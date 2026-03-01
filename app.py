import os
import sys
import torch
import numpy as np
import soundfile as sf
import shutil
import librosa
from neuttsair.neutts import NeuTTSAir
import gradio as gr

# ---------------------------
# eSpeak check (unchanged)
# ---------------------------
def check_espeak_installed():
    possible_paths = [
        "C:\\Program Files\\eSpeak NG",
        "C:\\Program Files (x86)\\eSpeak NG",
        "C:\\Program Files\\eSpeak",
        "C:\\Program Files (x86)\\eSpeak",
    ]

    found_exe_in_path = False
    for cmd in ['espeak-ng', 'espeak']:
        exe_path = shutil.which(cmd)
        if exe_path:
            print(f"Found {cmd} in PATH at: {exe_path}")
            found_exe_in_path = True

        dll_names = ['libespeak-ng.dll', 'espeak-ng.dll', 'libespeak.dll', 'espeak.dll']
        for exe_cmd in ['espeak-ng', 'espeak']:
            exe_path = shutil.which(exe_cmd)
            if exe_path:
                exe_dir = os.path.dirname(exe_path)
                for dll in dll_names:
                    candidate = os.path.join(exe_dir, dll)
                    if os.path.exists(candidate):
                        os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = candidate
                        print(f"Found espeak shared library at: {candidate}")
                        return True

        for path in possible_paths:
            if os.path.exists(path):
                for root, _, files in os.walk(path):
                    for dll in dll_names:
                        candidate = os.path.join(root, dll)
                        if os.path.exists(candidate):
                            os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = candidate
                            os.environ['PATH'] = f"{path};{os.environ['PATH']}"
                            return True
                bin_path = os.path.join(path, 'espeak-ng.exe')
                if os.path.exists(bin_path):
                    print(f"Found espeak-ng executable at: {bin_path}")
                    print("Adding to PATH...")
                    os.environ['PATH'] = f"{path};{os.environ['PATH']}"
                    break

    print("\nError: espeak-ng not found!")
    print("Install from https://github.com/espeak-ng/espeak-ng/releases")
    return False


if not check_espeak_installed():
    sys.exit(1)

# ---------------------------
# Model initialization
# ---------------------------
print("\nInitializing TTS model...")

# Clear CUDA cache before loading models to free up memory
if torch.cuda.is_available():
    print("Clearing CUDA cache...")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB total")

try:
    project_root = os.path.abspath(os.path.dirname(__file__))
    local_backbone = os.path.join(project_root, "Models", "neutts-air")

    def _resolve_hf_snapshot(root_path: str) -> str:
        try:
            # Check for HuggingFace cache structure
            for name in os.listdir(root_path):
                if name.startswith("models--"):
                    models_dir = os.path.join(root_path, name)
                    snapshots_dir = os.path.join(models_dir, "snapshots")
                    if os.path.isdir(snapshots_dir):
                        for snap in os.listdir(snapshots_dir):
                            snap_path = os.path.join(snapshots_dir, snap)
                            cfg = os.path.join(snap_path, "config.json")
                            if os.path.exists(cfg):
                                print(f"Found model in snapshots: {snap_path}")
                                return snap_path
        except Exception as e:
            print(f"Warning: Error resolving model path: {e}")
            pass
        return root_path

    backbone_arg = _resolve_hf_snapshot(local_backbone) if os.path.isdir(local_backbone) else "neutts-air-q4-gguf"
cd
    print(f"Using codec: neuphonic/neucodec")

    # For 4GB GPUs, load codec on CPU to save GPU memory
    # The codec is smaller and can run efficiently on CPU
    codec_device = "cuda"
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory_gb <= 4.5:  # For GPUs with 4GB or less
            print(f"Detected {gpu_memory_gb:.2f} GB GPU. Loading codec on CPU to save GPU memory.")
            codec_device = "cpu"
    
    tts = NeuTTSAir(
        backbone_repo=backbone_arg,
        backbone_device="cuda",
        codec_repo="neuphonic/neucodec",
        codec_device=codec_device,
    )
    
    # Clear cache again after loading to free any temporary allocations
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
except Exception as e:
    print(f"\nError initializing TTS model: {str(e)}")
    sys.exit(1)

# ---------------------------
# Voice loading logic
# ---------------------------
VOICES = {"samples": {}}
voice_dir = "samples"
os.makedirs(voice_dir, exist_ok=True)

for name in os.listdir(voice_dir):
    if name.endswith(".txt"):
        base = os.path.splitext(name)[0]
        txt_path = os.path.join(voice_dir, f"{base}.txt")
        wav_path = os.path.join(voice_dir, f"{base}.wav")
        pt_path = os.path.join(voice_dir, f"{base}.pt")

        if os.path.exists(txt_path) and (os.path.exists(wav_path) or os.path.exists(pt_path)):
            VOICES["samples"][base] = (txt_path, wav_path if os.path.exists(wav_path) else pt_path)

def format_voice_choice(name):
    return f"Voice: {name}"

# ---------------------------
# Core functions
# ---------------------------
def load_reference(voice_name):
    txt_path, audio_or_pt = VOICES["samples"][voice_name]
    ref_text = open(txt_path, "r").read().strip()

    if audio_or_pt.endswith(".pt"):
        ref_codes = torch.load(audio_or_pt)
    else:
        ref_codes = tts.encode_reference(audio_or_pt)
    return ref_text, ref_codes


def split_text_into_chunks(text, max_length=150):
    """Split text into smaller chunks preserving sentence and punctuation structure."""
    import re
    
    # Clean up the text first
    text = text.strip()
    if not text:
        return []

    # Split by sentence-ending punctuation while preserving the punctuation
    sentence_pattern = r'([.!?]+)'
    parts = re.split(sentence_pattern, text)

    # Reconstruct sentences with their punctuation
    sentences = []
    i = 0
    while i < len(parts):
        if parts[i].strip():
            sentence = parts[i].strip()
            # Add punctuation if it exists
            if i + 1 < len(parts) and parts[i + 1].strip():
                sentence += parts[i + 1]
                i += 2
            else:
                # If no punctuation follows, add a period (only once)
                if not sentence.endswith(('.', '!', '?')):
                    sentence += '.'
                i += 1
            sentences.append(sentence)
        else:
            i += 1

    # ✅ FIX: Avoid adding the last part twice when no punctuation present
    if len(parts) > 0 and parts[-1].strip():
        last_part = parts[-1].strip()
        # Add only if it's not already included
        if not any(last_part in s or s.startswith(last_part) for s in sentences):
            if not last_part.endswith(('.', '!', '?')):
                last_part += '.'
            sentences.append(last_part)

    # Group sentences into chunks
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If single sentence exceeds max_length, split by commas
        if len(sentence) > max_length:
            comma_parts = re.split(r'(,)', sentence)
            temp_sentence = ""
            
            i = 0
            while i < len(comma_parts):
                part = comma_parts[i].strip()
                comma = comma_parts[i + 1] if i + 1 < len(comma_parts) else ''
                
                # If part is still too long, split by words
                if len(part) > max_length:
                    words = part.split()
                    temp_words = []
                    
                    for word in words:
                        test_chunk = ' '.join(temp_words + [word])
                        if len(test_chunk) > max_length and temp_words:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                                current_chunk = ""
                            chunks.append(' '.join(temp_words))
                            temp_words = [word]
                        else:
                            temp_words.append(word)
                    
                    if temp_words:
                        part = ' '.join(temp_words) + comma
                        if current_chunk and len(current_chunk + ' ' + part) > max_length:
                            chunks.append(current_chunk.strip())
                            current_chunk = part
                        else:
                            current_chunk += (' ' if current_chunk else '') + part
                else:
                    part_with_comma = part + comma
                    if current_chunk and len(current_chunk + ' ' + part_with_comma) > max_length:
                        chunks.append(current_chunk.strip())
                        current_chunk = part_with_comma
                    else:
                        current_chunk += (' ' if current_chunk else '') + part_with_comma
                
                i += 2 if i + 1 < len(comma_parts) else 1
        else:
            # Normal sentence that fits within limit
            if current_chunk and len(current_chunk + ' ' + sentence) > max_length:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += (' ' if current_chunk else '') + sentence

    # CRITICAL: Always add remaining chunk at the end
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Filter out empty or duplicate chunks ✅
    final_chunks = []
    for chunk in chunks:
        if chunk.strip() and (not final_chunks or chunk.strip() != final_chunks[-1]):
            final_chunks.append(chunk.strip())

    return final_chunks


def process_chunk(chunk, ref_codes, ref_text, tts_model):
    """Process a single chunk of text and return the audio."""
    try:
        return tts_model.infer(chunk, ref_codes, ref_text)
    except Exception as e:
        # Swallow individual chunk errors and return None to let caller handle it
        return None

def estimate_generation_time(num_chunks):
    """Estimate the generation time based on number of chunks."""
    # Assuming average of 3 seconds per chunk plus overhead
    return num_chunks * 3 + 2

def format_time(seconds):
    """Format seconds into a readable time string."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    minutes = int(seconds // 60)
    seconds = seconds % 60
    return f"{minutes} minute{'s' if minutes != 1 else ''} {seconds:.1f} seconds"

def generate_speech(text, voice_name, speed_control="1x"):
    try:
        import time

                # Input validations
        if not text or not text.strip():
            yield 0, None, "❌ Error: Input text cannot be empty.", None
            return

        if not voice_name:
            yield 0, None, "❌ Error: No voice selected. Please select a voice.", None
            return

        if voice_name not in VOICES["samples"]:
            yield 0, None, f"❌ Error: Voice '{voice_name}' not found.", None
            return

        # Convert speed control string to float
        try:
            speed = float(speed_control.rstrip('x'))
        except ValueError:
            speed = 1.0  # Default to 1x if conversion fails

        start_time = time.time()
        
        # Load reference only once
        yield 10, None, "Loading voice reference...", None
        ref_text, ref_codes = load_reference(voice_name)
        
        # Split text into smaller chunks for better processing
        chunks = split_text_into_chunks(text)
        total_chunks = len(chunks)
        
        if total_chunks == 0:
            raise ValueError("No text to process")
        
        # Estimate total time
        estimated_time = estimate_generation_time(total_chunks)
        status = f"Estimated time to completion: {format_time(estimated_time)}\nProcessing {total_chunks} chunks..."
        yield 15, None, status, None
            
        # Process each chunk and store with its index
        chunk_results = []
        for i, chunk in enumerate(chunks, 1):
            chunk_start = time.time()
            
            # Update progress
            progress = int(15 + (75 * i / total_chunks))
            
            # Calculate and show time statistics
            elapsed_time = time.time() - start_time
            if i > 1:
                avg_time_per_chunk = elapsed_time / (i - 1)
                remaining_chunks = total_chunks - (i - 1)
                estimated_remaining = avg_time_per_chunk * remaining_chunks
                status = (
                    f"Processing chunk {i}/{total_chunks}\n"
                    f"Progress: {progress}% complete\n"
                    f"Est. remaining: {format_time(estimated_remaining)}"
                )
            else:
                status = f"Processing chunk {i}/{total_chunks}\nProgress: {progress}% complete"
            
            yield progress, None, status, None
            
            # Generate audio for this chunk
            chunk_wav = process_chunk(chunk, ref_codes, ref_text, tts)
            if chunk_wav is not None:
                # Store chunk with its index to maintain order
                chunk_results.append((i-1, chunk_wav))

        if not chunk_results:
            raise ValueError("Failed to generate any audio")

        # Update status for final processing
        yield 90, None, "Finalizing audio...\nOrdering and combining chunks...", None

        # Sort chunks by their original index and extract the audio data
        chunk_results.sort(key=lambda x: x[0])  # Sort by index
        processed_chunks = [chunk[1] for chunk in chunk_results]  # Extract audio data in order

        # Create silence once
        silence = np.zeros(int(24000 * 0.25))  # 0.25 seconds silence between chunks

        # Concatenate all chunks with silence in between
        all_wav = processed_chunks[0]
        for chunk_wav in processed_chunks[1:]:
            all_wav = np.concatenate([all_wav, silence, chunk_wav])

        # Apply speed adjustment if needed (pitch-preserving time-stretching)
        if speed != 1.0:
            # Use librosa for pitch-preserving time-stretching
            # rate > 1 speeds up, rate < 1 slows down
           all_wav = librosa.effects.time_stretch(all_wav.astype(np.float32), rate=speed)
           
        

        # Save the final audio
        temp_path = "temp_output.wav"
        sf.write(temp_path, all_wav, 24000)
        
        # Calculate and show total time taken
        total_time = time.time() - start_time
        final_status = f"✅ Generation complete!\nTotal time: {format_time(total_time)}"
        
        yield 100, temp_path, final_status, None
    except Exception as e:
        error_status = f"❌ Error generating speech: {str(e)}"
        yield 0, None, error_status, None


def delete_voice(voice_name):
    """Deletes a voice and its associated files."""
    try:
        if voice_name not in VOICES["samples"]:
            return f"❌ Voice '{voice_name}' not found!", gr.update()

        txt_path = f"samples/{voice_name}.txt"
        wav_path = f"samples/{voice_name}.wav"
        pt_path = f"samples/{voice_name}.pt"

        # Remove files if they exist
        for path in [txt_path, wav_path, pt_path]:
            if os.path.exists(path):
                os.remove(path)

        # Remove from VOICES dictionary
        del VOICES["samples"][voice_name]
        
        remaining_voices = list(VOICES["samples"].keys())
        new_selected = remaining_voices[0] if remaining_voices else None
        
        return f"✅ Voice '{voice_name}' deleted successfully!", gr.update(choices=remaining_voices, value=new_selected)
    except Exception as e:
        return f"❌ Error deleting voice: {e}", gr.update()

def clone_voice(new_name, txt, audio_file):
    """Encodes a new reference voice and saves its embedding."""
    try:

        
        # Input validations
        if not new_name or not new_name.strip():
            return "❌ Error: New Voice name cannot be empty.", gr.update()
        
        if not txt or not txt.strip():
            return "❌ Error: Reference text cannot be empty.", gr.update()
            
        if not audio_file:
            return "❌ Error: No reference audio file provided.", gr.update()
            
        if new_name in VOICES["samples"]:
            return f"❌ Error: Voice '{new_name}' already exists. Please choose a different name.", gr.update()
            
        os.makedirs("samples", exist_ok=True)
        txt_path = f"samples/{new_name}.txt"
        wav_path = f"samples/{new_name}.wav"
        pt_path = f"samples/{new_name}.pt"

        # Save reference text and audio
        with open(txt_path, "w") as f:
            f.write(txt.strip())
        shutil.copy(audio_file, wav_path)

        ref_codes = tts.encode_reference(wav_path)
        torch.save(ref_codes, pt_path)

        VOICES["samples"][new_name] = (txt_path, pt_path)
        return f"✅ Voice '{new_name}' cloned and saved successfully!", gr.update(choices=list(VOICES["samples"].keys()), value=new_name)
    except Exception as e:
        return f"❌ Error cloning voice: {e}", gr.update()


# ---------------------------
# UI
# ---------------------------
# Custom CSS for modern design
custom_css = """
footer {display: none !important;}
.footer {display: none !important;}
#api-docs-link {display: none !important;}

/* Modern header styling */
.heading-container {
    text-align: center;
    padding: 2rem 1rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 12px;
    margin-bottom: 2rem;
    color: white;
}

.heading-container h1 {
    margin: 0;
    font-size: 2.5rem;
    font-weight: 700;
    color: white;
}

.heading-container h3 {
    margin: 0.5rem 0 0 0;
    font-size: 1.1rem;
    font-weight: 400;
    color: rgba(255, 255, 255, 0.9);
}

/* Card-like containers */
.control-panel {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 12px;
    border: 1px solid #e9ecef;
    margin-bottom: 1rem;
}

.output-panel {
    background: #ffffff;
    padding: 1.5rem;
    border-radius: 12px;
    border: 2px solid #e9ecef;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

/* Button styling */
.primary-button {
    width: 100%;
    padding: 0.75rem;
    font-size: 1.1rem;
    font-weight: 600;
    border-radius: 8px;
    margin-top: 1rem;
}

/* Progress bar styling */
.progress-container {
    margin: 1rem 0;
}

/* Status box styling */
.status-box {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 1rem;
    min-height: 80px;
}

/* Audio player styling */
.audio-container {
    margin-top: 1rem;
    padding: 1rem;
    background: #f8f9fa;
    border-radius: 8px;
}

/* Voice selection styling */
.voice-controls {
    display: flex;
    gap: 0.5rem;
    align-items: flex-end;
}

/* Tab styling improvements */
.tab-nav {
    margin-bottom: 1.5rem;
}

/* Instructions content styling */
.instructions-content {
    background: #ffffff;
    padding: 2rem;
    border-radius: 12px;
    border: 1px solid #e9ecef;
    line-height: 1.8;
    max-width: 1200px;
    margin: 0 auto;
}

.instructions-content h1 {
    color: #667eea;
    border-bottom: 3px solid #667eea;
    padding-bottom: 0.5rem;
    margin-bottom: 1.5rem;
}

.instructions-content h2 {
    color: #764ba2;
    margin-top: 2rem;
    margin-bottom: 1rem;
    font-size: 1.5rem;
}

.instructions-content h3 {
    color: #5a67d8;
    margin-top: 1.5rem;
    margin-bottom: 0.75rem;
    font-size: 1.2rem;
}

.instructions-content ul, .instructions-content ol {
    margin-left: 1.5rem;
    margin-bottom: 1rem;
}

.instructions-content li {
    margin-bottom: 0.5rem;
}

.instructions-content code {
    background: #f1f3f5;
    padding: 0.2rem 0.4rem;
    border-radius: 4px;
    font-family: 'Courier New', monospace;
    font-size: 0.9em;
}

.instructions-content hr {
    border: none;
    border-top: 2px solid #e9ecef;
    margin: 2rem 0;
}

.instructions-content blockquote {
    border-left: 4px solid #667eea;
    padding-left: 1rem;
    margin-left: 0;
    color: #495057;
    font-style: italic;
}
"""

with gr.Blocks(
    title="Virtual Lab Voice Cloning",
    theme=gr.themes.Soft(
        primary_hue="purple",
        secondary_hue="blue"
    ),
    css=custom_css
) as app:
    
    # Modern header with gradient
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
                """
                <div class="heading-container">
                    <h1>🎙️ Virtual Lab Voice Cloning</h1>
                    <h3>High-Quality Text-to-Speech with Voice Cloning</h3>
                </div>
                """,
                elem_classes="heading"
            )

    with gr.Tab("🎯 Generate Speech", elem_classes="tab-nav"):
        with gr.Row(equal_height=True):
            # Left Column - Input Controls
            with gr.Column(scale=1, min_width=400):
                gr.Markdown("### 📝 Input Settings", elem_classes="control-panel")
                
                text_input = gr.Textbox(
                    label="📄 Text to Convert",
                    placeholder="Enter the text you want to convert to speech...",
                    lines=6,
                    elem_classes="text-input"
                )
                
                with gr.Row(elem_classes="voice-controls"):
                    voice_select = gr.Dropdown(
                        label="🎤 Select Voice",
                        choices=list(VOICES["samples"].keys()),
                        value=list(VOICES["samples"].keys())[0] if VOICES["samples"] else None,
                        interactive=True,
                        scale=3
                    )
                    delete_btn = gr.Button(
                        "🗑️",
                        variant="secondary",
                        size="sm",
                        scale=1,
                        min_width=50
                    )
                
                speed_control = gr.Dropdown(
                    label="⚡ Speech Speed",
                    choices=["1x", "1.1x", "1.2x", "1.3x", "1.4x", "1.5x"],
                    value="1x",
                    info="Select playback speed (preserves pitch and voice characteristics)"
                )
                
                generate_btn = gr.Button(
                    "🎙️ Generate Speech",
                    variant="primary",
                    size="lg",
                    elem_classes="primary-button"
                )
            
            # Right Column - Output & Status
            with gr.Column(scale=1, min_width=400):
                gr.Markdown("### 📊 Generation Status", elem_classes="output-panel")
                
                progress_bar = gr.Slider(
                    label="Progress",
                    minimum=0,
                    maximum=100,
                    value=0,
                    interactive=False,
                    elem_classes="progress-container"
                )
                
                status_box = gr.Textbox(
                    label="Status Information",
                    value="Ready to generate speech. Enter text and select a voice.",
                    lines=4,
                    interactive=False,
                    elem_classes="status-box"
                )
                
                delete_status = gr.Textbox(label="Status", visible=False)
                
                gr.Markdown("### 🎵 Audio Output", elem_classes="audio-container")
                audio_output = gr.Audio(
                    label="Generated Audio",
                    autoplay=True,
                    show_download_button=True
                )

        # Event handlers
        generate_btn.click(
            fn=generate_speech,
            inputs=[text_input, voice_select, speed_control],
            outputs=[progress_bar, audio_output, status_box, delete_status]
        )

        delete_btn.click(
            fn=delete_voice,
            inputs=[voice_select],
            outputs=[delete_status, voice_select]
        )

    with gr.Tab("🧬 Clone New Voice", elem_classes="tab-nav"):
        with gr.Row(equal_height=True):
            # Left Column - Voice Cloning Input
            with gr.Column(scale=1, min_width=400):
                gr.Markdown("### 🎤 Voice Cloning Setup", elem_classes="control-panel")
                
                new_voice_name = gr.Textbox(
                    label="📛 Voice Name",
                    placeholder="Enter a unique name for this voice...",
                    info="Choose a descriptive name for your cloned voice"
                )
                
                ref_text_input = gr.Textbox(
                    label="📝 Reference Text",
                    placeholder="Enter the exact text that is spoken in the audio sample...",
                    lines=4,
                    info="This should match the text spoken in your audio file"
                )
                
                ref_audio_input = gr.Audio(
                    label="🎵 Reference Audio File",
                    type="filepath"
                )
                gr.Markdown(
                    "<small>💡 Upload a WAV file containing the voice sample (recommended: 5-30 seconds)</small>",
                    elem_classes="info-text"
                )
                
                clone_btn = gr.Button(
                    "🧬 Clone Voice",
                    variant="primary",
                    size="lg",
                    elem_classes="primary-button"
                )
            
            # Right Column - Status
            with gr.Column(scale=1, min_width=400):
                gr.Markdown("### 📋 Cloning Status", elem_classes="output-panel")
                
                clone_status = gr.Textbox(
                    label="Status",
                    value="Ready to clone a new voice. Fill in the details on the left and upload an audio sample.",
                    lines=8,
                    interactive=False,
                    elem_classes="status-box"
                )
                
                gr.Markdown(
                    """
                    ### 💡 Tips for Best Results
                    - Use clear, high-quality audio recordings
                    - Ensure the reference text matches what's spoken
                    - Audio length: 5-30 seconds works best
                    - Speak naturally and clearly in the sample
                    - Avoid background noise when possible
                    """,
                    elem_classes="control-panel"
                )

        # Event handler
        clone_btn.click(
            fn=clone_voice,
            inputs=[new_voice_name, ref_text_input, ref_audio_input],
            outputs=[clone_status, voice_select]
        )

    with gr.Tab("📖 Instructions", elem_classes="tab-nav"):
        with gr.Column():
            gr.Markdown(
                """
                # 🎙️ Virtual Lab Voice Cloning - User Guide
                
                Welcome to the Virtual Lab Voice Cloning tool! This guide will help you get started with creating high-quality text-to-speech audio using voice cloning technology.
                
                ---
                
                ## 🎯 How to Generate Speech
                
                ### Step 1: Navigate to the "Generate Speech" Tab
                Click on the **"🎯 Generate Speech"** tab at the top of the interface.
                
                ### Step 2: Enter Your Text
                - Type or paste the text you want to convert to speech in the **"📄 Text to Convert"** text box
                - You can enter multiple sentences or paragraphs
                - The tool will automatically split long texts into manageable chunks
                
                ### Step 3: Select a Voice
                - Choose a voice from the **"🎤 Select Voice"** dropdown menu
                - Only voices that have been cloned and saved will appear in this list
                - You can delete a voice by clicking the 🗑️ button next to the voice selector
                
                ### Step 4: Adjust Speech Speed (Optional)
                - Use the **"⚡ Speech Speed"** dropdown to control playback speed
                - Options range from 1x (normal) to 1.5x (faster)
                - Speed adjustment preserves pitch and voice characteristics
                
                ### Step 5: Generate Audio
                - Click the **"🎙️ Generate Speech"** button
                - Monitor the progress bar and status messages
                - The generated audio will appear automatically when complete
                - You can play the audio directly in the browser or download it
                
                ---
                
                ## 🧬 How to Clone a New Voice
                
                ### Step 1: Navigate to the "Clone New Voice" Tab
                Click on the **"🧬 Clone New Voice"** tab at the top of the interface.
                
                ### Step 2: Prepare Your Audio Sample
                Before cloning, you'll need:
                - A clear audio recording (WAV format recommended)
                - 5-30 seconds of speech works best
                - High-quality audio with minimal background noise
                - Natural, clear speech
                
                ### Step 3: Enter Voice Details
                - **Voice Name**: Enter a unique, descriptive name for your cloned voice
                - **Reference Text**: Type the exact text that is spoken in your audio sample
                - **Reference Audio**: Upload your WAV audio file using the file uploader
                
                ### Step 4: Clone the Voice
                - Click the **"🧬 Clone Voice"** button
                - Wait for the cloning process to complete
                - Once successful, the new voice will be available in the voice selector
                
                ### Step 5: Use Your Cloned Voice
                - Navigate back to the "Generate Speech" tab
                - Your newly cloned voice will appear in the voice dropdown
                - Select it and generate speech as usual
                
                ---
                
                ## 💡 Best Practices & Tips
                
                ### For Voice Cloning:
                - ✅ Use high-quality, clear audio recordings
                - ✅ Ensure the reference text exactly matches what's spoken in the audio
                - ✅ Record in a quiet environment to minimize background noise
                - ✅ Speak naturally and at a normal pace
                - ✅ Use 5-30 seconds of audio for best results
                - ❌ Avoid very short clips (less than 3 seconds)
                - ❌ Avoid clips with heavy background noise or music
                - ❌ Don't use text that doesn't match the audio content
                
                ### For Speech Generation:
                - ✅ Use proper punctuation for better natural pauses
                - ✅ Break long texts into paragraphs for better processing
                - ✅ Review the generated audio and adjust speed if needed
                - ✅ The tool automatically handles long texts by splitting them into chunks
                - ✅ Generated audio is saved and can be downloaded
                
                ### Performance Tips:
                - The tool processes text in chunks for better performance
                - Longer texts will take more time to generate
                - Progress updates show estimated completion time
                - GPU acceleration is used when available for faster processing
                
                ---
                
                ## 🔧 Technical Information
                
                ### Supported Formats:
                - **Input Audio**: WAV format (recommended)
                - **Output Audio**: WAV format, 24kHz sample rate
                - **Text**: Plain text (UTF-8)
                
                ### System Requirements:
                - NVIDIA GPU recommended for best performance
                - CUDA support for GPU acceleration
                - eSpeak NG installed for phonemization
                
                ### Features:
                - High-quality neural text-to-speech
                - Voice cloning from short audio samples
                - Pitch-preserving speed control
                - Automatic text chunking for long inputs
                - Real-time progress tracking
                
                ---
                
                ## ❓ Troubleshooting
                
                ### Common Issues:
                
                **"No voice selected" error:**
                - Make sure you have cloned at least one voice
                - Check that the voice appears in the dropdown menu
                
                **"Input text cannot be empty" error:**
                - Ensure you've entered text in the text input box
                - Check for whitespace-only text
                
                **Audio generation fails:**
                - Verify your GPU has enough memory
                - Try generating shorter texts first
                - Check that the voice files are not corrupted
                
                **Voice cloning fails:**
                - Ensure the audio file is in WAV format
                - Verify the reference text matches the audio content
                - Check that the audio quality is sufficient
                - Make sure the voice name is unique
                
                ---
                
                ## 📝 Notes
                
                - All cloned voices are saved locally in the `samples` folder
                - Generated audio files are temporary and should be downloaded if you want to keep them
                - The tool uses advanced neural networks for high-quality voice synthesis
                - Processing time depends on text length and system performance
                
                ---
                
                **Enjoy creating amazing voice clones! 🎉**
                """,
                elem_classes="instructions-content"
            )

if __name__ == "__main__":
    import socket

    # Use PORT from environment (e.g. Render) or find a free port locally
    port = os.environ.get("PORT")
    if port is not None:
        port = int(port)
        server_name = "0.0.0.0"  # required for cloud hosts
        inbrowser = False
    else:
        def find_free_port(start=7860, end=7900):
            for p in range(start, end):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    try:
                        s.bind(("localhost", p))
                        return p
                    except OSError:
                        continue
            return start  # fallback
        port = find_free_port()
        server_name = "localhost"
        inbrowser = True

    print(f"\nLaunching on http://{server_name}:{port}")
    app.launch(server_name=server_name, server_port=port, share=False, inbrowser=inbrowser, show_error=True, show_api=False)
