# Virtual Lab Voice Cloning - Documentation

## 📖 Overview

Virtual Lab Voice Cloning is a powerful text-to-speech (TTS) application that allows you to generate natural-sounding speech from text using AI-powered voice cloning technology. The application can clone voices from audio samples and generate speech in those cloned voices, making it perfect for creating personalized voice content.

## ✨ Key Features

### 1. **Voice Cloning**
   - Clone any voice from a short audio sample (5-30 seconds recommended)
   - Automatically extracts voice characteristics from the audio
   - Stores cloned voices for future use
   - Supports multiple voice profiles simultaneously

### 2. **Text-to-Speech Generation**
   - Convert any text into natural-sounding speech
   - Uses cloned voices to match the original speaker's style
   - Handles long texts by automatically splitting them into manageable chunks
   - Preserves sentence structure and punctuation

### 3. **Speed Control**
   - Adjustable speech speed from 0.5x to 2.0x
   - Real-time speed adjustment without quality loss
   - Maintains natural voice characteristics at any speed

### 4. **Progress Tracking**
   - Real-time progress indicators during generation
   - Time estimates for completion
   - Detailed status messages
   - Visual progress bar

### 5. **Voice Management**
   - Create unlimited custom voices
   - Delete unwanted voices
   - Easy voice selection from dropdown menu
   - Voice data stored locally for privacy

### 6. **Modern Web Interface**
   - Clean, user-friendly design
   - Two main tabs: Generate Speech and Clone Voice
   - Responsive layout
   - Real-time audio playback

## 🛠️ Tech Stack & Libraries

### Core Technologies

#### **Python 3.12**
   - The main programming language used for the entire application

#### **PyTorch 2.8.0**
   - Deep learning framework for running AI models
   - Handles GPU acceleration for faster processing
   - Manages neural network computations

#### **Gradio 5.0+**
   - Web interface framework
   - Creates the user-friendly web UI
   - Handles user interactions and displays

### AI & Machine Learning Libraries

#### **Transformers 4.56.1**
   - Provides pre-trained language models
   - Handles the text-to-speech backbone model
   - Manages tokenization and text processing

#### **NeuCodec (>=0.0.4)**
   - Neural audio codec for encoding/decoding audio
   - Converts audio to compressed format for processing
   - Reconstructs high-quality audio from codes

#### **Resemble-Perth 1.0.1**
   - Watermarking technology
   - Adds invisible watermarks to generated audio
   - Helps identify AI-generated content

#### **Phonemizer 3.3.0**
   - Converts text to phonetic representation
   - Uses eSpeak-ng for phonemization
   - Helps the model understand pronunciation

### Audio Processing Libraries

#### **Librosa 0.11.0**
   - Audio analysis and processing
   - Handles audio file loading and manipulation
   - Provides audio feature extraction

#### **SoundFile 0.13.1**
   - Reads and writes audio files
   - Supports WAV format
   - Handles audio file I/O operations

#### **NumPy 2.2.6**
   - Numerical computing library
   - Handles array operations for audio data
   - Performs mathematical operations on audio samples

### Additional Libraries

#### **TorchAO 0.13.0**
   - PyTorch optimization library
   - Helps optimize model performance
   - Improves inference speed

## 🏗️ How It Works

### 1. **Voice Cloning Process**
   ```
   Audio Sample → Encoding → Voice Embedding → Storage
   ```
   - When you upload an audio sample, the system:
     1. Loads the audio file
     2. Encodes it using the neural codec
     3. Extracts voice characteristics
     4. Saves the voice profile for future use

### 2. **Speech Generation Process**
   ```
   Text Input → Phonemization → Model Processing → Audio Decoding → Output
   ```
   - When generating speech:
     1. Your text is converted to phonetic representation
     2. The AI model processes it with the cloned voice reference
     3. Audio codes are generated
     4. Codes are decoded into audio waveform
     5. Watermark is applied
     6. Final audio is output

### 3. **Text Chunking**
   - Long texts are automatically split into smaller chunks
   - Each chunk is processed separately
   - Chunks are combined with natural pauses
   - Maintains sentence boundaries and punctuation

## 📁 Project Structure

```
neutts-air NVIDIA GPU/
├── app.py                 # Main application file with UI and logic
├── requirements.txt       # Python dependencies
├── Run NeuTTS.bat         # Windows batch file to run the app
├── neuttsair/            # Core TTS module
│   ├── __init__.py
│   └── neutts.py         # NeuTTSAir class implementation
├── Models/               # AI model storage
│   └── neutts-air/       # Pre-trained TTS model
├── samples/              # Cloned voice storage
│   ├── [VoiceName].txt   # Reference text for each voice
│   ├── [VoiceName].wav   # Original audio sample
│   └── [VoiceName].pt    # Encoded voice embedding
└── temp_output.wav       # Temporary output file
```

## 🚀 Getting Started

### Prerequisites

1. **Python 3.12** installed
2. **eSpeak-ng** installed (for phonemization)
   - Download from: https://github.com/espeak-ng/espeak-ng/releases
   - Or use the included installer: `espeak-ng (1).msi`
3. **NVIDIA GPU** (recommended for faster processing)
   - CUDA-compatible GPU
   - At least 4GB VRAM (codec will use CPU if less)

### Installation

1. **Create Virtual Environment** (if not already created)
   ```bash
   python -m venv .venv
   ```

2. **Activate Virtual Environment**
   - Windows: `.venv\Scripts\Activate.ps1`
   - Or use the batch file: `Run NeuTTS.bat`

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

#### Option 1: Using Batch File (Windows)
   - Double-click `Run NeuTTS.bat`

#### Option 2: Using Python Directly
   ```bash
   # Activate virtual environment first
   .venv\Scripts\Activate.ps1
   
   # Set environment variables
   $env:HF_HOME = "$PWD\.hf_cache"
   $env:HUGGINGFACE_HUB_CACHE = "$PWD\.hf_cache\hub"
   $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
   $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
   
   # Run the app
   python app.py
   ```

The application will open in your browser at `http://localhost:7860`

## 📖 Usage Guide

### Cloning a Voice

1. Go to the **"🧬 Clone New Voice"** tab
2. Enter a unique name for your voice
3. Enter the exact text that was spoken in your audio sample
4. Upload a WAV audio file (5-30 seconds recommended)
5. Click **"🧬 Clone Voice"**
6. Wait for the cloning process to complete

**Tips for Best Results:**
- Use clear, high-quality audio recordings
- Ensure reference text matches what's spoken exactly
- Speak naturally and clearly
- Avoid background noise
- 5-30 seconds of audio works best

### Generating Speech

1. Go to the **"🎯 Generate Speech"** tab
2. Enter or paste the text you want to convert
3. Select a cloned voice from the dropdown
4. Adjust the speed slider if needed (1.0 = normal speed)
5. Click **"🎙️ Generate Speech"**
6. Wait for processing (progress bar shows status)
7. Listen to the generated audio

**Features:**
- Handles long texts automatically
- Shows real-time progress
- Displays time estimates
- Auto-plays generated audio

### Managing Voices

- **Select Voice**: Use the dropdown menu to choose a voice
- **Delete Voice**: Click the 🗑️ button next to the voice selector
- **Create New Voice**: Use the Clone Voice tab to add more voices

## ⚙️ Technical Details

### Model Architecture

- **Backbone Model**: NeuTTS-Air (transformer-based language model)
- **Codec**: NeuCodec (neural audio codec)
- **Watermarking**: PerthNet (implicit watermarking)

### Processing Pipeline

1. **Text Input** → Phonemization (text to phonemes)
2. **Phonemes + Voice Reference** → AI Model Processing
3. **Model Output** → Audio Code Generation
4. **Audio Codes** → Decoding to Waveform
5. **Waveform** → Watermarking
6. **Final Audio** → Output

### Memory Management

- Automatically detects GPU memory
- For GPUs with ≤4.5GB VRAM, codec runs on CPU
- Clears CUDA cache before and after model loading
- Optimized for efficient memory usage

### Audio Specifications

- **Sample Rate**: 24,000 Hz
- **Format**: WAV (PCM)
- **Chunk Processing**: Automatic splitting for long texts
- **Silence Between Chunks**: 0.25 seconds

## 🔧 Configuration

### Environment Variables

The application uses these environment variables:

- `HF_HOME`: Hugging Face cache directory
- `HUGGINGFACE_HUB_CACHE`: Model cache location
- `PYTORCH_CUDA_ALLOC_CONF`: CUDA memory allocation settings
- `HF_HUB_DISABLE_SYMLINKS_WARNING`: Disables symlink warnings
- `PHONEMIZER_ESPEAK_LIBRARY`: Path to eSpeak library

### Model Settings

- **Backbone Device**: CUDA (GPU) by default
- **Codec Device**: CUDA for GPUs >4.5GB, CPU otherwise
- **Max Context**: 2048 tokens
- **Temperature**: 1.0 (for generation)
- **Top-k**: 50 (for generation)

## 🐛 Troubleshooting

### Common Issues

1. **eSpeak not found**
   - Install eSpeak-ng from the releases page
   - Or run the included installer

2. **Out of Memory Errors**
   - The app automatically uses CPU for codec on low-memory GPUs
   - Close other applications to free GPU memory

3. **Slow Processing**
   - Ensure GPU is being used (check console output)
   - Reduce text length for faster processing
   - Lower-end GPUs may take longer

4. **Audio Quality Issues**
   - Use high-quality reference audio
   - Ensure reference text matches audio exactly
   - Try different audio samples

## 📝 File Formats

### Supported Input Formats
- **Text**: Plain text input
- **Audio**: WAV format for voice cloning

### Output Format
- **Audio**: WAV format, 24kHz sample rate
- **Temporary files**: Stored as `temp_output.wav`

## 🔒 Privacy & Security

- All voice data is stored locally
- No data is sent to external servers
- Voice embeddings stored in `.pt` files
- Audio samples stored in `samples/` directory
- Models cached locally in `Models/` directory

## 📚 Additional Resources

- **NeuTTS-Air Model**: https://huggingface.co/neuphonic/neutts-air
- **NeuCodec**: https://huggingface.co/neuphonic/neucodec
- **eSpeak-ng**: https://github.com/espeak-ng/espeak-ng

## 🎯 Use Cases

- **Content Creation**: Generate voiceovers for videos
- **Accessibility**: Text-to-speech for accessibility needs
- **Personalization**: Create custom voice assistants
- **Education**: Language learning and pronunciation
- **Entertainment**: Voice cloning for creative projects

## 📄 License

This project uses various open-source libraries. Please refer to individual library licenses for details.

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section
2. Review console output for error messages
3. Ensure all dependencies are installed correctly
4. Verify eSpeak-ng is properly installed

---

**Version**: 2.0  
**Last Updated**: 2025  
**Platform**: Windows (NVIDIA GPU)

