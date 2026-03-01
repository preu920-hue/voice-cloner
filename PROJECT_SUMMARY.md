# Project Summary - Virtual Lab Voice Cloning

## What Is This Project?

This is a **Voice Cloning and Text-to-Speech** application. In simple terms, it lets you:

1. **Clone someone's voice** from a short audio recording (like 5-30 seconds)
2. **Make that cloned voice say anything** you type in text

Think of it like having a voice actor that can read any script you give them, but using a voice you've recorded yourself!

## What Can You Do With It?

### Main Features:

- **🎤 Clone Voices**: Upload a short audio file of someone speaking, and the app learns their voice
- **📝 Generate Speech**: Type any text and have it spoken in your cloned voice
- **⚡ Speed Control**: Make the speech faster or slower (0.5x to 2x speed)
- **💾 Save Voices**: Keep multiple cloned voices and switch between them
- **🎵 Audio Output**: Get high-quality audio files you can download and use

## How Does It Work? (Simple Explanation)

### Step 1: Voice Cloning
When you upload an audio sample:
- The app listens to the audio
- It learns the unique characteristics of that voice (like tone, accent, speaking style)
- It saves this "voice profile" so you can use it later

### Step 2: Speech Generation
When you want to generate speech:
- You type the text you want to hear
- The app uses AI to convert your text into speech
- It uses the cloned voice profile to make it sound like the original speaker
- You get an audio file you can play or download

## What Technology Does It Use?

The app uses **Artificial Intelligence (AI)** to do all this magic. Specifically:

- **NeuTTS-Air**: The main AI model that converts text to speech
- **NeuCodec**: Helps process and understand audio
- **Python**: The programming language everything is written in
- **Gradio**: Creates the web interface you see and use
- **PyTorch**: Helps run the AI models (works best with NVIDIA graphics cards)

## What Do You Need to Run It?

### Required:
- **Windows computer** (this version is for Windows)
- **Python 3.12** installed
- **eSpeak-ng** installed (a text-to-speech library - there's an installer included)
- **NVIDIA Graphics Card** (recommended for faster processing)

### Optional but Helpful:
- At least 4GB of video memory (VRAM) on your graphics card
- Good internet connection (for first-time setup to download AI models)

## How to Use It

### First Time Setup:
1. Make sure Python 3.12 is installed
2. Install eSpeak-ng (use the included `espeak-ng (1).msi` file)
3. Double-click `Run NeuTTS.bat` to start the app
4. Wait for it to load (first time takes longer as it downloads AI models)
5. A web browser window will open automatically

### Cloning a Voice:
1. Go to the "🧬 Clone New Voice" tab
2. Give your voice a name (like "John's Voice")
3. Type the exact text that was spoken in your audio file
4. Upload a WAV audio file (5-30 seconds works best)
5. Click "🧬 Clone Voice"
6. Wait for it to finish processing

**Tips for best results:**
- Use clear, high-quality audio
- Make sure the text matches what's actually spoken
- Avoid background noise
- 5-30 seconds of audio is perfect

### Generating Speech:
1. Go to the "🎯 Generate Speech" tab
2. Type or paste the text you want to convert
3. Select a cloned voice from the dropdown menu
4. Adjust the speed slider if you want (1.0 = normal speed)
5. Click "🎙️ Generate Speech"
6. Wait for processing (you'll see a progress bar)
7. Listen to the generated audio!

## Project Structure (What Files Are Where)

```
neutts-air NVIDIA GPU/
├── app.py              ← Main application (the web interface)
├── requirements.txt    ← List of software packages needed
├── Run NeuTTS.bat     ← Easy way to start the app (double-click this!)
├── neuttsair/         ← Core voice cloning code
│   └── neutts.py      ← The brain that does the voice cloning
├── Models/            ← AI models stored here (downloaded automatically)
├── samples/           ← Your cloned voices are saved here
│   ├── [VoiceName].txt  ← The text that was spoken
│   ├── [VoiceName].wav  ← The original audio file
│   └── [VoiceName].pt   ← The voice profile (learned characteristics)
└── temp_output.wav    ← Temporary file for generated audio
```

## Important Notes

### Privacy:
- ✅ Everything runs on your computer - no data is sent to the internet
- ✅ Your voice samples stay on your computer
- ✅ All processing happens locally

### Performance:
- The app works best with an NVIDIA graphics card
- If you have a smaller graphics card (4GB or less), some parts will run on your CPU instead
- First-time setup takes longer because it downloads AI models
- Generating speech takes a few seconds to a few minutes depending on text length

### File Formats:
- **Input**: WAV audio files for voice cloning
- **Output**: WAV audio files (24,000 Hz sample rate)
- **Text**: Plain text input

## Common Use Cases

People use this for:
- **Content Creation**: Making voiceovers for videos
- **Accessibility**: Converting text to speech for people who need it
- **Education**: Language learning and pronunciation practice
- **Entertainment**: Creating fun voice projects
- **Personal Projects**: Making custom voice assistants

## Troubleshooting

**Problem**: App won't start
- **Solution**: Make sure eSpeak-ng is installed and Python 3.12 is installed

**Problem**: Slow processing
- **Solution**: Make sure your graphics card is being used (check the console messages)

**Problem**: Out of memory errors
- **Solution**: The app automatically uses CPU for some parts if your GPU is small. Close other programs to free up memory.

**Problem**: Audio quality isn't good
- **Solution**: Use higher quality reference audio, make sure the text matches what's spoken, try a different audio sample

## Summary

This is a powerful but easy-to-use tool that lets you clone voices and generate speech. It uses advanced AI technology but presents it in a simple web interface. Everything runs on your computer for privacy, and you can create unlimited voice clones. Perfect for content creators, educators, or anyone who wants to experiment with voice technology!

---

**Version**: 2.0  
**Platform**: Windows (NVIDIA GPU recommended)  
**Last Updated**: 2025



