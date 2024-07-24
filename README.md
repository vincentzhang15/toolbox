# toolbox
 
## About
Toolbox for performing common operations offline.

## Tech
- python, cv2, pillow
- regular expression
- ffmpeg
- ollama
- Tesseract OCR
- OpenAI Whisper (ASR), Nvidia NeMo Canary 1b (ASR)
- Alibaba Qwen2 72b

## Features
- Image Tools:
    - Image of document to scanned image.
    - Image of document to text file.
    - Convert any image (e.g., .HEIC, .webp) to .jpg.
    - Crop the largest square with face centered in the image using an OpenCV pre-trained Haar cascade for face detection.
- Audio Tools
    - Extract audio segment from an audio file.
    - Speech to text conversion using AI model (OpenAI Whisper & Nvidia NeMo Canary 1b).
    - Chop audio into intervals.
    - Automatically preprocess input to Nvidia NeMo Canary (speech-to-text) by converting audio to mono channel.
    - Merge audio files in a folder to a single .mp3 file.
    - Convert any audio file to mp3.
    - Convert all audio files in a directory to .mp3.
    - Normalize audio volume.
    - Compress audio file.
- Text Tools
    - Convert curl command to be windows compatible.
    - Text summarize using AI model (Alibaba Qwen2 72b).
    - LLM query feature that generalizes text summary feature.
- File Tools
    - Convert file to text string.
    - Delete folders.
    - Purge all node_modules folders in a directory and subdirectories.
    - Convert PDF to text.
