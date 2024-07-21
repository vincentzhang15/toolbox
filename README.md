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
- Text Tools
    - Convert curl command to be windows compatible.
    - Text summarize using AI model (Alibaba Qwen2 72b).
- File Tools
    - Convert file to text string.
    - Delete folders in the Shell.
