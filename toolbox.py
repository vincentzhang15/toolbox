"""
Toolbox for performing common operations offline.

FEATURES:
2024-07-12:
- Image of document to scanned image.
- Image of document to text file.

2024-07-13:
- Convert curl command to be windows compatible.

2024-07-18:
- Convert .HEIC image to .jpg [2024-07-20 combined into any2jpg].

2024-07-19:
- Extract audio segment from an audio file.
- Speech to text conversion using AI model (OpenAI Whisper & Nvidia NeMo Canary 1b).
- Convert file to text string.
- Text summarize using AI model (Alibaba Qwen2 72b).
- Chop audio into intervals.
- Delete folders in the Shell.

2024-07-20:
- Convert any image (e.g., .HEIC, .webp) to .jpg.
- Crop the largest square with face centered in the image using an OpenCV pre-trained Haar cascade for face detection.

2024-07-21:
- Purge all node_modules folders in a directory and subdirectories.

Created: 2024-07-12
Updated: 2024-07-21
"""

import cv2
import numpy as np
from PIL import Image
import pytesseract
from pytesseract import Output
import pyperclip
import re
import subprocess
import torch
import os
import shutil

class ImageOperations:
    def _draw_rectangles(img, objs):
        for o in objs:
            x,y,w,h = o[1:]
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite('rect.jpg', img)

    def img2scan(path):
        gray = cv2.cvtColor(np.array(Image.open(path)), cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0) # Smooth out noise with 5x5 pixel Gaussian kernel and auto-detected standard deviation.
        scan = cv2.adaptiveThreshold(
                    blur, # Single channel image.
                    255, # Max pixel intensity.
                    cv2.ADAPTIVE_THRESH_MEAN_C, # Threshold is the mean of the neighbourhood area minus the constant C.
                    cv2.THRESH_BINARY, # Pixels with values > threshold are set to 255.
                    19, # Neighbourhood area size to calculate the adaptive threshold value.
                    10 # Constant C to subtract from the mean.
                )
        cv2.imwrite('scan.jpg', scan)
        return scan

    def img2text(path):
        scan = img2scan(path)
        res = pytesseract.image_to_data(scan, output_type=Output.DICT, config=r'--oem 3 --psm 6', lang='eng')
        words = [(text,x,y,w,h) for text,x,y,w,h,conf in zip(res['text'],res['left'],res['top'],res['width'],res['height'],res['conf']) if text.strip() and int(conf) > 0]
        words.sort(key=lambda w: (w[2],w[1]))  # Sort by (y,x).

        ImageOperations._draw_rectangles(scan, words)

        lines = []
        line = []
        prev_y = None
        merge_line = lambda ln: ' '.join([x[0] for x in sorted(ln, key=lambda x: x[1])])
        for word in words:
            text,x,y,w,h = word
            elm = (text, x)
            if prev_y is None:
                line.append(elm)
            elif y - prev_y > h:
                lines.append(merge_line(line))
                line = [elm]
            else:
                line.append(elm)
            prev_y = y
        lines.append(merge_line(line))

        with open('ocr.txt', 'w') as f:
            f.write('\n'.join(lines))

        return scan

    def any2jpg(path):
        """Convert any image (e.g., .HEIC, .webp) to .jpg.

        HEIC Ref: https://stackoverflow.com/questions/54395735/how-to-work-with-heic-image-file-types-in-python
        """
        if os.path.isdir(path):
            return
        root, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext == ".jpg":
            return
        if ext == ".heic":
            from pillow_heif import register_heif_opener
            register_heif_opener()

        img = Image.open(path)
        img.save(root+".jpg")

    def crop_face_centered(path, output="cropped.jpg", method="max"):
        """Crop the largest square with face centered in the image.
        """
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            raise ValueError("No faces detected.")
        x,y,w,h = faces[0]
        cx,cy = x+w//2, y+h//2

        def center_strict():
            """Face will strictly be in the center of the square."""
            m = min(cx, cy, img.shape[1]-cx, img.shape[0]-cy)
            return cx-m, cx+m, cy-m, cy+m
        def center_max():
            """As much of the image will be included as possible with the face contained within."""
            length = min(img.shape[0], img.shape[1])//2
            x1 = cx - length
            x2 = cx + length
            y1 = cy - length
            y2 = cy + length
            if x1 < 0:
                x2 += -x1
                x1 = 0
            if x2 > img.shape[1]:
                x1 -= x2 - img.shape[1]
                x2 = img.shape[1]
            if y1 < 0:
                y2 += -y1
                y1 = 0
            if y2 > img.shape[0]:
                y1 -= y2 - img.shape[0]
                y2 = img.shape[0]
            return x1, x2, y1, y2

        x1,x2,y1,y2 = None,None,None,None
        if method == "strict":
            x1,x2,y1,y2 = center_strict()
        elif method == "max":
            x1,x2,y1,y2 = center_max()
        crop = img[y1:y2, x1:x2]        
        cv2.imwrite(output, crop)

class AudioOperations:
    def extract_audio(file, start="00:00:00", stop="00:00:15", output="extracted.mp3"):
        """Extract audio segment.
        Input and output files should both be .mp3.
        """
        cmd = rf"ffmpeg -i {file} -ss {start} -to {stop} -c copy -acodec copy -y {output}"
        print("Running:", cmd)
        subprocess.run(cmd, shell=True)
    def chop(file, interval="00:01:00", output_dir="chopped_output"):
        """Chop audio into intervals.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            raise ValueError(f"Output directory {output_dir} already exists.")
        cmd = rf"ffmpeg -i {file} -f segment -segment_time {interval} -c copy -y {output_dir}/out%03d.mp3"
        print("Running:", cmd)
        subprocess.run(cmd, shell=True)
    def speech2text(path, output="output.txt", model="canary"):
        """Convert speech to text.
        """
        def whisper():
            """Use OpenAI Whisper model.
            https://github.com/openai/whisper
            """
            import whisper
            model = whisper.load_model("large")
            result = model.transcribe(path)
            return result["text"]
        def canary():
            """Use Nvidia Canary 1b model.
            https://huggingface.co/nvidia/canary-1b
            """
            from nemo.collections.asr.models import EncDecMultiTaskModel

            # load model
            canary_model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b')

            # update dcode params
            decode_cfg = canary_model.cfg.decoding
            decode_cfg.beam.beam_size = 1
            canary_model.change_decoding_strategy(decode_cfg)

            try:
                predicted_text = canary_model.transcribe(path)

                return "\n".join(predicted_text)
            except torch.cuda.OutOfMemoryError:
                chop(path, interval="00:01:00", output_dir="chopped_output")
                res = ""
                for f in os.listdir("chopped_output"):
                    predicted_text = canary_model.transcribe(f"chopped_output/{f}")
                    res += "\n".join(predicted_text)
                    print(predicted_text)
                delete("chopped_output", force=True)
                return res

        r = None
        if model=="whisper":
            r = whisper()
        elif model=="canary":
            r = canary()
        else:
            raise ValueError("No speech2text model specified.")

        with open(output, "w") as f:
            f.write(r)
        print(r)

class FileOperations:
    def file2text(path):
        with open(path, "r") as f:
            return f.read()
    def delete(path, force=False):
        if not force:
            input(f"WARNING: Deleting {path}. Press Enter to continue.")
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"Deleted: {path}")
        else:
            raise ValueError(f"File {path} does not exist.")
    def purge_node_modules(path):
        """Delete all node_modules folders given directory and subdirectories.
        """
        paths = []
        for root, dirs, _ in os.walk(path, topdown=True):
            if "node_modules" in dirs:
                d = os.path.join(root, "node_modules")
                print(d)
                paths.append(d)
            dirs[:] = [d for d in dirs if d!="node_modules"]
        if len(paths) == 0:
            print(f"No node_modules folders found in {path}.")
            return
        if input("Purge? (Y/n): ") == "Y":
            for p in paths:
                delete(p, force=True)

class TextOperations:
    def curl2win():
        """Copy curl command to clipboard and this method will replace the
        clipboard with the converted windows-compatible command.
        """
        curl = pyperclip.paste()
        curl = curl.replace('\\', '^')
        a,b = curl.split('-d')
        b = b.replace("\r\n", " ")
        b = b.replace('\"', '\\"')
        b = b.replace("'", '"')
        b = re.sub("\s{2,}", "", b)

        curl = f"{a}-d{b}"
        pyperclip.copy(curl)
        print("Copied:", curl)

    def summarize(text, output="summary.txt"):
        """Summarize text using an ollama model.
        """
        import ollama

        prompt = f"""
            Create a bullet point summary of the following text:
            {text}
        """
        print("prompt:", prompt)
        stream = ollama.chat(
            model='qwen2:72b',
            messages=[{
                'role': 'user',
                'content': prompt}],
            stream=True,
        )
        r = ""
        for chunk in stream:
            print(chunk['message']['content'], end='', flush=True)
            r += chunk['message']['content']
        with open(output, "w") as f:
            f.write(r)

img2scan = ImageOperations.img2scan
img2text = ImageOperations.img2text
curl2win = TextOperations.curl2win
extract_audio = AudioOperations.extract_audio
speech2text = AudioOperations.speech2text
file2text = FileOperations.file2text
summarize = TextOperations.summarize
chop = AudioOperations.chop
delete = FileOperations.delete
any2jpg = ImageOperations.any2jpg
crop_face_centered = ImageOperations.crop_face_centered
purge_node_modules = FileOperations.purge_node_modules

if __name__ == "__main__":
    # img2scan('Consolidated.png')
    # img2text('Consolidated.png')
    # heic2jpg("image.HEIC")
    # curl2win()
    # extract_audio("audio.mp3", output="extracted.mp3")
    # chop("audio.mp3", interval="00:01:00", output_dir="chopped_output")
    # speech2text("audio.mp3", output="output.txt", model="canary")
    # summarize(file2text("text.txt"), output="summary.txt")
    # any2jpg("image.webp")
    # crop_face_centered("image.jpg", output="cropped.jpg")
    # purge_node_modules("D:\\")
    pass
