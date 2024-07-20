"""
Toolbox for performing common operations offline.

FEATURES:
2024-07-12:
- Image of document to scanned image.
- Image of document to text file.

2024-07-13:
- Convert curl command to be windows compatible.

2024-07-18:
- Convert .HEIC image to .jpg

2024-07-19:
- Extract audio segment from an audio file.
- Speech to text conversion using AI model (OpenAI Whisper & Nvidia NeMo Canary 1b).
- Convert file to text string.
- Text summarize using AI model (Alibaba Qwen2 72b).
- Chop audio into intervals.
- Delete folders in the Shell.

Created: 2024-07-12
Updated: 2024-07-19
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

    def heic2jpg(path):
        """HEIC to JPG conversion using pillow_heif.
        Ref:
        https://stackoverflow.com/questions/54395735/how-to-work-with-heic-image-file-types-in-python
        """
        from pillow_heif import register_heif_opener
        register_heif_opener()
        image = Image.open(path)
        image.save(path.rstrip('.HEIC').rstrip('.heic') + '.jpg')

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
            cmd = f"rm -r {path}"
            print("Running:", cmd)
            subprocess.run(cmd, shell=True)
        else:
            raise ValueError(f"File {path} does not exist.")

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
heic2jpg = ImageOperations.heic2jpg
curl2win = TextOperations.curl2win
extract_audio = AudioOperations.extract_audio
speech2text = AudioOperations.speech2text
file2text = FileOperations.file2text
summarize = TextOperations.summarize
chop = AudioOperations.chop
delete = FileOperations.delete

if __name__ == "__main__":
    # img2scan('Consolidated.png')
    # img2text('Consolidated.png')
    # heic2jpg("image.HEIC")
    # curl2win()
    # extract_audio("audio.mp3", output="extracted.mp3")
    # chop("audio.mp3", interval="00:01:00", output_dir="chopped_output")
    # speech2text("audio.mp3", output="output.txt", model="canary")
    # summarize(file2text("text.txt"), output="summary.txt")
    pass
