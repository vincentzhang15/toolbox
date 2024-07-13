"""
Toolbox for performing common operations offline.

FEATURES:
2024-07-12:
- Image of document to scanned image.
- Image of document to text file.

Created: 2024-07-12
Updated: 2024-07-12
"""

import cv2
import numpy as np
from PIL import Image
import pytesseract
from pytesseract import Output


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


img2scan = ImageOperations.img2scan
img2text = ImageOperations.img2text

if __name__ == "__main__":
    img2scan('Consolidated.png')
    # img2text('Consolidated.png')
