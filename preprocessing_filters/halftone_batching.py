import re
import cv2
import os
import glob
import numpy as np
from math import ceil
from os import path

def halftone(img, side=20, jump=None, bg_color=(255,255,255), fg_color=(0,0,0),
            alpha=1.0, invert=False):
    assert side > 0
    assert alpha > 0

    # img is expected grayscale (H,W)
    if img.ndim != 2:
        raise ValueError("Expected a grayscale image (H,W). Use cv2.IMREAD_GRAYSCALE or convert first.")

    height, width = img.shape
    if jump is None:
        jump = ceil(min(height, width) * 0.007)
    assert jump > 0

    # OpenCV uses BGR for 3-channel images, but black/white don't care; keep consistent anyway:
    bg = tuple(bg_color[::-1])
    fg = tuple(fg_color[::-1])

    out_h = side * ceil(height / jump)
    out_w = side * ceil(width / jump)

    canvas = np.empty((out_h, out_w, 3), np.uint8)
    canvas[:] = bg  # ensure background everywhere

    half = side // 2
    for oy, y in enumerate(range(0, height, jump)):
        for ox, x in enumerate(range(0, width, jump)):
            block = img[y:y+jump, x:x+jump]
            if block.size == 0:
                continue

            mean = float(block.mean())  # 0..255
            t = mean / 255.0            # 0..1

            # bigger dots for darker pixels by default
            intensity = t if invert else (1.0 - t)

            # compute radius and clamp
            r = int(round(alpha * intensity * half))
            r = max(0, min(r, half))

            # draw directly on the right tile region
            tile = canvas[oy*side:(oy+1)*side, ox*side:(ox+1)*side]
            tile[:] = bg
            if r > 0:
                cv2.circle(tile, (half, half), r, fg, -1)

    return canvas


def square_avg_value(square):
    '''
    Calculates the average grayscale value of the pixels in a square of the 
    original image 
    Argument:
        square: List of N lists, each with N integers whose value is between 0 
        and 255
    Returns:
        float: Average grayscale value
    '''
    total = 0
    n = 0
    for row in square:
        for pixel in row:
            total += pixel
            n += 1
    return total / n


def str_to_rgb(str_val):
    '''
    Receives a string with a rgb value and returns a tuple with the 
    corresponding rgb value
    '''
    split = str_val[1:-1].split(",")
    return (int(split[0]), int(split[1]), int(split[2]))


def validate_rgb_string(rgb_str):
    '''
    Validates if a string contains a valid RGB value
    Returns the tuple if valid, raises ValueError if invalid
    '''
    pat = re.compile(r"\([0-9]{1,3},[0-9]{1,3},[0-9]{1,3}\)")
    rgb_str = rgb_str.replace(" ", "")
    if (pat.match(rgb_str) and
        int(rgb_str[1:-1].split(",")[0]) < 256 and
        int(rgb_str[1:-1].split(",")[1]) < 256 and
        int(rgb_str[1:-1].split(",")[2]) < 256):
        return str_to_rgb(rgb_str)
    raise ValueError(f"Invalid RGB value: {rgb_str}")


if __name__ == "__main__":
    SRC_DIR = "data/flowers-102/jpg"
    DST_DIR = "data/flowers-102/jpg_halftone"    

    os.makedirs(DST_DIR, exist_ok=True)

    # Grab all jpg/JPG/jpeg/JPEG files in the source folder
    img_paths = []
    for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG"):
        img_paths.extend(glob.glob(os.path.join(SRC_DIR, ext)))
    img_paths.sort()

    if not img_paths:
        raise FileNotFoundError(f"No images found in: {SRC_DIR}")

    saved = 0
    failed = 0

    for in_path in img_paths:
        print("Halftone for image", in_path)
        image = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"[WARN] Failed to read: {in_path}")
            failed += 1
            continue

        try:
            out_img = halftone(
                    image, 
                    side=2, 
                    jump=2, 
                    bg_color=(255, 255, 255), 
                    fg_color=(0, 0, 0), 
                    alpha=1.0
                )
        except Exception as e:
            print(f"[WARN] Failed to process {in_path}: {e}")
            failed += 1
            continue

        base_name = os.path.basename(in_path)
        out_path = os.path.join(DST_DIR, base_name)

        ok = cv2.imwrite(out_path, out_img)
        if not ok:
            print(f"[WARN] Failed to write: {out_path}")
            failed += 1
            continue

        saved += 1
        if saved % 100 == 0:
            print(f"Saved {saved}/{len(img_paths)}...")

    print(f"Done. Saved: {saved}, Failed: {failed}, Total: {len(img_paths)}")
    print(f"Output folder: {DST_DIR}")