import os
import glob
import cv2
import numpy as np

SRC_DIR = "data/flowers-102/jpg"
DST_DIR = "data/flowers-102/jpg_pencil"

os.makedirs(DST_DIR, exist_ok=True)

# Grab all jpg/JPG/jpeg/JPEG files in the source folder
img_paths = []
for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG"):
    img_paths.extend(glob.glob(os.path.join(SRC_DIR, ext)))
img_paths.sort()

if not img_paths:
    raise FileNotFoundError(f"No images found in: {SRC_DIR}")

#PENCIL SKETCH EFFECT
def dodgeV2(image,mask):
    return cv2.divide(image,255-mask,scale=256)

def sketch_process(image: np.ndarray) -> np.ndarray:
    #reading imagae from file
    img = image
    kernel_sharpening = np.array([[-1,-1,-1], 
                                [-1, 9,-1],
                                [-1,-1,-1]])
    sharpened = cv2.filter2D(img,-1,kernel_sharpening)
    gray = cv2.cvtColor(sharpened , cv2.COLOR_BGR2GRAY)
    inv = 255-gray
    gaussgray = cv2.GaussianBlur(inv,ksize=(15,15),sigmaX=0,sigmaY=0)
    
    pencil_img = dodgeV2(gray,gaussgray)

    # darken / thicken lines by applying a binary threshold
    inv_pencil = 255 - pencil_img
    kernel = np.ones((2, 2), np.uint8)
    inv_pencil = cv2.dilate(inv_pencil, kernel, iterations=1)
    inv_pencil = cv2.convertScaleAbs(inv_pencil, alpha=1.4, beta=0)
    pencil_img = 255 - inv_pencil
    
    return pencil_img


saved = 0
failed = 0

for in_path in img_paths:
    image = cv2.imread(in_path)
    if image is None:
        print(f"[WARN] Failed to read: {in_path}")
        failed += 1
        continue

    out_img = sketch_process(image)

    # Keep the original filename, save into destination folder
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