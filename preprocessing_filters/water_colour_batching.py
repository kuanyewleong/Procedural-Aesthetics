import os
import glob
import cv2
import numpy as np

SRC_DIR = "data/flowers-102/jpg"
DST_DIR = "data/flowers-102/jpg_watercolor"

os.makedirs(DST_DIR, exist_ok=True)

# Grab all jpg/JPG/jpeg/JPEG files in the source folder
img_paths = []
for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG"):
    img_paths.extend(glob.glob(os.path.join(SRC_DIR, ext)))
img_paths.sort()

if not img_paths:
    raise FileNotFoundError(f"No images found in: {SRC_DIR}")

def watercolor_process(image: np.ndarray) -> np.ndarray:
    # resize image - using cubic interpolation (fx=fy=1 keeps size, but matches your pipeline)
    image_resize = cv2.resize(image, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)

    # removing impurity
    image_cleared = cv2.medianBlur(image_resize, 3)
    image_cleared = cv2.medianBlur(image_cleared, 3)
    image_cleared = cv2.medianBlur(image_cleared, 3)

    image_cleared = cv2.edgePreservingFilter(image_cleared, sigma_s=100, sigma_r=0.5)

    # bilateralFilter
    image_filtered = cv2.bilateralFilter(image_cleared, 3, 10, 5)

    for _ in range(2):
        image_filtered = cv2.bilateralFilter(image_filtered, 3, 20, 10)

    for _ in range(3):
        image_filtered = cv2.bilateralFilter(image_filtered, 5, 30, 10)

    # sharpening image
    gaussian_mask = cv2.GaussianBlur(image_filtered, (7, 7), 5)
    image_sharp = cv2.addWeighted(image_filtered, 1.5, gaussian_mask, -0.5, 0)
    image_sharp = cv2.addWeighted(image_sharp, 1.4, gaussian_mask, -0.2, 10)

    return image_sharp

saved = 0
failed = 0

for in_path in img_paths:
    image = cv2.imread(in_path)
    if image is None:
        print(f"[WARN] Failed to read: {in_path}")
        failed += 1
        continue

    out_img = watercolor_process(image)

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
