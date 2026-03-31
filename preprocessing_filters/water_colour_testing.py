import cv2
import numpy as np

# reading image
image = cv2.imread('data/flowers-102/jpg/image_00055.jpg')

# resize image - using cubic interpolation
image_resize = cv2.resize(image, None, fx=1, fy=1)

# removing impurity
image_cleared = cv2.medianBlur(image_resize, 3)
image_cleared = cv2.medianBlur(image_cleared, 3)
image_cleared = cv2.medianBlur(image_cleared, 3)

image_cleared = cv2.edgePreservingFilter(image_cleared, sigma_s = 100, sigma_r = 0.5)

# bilateralFilter
image_filtered = cv2.bilateralFilter(image_cleared, 3, 10, 5)

for _ in range(2):
    image_filtered = cv2.bilateralFilter(image_filtered, 3, 20, 10)

for _ in range(3):
    image_filtered = cv2.bilateralFilter(image_filtered, 5, 30, 10)

# sharpening image
gaussian_mask = cv2.GaussianBlur(image_filtered, (7,7), 5)
image_sharp = cv2.addWeighted(image_filtered, 1.5, gaussian_mask, -0.5, 0)
image_sharp = cv2.addWeighted(image_sharp, 1.4, gaussian_mask, -0.2, 10)

# resize to 512x512
h, w = image_sharp.shape[:2]
# scale so that both dimensions are >= 512 (cover)
scale = max(512 / w, 512 / h)
nw, nh = int(round(w * scale)), int(round(h * scale))

resized = cv2.resize(image_sharp, (nw, nh), interpolation=cv2.INTER_LANCZOS4)

# center crop to 512x512
x0 = (nw - 512) // 2
y0 = (nh - 512) // 2
cropped = resized[y0:y0+512, x0:x0+512]

# save sharpened & resized image
out_path = "flowers_00055_watercolor.jpg"
ok = cv2.imwrite(out_path, cropped)

if not ok:
    raise IOError(f"Failed to write output image to: {out_path}")

print(f"Saved: {out_path}")
