import cv2
import argparse
import math
import progressbar
import bisect
import scipy.spatial
import numpy as np
import random
from sklearn.cluster import KMeans

import os
import glob

# Helper functions for the pointillism pipeline
def compute_color_probabilities(pixels, palette, k=9):
    distances = scipy.spatial.distance.cdist(pixels, palette.colors)
    maxima = np.amax(distances, axis=1)

    distances = maxima[:, None] - distances
    summ = np.sum(distances, 1)
    distances /= summ[:, None]

    distances = np.exp(k*len(palette)*distances)
    summ = np.sum(distances, 1)
    distances /= summ[:, None]

    return np.cumsum(distances, axis=1, dtype=np.float32)


def color_select(probabilities, palette):
    r = random.uniform(0, 1)
    i = bisect.bisect_left(probabilities, r)
    return palette[i] if i < len(palette) else palette[-1]


def randomized_grid(h, w, scale):
    assert (scale > 0)

    r = scale//2

    grid = []
    for i in range(0, h, scale):
        for j in range(0, w, scale):
            y = random.randint(-r, r) + i
            x = random.randint(-r, r) + j

            grid.append((y % h, x % w))

    random.shuffle(grid)
    return grid

# ==========================

# Colour Pallete class
class ColorPalette:
    def __init__(self, colors, base_len=0):
        self.colors = colors
        self.base_len = base_len if base_len > 0 else len(colors)

    @staticmethod
    def from_image(img, n, max_img_size=200, n_init=10):
        # scale down the image to speedup kmeans
        img = limit_size(img, max_img_size)

        clt = KMeans(n_clusters=n, n_init=n_init)
        clt.fit(img.reshape(-1, 3))

        return ColorPalette(clt.cluster_centers_)

    def extend(self, extensions):
        extension = [regulate(self.colors.reshape((1, len(self.colors), 3)).astype(np.uint8), *x).reshape((-1, 3)) for x
                     in
                     extensions]

        return ColorPalette(np.vstack([self.colors.reshape((-1, 3))] + extension), self.base_len)

    def to_image(self):
        cols = self.base_len
        rows = int(math.ceil(len(self.colors) / cols))

        res = np.zeros((rows * 80, cols * 80, 3), dtype=np.uint8)
        for y in range(rows):
            for x in range(cols):
                if y * cols + x < len(self.colors):
                    color = [int(c) for c in self.colors[y * cols + x]]
                    cv2.rectangle(res, (x * 80, y * 80), (x * 80 + 80, y * 80 + 80), color, -1)

        return res

    def __len__(self):
        return len(self.colors)

    def __getitem__(self, item):
        return self.colors[item]

# ==========================

# Utility functions
def limit_size(img, max_x, max_y=0):
    if max_x == 0:
        return img

    if max_y == 0:
        max_y = max_x

    ratio = min(1.0, float(max_x) / img.shape[1], float(max_y) / img.shape[0])

    if ratio != 1.0:
        shape = (int(img.shape[1] * ratio), int(img.shape[0] * ratio))
        return cv2.resize(img, shape, interpolation=cv2.INTER_AREA)
    else:
        return img


def clipped_addition(img, x, _max=255, _min=0):
    if x > 0:
        mask = img > (_max - x)
        img += x
        np.putmask(img, mask, _max)
    if x < 0:
        mask = img < (_min - x)
        img += x
        np.putmask(img, mask, _min)


def regulate(img, hue=0, saturation=0, luminosity=0):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)
    if hue < 0:
        hue = 255 + hue
    hsv[:, :, 0] += hue
    clipped_addition(hsv[:, :, 1], saturation)
    clipped_addition(hsv[:, :, 2], luminosity)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR_FULL)

# ==========================

# Vector field class for the gradient
class VectorField:
    def __init__(self, fieldx, fieldy):
        self.fieldx = fieldx
        self.fieldy = fieldy

    @staticmethod
    def from_gradient(gray):
        fieldx = cv2.Scharr(gray, cv2.CV_32F, 1, 0) / 15.36
        fieldy = cv2.Scharr(gray, cv2.CV_32F, 0, 1) / 15.36

        return VectorField(fieldx, fieldy)

    def get_magnitude_image(self):
        res = np.sqrt(self.fieldx**2 + self.fieldy**2)
        
        return (res * 255/np.max(res)).astype(np.uint8)

    def smooth(self, radius, iterations=1):
        s = 2*radius + 1
        for _ in range(iterations):
            self.fieldx = cv2.GaussianBlur(self.fieldx, (s, s), 0)
            self.fieldy = cv2.GaussianBlur(self.fieldy, (s, s), 0)

    def direction(self, i, j):
        return math.atan2(self.fieldy[i, j], self.fieldx[i, j])

    def magnitude(self, i, j):
        return math.hypot(self.fieldx[i, j], self.fieldy[i, j])

# ==========================

# Main pointillism processing function
def process_image(img_path, palette_size=20, stroke_scale=0,
                  gradient_smoothing_radius=0, limit_image_size=0):
    """
    Process an image with a painterly rendering algorithm.

    Parameters
    ----------
    img_path : str
        Path to the input image.
    palette_size : int, optional
        Number of colors in the base palette (default 20).
    stroke_scale : int, optional
        Scale of brush strokes. If 0, automatically computed based on image size.
    gradient_smoothing_radius : int, optional
        Radius for smoothing the gradient field. If 0, automatically computed.
    limit_image_size : int, optional
        If > 0, resizes the image so its largest dimension does not exceed this value.

    Returns
    -------
    numpy.ndarray
        The resulting painted image.
    """
    # Load image    
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {img_path}")

    # Optional resizing
    if limit_image_size > 0:
        img = limit_size(img, limit_image_size)

    # Determine stroke scale
    if stroke_scale == 0:
        stroke_scale = int(math.ceil(max(img.shape) / 1000))
        print(f"Automatically chosen stroke scale: {stroke_scale}")
    else:
        stroke_scale = stroke_scale

    # Determine gradient smoothing radius
    if gradient_smoothing_radius == 0:
        gradient_smoothing_radius = int(round(max(img.shape) / 50))
        print(f"Automatically chosen gradient smoothing radius: {gradient_smoothing_radius}")
    else:
        gradient_smoothing_radius = gradient_smoothing_radius

    # Convert to grayscale for gradient computation
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create and extend color palette
    # print("Computing color palette...")
    palette = ColorPalette.from_image(img, palette_size)
    # print("Extending color palette...")
    palette = palette.extend([(0, 50, 0), (15, 30, 0), (-15, 30, 0)])

    # Compute and smooth gradient
    # print("Computing gradient...")
    gradient = VectorField.from_gradient(gray)
    # print("Smoothing gradient...")
    gradient.smooth(gradient_smoothing_radius)

    # Prepare base image (median blur)
    res = cv2.medianBlur(img, 11)

    # Generate random grid for strokes
    grid = randomized_grid(img.shape[0], img.shape[1], scale=3)
    batch_size = 10000

    # Draw strokes in batches
    bar = progressbar.ProgressBar()
    for h in bar(range(0, len(grid), batch_size)):
        batch_end = min(h + batch_size, len(grid))
        # Get pixel colors at grid points
        pixels = np.array([img[x[0], x[1]] for x in grid[h:batch_end]])
        # Compute color probabilities
        color_probs = compute_color_probabilities(pixels, palette, k=9)

        for i, (y, x) in enumerate(grid[h:batch_end]):
            color = color_select(color_probs[i], palette)
            angle = math.degrees(gradient.direction(y, x)) + 90
            length = int(round(stroke_scale + stroke_scale * math.sqrt(gradient.magnitude(y, x))))

            # Draw ellipse stroke
            cv2.ellipse(res, (x, y), (length, stroke_scale), angle, 0, 360,
                        color, -1, cv2.LINE_AA)

    return res


# Batching code to process all images in a folder
SRC_DIR = "data/flowers-102/jpg"
DST_DIR = "data/flowers-102/jpg_pointillism"

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
    print(in_path)
    image = cv2.imread(in_path)
    if image is None:
        print(f"[WARN] Failed to read: {in_path}")
        failed += 1
        continue

    out_img = process_image(in_path)

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