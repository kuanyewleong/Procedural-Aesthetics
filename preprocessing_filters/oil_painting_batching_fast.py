import os
import glob
import cv2
import numpy as np

SRC_DIR = "data/flowers-102/jpg"
DST_DIR = "data/flowers-102/jpg_oil"
A = 6  # kuwahara radius (window is (2A+1), quadrants are (A+1)x(A+1))

os.makedirs(DST_DIR, exist_ok=True)

# Grab all jpg/JPG/jpeg/JPEG files in the source folder
img_paths = []
for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG"):
    img_paths.extend(glob.glob(os.path.join(SRC_DIR, ext)))
img_paths.sort()

if not img_paths:
    raise FileNotFoundError(f"No images found in: {SRC_DIR}")

def _integral_sum(ii: np.ndarray, y0: int, y1: int, x0: int, x1: int, H: int, W: int) -> np.ndarray:
    """
    Compute per-pixel sum over windows using an integral image.
    ii: integral image of shape (Hp+1, Wp+1)
    Windows are [y0:y1] and [x0:x1] in the PADDED image coordinates (inclusive ranges),
    for all output pixels laid out as HxW via slicing.

    Here we use the standard integral sum formula with slicing:
    sum = ii[y1+1, x1+1] - ii[y0, x1+1] - ii[y1+1, x0] + ii[y0, x0]
    """
    # Convert inclusive endpoints into slice endpoints (+1)
    A = ii[y1+1 : y1+1+H, x1+1 : x1+1+W]
    B = ii[y0   : y0+H,     x1+1 : x1+1+W]
    C = ii[y1+1 : y1+1+H,   x0   : x0+W]
    D = ii[y0   : y0+H,     x0   : x0+W]
    return A - B - C + D

def kuwahara_fast_gray(gray_u8: np.ndarray, a: int = 6) -> np.ndarray:
    """
    Fast Kuwahara filter for a single-channel uint8 image using integral images.
    Uses replicate padding so every pixel uses a full (A+1)x(A+1) quadrant area.
    Output is float32; caller can convert to uint8.
    """
    if gray_u8.ndim != 2:
        raise ValueError("kuwahara_fast_gray expects a single-channel 2D array")

    H, W = gray_u8.shape
    a = int(a)
    q = a + 1
    area = float(q * q)

    # Replicate-pad by a so all windows are valid and same area
    pad = cv2.copyMakeBorder(gray_u8, a, a, a, a, borderType=cv2.BORDER_REPLICATE).astype(np.float32)
    Hp, Wp = pad.shape

    # Integral images for sum and sum of squares
    ii = cv2.integral(pad)  # float64 by default
    ii2 = cv2.integral(pad * pad)

    # Quadrant coordinates in the padded image (inclusive endpoints)
    # For output pixel (i,j) in original:
    # padded center is (i+a, j+a)
    # Q1: rows (i .. i+a)??  Actually we want:
    # Q1: [pi-a .. pi], [pj-a .. pj]  (top-left)
    # Q2: [pi .. pi+a], [pj-a .. pj]  (bottom-left)
    # Q3: [pi-a .. pi], [pj .. pj+a]  (top-right)
    # Q4: [pi .. pi+a], [pj .. pj+a]  (bottom-right)
    #
    # Using slicing over all pixels at once:
    # pi ranges [a .. a+H-1], pj ranges [a .. a+W-1]
    # so the window top-left indices become fixed slice origins.

    # Top-left quadrant (Q1): y0=0, y1=a; x0=0, x1=a in the per-pixel sliding sense.
    # In slicing form:
    # Q1 uses y0 = 0..H-1 and y1 = a..a+H-1 => pass y0=0, y1=a
    # and x0 = 0..W-1 and x1 = a..a+W-1 => pass x0=0, x1=a
    sum1  = _integral_sum(ii,  0,   a,  0,   a, H, W)
    sum21 = _integral_sum(ii2, 0,   a,  0,   a, H, W)

    # Bottom-left quadrant (Q2): y0=a, y1=2a; x0=0, x1=a
    sum2  = _integral_sum(ii,  a, 2*a,  0,   a, H, W)
    sum22 = _integral_sum(ii2, a, 2*a,  0,   a, H, W)

    # Top-right quadrant (Q3): y0=0, y1=a; x0=a, x1=2a
    sum3  = _integral_sum(ii,  0,   a,  a, 2*a, H, W)
    sum23 = _integral_sum(ii2, 0,   a,  a, 2*a, H, W)

    # Bottom-right quadrant (Q4): y0=a, y1=2a; x0=a, x1=2a
    sum4  = _integral_sum(ii,  a, 2*a,  a, 2*a, H, W)
    sum24 = _integral_sum(ii2, a, 2*a,  a, 2*a, H, W)

    # Means and variances
    m1 = sum1 / area
    m2 = sum2 / area
    m3 = sum3 / area
    m4 = sum4 / area

    v1 = (sum21 / area) - (m1 * m1)
    v2 = (sum22 / area) - (m2 * m2)
    v3 = (sum23 / area) - (m3 * m3)
    v4 = (sum24 / area) - (m4 * m4)

    # Stack and pick mean from quadrant with minimum variance
    vars_ = np.stack([v1, v2, v3, v4], axis=0)   # (4,H,W)
    means = np.stack([m1, m2, m3, m4], axis=0)   # (4,H,W)

    idx = np.argmin(vars_, axis=0)              # (H,W) values 0..3
    out = np.take_along_axis(means, idx[None, ...], axis=0)[0]  # (H,W)

    return out.astype(np.float32)

def kuwahara_color_bgr(img_bgr_u8: np.ndarray, a: int = 6) -> np.ndarray:
    """
    Apply fast Kuwahara per-channel on BGR uint8 image.
    Returns uint8 BGR.
    """
    if img_bgr_u8 is None or img_bgr_u8.ndim != 3 or img_bgr_u8.shape[2] != 3:
        raise ValueError("kuwahara_color_bgr expects a uint8 BGR image")

    b = kuwahara_fast_gray(img_bgr_u8[:, :, 0], a)
    g = kuwahara_fast_gray(img_bgr_u8[:, :, 1], a)
    r = kuwahara_fast_gray(img_bgr_u8[:, :, 2], a)

    out = np.dstack([b, g, r])
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def main():
    saved = 0
    failed = 0

    # Optional: let OpenCV use CPU threads
    # cv2.setUseOptimized(True)  # usually default True
    # cv2.setNumThreads(0)       # 0 lets OpenCV decide; or set e.g. 8/16

    for in_path in img_paths:
        image = cv2.imread(in_path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"[WARN] Failed to read: {in_path}")
            failed += 1
            continue

        try:
            out_img = kuwahara_color_bgr(image, A)
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

if __name__ == "__main__":
    main()
