import os
import glob
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.io import loadmat

from multi_flower_canvas_modules import MultiFlowerCanvas


SRC_DIR = "data/flowers-102/jpg"
LABELS_MAT = "data/flowers-102/imagelabels.mat"
DST_DIR = "data/flowers-102/jpg_multiflower_raw"

# output canvas size (should match your later training size)
CANVAS_SIZE = 512

# how many flowers per canvas
K_MIN, K_MAX = 12, 24

# reproducibility
SEED = 0


def load_flowers102_labels(labels_mat_path: str):
    """
    Returns labels_0based: list[int] length N (N=8189), where each is 0..101.
    The ordering corresponds to image_00001.jpg ... image_08189.jpg
    """
    mat = loadmat(labels_mat_path)
    labels = mat.get("labels", None)
    if labels is None:
        raise KeyError(f"{labels_mat_path} missing 'labels'. Keys: {list(mat.keys())}")
    labels = labels.reshape(-1).astype(np.int64)  # 1..102
    labels_0 = (labels - 1).tolist()
    return labels_0


def filename_to_index_zero_based(path: str) -> int:
    """
    Oxford filenames: image_00001.jpg ... image_08189.jpg
    Returns 0-based index.
    """
    stem = Path(path).stem  # "image_00001"
    try:
        n = int(stem.split("_")[-1])  # 1..8189
        return n - 1
    except Exception as e:
        raise ValueError(f"Unexpected filename format: {path}") from e


def read_image_as_torch_rgb01(path: str, size: int, device: str) -> torch.Tensor:
    """
    Returns torch tensor [3,H,W] in [0,1], RGB, float32
    """
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"cv2 failed to read: {path}")
    bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).to(device=device, dtype=torch.float32) / 255.0  # [H,W,3]
    x = x.permute(2, 0, 1).contiguous()  # [3,H,W]
    return x


def torch_rgb01_to_bgr_uint8(img01: torch.Tensor) -> np.ndarray:
    """
    img01: [3,H,W] RGB in [0,1]
    returns BGR uint8 [H,W,3] for cv2.imwrite
    """
    x = img01.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()  # [H,W,3] RGB
    x = (x * 255.0 + 0.5).astype(np.uint8)
    bgr = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    return bgr


def main():
    os.makedirs(DST_DIR, exist_ok=True)

    rng = random.Random(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    labels_0 = load_flowers102_labels(LABELS_MAT)
    N = len(labels_0)
    print("loaded labels:", N)

    # index filepaths by class label
    # (we assume SRC_DIR contains image_00001.jpg ... image_08189.jpg)
    img_paths = []
    for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG"):
        img_paths.extend(glob.glob(os.path.join(SRC_DIR, ext)))
    img_paths.sort()
    if not img_paths:
        raise FileNotFoundError(f"No images found in: {SRC_DIR}")

    by_label = {i: [] for i in range(102)}
    for p in img_paths:
        idx0 = filename_to_index_zero_based(p)
        if idx0 < 0 or idx0 >= N:
            continue
        lab = labels_0[idx0]
        by_label[lab].append(p)

    # build synthesis module
    synth = MultiFlowerCanvas(
        canvas_size=CANVAS_SIZE,
        k_min=K_MIN,
        k_max=K_MAX,
        allow_rhythmic_prob=0.5,
        background_mode=("plain"), # ("spiral", "stripe", "box"),
        sticker_radius=0.92,
        sticker_softness=0.08,
    )

    saved = 0
    failed = 0

    for in_path in img_paths:
        base_name = os.path.basename(in_path)
        out_path = os.path.join(DST_DIR, base_name)

        try:
            idx0 = filename_to_index_zero_based(in_path)
            if idx0 < 0 or idx0 >= N:
                raise ValueError(f"index out of range for {in_path}: {idx0}")

            label = labels_0[idx0]
            pool = by_label[label]
            if len(pool) < K_MIN:
                raise RuntimeError(f"Not enough images in class {label}: {len(pool)}")

            K = rng.randint(K_MIN, min(K_MAX, len(pool)))

            # sample K images from same class; include the current image for anchoring
            chosen = [in_path]
            if K > 1:
                chosen += rng.choices(pool, k=K - 1)

            # load to torch batch [K,3,H,W]
            flowers = torch.stack(
                [read_image_as_torch_rgb01(p, CANVAS_SIZE, device=device) for p in chosen],
                dim=0
            )  # [K,3,H,W]

            # synth canvas [3,H,W]
            canvas = synth(flowers)

            # save
            out_bgr = torch_rgb01_to_bgr_uint8(canvas)
            ok = cv2.imwrite(out_path, out_bgr)
            if not ok:
                raise RuntimeError(f"cv2.imwrite failed: {out_path}")

            saved += 1
            if saved % 100 == 0:
                print(f"Saved {saved}/{len(img_paths)}...")

        except Exception as e:
            print(f"[WARN] Failed on {in_path}: {e}")
            failed += 1

    print("Done.")
    print("saved:", saved)
    print("failed:", failed)


if __name__ == "__main__":
    main()