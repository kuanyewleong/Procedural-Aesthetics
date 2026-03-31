import json
import random
from pathlib import Path
from typing import List, Dict, Any

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

try:
    from scipy.io import loadmat
except Exception:
    loadmat = None


def build_transforms(image_size=512):
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.15), Image.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),  # [0,1]
    ])


def _find_flowers102_root(root: Path) -> Path:
    candidates = [root / "flowers-102", root / "Flowers102", root]
    for c in candidates:
        if (c / "jpg").exists() and ((c / "imagelabels.mat").exists() or (c / "setid.mat").exists()):
            return c
    if (root / "flowers-102").exists():
        return root / "flowers-102"
    return root


def _load_label_names(flowers_root: Path) -> List[str]:
    candidates = [
        flowers_root / "cat_to_name.json",
        flowers_root.parent / "cat_to_name.json",
        flowers_root / "CAT_TO_NAME.JSON",
    ]
    for p in candidates:
        if p.exists():
            data = json.loads(p.read_text())
            names = [""] * 102
            for k, v in data.items():
                i = int(k) - 1
                if 0 <= i < 102:
                    names[i] = str(v).strip()
            for i in range(102):
                if not names[i]:
                    names[i] = f"flower class {i}"
            return names
    return [f"flower class {i}" for i in range(102)]


def _load_mat_splits(flowers_root: Path):
    if loadmat is None:
        raise RuntimeError("scipy is required to read imagelabels.mat / setid.mat. Install scipy.")

    labels_path = flowers_root / "imagelabels.mat"
    setid_path = flowers_root / "setid.mat"

    if not labels_path.exists() or not setid_path.exists():
        raise FileNotFoundError(f"Missing imagelabels.mat or setid.mat under {flowers_root}")

    labels_mat = loadmat(str(labels_path))
    setid_mat = loadmat(str(setid_path))

    labels = labels_mat.get("labels", None)
    if labels is None:
        raise KeyError(f"imagelabels.mat missing 'labels'. Keys: {list(labels_mat.keys())}")

    labels = labels.reshape(-1).astype("int64")
    labels_0 = (labels - 1).tolist()  # 0..101

    def _get_ids(key: str):
        arr = setid_mat.get(key, None)
        if arr is None:
            raise KeyError(f"setid.mat missing '{key}'. Keys: {list(setid_mat.keys())}")
        arr = arr.reshape(-1).astype("int64")
        return (arr - 1).tolist()

    split_ids = {
        "train": _get_ids("trnid"),
        "val": _get_ids("valid"),
        "test": _get_ids("tstid"),
    }
    return labels_0, split_ids


class Flowers102Pointillism(Dataset):
    """
    Reads Oxford-102 split/labels from .mat, but loads images from:
      {flowers_root}/jpg_pointillism/image_XXXXX.jpg
    """
    def __init__(self, root="./data", split="train", image_size=512, seed=0, data_dirname="jpg_pointillism"):
        assert split in ["train", "val", "test"]
        self.root = Path(root)
        self.flowers_root = _find_flowers102_root(self.root)
        self.water_dir = self.flowers_root / data_dirname
        if not self.water_dir.exists():
            raise FileNotFoundError(f"Pointillism folder not found: {self.water_dir}")

        self.T = build_transforms(image_size)
        self.rng = random.Random(seed)

        self.names = _load_label_names(self.flowers_root)
        self.all_labels, self.split_ids = _load_mat_splits(self.flowers_root)
        self.indices = self.split_ids[split]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx) -> Dict[str, Any]:
        real_idx = self.indices[idx]
        label = int(self.all_labels[real_idx])  # 0..101

        name = self.names[label].replace("_", " ") if 0 <= label < len(self.names) else "flower"

        # watercolor file path must match original naming
        path = self.water_dir / f"image_{real_idx+1:05d}.jpg"

        img = Image.open(path).convert("RGB")
        img = self.T(img)

        return {
            "image": img,      # [0,1]
            "name": name,
            "label": label,
            "path": str(path),
        }
