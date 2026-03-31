# data_flowers.py
import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

try:
    from scipy.io import loadmat
except Exception:
    loadmat = None

CAP_TEMPLATES = [
    "a photograph of a {name} flower",
    "a close-up macro photo of a {name} blossom",
    "a botanical photo of {name} petals and stamens",
    "a natural scene featuring {name} flowers",
    "a high-detail photo of {name} with soft background",
]

# def build_transforms(image_size=256):
#     return transforms.Compose([
#         transforms.Resize(int(image_size * 1.15), Image.BICUBIC),
#         transforms.CenterCrop(image_size),
#         transforms.ToTensor(),  # [0,1]
#     ])

def build_transforms(image_size=256, train: bool = True):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.7, 1.0),
                ratio=(0.9, 1.1),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02
            ),
            transforms.ToTensor(),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.15), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])

def _find_flowers102_root(root: Path) -> Path:
    """
    Torchvision typically downloads into:
      root/flowers-102/
        jpg/
        imagelabels.mat
        setid.mat
    We try to locate that folder robustly.
    """
    candidates = [
        root / "flowers-102",
        root / "Flowers102",
        root,
    ]
    for c in candidates:
        if (c / "jpg").exists() and ((c / "imagelabels.mat").exists() or (c / "setid.mat").exists()):
            return c
    # last resort: if root/flowers-102 exists but not fully downloaded
    if (root / "flowers-102").exists():
        return root / "flowers-102"
    return root

def _load_label_names(flowers_root: Path) -> List[str]:
    """
    Priority:
      1) cat_to_name.json (most common)
      2) fallback "flower class {i}"
    Returns list length 102 indexed by 0..101
    """
    candidates = [
        flowers_root / "cat_to_name.json",
        flowers_root.parent / "cat_to_name.json",
        flowers_root / "CAT_TO_NAME.JSON",
    ]
    for p in candidates:
        if p.exists():
            try:
                data = json.loads(p.read_text())
                names = [""] * 102
                for k, v in data.items():
                    # common format: keys "1".."102"
                    i = int(k) - 1
                    if 0 <= i < 102:
                        names[i] = str(v).strip()
                for i in range(102):
                    if not names[i]:
                        names[i] = f"flower class {i}"
                return names
            except Exception:
                pass
    return [f"flower class {i}" for i in range(102)]

def _load_mat_splits(flowers_root: Path):
    """
    Returns:
      labels_0based: List[int] length N (N=8189)
      split_indices_0based: Dict[str, List[int]] for keys train/val/test
      image_paths: List[Path] length N
    """
    if loadmat is None:
        raise RuntimeError("scipy is required to read imagelabels.mat / setid.mat. Install scipy.")

    labels_path = flowers_root / "imagelabels.mat"
    setid_path = flowers_root / "setid.mat"
    jpg_dir = flowers_root / "jpg"

    if not labels_path.exists() or not setid_path.exists() or not jpg_dir.exists():
        raise FileNotFoundError(
            f"Could not find Oxford-102 files under {flowers_root}. "
            f"Need jpg/, imagelabels.mat, setid.mat"
        )

    labels_mat = loadmat(str(labels_path))
    setid_mat = loadmat(str(setid_path))

    # labels: shape (1, N) or (N, 1), values 1..102
    labels = labels_mat.get("labels", None)
    if labels is None:
        # sometimes key differs; common is 'labels'
        raise KeyError(f"imagelabels.mat does not contain 'labels' key. Keys: {list(labels_mat.keys())}")

    labels = labels.reshape(-1).astype("int64")
    labels_0 = (labels - 1).tolist()

    # setid.mat typically has 'trnid', 'valid', 'tstid' as 1-based indices into images
    def _get_ids(key: str) -> List[int]:
        arr = setid_mat.get(key, None)
        if arr is None:
            raise KeyError(f"setid.mat missing '{key}'. Keys: {list(setid_mat.keys())}")
        arr = arr.reshape(-1).astype("int64")
        return (arr - 1).tolist()  # 0-based

    train_ids = _get_ids("trnid")
    val_ids   = _get_ids("valid")
    test_ids  = _get_ids("tstid")

    # images are named image_00001.jpg ... image_08189.jpg
    # indices 0..8188 correspond to image_(idx+1)
    image_paths = [(jpg_dir / f"image_{i+1:05d}.jpg") for i in range(len(labels_0))]

    return labels_0, {"train": train_ids, "val": val_ids, "test": test_ids}, image_paths

class Flowers102Clean(Dataset):
    """
    Clean loader that reads the Oxford-102 canonical .mat files (labels + splits),
    and generates captions ONLY from dataset labels (no pixel-derived auto-captions).
    """
    def __init__(self, root="./data", split="train", image_size=256, seed=0, download=False):
        assert split in ["train", "val", "test"]
        self.root = Path(root)
        self.flowers_root = _find_flowers102_root(self.root)

        if download:
            # We deliberately avoid downloading here to keep behavior explicit/clean-room.
            # If you want download, do it once with torchvision.datasets.Flowers102(..., download=True).
            pass

        self.T = build_transforms(image_size, train=(split == "train"))
        self.rng = random.Random(seed)

        self.names = _load_label_names(self.flowers_root)

        labels_0, split_ids, image_paths = _load_mat_splits(self.flowers_root)

        self.all_labels = labels_0
        self.all_paths = image_paths
        self.indices = split_ids[split]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx) -> Dict[str, Any]:
        real_idx = self.indices[idx]
        path = self.all_paths[real_idx]
        label = int(self.all_labels[real_idx])  # 0..101
        
        if 0 <= label < len(self.names):
            name = self.names[label].replace("_", " ")
        else:
            name = "flower"

        # dataset-driven caption (not pixel-driven)
        caption = self.rng.choice(CAP_TEMPLATES).format(name=name)        

        img = Image.open(path).convert("RGB")
        img = self.T(img)  # [0,1]

        return {
            "image": img,
            "text": caption,
            "name": name,
            "label": label,
            "path": str(path),
        }
