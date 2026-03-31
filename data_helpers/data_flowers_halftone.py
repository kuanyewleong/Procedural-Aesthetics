import torch
from torch.utils.data import Dataset
from data_flowers import Flowers102Clean

class Flowers102Halftone(Dataset):
    """
    Stage-2 dataset: Oxford Flowers-102 -> halftone stylized images.
    Expects halftone_fn(img01:[3,H,W] in [0,1]) -> img01 in [0,1]
    """
    def __init__(self, split="train", image_size=512, halftone_fn=None, root="./data", seed=0):
        super().__init__()
        self.base = Flowers102Clean(root=root, split=split, image_size=image_size, seed=seed)
        assert halftone_fn is not None, "Provide halftone_fn(img01)->img01"
        self.halftone_fn = halftone_fn

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        b = self.base[idx]
        img01 = b["image"]  # torch [3,H,W] in [0,1]
        img01 = self.halftone_fn(img01).clamp(0, 1)

        out = dict(b)
        out["image"] = img01
        # keep: out["name"], out["label"], out["path"], etc.
        return out
