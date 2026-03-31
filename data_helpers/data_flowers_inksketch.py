import torch
from torch.utils.data import Dataset
from data_flowers import Flowers102Clean

class Flowers102InkSketch(Dataset):
    """
    Stage-2 dataset: applies an ink-sketch procedural pipeline to images.
    sketch_fn: callable(img01: [3,H,W] in [0,1]) -> img01
    """
    def __init__(self, split="train", image_size=512, sketch_fn=None, root="./data", seed=0):
        super().__init__()
        self.base = Flowers102Clean(root=root, split=split, image_size=image_size, seed=seed)
        assert sketch_fn is not None, "Provide sketch_fn(img01)->img01"
        self.sketch_fn = sketch_fn

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        b = self.base[idx]
        img01 = b["image"]  # torch [3,H,W] in [0,1]

        # apply procedural ink sketch (torch, GPU-friendly if moved later)
        img01 = self.sketch_fn(img01).clamp(0, 1)

        b = dict(b)
        b["image"] = img01
        return b
