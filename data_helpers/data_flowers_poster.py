import torch
from torch.utils.data import Dataset
from data_flowers import Flowers102Clean

class Flowers102Poster(Dataset):
    """
    Stage-2 dataset: returns poster-stylized images.
    posterize_fn: callable(img01: [3,H,W] in [0,1]) -> img01
    """
    def __init__(self, split="train", image_size=512, posterize_fn=None, root="./data", seed=0):
        super().__init__()
        self.base = Flowers102Clean(root=root, split=split, image_size=image_size, seed=seed)
        assert posterize_fn is not None, "Provide posterize_fn(img01)->img01"
        self.posterize_fn = posterize_fn

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        b = self.base[idx]
        img01 = b["image"]  # torch [3,H,W] in [0,1]
        img01 = self.posterize_fn(img01)
        img01 = img01.clamp(0, 1)

        b = dict(b)
        b["image"] = img01
        return b
