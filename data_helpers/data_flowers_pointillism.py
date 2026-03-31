import torch
from torch.utils.data import Dataset
from data_flowers import Flowers102Clean

class Flowers102Pointillism(Dataset):
    """
    Stage-2 dataset: returns pointillism-stylized images.
    Keeps original too:
      - image_photo: original [0,1]
      - image: stylized [0,1]
    """
    def __init__(self, split="train", image_size=512, pointillize_fn=None, root="./data", seed=0):
        super().__init__()
        self.base = Flowers102Clean(root=root, split=split, image_size=image_size, seed=seed)
        assert pointillize_fn is not None, "Provide pointillize_fn(img01)->img01"
        self.pointillize_fn = pointillize_fn

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        b = self.base[idx]
        img01 = b["image"]  # torch [3,H,W] in [0,1]

        # stylize
        styl = self.pointillize_fn(img01).clamp(0, 1)

        out = dict(b)
        out["image_photo"] = img01
        out["image"] = styl
        return out
