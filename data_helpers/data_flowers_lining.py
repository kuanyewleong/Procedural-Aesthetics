import torch
from torch.utils.data import Dataset
from data_helpers.data_flowers import Flowers102Clean
from auto_captions.auto_captions_lining import make_lining_caption

class Flowers102Lining(Dataset):
    """
    Stage-2 lining dataset:
      - base image from Flowers102Clean
      - run lining stylizer
      - caption includes canonical species prefix + lining style keywords
    """
    def __init__(
        self,
        root="./data",
        split="train",
        image_size=512,
        seed=0,
        stylize_fn=None,
        caption_rng_seed=0,
    ):
        super().__init__()
        self.base = Flowers102Clean(root=root, split=split, image_size=image_size, seed=seed)
        assert stylize_fn is not None, "Provide stylize_fn(img01)->img01 for felt style."
        self.stylize_fn = stylize_fn

        # deterministic-ish caption randomness (optional)
        import random
        self.rng = random.Random(caption_rng_seed)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        b = self.base[idx]
        img01 = b["image"]  # [3,H,W], [0,1]

        # apply fauvism stylization (must return [3,H,W] in [0,1])
        img01 = self.stylize_fn(img01).clamp(0.0, 1.0)

        # canonical lining caption based on true species
        name = b["name"]
        caption = make_lining_caption(name, rng=self.rng)

        # return new sample
        out = dict(b)
        out["image"] = img01
        out["text"] = caption
        return out
