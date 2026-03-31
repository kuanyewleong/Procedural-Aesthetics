import torch
from torch.utils.data import Dataset
from data_flowers import Flowers102Clean
from auto_captions.auto_captions_fauvism import make_fauvism_caption

class Flowers102Fauvism(Dataset):
    """
    Stage-2 fauvism dataset:
      - base image from Flowers102Clean
      - run fauvism stylizer
      - caption includes canonical species prefix + fauvism style keywords
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

        # canonical fauvism caption based on true species
        name = b["name"]
        caption = make_fauvism_caption(name, rng=self.rng)

        # return new sample
        out = dict(b)
        out["image"] = img01
        out["text"] = caption
        return out
