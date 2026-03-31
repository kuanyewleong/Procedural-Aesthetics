import random
from torch.utils.data import Dataset

from data_flowers import Flowers102Clean
from auto_captions.auto_captions_postimp import make_postimp_caption

class Flowers102PostImpressionist(Dataset):
    """
    Stage-2 dataset:
      - loads Flowers102Clean sample
      - applies procedural post-impressionist stylization to image
      - returns stylized caption with canonical species prefix
    """
    def __init__(
        self,
        root="./data",
        split="train",
        image_size=512,
        seed=0,
        stylize_fn=None,
        caption_seed_offset=12345,
        keep_original=False,
    ):
        super().__init__()
        assert stylize_fn is not None, "Provide stylize_fn(img01)->img01 (torch, [3,H,W], [0,1])"
        self.base = Flowers102Clean(root=root, split=split, image_size=image_size, seed=seed)
        self.stylize_fn = stylize_fn
        self.keep_original = keep_original

        # deterministic-ish caption RNG per dataset instance
        self.rng = random.Random(seed + caption_seed_offset)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        b = self.base[idx]
        img01 = b["image"]      # [3,H,W] in [0,1]
        name = b["name"]
        label = b["label"]

        img_styled = self.stylize_fn(img01).clamp(0, 1)

        # caption: canonical species + postimp keywords
        caption = make_postimp_caption(name, rng=self.rng)

        out = {
            "image": img_styled,
            "text": caption,
            "name": name,
            "label": label,
            "path": b.get("path", ""),
        }
        if self.keep_original:
            out["image_orig"] = img01

        return out
