import re
import random

HALFTONE_STYLE_PHRASES = [
    "halftone comic book style",
    "pop art halftone print",
    "lichtenstein style pop art",
    "newspaper halftone printing",
    "retro print halftone dots",
    "comic book pop art print",
]

DETAIL_PHRASES = [
    "dot screen shading",
    "ink outlines",
    "high contrast print",
    "flat graphic colors",
    "bold comic shading",
    "old newspaper texture",
]

def _clean_species_name(name: str) -> str:
    if name is None:
        return "flower"
    name = str(name).strip().lower().replace("_", " ")
    name = re.sub(r"\s+", " ", name)
    return name

def make_halftone_caption(flower_name: str, rng=None):
    """
    Canonical prefix + halftone style keywords.
    Example:
      "flower species: clematis. a halftone comic book print of a flower, pop art halftone print, dot screen shading."
    """
    if rng is None:
        rng = random
    species = _clean_species_name(flower_name)
    style = rng.choice(HALFTONE_STYLE_PHRASES)
    detail = rng.choice(DETAIL_PHRASES)
    return f"flower species: {species}. a halftone comic book print of a flower, {style}, {detail}."
