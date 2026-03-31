import re
import random

def _clean_species_name(name: str) -> str:
    if name is None:
        return "flower"
    name = str(name).strip().lower().replace("_", " ")
    name = re.sub(r"\s+", " ", name)
    return name

Lining_STYLE_PHRASES = [
    "lining art style",
    "square tile lining",
    "tiled lining artwork",
    "lining tile composition",
    "pixelated lining style",
]

TILE_DETAIL_PHRASES = [
    "made of small lines",
    "tiled with discrete square pieces",
    "each square filled with average local color",
    "flat colored tiles with simplified shading",
    "line-based rendering with reduced detail",
]

GROUT_PHRASES = [
    "with visible grout lines",
    "with thin grout lines between tiles",
    "with dark grout separating tiles",
    "with subtle grout outlines",
    "",  # sometimes no grout mention
]

COLOR_PHRASES = [
    "bold color blocks",
    "reduced color palette",
    "high contrast tile colors",
    "muted tile colors",
    "vibrant tile colors",
]

def make_lining_caption(flower_name: str, rng=None):
    """
    Canonical + lining style keywords:
      "flower species: {name}. a lining artwork of a flower, {style}, {tile_detail} {grout}, {color}."
    """    
    if rng is None:
        rng = random

    species = _clean_species_name(flower_name)
    style = rng.choice(Lining_STYLE_PHRASES)
    tile_detail = rng.choice(TILE_DETAIL_PHRASES)
    grout = rng.choice(GROUT_PHRASES)
    color = rng.choice(COLOR_PHRASES)

    # Keep canonical species prefix stable for class binding
    # Keep the "mosaic" terms explicit and redundant (helps learning)
    if grout:
        return f"flower species: {species}. a mosaic artwork of a flower, {style}, {tile_detail} {grout}, {color}."
    else:
        return f"flower species: {species}. a mosaic artwork of a flower, {style}, {tile_detail}, {color}."
