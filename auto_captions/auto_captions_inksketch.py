import re
import random

def _clean_species_name(name: str) -> str:
    if name is None:
        return "flower"
    name = str(name).strip().lower().replace("_", " ")
    name = re.sub(r"\s+", " ", name)
    return name

INK_STYLE_PHRASES = [
    "ink sketch",
    "pen and ink drawing",
    "cross-hatching ink sketch",
    "hatching illustration",
    "engraving-style line art",
    "ink crosshatch drawing",
]

INK_DETAIL_PHRASES = [
    "fine hatching lines",
    "dense cross-hatching in shadows",
    "line shading",
    "monochrome ink",
    "paper texture",
    "high-contrast ink lines",
]

CONTEXT_PHRASES = [
    "in the wild",
    "in a garden",
    "with foliage",
    "with a soft background",
    "on a stem",
    "",
]

def make_inksketch_caption(flower_name: str, rng=None):
    """
    Canonical + ink-sketch style keywords:
      "flower species: {name}. an ink sketch of a flower, {style}, {detail} {context}."
    """
    if rng is None:
        rng = random

    species = _clean_species_name(flower_name)
    style = rng.choice(INK_STYLE_PHRASES)
    detail = rng.choice(INK_DETAIL_PHRASES)
    context = rng.choice(CONTEXT_PHRASES).strip()

    prefix = f"flower species: {species}. an ink sketch of a flower, {style}, {detail}"
    if context:
        prefix += f", {context}"
    return prefix + "."
