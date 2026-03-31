import re

def _clean_species_name(name: str) -> str:
    if name is None:
        return "flower"
    name = str(name).strip().lower().replace("_", " ")
    name = re.sub(r"\s+", " ", name)
    return name

POSTER_STYLE_PHRASES = [
    "poster painting style",
    "posterized painting",
    "graphic poster illustration",
    "bold poster art style",
    "flat color poster painting",
    "painterly poster style",
]

DETAIL_PHRASES = [
    "bold flat colors",
    "simplified shapes",
    "reduced color palette",
    "high contrast",
    "clean edges",
    "minimal texture",
]

def make_poster_caption(flower_name: str, rng=None):
    """
    Canonical + poster style keywords:
      "flower species: {name}. a poster painting of a flower, {style}, {detail}."
    """
    import random
    if rng is None:
        rng = random

    species = _clean_species_name(flower_name)
    style = rng.choice(POSTER_STYLE_PHRASES)
    detail = rng.choice(DETAIL_PHRASES)

    # Keep the first clause canonical and stable
    return f"flower species: {species}. a poster painting of a flower, {style}, {detail}."
