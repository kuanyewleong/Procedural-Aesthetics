import re
import random

def _clean_species_name(name: str) -> str:
    if name is None:
        return "flower"
    name = str(name).strip().lower().replace("_", " ")
    name = re.sub(r"\s+", " ", name)
    return name

OIL_STYLE_PHRASES = [
    "oil painting",
    "painterly oil artwork",
    "brush-stroke oil painting",
    "oil on canvas style",
    "impasto oil painting",
]

BRUSH_PHRASES = [
    "visible brush strokes",
    "directional brush strokes",
    "thick paint strokes",
    "layered brushwork",
    "expressive brushwork",
]

FILTER_PHRASES = [
    "anisotropic kuwahara filter",
    "kuwahara painting filter",
    "edge-preserving painterly filter",
    "anisotropic painterly smoothing",
]

CANVAS_PHRASES = [
    "canvas texture",
    "subtle canvas grain",
    "painted texture",
    "rich pigment texture",
    "soft canvas weave",
]

LIGHT_PHRASES = [
    "soft natural light",
    "studio lighting",
    "warm lighting",
    "gentle highlights",
    "moody lighting",
]

COMPOSITION_PHRASES = [
    "close-up",
    "macro view",
    "botanical composition",
    "single bloom",
    "flowers in the wild",
    "in a garden",
]

def make_oil_caption(
    flower_name: str,
    rng=None,
    include_filter_keyword: bool = True,
):
    """
    Canonical + oil painting keywords:
      "flower species: {name}. an oil painting of a flower, {style}, {brush}, {canvas}, {light}, {comp}, {filter}."
    Keep the canonical prefix stable for species binding.
    """
    if rng is None:
        rng = random

    species = _clean_species_name(flower_name)

    style = rng.choice(OIL_STYLE_PHRASES)
    brush = rng.choice(BRUSH_PHRASES)
    canvas = rng.choice(CANVAS_PHRASES)
    light = rng.choice(LIGHT_PHRASES)
    comp = rng.choice(COMPOSITION_PHRASES)

    parts = [style, brush, canvas, light, comp]
    if include_filter_keyword:
        parts.append(rng.choice(FILTER_PHRASES))

    suffix = ", ".join(parts)

    # canonical prefix first
    return f"flower species: {species}. an oil painting of a flower, {suffix}."
