import re
import random

def _clean_species_name(name: str) -> str:
    if name is None:
        return "flower"
    name = str(name).strip().lower().replace("_", " ")
    name = re.sub(r"\s+", " ", name)
    return name

CHAOTIC_BRUSH_STYLE_PHRASES = [
    "chaotic brushes art style",
    "expressive brushstroke painting",
    "abstract painterly brushwork",
    "wild multi-directional brush painting",
    "scattered brushstroke composition",
]

BRUSH_STRUCTURE_PHRASES = [
    "made of scattered brush strokes",
    "built from layered strokes in many directions",
    "with long and short brush marks across the image",
    "formed by irregular painterly strokes",
    "with fragmented strokes and simplified painted structure",
]

BRUSH_MOTION_PHRASES = [
    "with energetic directional strokes",
    "with chaotic strokes flowing in many directions",
    "with loose and restless brush movement",
    "with scattered brush marks crossing the scene",
    "",  # sometimes no explicit motion phrase
]

COLOR_PHRASES = [
    "bold painted color blocks",
    "reduced painterly color palette",
    "high contrast brush colors",
    "muted painted tones",
    "vibrant expressive colors",
]

TEXTURE_PHRASES = [
    "rough painterly texture",
    "visible brush textures",
    "soft broken paint texture",
    "layered stroke texture",
    "hand-painted abstract texture",
]

def make_chaotic_brushes_caption(flower_name: str, rng=None):
    """
    Canonical + ChaoticBrushes style keywords:
      "flower species: {name}. an expressive painting of a flower, {style}, {structure} {motion}, {texture}, {color}."
    """
    if rng is None:
        rng = random

    species = _clean_species_name(flower_name)
    style = rng.choice(CHAOTIC_BRUSH_STYLE_PHRASES)
    structure = rng.choice(BRUSH_STRUCTURE_PHRASES)
    motion = rng.choice(BRUSH_MOTION_PHRASES)
    texture = rng.choice(TEXTURE_PHRASES)
    color = rng.choice(COLOR_PHRASES)

    if motion:
        return (
            f"flower species: {species}. "
            f"an expressive painting of a flower, {style}, {structure} {motion}, {texture}, {color}."
        )
    else:
        return (
            f"flower species: {species}. "
            f"an expressive painting of a flower, {style}, {structure}, {texture}, {color}."
        )