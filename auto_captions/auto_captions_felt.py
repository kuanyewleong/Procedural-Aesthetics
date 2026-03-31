import re
import random

def _clean_species_name(name: str) -> str:
    if name is None:
        return "flower"
    name = str(name).strip().lower().replace("_", " ")
    name = re.sub(r"\s+", " ", name)
    return name

FELT_STYLE_PHRASES = [
    "needle felt art style",
    "wool felt craft",
    "handmade felt artwork",
    "felt collage style",
    "soft felt textile art",
    "fiber art felt style",
]

FELT_DETAIL_PHRASES = [
    "soft fibrous texture",
    "visible wool fibers",
    "puffy felt shading",
    "layered felt shapes",
    "stitched edges",
    "cozy handmade texture",
]

SCENE_PHRASES = [
    "in the wild",
    "in a garden",
    "with simple background",
    "with minimal details",
    "on a soft backdrop",
    "",
]

def make_felt_caption(flower_name: str, rng=None) -> str:
    """
    Canonical + felt keywords:
      "flower species: {name}. a felt artwork of a flower, {style}, {detail}, {scene}."
    """
    if rng is None:
        rng = random

    species = _clean_species_name(flower_name)
    style = rng.choice(FELT_STYLE_PHRASES)
    detail = rng.choice(FELT_DETAIL_PHRASES)
    scene = rng.choice(SCENE_PHRASES).strip()

    if scene:
        return f"flower species: {species}. a felt artwork of a flower, {style}, {detail}, {scene}."
    return f"flower species: {species}. a felt artwork of a flower, {style}, {detail}."
