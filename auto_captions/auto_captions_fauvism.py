import re
import random

def _clean_species_name(name: str) -> str:
    if name is None:
        return "flower"
    name = str(name).strip().lower().replace("_", " ")
    name = re.sub(r"\s+", " ", name)
    return name

FAUVE_STYLE_PHRASES = [
    "fauvism painting style",
    "fauvist art style",
    "in the style of fauvism",
    "bold fauvist painting",
]

FAUVE_TRAITS = [
    "wild brushwork",
    "strident vivid colors",
    "simplified abstract forms",
    "high contrast color blocks",
    "expressive painterly strokes",
    "unnatural saturated palette",
]

SCENE_PHRASES = [
    "in the wild",
    "in a garden",
    "with foliage",
    "against a simple background",
    "outdoors",
    "with bold flat background",
]

def make_fauvism_caption(
    flower_name: str,
    rng: random.Random | None = None,
    include_scene: bool = True,
):
    """
    Strong, consistent caption:
      "flower species: {name}. a fauvism painting of a flower, {traits}, {scene}."
    """
    if rng is None:
        rng = random

    species = _clean_species_name(flower_name)
    style = rng.choice(FAUVE_STYLE_PHRASES)
    # pick 2 traits for richness but keep stable structure
    traits = rng.sample(FAUVE_TRAITS, k=2)
    scene = rng.choice(SCENE_PHRASES) if include_scene else ""

    tail = ", ".join([style] + traits)
    if include_scene and scene:
        tail = tail + f", {scene}"

    return f"flower species: {species}. a fauvism painting of a flower, {tail}."
