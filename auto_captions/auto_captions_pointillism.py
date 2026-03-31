import re
import random

POINTILLISM_STYLE_PHRASES = [
    "pointillism painting",
    "neo-impressionist pointillism",
    "stippling dot painting style",
    "pointillist artwork",
    "painted with thousands of dots",
]

POINTILLISM_DETAIL_PHRASES = [
    "dense colored dots",
    "stippled texture",
    "visible brush dabs as dots",
    "high-frequency dot pattern",
    "layered dot strokes",
]

CONTEXT_PHRASES = [
    "in the wild",
    "in a garden",
    "with foliage background",
    "with soft background",
    "",
]

def _clean_species_name(name: str) -> str:
    if name is None:
        return "flower"
    name = str(name).strip().lower().replace("_", " ")
    name = re.sub(r"\s+", " ", name)
    return name

def make_pointillism_caption(flower_name: str, rng=None, include_context=True):
    """
    Canonical prefix + pointillism keywords.
    Keep the first clause EXACT across training/inference.

    Example:
      "flower species: clematis. a pointillism painting of a flower, stippled texture, dense colored dots, in the wild."
    """
    if rng is None:
        rng = random

    species = _clean_species_name(flower_name)
    style = rng.choice(POINTILLISM_STYLE_PHRASES)
    detail = rng.choice(POINTILLISM_DETAIL_PHRASES)
    ctx = rng.choice(CONTEXT_PHRASES) if include_context else ""

    if ctx:
        return f"flower species: {species}. a {style} of a flower, {detail}, {ctx}."
    else:
        return f"flower species: {species}. a {style} of a flower, {detail}."
