import re
import random

def _clean_species_name(name: str) -> str:
    if name is None:
        return "flower"
    name = str(name).strip().lower().replace("_", " ")
    name = re.sub(r"\s+", " ", name)
    return name

POSTIMP_STYLE_PHRASES = [
    "post-impressionist painting",
    "post impressionism style painting",
    "post-impressionist artwork",
]

POSTIMP_TRAITS = [
    "visible brush strokes",
    "painterly brushwork",
    "thick paint strokes",
    "expressive color",
    "vivid color palette",
    "colorful textured paint",
    "soft blended edges",
]

COMPOSITION_HINTS = [
    "plein air feeling",
    "natural light",
    "organic shapes",
    "stylized painterly look",
    "artistic interpretation",
]

def make_postimp_caption(flower_name: str, rng: random.Random | None = None):
    """
    Strong canonical conditioning + post-impressionist keywords.
    """
    rng = rng or random
    species = _clean_species_name(flower_name)

    style = rng.choice(POSTIMP_STYLE_PHRASES)
    trait1 = rng.choice(POSTIMP_TRAITS)
    trait2 = rng.choice(POSTIMP_TRAITS)
    hint = rng.choice(COMPOSITION_HINTS)

    # Canonical prefix MUST be stable
    # Keep "a painting of a flower" stable too.
    return f"flower species: {species}. a painting of a flower, {style}, {trait1}, {trait2}, {hint}."
