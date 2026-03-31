import re
import random

def _clean_species_name(name: str) -> str:
    if name is None:
        return "flower"
    name = str(name).strip().lower().replace("_", " ")
    name = re.sub(r"\s+", " ", name)
    return name

WATERCOLOR_PHRASES = [
    "wet-on-wet watercolor painting",
    "soft watercolor washes",
    "watercolor illustration",
    "delicate watercolor wash",
    "watercolor on paper",
]

DETAILS = [
    "soft bleeding pigments",
    "gentle gradients and blooms",
    "paper texture and light granulation",
    "subtle edges with watery diffusion",
    "transparent layered washes",
]

def make_watercolor_caption(flower_name: str, rng=None):
    if rng is None:
        rng = random
    species = _clean_species_name(flower_name)
    style = rng.choice(WATERCOLOR_PHRASES)
    detail = rng.choice(DETAILS)
    return f"flower species: {species}. a {style} of a flower, {detail}."
