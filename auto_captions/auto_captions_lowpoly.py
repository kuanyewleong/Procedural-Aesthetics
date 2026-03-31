import re
import random

def _clean_species_name(name: str) -> str:
    if name is None:
        return "flower"
    name = str(name).strip().lower().replace("_", " ")
    name = re.sub(r"\s+", " ", name)
    return name

LOWPOLY_STYLE_PHRASES = [
    "low poly art style",
    "low-poly 3D-inspired illustration",
    "geometric low poly rendering",
    "triangulated low poly artwork",
    "polygonal low poly illustration",
]

TECH_PHRASES = [
    "delaunay triangulation mesh",
    "triangular facets",
    "flat-shaded triangles",
    "faceted polygon mesh",
    "triangulated surface",
]

VISUAL_PHRASES = [
    "clean geometric shapes",
    "hard edges and flat colors",
    "simplified forms",
    "angular facets with bold color blocks",
    "crisp polygon boundaries",
]

SCENE_PHRASES = [
    "in the wild",
    "in a garden",
    "with a simple background",
    "with natural foliage",
    "",
]

def make_lowpoly_caption(
    flower_name: str,
    rng=None,
    include_tech=True,
    include_scene=True,
):
    """
    Canonical + low-poly keywords.

    Example:
      "flower species: clematis. a low poly art style illustration of a flower, delaunay triangulation mesh,
       flat-shaded triangles, crisp polygon boundaries, in the wild."
    """
    if rng is None:
        rng = random

    species = _clean_species_name(flower_name)
    style = rng.choice(LOWPOLY_STYLE_PHRASES)
    visual = rng.choice(VISUAL_PHRASES)
    scene = rng.choice(SCENE_PHRASES) if include_scene else ""

    if include_tech:
        tech = rng.choice(TECH_PHRASES)
        caption = f"flower species: {species}. a {style} illustration of a flower, {tech}, {visual}"
    else:
        caption = f"flower species: {species}. a {style} illustration of a flower, {visual}"

    if scene:
        caption += f", {scene}"
    caption += "."
    return caption
