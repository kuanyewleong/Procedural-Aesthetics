import re
import random

def _clean_species_name(name: str) -> str:
    if name is None:
        return "flower"
    name = str(name).strip().lower().replace("_", " ")
    name = re.sub(r"\s+", " ", name)
    return name

# Core style keywords (multi-flower abstract canvas / collage)
MULTIFLOWER_CANVAS_PHRASES = [
    "multi-flower abstract canvas composition",
    "abstract floral collage on a canvas",
    "layered multi-flower poster collage",
    "rhythmic multi-flower arrangement on a canvas",
    "scattered multi-flower cutout composition",
]

# Composition / layout descriptors
COMPOSITION_DETAILS = [
    "multiple round flower cutouts in varied sizes, softly blended edges",
    "six to twelve flowers arranged with rhythmic spacing and gentle overlaps",
    "scattered flower stickers with partial overlaps and balanced negative space",
    "clustered blooms and offset placements creating an abstract floral pattern",
    "layered flowers placed across the canvas with soft feathered boundaries",
]

# Background descriptors (keep it flower-focused; can mention plain/soft or harmonic tri-color)
BACKGROUND_DETAILS = [
    "on a soft harmonious tri-color background",
    "on a minimal soft gradient background with harmonious colors",
    "on a plain background with gentle color harmony",
    "on a smooth low-contrast tri-color backdrop",
    "on a softly blended abstract background with harmonious colors",
]

# Optional extra cues about the synthetic process (light touch)
PROCESS_CUES = [
    "sticker-like round cutouts with feathered edges",
    "soft masked circular crops and blended layering",
    "collage layering with clean silhouettes and smooth blending",
    "cutout-style blooms blended into the canvas",
    "",
]

def make_multiflower_canvas_caption(flower_name: str, rng=None):
    """
    Canonical species prefix + multi-flower canvas style caption.
    Intended for synthetic multi-flower abstract compositions built from same-class samples.
    """
    if rng is None:
        rng = random

    species = _clean_species_name(flower_name)
    style = rng.choice(MULTIFLOWER_CANVAS_PHRASES)
    comp = rng.choice(COMPOSITION_DETAILS)
    bg = rng.choice(BACKGROUND_DETAILS)
    cue = rng.choice(PROCESS_CUES)

    tail = f"{comp}, {bg}"
    if cue:
        tail = f"{tail}, {cue}"

    return f"flower species: {species}. a {style}, {tail}."