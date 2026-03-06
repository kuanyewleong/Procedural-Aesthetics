# Generating Zero-Shot Aesthetics: Learning Pictorial Style from Procedural Filters — Not Art Datasets
## (A Procedural Style Anchoring Approach to Visual Art Generation)

1. ### The Problem: The "Imitation Trap" in Current AI Art
    Current text-to-image models like Stable Diffusion are incredible, but they have a fundamental flaw: they can only imitate styles they've already seen in their massive, often copyrighted, training data. This creates:

    - Legal & Ethical Risks: Dependence on copyrighted artwork.

    - Data Bias: Styles are limited to what's in the datasets (online / digitized archives).

2. ### Our Research Questions: 
    1. Can a text‑to‑image model learn by procedural filters to generate a recognizable visual art style without ever being trained on artwork samples?
    1. Can the ensemble of several models trained with procedural filters of different art style create some emerging art styles?

    We aim to build a "clean-slate" foundation model that learns the principles of a style from first principles.

3. ### Methodology: Procedural Style Anchoring
    1. Stage 0 – Representation Learning
    Train a VAE denoiser on Flower‑102 to learn latent representations of real flowers.

    1. Stage 1 – Text‑to‑Image Learning
    Train a UNet‑based diffusion model coupled with Google’s T5 text‑to‑text model, enabling text‑conditioned image generation.

    1. Stage 2 – Style Emergence via Procedural Bias
    Fine‑tune the UNet by applying procedural visual filters (e.g., poster‑colour effects) to the Flower‑102 images, introducing stylistic bias without using any artwork samples / models.

    1. Stage 3 - The Ensemble Artistry
    Multiple independently trained style-anchored models are ensembled during inference, allowing the final image to arise from negotiation between stylistic “personalities” — a controlled, generative tension rather than a single, fixed aesthetic. 

    All models were trained from scratch to avoid inherited artistic priors.

4. ### Core Setup

    - Subject-matter: flower-themed visual art (painting-like imagery)
    - Data source: Oxford Flower-102 — only real flower photographs
    - Primary constraint: no artwork images used at any stage (no art datasets, no art-trained models, text-conditioning T5 is pure language model)


# (codes are not revealed yet as there is a review process underway)
# Installation
pip install -r requirements.txt

# Models Training / Inference
## Dataset
Download and place Oxford Flower-102 dataset in 
```
"data/flowers-102"
```

## Train Models
```
set -e
mkdir -p ckpts samples data
python train_vae.py
python train_diffuser_stage1_class_emb.py
python train_diffuser_stage2_{style_name}.py
```

## Test image generation with prompt
```
python infer_stage2_{style_name}.py
```

# Example Outcomes 

## Direct Inference by Models trained with Procedural Anchoring
| Felt Style | Lowpoly Style | Oil Painting Style |
|:--:|:--:|:--:|
![Camellia - felt style](https://github.com/kuanyewleong/Procedural-Aesthetics/blob/main/figures/stage2_felt_camellia.png)|![Clematis - lowpoly style](https://github.com/kuanyewleong/Procedural-Aesthetics/blob/main/figures/stage2_lowpoly_clematis.png)|![King Protea - oil painting](https://github.com/kuanyewleong/Procedural-Aesthetics/raw/main/figures/image_oil_painting.jpg)|
|*Camellia*|*Clematis*|*King Protea*|

| Mosaic | Post-impressionism | Poster |
|:--:|:--:|:--:|
|![Buttercup - mosaic](https://github.com/kuanyewleong/Procedural-Aesthetics/blob/main/figures/stage2_mosaic_buttercup.png)|![Rose - post-impressionism](https://github.com/kuanyewleong/Procedural-Aesthetics/blob/main/figures/stage2_post-imp_rose.png)|![Morning Glory - poster](https://github.com/kuanyewleong/Procedural-Aesthetics/blob/main/figures/stage2_poster_morning_glory.png)|
|*Buttercup*|*Rose*|*Morning Glory*|

# The best part (and the highlight) of this approach is the ensembling creativity - 
## Emerging Styles by Ensembling different Models (weighted soft blend per block-wise mixing)