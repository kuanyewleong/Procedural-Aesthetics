import os
import re
import torch
import torch.nn as nn
from torchvision.utils import save_image

from models_ae import KL_VAE
from text_cond_t5_dual import T5DualTextCond
from diffusion_sd15 import CosineScheduler, p_sample_loop
from diffusers import UNet2DConditionModel
from data_helpers.data_flowers import Flowers102Clean


# ---------- helper: prompt -> label id ----------
def _norm(s: str) -> str:
    s = s.lower().replace("_", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def build_name_to_label(names):
    return {_norm(n): i for i, n in enumerate(names)}

def prompt_to_label_id(prompt: str, name_to_label: dict):
    p = _norm(prompt)
    best = None
    best_len = 0
    for name, lid in name_to_label.items():
        if name and name in p and len(name) > best_len:
            best = lid
            best_len = len(name)
    return best

def canonicalize_prompt(prompt: str, species_name: str, append_original: bool = True) -> str:
    """
    Convert arbitrary prompt into the canonical training format.
    """
    base = f"flower species: {species_name}. a photo of a flower."
    if append_original:
        # keep user's extra intent while still anchoring species prefix
        return base + " " + prompt.strip()
    return base


# ---------- modules (must match training) ----------
class VecToTokensTransformer(nn.Module):
    def __init__(
        self,
        d_in: int,
        n_tokens: int = 77,
        d_model: int = 512,
        d_out: int = 768,
        n_layers: int = 3,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        p_uncond: float = 0.0,   # <- inference: 0.0
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_tokens = n_tokens
        self.p_uncond = p_uncond

        self.ctx_proj = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.token_queries = nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)

        self.blocks = nn.ModuleList([
            _TokenGenBlock(d_model=d_model, n_heads=n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.norm_out = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_out)

    def forward(self, ctx):
        if ctx.dim() == 3:
            ctx = ctx.mean(dim=1)
        B = ctx.shape[0]
        c = self.ctx_proj(ctx)

        tok = self.token_queries[None, :, :].expand(B, -1, -1) + self.pos_emb[None, :, :]
        for blk in self.blocks:
            tok = blk(tok, c)

        tok = self.norm_out(tok)
        tok = self.out_proj(tok)  # [B,77,768]
        return tok


class _TokenGenBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model))

        self.to_ss1 = nn.Linear(d_model, 2 * d_model)
        self.to_ss2 = nn.Linear(d_model, 2 * d_model)

    def forward(self, x, ctx_vec):
        h = self.norm1(x)
        scale, shift = self.to_ss1(ctx_vec).unsqueeze(1).chunk(2, dim=-1)
        h = h * (1 + scale) + shift
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out

        h = self.norm2(x)
        scale, shift = self.to_ss2(ctx_vec).unsqueeze(1).chunk(2, dim=-1)
        h = h * (1 + scale) + shift
        x = x + self.mlp(h)
        return x


class ClassEmbedder(nn.Module):
    def __init__(self, num_classes: int, d_ctx: int):
        super().__init__()
        self.emb = nn.Embedding(num_classes, d_ctx)

    def forward(self, ctx_vec: torch.Tensor, labels: torch.Tensor):
        # ctx_vec: [B,D] or [B,1,D] or [B,S,D]
        if ctx_vec.dim() == 3:
            cls = self.emb(labels)[:, None, :]
        else:
            cls = self.emb(labels)
        return ctx_vec + cls


def build_sd15_unet(sample_size: int):
    return UNet2DConditionModel(
        sample_size=sample_size,
        in_channels=4,
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(192, 384, 768, 768),
        down_block_types=("CrossAttnDownBlock2D","CrossAttnDownBlock2D","CrossAttnDownBlock2D","DownBlock2D"),
        up_block_types=("UpBlock2D","CrossAttnUpBlock2D","CrossAttnUpBlock2D","CrossAttnUpBlock2D"),
        cross_attention_dim=768,
        attention_head_dim=8,
    )


@torch.no_grad()
def main(
    prompt_nat="a clematis in the wild",
    vae_ckpt="ckpts/vae_512_lpips.pt",
    sd15_unet_ckpt="ckpts/unet_stage1_clsemb_sd15.pt",
    out_png="samples/sample_sd15.png",
    image_size=512,
    timesteps=1000,
    guidance_scale=2.5,
    sample_steps=250,
    dyn_thresh=False,   # start off; enable only if needed
    dyn_p=0.999,
    canonicalize_prompt_on=True,
    canonicalize_append_original=True,
):
    os.makedirs("samples", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- VAE ----
    ck_vae = torch.load(vae_ckpt, map_location=device)
    if isinstance(ck_vae, dict) and "vae" in ck_vae:
        base = ck_vae.get("base", 96)
        z_ch = ck_vae.get("z_ch", 4)
        vae = KL_VAE(z_ch=z_ch, base=base).to(device).eval()
        vae.load_state_dict(ck_vae.get("ema", ck_vae["vae"]), strict=True)
    else:
        vae = KL_VAE(z_ch=4, base=96).to(device).eval()
        vae.load_state_dict(ck_vae, strict=True)

    for p in vae.parameters():
        p.requires_grad_(False)
    vae.eval()

    # ---- Text ----
    text = T5DualTextCond(device=device)

    # ---- UNet + adapters ----
    latent_size = image_size // 8
    unet = build_sd15_unet(sample_size=latent_size).to(device).eval()
    ctx_adapter = VecToTokensTransformer(d_in=text.d_ctx, n_layers=3).to(device).eval()
    class_embed = ClassEmbedder(num_classes=102, d_ctx=text.d_ctx).to(device).eval()

    ck = torch.load(sd15_unet_ckpt, map_location=device)
    unet.load_state_dict(ck["ema"] if "ema" in ck else ck["unet"], strict=True)
    ctx_adapter.load_state_dict(ck["ctx_adapter"], strict=True)
    if "class_embed" in ck:
        class_embed.load_state_dict(ck["class_embed"], strict=True)
    else:
        print("Warning: checkpoint has no class_embed; running prompt-only.")

    sched = CosineScheduler(timesteps=timesteps).to(device)

    # ---- auto map prompt species -> label id ----
    names = Flowers102Clean(split="train", image_size=image_size).names
    name_to_label = build_name_to_label(names)
    label_id = prompt_to_label_id(prompt_nat, name_to_label)

    labels = None
    if label_id is not None:
        species_name = names[label_id]
        print("Auto label:", label_id, species_name)
        labels = torch.tensor([label_id], device=device, dtype=torch.long)

        if canonicalize_prompt_on:
            prompt_nat = canonicalize_prompt(
                prompt_nat,
                species_name=species_name,
                append_original=canonicalize_append_original
            )
            print("Canonicalized prompt:", prompt_nat)
    else:
        print("No species match; prompt-only.")

    # ---- build ctx ----
    ctx_cond_raw = text.encode_dual([prompt_nat], None, w_nat=1.0, w_tech=0.0)
    ctx_uncond_raw = text.encode_dual([""], None, w_nat=1.0, w_tech=0.0)

    if labels is not None and "class_embed" in ck:
        ctx_cond_raw = class_embed(ctx_cond_raw, labels)

    ctx_cond = ctx_adapter(ctx_cond_raw)
    ctx_uncond = ctx_adapter(ctx_uncond_raw)

    # v-pred -> eps wrapper (your p_sample_loop expects eps)
    def model_eps_from_v(x, tt, ctx_tokens):
        v = unet(x, tt, encoder_hidden_states=ctx_tokens).sample
        abar = sched.alpha_bar[tt].view(-1, 1, 1, 1)
        eps = abar.sqrt() * v + (1 - abar).sqrt() * x
        return eps

    z = p_sample_loop(
        model=model_eps_from_v,
        sched=sched,
        shape=(1, 4, latent_size, latent_size),
        ctx_cond=ctx_cond,
        ctx_uncond=ctx_uncond,
        device=device,
        guidance_scale=guidance_scale,
        sample_steps=sample_steps,
        use_ddim=True,
        eta=0.0,
        dyn_thresh=dyn_thresh,
        dyn_p=dyn_p,
    )

    x = vae.decode(z).clamp(-1, 1)
    save_image((x + 1) / 2, out_png)
    print("Saved:", out_png)


if __name__ == "__main__":
    main(
        prompt_nat="a photo of hibiscus in the wild",
        out_png="samples/sample_sd15_clsemb.png",
        guidance_scale=2.5,
        sample_steps=300,
        dyn_thresh=True,
        dyn_p=0.999,
        canonicalize_prompt_on=True,
        canonicalize_append_original=True
        )