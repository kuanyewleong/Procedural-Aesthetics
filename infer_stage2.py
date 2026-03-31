import os
import re
import torch
from torchvision.utils import save_image

from models_ae import KL_VAE
from text_cond_t5_dual import T5DualTextCond
from diffusers import UNet2DConditionModel
from diffusion_sd15 import CosineScheduler, p_sample_loop

from data_helpers.data_flowers import Flowers102Clean
# from procedural_modules.kuwahara_modules import AnisoKuwaharaOilPipeline


# -------------------------
# Conditioning modules (must match Stage-2 training)
# -------------------------

class _TokenGenBlock(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.attn = torch.nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = torch.nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_model, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, d_model),
        )
        self.to_ss1 = torch.nn.Linear(d_model, 2 * d_model)
        self.to_ss2 = torch.nn.Linear(d_model, 2 * d_model)

    def forward(self, x, ctx_vec):
        h = self.norm1(x)
        ss = self.to_ss1(ctx_vec).unsqueeze(1)
        scale, shift = ss.chunk(2, dim=-1)
        h = h * (1 + scale) + shift
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out

        h = self.norm2(x)
        ss = self.to_ss2(ctx_vec).unsqueeze(1)
        scale, shift = ss.chunk(2, dim=-1)
        h = h * (1 + scale) + shift
        x = x + self.mlp(h)
        return x


class VecToTokensTransformer(torch.nn.Module):
    def __init__(
        self,
        d_in: int,
        n_tokens: int = 77,
        d_model: int = 512,
        d_out: int = 768,
        n_layers: int = 3,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        p_uncond: float = 0.0,   # inference: no dropout
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.p_uncond = p_uncond

        self.ctx_proj = torch.nn.Sequential(
            torch.nn.Linear(d_in, d_model),
            torch.nn.SiLU(),
            torch.nn.Linear(d_model, d_model),
        )
        self.token_queries = torch.nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)
        self.pos_emb = torch.nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)
        self.blocks = torch.nn.ModuleList([
            _TokenGenBlock(d_model=d_model, n_heads=n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm_out = torch.nn.LayerNorm(d_model)
        self.out_proj = torch.nn.Linear(d_model, d_out)

    def forward(self, ctx):
        if ctx.dim() == 3:
            ctx = ctx.mean(dim=1)
        B = ctx.shape[0]
        c = self.ctx_proj(ctx)
        tok = self.token_queries[None, :, :].expand(B, -1, -1) + self.pos_emb[None, :, :]
        for blk in self.blocks:
            tok = blk(tok, c)
        tok = self.out_proj(self.norm_out(tok))
        return tok


class ClassEmbedder(torch.nn.Module):
    def __init__(self, num_classes: int, d_ctx: int):
        super().__init__()
        self.emb = torch.nn.Embedding(num_classes, d_ctx)

    def forward(self, ctx_vec, labels):
        if ctx_vec.dim() == 3:
            cls = self.emb(labels)[:, None, :]
        else:
            cls = self.emb(labels)
        return ctx_vec + cls


def build_small_sd_unet(sample_size: int):
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


# -------------------------
# Prompt -> label auto-match (optional)
# -------------------------

def build_name_to_label(names):
    def norm(s):
        s = s.lower().replace("_", " ").strip()
        s = re.sub(r"\s+", " ", s)
        return s
    return {norm(n): i for i, n in enumerate(names)}

def prompt_to_label_id(prompt: str, name_to_label: dict):
    p = prompt.lower().replace("_", " ")
    p = re.sub(r"\s+", " ", p)
    best, best_len = None, 0
    for name, lid in name_to_label.items():
        if name and name in p and len(name) > best_len:
            best, best_len = lid, len(name)
    return best


# -------------------------
# Inference
# -------------------------

@torch.no_grad()
def main(
    prompt="a morning glory in the wild, poster painting style",
    ckpt_stage2="ckpts/unet_stage2_poster.pt",
    vae_ckpt="ckpts/vae_512_lpips.pt",
    out_png="samples/stage2_poster_sample.png",
    image_size=512,
    timesteps=1000,
    sample_steps=250,
    guidance_scale=2.5,
    use_ddim=True,
    eta=0.0,
    dyn_thresh=False,   # recommended OFF with stable sampler; turn on if needed
    dyn_p=0.999,
    seed=0,
    auto_label=True,
):
    os.makedirs("samples", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    # ---- load VAE ----
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

    # ---- load Stage-2 checkpoint ----
    ck = torch.load(ckpt_stage2, map_location=device)
    latent_size = int(ck.get("latent_size", image_size // 8))

    unet = build_small_sd_unet(sample_size=latent_size).to(device).eval()
    unet.load_state_dict(ck["ema"] if "ema" in ck else ck["unet"], strict=True)

    text = T5DualTextCond(device=device)

    ctx_adapter = VecToTokensTransformer(
        d_in=text.d_ctx, n_tokens=77, d_model=512, d_out=768, n_layers=3, n_heads=8, p_uncond=0.0
    ).to(device).eval()
    ctx_adapter.load_state_dict(ck["ctx_adapter"], strict=True)

    class_embed = ClassEmbedder(num_classes=102, d_ctx=text.d_ctx).to(device).eval()
    if "class_embed" in ck:
        class_embed.load_state_dict(ck["class_embed"], strict=True)

    # ---- auto label from prompt (optional) ----
    label_id = None
    if auto_label:
        ds_tmp = Flowers102Clean(split="train", image_size=image_size)
        name_to_label = build_name_to_label(ds_tmp.names)
        label_id = prompt_to_label_id(prompt, name_to_label)
        if label_id is not None:
            print(f"Auto species label: {label_id} ({ds_tmp.names[label_id]})")
        else:
            print("Auto species label: None (prompt-only)")

    # ---- build conditional/unconditional contexts ----
    ctx_cond_raw = text.encode_dual([prompt], None, w_nat=1.0, w_tech=0.0).to(device)
    if label_id is not None:
        labels = torch.tensor([label_id], device=device, dtype=torch.long)
        ctx_cond_raw = class_embed(ctx_cond_raw, labels)

    ctx_uncond_raw = text.encode_dual([""], None, w_nat=1.0, w_tech=0.0).to(device)
    # IMPORTANT: do NOT add class_embed to uncond

    ctx_cond = ctx_adapter(ctx_cond_raw)
    ctx_uncond = ctx_adapter(ctx_uncond_raw)

    # ---- v-pred -> eps wrapper for sampler ----
    sched = CosineScheduler(timesteps=timesteps).to(device)

    def model_eps(x, tt, ctx_tokens):
        # UNet predicts v (stage-2 trained with v)
        v = unet(x, tt, encoder_hidden_states=ctx_tokens).sample
        abar = sched.alpha_bar[tt].view(-1, 1, 1, 1)
        # eps = sqrt(abar)*v + sqrt(1-abar)*x_t
        eps = abar.sqrt() * v + (1 - abar).sqrt() * x
        return eps

    # ---- sample ----
    zshape = (1, 4, latent_size, latent_size)
    z = p_sample_loop(
        model=model_eps,
        sched=sched,
        shape=zshape,
        ctx_cond=ctx_cond,
        ctx_uncond=ctx_uncond,
        device=device,
        guidance_scale=guidance_scale,
        sample_steps=sample_steps,
        use_ddim=use_ddim,
        eta=eta,
        dyn_thresh=dyn_thresh,
        dyn_p=dyn_p
    )

    x = vae.decode(z).clamp(-1, 1)
    save_image((x + 1) / 2, out_png)
    print("Saved:", out_png)
    print("z stats:", float(z.mean()), float(z.std(unbiased=False)), float(z.min()), float(z.max()))
    print("x stats:", float(x.mean()), float(x.std(unbiased=False)), float(x.min()), float(x.max()))


if __name__ == "__main__":
    main(
        prompt="a oxeye daisy in the wild, in post impressionism style",
        ckpt_stage2="ckpts/unet_stage2_post-imp.pt",
        vae_ckpt="ckpts/vae_512_lpips.pt",
        out_png="stage2_post-imp_sample.png",
        guidance_scale=2.5,
        sample_steps=250,
        dyn_p=0.999,
        dyn_thresh=False,
    )
