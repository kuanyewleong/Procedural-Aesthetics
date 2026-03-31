import os
import itertools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models_ae import KL_VAE
from text_cond_t5_dual import T5DualTextCond
from diffusion_sd15 import CosineScheduler, q_sample
from diffusers import UNet2DConditionModel

from data_helpers.data_flowers_oil import Flowers102Oil
from auto_captions.auto_captions_oil import make_oil_caption
from procedural_modules.kuwahara_modules import AnisoKuwaharaOilPipeline


class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k in self.shadow:
            self.shadow[k].mul_(self.decay).add_(msd[k], alpha=1 - self.decay)


class _TokenGenBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.to_ss1 = nn.Linear(d_model, 2 * d_model)
        self.to_ss2 = nn.Linear(d_model, 2 * d_model)

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
        p_uncond: float = 0.1,
        dropout: float = 0.0,
    ):
        super().__init__()
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
        tok = self.out_proj(self.norm_out(tok))

        if self.training and self.p_uncond > 0:
            drop = (torch.rand(B, device=tok.device) < self.p_uncond).view(B, 1, 1)
            tok = torch.where(drop, torch.zeros_like(tok), tok)
        return tok


class ClassEmbedder(nn.Module):
    def __init__(self, num_classes: int, d_ctx: int, p_drop: float = 0.1):
        super().__init__()
        self.emb = nn.Embedding(num_classes, d_ctx)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.p_drop = p_drop

    def forward(self, ctx_vec, labels):
        if ctx_vec.dim() == 3:
            cls = self.emb(labels)[:, None, :]
        else:
            cls = self.emb(labels)
        if self.training and self.p_drop > 0:
            drop = (torch.rand(labels.shape[0], device=labels.device) < self.p_drop)
            cls = torch.where(drop.view(-1, 1) if cls.dim() == 2 else drop.view(-1, 1, 1),
                              torch.zeros_like(cls), cls)
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

# ---------- procedural modules ----------
_oil = AnisoKuwaharaOilPipeline(
    radius=10,         # stroke length (8–14)
    step=5,            # samples along stroke (4–7)
    sigma_color=0.8,   # preblur inside filter (0.5–1.2)
    coherence_gain=2.0,# stroke alignment strength (1.0–3.0)
    downsample=2,      # speed: 2 is good for 512
    contrast=0.25,
    edge_strength=0.10,
)

@torch.no_grad()
def stylize_fn(img01):
    return _oil(img01)


def train_stage2_poster(
    stage1_ckpt="ckpts/unet_stage1_clsemb_sd15.pt",
    vae_ckpt="ckpts/vae_512_lpips.pt",
    out="ckpts/unet_stage2_oil.pt",
    image_size=512,
    bs=16,
    lr=5e-5,          # set smaller for finetune
    wd=1e-2,
    timesteps=1000,
    steps_target=80000,
    log_every=200,
    save_every=5000,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    # ---- dataset ----
    ds = Flowers102Oil(split="train", image_size=image_size, stylize_fn=stylize_fn)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    it = itertools.cycle(dl)

    # ---- load VAE (frozen) ----
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

    # ---- text ----
    text = T5DualTextCond(device=device)

    # ---- sched ----
    sched = CosineScheduler(timesteps=timesteps).to(device)

    # ---- load Stage-1 UNet + adapters ----
    latent_size = image_size // 8
    unet = build_small_sd_unet(sample_size=latent_size).to(device)
    unet.enable_gradient_checkpointing()

    ctx_adapter = VecToTokensTransformer(
        d_in=text.d_ctx, n_tokens=77, d_model=512, d_out=768, n_layers=3, n_heads=8, p_uncond=0.1
    ).to(device)

    class_embed = ClassEmbedder(num_classes=102, d_ctx=text.d_ctx, p_drop=0.1).to(device)

    ck1 = torch.load(stage1_ckpt, map_location=device)
    # prefer EMA weights for initialization
    unet.load_state_dict(ck1["ema"] if "ema" in ck1 else ck1["unet"], strict=True)
    if "ctx_adapter" in ck1:
        ctx_adapter.load_state_dict(ck1["ctx_adapter"], strict=True)
    if "class_embed" in ck1:
        class_embed.load_state_dict(ck1["class_embed"], strict=True)

    unet.train()
    ctx_adapter.train()
    class_embed.train()

    # ---- optimizer ----
    opt = torch.optim.AdamW(
        list(unet.parameters()) + list(ctx_adapter.parameters()) + list(class_embed.parameters()),
        lr=lr,
        weight_decay=wd
    )

    ema = EMA(unet, decay=0.9999)

    pbar = tqdm(total=steps_target, desc="Stage-2 poster fine-tune (v-pred)")
    global_step = 0

    while global_step < steps_target:
        b = next(it)
        imgs01 = b["image"].to(device)
        labels = b["label"].to(device, dtype=torch.long)
        names = b["name"]  # list[str]

        # VAE encode deterministic mu
        with torch.no_grad():
            _, mu, _ = vae.encode(imgs01 * 2 - 1, sample=False)
            z0 = mu

        # captions: canonical species + oil painting keywords
        caps = [make_oil_caption(names[i]) for i in range(len(names))]

        ctx = text.encode_dual(caps, None, w_nat=1.0, w_tech=0.0)
        ctx = ctx.to(device)
        ctx = class_embed(ctx, labels)
        ctx_sd = ctx_adapter(ctx)

        t = torch.randint(0, timesteps, (z0.size(0),), device=device, dtype=torch.long)
        noise = torch.randn_like(z0)
        zt = q_sample(z0, t, noise, sched.alpha_bar)

        abar = sched.alpha_bar[t].view(-1, 1, 1, 1)

        # Min-SNR weights
        snr = abar / (1 - abar + 1e-8)
        gamma = 5.0
        w = torch.minimum(snr, torch.tensor(gamma, device=device)) / (snr + 1e-8)

        # v target
        v = abar.sqrt() * noise - (1 - abar).sqrt() * z0

        pred_v = unet(zt, t, encoder_hidden_states=ctx_sd).sample
        loss = (w * (pred_v - v).pow(2)).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(unet.parameters()) + list(ctx_adapter.parameters()) + list(class_embed.parameters()),
            1.0
        )
        opt.step()
        ema.update(unet)

        global_step += 1
        pbar.update(1)

        if global_step % log_every == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        if global_step % save_every == 0:
            # Build step-numbered output path so we don't overwrite
            out_dir = os.path.dirname(out) or "."
            base = os.path.basename(out)

            # Split "name.ext" safely (works even if name has multiple dots)
            stem, ext = os.path.splitext(base)
            step_out = os.path.join(out_dir, f"{stem}_step{global_step}{ext}")

            ckpt = {
                "unet": unet.state_dict(),
                "ema": ema.shadow,
                "ctx_adapter": ctx_adapter.state_dict(),
                "class_embed": class_embed.state_dict(),
                "latent_size": latent_size,
                "global_step": global_step,   # optional but useful metadata
            }

            torch.save(ckpt, step_out)
            pbar.write(f"Saved {step_out}")

    pbar.close()


if __name__ == "__main__":
    train_stage2_poster(
        out="ckpts/unet_stage2_oil.pt",
    )
