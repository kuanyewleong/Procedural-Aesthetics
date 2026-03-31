import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import itertools

from data_helpers.data_flowers import Flowers102Clean
from auto_captions.auto_captions_class_emb import make_natural_caption
from models_ae import KL_VAE
from text_cond_t5_dual import T5DualTextCond
from diffusion_sd15 import CosineScheduler, q_sample

# from models_diffuser import build_sd15_unet
from diffusers import UNet2DConditionModel


class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k in self.shadow:
            self.shadow[k].mul_(self.decay).add_(msd[k], alpha=1 - self.decay)

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)

class VecToTokensTransformer(nn.Module):
    """
    Single conditioning vector -> 77 token embeddings for SD1.5 cross-attn.
    - learned token queries (77 x d_model)
    - learned positional embeddings (77 x d_model)
    - FiLM-style conditioning from ctx vector into each block
    - 2-4 Transformer blocks
    - output projection to 768
    - CFG dropout: optionally zero tokens during training
    """
    def __init__(
        self,
        d_in: int,            # your ctx dim (text.d_ctx)
        n_tokens: int = 77,
        d_model: int = 512,   # internal transformer width
        d_out: int = 768,     # SD cross_attention_dim
        n_layers: int = 3,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        p_uncond: float = 0.1,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0

        self.n_tokens = n_tokens
        self.d_out = d_out
        self.p_uncond = p_uncond

        # ctx -> d_model
        self.ctx_proj = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # learned token queries + positional embeddings
        self.token_queries = nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)

        self.blocks = nn.ModuleList([
            _TokenGenBlock(
                d_model=d_model,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.norm_out = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_out)

    def forward(self, ctx):
        """
        ctx: [B,D] or [B,1,D] or [B,S,D]
        returns: [B,77,768]
        """
        if ctx.dim() == 3:
            ctx = ctx.mean(dim=1)  # simple pooling
        B = ctx.shape[0]

        c = self.ctx_proj(ctx)  # [B, d_model]

        # start tokens = learned queries + positional embeddings
        tok = self.token_queries[None, :, :].expand(B, -1, -1) + self.pos_emb[None, :, :]

        # transformer blocks with FiLM conditioning from c
        for blk in self.blocks:
            tok = blk(tok, c)

        tok = self.norm_out(tok)
        tok = self.out_proj(tok)  # [B,77,768]

        # CFG dropout: replace with zeros sometimes
        if self.training and self.p_uncond > 0:
            drop = (torch.rand(B, device=tok.device) < self.p_uncond).view(B, 1, 1)
            tok = torch.where(drop, torch.zeros_like(tok), tok)

        return tok


class _TokenGenBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

        # FiLM from ctx vector: scale+shift for attention and mlp
        self.to_ss1 = nn.Linear(d_model, 2 * d_model)
        self.to_ss2 = nn.Linear(d_model, 2 * d_model)

    def forward(self, x, ctx_vec):
        """
        x: [B,N,d_model]
        ctx_vec: [B,d_model]
        """
        # --- attn ---
        h = self.norm1(x)
        ss = self.to_ss1(ctx_vec).unsqueeze(1)  # [B,1,2d]
        scale, shift = ss.chunk(2, dim=-1)
        h = h * (1 + scale) + shift

        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out

        # --- mlp ---
        h = self.norm2(x)
        ss = self.to_ss2(ctx_vec).unsqueeze(1)
        scale, shift = ss.chunk(2, dim=-1)
        h = h * (1 + scale) + shift

        x = x + self.mlp(h)
        return x

class ClassEmbedder(nn.Module):
    """
    Learns a class embedding and adds it to a text conditioning vector.
    - num_classes: 102 for Flowers-102
    - d_ctx: must match your text.encode_dual output dim (text.d_ctx)
    - p_drop: probability to drop class conditioning during training (for CFG / robustness)
    """
    def __init__(self, num_classes: int, d_ctx: int, p_drop: float = 0.1):
        super().__init__()
        self.emb = nn.Embedding(num_classes, d_ctx)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.p_drop = p_drop

    def forward(self, ctx_vec: torch.Tensor, labels: torch.Tensor):
        """
        ctx_vec: [B, D] or [B,1,D] or [B,S,D]
        labels:  [B] long, values 0..num_classes-1
        returns ctx_vec with class info added (same shape as input ctx_vec)
        """
        if ctx_vec.dim() == 3:
            # broadcast add to all tokens/segments
            cls = self.emb(labels)[:, None, :]  # [B,1,D]
        else:
            cls = self.emb(labels)              # [B,D]

        if self.training and self.p_drop > 0:
            drop = (torch.rand(labels.shape[0], device=labels.device) < self.p_drop)
            if ctx_vec.dim() == 3:
                drop = drop.view(-1, 1, 1)
            else:
                drop = drop.view(-1, 1)
            cls = torch.where(drop, torch.zeros_like(cls), cls)

        return ctx_vec + cls


def train_stage1_t5_sd15_unet(
    vae_ckpt="ckpts/vae_512_lpips.pt",
    out="ckpts/unet_stage1_t5_sd15.pt",
    image_size=512,
    bs=16,
    lr=1e-4,
    wd=1e-2,
    timesteps=1000,
    ):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = Flowers102Clean(split="train", image_size=image_size)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    it = itertools.cycle(dl)

    # -------- load VAE safely --------
    ck_vae = torch.load(vae_ckpt, map_location=device)
    if isinstance(ck_vae, dict) and "vae" in ck_vae:
        base = ck_vae.get("base", 96)
        z_ch = ck_vae.get("z_ch", 4)
        vae = KL_VAE(z_ch=z_ch, base=base).to(device).eval()
        vae.load_state_dict(ck_vae.get("ema", ck_vae["vae"]), strict=True)
    else:
        # fallback: raw state_dict
        vae = KL_VAE(z_ch=4, base=96).to(device).eval()
        vae.load_state_dict(ck_vae, strict=True)

    for p in vae.parameters():
        p.requires_grad_(False)
    vae.eval()
    # --------------------------------

    text = T5DualTextCond(device=device)

    sched = CosineScheduler(timesteps=timesteps).to(device)

    latent_size = image_size // 8  # 64 for 512 with /8 VAE

    unet = UNet2DConditionModel(
        sample_size=latent_size,  # <- use latent_size variable
        in_channels=4,
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(192, 384, 768, 768),
        down_block_types=("CrossAttnDownBlock2D","CrossAttnDownBlock2D","CrossAttnDownBlock2D","DownBlock2D"),
        up_block_types=("UpBlock2D","CrossAttnUpBlock2D","CrossAttnUpBlock2D","CrossAttnUpBlock2D"),
        cross_attention_dim=768,
        attention_head_dim=8,
    ).to(device)
    unet.enable_gradient_checkpointing()

    ctx_adapter = VecToTokensTransformer(
        d_in=text.d_ctx,
        n_tokens=77,
        d_model=512,
        d_out=768,
        n_layers=3,
        n_heads=8,
        p_uncond=0.1,
    ).to(device)

    class_embed = ClassEmbedder(num_classes=102, d_ctx=text.d_ctx, p_drop=0.1).to(device)

    # make sure train mode
    unet.train()
    ctx_adapter.train()
    class_embed.train()

    opt = torch.optim.AdamW(
        list(unet.parameters()) + list(ctx_adapter.parameters()) + list(class_embed.parameters()),
        lr=lr,
        weight_decay=wd
    )

    steps_target = 160000
    log_every = 500
    save_every = 500

    ema = EMA(unet, decay=0.9999)

    pbar = tqdm(total=steps_target, desc="Stage-1 SD1.5-UNet (v-pred) training")
    global_step = 0

    while global_step < steps_target:
        b = next(it)
        imgs01 = b["image"].to(device)
        labels = b["label"].to(device, dtype=torch.long)

        with torch.no_grad():
            _, mu, _ = vae.encode(imgs01 * 2 - 1, sample=False)
            z0 = mu

        caps = [
            make_natural_caption(
                b["image"][i],
                flower_name=b["name"][i],
                watercolor=False,
                include_details=False
            )
            for i in range(len(imgs01))
        ]

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

        # IMPORTANT: include class_embed params too
        torch.nn.utils.clip_grad_norm_(
            list(unet.parameters()) + list(ctx_adapter.parameters()) + list(class_embed.parameters()),
            1.0
        )

        opt.step()
        ema.update(unet)

        global_step += 1
        pbar.update(1)

        if global_step % log_every == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}", latent=f"{latent_size}x{latent_size}")

        if global_step % save_every == 0:
            torch.save(
                {
                    "unet": unet.state_dict(),
                    "ema": ema.shadow,
                    "ctx_adapter": ctx_adapter.state_dict(),
                    "class_embed": class_embed.state_dict(),
                    "latent_size": latent_size,
                },
                out
            )
            pbar.write(f"Saved {out}")

    pbar.close()


if __name__ == "__main__":
    train_stage1_t5_sd15_unet(
        vae_ckpt="ckpts/vae_512_lpips.pt",
        out="ckpts/unet_stage1_clsemb_sd15.pt"
    )