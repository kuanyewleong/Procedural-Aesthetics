import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from models_ae import KL_VAE
from text_cond_t5_dual import T5DualTextCond
from diffusion_sd15 import CosineScheduler, p_sample_loop
from diffusers import UNet2DConditionModel


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


def build_sd15_unet(sample_size: int):
    return UNet2DConditionModel(
        sample_size=64,
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
    prompt_nat="morning glory",
    prompt_tech=None,
    w_nat=1.0, w_tech=0.0,
    vae_ckpt="ckpts/vae_512_lpips.pt",      # dict with keys: vae, ema, base, z_ch (ideally)
    sd15_unet_ckpt="ckpts/unet_stage1_t5_sd15.pt",
    out_png="samples/sample_sd15.png",
    image_size=512,
    timesteps=1000,
    guidance_scale=3.0,
    sample_steps=200,
    use_ddim=True,
    eta=0.0,
    dyn_thresh=True,
    dyn_p=0.999,
):
    os.makedirs("samples", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------- VAE load (base=96) --------
    ck_vae = torch.load(vae_ckpt, map_location=device)
    if isinstance(ck_vae, dict) and "vae" in ck_vae:
        base = ck_vae.get("base", 96)
        z_ch = ck_vae.get("z_ch", 4)
        vae = KL_VAE(z_ch=z_ch, base=base).to(device).eval()
        vae.load_state_dict(ck_vae.get("ema", ck_vae["vae"]), strict=True)
    else:
        # fallback: raw state_dict (older)
        vae = KL_VAE(z_ch=4, base=96).to(device).eval()
        vae.load_state_dict(ck_vae, strict=True)

    for p in vae.parameters():
        p.requires_grad_(False)
    vae.eval()

    # -------- Text encoder --------
    text = T5DualTextCond(device=device)

    # -------- UNet + ctx_adapter --------
    latent_size = image_size // 8  # 64 for 512 with /8 VAE
    unet = build_sd15_unet(sample_size=latent_size).to(device).eval()
    ctx_adapter = VecToTokensTransformer(
                                            d_in=text.d_ctx,
                                            n_tokens=77,
                                            d_model=512,
                                            d_out=768,
                                            n_layers=3,     # try 2 or 3 first
                                            n_heads=8,
                                            p_uncond=0.1,
                                        ).to(device).eval()

    ck = torch.load(sd15_unet_ckpt, map_location=device)
    if isinstance(ck, dict) and "unet" in ck:
        unet.load_state_dict(ck["ema"] if "ema" in ck else ck["unet"], strict=True)
        if "ctx_adapter" in ck:
            ctx_adapter.load_state_dict(ck["ctx_adapter"], strict=True)
        if "latent_size" in ck:
            print("ckpt latent_size:", ck["latent_size"], "current:", latent_size)
    else:
        unet.load_state_dict(ck, strict=True)

    sched = CosineScheduler(timesteps=timesteps).to(device)

    # -------- Build contexts --------
    ctx_cond_raw = text.encode_dual(
        [prompt_nat],
        [prompt_tech] if prompt_tech else None,
        w_nat=w_nat, w_tech=w_tech
    )
    ctx_uncond_raw = text.encode_dual([""], None, w_nat=1.0, w_tech=0.0)

    ctx_cond = ctx_adapter(ctx_cond_raw)
    ctx_uncond = ctx_adapter(ctx_uncond_raw)

    def model_eps_from_v(x, tt, ctx_tokens):
        v = unet(x, tt, encoder_hidden_states=ctx_tokens).sample  # v-pred
        abar = sched.alpha_bar[tt].view(-1,1,1,1)
        eps = abar.sqrt() * v + (1 - abar).sqrt() * x
        return eps

    # def model(x, tt, ctx_tokens):
    #     return unet(x, tt, encoder_hidden_states=ctx_tokens).sample

    # -------- Sample latents --------
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

    # -------- Decode --------
    x = vae.decode(z).clamp(-1, 1)
    save_image((x + 1) / 2, out_png)
    
    print("z stats:", float(z.mean()), float(z.std(unbiased=False)), float(z.min()), float(z.max()))
    print("x stats:", float(x.mean()), float(x.std(unbiased=False)), float(x.min()), float(x.max()))
    print("Saved:", out_png)
    

if __name__ == "__main__":
    main(
        prompt_nat="a photograph of yellow iris",
        prompt_tech=None,
        w_nat=1.0, w_tech=0.0,
        vae_ckpt="ckpts/vae_512_lpips.pt",
        sd15_unet_ckpt="ckpts/unet_stage1_clsemb_sd15.pt",
        out_png="samples/stage1_check_sd15.png",
        image_size=512,
        timesteps=1000,
        guidance_scale=2.0,
        sample_steps=250,
        use_ddim=True,
        eta=0.0, 
        dyn_thresh=True,
        dyn_p=0.999,
        )
