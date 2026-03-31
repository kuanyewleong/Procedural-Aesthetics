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

from data_helpers.data_flowers_watercolor import Flowers102Watercolor
from auto_captions.auto_captions_watercolor import make_watercolor_caption


class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k in self.shadow:
            self.shadow[k].mul_(self.decay).add_(msd[k], alpha=1 - self.decay)


# --- match stage-1 adapter/class modules ---
from train_diffuser_stage1_class_emb import VecToTokensTransformer, ClassEmbedder


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


def train_stage2_watercolor(
    stage1_ckpt="ckpts/unet_stage1_clsemb_sd15.pt",
    vae_ckpt="ckpts/vae_512_lpips.pt",
    out="ckpts/unet_stage2_watercolor.pt",
    image_size=512,
    bs=16,
    lr=5e-5,
    wd=1e-2,
    timesteps=1000,
    steps_target=100000,
    log_every=200,
    save_every=1000,
    watercolor_dirname="jpg_watercolor",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    # dataset from preprocessed folder
    ds = Flowers102Watercolor(split="train", image_size=image_size, watercolor_dirname=watercolor_dirname)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    it = itertools.cycle(dl)

    # load frozen VAE
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

    text = T5DualTextCond(device=device)
    sched = CosineScheduler(timesteps=timesteps).to(device)

    # load stage-1 components
    ck1 = torch.load(stage1_ckpt, map_location=device)
    latent_size = ck1.get("latent_size", image_size // 8)

    unet = build_small_sd_unet(sample_size=latent_size).to(device)
    unet.enable_gradient_checkpointing()

    ctx_adapter = VecToTokensTransformer(d_in=text.d_ctx, n_tokens=77, d_model=512, d_out=768, n_layers=3, n_heads=8, p_uncond=0.1).to(device)
    class_embed = ClassEmbedder(num_classes=102, d_ctx=text.d_ctx, p_drop=0.1).to(device)

    unet.load_state_dict(ck1["ema"] if "ema" in ck1 else ck1["unet"], strict=True)
    ctx_adapter.load_state_dict(ck1["ctx_adapter"], strict=True)
    if "class_embed" in ck1:
        class_embed.load_state_dict(ck1["class_embed"], strict=True)

    unet.train(); ctx_adapter.train(); class_embed.train()

    opt = torch.optim.AdamW(
        list(unet.parameters()) + list(ctx_adapter.parameters()) + list(class_embed.parameters()),
        lr=lr, weight_decay=wd
    )
    ema = EMA(unet, decay=0.9999)

    pbar = tqdm(total=steps_target, desc="Stage-2 watercolor fine-tune (v-pred)")
    global_step = 0

    while global_step < steps_target:
        b = next(it)
        imgs01 = b["image"].to(device)
        labels = b["label"].to(device, dtype=torch.long)
        names = b["name"]

        with torch.no_grad():
            _, mu, _ = vae.encode(imgs01 * 2 - 1, sample=False)
            z0 = mu

        caps = [make_watercolor_caption(names[i]) for i in range(len(names))]
        ctx = text.encode_dual(caps, None, w_nat=1.0, w_tech=0.0).to(device)
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
    train_stage2_watercolor(
        stage1_ckpt="ckpts/unet_stage1_clsemb_sd15.pt",
        vae_ckpt="ckpts/vae_512_lpips.pt",
        out="ckpts/unet_stage2_watercolor.pt",
        watercolor_dirname="jpg_watercolor",
    )
