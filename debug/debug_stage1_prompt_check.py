# quick_stage1_prompt_check.py
import os
import torch
from torchvision.utils import save_image

from models_ae import KL_VAE
from models_unet import UNetCond
from text_cond_t5_dual import T5DualTextCond
from diffusion import CosineScheduler, p_sample_loop

@torch.no_grad()
def main(
    prompt_nat="a photo of a lily",    
    prompt_tech=None,
    w_nat=0.7, w_tech=0.3,
    vae_ckpt="ckpts/vae.pt",
    unet_stage1_ckpt="ckpts/unet_stage1_t5.pt",
    out_png="samples/stage1_prompt_check.png",
    image_size=256,
    timesteps=200,
):
    os.makedirs("samples", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vae = KL_VAE(z_ch=4).to(device).eval()
    vae.load_state_dict(torch.load(vae_ckpt, map_location=device))

    text = T5DualTextCond(device=device)

    unet = UNetCond(z_ch=4, base=192, d_ctx=text.d_ctx).to(device).eval()
    # unet.load_state_dict(torch.load(unet_stage1_ckpt, map_location=device))
    ck = torch.load(unet_stage1_ckpt, map_location=device)
    unet.load_state_dict(ck["ema"] if "ema" in ck else ck)

    sched = CosineScheduler(timesteps=timesteps).to(device)

    w_nat=1.0; w_tech=0.0    
    prompt_tech=None
    ctx = text.encode_dual([prompt_nat], [prompt_tech] if prompt_tech else None, w_nat=w_nat, w_tech=w_tech)
    ctx_cond = text.encode_dual([prompt_nat], [prompt_tech] if prompt_tech else None, w_nat=w_nat, w_tech=w_tech)
    ctx_uncond = text.encode_dual([""], None, w_nat=1.0, w_tech=0.0)  # unconditional

    guidance_scale=0.0

    def model(x, tt, ctx_):
        return unet(x, tt, ctx_)

    zshape = (1, 4, image_size//8, image_size//8)
    z = p_sample_loop(
        model=lambda x, tt, ctx: unet(x, tt, ctx),
        sched=sched,
        shape=zshape,
        ctx_cond=ctx_cond,
        ctx_uncond=ctx_uncond,
        device=device,
        guidance_scale=3.0,     # try 2.0–6.0
    )

    # IMPORTANT: don't tanh here; clamp is safer for debugging
    x = vae.dec(z).clamp(-1, 1)
    save_image((x + 1) / 2, out_png)
    print("Saved:", out_png)
    print("z stats:", float(z.mean()), float(z.std(unbiased=False)), float(z.min()), float(z.max()))
    print("x stats:", float(x.mean()), float(x.std(unbiased=False)), float(x.min()), float(x.max()))

if __name__ == "__main__":
    main()
