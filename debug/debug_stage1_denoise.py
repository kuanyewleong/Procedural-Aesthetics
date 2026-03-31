import os, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from data_helpers.data_flowers import Flowers102Clean
from models_ae import KL_VAE
from models_unet import UNetCond
from text_cond_t5_dual import T5DualTextCond
from diffusion import CosineScheduler, q_sample

@torch.no_grad()
def main():
    os.makedirs("samples/debug", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vae = KL_VAE(z_ch=4).to(device).eval()
    vae.load_state_dict(torch.load("ckpts/vae.pt", map_location=device))

    text = T5DualTextCond(device=device)
    unet = UNetCond(z_ch=4, base=128, d_ctx=text.d_ctx).to(device).eval()
    unet.load_state_dict(torch.load("ckpts/unet_stage1_t5.pt", map_location=device))

    sched = CosineScheduler(timesteps=1000).to(device)

    dl = DataLoader(Flowers102Clean(split="val", image_size=256), batch_size=8, shuffle=True, num_workers=0)
    b = next(iter(dl))
    x01 = b["image"].to(device)                     # [0,1]
    captions = [b["text"][i] for i in range(len(x01))]
    ctx = text.encode_dual(captions, None, w_nat=1.0, w_tech=0.0)

    # encode to latent
    z0, _, _, _ = vae.enc(x01*2-1)                  # z0 is what diffusion sees
    noise = torch.randn_like(z0)

    # pick a mid timestep
    t = torch.full((z0.size(0),), 500, device=device, dtype=torch.long)
    # zt = q_sample(z0, t, noise, sched.alphas_cumprod)
    zt = q_sample(z0, t, noise, sched.alpha_bar)

    pred_eps = unet(zt, t, ctx)
    mse = F.mse_loss(pred_eps, noise).item()
    print("Stage1 eps-pred MSE @t=500:", mse)

    # reconstruct x0 from predicted eps
    abar = sched.alpha_bar[t].view(-1,1,1,1)
    z0_hat = (zt - (1-abar).sqrt()*pred_eps) / (abar.sqrt() + 1e-8)

    # decode
    xrec = vae.dec(z0_hat).clamp(-1,1)
    save_image(x01, "samples/debug/input_x01.png", nrow=4)
    save_image((xrec+1)/2, "samples/debug/denoise_x0hat.png", nrow=4)
    print("Saved samples/debug/input_x01.png and denoise_x0hat.png")

if __name__ == "__main__":
    main()
