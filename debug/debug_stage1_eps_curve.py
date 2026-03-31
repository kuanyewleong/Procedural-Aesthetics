import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_helpers.data_flowers import Flowers102Clean
from models_ae import KL_VAE
from models_unet import UNetCond
from text_cond_t5_dual import T5DualTextCond
from diffusion import CosineScheduler, q_sample

@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vae = KL_VAE(z_ch=4).to(device).eval()
    vae.load_state_dict(torch.load("ckpts/vae.pt", map_location=device))

    text = T5DualTextCond(device=device)
    unet = UNetCond(z_ch=4, base=128, d_ctx=text.d_ctx).to(device).eval()
    unet.load_state_dict(torch.load("ckpts/unet_stage1_t5.pt", map_location=device))

    sched = CosineScheduler(timesteps=1000).to(device)

    dl = DataLoader(Flowers102Clean(split="val", image_size=256), batch_size=16, shuffle=True, num_workers=0)
    b = next(iter(dl))
    x01 = b["image"].to(device)
    caps = [b["text"][i] for i in range(len(x01))]
    ctx = text.encode_dual(caps, None, w_nat=1.0, w_tech=0.0)

    z0, _, _, _ = vae.enc(x01*2-1)
    noise = torch.randn_like(z0)

    # probe timesteps
    ts = [0, 10, 50, 100, 250, 500, 750, 900, 950, 990, 999]
    for tval in ts:
        t = torch.full((z0.size(0),), tval, device=device, dtype=torch.long)
        zt = q_sample(z0, t, noise, sched.alpha_bar)
        pred = unet(zt, t, ctx)
        mse = F.mse_loss(pred, noise).item()
        print(f"t={tval:4d}  mse={mse:.4f}  pred_mean={pred.mean().item():+.3f}  pred_std={pred.std(unbiased=False).item():.3f}")

if __name__ == "__main__":
    main()
