import torch
from models_ae import KL_VAE
from models_unet import UNetCond
from text_cond_t5_dual import T5DualTextCond
from diffusion import CosineScheduler, p_sample_loop

@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vae = KL_VAE(z_ch=4).to(device).eval()
    vae.load_state_dict(torch.load("ckpts/vae.pt", map_location=device))

    text = T5DualTextCond(device=device)
    unet = UNetCond(z_ch=4, base=128, d_ctx=text.d_ctx).to(device).eval()
    unet.load_state_dict(torch.load("ckpts/unet_stage2_psa_t5.pt", map_location=device))

    sched = CosineScheduler(timesteps=1000).to(device)

    prompt = "watercolor painting of a tulip, soft transparent washes, paper texture"
    ctx = text.encode_dual([prompt], None, w_nat=1.0, w_tech=0.0)

    def model(x, tt, ctx):
        eps = unet(x, tt, ctx)
        return eps

    zshape = (1, 4, 256//8, 256//8)
    z = p_sample_loop(model, sched, zshape, ctx, device)

    print("sampled z stats:",
          "mean", z.mean().item(),
          "std", z.std(unbiased=False).item(),
          "min", z.min().item(),
          "max", z.max().item())

    x = vae.dec(z).clamp(-1, 1)
    print("decoded x stats:",
          "mean", x.mean().item(),
          "std", x.std(unbiased=False).item(),
          "min", x.min().item(),
          "max", x.max().item())

if __name__ == "__main__":
    main()
