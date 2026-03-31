import torch
from models_ae import KL_VAE

device = "cuda" if torch.cuda.is_available() else "cpu"

vae = KL_VAE(z_ch=4).to(device).eval()
vae.load_state_dict(torch.load("ckpts/vae.pt", map_location=device))

# make a dummy 256x256 image batch in [-1, 1]
imgs = torch.randn(1, 3, 256, 256, device=device).clamp(-1, 1)

with torch.no_grad():
    z0, mu, logv, _ = vae.enc(imgs)

print("imgs.shape:", imgs.shape)  # [B,3,256,256]
print("z0.shape:", z0.shape)      # expected [B,4,32,32] for your VAE

