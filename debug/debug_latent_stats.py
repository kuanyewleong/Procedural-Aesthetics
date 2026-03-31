import torch
from torch.utils.data import DataLoader
from data_helpers.data_flowers import Flowers102Clean
from models_ae import KL_VAE

device = "cuda" if torch.cuda.is_available() else "cpu"
vae = KL_VAE(z_ch=4).to(device).eval()
vae.load_state_dict(torch.load("ckpts/vae.pt", map_location=device))

dl = DataLoader(Flowers102Clean(split="train", image_size=256), batch_size=32, shuffle=True, num_workers=0)

zs = []
with torch.no_grad():
    for i, b in enumerate(dl):
        x = b["image"].to(device) * 2 - 1
        z, _, _, _ = vae.enc(x)
        zs.append(z.flatten(1).cpu())
        if i == 200: break

Z = torch.cat(zs, 0)
print("latent mean:", Z.mean().item())
print("latent std :", Z.std(unbiased=False).item())
print("latent min/max:", Z.min().item(), Z.max().item())
