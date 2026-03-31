import torch
from torchvision.utils import save_image
from data_helpers.data_flowers import Flowers102Clean
from models_ae import KL_VAE

device = "cuda" if torch.cuda.is_available() else "cpu"
vae = KL_VAE(z_ch=4).to(device).eval()
vae.load_state_dict(torch.load("ckpts/vae.pt", map_location=device))

ds = Flowers102Clean(split="val", image_size=256)
x = ds[120]["image"].unsqueeze(0).to(device)          # [0,1]
with torch.no_grad():
    z, _, _, _ = vae.enc(x*2-1)
    save_image(z, "samples/vae_raw.png")
    xrec = vae.dec(z)                                # trained in [-1,1] space
# IMPORTANT: do NOT tanh here unless you trained decoder with tanh
save_image((x*1.0), "samples/vae_input.png")
save_image((xrec.clamp(-1,1)+1)/2, "samples/vae_recon.png")
print("saved samples/vae_input.png and samples/vae_recon.png")
