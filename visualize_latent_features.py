import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
from data_helpers.data_flowers import Flowers102Clean
from models_ae import KL_VAE

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load Model
vae = KL_VAE(z_ch=4).to(device).eval()
vae.load_state_dict(torch.load("ckpts/vae.pt", map_location=device))

# 2. Load Data (Get two different flowers for interpolation)
ds = Flowers102Clean(split="val", image_size=256)
img1 = ds[120]["image"].unsqueeze(0).to(device) # Flower A
img2 = ds[180]["image"].unsqueeze(0).to(device) # Flower B

with torch.no_grad():
    # --- PART A: LATENT HEATMAPS ---
    # Encode Flower A
    z1, _, _, _ = vae.enc(img1 * 2 - 1) 
    
    # Normalize z1 channels individually for better heatmap visualization
    # We take the 4 channels of the first batch [1, 4, H, W]
    z_visual = z1[0].cpu() 
    heatmaps = []
    for i in range(4):
        ch = z_visual[i]
        # Normalize to 0-1 range for display
        ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
        heatmaps.append(ch.unsqueeze(0))
    
    # Create a grid of the 4 latent channels    
        # 1. Create the grid (this results in [3, H, W])
    latent_grid_tensor = make_grid(heatmaps, nrow=4, padding=4)
    
    # 2. Move channels to the end: [3, H, W] -> [H, W, 3]
    # We also move it to CPU and convert to numpy
    latent_grid_np = latent_grid_tensor.permute(1, 2, 0).cpu().numpy()

    # 3. Plot using Matplotlib
    plt.figure(figsize=(12, 4))
    
    # NOTE: Since make_grid already made it 3-channel, 
    # we take just one channel to apply the colormap effectively
    plt.imshow(latent_grid_np[:, :, 0], cmap='inferno') 
    
    plt.axis('off')
    plt.colorbar(label="Activation Intensity")
    plt.title("Latent Space Feature Activations")
    plt.savefig("samples/latent_heatmaps_colorized.png", bbox_inches='tight')


    # Also save the raw grid without color map for reference
    latent_grid = make_grid(heatmaps, nrow=4, padding=2)
    save_image(latent_grid, "samples/latent_heatmaps.png")

    # --- PART B: LATENT INTERPOLATION ---
    z2, _, _, _ = vae.enc(img2 * 2 - 1)
    
    interpolation_steps = 8
    interp_list = []
    
    for i in range(interpolation_steps):
        alpha = i / (interpolation_steps - 1)
        # Linear interpolation between latent vectors
        z_interp = torch.lerp(z1, z2, alpha)
        
        # Decode the interpolated point
        out = vae.dec(z_interp)
        # Convert back to [0, 1] range
        out = (out + 1) / 2
        interp_list.append(out.squeeze(0))
    
    # Create a long strip showing the morphing process
    comparison_strip = make_grid(interp_list, nrow=interpolation_steps, padding=4)
    save_image(comparison_strip, "samples/latent_interpolation.png")

print("Visualizations saved: 'samples/latent_heatmaps.png' and 'samples/latent_interpolation.png'")
