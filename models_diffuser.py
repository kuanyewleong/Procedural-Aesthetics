import torch
from diffusers import UNet2DConditionModel

def build_sd15_unet(sample_size: int):
    # SD1.5-ish config but with sample_size matching YOUR latent spatial size.
    return UNet2DConditionModel(
        sample_size=sample_size,          # <-- IMPORTANT: match z0 H/W (likely image_size if your VAE keeps 256x256)
        in_channels=4,
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(320, 640, 1280, 1280),
        down_block_types=(
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        cross_attention_dim=768,           # SD1.5 text hidden size
        attention_head_dim=8,
    )

# # usage example
# unet = build_sd15_unet()

# B, H, W = 2, 64, 64
# x   = torch.randn(B, 4, H, W)               # latent noisy sample x_t
# t   = torch.randint(0, 1000, (B,), dtype=torch.long)
# ctx = torch.randn(B, 77, 768)               # CLIP text hidden states

# eps_hat = unet(x, t, encoder_hidden_states=ctx).sample
# print(eps_hat.shape)  # [B, 4, H, W]
