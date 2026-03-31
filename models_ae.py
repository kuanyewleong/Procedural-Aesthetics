# models_ae.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def swish(x):  # SiLU
    return x * torch.sigmoid(x)


class ResBlock(nn.Module):
    def __init__(self, c, c_out=None, groups=8):
        super().__init__()
        c_out = c if c_out is None else c_out
        self.norm1 = nn.GroupNorm(groups, c)
        self.conv1 = nn.Conv2d(c, c_out, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Identity() if c == c_out else nn.Conv2d(c, c_out, 1)

    def forward(self, x):
        h = self.conv1(swish(self.norm1(x)))
        h = self.conv2(swish(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class Encoder(nn.Module):
    """
    512 -> 256 -> 128 -> 64 latent (3 downsamples = /8)
    """
    def __init__(self, in_ch=3, z_ch=4, base=128):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 4

        self.inp = nn.Conv2d(in_ch, c1, 3, padding=1)

        self.block1 = nn.Sequential(ResBlock(c1), ResBlock(c1))
        self.down1 = Downsample(c1)

        self.block2 = nn.Sequential(ResBlock(c1, c2), ResBlock(c2))
        self.down2 = Downsample(c2)

        self.block3 = nn.Sequential(ResBlock(c2, c3), ResBlock(c3))
        self.down3 = Downsample(c3)

        self.block4 = nn.Sequential(ResBlock(c3, c4), ResBlock(c4))
        self.mid = nn.Sequential(ResBlock(c4), ResBlock(c4))

        self.to_mu = nn.Conv2d(c4, z_ch, 1)
        self.to_logv = nn.Conv2d(c4, z_ch, 1)

    def forward(self, x, sample=True):
        h = self.inp(x)

        h = self.block1(h)
        h = self.down1(h)

        h = self.block2(h)
        h = self.down2(h)

        h = self.block3(h)
        h = self.down3(h)

        h = self.block4(h)
        h = self.mid(h)

        mu = self.to_mu(h)
        logv = self.to_logv(h).clamp(-20.0, 20.0)  # stability

        if sample:
            std = torch.exp(0.5 * logv)
            z = mu + std * torch.randn_like(std)
        else:
            z = mu

        return z, mu, logv


class Decoder(nn.Module):
    """
    latent 64 -> 128 -> 256 -> 512
    """
    def __init__(self, out_ch=3, z_ch=4, base=128):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 4

        self.inp = nn.Conv2d(z_ch, c4, 3, padding=1)

        self.mid = nn.Sequential(ResBlock(c4), ResBlock(c4))

        self.up3 = Upsample(c4)
        self.block3 = nn.Sequential(ResBlock(c4, c3), ResBlock(c3))

        self.up2 = Upsample(c3)
        self.block2 = nn.Sequential(ResBlock(c3, c2), ResBlock(c2))

        self.up1 = Upsample(c2)
        self.block1 = nn.Sequential(ResBlock(c2, c1), ResBlock(c1))

        self.out = nn.Sequential(
            nn.GroupNorm(8, c1),
            nn.SiLU(),
            nn.Conv2d(c1, out_ch, 3, padding=1),
        )

    def forward(self, z):
        h = self.inp(z)
        h = self.mid(h)

        h = self.up3(h)
        h = self.block3(h)

        h = self.up2(h)
        h = self.block2(h)

        h = self.up1(h)
        h = self.block1(h)

        return self.out(h)


class KL_VAE(nn.Module):
    def __init__(self, z_ch=4, base=128):
        super().__init__()
        self.enc = Encoder(in_ch=3, z_ch=z_ch, base=base)
        self.dec = Decoder(out_ch=3, z_ch=z_ch, base=base)

    def encode(self, x, sample=True):
        return self.enc(x, sample=sample)

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        z, mu, logv = self.encode(x, sample=True)
        xrec = self.decode(z)
        return xrec, mu, logv, z

def kl_free_bits(mu, logv, free_bits=0.5):
    # returns mean KL with free bits threshold (in nats)
    kl = 0.5 * (mu.pow(2) + logv.exp() - 1 - logv)   # [B,C,H,W]
    kl = kl.mean(dim=(0,2,3))                        # [C]
    kl = torch.clamp(kl, min=free_bits).mean()
    return kl
    
# def kl_loss(mu, logv):
#     return -0.5 * torch.mean(1 + logv - mu.pow(2) - logv.exp())
