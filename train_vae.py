import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_helpers.data_flowers import Flowers102Clean
from models_ae import KL_VAE, kl_free_bits

import lpips


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k in self.shadow:
            self.shadow[k].mul_(self.decay).add_(msd[k], alpha=1 - self.decay)


def beta_warmup(epoch, warmup_epochs=20, beta_max=1e-3):
    if epoch >= warmup_epochs:
        return beta_max
    return beta_max * (epoch + 1) / warmup_epochs


@torch.no_grad()
def save_recon_grid(vae, batch, out_path, device):
    from torchvision.utils import save_image
    x01 = batch["image"].to(device)
    x = x01 * 2 - 1
    z, mu, logv = vae.encode(x, sample=False)
    xrec = vae.decode(mu).clamp(-1, 1)
    grid = torch.cat([x, xrec], dim=0)
    save_image((grid + 1) / 2, out_path, nrow=x.size(0))


def train_vae(
    out="ckpts/vae_512_lpips.pt",
    image_size=512,
    epochs=128,
    bs=2,
    lr=2e-4,
    wd=1e-6,
    base=96,
    beta_max=5e-3,
    warmup_epochs=10,
    grad_clip=1.0,
    use_ema=True,
    # loss weights
    w_l1=1.0,
    w_l2=0.05,
    w_lpips=0.4,
):
    os.makedirs("ckpts", exist_ok=True)
    os.makedirs("samples", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = Flowers102Clean(split="train", image_size=image_size)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=4,
                    pin_memory=True, drop_last=True)

    vae = KL_VAE(z_ch=4, base=base).to(device)
    opt = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=wd)

    # LPIPS network for perceptual loss
    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    ema = EMA(vae, decay=0.999) if use_ema else None

    for ep in range(epochs):
        vae.train()
        beta = beta_warmup(ep, warmup_epochs=warmup_epochs, beta_max=beta_max)

        pbar = tqdm(dl, desc=f"VAE {ep+1}/{epochs} | beta={beta:.2e}")
        for b in pbar:
            x = b["image"].to(device) * 2 - 1  # [-1,1]
            xrec, mu, logv, _ = vae(x)
            xrec = xrec.clamp(-1, 1)

            l1 = F.l1_loss(xrec, x)
            l2 = F.mse_loss(xrec, x)

            # LPIPS returns [B,1,1,1] or [B,1] depending on version; take mean
            lp = lpips_fn(xrec, x).mean()

            recon = w_l1 * l1 + w_l2 * l2 + w_lpips * lp
            kll = kl_free_bits(mu, logv)

            loss = recon + beta * kll

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(vae.parameters(), grad_clip)
            opt.step()

            if ema is not None:
                ema.update(vae)

            with torch.no_grad():
                mu_std = float(mu.std(unbiased=False))
                pbar.set_postfix(
                    loss=float(loss),
                    recon=float(recon),
                    l1=float(l1),
                    lpips=float(lp),
                    kl=float(kll),
                    mu_std=mu_std
                )

        # snapshot recon
        vae.eval()
        sample_batch = next(iter(dl))
        save_recon_grid(vae, sample_batch, f"samples/vae_recon_ep{ep+1:03d}.png", device)

        ck = {"vae": vae.state_dict()}
        if ema is not None:
            ck["ema"] = ema.shadow
        torch.save(ck, out)
        print("Saved", out)

    return out


if __name__ == "__main__":
    train_vae()
