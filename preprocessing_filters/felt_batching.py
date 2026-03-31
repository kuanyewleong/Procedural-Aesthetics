import torch
import os
import glob
import cv2
import torch.nn.functional as F
import numpy as np

def _ensure_bchw(img01: torch.Tensor) -> torch.Tensor:
    assert img01.dim() == 3 and img01.shape[0] == 3, "expected [3,H,W]"
    return img01.unsqueeze(0)

def _back_to_chw(x: torch.Tensor) -> torch.Tensor:
    return x.squeeze(0).clamp(0.0, 1.0)

def _gray(bchw: torch.Tensor) -> torch.Tensor:
    r, g, b = bchw[:, 0:1], bchw[:, 1:2], bchw[:, 2:3]
    return (0.2989 * r + 0.5870 * g + 0.1140 * b)

def _box_blur(bchw: torch.Tensor, k: int) -> torch.Tensor:
    pad = k // 2
    x = F.pad(bchw, (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(x, kernel_size=k, stride=1)

def _sobel(g: torch.Tensor):
    device, dtype = g.device, g.dtype
    kx = torch.tensor([[[[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]]]], device=device, dtype=dtype)
    ky = torch.tensor([[[[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]]]], device=device, dtype=dtype)
    gx = F.conv2d(g, kx, padding=1)
    gy = F.conv2d(g, ky, padding=1)
    return gx, gy

def _edge_mag(g: torch.Tensor) -> torch.Tensor:
    gx, gy = _sobel(g)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)

def _normalize01(x: torch.Tensor, eps=1e-8) -> torch.Tensor:
    mn = x.amin(dim=(2, 3), keepdim=True)
    mx = x.amax(dim=(2, 3), keepdim=True)
    return (x - mn) / (mx - mn + eps)

# ---------------------------
# Felt Modules
# ---------------------------

class FeltSoften:
    """
    Felt looks puffy and slightly out-of-focus. This reduces fine detail.
    """
    def __init__(self, k: int = 9, mix: float = 0.85):
        self.k = int(k)
        self.mix = float(mix)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = _ensure_bchw(img01)
        blur = _box_blur(x, self.k)
        out = x * (1 - self.mix) + blur * self.mix
        return _back_to_chw(out)


class FeltPalette:
    """
    Felt usually has limited dyes/palette. Softly quantize colors.
    """
    def __init__(self, levels: int = 10, softness: float = 10.0):
        self.levels = int(levels)
        self.softness = float(softness)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = img01.clamp(0, 1)
        L = max(self.levels, 2)
        hard = torch.round(x * (L - 1)) / (L - 1)
        # soft pull toward bins to avoid harsh banding
        out = x + torch.tanh(self.softness * (hard - x)) * (hard - x)
        return out.clamp(0, 1)


class FeltShading:
    """
    Adds a subtle 'puffy' shading by using blurred luminance and a gentle curve.
    """
    def __init__(self, blur_k: int = 21, amount: float = 0.25):
        self.blur_k = int(blur_k)
        self.amount = float(amount)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = _ensure_bchw(img01)
        g = _gray(x)
        g_blur = _box_blur(g, self.blur_k)

        # shading factor around 1.0
        # push midtones slightly, like felt thickness
        shade = 1.0 + self.amount * (g_blur - 0.5) * 2.0
        shade = shade.clamp(0.75, 1.25)

        out = x * shade
        return _back_to_chw(out)


class FiberNoise:
    """
    Creates a fibrous texture map (wool-like grain) using multi-scale blurred noise.
    This is NOT Perlin, but it looks surprisingly felt-like once combined with flow.
    """
    def __init__(self, strength: float = 0.18, k_small: int = 7, k_large: int = 31):
        self.strength = float(strength)
        self.k_small = int(k_small)
        self.k_large = int(k_large)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = _ensure_bchw(img01)
        B, C, H, W = x.shape

        n = torch.randn((B, 1, H, W), device=x.device, dtype=x.dtype)
        n1 = _box_blur(n, self.k_small)
        n2 = _box_blur(n, self.k_large)

        # band-pass-ish: small - large, normalized
        tex = _normalize01(n1 - 0.5 * n2)
        tex = (tex - 0.5) * 2.0  # [-1,1] roughly

        out = x + self.strength * tex.repeat(1, 3, 1, 1)
        return _back_to_chw(out)


class FiberFlow:
    def __init__(self, strength: float = 0.22, angle_bins: int = 8, blur_len: int = 11):
        self.strength = float(strength)
        self.angle_bins = int(angle_bins)
        self.blur_len = int(blur_len)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = _ensure_bchw(img01)
        B, C, H, W = x.shape

        tex = torch.randn((B, 1, H, W), device=x.device, dtype=x.dtype)
        tex = _box_blur(tex, 5)

        g = _gray(x)
        gx, gy = _sobel(g)
        ang = torch.atan2(gy, gx)  # [-pi, pi]
        bins = ((ang + torch.pi) / (2 * torch.pi) * self.angle_bins).long().clamp(0, self.angle_bins - 1)

        k = self.blur_len
        pad = k // 2

        # kernels
        w_h = torch.ones((1, 1, 1, k), device=x.device, dtype=x.dtype) / k  # horizontal
        w_v = torch.ones((1, 1, k, 1), device=x.device, dtype=x.dtype) / k  # vertical

        # IMPORTANT: pad only along the kernel direction
        tex_h = F.pad(tex, (pad, pad, 0, 0), mode="reflect")  # left/right only
        th = F.conv2d(tex_h, w_h)  # -> [B,1,H,W]

        tex_v = F.pad(tex, (0, 0, pad, pad), mode="reflect")  # top/bottom only
        tv = F.conv2d(tex_v, w_v)  # -> [B,1,H,W]

        # cheap diagonals from horizontal blur (same shape)
        td1 = 0.5 * (torch.roll(th, shifts=1, dims=2) + torch.roll(th, shifts=-1, dims=2))  # \
        td2 = 0.5 * (torch.roll(th, shifts=1, dims=3) + torch.roll(th, shifts=-1, dims=3))  # / (approx)

        # map bins to 4 directions
        dir_id = (bins % 4)  # [B,1,H,W] long
        m0 = (dir_id == 0).float()
        m1 = (dir_id == 1).float()
        m2 = (dir_id == 2).float()
        m3 = (dir_id == 3).float()

        tex_dir = th * m0 + td1 * m1 + tv * m2 + td2 * m3  # all [B,1,H,W]
        tex_dir = _normalize01(tex_dir)
        tex_dir = (tex_dir - 0.5) * 2.0  # [-1,1]

        out = x + self.strength * tex_dir.repeat(1, 3, 1, 1)
        return _back_to_chw(out)


class FeltSeams:
    """
    Adds 'stitch/seam' like darker outlines along strong edges.
    """
    def __init__(self, strength: float = 0.35, threshold: float = 0.18, blur_k: int = 5):
        self.strength = float(strength)
        self.threshold = float(threshold)
        self.blur_k = int(blur_k)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = _ensure_bchw(img01)
        g = _gray(x)
        e = _edge_mag(g)
        e = e / (e.amax(dim=(2, 3), keepdim=True) + 1e-8)
        if self.blur_k > 1:
            e = _box_blur(e, self.blur_k)

        seam = (e - self.threshold).clamp(min=0.0)
        seam = seam / (seam.amax(dim=(2, 3), keepdim=True) + 1e-8)
        seam = seam.clamp(0.0, 1.0)

        # darken along seams
        out = x * (1.0 - self.strength * seam)
        return _back_to_chw(out)


class FeltPipeline:
    """
    Default felt-art pipeline:
      soften -> shading -> palette -> fiber noise -> fiber flow -> seams
    """
    def __init__(
        self,
        soften_k: int = 9,
        soften_mix: float = 0.85,
        shade_k: int = 21,
        shade_amount: float = 0.22,
        palette_levels: int = 10,
        palette_softness: float = 10.0,
        fiber_noise_strength: float = 0.12,
        fiber_flow_strength: float = 0.18,
        seam_strength: float = 0.30,
        seam_threshold: float = 0.18,
    ):
        self.soften = FeltSoften(k=soften_k, mix=soften_mix)
        self.shade = FeltShading(blur_k=shade_k, amount=shade_amount)
        self.palette = FeltPalette(levels=palette_levels, softness=palette_softness)
        self.noise = FiberNoise(strength=fiber_noise_strength, k_small=7, k_large=31)
        self.flow = FiberFlow(strength=fiber_flow_strength, angle_bins=8, blur_len=11)
        self.seams = FeltSeams(strength=seam_strength, threshold=seam_threshold, blur_k=5)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = img01
        x = self.soften(x)
        x = self.shade(x)
        x = self.palette(x)
        x = self.noise(x)
        x = self.flow(x)
        x = self.seams(x)
        return x.clamp(0, 1)


# ---------------------------
# img_preprocess function
# ---------------------------
def img_preprocess(image: np.ndarray, pipeline: FeltPipeline = None) -> np.ndarray:
    """
    Apply the felt-art pipeline to an OpenCV image.

    Args:
        image: numpy array (H, W, 3) in BGR order, dtype uint8.
        pipeline: optional FeltPipeline instance; uses default if None.

    Returns:
        numpy array (H, W, 3) in BGR order, dtype uint8, after felt effect.
    """
    # Convert BGR to RGB – .copy() resolves negative stride issue
    img_rgb = image[..., ::-1].copy()          # (H,W,3) uint8, RGB

    # Convert to torch tensor: [0,1] float, shape (3,H,W)
    img_tensor = torch.from_numpy(img_rgb).float().div(255.0).permute(2, 0, 1)

    # Apply pipeline (default if none given)
    if pipeline is None:
        pipeline = FeltPipeline()
    out_tensor = pipeline(img_tensor)           # still (3,H,W) in [0,1]

    # Convert back to numpy uint8 BGR
    out_np = (out_tensor.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    out_bgr = out_np[..., ::-1]

    return out_bgr

##########################################################################

SRC_DIR = "data/flowers-102/jpg_multiflower_raw"
DST_DIR = "data/flowers-102/jpg_felt"

os.makedirs(DST_DIR, exist_ok=True)

# Grab all jpg/JPG/jpeg/JPEG files in the source folder
img_paths = []
for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG"):
    img_paths.extend(glob.glob(os.path.join(SRC_DIR, ext)))
img_paths.sort()

if not img_paths:
    raise FileNotFoundError(f"No images found in: {SRC_DIR}")


saved = 0
failed = 0

for in_path in img_paths:
    image = cv2.imread(in_path)
    if image is None:
        print(f"[WARN] Failed to read: {in_path}")
        failed += 1
        continue

    out_img = img_preprocess(image)

    # Keep the original filename, save into destination folder
    base_name = os.path.basename(in_path)
    out_path = os.path.join(DST_DIR, base_name)

    ok = cv2.imwrite(out_path, out_img)
    if not ok:
        print(f"[WARN] Failed to write: {out_path}")
        failed += 1
        continue

    saved += 1
    if saved % 100 == 0:
        print(f"Saved {saved}/{len(img_paths)}...")

print(f"Done. Saved: {saved}, Failed: {failed}, Total: {len(img_paths)}")
print(f"Output folder: {DST_DIR}")
