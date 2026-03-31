import torch
import torch.nn.functional as F

# ---------------------------
# Helpers
# ---------------------------

def _ensure_bchw(img01: torch.Tensor) -> torch.Tensor:
    assert img01.dim() == 3 and img01.shape[0] == 3, "expected [3,H,W]"
    return img01.unsqueeze(0)  # [1,3,H,W]

def _back_to_chw(x: torch.Tensor) -> torch.Tensor:
    return x.squeeze(0).clamp(0.0, 1.0)

def _gray(bchw: torch.Tensor) -> torch.Tensor:
    # bchw: [B,3,H,W]
    r, g, b = bchw[:, 0:1], bchw[:, 1:2], bchw[:, 2:3]
    return (0.2989 * r + 0.5870 * g + 0.1140 * b)

def _box_blur(bchw: torch.Tensor, k: int) -> torch.Tensor:
    # fast average blur using pooling
    pad = k // 2
    x = F.pad(bchw, (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(x, kernel_size=k, stride=1)

def _gaussian_blur(bchw: torch.Tensor, k: int = 5, sigma: float = 1.0) -> torch.Tensor:
    # separable gaussian blur (small k)
    assert k % 2 == 1
    device = bchw.device
    dtype = bchw.dtype
    x = torch.arange(k, device=device, dtype=dtype) - (k // 2)
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = g / g.sum()
    g1 = g.view(1, 1, 1, k)  # horizontal
    g2 = g.view(1, 1, k, 1)  # vertical

    B, C, H, W = bchw.shape
    # depthwise conv
    w1 = g1.repeat(C, 1, 1, 1)
    w2 = g2.repeat(C, 1, 1, 1)

    x = F.pad(bchw, (k//2, k//2, 0, 0), mode="reflect")
    x = F.conv2d(x, w1, groups=C)
    x = F.pad(x, (0, 0, k//2, k//2), mode="reflect")
    x = F.conv2d(x, w2, groups=C)
    return x

def _sobel_edges(gray: torch.Tensor) -> torch.Tensor:
    # gray: [B,1,H,W] -> edges: [B,1,H,W] in [0, ~]
    device, dtype = gray.device, gray.dtype
    kx = torch.tensor([[[[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]]]], device=device, dtype=dtype)
    ky = torch.tensor([[[[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]]]], device=device, dtype=dtype)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)

# ---------------------------
# Poster Modules
# ---------------------------

class EdgePreservingSmooth:
    """
    Simple edge-preserving smoothing:
    - blur
    - blend original towards blur where gradients are low
    Produces flatter regions while keeping edges.
    """
    def __init__(self, blur_k: int = 7, edge_strength: float = 8.0, mix: float = 0.8):
        self.blur_k = blur_k
        self.edge_strength = edge_strength
        self.mix = mix

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = _ensure_bchw(img01)
        blur = _box_blur(x, self.blur_k)

        g = _gray(x)
        e = _sobel_edges(g)  # [B,1,H,W]
        # low edge -> allow smoothing; high edge -> keep original
        w = torch.exp(-self.edge_strength * e).clamp(0.0, 1.0)  # [B,1,H,W]
        w = w.repeat(1, 3, 1, 1)

        out = x * (1 - self.mix * w) + blur * (self.mix * w)
        return _back_to_chw(out)


class ColorQuantize:
    """
    Hard posterization / palette reduction by uniform quantization per channel.
    levels=6..10 works well.
    """
    def __init__(self, levels: int = 8):
        self.levels = int(levels)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = img01.clamp(0, 1)
        L = max(self.levels, 2)
        q = torch.round(x * (L - 1)) / (L - 1)
        return q.clamp(0, 1)


class SoftColorQuantize:
    """
    Soft quantization: avoids harsh banding by blending towards nearest bin.
    alpha controls how close to hard quantization (higher = harder).
    """
    def __init__(self, levels: int = 8, alpha: float = 12.0):
        self.levels = int(levels)
        self.alpha = float(alpha)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = img01.clamp(0, 1)
        L = max(self.levels, 2)
        hard = torch.round(x * (L - 1)) / (L - 1)
        out = x + torch.tanh(self.alpha * (hard - x)) * (hard - x)
        return out.clamp(0, 1)


class ContrastCurve:
    """
    Poster-like contrast shaping using a simple S-curve.
    amount: 0..1
    """
    def __init__(self, amount: float = 0.35):
        self.amount = float(amount)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = img01.clamp(0, 1)
        a = self.amount
        # smooth S-curve around 0.5
        out = x + a * (x - x * x) * (x - 0.5) * 4.0
        return out.clamp(0, 1)


class InkEdges:
    """
    Draw dark ink lines on edges.
    strength: how dark edges are
    threshold: edge magnitude threshold (lower = more lines)
    """
    def __init__(self, strength: float = 0.75, threshold: float = 0.12, blur_k: int = 3):
        self.strength = float(strength)
        self.threshold = float(threshold)
        self.blur_k = int(blur_k)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = _ensure_bchw(img01)
        g = _gray(x)
        e = _sobel_edges(g)

        # normalize edge magnitude roughly
        e = e / (e.amax(dim=(2, 3), keepdim=True) + 1e-8)
        if self.blur_k > 1:
            e = _box_blur(e, self.blur_k)

        # edge mask: 0 no edge -> 1 edge
        mask = (e - self.threshold).clamp(min=0.0)
        mask = mask / (mask.amax(dim=(2, 3), keepdim=True) + 1e-8)
        mask = mask.clamp(0.0, 1.0)

        # darken along edges
        ink = 1.0 - self.strength * mask
        out = x * ink
        return _back_to_chw(out)


class PosterPipeline:
    """
    A reasonable default poster-painting pipeline:
      edge-preserving smooth -> contrast -> (soft) quantize -> ink edges
    """
    def __init__(
        self,
        smooth_k: int = 7,
        smooth_mix: float = 0.85,
        smooth_edge: float = 10.0,
        contrast: float = 0.35,
        quant_levels: int = 8,
        quant_soft: bool = True,
        ink_strength: float = 0.7,
        ink_threshold: float = 0.12,
    ):
        self.smooth = EdgePreservingSmooth(blur_k=smooth_k, edge_strength=smooth_edge, mix=smooth_mix)
        self.contrast = ContrastCurve(amount=contrast)
        self.quant = SoftColorQuantize(levels=quant_levels, alpha=12.0) if quant_soft else ColorQuantize(levels=quant_levels)
        self.ink = InkEdges(strength=ink_strength, threshold=ink_threshold, blur_k=3)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = img01
        x = self.smooth(x)
        x = self.contrast(x)
        x = self.quant(x)
        x = self.ink(x)
        return x.clamp(0, 1)
