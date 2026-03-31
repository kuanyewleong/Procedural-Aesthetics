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
    r, g, b = bchw[:, 0:1], bchw[:, 1:2], bchw[:, 2:3]
    return (0.2989 * r + 0.5870 * g + 0.1140 * b)

def _box_blur(bchw: torch.Tensor, k: int) -> torch.Tensor:
    pad = k // 2
    x = F.pad(bchw, (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(x, kernel_size=k, stride=1)

def _gaussian_kernel_1d(k: int, sigma: float, device, dtype):
    x = torch.arange(k, device=device, dtype=dtype) - (k // 2)
    g = torch.exp(-(x**2) / (2 * sigma**2))
    return g / g.sum()

def _gaussian_blur(bchw: torch.Tensor, k: int = 9, sigma: float = 2.0) -> torch.Tensor:
    assert k % 2 == 1
    B, C, H, W = bchw.shape
    device, dtype = bchw.device, bchw.dtype

    g = _gaussian_kernel_1d(k, sigma, device, dtype)
    w1 = g.view(1, 1, 1, k).repeat(C, 1, 1, 1)
    w2 = g.view(1, 1, k, 1).repeat(C, 1, 1, 1)

    x = F.pad(bchw, (k//2, k//2, 0, 0), mode="reflect")
    x = F.conv2d(x, w1, groups=C)
    x = F.pad(x, (0, 0, k//2, k//2), mode="reflect")
    x = F.conv2d(x, w2, groups=C)
    return x

def _sobel_edges(gray: torch.Tensor) -> torch.Tensor:
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

def _normalize01(x: torch.Tensor) -> torch.Tensor:
    mn = x.amin(dim=(-2, -1), keepdim=True)
    mx = x.amax(dim=(-2, -1), keepdim=True)
    return (x - mn) / (mx - mn + 1e-8)

# ---------------------------
# Post-Impressionist Modules
# ---------------------------

class ChromaticBoost:
    """
    Slightly pushes saturation + warms highlights / cools shadows.
    Gives more expressive Post-Impressionist color.
    """
    def __init__(self, sat: float = 1.15, warm: float = 0.06):
        self.sat = float(sat)
        self.warm = float(warm)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = _ensure_bchw(img01).clamp(0, 1)
        g = _gray(x)  # [1,1,H,W]

        # saturation: push away from gray
        x = g + (x - g) * self.sat

        # warm highlights / cool shadows using luminance
        # highlights -> add to R,G; shadows -> add to B slightly
        lum = g
        warm_mask = (lum - 0.5).clamp(min=0.0) * 2.0
        cool_mask = (0.5 - lum).clamp(min=0.0) * 2.0

        r, gg, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        r = r + self.warm * warm_mask
        gg = gg + 0.5 * self.warm * warm_mask
        b = b + 0.5 * self.warm * cool_mask

        x = torch.cat([r, gg, b], dim=1)
        return _back_to_chw(x)


class CoarsePaintSmoothing:
    """
    Paint-like smoothing: stronger blur in flat areas, preserves edges.
    Produces coherent color regions like brush paint masses.
    """
    def __init__(self, blur_k: int = 11, edge_strength: float = 12.0, mix: float = 0.85):
        self.blur_k = int(blur_k)
        self.edge_strength = float(edge_strength)
        self.mix = float(mix)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = _ensure_bchw(img01).clamp(0, 1)
        blur = _gaussian_blur(x, k=max(3, self.blur_k | 1), sigma=max(0.6, self.blur_k / 5))

        e = _sobel_edges(_gray(x))  # [1,1,H,W]
        # low edges => heavy smoothing, high edges => keep original
        w = torch.exp(-self.edge_strength * e).clamp(0, 1).repeat(1, 3, 1, 1)

        out = x * (1 - self.mix * w) + blur * (self.mix * w)
        return _back_to_chw(out)


class BrushStrokeField:
    """
    Adds directional stroke texture guided by image gradients.
    This is not true stroke rendering, but a convincing procedural approximation:
    - build a stroke "noise" field
    - blur it anisotropically along gradient direction (approx via oriented mix)
    """
    def __init__(self, strength: float = 0.22, scale: int = 6):
        self.strength = float(strength)
        self.scale = int(scale)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = _ensure_bchw(img01).clamp(0, 1)
        B, C, H, W = x.shape

        g = _gray(x)
        device, dtype = x.device, x.dtype

        # gradient direction
        kx = torch.tensor([[[[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]]]], device=device, dtype=dtype)
        ky = torch.tensor([[[[-1, -2, -1],
                             [ 0,  0,  0],
                             [ 1,  2,  1]]]], device=device, dtype=dtype)
        gx = F.conv2d(g, kx, padding=1)
        gy = F.conv2d(g, ky, padding=1)
        mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
        mag = _normalize01(mag)

        # base stroke noise (low frequency)
        nh, nw = max(8, H // self.scale), max(8, W // self.scale)
        noise = torch.randn((B, 1, nh, nw), device=device, dtype=dtype)
        noise = F.interpolate(noise, size=(H, W), mode="bilinear", align_corners=False)
        noise = _gaussian_blur(noise.repeat(1, 3, 1, 1), k=9, sigma=2.0)  # [B,3,H,W]

        # anisotropic-ish: mix horizontal vs vertical blur based on gradient direction
        # Where gx dominates => strokes vertical-ish; where gy dominates => horizontal-ish
        a = (gx.abs() / (gx.abs() + gy.abs() + 1e-8)).clamp(0, 1)  # [B,1,H,W]
        # horizontal blur
        hblur = F.avg_pool2d(F.pad(noise, (6, 6, 0, 0), mode="reflect"), kernel_size=(1, 13), stride=1)
        # vertical blur
        vblur = F.avg_pool2d(F.pad(noise, (0, 0, 6, 6), mode="reflect"), kernel_size=(13, 1), stride=1)
        stroke = (a.repeat(1,3,1,1) * vblur + (1 - a).repeat(1,3,1,1) * hblur)

        # apply more stroke texture where edges exist (mag)
        m = mag.repeat(1, 3, 1, 1)
        out = x + self.strength * (stroke - stroke.mean(dim=(2,3), keepdim=True)) * (0.35 + 0.65 * m)
        return _back_to_chw(out)


class PaletteSnap:
    """
    Soft palette snapping (like painterly reduced palette).
    Uses soft quantization and optional per-channel bias.
    """
    def __init__(self, levels: int = 10, alpha: float = 10.0):
        self.levels = int(levels)
        self.alpha = float(alpha)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = img01.clamp(0, 1)
        L = max(self.levels, 2)
        hard = torch.round(x * (L - 1)) / (L - 1)
        out = x + torch.tanh(self.alpha * (hard - x)) * (hard - x)
        return out.clamp(0, 1)


class CanvasTexture:
    """
    Adds subtle canvas/paint grain.
    """
    def __init__(self, strength: float = 0.06, scale: int = 3):
        self.strength = float(strength)
        self.scale = int(scale)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = _ensure_bchw(img01).clamp(0, 1)
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        nh, nw = max(8, H // (self.scale * 2)), max(8, W // (self.scale * 2))
        tex = torch.randn((B, 1, nh, nw), device=device, dtype=dtype)
        tex = F.interpolate(tex, size=(H, W), mode="bilinear", align_corners=False)
        tex = _gaussian_blur(tex, k=7, sigma=1.5)
        tex = tex.repeat(1, 3, 1, 1)

        # slightly stronger texture in midtones (paint catches light)
        g = _gray(x)
        mid = (1.0 - (g - 0.5).abs() * 2.0).clamp(0, 1).repeat(1, 3, 1, 1)

        out = x + self.strength * tex * (0.4 + 0.6 * mid)
        return _back_to_chw(out)


class PostImpressionistPipeline:
    """
    A reasonable default Post-Impressionist approximation:
      smoothing (paint masses) -> chroma boost -> strokes -> palette snap -> canvas texture
    """
    def __init__(
        self,
        smooth_k: int = 11,
        smooth_mix: float = 0.85,
        smooth_edge: float = 12.0,
        sat: float = 1.15,
        warm: float = 0.06,
        stroke_strength: float = 0.22,
        stroke_scale: int = 6,
        palette_levels: int = 10,
        palette_alpha: float = 10.0,
        canvas_strength: float = 0.06,
        canvas_scale: int = 3,
    ):
        self.smooth = CoarsePaintSmoothing(blur_k=smooth_k, edge_strength=smooth_edge, mix=smooth_mix)
        self.chroma = ChromaticBoost(sat=sat, warm=warm)
        self.strokes = BrushStrokeField(strength=stroke_strength, scale=stroke_scale)
        self.palette = PaletteSnap(levels=palette_levels, alpha=palette_alpha)
        self.canvas = CanvasTexture(strength=canvas_strength, scale=canvas_scale)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = img01
        x = self.smooth(x)
        x = self.chroma(x)
        x = self.strokes(x)
        x = self.palette(x)
        x = self.canvas(x)
        return x.clamp(0, 1)
