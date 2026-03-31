import math
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
    return 0.2989 * r + 0.5870 * g + 0.1140 * b

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

def _soft_brush_blur(mask: torch.Tensor, k: int = 5, sigma: float = 1.0) -> torch.Tensor:
    """
    mask: [1,1,H,W]
    Softens brush-mask edges.
    """
    if k <= 1:
        return mask
    device, dtype = mask.device, mask.dtype
    ax = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2
    g = torch.exp(-(ax * ax) / (2 * sigma * sigma))
    g = g / g.sum()
    k2d = (g[:, None] * g[None, :]).view(1, 1, k, k)
    return F.conv2d(mask, k2d, padding=k // 2)

def _brush_stroke_mask(
    h: int,
    w: int,
    cx: float,
    cy: float,
    angle: float,
    length: float,
    thickness: float,
    device,
    dtype,
):
    """
    Soft mask for one brush stroke segment.
    Returns [1,1,H,W].
    """
    yy = torch.arange(h, device=device, dtype=dtype).view(1, 1, h, 1)
    xx = torch.arange(w, device=device, dtype=dtype).view(1, 1, 1, w)

    dx = xx - cx
    dy = yy - cy

    ca = math.cos(angle)
    sa = math.sin(angle)

    # local coordinates relative to brush direction
    u = dx * ca + dy * sa
    v = -dx * sa + dy * ca

    half_len = max(length * 0.5, 1.0)
    thick = max(thickness, 1.0)

    inside_u = (u.abs() <= half_len).to(dtype)
    brush_core = torch.exp(-(v * v) / (2 * (thick * 0.5) ** 2))
    tip_falloff = torch.clamp(1.0 - (u.abs() - half_len + thick) / thick, 0.0, 1.0)

    m = brush_core * torch.maximum(inside_u, tip_falloff)
    return m.clamp(0.0, 1.0)

def _chaotic_brushes_mask(
    h: int,
    w: int,
    n_brushes: int,
    min_length: float,
    max_length: float,
    min_thickness: float,
    max_thickness: float,
    chaos: float,
    device,
    dtype,
):
    """
    Build a mask made of many scattered brush strokes.
    Returns [1,1,H,W], where 1 means brush lining region.
    """
    mask = torch.zeros((1, 1, h, w), device=device, dtype=dtype)

    base_dirs = torch.tensor(
        [0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4],
        device=device,
        dtype=dtype,
    )

    for _ in range(n_brushes):
        cx = torch.rand((), device=device, dtype=dtype) * (w - 1)
        cy = torch.rand((), device=device, dtype=dtype) * (h - 1)

        if chaos < 0.35:
            a0 = base_dirs[torch.randint(0, len(base_dirs), (1,), device=device)].item()
            angle = a0 + (torch.rand((), device=device).item() * 2 - 1) * (math.pi / 10)
        else:
            if torch.rand((), device=device).item() < (1.0 - chaos * 0.5):
                a0 = base_dirs[torch.randint(0, len(base_dirs), (1,), device=device)].item()
                angle = a0 + (torch.rand((), device=device).item() * 2 - 1) * (math.pi / 4) * chaos
            else:
                angle = torch.rand((), device=device).item() * math.pi

        length = min_length + torch.rand((), device=device).item() * (max_length - min_length)
        thickness = min_thickness + torch.rand((), device=device).item() * (max_thickness - min_thickness)

        stroke = _brush_stroke_mask(
            h=h,
            w=w,
            cx=float(cx.item()),
            cy=float(cy.item()),
            angle=float(angle),
            length=float(length),
            thickness=float(thickness),
            device=device,
            dtype=dtype,
        )

        strength = 0.55 + 0.45 * torch.rand((), device=device).item()
        stroke = stroke * strength

        mask = torch.maximum(mask, stroke)

    noise = torch.rand((1, 1, h, w), device=device, dtype=dtype)
    mask = mask * (0.85 + 0.30 * noise)

    return mask.clamp(0.0, 1.0)

# ---------------------------
# Mosaic-style base modules
# ---------------------------

class MosaicDownUp:
    """
    Basic mosaic: average pool into tiles, then nearest upsample.
    """
    def __init__(self, tile: int = 16):
        self.tile = int(tile)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = _ensure_bchw(img01)
        _, _, H, W = x.shape
        t = max(self.tile, 1)

        pad_h = (t - (H % t)) % t
        pad_w = (t - (W % t)) % t
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        pooled = F.avg_pool2d(x_pad, kernel_size=t, stride=t)
        up = F.interpolate(pooled, scale_factor=t, mode="nearest")
        up = up[:, :, :H, :W]
        return _back_to_chw(up)

class MosaicJitter:
    def __init__(self, tile: int = 16, jitter_brightness: float = 0.08, jitter_color: float = 0.04):
        self.tile = int(tile)
        self.jb = float(jitter_brightness)
        self.jc = float(jitter_color)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = _ensure_bchw(img01)
        B, C, H, W = x.shape
        t = max(self.tile, 1)

        pad_h = (t - (H % t)) % t
        pad_w = (t - (W % t)) % t
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        pooled = F.avg_pool2d(x_pad, kernel_size=t, stride=t)
        Bh, Cc, Ht, Wt = pooled.shape

        if self.jb > 0:
            br = (torch.rand(Bh, 1, Ht, Wt, device=x.device, dtype=x.dtype) * 2 - 1) * self.jb
        else:
            br = 0.0

        if self.jc > 0:
            cj = (torch.rand(Bh, 3, Ht, Wt, device=x.device, dtype=x.dtype) * 2 - 1) * self.jc
        else:
            cj = 0.0

        pooled_j = (pooled + cj + br).clamp(0, 1)
        up = F.interpolate(pooled_j, scale_factor=t, mode="nearest")
        up = up[:, :, :H, :W]
        return _back_to_chw(up)

class EdgeAwareMosaic:
    def __init__(self, tile_small: int = 8, tile_big: int = 24, edge_strength: float = 8.0):
        self.tile_small = int(tile_small)
        self.tile_big = int(tile_big)
        self.edge_strength = float(edge_strength)

        self.ms = MosaicDownUp(tile_small)
        self.mb = MosaicDownUp(tile_big)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = _ensure_bchw(img01)
        g = _gray(x)
        e = _sobel_edges(g)
        e = e / (e.amax(dim=(2, 3), keepdim=True) + 1e-8)
        w = torch.clamp(e * self.edge_strength, 0.0, 1.0)
        w3 = w.repeat(1, 3, 1, 1)

        small = _ensure_bchw(self.ms(img01))
        big = _ensure_bchw(self.mb(img01))
        out = small * w3 + big * (1 - w3)
        return _back_to_chw(out)

# ---------------------------
# ChaoticBrushes modules
# ---------------------------

class ChaoticBrushes:
    """
    Mosaic base with scattered multi-directional chaotic brush strokes.

    n_brushes: how many scattered brush strokes to generate
    min_length/max_length: brush length range in pixels
    min_thickness/max_thickness: brush thickness range in pixels
    chaos: 0..1, larger = more random directions and placement
    softness: blur amount for softer brush edges
    threshold: lower -> more visible brush coverage
    brush_color: overlay color for the brush strokes
    """
    def __init__(
        self,
        tile: int = 16,
        brush_color=(0.05, 0.05, 0.05),
        n_brushes: int = 140,
        min_length: float = 10.0,
        max_length: float = 80.0,
        min_thickness: float = 1.0,
        max_thickness: float = 3.5,
        chaos: float = 0.85,
        softness: float = 1.2,
        threshold: float = 0.42,
    ):
        self.tile = int(tile)
        self.n_brushes = int(n_brushes)
        self.min_length = float(min_length)
        self.max_length = float(max_length)
        self.min_thickness = float(min_thickness)
        self.max_thickness = float(max_thickness)
        self.chaos = float(chaos)
        self.softness = float(softness)
        self.threshold = float(threshold)

        if isinstance(brush_color, (float, int)):
            brush_color = (float(brush_color),) * 3
        self.brush_color = torch.tensor(brush_color, dtype=torch.float32).view(1, 3, 1, 1)

        self.mosaic = MosaicDownUp(tile=tile)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = _ensure_bchw(img01)
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        base_tiles = _ensure_bchw(self.mosaic(img01))

        brush_mask = _chaotic_brushes_mask(
            h=H,
            w=W,
            n_brushes=self.n_brushes,
            min_length=self.min_length,
            max_length=self.max_length,
            min_thickness=self.min_thickness,
            max_thickness=self.max_thickness,
            chaos=self.chaos,
            device=device,
            dtype=dtype,
        )

        if self.softness > 0:
            k = max(3, int(round(self.softness * 4)) | 1)
            brush_mask = _soft_brush_blur(brush_mask, k=k, sigma=max(0.5, self.softness))

        brush_mask = (brush_mask > self.threshold).to(dtype)
        brush_mask3 = brush_mask.repeat(1, 3, 1, 1)

        brush_color = self.brush_color.to(device=device, dtype=dtype).expand(B, -1, H, W)
        out = torch.where(brush_mask3 > 0.5, brush_color, base_tiles)
        return _back_to_chw(out)

class ChaoticBrushesPipeline:
    """
    A painterly chaotic-brush pipeline:
      edge-aware tiling -> jitter -> chaotic brush overlay
    """
    def __init__(
        self,
        tile_small: int = 8,
        tile_big: int = 24,
        edge_strength: float = 8.0,
        jitter_brightness: float = 0.06,
        jitter_color: float = 0.03,
        brush_color=(0.05, 0.05, 0.05),
        n_brushes: int = 160,
        min_length: float = 8.0,
        max_length: float = 90.0,
        min_thickness: float = 1.0,
        max_thickness: float = 3.0,
        chaos: float = 0.9,
        softness: float = 1.2,
        threshold: float = 0.40,
    ):
        self.edge = EdgeAwareMosaic(tile_small=tile_small, tile_big=tile_big, edge_strength=edge_strength)
        self.jitter = MosaicJitter(tile=tile_small, jitter_brightness=jitter_brightness, jitter_color=jitter_color)
        self.brushes = ChaoticBrushes(
            tile=tile_small,
            brush_color=brush_color,
            n_brushes=n_brushes,
            min_length=min_length,
            max_length=max_length,
            min_thickness=min_thickness,
            max_thickness=max_thickness,
            chaos=chaos,
            softness=softness,
            threshold=threshold,
        )

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = self.edge(img01)
        x = self.jitter(x)
        x = self.brushes(x)
        return x.clamp(0, 1)