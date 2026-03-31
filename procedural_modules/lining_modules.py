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

def _tile_mask(h: int, w: int, tile: int, grout: int, device, dtype):
    """
    Original axis-aligned grout mask.
    Returns [1,1,H,W] grout mask (1 = grout line, 0 = tile face).
    """
    yy = torch.arange(h, device=device).view(1, 1, h, 1)
    xx = torch.arange(w, device=device).view(1, 1, 1, w)
    m = ((yy % tile) < grout) | ((xx % tile) < grout)
    return m.to(dtype)

def _diag_tile_mask(h: int, w: int, tile: int, grout: int, device, dtype, mode="cross"):
    """
    Diagonal grout mask using rotated coordinates.

    mode:
      - "slash"      -> one diagonal family: /
      - "backslash"  -> one diagonal family: \
      - "cross"      -> both diagonal families: X / diamond pattern
    """
    yy = torch.arange(h, device=device).view(1, 1, h, 1)
    xx = torch.arange(w, device=device).view(1, 1, 1, w)

    # Rotated coordinates
    # /  diagonals are constant for (x + y)
    # \ diagonals are constant for (x - y)
    u = xx + yy
    v = xx - yy

    if mode == "slash":
        m = (u % tile) < grout
    elif mode == "backslash":
        m = (torch.remainder(v, tile)) < grout
    elif mode == "cross":
        m = ((u % tile) < grout) | ((torch.remainder(v, tile)) < grout)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return m.to(dtype)

# ---------------------------
# Lining modules
# ---------------------------

class LiningDownUp:
    """
    Basic Lining: average pool into tiles, then nearest upsample.
    tile: tile size in pixels
    """
    def __init__(self, tile: int = 16):
        self.tile = int(tile)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = _ensure_bchw(img01)
        B, C, H, W = x.shape
        t = max(self.tile, 1)

        pad_h = (t - (H % t)) % t
        pad_w = (t - (W % t)) % t
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        pooled = F.avg_pool2d(x_pad, kernel_size=t, stride=t)
        up = F.interpolate(pooled, scale_factor=t, mode="nearest")

        up = up[:, :, :H, :W]
        return _back_to_chw(up)


class LiningWithGrout:
    """
    Lining tiles + grout lines.
    pattern:
      - "grid"       -> original x/y lines
      - "slash"      -> / diagonal lines
      - "backslash"  -> \ diagonal lines
      - "cross"      -> both diagonals
    """
    def __init__(
        self,
        tile: int = 16,
        grout: int = 2,
        grout_color=(0.05, 0.05, 0.05),
        pattern: str = "grid",
    ):
        self.tile = int(tile)
        self.grout = int(grout)
        self.pattern = pattern

        if isinstance(grout_color, (float, int)):
            grout_color = (float(grout_color),) * 3
        self.grout_color = torch.tensor(grout_color, dtype=torch.float32).view(1, 3, 1, 1)

        self.Lining = LiningDownUp(tile=tile)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = _ensure_bchw(img01)
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype

        tiles = _ensure_bchw(self.Lining(img01))  # [1,3,H,W]

        if self.pattern == "grid":
            grout_mask = _tile_mask(H, W, self.tile, self.grout, device, dtype)
        else:
            grout_mask = _diag_tile_mask(
                H, W, self.tile, self.grout, device, dtype, mode=self.pattern
            )

        grout_mask3 = grout_mask.repeat(1, 3, 1, 1)
        grout_color = self.grout_color.to(device=device, dtype=dtype).expand(B, -1, H, W)

        out = torch.where(grout_mask3 > 0.5, grout_color, tiles)
        return _back_to_chw(out)


class LiningJitter:
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


class EdgeAwareLining:
    def __init__(self, tile_small: int = 8, tile_big: int = 24, edge_strength: float = 8.0):
        self.tile_small = int(tile_small)
        self.tile_big = int(tile_big)
        self.edge_strength = float(edge_strength)

        self.ms = LiningDownUp(tile_small)
        self.mb = LiningDownUp(tile_big)

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


class LiningPipeline:
    def __init__(
        self,
        tile_small: int = 8,
        tile_big: int = 24,
        edge_strength: float = 8.0,
        jitter_brightness: float = 0.06,
        jitter_color: float = 0.03,
        grout: int = 2,
        grout_color=(0.05, 0.05, 0.05),
        grout_pattern: str = "cross",  # <-- changed
    ):
        self.edge = EdgeAwareLining(tile_small=tile_small, tile_big=tile_big, edge_strength=edge_strength)
        self.jitter = LiningJitter(tile=tile_small, jitter_brightness=jitter_brightness, jitter_color=jitter_color)
        self.grout = LiningWithGrout(
            tile=tile_small,
            grout=grout,
            grout_color=grout_color,
            pattern=grout_pattern,
        )

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = self.edge(img01)
        x = self.jitter(x)
        x = self.grout(x)
        return x.clamp(0, 1)