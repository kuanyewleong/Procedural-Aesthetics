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

def _box_blur(bchw: torch.Tensor, k: int) -> torch.Tensor:
    pad = k // 2
    x = F.pad(bchw, (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(x, kernel_size=k, stride=1)

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

# ---------------------------
# Core Halftone (AM screening)
# ---------------------------

class HalftoneDots:
    """
    AM halftone dots using a grid:
      - Lay a grid (cell_size pixels)
      - Sample brightness at cell center
      - Dot radius depends on darkness (1 - brightness)
      - Render circle by distance field

    Output is a 1-channel mask in [0,1] where 1=ink, 0=paper (or inverted).
    """
    def __init__(
        self,
        cell_size: int = 8,
        gamma: float = 1.0,      # gamma applied to brightness before mapping to radius
        max_radius: float = 0.48,# relative to cell_size (0.5 touches borders)
        softness: float = 1.5,   # edge softness in pixels (higher = softer dots)
        invert: bool = False,    # if True: dots represent light instead of dark
        jitter: float = 0.0,     # jitter dot centers within cell in pixels (0..~1.5)
    ):
        self.cell_size = int(cell_size)
        self.gamma = float(gamma)
        self.max_radius = float(max_radius)
        self.softness = float(softness)
        self.invert = bool(invert)
        self.jitter = float(jitter)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = _ensure_bchw(img01)   # [1,3,H,W]
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        gray = _gray(x)  # [1,1,H,W] brightness in [0,1]

        cs = self.cell_size
        # number of cells
        gh = (H + cs - 1) // cs
        gw = (W + cs - 1) // cs

        # pad to full grid
        pad_h = gh * cs - H
        pad_w = gw * cs - W
        gray_p = F.pad(gray, (0, pad_w, 0, pad_h), mode="reflect")  # [1,1,gh*cs,gw*cs]

        # sample brightness at cell centers by pooling then upsampling
        # avg pool gives stable tone sampling (center sample can be noisy)
        tone = F.avg_pool2d(gray_p, kernel_size=cs, stride=cs)  # [1,1,gh,gw]
        tone = tone.clamp(0, 1)

        # map brightness->radius (dark area => bigger dot)
        # darkness d = 1 - tone
        if self.invert:
            d = tone
        else:
            d = 1.0 - tone

        # optional gamma shaping
        d = d.clamp(0, 1) ** self.gamma

        # radius in pixels
        r = d * (self.max_radius * cs)  # [1,1,gh,gw]

        # build per-pixel coordinates within each cell
        # coordinates 0..cs-1
        yy = torch.arange(gh * cs, device=device, dtype=dtype).view(1, 1, gh * cs, 1)
        xx = torch.arange(gw * cs, device=device, dtype=dtype).view(1, 1, 1, gw * cs)

        # local coords within cell
        y0 = (yy % cs) + 0.5
        x0 = (xx % cs) + 0.5

        # dot center (normally at cs/2, cs/2), with optional jitter per cell
        cy = cs * 0.5
        cx = cs * 0.5

        if self.jitter > 0:
            # jitter per cell, then broadcast
            j = self.jitter
            jy = (torch.rand(1, 1, gh, gw, device=device, dtype=dtype) * 2 - 1) * j
            jx = (torch.rand(1, 1, gh, gw, device=device, dtype=dtype) * 2 - 1) * j
            # upsample to per-pixel
            jy = jy.repeat_interleave(cs, dim=2).repeat_interleave(cs, dim=3)
            jx = jx.repeat_interleave(cs, dim=2).repeat_interleave(cs, dim=3)
        else:
            jy = 0.0
            jx = 0.0

        # distance from dot center
        dy = (y0 - (cy + jy))
        dx = (x0 - (cx + jx))
        dist = torch.sqrt(dx * dx + dy * dy + 1e-8)  # [1,1,gh*cs,gw*cs]

        # expand radius to per-pixel by repeating cells
        r_px = r.repeat_interleave(cs, dim=2).repeat_interleave(cs, dim=3)  # [1,1,gh*cs,gw*cs]

        # smooth circle: ink=1 inside radius, 0 outside, with softness
        # use a smoothstep-ish transition
        s = max(self.softness, 1e-3)
        ink = torch.clamp((r_px - dist) / s, min=0.0, max=1.0)
        # crop back to original size
        ink = ink[:, :, :H, :W]
        return ink.squeeze(0)  # [1,H,W]


class HalftoneMono:
    """
    Applies halftone dots to grayscale and composites onto paper/ink.
    Output: [3,H,W]
    """
    def __init__(
        self,
        cell_size: int = 8,
        gamma: float = 1.0,
        max_radius: float = 0.48,
        softness: float = 1.5,
        ink_color=(0.05, 0.05, 0.05),
        paper_color=(1.0, 1.0, 1.0),
        jitter: float = 0.0,
    ):
        self.dots = HalftoneDots(cell_size=cell_size, gamma=gamma,
                                 max_radius=max_radius, softness=softness,
                                 invert=False, jitter=jitter)
        self.ink_color = ink_color
        self.paper_color = paper_color

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        ink = self.dots(img01)  # [1,H,W] ink mask
        H, W = ink.shape[-2], ink.shape[-1]
        ink3 = ink.repeat(3, 1, 1)

        ink_c = torch.tensor(self.ink_color, device=img01.device, dtype=img01.dtype).view(3,1,1)
        pap_c = torch.tensor(self.paper_color, device=img01.device, dtype=img01.dtype).view(3,1,1)

        out = pap_c * (1 - ink3) + ink_c * ink3
        return out.clamp(0, 1)


# ---------------------------
# Color Halftone (CMY-ish screen)
# ---------------------------

class HalftoneCMY:
    """
    Simple pop-art halftone via CMY-ish separation:
      - Convert RGB->CMY (approx): C=1-R, M=1-G, Y=1-B
      - For each channel, generate its own dot screen with different angle/offset feel
      - Composite by subtracting inks from paper
    """
    def __init__(
        self,
        cell_size: int = 8,
        gamma: float = 1.0,
        max_radius: float = 0.48,
        softness: float = 1.3,
        paper_color=(1.0, 0.98, 0.95),
        cmy_colors=((0.0, 0.75, 0.75), (0.85, 0.0, 0.85), (0.95, 0.85, 0.0)),
        jitter: float = 0.0,
        # per-channel offsets (in pixels) to mimic rotated screens without full rotation
        offsets=((0.0, 0.0), (0.33, 0.66), (0.66, 0.33)),
    ):
        self.cell_size = int(cell_size)
        self.gamma = float(gamma)
        self.max_radius = float(max_radius)
        self.softness = float(softness)
        self.paper_color = paper_color
        self.cmy_colors = cmy_colors
        self.jitter = float(jitter)
        self.offsets = offsets

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = img01.clamp(0, 1)
        C = (1.0 - x[0:1]).clamp(0, 1)
        M = (1.0 - x[1:2]).clamp(0, 1)
        Y = (1.0 - x[2:3]).clamp(0, 1)
        chans = [C, M, Y]

        H, W = x.shape[-2], x.shape[-1]
        device, dtype = x.device, x.dtype

        paper = torch.tensor(self.paper_color, device=device, dtype=dtype).view(3,1,1).expand(3,H,W).clone()
        out = paper

        # Per-channel dot masks
        for i, ch in enumerate(chans):
            # build a fake "image" where brightness=1-ink amount, so dots represent ink
            # We'll pass a 3ch image made from channel for HalftoneDots
            fake_rgb = torch.cat([1 - ch, 1 - ch, 1 - ch], dim=0)  # [3,H,W]

            # apply offset by rolling image (cheap screen rotation feel)
            oy, ox = self.offsets[i]
            ry = int(round(oy * self.cell_size))
            rx = int(round(ox * self.cell_size))
            if ry != 0 or rx != 0:
                fake_rgb = torch.roll(fake_rgb, shifts=(ry, rx), dims=(1, 2))

            dots = HalftoneDots(
                cell_size=self.cell_size,
                gamma=self.gamma,
                max_radius=self.max_radius,
                softness=self.softness,
                invert=False,
                jitter=self.jitter,
            )(fake_rgb)  # [1,H,W], 1=ink

            # unroll back to keep alignment stable
            if ry != 0 or rx != 0:
                dots = torch.roll(dots, shifts=(-ry, -rx), dims=(1, 2))

            ink_color = torch.tensor(self.cmy_colors[i], device=device, dtype=dtype).view(3,1,1)
            out = out * (1 - dots) + ink_color * dots  # overlay ink on paper

        return out.clamp(0, 1)


# ---------------------------
# Extras: Ink outline + pop contrast
# ---------------------------

class InkEdges:
    """
    Dark line art overlay using Sobel edges.
    """
    def __init__(self, strength: float = 0.6, threshold: float = 0.12, blur_k: int = 3):
        self.strength = float(strength)
        self.threshold = float(threshold)
        self.blur_k = int(blur_k)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = _ensure_bchw(img01)
        g = _gray(x)
        e = _sobel_edges(g)
        e = e / (e.amax(dim=(2, 3), keepdim=True) + 1e-8)
        if self.blur_k > 1:
            e = _box_blur(e, self.blur_k)

        mask = (e - self.threshold).clamp(min=0.0)
        mask = mask / (mask.amax(dim=(2, 3), keepdim=True) + 1e-8)
        mask = mask.clamp(0.0, 1.0)

        ink = 1.0 - self.strength * mask
        out = x * ink
        return _back_to_chw(out)


class PopContrast:
    """
    Stronger contrast / saturation-ish push to mimic pop print.
    """
    def __init__(self, contrast: float = 0.5, sat: float = 0.25):
        self.contrast = float(contrast)
        self.sat = float(sat)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = img01.clamp(0, 1)
        # contrast around mid gray
        c = self.contrast
        x = (x - 0.5) * (1.0 + c) + 0.5

        # fake saturation: push away from luminance
        g = (0.2989 * x[0:1] + 0.5870 * x[1:2] + 0.1140 * x[2:3])
        x = x + self.sat * (x - g)

        return x.clamp(0, 1)


# ---------------------------
# Pipelines
# ---------------------------

class HalftonePipelineMono:
    """
    Newspaper-ish mono halftone + ink edges.
    """
    def __init__(
        self,
        cell_size: int = 8,
        gamma: float = 1.0,
        max_radius: float = 0.48,
        softness: float = 1.5,
        paper_color=(0.98, 0.97, 0.94),
        ink_color=(0.08, 0.07, 0.06),
        edge_strength: float = 0.6,
    ):
        self.halftone = HalftoneMono(
            cell_size=cell_size, gamma=gamma, max_radius=max_radius, softness=softness,
            ink_color=ink_color, paper_color=paper_color, jitter=0.5
        )
        self.edges = InkEdges(strength=edge_strength, threshold=0.12, blur_k=3)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = self.halftone(img01)
        x = self.edges(x)
        return x.clamp(0, 1)


class HalftonePipelinePopArt:
    """
    Lichtenstein-ish pop-art:
      - CMY halftone screens + pop contrast + optional ink edges
    """
    def __init__(
        self,
        cell_size: int = 8,
        gamma: float = 1.0,
        max_radius: float = 0.48,
        softness: float = 1.2,
        edge_strength: float = 0.45,
    ):
        self.cmy = HalftoneCMY(
            cell_size=cell_size, gamma=gamma, max_radius=max_radius, softness=softness,
            paper_color=(1.0, 0.98, 0.95),
            cmy_colors=((0.05, 0.75, 0.85), (0.95, 0.15, 0.75), (0.98, 0.9, 0.15)),
            jitter=0.35,
            offsets=((0.0, 0.0), (0.33, 0.66), (0.66, 0.33)),
        )
        self.pop = PopContrast(contrast=0.65, sat=0.45)
        self.edges = InkEdges(strength=edge_strength, threshold=0.14, blur_k=3)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = self.cmy(img01)
        x = self.pop(x)
        x = self.edges(x)
        return x.clamp(0, 1)
