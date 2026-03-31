import torch
import torch.nn.functional as F

# ---------------------------
# Helpers
# ---------------------------

def _bchw(img01: torch.Tensor) -> torch.Tensor:
    assert img01.dim() == 3 and img01.shape[0] == 3, "expected [3,H,W]"
    return img01.unsqueeze(0)

def _chw(x: torch.Tensor) -> torch.Tensor:
    return x.squeeze(0).clamp(0.0, 1.0)

def _gray(bchw: torch.Tensor) -> torch.Tensor:
    r, g, b = bchw[:, 0:1], bchw[:, 1:2], bchw[:, 2:3]
    return (0.2989 * r + 0.5870 * g + 0.1140 * b)

def _sobel(gray: torch.Tensor):
    # gray: [B,1,H,W]
    device, dtype = gray.device, gray.dtype
    kx = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]], device=device, dtype=dtype)
    ky = torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]], device=device, dtype=dtype)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return gx, gy

def _gaussian_kernel_1d(k: int, sigma: float, device, dtype):
    assert k % 2 == 1
    x = torch.arange(k, device=device, dtype=dtype) - (k // 2)
    g = torch.exp(-(x**2) / (2 * sigma**2))
    return g / g.sum()

def _gaussian_blur(gray: torch.Tensor, k: int, sigma: float):
    # depthwise separable gaussian blur for [B,1,H,W] or [B,C,H,W]
    assert k % 2 == 1
    device, dtype = gray.device, gray.dtype
    g = _gaussian_kernel_1d(k, sigma, device, dtype)
    g1 = g.view(1, 1, 1, k)
    g2 = g.view(1, 1, k, 1)

    B, C, H, W = gray.shape
    w1 = g1.repeat(C, 1, 1, 1)
    w2 = g2.repeat(C, 1, 1, 1)

    x = F.pad(gray, (k//2, k//2, 0, 0), mode="reflect")
    x = F.conv2d(x, w1, groups=C)
    x = F.pad(x, (0, 0, k//2, k//2), mode="reflect")
    x = F.conv2d(x, w2, groups=C)
    return x

def _make_base_grid(B, H, W, device, dtype):
    # returns grid [B,H,W,2] in [-1,1]
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device, dtype=dtype),
        torch.linspace(-1, 1, W, device=device, dtype=dtype),
        indexing="ij",
    )
    grid = torch.stack([xx, yy], dim=-1)  # [H,W,2]
    return grid.unsqueeze(0).expand(B, -1, -1, -1).contiguous()

# ---------------------------
# Modules
# ---------------------------

class StructureTensorOrientation:
    """
    Computes local orientation theta and coherence (anisotropy strength) from structure tensor.
    """
    def __init__(self, sigma_grad: float = 1.0, sigma_tensor: float = 2.0, k: int = 7):
        self.sigma_grad = float(sigma_grad)
        self.sigma_tensor = float(sigma_tensor)
        self.k = int(k if k % 2 == 1 else k + 1)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor):
        x = _bchw(img01)
        g = _gray(x)

        # smooth a bit before gradients
        g_s = _gaussian_blur(g, k=self.k, sigma=self.sigma_grad)

        gx, gy = _sobel(g_s)

        # structure tensor components
        Jxx = gx * gx
        Jyy = gy * gy
        Jxy = gx * gy

        # smooth tensor
        Jxx = _gaussian_blur(Jxx, k=self.k, sigma=self.sigma_tensor)
        Jyy = _gaussian_blur(Jyy, k=self.k, sigma=self.sigma_tensor)
        Jxy = _gaussian_blur(Jxy, k=self.k, sigma=self.sigma_tensor)

        # orientation: theta = 0.5 * atan2(2Jxy, Jxx - Jyy)
        theta = 0.5 * torch.atan2(2.0 * Jxy, (Jxx - Jyy + 1e-8))  # [B,1,H,W]

        # coherence: (lambda1 - lambda2) / (lambda1 + lambda2)
        trace = Jxx + Jyy
        det = Jxx * Jyy - Jxy * Jxy
        tmp = torch.sqrt(torch.clamp(trace * trace - 4.0 * det, min=0.0))
        lam1 = 0.5 * (trace + tmp)
        lam2 = 0.5 * (trace - tmp)
        coherence = (lam1 - lam2) / (lam1 + lam2 + 1e-8)  # [B,1,H,W] in [0,1-ish]
        coherence = coherence.clamp(0.0, 1.0)

        return theta, coherence


class AnisotropicKuwaharaLite:
    """
    "Lite" anisotropic Kuwahara:
    - for each pixel, sample 4 oriented regions (forward/back along tangent, left/right across normal)
    - compute mean color + color variance proxy per region
    - choose region with minimal variance -> output its mean
    This creates brush-stroke smoothing aligned to orientation.

    Performance knobs:
    - radius: stroke length (in pixels)
    - step: number of samples per half-stroke (higher = smoother but slower)
    """
    def __init__(
        self,
        radius: int = 8,
        step: int = 4,
        sigma_color: float = 0.0,      # optional pre-blur
        tensor: StructureTensorOrientation | None = None,
        coherence_gain: float = 1.5,   # how strongly anisotropy follows coherence
        downsample: int = 1,           # set 2 for speed, then upsample
    ):
        self.radius = int(radius)
        self.step = int(step)
        self.sigma_color = float(sigma_color)
        self.tensor = tensor if tensor is not None else StructureTensorOrientation()
        self.coherence_gain = float(coherence_gain)
        self.downsample = int(downsample)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = _bchw(img01)
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        # optional pre-blur (helps reduce noise)
        if self.sigma_color > 0:
            k = 7
            x = _gaussian_blur(x, k=k, sigma=self.sigma_color)

        # optional downsample for speed
        if self.downsample > 1:
            ds = self.downsample
            x_small = F.interpolate(x, scale_factor=1.0/ds, mode="area")
            Hs, Ws = x_small.shape[-2:]
            theta, coh = self.tensor(_chw(x_small))  # expects [3,H,W]
            # tensor returns [1,1,Hs,Ws]
            theta = theta.to(device=device, dtype=dtype)
            coh = coh.to(device=device, dtype=dtype)
            base_grid = _make_base_grid(B, Hs, Ws, device, dtype)
        else:
            x_small = x
            theta, coh = self.tensor(img01)  # [1,1,H,W]
            theta = theta.to(device=device, dtype=dtype)
            coh = coh.to(device=device, dtype=dtype)
            base_grid = _make_base_grid(B, H, W, device, dtype)

        # compute tangent and normal unit vectors in grid coords
        # grid coords: x in [-1,1] corresponds to W, y corresponds to H
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        # tangent direction (along brush stroke)
        tx = cos_t
        ty = sin_t
        # normal direction (perpendicular)
        nx = -sin_t
        ny = cos_t

        # scale offsets from pixel units to normalized grid units
        if self.downsample > 1:
            Hc, Wc = x_small.shape[-2:]
        else:
            Hc, Wc = H, W
        sx = 2.0 / max(Wc - 1, 1)
        sy = 2.0 / max(Hc - 1, 1)

        # anisotropy scale: stronger along tangent when coherence is high
        # along_scale in [1, 1+gain], across_scale in [1, 1+gain] inverse-ish
        a = (1.0 + self.coherence_gain * coh)  # [B,1,H,W]
        along = a
        across = 1.0 / a.clamp(min=1e-3)

        # sample points along tangent and normal
        # offsets: positions in pixels -> normalized
        # we create a small set of offsets: [-r..-1] and [1..r] (no 0)
        r = self.radius
        s = max(self.step, 1)
        # sample distances: s points from 1..r
        d = torch.linspace(1.0, float(r), steps=s, device=device, dtype=dtype).view(1, 1, 1, 1, s)

        # build 4 directional grids: +t, -t, +n, -n
        # each grid: [B,H,W,2] per sample -> we'll sample and aggregate
        def sample_dir(dx, dy):
            # dx,dy are [B,1,H,W]
            # offsets [B,1,H,W,s]
            ox = dx.unsqueeze(-1) * d * sx
            oy = dy.unsqueeze(-1) * d * sy
            # base_grid [B,H,W,2] -> expand samples
            gx = base_grid[..., 0].unsqueeze(-1) + ox.squeeze(1)
            gy = base_grid[..., 1].unsqueeze(-1) + oy.squeeze(1)
            grid = torch.stack([gx, gy], dim=-1)  # [B,H,W,s,2]
            # reshape to sample all s in one call
            grid2 = grid.view(B, Hc, Wc * s, 2)
            samp = F.grid_sample(x_small, grid2, mode="bilinear", padding_mode="border", align_corners=True)
            samp = samp.view(B, C, Hc, Wc, s)
            return samp

        # apply anisotropy scaling
        tdx = tx * along
        tdy = ty * along
        ndx = nx * across
        ndy = ny * across

        samp_t_pos = sample_dir(tdx, tdy)    # [B,C,H,W,s]
        samp_t_neg = sample_dir(-tdx, -tdy)
        samp_n_pos = sample_dir(ndx, ndy)
        samp_n_neg = sample_dir(-ndx, -ndy)

        # include center pixel as well (helps stability)
        center = x_small.unsqueeze(-1)  # [B,C,H,W,1]

        # define 4 regions (each includes center + one direction)
        regions = [
            torch.cat([center, samp_t_pos], dim=-1),
            torch.cat([center, samp_t_neg], dim=-1),
            torch.cat([center, samp_n_pos], dim=-1),
            torch.cat([center, samp_n_neg], dim=-1),
        ]

        # compute mean + variance proxy per region
        means = []
        vars_ = []
        for reg in regions:
            m = reg.mean(dim=-1)  # [B,C,H,W]
            # variance proxy: mean squared deviation across channels/samples
            v = (reg - m.unsqueeze(-1)).pow(2).mean(dim=-1).mean(dim=1, keepdim=True)  # [B,1,H,W]
            means.append(m)
            vars_.append(v)

        # choose region with minimal variance
        var_stack = torch.cat(vars_, dim=1)  # [B,4,H,W]
        idx = torch.argmin(var_stack, dim=1, keepdim=True)  # [B,1,H,W]

        # gather means
        mean_stack = torch.stack(means, dim=1)  # [B,4,C,H,W]
        idx5 = idx.unsqueeze(2)  # [B,1,1,H,W]
        out = torch.gather(mean_stack, dim=1, index=idx5.expand(-1, -1, mean_stack.size(2), -1, -1)).squeeze(1)

        # upsample back if needed
        if self.downsample > 1:
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

        return _chw(out)


class OilPaintPolish:
    """
    Optional final touches:
    - mild contrast curve
    - optional edge darkening (subtle)
    """
    def __init__(self, contrast: float = 0.25, edge_strength: float = 0.15):
        self.contrast = float(contrast)
        self.edge_strength = float(edge_strength)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = _bchw(img01)
        # gentle contrast curve around midtones
        a = self.contrast
        y = x + a * (x - x * x) * (x - 0.5) * 4.0
        y = y.clamp(0, 1)

        if self.edge_strength > 0:
            g = _gray(y)
            gx, gy = _sobel(g)
            e = torch.sqrt(gx * gx + gy * gy + 1e-8)                 # [B,1,H,W]
            e = e / (e.amax(dim=(2, 3), keepdim=True) + 1e-8)         # normalize
            ink = 1.0 - self.edge_strength * e
            y = (y * ink).clamp(0, 1)

        return _chw(y)


class AnisoKuwaharaOilPipeline:
    """
    Reasonable default "oil painting" pipeline:
      (optional preblur in kuwahara) -> anisotropic kuwahara -> polish
    """
    def __init__(
        self,
        radius: int = 10,
        step: int = 5,
        sigma_color: float = 0.8,
        coherence_gain: float = 2.0,
        downsample: int = 1,       # 2 speeds up a lot; set 1 for best quality
        contrast: float = 0.25,
        edge_strength: float = 0.10,
    ):
        self.kuw = AnisotropicKuwaharaLite(
            radius=radius,
            step=step,
            sigma_color=sigma_color,
            coherence_gain=coherence_gain,
            downsample=downsample,
        )
        self.polish = OilPaintPolish(contrast=contrast, edge_strength=edge_strength)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = self.kuw(img01)
        x = self.polish(x)
        return x.clamp(0, 1)
