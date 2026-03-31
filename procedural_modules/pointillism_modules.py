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

def _make_disk_kernel(radius: int, softness: float = 1.5, device=None, dtype=None):
    """
    Soft disk kernel for dot stamping.
    radius: dot radius in pixels
    softness: larger -> softer edges (Gaussian-like)
    returns kernel [1,1,k,k] with max 1.
    """
    r = int(radius)
    k = 2 * r + 1
    yy, xx = torch.meshgrid(
        torch.arange(k, device=device, dtype=dtype),
        torch.arange(k, device=device, dtype=dtype),
        indexing="ij"
    )
    cy = r
    cx = r
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    # soft disk: exp falloff
    kern = torch.exp(-dist2 / (2.0 * (softness ** 2) + 1e-8))
    kern = kern / (kern.max() + 1e-8)
    return kern.view(1, 1, k, k)

# ---------------------------
# Modules
# ---------------------------

class EdgePreservingFlatten:
    """
    Flattens regions while keeping edges (helps pointillism read as paint).
    """
    def __init__(self, blur_k: int = 7, edge_strength: float = 10.0, mix: float = 0.7):
        self.blur_k = blur_k
        self.edge_strength = edge_strength
        self.mix = mix

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = _ensure_bchw(img01)
        blur = _box_blur(x, self.blur_k)
        e = _sobel_edges(_gray(x))
        w = torch.exp(-self.edge_strength * e).clamp(0.0, 1.0)  # low edge -> more smoothing
        w = w.repeat(1, 3, 1, 1)
        out = x * (1 - self.mix * w) + blur * (self.mix * w)
        return _back_to_chw(out)

class ContrastCurve:
    """
    Slight contrast shaping to make dots pop.
    """
    def __init__(self, amount: float = 0.25):
        self.amount = float(amount)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = img01.clamp(0, 1)
        a = self.amount
        out = x + a * (x - x * x) * (x - 0.5) * 4.0
        return out.clamp(0, 1)

class PointillizeLayer:
    """
    One dot layer:
    - sample colors at jittered grid points
    - stamp soft disks at those points
    - returns a painted canvas [3,H,W] in [0,1]
    """
    def __init__(
        self,
        spacing: int = 8,          # grid spacing in pixels (smaller = denser)
        radius: int = 3,           # dot radius in pixels
        jitter: float = 0.7,       # jitter fraction of spacing (0..1)
        softness: float = 1.6,     # dot softness
        coverage: float = 1.0,     # 0..1 fraction of grid points used
        opacity: float = 1.0,      # layer opacity
        seed: int = 0
    ):
        self.spacing = int(spacing)
        self.radius = int(radius)
        self.jitter = float(jitter)
        self.softness = float(softness)
        self.coverage = float(coverage)
        self.opacity = float(opacity)
        self.seed = int(seed)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = _ensure_bchw(img01)  # [1,3,H,W]
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype

        # build grid points
        s = self.spacing
        ys = torch.arange(0, H, s, device=device, dtype=dtype)
        xs = torch.arange(0, W, s, device=device, dtype=dtype)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        pts = torch.stack([gy, gx], dim=-1).view(-1, 2)  # [N,2], (y,x)

        # random subset for coverage
        if self.coverage < 1.0:
            g = torch.Generator(device=device)
            g.manual_seed(self.seed)
            keep = torch.rand((pts.shape[0],), generator=g, device=device) < self.coverage
            pts = pts[keep]

        # jitter points
        g = torch.Generator(device=device)
        g.manual_seed(self.seed + 12345)
        j = (torch.rand((pts.shape[0], 2), generator=g, device=device, dtype=dtype) * 2 - 1)
        j = j * (self.jitter * s)
        pts_j = pts + j
        pts_j[:, 0] = pts_j[:, 0].clamp(0, H - 1)
        pts_j[:, 1] = pts_j[:, 1].clamp(0, W - 1)

        # sample colors from image at jittered points (nearest)
        yi = pts_j[:, 0].round().long()
        xi = pts_j[:, 1].round().long()
        colors = x[0, :, yi, xi].t().contiguous()  # [N,3]

        # dot kernel
        k = _make_disk_kernel(self.radius, softness=self.softness, device=device, dtype=dtype)  # [1,1,kh,kw]
        kh = k.shape[-2]
        kw = k.shape[-1]
        pad_y = kh // 2
        pad_x = kw // 2

        # canvases accumulate weighted color and weights
        canvas = torch.zeros((1, 3, H, W), device=device, dtype=dtype)
        weight = torch.zeros((1, 1, H, W), device=device, dtype=dtype)

        # stamp dots (loop over points; GPU ok for moderate N; use fewer points if huge images)
        # N ~ (H/s)*(W/s). For 512 and spacing=8 -> (64*64)=4096 points per layer.
        for idx in range(pts_j.shape[0]):
            y0 = int(yi[idx].item())
            x0 = int(xi[idx].item())

            y1 = y0 - pad_y
            y2 = y0 + pad_y + 1
            x1 = x0 - pad_x
            x2 = x0 + pad_x + 1

            ky1 = 0
            ky2 = kh
            kx1 = 0
            kx2 = kw

            # clip to image bounds
            if y1 < 0:
                ky1 = -y1
                y1 = 0
            if x1 < 0:
                kx1 = -x1
                x1 = 0
            if y2 > H:
                ky2 = kh - (y2 - H)
                y2 = H
            if x2 > W:
                kx2 = kw - (x2 - W)
                x2 = W

            kk = k[:, :, ky1:ky2, kx1:kx2]  # [1,1,*,*]
            col = colors[idx].view(1, 3, 1, 1)  # [1,3,1,1]
            canvas[:, :, y1:y2, x1:x2] += col * kk
            weight[:, :, y1:y2, x1:x2] += kk

        out = canvas / (weight + 1e-6)
        
        # White background
        bg = torch.ones_like(out)

        # Normalize paint coverage map (a in [0,1])
        a = (weight / (weight.amax(dim=(2, 3), keepdim=True) + 1e-8)).clamp(0, 1)

        # Force some paint everywhere (prevents all-white collapse)
        min_paint = 0.25   # try 0.20–0.40
        a = min_paint + (1.0 - min_paint) * a

        out = bg * (1 - a) + out * a

        # layer opacity
        out = (1 - self.opacity) * img01.unsqueeze(0) + self.opacity * out
        return _back_to_chw(out)

class LightInkEdges:
    """
    Optional: lightly darken edges so pointillism stays readable.
    """
    def __init__(self, strength: float = 0.25, threshold: float = 0.15, blur_k: int = 3):
        self.strength = float(strength)
        self.threshold = float(threshold)
        self.blur_k = int(blur_k)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = _ensure_bchw(img01)
        e = _sobel_edges(_gray(x))
        e = e / (e.amax(dim=(2, 3), keepdim=True) + 1e-8)
        if self.blur_k > 1:
            e = _box_blur(e, self.blur_k)
        mask = (e - self.threshold).clamp(min=0.0)
        mask = mask / (mask.amax(dim=(2, 3), keepdim=True) + 1e-8)
        ink = 1.0 - self.strength * mask
        out = x * ink
        return _back_to_chw(out)

class PointillismPipeline:
    """
    Strong pointillism:
      flatten -> contrast -> multi-scale dots -> (optional) light ink edges
    """
    def __init__(
        self,
        flatten: bool = True,
        flatten_k: int = 7,
        flatten_mix: float = 0.75,
        flatten_edge: float = 10.0,
        contrast: float = 0.25,
        # multi-scale dot layers (coarse -> fine)
        layers=(
            dict(spacing=10, radius=4, jitter=0.8, softness=1.8, coverage=0.85, opacity=1.0, seed=1),
            dict(spacing=7,  radius=3, jitter=0.8, softness=1.6, coverage=0.90, opacity=1.0, seed=2),
            dict(spacing=5,  radius=2, jitter=0.9, softness=1.4, coverage=0.95, opacity=1.0, seed=3),
        ),
        ink_edges: bool = True,
        ink_strength: float = 0.20,
        ink_threshold: float = 0.15,
    ):
        self.do_flatten = flatten
        self.flatten = EdgePreservingFlatten(blur_k=flatten_k, edge_strength=flatten_edge, mix=flatten_mix)
        self.contrast = ContrastCurve(amount=contrast)
        self.dot_layers = [PointillizeLayer(**cfg) for cfg in layers]
        self.do_ink = ink_edges
        self.ink = LightInkEdges(strength=ink_strength, threshold=ink_threshold, blur_k=3)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = img01.clamp(0, 1)
        if self.do_flatten:
            x = self.flatten(x)
        x = self.contrast(x)
        for layer in self.dot_layers:
            x = layer(x)
        if self.do_ink:
            x = self.ink(x)
        return x.clamp(0, 1)
