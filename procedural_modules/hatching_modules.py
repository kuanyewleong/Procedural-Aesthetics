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

def _normalize01(x: torch.Tensor, eps=1e-8) -> torch.Tensor:
    mn = x.amin(dim=(2,3), keepdim=True)
    mx = x.amax(dim=(2,3), keepdim=True)
    return (x - mn) / (mx - mn + eps)

def _gamma(x: torch.Tensor, g: float) -> torch.Tensor:
    return torch.clamp(x, 0, 1) ** g

# ---------------------------
# Core hatch generator
# ---------------------------

def _coord_grid(B, H, W, device, dtype):
    # returns X,Y in [0,1] with shape [B,1,H,W]
    yy = torch.linspace(0, 1, H, device=device, dtype=dtype).view(1,1,H,1).expand(B,1,H,W)
    xx = torch.linspace(0, 1, W, device=device, dtype=dtype).view(1,1,1,W).expand(B,1,H,W)
    return xx, yy

def _oriented_stripes(B, H, W, device, dtype, angle_deg: float, freq: float, phase: float = 0.0):
    """
    Sinusoidal stripes oriented by angle.
    Returns values in [0,1], where 1 ~ bright, 0 ~ dark stripe.
    """
    xx, yy = _coord_grid(B, H, W, device, dtype)
    th = torch.tensor(angle_deg * 3.14159265 / 180.0, device=device, dtype=dtype)
    # rotate coords
    u = xx * torch.cos(th) + yy * torch.sin(th)
    # stripes
    s = torch.cos(2 * 3.14159265 * (freq * u + phase))
    s = (s + 1.0) * 0.5
    return s

def _stripe_to_ink(stripe01: torch.Tensor, sharpness: float = 10.0):
    """
    Convert smooth stripes to ink lines: make dark lines thin and crisp.
    """
    # push values toward 0/1
    # lines correspond to low values
    x = stripe01
    x = torch.sigmoid((x - 0.5) * sharpness)
    return x  # ~0 on lines, ~1 background

# ---------------------------
# Modules
# ---------------------------

class ToneMap:
    """
    Convert RGB image to a smoothed tone map (darkness) in [0,1].
    darkness=1 means very dark region (needs dense hatching).
    """
    def __init__(self, blur_k: int = 9, gamma: float = 1.0):
        self.blur_k = int(blur_k)
        self.gamma = float(gamma)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = _ensure_bchw(img01)
        g = _gray(x)
        g = _box_blur(g, self.blur_k) if self.blur_k > 1 else g
        g = _normalize01(g)
        g = _gamma(g, self.gamma)
        darkness = 1.0 - g
        return darkness  # [1,1,H,W]


class HatchLayer:
    """
    Single-angle hatching controlled by darkness.
    Produces an 'ink mask' in [0,1] (1=paper, 0=ink).
    """
    def __init__(
        self,
        angle_deg: float = 45.0,
        freq: float = 55.0,         # stripes per unit (increase => denser lines)
        sharpness: float = 14.0,    # higher => crisper lines
        darkness_gain: float = 1.0, # how strongly tone modulates this layer
        threshold: float = 0.35,    # when this layer starts appearing (darkness > threshold)
        softness: float = 0.08,     # smooth thresholding
        phase: float = 0.0,
    ):
        self.angle_deg = float(angle_deg)
        self.freq = float(freq)
        self.sharpness = float(sharpness)
        self.darkness_gain = float(darkness_gain)
        self.threshold = float(threshold)
        self.softness = float(softness)
        self.phase = float(phase)

    @torch.no_grad()
    def __call__(self, darkness: torch.Tensor) -> torch.Tensor:
        # darkness: [B,1,H,W] in [0,1]
        B, _, H, W = darkness.shape
        device, dtype = darkness.device, darkness.dtype

        stripes = _oriented_stripes(B, H, W, device, dtype, self.angle_deg, self.freq, self.phase)
        ink_bg = _stripe_to_ink(stripes, sharpness=self.sharpness)  # [B,1,H,W] like, but currently [B,1,H,W]? -> stripes returns [B,1,H,W]
        if ink_bg.dim() == 3:
            ink_bg = ink_bg.unsqueeze(1)

        # gate by darkness: appear mostly where dark
        d = torch.clamp(darkness * self.darkness_gain, 0.0, 1.0)
        # smooth step around threshold
        gate = torch.clamp((d - self.threshold) / (self.softness + 1e-8), 0.0, 1.0)

        # combine: where gate=0 => no ink (paper=1); where gate=1 => use ink_bg
        layer = 1.0 - gate + gate * ink_bg
        return layer.clamp(0.0, 1.0)  # [B,1,H,W]


class CrossHatch:
    """
    Combine multiple HatchLayer angles into a cross-hatching mask.
    Output is ink mask in [0,1] (1=paper, 0=ink).
    """
    def __init__(self, layers):
        self.layers = layers

    @torch.no_grad()
    def __call__(self, darkness: torch.Tensor) -> torch.Tensor:
        ink_mask = torch.ones_like(darkness)
        for layer in self.layers:
            ink_mask = ink_mask * layer(darkness)
        return ink_mask.clamp(0.0, 1.0)


class InkEdges:
    """
    Add edge ink on top of hatching. Output ink mask in [0,1].
    """
    def __init__(self, strength: float = 0.55, threshold: float = 0.12, blur_k: int = 3):
        self.strength = float(strength)
        self.threshold = float(threshold)
        self.blur_k = int(blur_k)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = _ensure_bchw(img01)
        g = _gray(x)
        e = _sobel_edges(g)
        e = e / (e.amax(dim=(2,3), keepdim=True) + 1e-8)
        if self.blur_k > 1:
            e = _box_blur(e, self.blur_k)

        mask = (e - self.threshold).clamp(min=0.0)
        mask = mask / (mask.amax(dim=(2,3), keepdim=True) + 1e-8)
        mask = mask.clamp(0.0, 1.0)

        # ink where edges: 1->paper, 0->ink
        edge_ink = 1.0 - self.strength * mask
        return edge_ink.clamp(0.0, 1.0)


class PaperTexture:
    """
    Simple procedural paper grain using low-frequency noise (no external deps).
    """
    def __init__(self, strength: float = 0.08, blur_k: int = 31):
        self.strength = float(strength)
        self.blur_k = int(blur_k)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = _ensure_bchw(img01)
        B, _, H, W = x.shape
        n = torch.rand((B, 1, H, W), device=x.device, dtype=x.dtype)
        n = _box_blur(n, self.blur_k)
        n = _normalize01(n)
        # paper brightness modulation around 1
        paper = (1.0 - self.strength) + self.strength * n
        return paper.clamp(0.0, 1.0)  # [B,1,H,W]


class InkSketchPipeline:
    """
    Full ink sketch look:
      tone map -> multi-angle hatch -> optional edges -> paper -> composite to RGB
    Output is RGB in [0,1].
    """
    def __init__(
        self,
        tone_blur_k: int = 9,
        tone_gamma: float = 1.0,
        # hatch parameters
        freq: float = 55.0,
        sharpness: float = 14.0,
        # density thresholds for successive layers
        thresholds=(0.25, 0.40, 0.55),
        angles=(0.0, 45.0, -45.0, 90.0),
        edge_strength: float = 0.55,
        edge_threshold: float = 0.12,
        paper_strength: float = 0.08,
        use_edges: bool = True,
    ):
        self.tone = ToneMap(blur_k=tone_blur_k, gamma=tone_gamma)

        # build hatch layers with progressive activation
        # First layer appears earliest; later layers require darker regions
        layers = []
        # use first 3 thresholds for 3 layers; if 4 angles provided, last angle uses last threshold
        thr = list(thresholds)
        while len(thr) < len(angles):
            thr.append(thr[-1] + 0.10)

        for i, ang in enumerate(angles):
            layers.append(HatchLayer(
                angle_deg=ang,
                freq=freq,
                sharpness=sharpness,
                darkness_gain=1.0,
                threshold=thr[i],
                softness=0.08,
                phase=0.0
            ))
        self.hatch = CrossHatch(layers)

        self.edges = InkEdges(strength=edge_strength, threshold=edge_threshold, blur_k=3)
        self.paper = PaperTexture(strength=paper_strength, blur_k=31)
        self.use_edges = bool(use_edges)

    @torch.no_grad()
    def __call__(self, img01: torch.Tensor) -> torch.Tensor:
        x = img01.clamp(0, 1)
        xb = _ensure_bchw(x)

        darkness = self.tone(x)                 # [1,1,H,W]
        hatch_mask = self.hatch(darkness)       # [1,1,H,W] 1 paper / 0 ink

        if self.use_edges:
            edge_mask = self.edges(x)           # [1,1,H,W]
            ink_mask = hatch_mask * edge_mask
        else:
            ink_mask = hatch_mask

        paper = self.paper(x)                   # [1,1,H,W]

        # Composite: paper background with ink_mask controlling ink amount.
        # ink_mask=1 => paper, 0 => black ink
        out_gray = paper * ink_mask
        out_rgb = out_gray.repeat(1, 3, 1, 1)

        return _back_to_chw(out_rgb)
