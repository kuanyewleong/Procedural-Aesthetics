import torch
import torch.nn.functional as F
import math

# ---------------------------
# Helpers
# ---------------------------

def _ensure_bchw(img01: torch.Tensor) -> torch.Tensor:
    assert img01.dim() == 3 and img01.shape[0] == 3, "expected [3,H,W]"
    return img01.unsqueeze(0)

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

def _rgb_to_hsv(bchw: torch.Tensor):
    # bchw [B,3,H,W] -> h,s,v each [B,1,H,W], h in [0,1]
    r, g, b = bchw[:, 0], bchw[:, 1], bchw[:, 2]
    maxc = torch.maximum(torch.maximum(r, g), b)
    minc = torch.minimum(torch.minimum(r, g), b)
    v = maxc
    delt = maxc - minc + 1e-8
    s = delt / (maxc + 1e-8)

    hr = (((g - b) / delt) % 6.0) * (maxc == r)
    hg = (((b - r) / delt) + 2.0) * (maxc == g)
    hb = (((r - g) / delt) + 4.0) * (maxc == b)
    h = (hr + hg + hb) / 6.0
    h = torch.where(delt < 1e-6, torch.zeros_like(h), h)

    return h.unsqueeze(1), s.unsqueeze(1), v.unsqueeze(1)

def _hsv_to_rgb(h, s, v):
    # h,s,v: [B,1,H,W] with h in [0,1]
    h6 = (h % 1.0) * 6.0
    c = v * s
    x = c * (1 - torch.abs((h6 % 2) - 1))
    m = v - c

    z = torch.zeros_like(h6)
    r = torch.zeros_like(h6)
    g = torch.zeros_like(h6)
    b = torch.zeros_like(h6)

    cond0 = (0 <= h6) & (h6 < 1)
    cond1 = (1 <= h6) & (h6 < 2)
    cond2 = (2 <= h6) & (h6 < 3)
    cond3 = (3 <= h6) & (h6 < 4)
    cond4 = (4 <= h6) & (h6 < 5)
    cond5 = (5 <= h6) & (h6 < 6)

    r = torch.where(cond0, c, r); g = torch.where(cond0, x, g); b = torch.where(cond0, z, b)
    r = torch.where(cond1, x, r); g = torch.where(cond1, c, g); b = torch.where(cond1, z, b)
    r = torch.where(cond2, z, r); g = torch.where(cond2, c, g); b = torch.where(cond2, x, b)
    r = torch.where(cond3, z, r); g = torch.where(cond3, x, g); b = torch.where(cond3, c, b)
    r = torch.where(cond4, x, r); g = torch.where(cond4, z, g); b = torch.where(cond4, c, b)
    r = torch.where(cond5, c, r); g = torch.where(cond5, z, g); b = torch.where(cond5, x, b)

    return torch.cat([r + m, g + m, b + m], dim=1)

def _randn_lowfreq(B, H, W, device, dtype, k=41):
    n = torch.randn(B, 1, H, W, device=device, dtype=dtype)
    n = _box_blur(n, k)
    n = n / (n.std(dim=(2,3), keepdim=True) + 1e-8)
    return n

def _complement_hue(h):
    # hue complement: rotate by 180 degrees on color wheel
    return (h + 0.5) % 1.0

# ---------------------------
# Fauvism Modules (v2)
# ---------------------------

class FauveSimplify:
    """
    Strong abstraction/simplification via repeated edge-preserving smoothing.
    """
    def __init__(self, blur_k=9, edge_strength=10.0, mix=0.9, passes=2):
        self.blur_k = int(blur_k)
        self.edge_strength = float(edge_strength)
        self.mix = float(mix)
        self.passes = int(passes)

    @torch.no_grad()
    def __call__(self, img01):
        x = _ensure_bchw(img01).clamp(0, 1)
        for _ in range(self.passes):
            blur = _box_blur(x, self.blur_k)
            e = _sobel_edges(_gray(x))
            w = torch.exp(-self.edge_strength * e).clamp(0, 1)
            w = w.repeat(1, 3, 1, 1)
            x = x * (1 - self.mix * w) + blur * (self.mix * w)
        return _back_to_chw(x)


class FauveEmotionColorMap:
    """
    Arbitrary/emotion-driven color mapping:
    - picks a global mood (or user-provided) that shifts hue/sat/value non-physically
    - also applies low-frequency spatial hue drift so skies/foliage/petals can become "wrong" colors
    """
    MOODS = {
        # hue_shift, sat_mul, val_mul
        "joy":      ( 0.08, 2.10, 1.05),
        "anger":    ( 0.00, 2.30, 1.00),
        "melancholy":(0.55, 1.60, 0.95),
        "dream":    ( 0.18, 1.90, 1.03),
        "wild":     ( 0.33, 2.40, 1.00),
        "calm":     ( 0.10, 1.50, 1.02),
    }

    def __init__(self, mood="wild", drift=0.10, drift_k=61, local_jitter=0.03):
        self.mood = mood
        self.drift = float(drift)
        self.drift_k = int(drift_k)
        self.local_jitter = float(local_jitter)

    @torch.no_grad()
    def __call__(self, img01):
        x = _ensure_bchw(img01).clamp(0, 1)
        h, s, v = _rgb_to_hsv(x)
        B, _, H, W = h.shape
        device, dtype = x.device, x.dtype

        hs, sm, vm = self.MOODS.get(self.mood, self.MOODS["wild"])

        # global non-physical hue shift
        h = (h + hs) % 1.0

        # low-frequency "emotion drift" across canvas (arbitrary mapping)
        drift = _randn_lowfreq(B, H, W, device, dtype, k=self.drift_k)
        h = (h + self.drift * drift) % 1.0

        # a bit of local hue jitter to break realism
        jit = _randn_lowfreq(B, H, W, device, dtype, k=21)
        h = (h + self.local_jitter * jit) % 1.0

        # strident saturation
        s = torch.clamp(s * sm, 0, 1)

        # slight value lift/crush
        v = torch.clamp(v * vm, 0, 1)

        out = _hsv_to_rgb(h, s, v)
        return _back_to_chw(out)


class FauveComplementVibration:
    """
    Complementary Color Theory injection:
    - compute dominant hue field
    - create a region mask and push adjacent regions toward complementary hues
    - emphasizes orange/blue, red/cyan, green/magenta contrasts
    """
    def __init__(self, region_k=31, strength=0.45, edge_boost=0.25):
        self.region_k = int(region_k)
        self.strength = float(strength)
        self.edge_boost = float(edge_boost)

    @torch.no_grad()
    def __call__(self, img01):
        x = _ensure_bchw(img01).clamp(0, 1)
        h, s, v = _rgb_to_hsv(x)

        # build a coarse region map from smoothed hue and value
        h_s = _box_blur(h, self.region_k)
        v_s = _box_blur(v, self.region_k)

        # region score: values in [0,1] that alternate areas (like soft segmentation)
        # use a sinusoid over hue + value for "patchy regions" (gives painterly blocks)
        region = 0.5 + 0.5 * torch.sin(2 * math.pi * (h_s * 1.0 + v_s * 0.35))
        region = region.clamp(0, 1)

        # compute complement hue
        h_comp = _complement_hue(h)

        # push some regions toward complement
        # region acts like mask: 0 keeps original, 1 moves to complement
        mask = region

        # boost mask near edges to make complement adjacency more visible
        edges = _sobel_edges(_gray(x))
        edges = edges / (edges.amax(dim=(2,3), keepdim=True) + 1e-8)
        mask = torch.clamp(mask + self.edge_boost * edges, 0, 1)

        # blend hue toward complement hue (circular blend)
        # compute shortest direction on hue circle
        dh = h_comp - h
        dh = (dh + 0.5) % 1.0 - 0.5  # wrap to [-0.5,0.5]
        h_new = (h + self.strength * mask * dh) % 1.0

        # increase saturation a bit in complemented zones for vibration
        s_new = torch.clamp(s * (1.0 + 0.35 * mask), 0, 1)

        out = _hsv_to_rgb(h_new, s_new, v)
        return _back_to_chw(out)


class FauveBrushStrokes:
    """
    Wild brush work by multi-directional smear strokes.
    """
    def __init__(self, n_dirs=6, stroke_len=6, stroke_strength=0.65):
        self.n_dirs = int(n_dirs)
        self.stroke_len = int(stroke_len)
        self.stroke_strength = float(stroke_strength)

    @torch.no_grad()
    def __call__(self, img01):
        x = _ensure_bchw(img01).clamp(0, 1)
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        # deterministic-ish set of directions (not random every call)
        angles = torch.linspace(0, math.pi, steps=self.n_dirs, device=device, dtype=dtype)
        dirs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)  # [n,2]

        out = x
        for i in range(self.n_dirs):
            dx = float(dirs[i, 0].item())
            dy = float(dirs[i, 1].item())

            acc = x
            for k in range(1, self.stroke_len + 1):
                sx = int(round(dx * k))
                sy = int(round(dy * k))
                shifted = torch.roll(x, shifts=(sy, sx), dims=(2, 3))
                acc = acc + shifted
            acc = acc / (self.stroke_len + 1)

            out = out * (1 - self.stroke_strength / self.n_dirs) + acc * (self.stroke_strength / self.n_dirs)

        return _back_to_chw(out)


class FauveImpastoTexture:
    """
    Subtle impasto/painterly texture:
    - low-frequency noise modulates brightness and slight chroma.
    """
    def __init__(self, strength=0.10, noise_k=41, chroma=0.04):
        self.strength = float(strength)
        self.noise_k = int(noise_k)
        self.chroma = float(chroma)

    @torch.no_grad()
    def __call__(self, img01):
        x = _ensure_bchw(img01).clamp(0, 1)
        B, _, H, W = x.shape
        n = _randn_lowfreq(B, H, W, x.device, x.dtype, k=self.noise_k)
        x = torch.clamp(x + self.strength * n.repeat(1, 3, 1, 1), 0, 1)

        if self.chroma > 0:
            cj = (torch.randn(B, 3, 1, 1, device=x.device, dtype=x.dtype)) * self.chroma
            x = torch.clamp(x + cj, 0, 1)

        return _back_to_chw(x)


class FauveContours:
    """
    Bold expressive contours (optional).
    """
    def __init__(self, strength=0.55, threshold=0.10, blur_k=3):
        self.strength = float(strength)
        self.threshold = float(threshold)
        self.blur_k = int(blur_k)

    @torch.no_grad()
    def __call__(self, img01):
        x = _ensure_bchw(img01).clamp(0, 1)
        e = _sobel_edges(_gray(x))
        e = e / (e.amax(dim=(2,3), keepdim=True) + 1e-8)
        if self.blur_k > 1:
            e = _box_blur(e, self.blur_k)

        mask = (e - self.threshold).clamp(min=0.0)
        mask = mask / (mask.amax(dim=(2,3), keepdim=True) + 1e-8)
        mask = mask.clamp(0, 1)

        ink = 1.0 - self.strength * mask
        out = x * ink
        return _back_to_chw(out)


class FauvismPipeline:
    """
    Fauvism pipeline with explicit:
      - emotion-driven arbitrary color mapping
      - complementary vibration enforcement
      - simplified forms + wild brush strokes
    """
    def __init__(
        self,
        mood="wild",
        simplify=True,
        contours=True,
        # simplify
        simplify_k=9,
        simplify_passes=2,
        simplify_mix=0.9,
        simplify_edge=10.0,
        # emotion color map
        drift=0.10,
        drift_k=61,
        # complement vibration
        comp_region_k=31,
        comp_strength=0.45,
        # brushwork
        n_dirs=6,
        stroke_len=6,
        stroke_strength=0.65,
        # texture
        impasto_strength=0.10,
    ):
        self.do_simplify = bool(simplify)
        self.do_contours = bool(contours)

        self.simplify = FauveSimplify(
            blur_k=simplify_k,
            edge_strength=simplify_edge,
            mix=simplify_mix,
            passes=simplify_passes,
        )
        self.emotion = FauveEmotionColorMap(
            mood=mood,
            drift=drift,
            drift_k=drift_k,
        )
        self.complement = FauveComplementVibration(
            region_k=comp_region_k,
            strength=comp_strength,
            edge_boost=0.25,
        )
        self.brush = FauveBrushStrokes(
            n_dirs=n_dirs,
            stroke_len=stroke_len,
            stroke_strength=stroke_strength,
        )
        self.impasto = FauveImpastoTexture(strength=impasto_strength)
        self.contours = FauveContours(strength=0.55)

    @torch.no_grad()
    def __call__(self, img01):
        x = img01
        if self.do_simplify:
            x = self.simplify(x)
        x = self.emotion(x)
        x = self.complement(x)
        x = self.brush(x)
        x = self.impasto(x)
        if self.do_contours:
            x = self.contours(x)
        return x.clamp(0, 1)
