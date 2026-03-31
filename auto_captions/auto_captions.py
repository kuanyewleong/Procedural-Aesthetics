import torch, random, torch.nn.functional as F

def rgb_to_hsv(x):
    r, g, b = x[:,0], x[:,1], x[:,2]
    maxc, _ = torch.max(x, dim=1); minc, _ = torch.min(x, dim=1)
    v = maxc; delt = maxc - minc + 1e-6; s = delt / (maxc + 1e-6)
    hr = (((g - b) / delt) % 6) * (maxc == r)
    hg = (((b - r) / delt) + 2) * (maxc == g)
    hb = (((r - g) / delt) + 4) * (maxc == b)
    h = (hr + hg + hb) / 6.0; h[delt < 1e-5] = 0.0
    return h.unsqueeze(1), s.unsqueeze(1), v.unsqueeze(1)

COLOR_BINS = [
    ("red",(350/360,10/360)),("orange",(10/360,40/360)),("yellow",(40/360,70/360)),
    ("green",(70/360,160/360)),("cyan",(160/360,190/360)),("blue",(190/360,260/360)),
    ("purple",(260/360,300/360)),("magenta",(300/360,350/360)),
]

def hue_to_name(hmean, smean, vmean):
    if smean < 0.18:
        if vmean > 0.75: return "white"
        if vmean < 0.25: return "black"
        return "gray"
    for name,(lo,hi) in COLOR_BINS:
        if lo <= hi and (hmean >= lo and hmean < hi): return name
        if lo > hi and (hmean >= lo or hmean < hi): return name
    return "red"

def dominant_colors(img01, k=3):
    x = img01.unsqueeze(0)
    h,s,v = rgb_to_hsv(x)
    x_small = F.interpolate(x, size=(96,96), mode="area")
    h_small, s_small, v_small = rgb_to_hsv(x_small)
    h_flat = h_small.flatten().mean().item()
    s_mean = s.flatten().mean().item()
    v_mean = v.flatten().mean().item()
    main = hue_to_name(h_flat, s_mean, v_mean)
    BINS=12
    hq = (h_small.flatten()*BINS).clamp(0,BINS-1).long()
    hist = torch.bincount(hq, minlength=BINS).float()
    topk = torch.topk(hist, k=min(k, BINS)).indices.tolist()
    names=[]; seen=set()
    for bi in topk:
        h_center = (bi+0.5)/BINS
        n = hue_to_name(h_center, s_mean, v_mean)
        if n not in seen: names.append(n); seen.add(n)
    if main not in names: names=[main]+names
    return names[:k]

def edge_density(img01):
    g = 0.2989*img01[0:1]+0.5870*img01[1:2]+0.1140*img01[2:3]
    kx = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]] , device=g.device, dtype=g.dtype)
    ky = torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]] , device=g.device, dtype=g.dtype)
    gx = F.conv2d(g.unsqueeze(0), kx, padding=1); gy = F.conv2d(g.unsqueeze(0), ky, padding=1)
    return (gx**2 + gy**2).sqrt().mean().item()

def vibe_from_sv(s_mean, v_mean):
    if s_mean < 0.25 and v_mean > 0.7: return "soft pastel"
    if s_mean > 0.55 and v_mean > 0.65: return "vibrant"
    if v_mean < 0.35: return "moody"
    return "natural"

PHOTO_TEMPLATES = [
    "a {shot} photo of {colors} {flower}{context}, {vibe}",
    "a {shot} botanical photograph of {colors} {flower}{context}, {vibe}",
    "a {shot} close-up of {colors} {flower}{context}, {vibe}"
]

WATERCOLOR_TEMPLATES = [
    "watercolor painting of {colors} {flower}{context}, soft transparent washes, {vibe} tones",
    "a botanical watercolor of {colors} {flower}{context}, gentle brush edges and paper texture, {vibe}",
    "watercolor artwork of {colors} {flower}{context}, delicate blooms and light granulation, {vibe}"
]

CONTEXT_PHRASES = [
    " in the wild", " in a garden", " with soft bokeh background",
    " on a stem", " amid green foliage", ""
]

def join_colors(words):
    if   len(words)==0: return "unknown"
    elif len(words)==1: return words[0]
    elif len(words)==2: return f"{words[0]} and {words[1]}"
    else: return f"{words[0]}, {words[1]} and {words[2]}"

def make_natural_caption(img01, flower_name=None, watercolor=False, rng=None):
    if flower_name is None:
        flower_name = "flower"
    if rng is None: rng = random
    x = img01.to(torch.float32).clamp(0,1)
    _, s, v = rgb_to_hsv(x.unsqueeze(0))
    s_mean, v_mean = s.mean().item(), v.mean().item()
    vibe = vibe_from_sv(s_mean, v_mean)
    colors = join_colors(dominant_colors(x, k=3))
    shot = "macro" if edge_density(x) > 0.12 else "standard"
    context = rng.choice(CONTEXT_PHRASES)
    tpl = rng.choice(WATERCOLOR_TEMPLATES if watercolor else PHOTO_TEMPLATES)
    return tpl.format(shot=shot, colors=colors, flower=flower_name, context=context, vibe=vibe)

def tech_caption_from_steps(base_caption, steps):
    if not steps: return base_caption + ", no watercolor modules applied"
    names = ", ".join([s[0] for s in steps]); desc = "; ".join([s[1] for s in steps])
    return base_caption + f", watercolor simulation applying modules {names}: {desc}"
