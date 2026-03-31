import os
import re
import copy
import torch
import copy
import torch.nn as nn
from torchvision.utils import save_image

from models_ae import KL_VAE
from text_cond_t5_dual import T5DualTextCond
from diffusers import UNet2DConditionModel
from diffusion_sd15 import CosineScheduler, p_sample_loop

# from hatching_modules import InkSketchPipeline
from data_helpers.data_flowers import Flowers102Clean


# -------------------------
# Conditioning modules (must match Stage-2 training)
# -------------------------

class _TokenGenBlock(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.attn = torch.nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = torch.nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_model, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, d_model),
        )
        self.to_ss1 = torch.nn.Linear(d_model, 2 * d_model)
        self.to_ss2 = torch.nn.Linear(d_model, 2 * d_model)

    def forward(self, x, ctx_vec):
        h = self.norm1(x)
        ss = self.to_ss1(ctx_vec).unsqueeze(1)
        scale, shift = ss.chunk(2, dim=-1)
        h = h * (1 + scale) + shift
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out

        h = self.norm2(x)
        ss = self.to_ss2(ctx_vec).unsqueeze(1)
        scale, shift = ss.chunk(2, dim=-1)
        h = h * (1 + scale) + shift
        x = x + self.mlp(h)
        return x


class VecToTokensTransformer(torch.nn.Module):
    def __init__(
        self,
        d_in: int,
        n_tokens: int = 77,
        d_model: int = 512,
        d_out: int = 768,
        n_layers: int = 3,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        p_uncond: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.p_uncond = p_uncond

        self.ctx_proj = torch.nn.Sequential(
            torch.nn.Linear(d_in, d_model),
            torch.nn.SiLU(),
            torch.nn.Linear(d_model, d_model),
        )
        self.token_queries = torch.nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)
        self.pos_emb = torch.nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)
        self.blocks = torch.nn.ModuleList([
            _TokenGenBlock(d_model=d_model, n_heads=n_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm_out = torch.nn.LayerNorm(d_model)
        self.out_proj = torch.nn.Linear(d_model, d_out)

    def forward(self, ctx):
        if ctx.dim() == 3:
            ctx = ctx.mean(dim=1)
        B = ctx.shape[0]
        c = self.ctx_proj(ctx)
        tok = self.token_queries[None, :, :].expand(B, -1, -1) + self.pos_emb[None, :, :]
        for blk in self.blocks:
            tok = blk(tok, c)
        tok = self.out_proj(self.norm_out(tok))
        return tok


class ClassEmbedder(torch.nn.Module):
    def __init__(self, num_classes: int, d_ctx: int):
        super().__init__()
        self.emb = torch.nn.Embedding(num_classes, d_ctx)

    def forward(self, ctx_vec, labels):
        if ctx_vec.dim() == 3:
            cls = self.emb(labels)[:, None, :]
        else:
            cls = self.emb(labels)
        return ctx_vec + cls


def build_small_sd_unet(sample_size: int):
    return UNet2DConditionModel(
        sample_size=sample_size,
        in_channels=4,
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(192, 384, 768, 768),
        down_block_types=("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=768,
        attention_head_dim=8,
    )


# -------------------------
# Prompt -> label auto-match (optional)
# -------------------------

def build_name_to_label(names):
    def norm(s):
        s = s.lower().replace("_", " ").strip()
        s = re.sub(r"\s+", " ", s)
        return s
    return {norm(n): i for i, n in enumerate(names)}

def prompt_to_label_id(prompt: str, name_to_label: dict):
    p = prompt.lower().replace("_", " ")
    p = re.sub(r"\s+", " ", p)
    best, best_len = None, 0
    for name, lid in name_to_label.items():
        if name and name in p and len(name) > best_len:
            best, best_len = lid, len(name)
    return best


# -------------------------
# NEW: Load multiple UNets and block-wise mix them
# -------------------------

def _load_stage2_bundle(ckpt_path: str, device: str, image_size: int):
    """
    Loads:
      - unet (EMA if available)
      - ctx_adapter
      - class_embed (if present)
      - latent_size from checkpoint
    Returns dict with modules + latent_size.
    """
    ck = torch.load(ckpt_path, map_location=device)
    latent_size = int(ck.get("latent_size", image_size // 8))

    unet = build_small_sd_unet(sample_size=latent_size).to(device).eval()
    unet.load_state_dict(ck["ema"] if "ema" in ck else ck["unet"], strict=True)

    text = T5DualTextCond(device=device)

    ctx_adapter = VecToTokensTransformer(
        d_in=text.d_ctx, n_tokens=77, d_model=512, d_out=768, n_layers=3, n_heads=8, p_uncond=0.0
    ).to(device).eval()
    ctx_adapter.load_state_dict(ck["ctx_adapter"], strict=True)

    class_embed = ClassEmbedder(num_classes=102, d_ctx=text.d_ctx).to(device).eval()
    if "class_embed" in ck:
        class_embed.load_state_dict(ck["class_embed"], strict=True)

    # freeze
    for m in (unet, ctx_adapter, class_embed):
        for p in m.parameters():
            p.requires_grad_(False)

    return {
        "ck": ck,
        "latent_size": latent_size,
        "unet": unet,
        "ctx_adapter": ctx_adapter,
        "class_embed": class_embed,
        "text": text,  # shared interface; we will use the first bundle's text
    }

# soft blending wrapper
def _normalize_weights(w):
    w = torch.tensor(w, dtype=torch.float32)
    w = w / (w.sum() + 1e-12)
    return w.tolist()

def _blend_tensors(ts, ws):
    out = None
    for t, w in zip(ts, ws):
        out = t.mul(w) if out is None else out.add(t, alpha=w)
    return out

def _blend_res_samples(res_list, ws):
    # res_list: list of tuples, each tuple has same length
    K = len(res_list[0])
    blended = []
    for k in range(K):
        blended.append(_blend_tensors([r[k] for r in res_list], ws))
    return tuple(blended)

class SoftBlendDownBlock(nn.Module):
    def __init__(self, blocks, weights):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.weights = _normalize_weights(weights)

        proto = blocks[0]
        self.has_cross_attention = getattr(proto, "has_cross_attention", False)
        self.resnets = getattr(proto, "resnets", None)
        self.attentions = getattr(proto, "attentions", None)
        self.downsamplers = getattr(proto, "downsamplers", None)
        self.gradient_checkpointing = getattr(proto, "gradient_checkpointing", False)

    def forward(self, *args, **kwargs):
        outs = [blk(*args, **kwargs) for blk in self.blocks]
        if not (isinstance(outs[0], tuple) and len(outs[0]) == 2):
            raise RuntimeError(f"Down block output unexpected type: {type(outs[0])}")
        hs = _blend_tensors([o[0] for o in outs], self.weights)
        res = _blend_res_samples([o[1] for o in outs], self.weights)
        return hs, res

class SoftBlendMidBlock(nn.Module):
    def __init__(self, blocks, weights):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.weights = _normalize_weights(weights)

        proto = blocks[0]
        self.has_cross_attention = getattr(proto, "has_cross_attention", False)
        self.resnets = getattr(proto, "resnets", None)
        self.attentions = getattr(proto, "attentions", None)
        self.gradient_checkpointing = getattr(proto, "gradient_checkpointing", False)

    def forward(self, *args, **kwargs):
        outs = [blk(*args, **kwargs) for blk in self.blocks]
        if isinstance(outs[0], tuple):
            outs = [o[0] for o in outs]
        return _blend_tensors(outs, self.weights)


class SoftBlendUpBlock(nn.Module):
    def __init__(self, blocks, weights):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.weights = _normalize_weights(weights)

        # Preserve UNet block interface expected by diffusers forward()
        proto = blocks[0]
        self.has_cross_attention = getattr(proto, "has_cross_attention", False)
        self.resnets = getattr(proto, "resnets", None)
        self.attentions = getattr(proto, "attentions", None)
        self.upsamplers = getattr(proto, "upsamplers", None)
        self.gradient_checkpointing = getattr(proto, "gradient_checkpointing", False)

    def forward(self, *args, **kwargs):
        outs = [blk(*args, **kwargs) for blk in self.blocks]
        if isinstance(outs[0], tuple):
            outs = [o[0] for o in outs]
        return _blend_tensors(outs, self.weights)


# Build a new UNet where each block is a soft blend of the corresponding blocks from multiple UNets
def build_blockwise_ensemble_unet(
    unets,
    mix_plan=None,
    mix_mode="round_robin",
    ensemble_mode="hard",          # "hard" or "soft"
    soft_weights=None,             # list of floats, len == len(unets)
    soft_where="all",              # "all" or "attn_only"
):
    import copy
    assert len(unets) >= 1
    n_models = len(unets)

    base = copy.deepcopy(unets[0]).eval()
    n_down = len(base.down_blocks)
    n_up = len(base.up_blocks)

    if soft_weights is None:
        soft_weights = [1.0 / n_models] * n_models
    if len(soft_weights) != n_models:
        raise ValueError(f"soft_weights must have len {n_models}, got {len(soft_weights)}")

    # default hard plan
    if mix_plan is None:
        if mix_mode == "first":
            mix_plan = {"down": [0]*n_down, "mid": 0, "up": [0]*n_up}
        elif mix_mode == "round_robin":
            mix_plan = {
                "down": [(i % n_models) for i in range(n_down)],
                "mid": (n_models - 1),
                "up":  [((i + 1) % n_models) for i in range(n_up)],
            }
        else:
            raise ValueError(f"Unknown mix_mode: {mix_mode}")

    def _is_cross_attn_block(b):
        return "crossattn" in b.__class__.__name__.lower()

    # --- down blocks
    for i in range(n_down):
        if ensemble_mode == "soft" and (soft_where == "all" or _is_cross_attn_block(base.down_blocks[i])):
            base.down_blocks[i] = SoftBlendDownBlock([u.down_blocks[i] for u in unets], soft_weights)
        else:
            base.down_blocks[i] = unets[mix_plan["down"][i]].down_blocks[i]

    # --- mid block
    if ensemble_mode == "soft":
        base.mid_block = SoftBlendMidBlock([u.mid_block for u in unets], soft_weights)
    else:
        base.mid_block = unets[mix_plan["mid"]].mid_block

    # --- up blocks
    for i in range(n_up):
        if ensemble_mode == "soft" and (soft_where == "all" or _is_cross_attn_block(base.up_blocks[i])):
            base.up_blocks[i] = SoftBlendUpBlock([u.up_blocks[i] for u in unets], soft_weights)
        else:
            base.up_blocks[i] = unets[mix_plan["up"][i]].up_blocks[i]

    return base


# -------------------------
# Inference
# -------------------------

@torch.no_grad()
def main(
    prompt="a morning glory in the wild, new creative style",
    ckpt_stage2_list=(
        "ckpts/unet_stage2_hatching_step5000.pt",
        "ckpts/unet_stage2_poster.pt",
    ),
    vae_ckpt="ckpts/vae_512_lpips.pt",
    out_png="samples/stage2_ensemble_mix.png",
    image_size=512,
    timesteps=1000,
    sample_steps=250,
    guidance_scale=2.5,
    use_ddim=True,
    eta=0.0,
    dyn_thresh=False,
    dyn_p=0.999,
    seed=0,
    auto_label=True,
    mix_mode="round_robin",
    mix_plan=None,
    ensemble_mode="hard",   # "hard" or "soft"
    soft_weights=None,      # e.g. [0.2, 0.5, 0.3]
    soft_where="all",       # "all" or "attn_only"
):
    os.makedirs("samples", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator(device=device).manual_seed(seed)

    # ---- load VAE ----
    ck_vae = torch.load(vae_ckpt, map_location=device)
    if isinstance(ck_vae, dict) and "vae" in ck_vae:
        base_ch = ck_vae.get("base", 96)
        z_ch = ck_vae.get("z_ch", 4)
        vae = KL_VAE(z_ch=z_ch, base=base_ch).to(device).eval()
        vae.load_state_dict(ck_vae.get("ema", ck_vae["vae"]), strict=True)
    else:
        vae = KL_VAE(z_ch=4, base=96).to(device).eval()
        vae.load_state_dict(ck_vae, strict=True)
    for p in vae.parameters():
        p.requires_grad_(False)
    vae.eval()

    # ---- load multiple Stage-2 bundles ----
    bundles = [_load_stage2_bundle(p, device=device, image_size=image_size) for p in ckpt_stage2_list]

    # sanity: all latent sizes should match for clean mixing
    latent_sizes = [b["latent_size"] for b in bundles]
    if len(set(latent_sizes)) != 1:
        raise ValueError(f"All stage-2 checkpoints must have same latent_size. Got: {latent_sizes}")
    latent_size = latent_sizes[0]

    # choose text/ctx/class modules from the FIRST bundle (must be compatible across your training)
    text = bundles[0]["text"]
    ctx_adapter = bundles[0]["ctx_adapter"]
    class_embed = bundles[0]["class_embed"]

    # build mixed UNet from all UNets
    unets = [b["unet"] for b in bundles]
    unet = build_blockwise_ensemble_unet(
                unets,
                mix_plan=mix_plan,
                mix_mode=mix_mode,
                ensemble_mode=ensemble_mode,
                soft_weights=soft_weights,
                soft_where=soft_where,
            ).to(device).eval()

    # ---- auto label from prompt (optional) ----
    label_id = None
    if auto_label:
        ds_tmp = Flowers102Clean(split="train", image_size=image_size)
        name_to_label = build_name_to_label(ds_tmp.names)
        label_id = prompt_to_label_id(prompt, name_to_label)
        if label_id is not None:
            print(f"Auto species label: {label_id} ({ds_tmp.names[label_id]})")
        else:
            print("Auto species label: None (prompt-only)")

    # ---- build conditional/unconditional contexts ----
    ctx_cond_raw = text.encode_dual([prompt], None, w_nat=1.0, w_tech=0.0).to(device)
    if label_id is not None:
        labels = torch.tensor([label_id], device=device, dtype=torch.long)
        ctx_cond_raw = class_embed(ctx_cond_raw, labels)

    ctx_uncond_raw = text.encode_dual([""], None, w_nat=1.0, w_tech=0.0).to(device)
    # IMPORTANT: do NOT add class_embed to uncond

    ctx_cond = ctx_adapter(ctx_cond_raw)
    ctx_uncond = ctx_adapter(ctx_uncond_raw)

    # ---- v-pred -> eps wrapper for sampler ----
    sched = CosineScheduler(timesteps=timesteps).to(device)

    def model_eps(x, tt, ctx_tokens):
        v = unet(x, tt, encoder_hidden_states=ctx_tokens).sample
        abar = sched.alpha_bar[tt].view(-1, 1, 1, 1)
        eps = abar.sqrt() * v + (1 - abar).sqrt() * x
        return eps

    # ---- sample ----
    zshape = (1, 4, latent_size, latent_size)
    z = p_sample_loop(
        model=model_eps,
        sched=sched,
        shape=zshape,
        ctx_cond=ctx_cond,
        ctx_uncond=ctx_uncond,
        device=device,
        guidance_scale=guidance_scale,
        sample_steps=sample_steps,
        use_ddim=use_ddim,
        eta=eta,
        dyn_thresh=dyn_thresh,
        dyn_p=dyn_p
    )

    x = vae.decode(z).clamp(-1, 1)
    save_image((x + 1) / 2, out_png)
    print("Saved:", out_png)
    print("z stats:", float(z.mean()), float(z.std(unbiased=False)), float(z.min()), float(z.max()))
    print("x stats:", float(x.mean()), float(x.std(unbiased=False)), float(x.min()), float(x.max()))

    # Optional: show what the procedural posterizer would do (side-by-side debug)
    # art_style = InkSketchPipeline()
    # x01 = ((x + 1) / 2).clamp(0, 1).squeeze(0)
    # x_post = art_style(x01).unsqueeze(0)
    # out_png2 = out_png.replace(".png", "_postproc.png")
    # save_image(x_post, out_png2)
    # print("Saved (postproc preview):", out_png2)

if __name__ == "__main__":
    # Example 1: soft-blend
    # main(
    #     prompt="a hibiscus in the wild painted with a new creative hybrid art style",
    #     ckpt_stage2_list=(
    #         "ckpts/unet_stage2_halftone_step10000.pt",
    #         "ckpts/unet_stage2_poster.pt",
    #         "ckpts/unet_stage2_felt.pt",
    #     ),
    #     out_png="samples/ensemble_soft.png",
    #     ensemble_mode="soft",
    #     soft_weights=[0.25, 0.30, 0.45], # soft_weights=[0.20, 0.20, 0.60],
    #     soft_where="attn_only",
    # )
    # flower_name = "lotus"
    # Example 2: explicit block plan (more controllable “style grafting”)
    # main(
    #         prompt=f"a {flower_name} in the wild, painted with a new creative hybrid art style",
    #         ckpt_stage2_list=(
    #             "ckpts/lowpoly/unet_stage2_lowpoly_step55000.pt",
    #             "ckpts/mosaic/unet_stage2_mosaic_step20000.pt",
    #             "ckpts/pointillism/unet_stage2_pointillism_step60000.pt",
    #         ),
    #         out_png=f"samples/emerging_style/mix_plan_lowpoly_mosaic_pointillism/{flower_name}.png",
    #         mix_plan={
    #             "down": [1, 0, 1, 2],
    #             "mid": 2,
    #             "up": [1, 1, 0, 1],
    #         },
    #         sample_steps=600
    #     )


    # loop through all names in json and generate samples 
    # using mix-plan
    # import json
    # with open('data/flowers-102/cat_to_name.json', 'r') as f:
    #     cat_to_name = json.load(f)
    
    # total_flowers = len(cat_to_name)
    # for idx, (flower_id, flower_name) in enumerate(cat_to_name.items(), 1):
    #     print(f"Processing {idx}/{total_flowers}: {flower_name}")
        
    #     main(
    #         prompt=f"a {flower_name} in the wild, painted with a new creative hybrid art style",
    #         ckpt_stage2_list=(
    #             "ckpts/oil/unet_stage2_oil_step50000.pt",
    #             "ckpts/unet_stage2_poster.pt",
    #             "ckpts/unet_stage2_felt.pt",
    #         ),
    #         out_png=f"samples/emerging_style/mix_plan_step250/{flower_name}.png",
    #         mix_plan={
    #             "down": [0, 1, 0, 2],
    #             "mid": 2,
    #             "up": [1, 2, 0, 1],
    #         },
    #     )

    # using soft-blend
    import json
    with open('data/flowers-102/cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    total_flowers = len(cat_to_name)
    for idx, (flower_id, flower_name) in enumerate(cat_to_name.items(), 1):
        print(f"Processing {idx}/{total_flowers}: {flower_name}")

        # main(
        #     prompt=f"a {flower_name} in the wild, painted with a new creative hybrid art style",
        #     ckpt_stage2_list=(                
        #         "ckpts/fauvism/unet_stage2_fauvism.pt",
        #         "ckpts/lining/unet_stage2_lining_step75000.pt",
        #         "ckpts/pointillism/unet_stage2_pointillism_step60000.pt",
        #     ),
        #     out_png=f"samples/emerging_style/fauvism-10_lining-40_pointillism-50/{flower_name}.png",            
        #     ensemble_mode="soft",
        #     soft_weights=[0.10, 0.40, 0.50],
        #     soft_where="attn_only",
        #     dyn_thresh=False,
        #     sample_steps=250
        # )
        
        
        # main(
        #     prompt=f"a {flower_name} in the wild, painted with a new creative hybrid art style",
        #     ckpt_stage2_list=(
        #         "ckpts/lining/unet_stage2_lining_step75000.pt",                              
        #         "ckpts/chaoticbrush/unet_stage2_cbrush_step20000.pt",
        #         "ckpts/unet_stage2_felt.pt",
        #     ),
        #     out_png=f"samples/emerging_style/mix_plan_felt_lining_chaoticbrush/{flower_name}.png",
        #     mix_plan={
        #         "down": [0, 0, 0, 0],
        #         "mid": 2,
        #         "up": [1, 1, 1, 1],
        #     },
        #     sample_steps=250
        # )

        main(
            prompt=f"a {flower_name} in the wild, painted with a new creative hybrid art style",
            ckpt_stage2_list=(
                "ckpts/chaoticbrush/unet_stage2_cbrush_step25000.pt",
                "ckpts/oil_preproc/unet_stage2_oil_step47000.pt",                
                "ckpts/fauvism/unet_stage2_fauvism_step20000.pt",
                "ckpts/lining/unet_stage2_lining_step70000.pt"
            ),
            out_png=f"samples/emerging_style/mix_plan_cbrush_oil_fauvism_lining/{flower_name}.png",
            mix_plan={
                "down": [0, 0, 3, 1],
                "mid": 2,
                "up": [3, 3, 0, 0],
            },
        )

        # main(
        #     prompt=f"a {flower_name} in the wild, painted with a new creative hybrid art style",
        #     ckpt_stage2_list=(                
        #         "ckpts/unet_stage2_watercolor_step30000.pt",
        #         "ckpts/lowpoly/unet_stage2_lowpoly_step55000.pt",
        #         "ckpts/pointillism/unet_stage2_pointillism_step26000.pt"
        #     ),
        #     out_png=f"samples/emerging_style/watercolor-40_lowpoly-10_pointillism-50/{flower_name}.png",            
        #     ensemble_mode="soft",
        #     soft_weights=[0.40, 0.10, 0.50], # soft_weights=[0.20, 0.20, 0.60],
        #     soft_where="all",
        # )
