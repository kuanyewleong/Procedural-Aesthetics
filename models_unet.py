import torch, torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

def timestep_embedding(t, dim):
    # sinusoidal embedding, t: [B]
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device).float() / (half - 1))
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb  # [B, dim]

class CondFiLM(nn.Module):
    def __init__(self, ch, d_ctx):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.to_scale_shift = nn.Linear(d_ctx, 2 * ch)

    def forward(self, x, ctx):  # ctx: [B,1,D] or [B,D]
        if ctx.dim() == 3:
            ctx = ctx[:, 0]
        h = self.norm(x)
        ss = self.to_scale_shift(ctx).unsqueeze(-1).unsqueeze(-1)  # [B,2C,1,1]
        scale, shift = ss.chunk(2, dim=1)
        return x + h * (1 + scale) + shift

class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, t_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, 1, 1)
        self.gn1 = nn.GroupNorm(8, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1)
        self.gn2 = nn.GroupNorm(8, c_out)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.to_scale_shift = nn.Linear(t_dim, 2 * c_out)  # FiLM-style injection

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.gn1(h)
        ss = self.to_scale_shift(t_emb).unsqueeze(-1).unsqueeze(-1)
        scale, shift = ss.chunk(2, dim=1)
        h = h * (1 + scale) + shift
        h = self.act(h)
        h = self.conv2(h)
        h = self.gn2(h)
        h = self.act(h)
        return h + self.skip(x)

class SelfAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.q = nn.Conv2d(ch, ch, 1)
        self.k = nn.Conv2d(ch, ch, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q = rearrange(self.q(x), "b c h w -> b (h w) c")
        k = rearrange(self.k(x), "b c h w -> b (h w) c")
        v = rearrange(self.v(x), "b c h w -> b (h w) c")
        attn = torch.softmax((q @ k.transpose(1, 2)) / (c**0.5), dim=-1)
        out = attn @ v
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)
        return x + self.proj(out)

class DownStage(nn.Module):    
    def __init__(self, c_in, c_out, t_dim, d_ctx=None):
        super().__init__()
        self.rb1 = ResBlock(c_in,  c_out, t_dim)
        self.rb2 = ResBlock(c_out, c_out, t_dim)
        self.ca = CondFiLM(c_out, d_ctx) if d_ctx is not None else None

    def forward(self, x, t_emb, ctx=None):
        x = self.rb1(x, t_emb)
        x = self.rb2(x, t_emb)
        if self.ca is not None:
            x = self.ca(x, ctx)
        return x

class UpStage(nn.Module):    
    def __init__(self, c_in, c_out, t_dim):
        super().__init__()
        self.rb1 = ResBlock(c_in,  c_out, t_dim)
        self.rb2 = ResBlock(c_out, c_out, t_dim)

    def forward(self, x, t_emb):
        x = self.rb1(x, t_emb)
        x = self.rb2(x, t_emb)
        return x

class UNetCond(nn.Module):
    def __init__(self, z_ch=4, base=128, d_ctx=512, t_dim=256):
        super().__init__()
        self.t_dim = t_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim * 4),
            nn.SiLU(),
            nn.Linear(t_dim * 4, t_dim),
        )

        self.inp = nn.Conv2d(z_ch, base, 3, 1, 1)
        self.pool = nn.AvgPool2d(2)

        # Encoder: 4 downs (extra stage to reach 16x16)
        self.down1 = DownStage(base,     base,     t_dim, d_ctx=d_ctx)      # 256
        self.down2 = DownStage(base,     base * 2, t_dim, d_ctx=d_ctx)      # 128
        self.down3 = DownStage(base * 2, base * 4, t_dim, d_ctx=d_ctx)      # 64
        self.down4 = DownStage(base * 4, base * 8, t_dim, d_ctx=d_ctx)      # 32

        # Mid at 16x16
        self.mid_rb1 = ResBlock(base * 8, base * 8, t_dim)
        self.mid_ca  = CondFiLM(base * 8, d_ctx)
        self.mid_sa  = SelfAttention(base * 8)
        self.mid_rb2 = ResBlock(base * 8, base * 8, t_dim)

        # Decoder: 4 ups (mirrors encoder)
        self.up4 = UpStage(base * 16, base * 4, t_dim)  # (8b + 8b) -> 4b
        self.up3 = UpStage(base * 8,  base * 2, t_dim)  # (4b + 4b) -> 2b
        self.up2 = UpStage(base * 4,  base,     t_dim)  # (2b + 2b) -> b
        self.up1 = UpStage(base * 2,  base,     t_dim)  # (b + b)   -> b

        self.out = nn.Conv2d(base, z_ch, 3, 1, 1)

    def forward(self, x, t, ctx):
        t_emb = timestep_embedding(t, self.t_dim)
        t_emb = self.time_mlp(t_emb)

        x0 = self.inp(x)  # [B, base, 256, 256]

        d1 = self.down1(x0, t_emb, ctx); x1 = self.pool(d1)  # -> 128
        d2 = self.down2(x1, t_emb, ctx); x2 = self.pool(d2)  # -> 64
        d3 = self.down3(x2, t_emb, ctx); x3 = self.pool(d3)  # -> 32
        d4 = self.down4(x3, t_emb, ctx); x4 = self.pool(d4)  # -> 16

        m = self.mid_rb1(x4, t_emb)
        m = self.mid_ca(m, ctx)
        m = self.mid_sa(m)
        m = self.mid_rb2(m, t_emb)

        u4 = F.interpolate(m, scale_factor=2, mode="nearest")         # 16 -> 32
        u4 = self.up4(torch.cat([u4, d4], dim=1), t_emb)

        u3 = F.interpolate(u4, scale_factor=2, mode="nearest")        # 32 -> 64
        u3 = self.up3(torch.cat([u3, d3], dim=1), t_emb)

        u2 = F.interpolate(u3, scale_factor=2, mode="nearest")        # 64 -> 128
        u2 = self.up2(torch.cat([u2, d2], dim=1), t_emb)

        u1 = F.interpolate(u2, scale_factor=2, mode="nearest")        # 128 -> 256
        u1 = self.up1(torch.cat([u1, d1], dim=1), t_emb)

        return self.out(u1)
