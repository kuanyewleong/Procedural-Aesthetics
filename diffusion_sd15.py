import torch, math

class CosineScheduler:
    """
    Precomputes:
      betas[t], alphas[t], alpha_bar[t] for t=0..T-1
    """
    def __init__(self, timesteps=1000, s=0.008):
        self.T = timesteps
        steps = torch.arange(timesteps + 1, dtype=torch.float32)

        # cosine alpha_bar per Nichol & Dhariwal
        alpha_bar = torch.cos(((steps / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]  # alpha_bar[0]=1

        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        betas = betas.clamp(1e-6, 0.999)

        alphas = 1 - betas

        self.betas = betas                      # [T]
        self.alphas = alphas                    # [T]
        self.alpha_bar = torch.cumprod(alphas, dim=0)  # [T]

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        return self


def q_sample(x0, t, noise, alpha_bar):
    """
    x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*eps
    t: [B] long
    alpha_bar: [T]
    """
    a = alpha_bar[t].view(-1, 1, 1, 1)
    return a.sqrt() * x0 + (1 - a).sqrt() * noise

def _dynamic_threshold(x0, p=0.999):
    # Latent-friendly: clip extremes but do NOT normalize by s
    s = torch.quantile(x0.abs().flatten(1), p, dim=1).view(-1, 1, 1, 1)
    return torch.clamp(x0, -s, s)

def make_ddim_timesteps(T, steps, device):
    stride = max(T // steps, 1)
    ts = torch.arange(T-1, -1, -stride, device=device, dtype=torch.long)
    if ts[-1].item() != 0:
        ts = torch.cat([ts, torch.zeros(1, device=device, dtype=torch.long)])
    return ts

@torch.no_grad()
def p_sample_loop(
        model,
        sched,
        shape,
        ctx_cond,
        ctx_uncond,
        device,
        guidance_scale=3.0,
        sample_steps=None,
        use_ddim=True,
        eta=0.0,
        dyn_thresh=False,
        dyn_p=0.999,
        cfg_rescale=True,
        x0_clip=3.0,
        eps_clip=10.0,
    ):
    T = sched.T
    if sample_steps is None or sample_steps >= T:
        t_list = torch.arange(T - 1, -1, -1, device=device, dtype=torch.long)
    else:
        t_list = make_ddim_timesteps(T, sample_steps, device)

    x = torch.randn(shape, device=device)

    for i, t in enumerate(t_list):
        tt = torch.full((shape[0],), int(t.item()), device=device, dtype=torch.long)

        abar_t = sched.alpha_bar[t]
        sqrt_abar_t = torch.sqrt(abar_t)
        sqrt_one_minus_abar_t = torch.sqrt(1 - abar_t)

        # next timestep
        if i == len(t_list) - 1:
            abar_prev = torch.tensor(1.0, device=device)
        else:
            abar_prev = sched.alpha_bar[t_list[i + 1]]
        sqrt_abar_prev = torch.sqrt(abar_prev)
        sqrt_one_minus_abar_prev = torch.sqrt(1 - abar_prev)

        # CFG eps
        eps_u = model(x, tt, ctx_uncond) if ctx_uncond is not None else 0.0
        eps_c = model(x, tt, ctx_cond)   if ctx_cond   is not None else eps_u
        eps = eps_u + guidance_scale * (eps_c - eps_u)

        # optional: CFG rescale to prevent blow-up
        if cfg_rescale and guidance_scale > 0 and isinstance(eps_u, torch.Tensor):
            std_u = eps_u.std(dim=(1,2,3), keepdim=True)
            std_e = eps.std(dim=(1,2,3), keepdim=True)
            eps = eps * (std_u / (std_e + 1e-8))

        # eps safety clip (helps with rare spikes)
        if eps_clip is not None:
            eps = torch.clamp(eps, -eps_clip, eps_clip)

        # predict x0
        x0_hat = (x - sqrt_one_minus_abar_t * eps) / (sqrt_abar_t + 1e-8)

        # latent clamp (VERY important for stability)
        if x0_clip is not None:
            x0_hat = torch.clamp(x0_hat, -x0_clip, x0_clip)

        # deterministic DDIM update (eta=0)
        x = sqrt_abar_prev * x0_hat + sqrt_one_minus_abar_prev * eps

        # debug: print stats at key steps
        if int(t.item()) in [999, 500, 200, 50, 0]:
            print("t", int(t.item()), "x std", float(x.std()), "x0hat std", float(x0_hat.std()))  
        if int(t.item()) == 999:
            print("eps std", float(eps.std()), "eps_u std", float(eps_u.std()), "eps_c std", float(eps_c.std()))
    
    return x






# @torch.no_grad()
# def p_sample_loop(
#     model,
#     sched: CosineScheduler,
#     shape,
#     ctx_cond,
#     ctx_uncond,
#     device,
#     guidance_scale=3.0,
#     sample_steps=None,          # e.g. 500; if None uses sched.T
#     use_ddim=False,             # deterministic sampling option
#     eta=0.0,                    # DDIM noise: 0.0 deterministic; >0 adds noise
#     dyn_thresh=False,           # optional, off by default
#     dyn_p=0.995
# ):
#     """
#     model(x, t, ctx) -> eps_hat  (epsilon prediction)
#     """

#     # choose timestep schedule
#     T = sched.T
#     if sample_steps is None or sample_steps >= T:
#         t_list = torch.arange(T - 1, -1, -1, device=device, dtype=torch.long)
#     else:
#         t_list = torch.linspace(T - 1, 0, steps=sample_steps, device=device).long()
#         t_list[-1] = 0

#     if t_list[-1].item() != 0:
#         t_list = torch.cat([t_list, torch.zeros(1, device=device, dtype=torch.long)], dim=0)


#     x = torch.randn(shape, device=device)

#     for i, t in enumerate(t_list):
#         tt = torch.full((shape[0],), int(t.item()), device=device, dtype=torch.long)

#         abar_t = sched.alpha_bar[t]                          # scalar
#         sqrt_abar_t = torch.sqrt(abar_t)
#         sqrt_one_minus_abar_t = torch.sqrt(1 - abar_t)

#         # ---- CFG (optional) ----
#         eps_u = model(x, tt, ctx_uncond) if ctx_uncond is not None else 0.0
#         eps_c = model(x, tt, ctx_cond)   if ctx_cond   is not None else eps_u
#         eps = eps_u + guidance_scale * (eps_c - eps_u)
#         # ------------------------

#         # eps-pred -> x0_hat
#         x0_hat = (x - sqrt_one_minus_abar_t * eps) / (sqrt_abar_t + 1e-8)
#         # HARD latent clamp (debug stabilizer; keep for now)
#         # x0_hat = x0_hat.clamp(-3.0, 3.0)

#         # optional gentler clip (can keep after things work)
#         if dyn_thresh:
#             x0_hat = _dynamic_threshold(x0_hat, p=dyn_p)

#         # # eps-pred -> x0_hat
#         # x0_hat = (x - sqrt_one_minus_abar_t * eps) / (sqrt_abar_t + 1e-8)

#         # previous alpha_bar (for last step, treat prev as 1.0)
#         if i == len(t_list) - 1:
#             abar_prev = torch.tensor(1.0, device=device)
#         else:
#             t_prev = t_list[i + 1]
#             abar_prev = sched.alpha_bar[t_prev]

#         sqrt_abar_prev = torch.sqrt(abar_prev)
#         sqrt_one_minus_abar_prev = torch.sqrt(1 - abar_prev)

#         if use_ddim:
#             if eta == 0.0:
#                 x = sqrt_abar_prev * x0_hat + sqrt_one_minus_abar_prev * eps
#             else:
#                 sigma = eta * torch.sqrt((1 - abar_prev) / (1 - abar_t + 1e-8)) * torch.sqrt(
#                     1 - abar_t / (abar_prev + 1e-8)
#                 )
#                 # direction term
#                 dir_xt = torch.sqrt(torch.clamp(1 - abar_prev - sigma**2, min=0.0)) * eps
#                 x = sqrt_abar_prev * x0_hat + dir_xt + sigma * torch.randn_like(x)
#         else:            
#             beta_t = sched.betas[t]
#             alpha_t = sched.alphas[t]
#             c1 = sqrt_abar_prev * beta_t / (1 - abar_t + 1e-8)
#             c2 = torch.sqrt(alpha_t) * (1 - abar_prev) / (1 - abar_t + 1e-8)
#             mean = c1 * x0_hat + c2 * x

#             if i == len(t_list) - 1:
#                 x = mean
#             else:
#                 # posterior variance
#                 var = beta_t * (1 - abar_prev) / (1 - abar_t + 1e-8)
#                 x = mean + torch.sqrt(var + 1e-8) * torch.randn_like(x)

#         # debug: print stats at key steps
#         if int(t.item()) in [999, 500, 200, 50, 0]:
#             print("t", int(t.item()), "x std", float(x.std()), "x0hat std", float(x0_hat.std()))  
#         if int(t.item()) == 999:
#             print("eps std", float(eps.std()), "eps_u std", float(eps_u.std()), "eps_c std", float(eps_c.std()))
    
#     return x
