"""Gradient- and derivative-health checks for physics-informed models.

Why this module exists
----------------------
Two failure modes have already cost this project weeks, and both were silent:

  * D6 residual normalisation shrank gradients ~5e4x.  Nothing crashed; the
    L-BFGS first step (t = min(1, 1/||g||_1)*lr) simply overshot and every
    architecture "just performed badly".
  * The hand-rolled SplineKAN produced max|u_tt| ~ 5e5 at initialisation against
    the Fourier PINN's 1.8e2.  Again nothing crashed -- it collapsed to u = 0.

Both are detectable in a few forward/backward passes BEFORE spending GPU-hours.
So every architecture entering the Phase 8 ladder is screened here first.

The checks are deliberately *scale-aware*.  A PDE residual built on second
derivatives amplifies any pathology in the basis quadratically, so we measure the
derivatives the operator actually consumes (u_x, u_xx, u_t, u_tt) rather than
just the network output.
"""
from __future__ import annotations
import math
import torch

from .operators import make_ansatz, wave_operator, _grad
from .losses import pde_loss, bc_loss
from .problem import Material

# Thresholds.  These are ORDER-OF-MAGNITUDE screens, not tuned constants; each
# is justified against a measured reference in the comment beside it.
VANISH_RMS   = 1e-12      # per-group grad RMS below this is numerically dead
EXPLODE_RMS  = 1e8        # per-group grad RMS above this will blow a line search
DEPTH_RATIO  = 1e6        # max/min group RMS: >1e6 means the deep end is starved
UTT_HI       = 1e4        # measured: Fourier PINN 1.8e2, pykan 5.2e-3 at init
UTT_LO       = 1e-8       # a model whose u_tt is ~0 at init cannot feel the PDE
DEAD_OUT     = 1e-12      # raw network output std; catches a constant network
FP32_EPS     = 1.19e-7


def _groups(model):
    """Group parameters by their top-two name components (~ one per layer)."""
    out = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        key = ".".join(name.split(".")[:-1]) or name   # owning module = one layer
        out.setdefault(key, []).append((name, p))
    return out


@torch.no_grad()
def _rms(ps):
    tot, n = 0.0, 0
    for _, p in ps:
        g = p.grad
        if g is None:
            continue
        tot += float((g.double() ** 2).sum())
        n += g.numel()
    return math.sqrt(tot / n) if n else 0.0


def derivative_scale(model, material: Material, *, ansatz_kind="legacy",
                     sigma_g=0.1, t_max=1.0, n=4096, device=None, seed=0):
    """Magnitudes of the derivatives the wave operator consumes, at the model's
    CURRENT parameters, under the same hard-IC ansatz the trainer uses.

    Returns max-abs of u, u_x, u_xx, u_t, u_tt and of the raw PDE residual.
    """
    device = device or next(model.parameters()).device
    g = torch.Generator(device=device).manual_seed(seed)
    x = (torch.rand(n, 1, generator=g, device=device)
         * (material.x_max - material.x_min) + material.x_min).requires_grad_(True)
    t = (torch.rand(n, 1, generator=g, device=device) * t_max).requires_grad_(True)

    ansatz = make_ansatz(ansatz_kind, sigma_g=sigma_g)
    raw = model(x, t)
    u = ansatz(raw, x, t)
    u_x = _grad(u, x); u_xx = _grad(u_x, x)
    u_t = _grad(u, t); u_tt = _grad(u_t, t)
    r = material.rho(x) * u_tt - _grad(material.E(x) * u_x, x)
    m = lambda v: float(v.detach().abs().max())
    # The RAW network output must be measured separately.  Under the hard-IC
    # ansatz u = g(x)decay(t) + growth(t)N(x,t), the g*decay term contributes
    # u_tt on its own, so a network that outputs a constant -- PirateNet without
    # its LSQ warm start is identically zero -- still shows a healthy |u_tt|.
    # Only the raw output's spread reveals it.
    return dict(u=m(u), u_x=m(u_x), u_xx=m(u_xx), u_t=m(u_t), u_tt=m(u_tt),
                residual=m(r), raw_std=float(raw.detach().std()),
                raw_absmax=float(raw.detach().abs().max()),
                finite=bool(torch.isfinite(r).all() and torch.isfinite(u_tt).all()))


def gradient_health(model, material: Material, *, ansatz_kind="legacy",
                    sigma_g=0.1, t_max=1.0, n_col=4096, n_bc=256,
                    device=None, seed=0, use_bc=True):
    """One backward pass through the real physics loss; report per-group scale.

    Diagnoses vanishing/exploding gradients *as the optimiser would see them*:
    the loss is the unnormalised PDE + BC loss the L-BFGS path actually uses.
    """
    device = device or next(model.parameters()).device
    g = torch.Generator(device=device).manual_seed(seed)
    x = (torch.rand(n_col, 1, generator=g, device=device)
         * (material.x_max - material.x_min) + material.x_min)
    t = torch.rand(n_col, 1, generator=g, device=device) * t_max
    tb = torch.rand(n_bc, 1, generator=g, device=device) * t_max

    ansatz = make_ansatz(ansatz_kind, sigma_g=sigma_g)
    u_fn = lambda a, b: ansatz(model(a, b), a, b)

    model.zero_grad(set_to_none=True)
    loss, _, _ = pde_loss(u_fn, x.clone(), t.clone(), material, 1.0, eps=None)
    if use_bc:
        loss = loss + bc_loss(u_fn, tb.clone(), material, 1.0)
    loss.backward()

    grp = {k: _rms(v) for k, v in _groups(model).items()}
    live = {k: v for k, v in grp.items() if v > 0.0}
    gn = math.sqrt(sum(v ** 2 for v in grp.values()))
    n_none = sum(1 for _, p in model.named_parameters()
                 if p.requires_grad and p.grad is None)
    out = dict(loss=float(loss.detach()), global_grad_norm=gn, per_group=grp,
               n_groups=len(grp), n_no_grad=n_none,
               min_group_rms=min(live.values()) if live else 0.0,
               max_group_rms=max(grp.values()) if grp else 0.0)
    out["depth_ratio"] = (out["max_group_rms"] / out["min_group_rms"]
                          if out["min_group_rms"] > 0 else math.inf)
    model.zero_grad(set_to_none=True)
    return out


def screen(model, material: Material, *, name="model", **kw):
    """Run both checks and return (ok, warnings, report).

    `ok` is False only for conditions that make training meaningless, not for
    merely-suboptimal ones -- the caller decides whether to proceed.
    """
    d = derivative_scale(model, material, **{k: v for k, v in kw.items()
                                             if k in ("ansatz_kind", "sigma_g", "t_max",
                                                      "device", "seed", "n")})
    h = gradient_health(model, material, **{k: v for k, v in kw.items()
                                            if k in ("ansatz_kind", "sigma_g", "t_max",
                                                     "device", "seed", "n_col", "n_bc",
                                                     "use_bc")})
    w, fatal = [], False
    if not d["finite"]:
        w.append("FATAL non-finite u_tt or residual at init"); fatal = True
    if not math.isfinite(h["loss"]):
        w.append("FATAL non-finite loss at init"); fatal = True
    if d["u_tt"] > UTT_HI:
        w.append(f"u_tt={d['u_tt']:.3g} > {UTT_HI:g} -- basis too stiff for a "
                 f"2nd-order residual (SplineKAN failed this at 5e5)")
    if d["u_tt"] < UTT_LO:
        w.append(f"u_tt={d['u_tt']:.3g} < {UTT_LO:g} -- model is ~flat in t, "
                 f"the PDE term cannot drive learning")
    if d["raw_std"] < DEAD_OUT:
        w.append(f"DEAD OUTPUT: raw network std {d['raw_std']:.3g} < {DEAD_OUT:g} "
                 f"-- the network is constant, only the ansatz's IC term is live "
                 f"(this is PirateNet without physics_informed_init)")
    if h["n_no_grad"]:
        w.append(f"{h['n_no_grad']} trainable tensors received NO gradient "
                 f"(disconnected from the loss)")
    if 0.0 < h["min_group_rms"] < VANISH_RMS:
        w.append(f"VANISHING: slowest group RMS {h['min_group_rms']:.3g} "
                 f"< {VANISH_RMS:g}")
    if h["max_group_rms"] > EXPLODE_RMS:
        w.append(f"EXPLODING: fastest group RMS {h['max_group_rms']:.3g} "
                 f"> {EXPLODE_RMS:g}")
    if h["depth_ratio"] > DEPTH_RATIO:
        w.append(f"DEPTH IMBALANCE: max/min group RMS = {h['depth_ratio']:.3g}")
    return (not fatal), w, dict(name=name, **d, **{k: v for k, v in h.items()
                                                   if k != "per_group"},
                                per_group=h["per_group"])


def format_report(rep, warnings):
    L = [f"  {rep['name']}",
         f"    |u|={rep['u']:.3e}  |u_x|={rep['u_x']:.3e}  |u_xx|={rep['u_xx']:.3e}",
         f"    |u_t|={rep['u_t']:.3e}  |u_tt|={rep['u_tt']:.3e}  |resid|={rep['residual']:.3e}",
         f"    raw_std={rep['raw_std']:.3e}  raw_absmax={rep['raw_absmax']:.3e}",
         f"    loss={rep['loss']:.4e}  ||grad||={rep['global_grad_norm']:.4e}  "
         f"groups={rep['n_groups']}  depth_ratio={rep['depth_ratio']:.3g}"]
    for w in warnings:
        L.append(f"    !! {w}")
    return "\n".join(L)
