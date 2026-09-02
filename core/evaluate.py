"""Evaluation with held-out residuals and a trivial-solution guard.

Two failure modes this is built to catch, both observed in this project:

A. REPORTING TRAINING RESIDUALS.  The rak branch reports the last optimiser
   closure value.  For Vanilla/Fourier that closure runs on the R3 point set,
   which by construction RETAINS the top-30% highest-residual points; PirateNet's
   runs on plain Sobol.  Comparing those two numbers is meaningless.  Residuals
   here are always computed on a freshly sampled, held-out set.

B. THE TRIVIAL BASIN.  A model that decays to ~0 after the ansatz handover pays
   almost no residual over most of the domain while being ~95% wrong.  Low loss
   plus high error looks like ill-posedness but is usually this.  `norm_ratio`
   detects it directly.
"""
from __future__ import annotations
import numpy as np
import torch
from .problem import Material
from .operators import wave_operator
from .metrics import (spacetime_rel_l2, per_snapshot_rel_l2,
                      per_snapshot_fixed_den, energy_norm_ratio)

TRIVIAL_THRESHOLD = 0.30


@torch.no_grad()
def _predict(u_fn, x_np, t_np, device, dtype):
    X = torch.tensor(x_np, dtype=dtype, device=device).reshape(-1, 1)
    out = np.empty((len(t_np), len(x_np)))
    for i, tv in enumerate(t_np):
        T = torch.full_like(X, float(tv))
        out[i] = u_fn(X, T).detach().cpu().numpy().ravel()
    return out


def evaluate(u_fn, material: Material, x_ref, t_ref, u_ref, *,
             scale: float, n_residual: int = 20000, t_max: float = 1.0,
             device=None, dtype=torch.float64, seed: int = 0) -> dict:
    """Full evaluation.  `u_fn(x,t)` must already include any ansatz."""
    device = device or torch.device("cpu")

    # -- accuracy (grad not needed) --
    u_pred = _predict(u_fn, x_ref, t_ref, device, dtype)

    # -- residual on a HELD-OUT set, freshly sampled, never the training points --
    g = torch.Generator(device="cpu").manual_seed(seed)
    xr = (torch.rand(n_residual, 1, generator=g, dtype=dtype)
          * (material.x_max - material.x_min) + material.x_min).to(device)
    tr = (torch.rand(n_residual, 1, generator=g, dtype=dtype) * t_max).to(device)
    R = wave_operator(u_fn, xr, tr, material) / scale
    res_rms = float(R.pow(2).mean().sqrt().detach())

    ratio = energy_norm_ratio(u_pred, u_ref)
    late = ratio[len(ratio) // 2:]
    trivial = bool(np.min(late) < TRIVIAL_THRESHOLD)

    return {
        "rel_l2": spacetime_rel_l2(u_pred, u_ref),          # headline, fixed denominator
        "per_t_fixed": per_snapshot_fixed_den(u_pred, u_ref),
        "per_t_drifting": per_snapshot_rel_l2(u_pred, u_ref),  # diagnostic only
        "residual_rms_heldout": res_rms,
        "norm_ratio": ratio,
        "trivial_collapse": trivial,
        "min_late_norm_ratio": float(np.min(late)),
        "u_pred": u_pred,
    }


def report(name: str, m: dict) -> str:
    flag = "  *** TRIVIAL COLLAPSE ***" if m["trivial_collapse"] else ""
    return (f"{name:<22} rel-L2 {m['rel_l2']:7.2f}%   "
            f"heldout-resid {m['residual_rms_heldout']:9.3e}   "
            f"min|u|/|u_ref| {m['min_late_norm_ratio']:5.2f}{flag}")
