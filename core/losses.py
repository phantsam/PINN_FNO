"""Loss terms for the frozen problem.  Identical for every architecture.

Three deliberate departures from the repo's version, each justified:

1. RESIDUAL NORMALISATION.  The raw residual rho*u_tt - d/dx(E u_x) has scale
   O(|d/dx(E g')|) ~ 2e2..1e4 depending on material and on whether the branch
   divides by rho.  Every downstream constant (causal epsilon, loss weights,
   convergence thresholds) then has to be re-tuned per material.  We divide by a
   FIXED, problem-derived constant so the residual is O(1).  Fixed -- not adaptive
   -- because an adaptive normaliser makes the objective non-stationary.

2. CAUSAL WEIGHTING PER THE ACTUAL PAPER.  Wang, Sankaran & Perdikaris
   (arXiv:2203.07404) Algorithm 1 prescribes an ANNEALED increasing sequence
   eps in [1e-2, 1e-1, 1, 10, 100] with advance criterion min_i w_i > delta,
   delta = 0.99 -- not a single fixed epsilon.  Their epsilon values are only
   meaningful when L = O(1), which is why (1) must come first.

3. GRAD-NORM BALANCING between PDE and BC terms (inverse gradient magnitude),
   never the inverted rule w_i = L_i / sum(L) found in src/train.py:271.
"""
from __future__ import annotations
import numpy as np
import torch
from .problem import Material, gaussian_ic, gaussian_ic_dx
from .operators import wave_operator, abc_operator


# ─────────────────────────────────────────────────────────────
# 1. Fixed residual scale
# ─────────────────────────────────────────────────────────────
def residual_scale(material: Material, sigma_g: float = 0.1, n: int = 20001) -> float:
    """max |d/dx( E(x) g'(x) )| -- the magnitude of the initial acceleration.

    Computed once at setup from the problem definition alone; independent of the
    network, the grid used at training time, and the optimiser.
    """
    x = np.linspace(material.x_min, material.x_max, n)
    E = np.asarray(material.E(x), float)
    gx = np.asarray(gaussian_ic_dx(x, sigma_g), float)
    return float(np.max(np.abs(np.gradient(E * gx, x))))


# ─────────────────────────────────────────────────────────────
# 2. Causal weighting (Wang et al. 2022, Algorithm 1)
# ─────────────────────────────────────────────────────────────
def causal_weights(chunk_losses: torch.Tensor, eps: float) -> torch.Tensor:
    """w_i = exp(-eps * sum_{k<i} L_k);  w_1 = 1.  Cumsum is detached."""
    cum = torch.zeros_like(chunk_losses)
    cum[1:] = torch.cumsum(chunk_losses[:-1].detach(), dim=0)
    return torch.exp(-eps * cum)


class CausalScheduler:
    """Anneals eps over an increasing sequence, advancing when min_i w_i > delta."""

    def __init__(self, eps_seq=(1e-2, 1e-1, 1.0, 10.0, 100.0), delta: float = 0.99):
        self.eps_seq = list(eps_seq)
        self.delta = float(delta)
        self.k = 0

    @property
    def eps(self) -> float:
        return self.eps_seq[self.k]

    @property
    def finished(self) -> bool:
        return self.k >= len(self.eps_seq) - 1

    def step(self, weights: torch.Tensor) -> bool:
        """Advance if the criterion is met.  Returns True if eps was advanced."""
        if not self.finished and float(weights.min()) > self.delta:
            self.k += 1
            return True
        return False


# ─────────────────────────────────────────────────────────────
# 3. Loss terms
# ─────────────────────────────────────────────────────────────
def pde_loss(u_fn, x, t, material: Material, scale: float, *,
             n_chunks: int = 32, eps: float | None = None, t_max: float = 1.0):
    """Normalised PDE residual loss.  Returns (loss, chunk_losses, weights)."""
    R = wave_operator(u_fn, x, t, material) / scale
    if eps is None or n_chunks <= 1:
        return (R**2).mean(), None, None

    # bucket by time; include the right edge so t == t_max is not dropped
    idx = torch.clamp((t.squeeze(-1) / t_max * n_chunks).long(), 0, n_chunks - 1)
    sq = (R**2).squeeze(-1)
    cnt = torch.zeros(n_chunks, device=R.device).index_add_(0, idx, torch.ones_like(sq))
    tot = torch.zeros(n_chunks, device=R.device).index_add_(0, idx, sq)
    chunk = tot / cnt.clamp(min=1.0)
    w = causal_weights(chunk, eps)
    return (w * chunk).mean(), chunk.detach(), w.detach()


def bc_scale(material: Material, sigma_g: float = 0.1, n: int = 20001) -> float:
    """c_max * max|g'(x)| -- the magnitude of u_t for a travelling pulse.

    The ABC residual u_t +- c u_x has units of u/time, NOT of the PDE residual
    (u/time^2 * density).  Dividing it by `residual_scale` would be a ~5-order
    dimensional mismatch, leaving the BC term numerically negligible before
    grad-norm balancing ever sees it.
    """
    x = np.linspace(material.x_min, material.x_max, n)
    return float(material.c_max * np.max(np.abs(gaussian_ic_dx(x, sigma_g))))


def bc_loss(u_fn, t, material: Material, scale: float):
    """Absorbing-BC residual at both ends.  `scale` must come from `bc_scale`,
    not from `residual_scale` -- the two have different physical dimensions."""
    xL = torch.full_like(t, material.x_min)
    xR = torch.full_like(t, material.x_max)
    rL = abc_operator(u_fn, xL, t.clone(), material, "left")
    rR = abc_operator(u_fn, xR, t.clone(), material, "right")
    return ((rL**2).mean() + (rR**2).mean()) / scale**2


def grad_norm_weights(model, loss_a, loss_b, eps: float = 1e-12):
    """Inverse mean-|grad| balancing.  NOT L_i/sum(L) -- that amplifies the
    dominant term and annihilates the other (see AUDIT.md 4.4)."""
    # only differentiable leaves: pykan registers some non-trainable parameters,
    # and torch.autograd.grad raises if any target does not require grad
    params = [p for p in model.parameters() if p.requires_grad]

    def mag(loss):
        if not loss.requires_grad or not params:
            return 1.0
        g = torch.autograd.grad(loss, params,
                                retain_graph=True, allow_unused=True)
        v = [gi.abs().mean().item() for gi in g if gi is not None]
        return float(sum(v) / max(len(v), 1))
    a, b = mag(loss_a), mag(loss_b)
    ref = 0.5 * (a + b) + eps
    return ref / (a + eps), ref / (b + eps)
