"""Differential operators for the 1D elastic wave problem.

The key design point: `wave_operator` is a PURE operator on an arbitrary
callable u(x,t).  It knows nothing about networks or the ansatz.  That is what
makes Method-of-Manufactured-Solutions verification possible -- you can feed it
any analytic u* and check it against a symbolically derived source term.

The network path is a thin wrapper that supplies `ansatz(model)` as the callable.
"""
from __future__ import annotations
import torch
from .problem import Material, gaussian_ic


def _grad(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                               create_graph=True)[0]


# ─────────────────────────────────────────────────────────────
# 1. Pure operators (no ansatz, no model)
# ─────────────────────────────────────────────────────────────
def wave_operator(u_fn, x, t, material: Material, *, drop_E_prime: bool = False):
    """rho(x) u_tt - d/dx( E(x) u_x )   evaluated on the callable u_fn(x,t).

    `drop_E_prime` deliberately BREAKS the operator (uses E*u_xx instead of the
    full divergence).  It exists solely so the mutation tests can prove the
    verification suite actually detects a wrong equation.
    """
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)
    u = u_fn(x, t)
    u_t = _grad(u, t)
    u_tt = _grad(u_t, t)
    u_x = _grad(u, x)
    if drop_E_prime:
        u_xx = _grad(u_x, x)
        div = material.E(x) * u_xx                    # WRONG: misses E'(x) u_x
    else:
        div = _grad(material.E(x) * u_x, x)           # correct divergence form
    return material.rho(x) * u_tt - div


def abc_operator(u_fn, x, t, material: Material, side: str):
    """Outgoing (absorbing) boundary residual.

    right (x_max): u = F(x-ct) => u_t + c u_x = 0
    left  (x_min): u = G(x+ct) => u_t - c u_x = 0
    """
    assert side in ("left", "right")
    x = x.requires_grad_(True)
    t = t.requires_grad_(True)
    u = u_fn(x, t)
    u_t, u_x = _grad(u, t), _grad(u, x)
    n = 1.0 if side == "right" else -1.0
    return u_t + n * material.c(x) * u_x


# ─────────────────────────────────────────────────────────────
# 2. Hard-constraint ansatz
# ─────────────────────────────────────────────────────────────
def make_ansatz(kind: str = "poly", sigma_g: float = 0.1, decay_rate: float = 15.0,
                growth_rate: float = 25.0):
    """Return f(nn_out, x, t) enforcing u(x,0)=g(x) and u_t(x,0)=0 exactly.

    kind='legacy' : g*exp(-.5(a t)^2) + tanh(b t)^2 * N     (the repo's form)
    kind='poly'   : g(x) + t^2 * N                          (no dead decay term,
                    no >1 overshoot, no trivial basin -- see Phase 3 A/B)
    """
    if kind == "legacy":
        def f(nn_out, x, t):
            g = gaussian_ic(x, sigma_g)
            return (g * torch.exp(-0.5 * (decay_rate * t) ** 2)
                    + torch.tanh(growth_rate * t) ** 2 * nn_out)
    elif kind == "poly":
        def f(nn_out, x, t):
            return gaussian_ic(x, sigma_g) + t**2 * nn_out
    else:
        raise ValueError(kind)
    return f


def ansatz_callable(model, ansatz, ):
    """Wrap a model(x,t) -> nn_out into a plain u(x,t) callable."""
    def u_fn(x, t):
        return ansatz(model(x, t), x, t)
    return u_fn
