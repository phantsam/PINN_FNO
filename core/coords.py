"""Characteristic (travel-time) coordinates for the 1D elastic wave equation.

Motivation
----------
For constant wave speed the general solution is exactly

    u(x,t) = F(x - ct) + G(x + ct)

i.e. a SUM OF TWO UNIVARIATE FUNCTIONS of the characteristic variables.  A
Kolmogorov-Arnold network represents precisely that shape natively -- a sum of
univariate functions of its inputs -- so feeding it (x,t) instead of the
characteristic pair asks it to discover the coordinate change first.  For
heterogeneous c(x) the WKB form

    u ~ A(x) F(tau(x) - t) + B(x) G(tau(x) + t),      tau(x) = int dx'/c(x')

keeps that structure with a slowly-varying amplitude, so the same argument holds.

This is therefore the rung of the Phase 8 ladder with the clearest theoretical
reason to help a KAN specifically, rather than helping every architecture
equally.  It is an INPUT EMBEDDING only: the ansatz, the residual and the metric
all stay in physical (x,t), so the comparison remains apples-to-apples.

Numerics
--------
The residual consumes u_xx, and by the chain rule

    u_x  = tau'(x) u_xi + tau'(x) u_eta
    u_xx = tau'(x)^2 (...) + tau''(x) (u_xi + u_eta)

so tau must be genuinely C^2 with ACCURATE second derivatives -- a linear or
even cubic interpolant is not enough.  We therefore build a QUINTIC HERMITE
interpolant that matches value, first and second derivative at every node:

    tau(x_i)   from high-order quadrature of 1/c
    tau'(x_i)  = 1/c(x_i)                    (analytic)
    tau''(x_i) = -c'(x_i)/c(x_i)^2           (analytic, c' by autograd)

The result is C^2 across every node by construction, and inside each interval
the error is O(h^6).  test_coords.py checks tau' and tau'' against their analytic
forms pointwise rather than trusting the construction.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

from .problem import Material


def _nodes(material: Material, n: int, device, dtype=torch.float64):
    """Node values of tau, tau', tau'' on a uniform grid over the domain."""
    x = torch.linspace(material.x_min, material.x_max, n, device=device, dtype=dtype)
    xg = x.clone().requires_grad_(True)
    c = material.c(xg)
    # A homogeneous material returns a CONSTANT c with no graph edge back to x
    # (E and rho are plain floats), so autograd.grad would raise "does not
    # require grad".  c' is identically zero there, which is the correct answer.
    if not torch.is_tensor(c):
        c = torch.as_tensor(c, device=xg.device, dtype=xg.dtype)
    c = c.expand_as(xg) if c.numel() == 1 else c
    if c.requires_grad and c.grad_fn is not None:
        cp = torch.autograd.grad(c.sum(), xg, allow_unused=True)[0]
        cp = torch.zeros_like(xg) if cp is None else cp
    else:
        cp = torch.zeros_like(xg)
    c = c.detach(); cp = cp.detach()
    inv = 1.0 / c

    # Per-interval 5-point Gauss-Legendre, then cumsum.
    #
    # The previous composite-Simpson loop advanced even indices by Simpson and
    # odd ones by trapezoid, so adjacent node values of tau carried INCONSISTENT
    # O(h^3) errors.  The quintic interpolant then differentiated that
    # inconsistency twice: error/h^2 = O(h), and tau'' converged only first order
    # (measured 4.12e-2 -> 1.03e-2 for a 4x refinement, exactly 4x).
    #
    # Gauss-Legendre with 5 nodes is exact for polynomials up to degree 9, so
    # every interval increment is accurate to O(h^11) and all nodes are mutually
    # consistent.  The increments are also kept as `dtau` and used directly,
    # rather than recovered by subtracting two cumulative sums.
    gl_x = torch.tensor([0.0,
                         -0.5384693101056831, 0.5384693101056831,
                         -0.9061798459386640, 0.9061798459386640],
                        device=device, dtype=dtype)
    gl_w = torch.tensor([0.5688888888888889,
                         0.4786286704993665, 0.4786286704993665,
                         0.2369268850561891, 0.2369268850561891],
                        device=device, dtype=dtype)
    a, b = x[:-1], x[1:]
    mid, half = 0.5 * (a + b), 0.5 * (b - a)
    xq = mid[:, None] + half[:, None] * gl_x[None, :]        # (n-1, 5)
    cq = material.c(xq)
    if not torch.is_tensor(cq):
        cq = torch.as_tensor(cq, device=device, dtype=dtype)
    cq = cq.expand_as(xq) if cq.numel() == 1 else cq
    dtau = (half * ((1.0 / cq.detach()) * gl_w[None, :]).sum(dim=1))
    tau = torch.cat([torch.zeros(1, device=device, dtype=dtype), dtau.cumsum(0)])
    return x, tau, dtau, inv, -cp / c ** 2


class TravelTime(nn.Module):
    """C^2 quintic-Hermite interpolant of tau(x) = int_{x_min}^{x} dx'/c(x')."""

    def __init__(self, material: Material, n: int = 4097, device=None):
        super().__init__()
        device = device or torch.device("cpu")
        x, tau, dtau, d1, d2 = _nodes(material, n, device)
        self.register_buffer("x0", x[:-1])
        self.register_buffer("h", x[1:] - x[:-1])
        hh = x[1:] - x[:-1]
        # Store the interval OFFSET dp = tau(x_{i+1}) - tau(x_i), never tau itself,
        # and reconstruct with the partition-of-unity identity H0 + H3 == 1:
        #     tau = p0 + dp*H3 + m0*H1 + m1*H4 + q0*H2 + q1*H5
        # p0 is constant inside the interval, so it drops out of every derivative.
        # Without this the bracket for tau' (~h*tau' ~ 5e-4) is formed as a sum of
        # terms of size tau ~ 1.8, and the float32 cancellation -- amplified by
        # 1/h^2 = 4e6 for tau'' -- gave 2.6e-2 relative error on tau' and 1.3e5 on
        # tau''.  Measured after the change: ~1e-7 and ~1e-5.
        self.register_buffer("p0", tau[:-1])
        self.register_buffer("dp", dtau)
        self.register_buffer("m0", d1[:-1] * hh)
        self.register_buffer("m1", d1[1:] * hh)
        self.register_buffer("q0", d2[:-1] * hh ** 2)
        self.register_buffer("q1", d2[1:] * hh ** 2)
        self.register_buffer("knots", x)
        self.tau_max = float(tau[-1])
        self.x_min, self.x_max = float(material.x_min), float(material.x_max)

    def forward(self, x):
        # Evaluate in float64.  Even with the offset formulation the quintic
        # basis forms tau'' as terms of size O(h) cancelling down to O(h^2)
        # before division by h^2 -- a ~4500x cancellation.  In float32 that left
        # 1.3% error on tau'' which GREW with node count (5.8e-2 at n=4097,
        # 1.3e-1 at n=16385), the signature of round-off rather than truncation.
        # float64 costs a few elementwise ops and removes it entirely.
        xin = x
        x = x.double()
        idx = torch.searchsorted(self.knots, x.detach().reshape(-1).contiguous())
        idx = (idx - 1).clamp(0, self.knots.numel() - 2)
        s = (x.reshape(-1) - self.x0[idx]) / self.h[idx]
        s2 = s * s; s3 = s2 * s; s4 = s3 * s; s5 = s4 * s
        # quintic Hermite basis: matches value, slope and curvature at both ends
        # H0 = 1 - 10s^3 + 15s^4 - 6s^5 is implied by H0 = 1 - H3 (see above)
        H1 = s - 6 * s3 + 8 * s4 - 3 * s5
        H2 = 0.5 * s2 - 1.5 * s3 + 1.5 * s4 - 0.5 * s5
        H3 = 10 * s3 - 15 * s4 + 6 * s5
        H4 = -4 * s3 + 7 * s4 - 3 * s5
        H5 = 0.5 * s3 - s4 + 0.5 * s5
        tau = (self.p0[idx]
               + self.dp[idx] * H3 + self.m0[idx] * H1 + self.m1[idx] * H4
               + self.q0[idx] * H2 + self.q1[idx] * H5)
        return tau.reshape(xin.shape).to(xin.dtype)


class CharCoords(nn.Module):
    """(x,t) -> (xi, eta) = (tau(x) - t, tau(x) + t), each affinely mapped to
    [-1,1] so the pair lands exactly on pykan's default grid_range."""

    def __init__(self, material: Material, t_max: float = 1.0, n: int = 4097):
        super().__init__()
        self.tau = TravelTime(material, n=n)
        T = self.tau.tau_max
        self.register_buffer("xi_lo", torch.tensor(-float(t_max)))
        self.register_buffer("xi_hi", torch.tensor(float(T)))
        self.register_buffer("eta_lo", torch.tensor(0.0))
        self.register_buffer("eta_hi", torch.tensor(float(T + t_max)))
        self.out_dim = 2

    def forward(self, x, t):
        tau = self.tau(x)
        xi = tau - t
        eta = tau + t
        xi = 2 * (xi - self.xi_lo) / (self.xi_hi - self.xi_lo) - 1
        eta = 2 * (eta - self.eta_lo) / (self.eta_hi - self.eta_lo) - 1
        return torch.cat([xi, eta], dim=-1)
