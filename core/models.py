"""Architectures.  Interchangeable; the problem is not.

Every Fourier-embedded model shares ONE embedding with ONE bandwidth
(DECISIONS.md D4: sigma_B = 10, derived from the solution's measured spectrum),
so bandwidth can never again be confounded with architecture.

All models expose the same interface:  forward(x, t) -> raw network output.
The hard-IC ansatz is applied outside, by the trainer, identically for all.
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SIGMA_B = 10.0          # DECISIONS.md D4


class FourierEmbed(nn.Module):
    """Shared random Fourier feature map.  B is a fixed buffer (not trained), so
    the bandwidth stays at its declared value for the whole run."""

    def __init__(self, n_fourier: int = 64, sigma: float = SIGMA_B, in_dim: int = 2):
        super().__init__()
        self.register_buffer("B", torch.randn(n_fourier, in_dim) * sigma)
        self.out_dim = 2 * n_fourier

    def forward(self, x, t):
        p = torch.cat([x, t], dim=-1) @ self.B.T
        return torch.cat([torch.cos(p), torch.sin(p)], dim=-1)


# ─────────────────────────────── PINNs ───────────────────────────────
class MLP(nn.Module):
    """Plain tanh MLP on raw (x,t).  No Fourier features -- the control."""

    def __init__(self, layers: int = 5, units: int = 128):
        super().__init__()
        L = [nn.Linear(2, units), nn.Tanh()]
        for _ in range(layers - 1):
            L += [nn.Linear(units, units), nn.Tanh()]
        L.append(nn.Linear(units, 1))
        self.net = nn.Sequential(*L)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=-1))


class FourierMLP(nn.Module):
    def __init__(self, layers: int = 5, units: int = 128, n_fourier: int = 64,
                 sigma: float = SIGMA_B):
        super().__init__()
        self.embed = FourierEmbed(n_fourier, sigma)
        L = [nn.Linear(self.embed.out_dim, units), nn.Tanh()]
        for _ in range(layers - 1):
            L += [nn.Linear(units, units), nn.Tanh()]
        L.append(nn.Linear(units, 1))
        self.net = nn.Sequential(*L)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x, t):
        return self.net(self.embed(x, t))


class _RWFLinear(nn.Module):
    """Random weight factorisation (PirateNet)."""

    def __init__(self, i, o, mu=1.0, sd=0.1):
        super().__init__()
        self.V = nn.Parameter(torch.empty(o, i)); nn.init.xavier_uniform_(self.V)
        self.s = nn.Parameter(torch.normal(mu, sd, (o,)))
        self.bias = nn.Parameter(torch.zeros(o))

    def forward(self, x):
        return F.linear(x, self.s.unsqueeze(1) * self.V, self.bias)


class _PirateBlock(nn.Module):
    def __init__(self, u):
        super().__init__()
        self.W1, self.W2, self.W3 = _RWFLinear(u, u), _RWFLinear(u, u), _RWFLinear(u, u)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, h, U, V):
        f = torch.tanh(self.W1(h)); z1 = f * U + (1 - f) * V
        g = torch.tanh(self.W2(z1)); z2 = g * U + (1 - g) * V
        return self.alpha * torch.tanh(self.W3(z2)) + (1 - self.alpha) * h


class PirateNet(nn.Module):
    def __init__(self, blocks: int = 3, units: int = 128, n_fourier: int = 64,
                 sigma: float = SIGMA_B):
        super().__init__()
        self.embed = FourierEmbed(n_fourier, sigma)
        d = self.embed.out_dim
        self.encU, self.encV, self.proj = _RWFLinear(d, units), _RWFLinear(d, units), _RWFLinear(d, units)
        self.blocks = nn.ModuleList([_PirateBlock(units) for _ in range(blocks)])
        self.out = nn.Linear(units, 1, bias=False); nn.init.zeros_(self.out.weight)

    def _features(self, x, t):
        phi = self.embed(x, t)
        U, V = torch.tanh(self.encU(phi)), torch.tanh(self.encV(phi))
        h = torch.tanh(self.proj(phi))
        for b in self.blocks:
            h = b(h, U, V)
        return h

    def forward(self, x, t):
        return self.out(self._features(x, t))

    @torch.no_grad()
    def physics_informed_init(self, x, t, y):
        """Least-squares-fit the (zero-initialised) output layer to `y`.

        PirateNet zero-initialises `out` on purpose, but that is only valid when
        followed by this LSQ step.  Without it the network is identically zero at
        init: u_tt == 0 and every backbone parameter is dead (caught by
        test_models.py sections 3-5).
        """
        self.eval()
        Phi = self._features(x, t)
        W = torch.linalg.lstsq(Phi, y).solution
        self.out.weight.copy_(W.T)
        self.train()


# ─────────────────────────────── KANs ────────────────────────────────
class _SplineKANLayer(nn.Module):
    def __init__(self, i, o, grid_size=10, order=4):
        super().__init__()
        self.n_basis = grid_size + order; self.order = order
        self.base = nn.Parameter(torch.empty(o, i)); nn.init.kaiming_uniform_(self.base, a=math.sqrt(5))
        self.coef = nn.Parameter(torch.empty(o, i, self.n_basis)); nn.init.xavier_uniform_(self.coef)
        g = torch.linspace(-1, 1, grid_size + 1); step = g[1] - g[0]
        self.register_buffer("grid", torch.cat([
            torch.linspace(g[0] - order * step, g[0] - step, order), g,
            torch.linspace(g[-1] + step, g[-1] + order * step, order)]))

    def _bspline(self, x):
        x = x.unsqueeze(-1); g = self.grid
        b = ((x >= g[:-1]) & (x < g[1:])).to(x.dtype)
        for k in range(1, self.order + 1):
            b = ((x - g[:-k-1]) / (g[k:-1] - g[:-k-1]) * b[:, :, :-1]
                 + (g[k+1:] - x) / (g[k+1:] - g[1:-k]) * b[:, :, 1:])
        return b.contiguous()

    def forward(self, x):
        return F.linear(F.silu(x), self.base) + torch.einsum("bik,oik->bo", self._bspline(x), self.coef)


class SplineKAN(nn.Module):
    """B-spline KAN on the shared Fourier embedding (the canonical 'PIKAN')."""

    def __init__(self, layers: int = 3, units: int = 64, grid_size: int = 10,
                 n_fourier: int = 64, sigma: float = SIGMA_B):
        super().__init__()
        self.embed = FourierEmbed(n_fourier, sigma)
        self.l0 = _SplineKANLayer(self.embed.out_dim, units, grid_size)
        self.n0 = nn.LayerNorm(units)
        self.mid = nn.ModuleList([_SplineKANLayer(units, units, grid_size) for _ in range(layers - 1)])
        self.norms = nn.ModuleList([nn.LayerNorm(units) for _ in range(layers - 1)])
        self.out = nn.Linear(units, 1, bias=False); nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x, t):
        h = torch.tanh(self.n0(self.l0(self.embed(x, t))))
        for l, n in zip(self.mid, self.norms):
            h = torch.tanh(n(l(h)))
        return self.out(h)


class PyKAN(nn.Module):
    """Spline KAN backed by the REFERENCE implementation (pykan), not a hand-rolled
    layer.  Operates on raw (x,t) -- the design used by arXiv:2602.15068, whose
    650-parameter config `ML/PIKAN.py` was adapted from.

    Rationale: our own `_SplineKANLayer` on a 128-d Fourier embedding produced
    max|u_tt| ~ 5e5 at init against the Fourier PINN's 1.8e2, and collapsed under
    the physics loss.  pykan is the implementation the published numbers were
    obtained with, so it removes our layer as a variable.
    """

    def __init__(self, width=(2, 5, 5, 5, 1), grid=5, k=3, seed=None,
                 ckpt="/tmp/_pykan_ckpt"):
        super().__init__()
        from kan import KAN
        from . import kan_patch
        # Two defects in pykan's initialisation, both measured:
        #
        # 1. SEED HIJACK.  MultKAN.__init__ calls torch.manual_seed(its own
        #    `seed`, default 1), overriding whatever the trainer set.  Measured:
        #    building under torch.manual_seed(0) and torch.manual_seed(2) gave
        #    parameters differing by 1.49e-08 -- the same magnitude as building
        #    twice under the SAME seed.  So the caller's seed never reached the
        #    model, and a multi-seed sweep varied only the collocation set.
        #    Fix: draw pykan's seed FROM the ambient RNG, which the trainer has
        #    already seeded, so distinct trainer seeds give distinct inits.
        #
        # 2. NON-DETERMINISM.  coef comes from curve2coef, whose torch.linalg.lstsq
        #    is not bitwise reproducible (2.2e-8 between identical builds).  Over
        #    ~585 L-BFGS epochs that amplified to a 6% relative difference in the
        #    final error (0.142% vs 0.151% on the same cell).  The ridge solve is
        #    exactly reproducible (0.0e+00).
        kan_patch.install()
        if seed is None:
            seed = int(torch.randint(0, 2 ** 31 - 1, (1,)).item())
        self.seed = int(seed)
        self.kan = KAN(width=list(width), grid=grid, k=k, seed=self.seed,
                       symbolic_enabled=False, auto_save=False, ckpt_path=ckpt)
        # symbolic_enabled=False means the symbolic branch never enters forward(),
        # but pykan still registers its `affine` tensors as requires_grad leaves.
        # They receive no gradient and never move, yet they were counted by
        # n_params (12,040 reported vs 8,600 real for pykan_wide -- 28.6% phantom)
        # and padded the L-BFGS history with permanent zeros.  Freeze them so the
        # parameter counts are honest and the optimiser state is minimal.
        for _m in self.kan.symbolic_fun:
            for _q in _m.parameters():
                _q.requires_grad_(False)

    def forward(self, x, t):
        return self.kan(torch.cat([x, t], dim=-1))


class _WavKANLayer(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(o, i))
        self.trans = nn.Parameter(torch.zeros(o, i))
        self.w = nn.Parameter(torch.empty(o, i)); nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))

    def forward(self, x):
        z = (x.unsqueeze(1) - self.trans.unsqueeze(0)) / self.scale.unsqueeze(0)
        psi = (2 / (math.sqrt(3) * math.pi ** 0.25)) * (z**2 - 1) * torch.exp(-0.5 * z**2)
        return (psi * self.w.unsqueeze(0)).sum(dim=2)


class WavKAN(nn.Module):
    """Mexican-hat wavelet KAN.  Operates on raw (x,t): the wavelet scales are
    themselves learnable, so it does not take the Fourier embedding."""

    def __init__(self, layers: int = 7, units: int = 32):
        super().__init__()
        dims = [2] + [units] * layers + [1]
        self.layers = nn.ModuleList([_WavKANLayer(a, b) for a, b in zip(dims[:-1], dims[1:])])

    def forward(self, x, t):
        h = torch.cat([x, t], dim=-1)
        for l in self.layers:
            h = l(h)
        return h


REGISTRY = {
    "mlp":        lambda: MLP(5, 128),
    "fourier":    lambda: FourierMLP(5, 128, 64, SIGMA_B),
    "pirate":     lambda: PirateNet(3, 128, 64, SIGMA_B),
    "splinekan":  lambda: SplineKAN(3, 64, 10, 64, SIGMA_B),   # hand-rolled; see CORRECTIONS C7b
    "pykan":      lambda: PyKAN((2, 5, 5, 5, 1), 5, 3),        # reference implementation
    "pykan_wide": lambda: PyKAN((2, 20, 20, 20, 1), 5, 3),
    "wavkan":     lambda: WavKAN(7, 32),
}


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
