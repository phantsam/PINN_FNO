"""Frozen problem specification for the 1D elastic wave equation.

Every architecture (PINN, KAN, ...) must import this module and may not
override any of it.  The *model* is interchangeable; the *problem* is not.

    PDE        rho(x) u_tt = d/dx( E(x) u_x )      x in [x_min,x_max], t in [0,T]
    IC         u(x,0) = g(x),  u_t(x,0) = 0
    BC         u_t -+ c(x) u_x = 0   (outgoing) at x_min / x_max
    metric     fixed-denominator relative L2 over a stated (x,t) window
"""
from __future__ import annotations
import math
import numpy as np
import torch

ArrayLike = "np.ndarray | torch.Tensor"


# ─────────────────────────────────────────────────────────────
# Materials.  E and rho are non-dimensional; x is non-dimensional.
# ─────────────────────────────────────────────────────────────
class Material:
    """Base class.  Subclasses define E(x) and rho(x) on [x_min, x_max]."""

    name: str = "base"

    def __init__(self, x_min: float, x_max: float, E_ref: float, rho_ref: float):
        self.x_min, self.x_max = float(x_min), float(x_max)
        self.E_ref, self.rho_ref = float(E_ref), float(rho_ref)

    # -- material fields (must accept torch tensors AND numpy arrays) --
    def E(self, x):
        raise NotImplementedError

    def rho(self, x):
        raise NotImplementedError

    def c(self, x):
        """Local wave speed sqrt(E/rho)."""
        E, r = self.E(x), self.rho(x)
        return torch.sqrt(E / r) if torch.is_tensor(x) else np.sqrt(E / r)

    @property
    def c_max(self) -> float:
        xs = np.linspace(self.x_min, self.x_max, 4001)
        return float(np.max(self.c(xs)))

    @property
    def interfaces(self) -> list[float]:
        """x-positions of material transitions (empty if homogeneous)."""
        return []

    def __repr__(self):
        return f"<{self.name} x=[{self.x_min},{self.x_max}] c_max={self.c_max:.4f}>"


def _ones_like(x, v):
    return torch.full_like(x, v) if torch.is_tensor(x) else np.full_like(np.asarray(x, float), v)


def _tanh(x):
    return torch.tanh(x) if torch.is_tensor(x) else np.tanh(x)


class Homogeneous(Material):
    name = "Homogeneous"

    def __init__(self):
        super().__init__(-1.0, 1.0, E_ref=80.0, rho_ref=100.0)

    def E(self, x):
        return _ones_like(x, 1.0)

    def rho(self, x):
        return _ones_like(x, 1.0)


class TwoLayer(Material):
    """E steps 1.0 -> 1.5 at x=0, smoothed over transition width `w`."""

    name = "TwoLayer"

    def __init__(self, w: float = 0.02):
        super().__init__(-1.0, 1.0, E_ref=80.0, rho_ref=100.0)
        self.w = float(w)
        self._E1, self._E2 = 1.0, 1.5

    def E(self, x):
        a = 0.5 * (1.0 + _tanh(x / self.w))
        return self._E1 * (1.0 - a) + self._E2 * a

    def rho(self, x):
        return _ones_like(x, 1.0)

    @property
    def interfaces(self):
        return [0.0]


class MultiLayer(Material):
    """6 layers, E from 1.0 to 2.5, each transition smoothed over width `w`."""

    name = "MultiLayer"

    def __init__(self, w: float = 0.05, n_layers: int = 6):
        # w=0.05 per DECISIONS.md D3: literature-equivalent is 0.067
        # (arXiv:2305.05150 uses transition/layer = 0.2); main's 0.02 was ~3x sharper.
        super().__init__(-1.0, 1.0, E_ref=60.0, rho_ref=100.0)
        self.w = float(w)
        self.n_layers = int(n_layers)
        self.E_vals = np.linspace(60.0, 150.0, self.n_layers) / 60.0

    @property
    def interfaces(self):
        lw = (self.x_max - self.x_min) / self.n_layers
        return [self.x_min + (k + 1) * lw for k in range(self.n_layers - 1)]

    def E(self, x):
        val = _ones_like(x, float(self.E_vals[0]))
        for k, b in enumerate(self.interfaces):
            a = 0.5 * (1.0 + _tanh((x - b) / self.w))
            val = val * (1.0 - a) + float(self.E_vals[k + 1]) * a
        return val

    def rho(self, x):
        return _ones_like(x, 1.0)


class VariableDensity(Material):
    """rho(x) AND E(x) both vary.  Exists so the rho half of the PDE is actually
    covered: with rho == 1 everywhere (as in the other three materials) a bug that
    drops or inverts rho is undetectable by construction."""

    name = "VariableDensity"

    def __init__(self, w: float = 0.05):
        super().__init__(-1.0, 1.0, E_ref=80.0, rho_ref=100.0)
        self.w = float(w)

    def E(self, x):
        return 1.0 + 0.5 * (1.0 + _tanh(x / self.w))          # 1.0 -> 2.0

    def rho(self, x):
        # smooth, strictly positive, genuinely non-constant
        if torch.is_tensor(x):
            return 1.0 + 0.4 * torch.sin(2.0 * x) + 0.2 * torch.cos(3.0 * x)
        xa = np.asarray(x, float)
        return 1.0 + 0.4 * np.sin(2.0 * xa) + 0.2 * np.cos(3.0 * xa)

    @property
    def interfaces(self):
        return [0.0]


MATERIALS = {"homogeneous": Homogeneous, "twolayer": TwoLayer,
             "multilayer": MultiLayer, "variabledensity": VariableDensity}


# ─────────────────────────────────────────────────────────────
# Initial condition
# ─────────────────────────────────────────────────────────────
def gaussian_ic(x, sigma_g: float = 0.1, x0: float = 0.0, normalize: bool = True):
    """Derivative of a Gaussian, peak-normalised to max|g| = 1.

    `normalize` uses the *analytic* peak, not the sampled max, so the IC is a
    fixed function of x.  Note: a sampled max does NOT measurably break the grid
    convergence study (measured 2.00/1.99/2.01 either way -- the perturbation is
    ~30x below discretisation error).  The real reason is training time: a PINN
    draws RANDOM collocation batches, where a sampled max would vary per batch
    and silently change the IC from step to step.
    """
    if torch.is_tensor(x):
        f = torch.exp(-0.5 * ((x - x0) / sigma_g) ** 2)
    else:
        f = np.exp(-0.5 * ((np.asarray(x, float) - x0) / sigma_g) ** 2)
    dfdx = -(x - x0) / sigma_g**2 * f
    if not normalize:
        return dfdx
    # analytic peak of |d/dx exp(-x^2/2s^2)| is at x = +-s, value = 1/(s*sqrt(e))
    peak = 1.0 / (sigma_g * math.sqrt(math.e))
    return dfdx / peak


def gaussian_ic_dx(x, sigma_g: float = 0.1, x0: float = 0.0):
    """Analytic d/dx of gaussian_ic (needed for the stress IC and for MMS)."""
    s2 = sigma_g**2
    if torch.is_tensor(x):
        f = torch.exp(-0.5 * ((x - x0) / sigma_g) ** 2)
    else:
        f = np.exp(-0.5 * ((np.asarray(x, float) - x0) / sigma_g) ** 2)
    d2 = ((x - x0) ** 2 / s2 - 1.0) / s2 * f
    peak = 1.0 / (sigma_g * math.sqrt(math.e))
    return d2 / peak
