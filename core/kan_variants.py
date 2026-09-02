"""Strengthened KAN family for the Phase 8 ladder.

Phase 6 ran pykan at its library defaults and lost to the Fourier PINN.  That
comparison was not fair, and the unfairness is measurable:

  * layer-0 knot spacing was 0.400 while the Gaussian pulse FWHM is 0.236 -- the
    splines were COARSER THAN THE ENTIRE WAVE PACKET.  Representing the measured
    k_peak ~ 9.4 at 4 knots/wavelength needs grid >= 13; resolving the pulse
    shape needs grid ~ 33.  We ran grid=5.
  * grid_range defaults to [-1,1] but t lies in [0,1], so half the knots in the
    time direction were never reachable.
  * meanwhile the PINN arm got sigma_B = 10 fitted to that same measured
    spectrum (DECISIONS.md D4).

This module exposes each of those as an explicit, independently-varied knob so
the ladder can attribute any change to one cause at a time.

Every variant keeps the project-wide contract: forward(x, t) -> raw output, with
the hard-IC ansatz applied outside by the trainer.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

from .models import FourierEmbed, SIGMA_B
from .coords import CharCoords
from . import kan_patch

# pykan's curve2coef uses torch.linalg.lstsq, whose only CUDA driver (gels)
# assumes full rank and returns NaN -- silently -- otherwise.  KAN activations
# contract with depth (measured: layer 3 visits only [-0.09, 0.07] of its [-1,1]
# grid), so every refine() past layer 0 produced NaN coefficients on GPU.  It is
# also numerically unstable at k=5 even at full rank: LSQ residual 58.65 vs the
# ridge solve's 0.0022 on an identical system.  Both are fatal to this ladder,
# since grid refinement and k=5 are two of the rungs.  See kan_patch.py.
_N_PATCHED = kan_patch.install()


class Sin(nn.Module):
    """SIREN-style sine. For a travelling wave this matches the solution's own
    structure, and unlike ReLU its derivatives never vanish or jump."""
    def __init__(self, w0: float = 1.0):
        super().__init__(); self.w0 = float(w0)
    def forward(self, x):
        return torch.sin(self.w0 * x)


# pykan itself only understands 'silu' | 'identity' | 'zero'; any other string is
# passed through untouched and then *called*, which raises inside KANLayer.  So
# every base function here is installed by post-construction substitution.
BASE_FUNS = {
    "silu":     lambda: nn.SiLU(),      # pykan default
    "tanh":     lambda: nn.Tanh(),      # the standard PINN activation
    "sin":      lambda: Sin(1.0),       # SIREN-style, matches wave structure
    "gelu":     lambda: nn.GELU(),
    "identity": lambda: nn.Identity(),
    # ReLU is included as a THEORY CONTROL, expected to underperform: its first
    # derivative jumps at every kink, so u_x inherits those jumps and the u_xx
    # the residual is built on is zero a.e. but Dirac-singular at the kinks.  The
    # spline branch still supplies curvature, so this handicaps rather than
    # destroys -- which is exactly what makes it a useful control.
    "relu":     lambda: nn.ReLU(),
}


class _Affine(nn.Module):
    """Map the physical domain onto pykan's [-1,1] grid_range, exactly.

    x in [x_min,x_max] -> [-1,1];  t in [0,t_max] -> [-1,1].

    Without this the time direction wastes half its knots.  The transform is
    affine, so autograd rescales u_t and u_tt by the constant Jacobian and the
    residual is unaffected apart from the intended resolution gain.
    """
    def __init__(self, x_min, x_max, t_max):
        super().__init__()
        self.register_buffer("xs", torch.tensor(2.0 / (x_max - x_min)))
        self.register_buffer("xo", torch.tensor(-(x_max + x_min) / (x_max - x_min)))
        self.register_buffer("ts", torch.tensor(2.0 / t_max))
    def forward(self, x, t):
        return torch.cat([x * self.xs + self.xo, t * self.ts - 1.0], dim=-1)


class TunedKAN(nn.Module):
    """pykan with every Phase 8 knob exposed.

    Parameters
    ----------
    width        : layer widths, e.g. (2, 20, 20, 20, 1)
    grid         : spline intervals per activation (the resolution knob)
    k            : spline order.  k=3 makes u C^2, so u_xx -- which the residual
                   consumes -- is only C^0, piecewise linear with kinks at every
                   knot.  k=5 makes u C^4 and u_xx C^2.  For a second-order PDE
                   this is a principled change, not a tweak.
    base_fun     : key into BASE_FUNS
    normalise    : apply _Affine so both inputs fill [-1,1]
    n_fourier    : if > 0, prepend the shared Fourier embedding.  cos/sin land in
                   [-1,1] natively, which is exactly pykan's grid_range, so no
                   further scaling is needed and `normalise` is bypassed.
    """

    def __init__(self, width=(2, 20, 20, 20, 1), grid=5, k=3, base_fun="silu",
                 normalise=True, x_min=-1.0, x_max=1.0, t_max=1.0,
                 n_fourier=0, sigma=SIGMA_B, char_coords=False, material=None,
                 grid_eps=0.02, noise_scale=0.3, seed=1, ckpt="/tmp/_pykan_ckpt"):
        super().__init__()
        from kan import KAN
        if char_coords and material is None:
            raise ValueError("char_coords=True needs the material to build tau(x)")
        if char_coords and n_fourier:
            raise ValueError("char_coords and n_fourier are alternative embeddings")
        # Characteristic coordinates (tau(x)-t, tau(x)+t) already land in [-1,1],
        # as do the Fourier cos/sin pair, so both bypass the affine scaler.
        self.chars = CharCoords(material, t_max=t_max) if char_coords else None
        self.embed = FourierEmbed(n_fourier, sigma) if n_fourier else None
        in_dim = self.embed.out_dim if self.embed else 2
        width = (in_dim,) + tuple(width)[1:]
        self.scaler = (None if (self.embed or self.chars or not normalise)
                       else _Affine(x_min, x_max, t_max))
        self.cfg = dict(width=width, grid=grid, k=k, base_fun=base_fun,
                        n_fourier=n_fourier, normalise=normalise,
                        char_coords=char_coords,
                        grid_eps=grid_eps, noise_scale=noise_scale, seed=seed)
        self._ckpt = ckpt
        self.kan = KAN(width=list(width), grid=grid, k=k, base_fun="silu",
                       grid_eps=grid_eps, noise_scale=noise_scale, seed=seed,
                       symbolic_enabled=False, auto_save=False, ckpt_path=ckpt)
        self._install(base_fun)

    def _install(self, base_fun):
        """Swap the base branch and freeze the inert symbolic branch."""
        if base_fun not in BASE_FUNS:
            raise KeyError(f"unknown base_fun {base_fun!r}; have {sorted(BASE_FUNS)}")
        fn = BASE_FUNS[base_fun]()
        self.kan.base_fun = fn
        self.kan.base_fun_name = base_fun
        for layer in self.kan.act_fun:
            layer.base_fun = fn
        # symbolic_enabled=False keeps this branch out of forward(), but pykan
        # still registers its `affine` tensors as requires_grad leaves.  Left
        # alone they inflate n_params by ~29% and pad every L-BFGS history vector
        # with permanent zeros.
        for m in self.kan.symbolic_fun:
            for q in m.parameters():
                q.requires_grad_(False)

    def to(self, *args, **kwargs):
        """Keep pykan's own device bookkeeping in sync.

        MultKAN and KANLayer each cache a `.device` attribute and define their
        own .to().  nn.Module.to() dispatches through _apply, which moves the
        tensors but never calls those overrides -- so `self.kan.device` stayed
        'cpu' after .to('cuda') and refine() then built the new grid on the CPU,
        raising "Expected all tensors to be on the same device".
        """
        super().to(*args, **kwargs)
        dev = torch._C._nn._parse_to(*args, **kwargs)[0]
        if dev is not None:
            self.kan.to(dev)
        return self

    def _inputs(self, x, t):
        if self.chars is not None:
            return self.chars(x, t)
        if self.embed is not None:
            return self.embed(x, t)
        if self.scaler is not None:
            return self.scaler(x, t)
        return torch.cat([x, t], dim=-1)

    def forward(self, x, t):
        return self.kan(self._inputs(x, t))

    # ── grid refinement (the KAN paper's own accuracy mechanism) ──────────
    @torch.no_grad()
    def refine_(self, new_grid, x, t):
        """Extend the spline grid in place, preserving the learnt function.

        pykan's refine() needs cached activations to place the new knots, and it
        returns a NEW module, so this re-points self.kan and re-installs the base
        branch and the symbolic freeze.  Coefficients are re-fitted by pykan so
        the represented function is preserved across the refinement.
        """
        prev = self.kan.save_act
        self.kan.save_act = True
        with torch.enable_grad():
            self.kan(self._inputs(x, t))          # populate the activation cache
        refined = self.kan.refine(new_grid)
        self.kan = refined
        self.kan.save_act = prev
        self.kan.symbolic_enabled = False
        self._install(self.cfg["base_fun"])
        self.cfg["grid"] = new_grid
        return self

    @torch.no_grad()
    def update_grid_(self, x, t):
        """Re-fit every layer's spline grid to the activations it actually sees.

        This is pykan's own default during fit() (update_grid=True, 10 updates
        over the first half of training) and our custom L-BFGS loop never called
        it.  It is the direct remedy for the grid/activation mismatch measured on
        a TRAINED network: layer 1 had overflowed its grid (activations reaching
        +3.06 against a last knot of +2.20, so those inputs got no spline response
        at all) while layer 3 used 31% of its range, ~3 knots, leaving the deepest
        layer nearly linear.  Measured effect of one call: layer-2 knots contract
        from [-2.20, 2.20] to [-0.57, 0.49] with 3e-3 function drift.
        """
        prev = self.kan.save_act
        self.kan.save_act = True
        self.kan.update_grid(self._inputs(x, t))
        self.kan.save_act = prev
        return self

    def set_save_act(self, flag: bool):
        """Activation caching costs memory on 10k collocation points and is only
        needed for refinement, so training turns it off."""
        self.kan.save_act = bool(flag)
        return self


def make(**kw):
    """Registry-friendly factory: `lambda: make(grid=20, k=5, ...)`."""
    return TunedKAN(**kw)
