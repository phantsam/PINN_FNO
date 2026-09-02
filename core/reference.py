"""Verified finite-difference reference solver.

Second-order in space AND time:
  * conservative flux stencil   L(u)_i = (F_{i+1/2} - F_{i-1/2}) / (rho_i dx),
                                F_{i+1/2} = E_{i+1/2} (u_{i+1}-u_i)/dx
  * leapfrog                    u^{n+1} = 2u^n - u^{n-1} + dt^2 L(u^n)
  * Taylor start-up             u^1 = u^0 + dt*v0 + 0.5 dt^2 L(u^0)     <-- 2nd order
  * Mur first-order ABC at both ends

Two bugs present in the original implementations are fixed here:
  (1) dt = T/nt  while  t = linspace(0,T,nt)  has spacing T/(nt-1)   -> every
      frame was mislabelled in time by a factor nt/(nt-1).
  (2) u^1 = u^0  is only FIRST-order accurate; it caps the whole scheme at
      order 1 regardless of the spatial stencil.
"""
from __future__ import annotations
import numpy as np
from .problem import Material, gaussian_ic

CFL_SAFETY = 0.45


def _laplacian(u, E_half, rho, dx):
    """Conservative (1/rho) d/dx (E du/dx); returns interior values only."""
    flux = E_half * (u[1:] - u[:-1]) / dx          # at i+1/2, length nx-1
    return (flux[1:] - flux[:-1]) / (rho[1:-1] * dx)


def fd_reference(material: Material, nx: int = 512, nt: int | None = None,
                 T: float = 1.0, sigma_g: float = 0.1, x0: float = 0.0,
                 cfl: float = CFL_SAFETY, dtype=np.float64):
    """Return (x, t, u) with u.shape == (nt, nx).

    If `nt` is None it is chosen from the CFL condition, so refining `nx` alone
    refines `dt` proportionally (needed for a clean joint-refinement study).
    """
    x = np.linspace(material.x_min, material.x_max, nx, dtype=dtype)
    dx = x[1] - x[0]

    E = np.asarray(material.E(x), dtype=dtype)
    rho = np.asarray(material.rho(x), dtype=dtype)
    E_half = 0.5 * (E[:-1] + E[1:])
    c_max = float(np.max(np.sqrt(E / rho)))

    if nt is None:
        nt = int(np.ceil(T / (cfl * dx / c_max))) + 1
    t = np.linspace(0.0, T, nt, dtype=dtype)
    dt = t[1] - t[0]                                   # FIX (1): consistent with t

    if dt > dx / c_max:
        raise ValueError(f"CFL violated: dt={dt:.3e} > dx/c={dx/c_max:.3e} "
                         f"(nx={nx}, nt={nt}). Increase nt.")

    u = np.zeros((nt, nx), dtype=dtype)
    u0 = np.asarray(gaussian_ic(x, sigma_g, x0), dtype=dtype)
    u[0] = u0

    # FIX (2): second-order Taylor start-up, u_t(x,0) = 0
    u1 = u0.copy()
    u1[1:-1] = u0[1:-1] + 0.5 * dt**2 * _laplacian(u0, E_half, rho, dx)
    _mur(u1, u0, u0, dx, dt, np.sqrt(E[0] / rho[0]), np.sqrt(E[-1] / rho[-1]))
    u[1] = u1

    for n in range(1, nt - 1):
        un, um = u[n], u[n - 1]
        nxt = np.zeros_like(un)
        nxt[1:-1] = 2 * un[1:-1] - um[1:-1] + dt**2 * _laplacian(un, E_half, rho, dx)
        _mur(nxt, un, um, dx, dt, np.sqrt(E[0] / rho[0]), np.sqrt(E[-1] / rho[-1]))
        u[n + 1] = nxt

    return x, t, u


def _mur(new, cur, prev, dx, dt, cL, cR):
    """First-order Mur absorbing BC, applied in place to `new`."""
    kL = (cL * dt - dx) / (cL * dt + dx)
    kR = (cR * dt - dx) / (cR * dt + dx)
    new[0] = cur[1] + kL * (new[1] - cur[0])
    new[-1] = cur[-2] + kR * (new[-2] - cur[-1])


def dalembert(x, t, sigma_g=0.1, x0=0.0, c=1.0):
    """Exact homogeneous solution: u = 1/2[g(x-ct) + g(x+ct)] (free space)."""
    return 0.5 * (gaussian_ic(x - c * t, sigma_g, x0) +
                  gaussian_ic(x + c * t, sigma_g, x0))
