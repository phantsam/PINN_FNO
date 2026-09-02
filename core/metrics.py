"""Evaluation metric with a FIXED denominator.

The repo's metric averages per-snapshot relative errors
    mean_k  ||u_pred(.,t_k) - u_ref(.,t_k)|| / ||u_ref(.,t_k)||
whose denominator DRIFTS: ||u_ref(.,t)|| falls to ~0.18 of its initial value by
t=1 on MultiLayer, so an identical absolute error scores ~4x worse late.  The
universally observed "error grows with time" is therefore partly an artefact.

Headline metric here is a single space-time relative L2 with one fixed
denominator.  The per-snapshot curve is kept, clearly labelled as a diagnostic.
"""
from __future__ import annotations
import numpy as np


def spacetime_rel_l2(u_pred: np.ndarray, u_ref: np.ndarray) -> float:
    """||u_pred - u_ref||_{L2(x,t)} / ||u_ref||_{L2(x,t)}  in percent."""
    den = float(np.linalg.norm(u_ref))
    return 100.0 * float(np.linalg.norm(u_pred - u_ref)) / max(den, 1e-30)


def per_snapshot_rel_l2(u_pred: np.ndarray, u_ref: np.ndarray) -> np.ndarray:
    """Diagnostic only -- drifting denominator, do not use as a headline."""
    num = np.linalg.norm(u_pred - u_ref, axis=1)
    den = np.linalg.norm(u_ref, axis=1)
    return 100.0 * num / np.maximum(den, 1e-30)


def per_snapshot_fixed_den(u_pred: np.ndarray, u_ref: np.ndarray) -> np.ndarray:
    """Per-time error against a FIXED denominator (the t=0 reference norm)."""
    den = np.linalg.norm(u_ref[0])
    return 100.0 * np.linalg.norm(u_pred - u_ref, axis=1) / max(den, 1e-30)


def amplitude_norm_ratio(u_pred: np.ndarray, u_ref: np.ndarray) -> np.ndarray:
    """||u_pred(.,t)|| / ||u_ref(.,t)||.  NOT an energy -- no rho or E weighting.

    Must be checked TWO-SIDED.  A one-sided "< 0.3" test catches collapse-to-zero
    (the legacy ansatz's failure mode) but is completely silent on a FROZEN IC
    (the poly ansatz's failure mode, N -> 0), which gives ratio 1.0 rising to ~5.9
    at ~188% error.  Flag outside [LO, HI].
    """
    return (np.linalg.norm(u_pred, axis=1)
            / np.maximum(np.linalg.norm(u_ref, axis=1), 1e-30))


# backwards-compatible alias; the name "energy" was a misnomer
energy_norm_ratio = amplitude_norm_ratio

RATIO_LO, RATIO_HI = 0.30, 1.50


def discrete_energy(u: np.ndarray, x: np.ndarray, t: np.ndarray, material) -> np.ndarray:
    """Reference-free monitor: 1/2 integral( rho u_t^2 + E u_x^2 ) dx.

    Needs no ground truth, so it can be watched during training.
    """
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    E = np.asarray(material.E(x), float)
    rho = np.asarray(material.rho(x), float)
    ut = np.gradient(u, dt, axis=0)
    ux = np.gradient(u, dx, axis=1)
    return 0.5 * np.sum(rho * ut**2 + E * ux**2, axis=1) * dx
