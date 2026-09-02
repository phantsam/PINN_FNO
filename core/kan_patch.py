"""Numerically stable replacement for pykan's `curve2coef`.

The bug
-------
pykan refits B-spline coefficients by least squares whenever a grid is created
or refined.  Its own source shows the author was aware of the hazard:

    #coef = torch.linalg.lstsq(mat, y_eval, driver='gelsy' if device=='cpu' else 'gels')...
    try:
        coef = torch.linalg.lstsq(mat, y_eval).solution[:,:,:,0]
    except:
        print('lstsq failed')

On CUDA `torch.linalg.lstsq` offers only the `gels` driver, which assumes the
design matrix has FULL RANK.  Given a rank-deficient one it returns NaN and does
not raise -- so the `try/except` never fires and the NaN propagates into every
coefficient.

Why it fires here
-----------------
KAN activations contract with depth.  Measured on a (2,20,20,20,1) net at init:

    layer 0  acts in [-1.00, 1.00]  design rank 8 of 8   full rank
    layer 1  acts in [-0.71, 0.65]  design rank 7 of 8   deficient
    layer 2  acts in [-0.28, 0.22]  design rank 4 of 8   deficient
    layer 3  acts in [-0.09, 0.07]  design rank 4 of 8   deficient

Deep layers only ever visit a sliver of their [-1,1] grid, so most B-spline
basis columns are identically zero over the sample and the system is singular.
Every refine() past layer 0 therefore produced NaN coefficients on GPU.

The fix
-------
Solve the ridge-regularised normal equations instead:

    (X'X + lambda*mean(diag(X'X))*I) c = X'y

which is exactly the fallback pykan left commented out, with a scale-relative
lambda so the regularisation is invariant to the magnitude of the activations.
On a full-rank system this agrees with the unregularised solution to ~1e-7; on a
rank-deficient one it returns the minimum-norm solution instead of NaN.
"""
from __future__ import annotations
import torch
import kan.spline as _spline
from kan.spline import B_batch

# Captured at import, i.e. BEFORE install() runs.  Tests need the genuine pykan
# routine to demonstrate the bug; once install() has fired, every module-level
# `curve2coef` name -- kan.spline's included -- points at the replacement.
ORIGINAL_CURVE2COEF = _spline.curve2coef

RIDGE = 1e-8          # relative to mean(diag(X'X)); see agreement test in tests/


def curve2coef_stable(x_eval, y_eval, grid, k, lamb: float = RIDGE):
    """x_eval (batch,in_dim); y_eval (batch,in_dim,out_dim); grid (in_dim,G+2k+1).
    Returns coef (in_dim, out_dim, G+k) -- pykan's exact contract."""
    n_coef = grid.shape[1] - k - 1
    mat = B_batch(x_eval, grid, k).permute(1, 0, 2)          # (in, batch, n_coef)
    Y = y_eval.permute(1, 2, 0)                              # (in, out, batch)
    XtX = mat.transpose(1, 2) @ mat                          # (in, n_coef, n_coef)
    XtY = torch.einsum("ibn,iob->ion", mat, Y)               # (in, out, n_coef)
    scale = torch.diagonal(XtX, dim1=1, dim2=2).mean(dim=1).clamp_min(1e-30)
    eye = torch.eye(n_coef, device=mat.device, dtype=mat.dtype)
    A = XtX + (lamb * scale)[:, None, None] * eye
    coef = torch.linalg.solve(A, XtY.transpose(1, 2))        # (in, n_coef, out)
    return coef.transpose(1, 2).contiguous()                 # (in, out, n_coef)


def install():
    """Patch every module that bound the name via `from .spline import *`.

    Resolve through sys.modules, not attribute access: kan/__init__.py rebinds
    `kan.KANLayer` to the CLASS of that name, shadowing the module, so
    `hasattr(kan.KANLayer, "curve2coef")` is False and the real module -- the one
    whose global actually gets called -- would be left unpatched.
    """
    import sys
    import kan.spline, kan.KANLayer, kan.MultKAN      # ensure all are imported
    n = 0
    for name in ("kan.spline", "kan.KANLayer", "kan.MultKAN"):
        mod = sys.modules.get(name)
        if mod is not None and hasattr(mod, "curve2coef"):
            mod.curve2coef = curve2coef_stable
            n += 1
    if n < 3:
        raise RuntimeError(f"curve2coef patch reached only {n}/3 pykan modules")
    return n
