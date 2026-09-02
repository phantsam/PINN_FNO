"""Pin the problem SPECIFICATION against independent literals.

The MMS validates the OPERATOR.  It cannot validate the SPEC, because its
symbolic E/rho are built by reading the same attributes they would need to check
(a mirror compared to its object).  Every constant below is written out by hand
from the intended problem definition, so a changed interface position, stiffness
value, wave speed or IC amplitude fails here.

Closes audit misses #3 (interface off-by-one), #4 (wrong E_vals),
#5 (absolute stiffness x2 with impedance ratio preserved), #13 (IC peak).
"""
import sys
import numpy as np, torch
from core.problem import (Homogeneous, TwoLayer, MultiLayer, VariableDensity,
                          gaussian_ic, gaussian_ic_dx)

R = []
def check(n, ok, d=""):
    R.append(ok); print(f"  [{'PASS' if ok else 'FAIL'}] {n:<54} {d}")

def close(a, b, tol=1e-12):
    return bool(np.all(np.abs(np.asarray(a) - np.asarray(b)) < tol))

print("\n1. Domain")
for M, lo, hi in ((Homogeneous(), -1.0, 1.0), (TwoLayer(), -1.0, 1.0),
                  (MultiLayer(), -1.0, 1.0), (VariableDensity(), -1.0, 1.0)):
    check(f"{M.name} domain == [{lo},{hi}]", M.x_min == lo and M.x_max == hi)

print("\n2. Absolute stiffness values (NOT just impedance ratios)")
check("Homogeneous E == 1 everywhere", close(Homogeneous().E(np.linspace(-1, 1, 51)), 1.0))
T = TwoLayer()
check("TwoLayer E(-0.5) == 1.0", close(T.E(np.array([-0.5])), 1.0, 1e-9))
check("TwoLayer E(+0.5) == 1.5", close(T.E(np.array([0.5])), 1.5, 1e-9))
check("TwoLayer E(0) == 1.25 (midpoint)", close(T.E(np.array([0.0])), 1.25, 1e-12))
check("TwoLayer c_max == sqrt(1.5) == 1.2247449", abs(T.c_max - 1.2247448714) < 1e-6,
      f"{T.c_max:.7f}")
M6 = MultiLayer()
check("MultiLayer E_vals == [1.0,1.3,1.6,1.9,2.2,2.5]",
      close(M6.E_vals, [1.0, 1.3, 1.6, 1.9, 2.2, 2.5], 1e-12),
      np.array2string(np.round(M6.E_vals, 3)))
check("MultiLayer c_max == sqrt(2.5) == 1.5811388", abs(M6.c_max - 1.5811388) < 1e-5,
      f"{M6.c_max:.7f}")

print("\n3. Interface positions")
check("Homogeneous has no interfaces", MultiLayer() and Homogeneous().interfaces == [])
check("TwoLayer interface == [0.0]", close(TwoLayer().interfaces, [0.0]))
check("MultiLayer interfaces == [-2/3,-1/3,0,1/3,2/3]",
      close(M6.interfaces, [-2/3, -1/3, 0.0, 1/3, 2/3], 1e-12),
      np.array2string(np.round(M6.interfaces, 4)))
check("no interface sits on the domain boundary",
      all(M.x_min < b < M.x_max for M in (T, M6) for b in M.interfaces))

print("\n4. Density")
for M in (Homogeneous(), TwoLayer(), MultiLayer()):
    check(f"{M.name} rho == 1", close(M.rho(np.linspace(-1, 1, 51)), 1.0))
V = VariableDensity()
r = V.rho(np.linspace(-1, 1, 2001))
check("VariableDensity rho is non-constant", float(np.ptp(r)) > 0.5, f"ptp={np.ptp(r):.3f}")
check("VariableDensity rho strictly positive", float(np.min(r)) > 0.1, f"min={np.min(r):.3f}")

print("\n5. Initial condition amplitude and shape")
xs = np.linspace(-1, 1, 200001)
g = gaussian_ic(xs, 0.1)
check("max|g| == 1 (analytic peak normalisation)", abs(np.max(np.abs(g)) - 1.0) < 1e-6,
      f"{np.max(np.abs(g)):.8f}")
check("g is odd about x0", close(g[::-1], -g, 1e-9))
check("g extrema at x = +-sigma", abs(xs[np.argmax(g)] + 0.1) < 1e-3 and
      abs(xs[np.argmin(g)] - 0.1) < 1e-3, f"argmax={xs[np.argmax(g)]:+.4f}")
check("g(0) == 0", abs(np.asarray(gaussian_ic(np.array([0.0]), 0.1)).item()) < 1e-15)
# gaussian_ic_dx must be the true derivative (zero coverage before -- miss #14)
d_num = np.gradient(g, xs)
d_an = gaussian_ic_dx(xs, 0.1)
inner = np.abs(xs) < 0.8
check("gaussian_ic_dx == d/dx gaussian_ic",
      float(np.max(np.abs(d_num[inner] - d_an[inner]))) < 1e-4,
      f"max diff {np.max(np.abs(d_num[inner]-d_an[inner])):.2e}")
_d0 = np.asarray(gaussian_ic_dx(np.array([0.0]), 0.1)).item()
check("gaussian_ic_dx sign correct at x=0 (must be NEGATIVE)", _d0 < 0, f"{_d0:.3f}")

print("\n6. numpy / torch paths agree (miss #10: silently different problems)")
xt = np.linspace(-1, 1, 1001)
for M in (Homogeneous(), TwoLayer(), MultiLayer(), VariableDensity()):
    for fn in ("E", "rho", "c"):
        a = np.asarray(getattr(M, fn)(xt), float)
        b = getattr(M, fn)(torch.tensor(xt, dtype=torch.float64)).numpy()
        check(f"{M.name}.{fn} numpy==torch", close(a, b, 1e-14))
gn = np.asarray(gaussian_ic(xt, 0.1), float)
gt = gaussian_ic(torch.tensor(xt, dtype=torch.float64), 0.1).numpy()
check("gaussian_ic numpy==torch", close(gn, gt, 1e-14))
dn = np.asarray(gaussian_ic_dx(xt, 0.1), float)
dt_ = gaussian_ic_dx(torch.tensor(xt, dtype=torch.float64), 0.1).numpy()
check("gaussian_ic_dx numpy==torch", close(dn, dt_, 1e-14))

print("\n" + "=" * 76)
nf = len(R) - sum(R)
print(f"{sum(R)}/{len(R)} passed" + ("" if nf == 0 else f"   *** {nf} FAILURES ***"))
sys.exit(1 if nf else 0)
