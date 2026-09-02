"""Coverage for coords.py -- the travel-time map tau(x) = int dx'/c(x').

tau feeds a residual built on u_xx, and by the chain rule that residual contains
tau'(x)^2 and tau''(x).  A map that merely *looks* smooth is not enough, so every
check below compares against the ANALYTIC derivative pointwise:

    tau'(x)  = 1/c(x)
    tau''(x) = -c'(x)/c(x)^2

Two numerical traps were found building this and both are pinned by regression
checks here, because both produced a plausible-looking but wrong tau'':
  * float32 cancellation in the quintic basis (error GREW with node count)
  * composite Simpson advancing odd indices by trapezoid, leaving adjacent nodes
    inconsistent at O(h^3) and collapsing tau'' to first-order accuracy
"""
import sys
import torch

from core.problem import MATERIALS
from core.coords import TravelTime, CharCoords

R = []
def check(n, ok, d=""):
    R.append(bool(ok)); print(f"  [{'PASS' if ok else 'FAIL'}] {n:<56} {d}")

MATS = ["homogeneous", "twolayer", "multilayer", "variabledensity"]


def analytic(M, x):
    xg = x.detach().clone().requires_grad_(True)
    c = M.c(xg)
    if torch.is_tensor(c) and c.requires_grad and c.grad_fn is not None:
        cp = torch.autograd.grad(c.sum(), xg, allow_unused=True)[0]
        cp = torch.zeros_like(xg) if cp is None else cp
    else:
        cp = torch.zeros_like(xg)
    c = torch.as_tensor(c)
    c = c.detach().expand_as(xg) if c.numel() == 1 else c.detach()
    return 1.0 / c, -cp.detach() / c ** 2


def derivs(tt, M, n_pts=20000):
    x = torch.linspace(M.x_min + 1e-4, M.x_max - 1e-4, n_pts).unsqueeze(1).requires_grad_(True)
    d1 = torch.autograd.grad(tt(x).sum(), x, create_graph=True)[0]
    d2 = torch.autograd.grad(d1.sum(), x)[0]
    return x, d1.detach(), d2.detach()


print("\n1. tau' reproduces 1/c(x) pointwise")
for mk in MATS:
    M = MATERIALS[mk](); tt = TravelTime(M, n=4097)
    x, d1, _ = derivs(tt, M)
    r1, _ = analytic(M, x)
    e = float(((d1 - r1).abs() / r1.abs()).max())
    check(f"{mk:<16} max rel err on tau'", e < 1e-5, f"{e:.3e}")

print("\n2. tau'' reproduces -c'(x)/c(x)^2 pointwise")
for mk in MATS:
    M = MATERIALS[mk](); tt = TravelTime(M, n=4097)
    x, _, d2 = derivs(tt, M)
    _, r2 = analytic(M, x)
    scale = max(float(r2.abs().max()), 1e-12)
    e = float((d2 - r2).abs().max())
    check(f"{mk:<16} max abs err on tau''", e < 1e-3 * max(scale, 1.0),
          f"{e:.3e} vs peak {scale:.4f}")

print("\n3. Regression: tau'' accuracy must NOT degrade as nodes are added")
# float32 cancellation made the error grow with n (5.8e-2 at 4097, 1.3e-1 at
# 16385).  With float64 evaluation it must stay flat or improve.
M = MATERIALS["twolayer"]()
errs = []
for n in (1025, 4097, 16385):
    tt = TravelTime(M, n=n)
    x, _, d2 = derivs(tt, M, n_pts=20000)
    _, r2 = analytic(M, x)
    errs.append(float((d2 - r2).abs().max()))
check("error does not grow with node count", errs[-1] <= errs[0] * 1.5,
      f"n=1025 {errs[0]:.2e} -> n=16385 {errs[2]:.2e}")
check("all node counts are accurate", max(errs) < 1e-3, f"max {max(errs):.2e}")

print("\n4. Closed form for constant wave speed")
M = MATERIALS["homogeneous"]()
tt = TravelTime(M, n=4097)
xs = torch.linspace(M.x_min, M.x_max, 501).unsqueeze(1)
c0 = float(torch.as_tensor(M.c(torch.tensor([[0.0]]))).reshape(-1)[0])
check("tau == (x - x_min)/c exactly",
      float((tt(xs) - (xs - M.x_min) / c0).abs().max()) < 1e-6,
      f"{float((tt(xs) - (xs - M.x_min)/c0).abs().max()):.3e}")
check("tau(x_min) == 0", abs(float(tt(torch.tensor([[M.x_min]])))) < 1e-12)

print("\n5. Monotonicity and travel-time ordering")
for mk in MATS:
    M = MATERIALS[mk](); tt = TravelTime(M, n=4097)
    xs = torch.linspace(M.x_min, M.x_max, 8000).unsqueeze(1)
    v = tt(xs).squeeze()
    check(f"{mk:<16} tau strictly increasing", bool((v.diff() > 0).all()),
          f"tau_max={float(v[-1]):.5f}")

print("\n6. C^2 continuity across interior knots")
M = MATERIALS["twolayer"](); tt = TravelTime(M, n=1025)
kn = tt.knots[len(tt.knots) // 3].item()
eps = 1e-6
xq = torch.tensor([[kn - eps], [kn + eps]], dtype=torch.float32).requires_grad_(True)
d1 = torch.autograd.grad(tt(xq).sum(), xq, create_graph=True)[0]
d2 = torch.autograd.grad(d1.sum(), xq)[0]
check("tau' continuous across a knot", abs(float(d1[0] - d1[1])) < 1e-4,
      f"jump={abs(float(d1[0]-d1[1])):.2e}")
check("tau'' continuous across a knot", abs(float(d2[0] - d2[1])) < 1e-2,
      f"jump={abs(float(d2[0]-d2[1])):.2e}")

print("\n7. CharCoords lands inside [-1,1] and stays differentiable twice")
for mk in MATS:
    M = MATERIALS[mk](); cc = CharCoords(M, t_max=1.0)
    x = (torch.rand(20000, 1) * (M.x_max - M.x_min) + M.x_min)
    t = torch.rand(20000, 1)
    o = cc(x, t)
    check(f"{mk:<16} outputs within [-1,1]",
          float(o.min()) >= -1.0 - 1e-5 and float(o.max()) <= 1.0 + 1e-5,
          f"[{float(o.min()):.4f}, {float(o.max()):.4f}]")
    xr = x[:256].clone().requires_grad_(True); tr = t[:256].clone().requires_grad_(True)
    v = cc(xr, tr).sum(dim=-1, keepdim=True)
    g1 = torch.autograd.grad(v.sum(), xr, create_graph=True)[0]
    g2 = torch.autograd.grad(g1.sum(), xr)[0]
    check(f"{mk:<16} second derivative finite", bool(torch.isfinite(g2).all()))

print("\n8. xi and eta really are the characteristic pair")
M = MATERIALS["homogeneous"](); cc = CharCoords(M, t_max=1.0)
c0 = float(torch.as_tensor(M.c(torch.tensor([[0.0]]))).reshape(-1)[0])
# along a rightgoing characteristic x = x0 + c*t, xi = tau(x)-t must be constant
t = torch.linspace(0.0, 0.5, 60).unsqueeze(1)
x = -0.5 + c0 * t
o = cc(x, t)
check("xi constant along a rightgoing characteristic",
      float(o[:, 0].std()) < 1e-5, f"std={float(o[:,0].std()):.2e}")
check("eta varies along it", float(o[:, 1].std()) > 1e-2, f"std={float(o[:,1].std()):.2e}")

print("\n" + "=" * 78)
print(f"  {sum(R)}/{len(R)} passed")
sys.exit(0 if all(R) else 1)
