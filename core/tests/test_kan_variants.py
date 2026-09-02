"""Coverage for kan_variants.py and kan_patch.py.

These modules exist to make the Phase 8 ladder possible, and two of the bugs they
work around are SILENT (NaN with no exception, and a 26000x-wrong least-squares
solution at full rank).  Silent bugs need tests that assert numbers, not that
assert "it ran", so every check below compares against an independent quantity.
"""
import sys
import math
import torch

from core.problem import MATERIALS
from core.models import n_params
from core.kan_patch import curve2coef_stable
from core.kan_variants import TunedKAN, _Affine, BASE_FUNS, Sin

R = []
def check(n, ok, d=""):
    R.append(bool(ok)); print(f"  [{'PASS' if ok else 'FAIL'}] {n:<58} {d}")

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
W = (2, 12, 12, 1)


def mkgrid(in_dim, G, k, lo=-1.0, hi=1.0, dev="cpu"):
    h = (hi - lo) / G
    return (torch.arange(-k, G + k + 1, device=dev, dtype=torch.float32) * h + lo
            ).expand(in_dim, -1).contiguous()


print("\n1. _Affine maps the physical domain exactly onto pykan's [-1,1]")
for (xmin, xmax, tmax) in [(-1.0, 1.0, 1.0), (0.0, 2.0, 0.5), (-3.0, 5.0, 2.0)]:
    a = _Affine(xmin, xmax, tmax)
    xs = torch.tensor([[xmin], [xmax], [(xmin + xmax) / 2]])
    ts = torch.tensor([[0.0], [tmax], [tmax / 2]])
    o = a(xs, ts)
    check(f"x[{xmin},{xmax}] -> [-1,1]",
          torch.allclose(o[:, 0], torch.tensor([-1., 1., 0.]), atol=1e-6),
          f"{[round(float(v),4) for v in o[:,0]]}")
    check(f"t[0,{tmax}] -> [-1,1]",
          torch.allclose(o[:, 1], torch.tensor([-1., 1., 0.]), atol=1e-6),
          f"{[round(float(v),4) for v in o[:,1]]}")
# the whole point of the transform: without it, t occupies only half the knots
a = _Affine(-1.0, 1.0, 1.0)
t_span_raw = 1.0 - 0.0                      # t in [0,1] against a [-1,1] grid
t_span_scaled = float(a(torch.zeros(2, 1), torch.tensor([[0.], [1.]]))[:, 1].diff())
check("normalisation doubles the usable time span", abs(t_span_scaled / t_span_raw - 2.0) < 1e-6,
      f"{t_span_raw} -> {t_span_scaled}")

print("\n2. curve2coef ridge solve: agrees where pykan is valid, finite where it is not")
from kan.spline import coef2curve
from core.kan_patch import ORIGINAL_CURVE2COEF as pk_curve2coef
torch.manual_seed(0)
for G, k in [(5, 3), (10, 3), (20, 3)]:
    g = mkgrid(2, G, k); x = torch.rand(3000, 2) * 2 - 1
    y = torch.stack([torch.sin(3 * x[:, 0]), torch.cos(4 * x[:, 1])], 1).unsqueeze(-1)
    a_ = pk_curve2coef(x, y, g, k); b_ = curve2coef_stable(x, y, g, k)
    check(f"shape contract preserved G={G} k={k}", a_.shape == b_.shape,
          f"{tuple(b_.shape)}")
    ra = float((coef2curve(x, g, a_, k) - y).norm())
    rb = float((coef2curve(x, g, b_, k) - y).norm())
    check(f"full-rank residual matches pykan G={G} k={k}", abs(ra - rb) < 1e-4 * max(1.0, ra),
          f"{ra:.6f} vs {rb:.6f}")
# k=5: pykan's gels is unstable even at full rank -- ridge must be far better
g = mkgrid(2, 20, 5); x = torch.rand(6000, 2) * 2 - 1
y = torch.stack([torch.sin(3 * x[:, 0]), torch.cos(4 * x[:, 1])], 1).unsqueeze(-1)
ra = float((coef2curve(x, g, pk_curve2coef(x, y, g, 5), 5) - y).norm())
rb = float((coef2curve(x, g, curve2coef_stable(x, y, g, 5), 5) - y).norm())
check("k=5 ridge beats pykan's unstable LSQ by >100x", rb * 100 < ra, f"{ra:.4f} vs {rb:.6f}")
check("k=5 ridge residual is actually small", rb < 1e-2, f"{rb:.6f}")
# rank-deficient: the case that broke refine()
if DEV.type == "cuda":
    gd = mkgrid(2, 10, 3, dev=DEV)
    xd = torch.rand(3000, 2, device=DEV) * 0.16 - 0.09        # narrow band
    yd = torch.randn(3000, 2, 2, device=DEV)
    check("pykan NaNs on rank-deficient CUDA input (the bug)",
          not bool(torch.isfinite(pk_curve2coef(xd, yd, gd, 3)).all()))
    check("ridge solve stays finite there",
          bool(torch.isfinite(curve2coef_stable(xd, yd, gd, 3)).all()))

print("\n3. Model construction, parameter honesty, device sync")
m = TunedKAN(width=W, grid=5, k=3).to(DEV)
check("forward returns (batch,1)",
      tuple(m(torch.rand(7, 1, device=DEV), torch.rand(7, 1, device=DEV)).shape) == (7, 1))
check("symbolic branch is frozen",
      all(not q.requires_grad for mm in m.kan.symbolic_fun for q in mm.parameters()))
sym = sum(q.numel() for mm in m.kan.symbolic_fun for q in mm.parameters())
live = sum(p.numel() for p in m.parameters() if p.requires_grad)
check("n_params counts exactly the live leaves", n_params(m) == live, f"{n_params(m)}")
check("no symbolic parameter is counted", sym > 0 and n_params(m) + sym <= sum(
      p.numel() for p in m.parameters()), f"{n_params(m)} live, {sym} symbolic")
# pykan also registers node_bias/node_scale/subnode_* with affine_trainable=False,
# so the frozen total legitimately exceeds the symbolic branch alone.
other = sum(p.numel() for p in m.parameters() if not p.requires_grad) - sym
check("other frozen pykan affines accounted for", other >= 0,
      f"{other} frozen non-symbolic (node/subnode affines)")
check("pykan's own device bookkeeping followed .to()",
      str(m.kan.device).startswith(DEV.type), f"kan.device={m.kan.device}")

print("\n4. base_fun substitution reaches every layer")
for bf in BASE_FUNS:
    mm = TunedKAN(width=W, grid=5, k=3, base_fun=bf).to(DEV)
    want = type(BASE_FUNS[bf]()).__name__
    got = {type(l.base_fun).__name__ for l in mm.kan.act_fun}
    check(f"{bf:<9} installed on all layers", got == {want}, f"{got}")
try:
    TunedKAN(width=W, base_fun="nope")
    check("unknown base_fun rejected", False)
except KeyError:
    check("unknown base_fun rejected", True)
check("Sin is a genuine sine", torch.allclose(Sin()(torch.tensor([0.0, math.pi / 2])),
                                              torch.tensor([0.0, 1.0]), atol=1e-6))

print("\n5. Grid refinement: resolution rises, represented function is preserved")
torch.manual_seed(0)
m = TunedKAN(width=W, grid=5, k=3).to(DEV)
xc = torch.rand(3000, 1, device=DEV) * 2 - 1
tc = torch.rand(3000, 1, device=DEV)
spacing = lambda mm: float(mm.kan.act_fun[0].grid[0][1] - mm.kan.act_fun[0].grid[0][0])
prev_sp, prev_out, prev_np = spacing(m), m(xc, tc).detach().clone(), n_params(m)
for G in (10, 20):
    m.refine_(G, xc, tc)
    out = m(xc, tc).detach()
    check(f"grid->{G} output finite", bool(torch.isfinite(out).all()))
    drift = float((out - prev_out).norm() / prev_out.norm())
    check(f"grid->{G} preserves the function", drift < 1e-2, f"drift={drift:.3e}")
    check(f"grid->{G} halves knot spacing", spacing(m) < prev_sp * 0.75,
          f"{prev_sp:.4f} -> {spacing(m):.4f}")
    check(f"grid->{G} adds parameters", n_params(m) > prev_np,
          f"{prev_np} -> {n_params(m)}")
    prev_sp, prev_out, prev_np = spacing(m), out.clone(), n_params(m)
check("refined grid finally resolves the pulse (FWHM 0.2355)", spacing(m) < 0.2355 / 2,
      f"spacing={spacing(m):.4f}")

print("\n6. Fourier-embedded variant")
for nf in (8, 16):
    mm = TunedKAN(width=W, grid=10, k=3, n_fourier=nf).to(DEV)
    check(f"n_fourier={nf} sets KAN input dim to 2*nf", mm.cfg["width"][0] == 2 * nf,
          f"{mm.cfg['width'][0]}")
    o = mm(torch.rand(5, 1, device=DEV), torch.rand(5, 1, device=DEV))
    check(f"n_fourier={nf} forward finite", bool(torch.isfinite(o).all()))
    check(f"n_fourier={nf} bypasses the affine scaler", mm.scaler is None)

print("\n7. Second derivatives exist and are finite (the residual needs u_xx)")
M = MATERIALS["twolayer"]()
for cfg in [dict(grid=5, k=3), dict(grid=20, k=3), dict(grid=20, k=5)]:
    mm = TunedKAN(width=W, **cfg).to(DEV)
    x = (torch.rand(256, 1, device=DEV) * 2 - 1).requires_grad_(True)
    t = torch.rand(256, 1, device=DEV).requires_grad_(True)
    u = mm(x, t)
    ux = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    uxx = torch.autograd.grad(ux.sum(), x, create_graph=True)[0]
    check(f"grid={cfg['grid']} k={cfg['k']}: u_xx finite and non-trivial",
          bool(torch.isfinite(uxx).all()) and float(uxx.abs().max()) > 0,
          f"max|u_xx|={float(uxx.abs().max()):.3e}")

print("\n" + "=" * 78)
print(f"  {sum(R)}/{len(R)} passed")
sys.exit(0 if all(R) else 1)
