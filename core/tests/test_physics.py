"""Physics verification.

Additions closing audit misses:
  #6  heterogeneous self-convergence (previously only Homogeneous was tested)
  #8/#9 Mur ABC reflection coefficient in the reference (was entirely unverified;
        a sign-flipped kL turns the left wall into a |R|=0.66 mirror, silently)
  #11 ansatz must not OVER-constrain: du_tt(x,0)/dN must be nonzero, else the
        residual at t=0 is irreducible for any network (t**3 passes both IC checks)
  #12 ABC operator exercised with c != 1 (was only ever run on Homogeneous)
  CFL guard asserted at its boundary
Also: the w/dx>=20 assertion is replaced -- it was decorative (measured -0.00%
at w/dx = 4.1, 2.0 and 0.4).  The condition that actually binds is pulse
separation, t_inc*c >= 3*sigma, which was an unguarded magic number.
"""
import sys
import numpy as np, torch
from core.problem import (Homogeneous, TwoLayer, MultiLayer, VariableDensity,
                          gaussian_ic, gaussian_ic_dx)
from core.operators import wave_operator, abc_operator, make_ansatz
from core.reference import fd_reference, dalembert

torch.set_default_dtype(torch.float64)
R = []
def check(n, ok, d=""):
    R.append(ok); print(f"  [{'PASS' if ok else 'FAIL'}] {n:<52} {d}")


print("\n1. Hard-constraint ansatz: exact ICs, and NOT over-constrained")
for kind in ("legacy", "poly"):
    a = make_ansatz(kind, sigma_g=0.1)
    net = lambda x, t: 50.0 * torch.sin(7 * x + 3 * t) + 20.0
    x = torch.linspace(-1, 1, 513).reshape(-1, 1).requires_grad_(True)
    t = torch.zeros_like(x).requires_grad_(True)
    u = a(net(x, t), x, t)
    ut = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    check(f"{kind}: max|u(x,0)-g(x)|", (u - gaussian_ic(x, 0.1)).abs().max().item() < 1e-14)
    check(f"{kind}: max|u_t(x,0)|", ut.abs().max().item() < 1e-14)
    # -- over-constraint probe: u_tt(x,0) must remain steerable by the network --
    A = torch.tensor(1.0, requires_grad=True)
    xa = torch.linspace(-0.5, 0.5, 33).reshape(-1, 1).requires_grad_(True)
    ta = torch.zeros_like(xa).requires_grad_(True)
    ua = a(A * net(xa, ta), xa, ta)
    u1 = torch.autograd.grad(ua, ta, torch.ones_like(ua), create_graph=True)[0]
    u2 = torch.autograd.grad(u1, ta, torch.ones_like(u1), create_graph=True)[0]
    sens = torch.autograd.grad(u2.abs().sum(), A, allow_unused=True)[0]
    sv = 0.0 if sens is None else abs(float(sens))
    check(f"{kind}: u_tt(x,0) is steerable (d/dN != 0)", sv > 1e-8, f"|d u_tt/dN|={sv:.3f}")


print("\n2. ABC operator -- including c != 1 (miss #12)")
# Use the ACTUAL local wave speed at the boundary, not the asymptotic layer value:
# with a finite transition width the tanh has not fully saturated at x_max, so
# c(x_max) differs from sqrt(E_last) by O(exp(-d/w)).
for M in (Homogeneous(), TwoLayer(), MultiLayer()):
    cb = float(np.asarray(M.c(np.array([M.x_max])))[0])
    k = 2 * np.pi
    right = lambda x, t, c=cb: torch.sin(k * (x - c * t))
    left = lambda x, t, c=cb: torch.sin(k * (x + c * t))
    tt = torch.linspace(0.05, 0.9, 64).reshape(-1, 1)
    xR = torch.full((64, 1), M.x_max); xL = torch.full((64, 1), M.x_min)
    rr = abc_operator(right, xR.clone(), tt.clone(), M, "right").abs().max().item()
    wr = abc_operator(left, xR.clone(), tt.clone(), M, "right").abs().max().item()
    check(f"{M.name}: right BC kills outgoing (c={cb:.4f})", rr < 1e-10, f"{rr:.2e}")
    check(f"{M.name}: right BC rejects incoming", wr > 1.0, f"{wr:.2e}")


print("\n3. Reference convergence -- homogeneous (vs exact) AND heterogeneous (self)")
errs = []
for nx in (256, 512, 1024, 2048):
    x, t, u = fd_reference(Homogeneous(), nx=nx, T=0.5, sigma_g=0.1)
    w = np.abs(x) < 0.85
    errs.append(np.max(np.abs(u[-1][w] - dalembert(x, t[-1], 0.1)[w])))
o = [np.log2(errs[i] / errs[i + 1]) for i in range(3)]
check("Homogeneous order ~2 (vs exact d'Alembert)", all(abs(v - 2) < 0.15 for v in o),
      ", ".join(f"{v:.2f}" for v in o))

for M in (TwoLayer(), MultiLayer(), VariableDensity()):          # miss #6
    ref_nx = 6145
    xf, tf, uf = fd_reference(M, nx=ref_nx, T=0.6, sigma_g=0.1, x0=-0.5)
    es = []
    for nx in (385, 769, 1537):
        xc, tc, uc = fd_reference(M, nx=nx, T=0.6, sigma_g=0.1, x0=-0.5)
        ic = np.abs(xc) < 0.9
        es.append(np.max(np.abs(uc[-1][ic] - np.interp(xc[ic], xf, uf[-1]))))
    oh = [np.log2(es[i] / es[i + 1]) for i in range(2)]
    check(f"{M.name} self-convergence order >= 1.8", all(v > 1.8 for v in oh),
          ", ".join(f"{v:.2f}" for v in oh))


print("\n4. Mur ABC in the reference -- reflection coefficient (misses #8, #9)")
M = Homogeneous()
Rs = []
for nx in (1024, 4096):
    x, t, u = fd_reference(M, nx=nx, T=1.9, sigma_g=0.08, x0=0.0)
    inc = np.max(np.abs(u[:len(t)//4]))
    late = u[int(0.85 * len(t)):]                # both pulses have exited by now
    Rs.append(np.max(np.abs(late)) / inc)
check("|R| < 1e-3 at nx=1024 (not a mirror)", Rs[0] < 1e-3, f"|R|={Rs[0]:.2e}")
check("|R| decreases with refinement", Rs[1] < Rs[0], f"{Rs[0]:.2e} -> {Rs[1]:.2e}")
# heterogeneous: exercises the RIGHT wall at c=sqrt(1.5) (miss #9)
x, t, u = fd_reference(TwoLayer(), nx=2048, T=1.9, sigma_g=0.08, x0=0.0)
inc = np.max(np.abs(u[:len(t)//4]))
Rh = np.max(np.abs(u[int(0.85 * len(t)):])) / inc
check("|R| < 1e-2 on TwoLayer (right wall c!=1)", Rh < 1e-2, f"|R|={Rh:.2e}")


print("\n5. Interface transmission (theory T = 2 Z1/(Z1+Z2))")
W, SG, NX, T_INC = 0.01, 0.10, 8192, 0.20
M2 = TwoLayer(w=W)
T_th = 2 * 1.0 / (1.0 + np.sqrt(1.5))
x, t, u = fd_reference(M2, nx=NX, T=0.62, sigma_g=SG, x0=-0.5)
check("thin vs pulse (w/sigma <= 0.2) -- the binding condition", W / SG <= 0.2, f"{W/SG:.2f}")
# the two counter-propagating halves sit at x0 +- c*t, so SEPARATION = 2*c*t
check("pulse halves separated (2*t_inc*c >= 3 sigma)", 2 * T_INC * 1.0 >= 3 * SG,
      f"sep={2*T_INC/SG:.1f} sigma")
check("pulse has not yet reached interface (t_inc*c <= |x0|)", T_INC * 1.0 <= 0.5)
inc = np.max(np.abs(u[np.argmin(np.abs(t - T_INC))][(x > -0.48) & (x < -0.12)]))
tr = np.max(np.abs(u[np.argmin(np.abs(t - 0.55))][(x > 0.10) & (x < 0.70)]))
meas = tr / inc
check("transmission coefficient", abs(meas - T_th) / T_th < 0.01,
      f"meas={meas:.4f} theory={T_th:.4f} err={100*(meas-T_th)/T_th:+.2f}%")


print("\n6. CFL guard fires at the leapfrog limit")
try:
    fd_reference(Homogeneous(), nx=512, nt=200, T=1.0)      # nu >> 1
    check("CFL violation raises", False, "no exception")
except ValueError:
    check("CFL violation raises ValueError", True)
try:
    fd_reference(Homogeneous(), nx=512, T=1.0); check("valid CFL runs", True)
except Exception as e:
    check("valid CFL runs", False, str(e))


print("\n7. Regression -- grad through cat(x,t) drops ansatz terms")
a = make_ansatz("legacy", 0.1)
net = lambda x, t: torch.sin(4 * x) * torch.cos(2 * t)
worst = 0.0
for tv in (0.005, 0.02, 0.05):
    xx = torch.full((1, 1), 0.15); t2 = torch.full((1, 1), tv)
    xm = xx.clone().requires_grad_(True); tm = t2.clone().requires_grad_(True)
    cat = torch.cat((xm, tm), 1)
    ub = a(net(cat[:, 0:1], cat[:, 1:2]), xm, tm)
    utb = torch.autograd.grad(ub, cat, torch.ones_like(ub), create_graph=True)[0][:, 1:2]
    xc = xx.clone().requires_grad_(True); tc = t2.clone().requires_grad_(True)
    uo = a(net(xc, tc), xc, tc)
    uto = torch.autograd.grad(uo, tc, torch.ones_like(uo), create_graph=True)[0]
    worst = max(worst, abs(utb.item() - uto.item()))
check("cat-gradient bug detectable (core/ never uses it)", worst > 1.0, f"{worst:.2f}")

print("\n" + "=" * 74)
nf = len(R) - sum(R)
print(f"{sum(R)}/{len(R)} passed" + ("" if nf == 0 else f"   *** {nf} FAILURES ***"))
sys.exit(1 if nf else 0)
