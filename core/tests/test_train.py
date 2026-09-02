"""Coverage for core/train.py -- previously ZERO.

Targets the three things added last, plus the R3 port whose original had a bug
that silently disabled it (rak: `if stopped: return` fired before the sampler was
ever built, so 3 of 6 runs labelled "R3" ran plain Sobol).
"""
import sys, os, tempfile
import numpy as np, torch
from core.problem import Homogeneous, TwoLayer
from core.reference import fd_reference
from core.models import REGISTRY
from core.operators import make_ansatz, wave_operator
from core.losses import residual_scale
from core.train import train, R3Sampler, sample, lbfgs_refine

torch.set_default_dtype(torch.float32)
DEV = torch.device("cpu")
R = []
def check(n, ok, d=""):
    R.append(ok); print(f"  [{'PASS' if ok else 'FAIL'}] {n:<52} {d}")

M = Homogeneous()
x_ref, t_ref, u_ref = fd_reference(M, nx=256, T=1.0, sigma_g=0.1)
idx = [int(np.argmin(np.abs(t_ref - v))) for v in np.linspace(0.05, 1.0, 8)]
KW = dict(steps=60, n_int=512, n_bc=64, device=DEV, eval_every=10**9,
          x_ref=x_ref, t_ref=t_ref[idx], u_ref=u_ref[idx])

print("\n1. Training runs, produces finite metrics, reduces the residual")
_, m0 = train(REGISTRY["fourier"], M, steps=5, n_int=512, n_bc=64, device=DEV,
              eval_every=10**9, x_ref=x_ref, t_ref=t_ref[idx], u_ref=u_ref[idx])
_, m1 = train(REGISTRY["fourier"], M, **KW)
check("metrics finite", np.isfinite(m1["rel_l2"]) and np.isfinite(m1["residual_rms_heldout"]))
check("60 steps beats 5 steps on held-out residual",
      m1["residual_rms_heldout"] < m0["residual_rms_heldout"],
      f"{m0['residual_rms_heldout']:.3e} -> {m1['residual_rms_heldout']:.3e}")

print("\n2. Reproducibility")
_, a = train(REGISTRY["fourier"], M, seed=3, **KW)
_, b = train(REGISTRY["fourier"], M, seed=3, **KW)
_, c = train(REGISTRY["fourier"], M, seed=4, **KW)
check("same seed -> identical rel_l2", a["rel_l2"] == b["rel_l2"], f"{a['rel_l2']:.6f}")
check("different seed -> different rel_l2", a["rel_l2"] != c["rel_l2"],
      f"{a['rel_l2']:.4f} vs {c['rel_l2']:.4f}")

print("\n3. R3 sampler retains HIGH-residual points and actually fires")
gen = torch.Generator(device=DEV).manual_seed(0)
r3 = R3Sampler(2000, M, 1.0, DEV, gen)
mdl = REGISTRY["fourier"]().to(DEV)
ans = make_ansatz("legacy", 0.1)
u_fn = lambda X, T: ans(mdl(X, T), X, T)
rs = residual_scale(M)
before = (wave_operator(u_fn, r3.x.clone(), r3.t.clone(), M) / rs).pow(2).detach()
r3.update(u_fn, rs)
after = (wave_operator(u_fn, r3.x.clone(), r3.t.clone(), M) / rs).pow(2).detach()
check("R3 fired (retained > 0)", r3.n_retained > 0, f"retained {r3.n_retained}/2000")
check("R3 keeps point count constant", r3.x.shape[0] == 2000)
check("R3 raises mean residual of the pool (keeps the hard points)",
      float(after.mean()) > float(before.mean()),
      f"{float(before.mean()):.3e} -> {float(after.mean()):.3e}")
_, mr = train(REGISTRY["fourier"], M, use_r3=True, r3_every=20, **KW)
check("train(use_r3=True) reports retention", mr["r3_retained"] is not None
      and mr["r3_retained"] > 0, f"retained={mr['r3_retained']}")
check("train(use_r3=False) reports None", m1["r3_retained"] is None)

print("\n4. L-BFGS refinement actually runs (strong_wolfe spelled correctly)")
_, ml = train(REGISTRY["fourier"], M, lbfgs_iters=30, **KW)
check("L-BFGS performed evaluations", ml["lbfgs_evals"] > 0, f"{ml['lbfgs_evals']} evals")
check("L-BFGS loss finite", np.isfinite(ml["lbfgs_loss"]), f"{ml['lbfgs_loss']:.3e}")
check("no L-BFGS when lbfgs_iters=0", m1["lbfgs_evals"] == 0)

print("\n5. Checkpoint round-trip (the stale-weights pitfall)")
with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "m.pt")
    mdl2, ms = train(REGISTRY["fourier"], M, seed=11, save_path=p, **KW)
    check("checkpoint written", os.path.exists(p))
    ck = torch.load(p, map_location=DEV, weights_only=False)
    fresh = REGISTRY["fourier"]().to(DEV)
    fresh.load_state_dict(ck["state_dict"])
    a1 = ans(mdl2(torch.zeros(5, 1), torch.full((5, 1), .5)), torch.zeros(5, 1), torch.full((5, 1), .5))
    a2 = ans(fresh(torch.zeros(5, 1), torch.full((5, 1), .5)), torch.zeros(5, 1), torch.full((5, 1), .5))
    check("reloaded model reproduces output exactly", torch.allclose(a1, a2))
    check("metrics stored alongside weights", abs(ck["metrics"]["rel_l2"] - ms["rel_l2"]) < 1e-9)

print("\n6. Causal scheduler advances during training")
_, ms2 = train(REGISTRY["fourier"], M, steps=400, n_int=512, n_bc=64, device=DEV,
               eval_every=10**9, x_ref=x_ref, t_ref=t_ref[idx], u_ref=u_ref[idx])
check("eps advanced beyond its initial 1e-2", ms2["final_eps"] > 1e-2, f"eps={ms2['final_eps']:.0e}")

print("\n" + "=" * 74)
nf = len(R) - sum(R)
print(f"{sum(R)}/{len(R)} passed" + ("" if nf == 0 else f"   *** {nf} FAILURES ***"))
sys.exit(1 if nf else 0)
