"""Phase 4 -- the comparison.

PINN vs KAN under the frozen spec.  Every factor fixed except the network:
identical PDE residual, verified reference, hard-IC ansatz (`legacy`, chosen by
the Phase-3 A/B), absorbing-BC loss, normalised residual, Wang et al. eps
annealing, grad-norm balancing, Fourier bandwidth sigma_B=10 (DECISIONS.md D4),
optimiser, LR schedule, collocation counts and step budget.

Reported per run: rel-L2 (fixed denominator), held-out residual, parameter count,
wall-clock, and the two-sided trivial-solution guard.  >=3 seeds per cell.
"""
import argparse, json, os
import numpy as np, torch
from core.problem import MATERIALS
from core.reference import fd_reference
from core.models import REGISTRY, n_params
from core.train import train

ap = argparse.ArgumentParser()
ap.add_argument("--cells", required=True, help="material:seed,material:seed,...")
ap.add_argument("--archs", default="mlp,fourier,pirate,splinekan,wavkan")
ap.add_argument("--steps", type=int, default=8000)
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--out", required=True)
ap.add_argument("--lbfgs", type=int, default=0)
ap.add_argument("--r3", action="store_true")
ap.add_argument("--ckpt_dir", default=None)
A = ap.parse_args()
dev = torch.device(f"cuda:{A.gpu}")

rows = json.load(open(A.out)) if os.path.exists(A.out) else []
done = {(r["material"], r["seed"], r["arch"]) for r in rows}
refs = {}

for cell in A.cells.split(","):
    mk, sd = cell.split(":"); sd = int(sd)
    if mk not in refs:
        M = MATERIALS[mk]()
        x, t, u = fd_reference(M, nx=512, T=1.0, sigma_g=0.1)
        idx = [int(np.argmin(np.abs(t - v))) for v in np.linspace(0.05, 1.0, 20)]
        refs[mk] = (M, x, t[idx], u[idx])
    M, x, tt, uu = refs[mk]
    for ak in A.archs.split(","):
        if (mk, sd, ak) in done:
            continue
        sp = None
        if A.ckpt_dir:
            os.makedirs(A.ckpt_dir, exist_ok=True)
            sp = os.path.join(A.ckpt_dir, f"{mk}_s{sd}_{ak}.pt")
        mdl, m = train(REGISTRY[ak], M, ansatz_kind="legacy", steps=A.steps, seed=sd,
                       device=dev, eval_every=A.steps // 2, use_r3=A.r3,
                       lbfgs_iters=A.lbfgs, save_path=sp,
                       x_ref=x, t_ref=tt, u_ref=uu)
        rows.append(dict(material=mk, seed=sd, arch=ak, params=n_params(mdl),
                         rel_l2=m["rel_l2"], resid=m["residual_rms_heldout"],
                         collapse=m["trivial_collapse"],
                         min_ratio=m["min_late_norm_ratio"], wall_s=m["wall_s"],
                         lbfgs_evals=m["lbfgs_evals"], r3=A.r3))
        print(f"{mk:<12}s{sd} {ak:<10} L2 {m['rel_l2']:8.3f}%  "
              f"resid {m['residual_rms_heldout']:.2e}  p={n_params(mdl):>7,}  "
              f"{m['wall_s']:5.0f}s{'  COLLAPSE' if m['trivial_collapse'] else ''}", flush=True)
        json.dump(rows, open(A.out, "w"), indent=1)
print("done")
