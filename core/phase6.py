"""Phase 6 -- the definitive comparison, with the L-BFGS scaling bug fixed.

Supersedes Phase 5, whose entire L-BFGS column was invalid: normalising the
residual (D6) shrank gradients ~5e4x, and torch's LBFGS picks its first step as
min(1, 1/||grad||_1)*lr, so step one overshot catastrophically.  Measured effect
of removing it: MLP/twolayer 95.59% -> 0.86%, WavKAN 92.63% -> 0.88%.

Protocol
  * L-BFGS on a FIXED Sobol collocation set, UNNORMALISED residual (rak's recipe)
  * each architecture run with and without R3 adaptive resampling
  * 5 architectures x 3 materials x 3 seeds x {plain, R3}
  * everything else identical: residual, ansatz, BC loss, metric, bandwidth
"""
import argparse, json, os
import numpy as np, torch
from core.problem import MATERIALS
from core.reference import fd_reference
from core.models import REGISTRY, n_params
from core.train import train_lbfgs

ap = argparse.ArgumentParser()
ap.add_argument("--cells", required=True)
ap.add_argument("--archs", default="mlp,fourier,pirate,wavkan,pykan_wide")
ap.add_argument("--epochs", type=int, default=700)
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--out", required=True)
ap.add_argument("--ckpt", default=None)
A = ap.parse_args()
dev = torch.device(f"cuda:{A.gpu}")

rows = json.load(open(A.out)) if os.path.exists(A.out) else []
done = {(r["material"], r["seed"], r["arch"], r["r3"]) for r in rows}
refs = {}

for cell in A.cells.split(","):
    mk, sd = cell.split(":"); sd = int(sd)
    if mk not in refs:
        M = MATERIALS[mk]()
        x, t, u = fd_reference(M, nx=512, T=1.0, sigma_g=0.1)
        i = [int(np.argmin(np.abs(t - v))) for v in np.linspace(0.05, 1.0, 20)]
        refs[mk] = (M, x, t[i], u[i])
    M, x, tt, uu = refs[mk]
    for ak in A.archs.split(","):
        for r3 in (False, True):
            if (mk, sd, ak, r3) in done:
                continue
            sp = None
            if A.ckpt:
                os.makedirs(A.ckpt, exist_ok=True)
                sp = os.path.join(A.ckpt, f"{mk}_s{sd}_{ak}_{'r3' if r3 else 'plain'}.pt")
            try:
                mdl, m = train_lbfgs(REGISTRY[ak], M, epochs=A.epochs, seed=sd,
                                     device=dev, use_r3=r3, save_path=sp,
                                     x_ref=x, t_ref=tt, u_ref=uu)
            except Exception as e:
                print(f"{mk:<12}s{sd} {ak:<11}{'R3' if r3 else '--':<4} FAILED {type(e).__name__}: {e}", flush=True)
                continue
            rows.append(dict(material=mk, seed=sd, arch=ak, r3=r3,
                             params=n_params(mdl), rel_l2=m["rel_l2"],
                             resid=m["residual_rms_heldout"], collapse=m["trivial_collapse"],
                             epochs_run=m["epochs_run"], wall_s=m["wall_s"],
                             r3_retained=m.get("r3_retained")))
            print(f"{mk:<12}s{sd} {ak:<11}{'R3' if r3 else '--':<4} L2 {m['rel_l2']:7.3f}%  "
                  f"{m['epochs_run']:>4}ep {m['wall_s']:5.0f}s"
                  + ("  COLLAPSE" if m["trivial_collapse"] else ""), flush=True)
            json.dump(rows, open(A.out, "w"), indent=1)
print("done")
