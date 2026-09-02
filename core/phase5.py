"""Phase 5 -- the definitive comparison.

Supersedes Phase 4, which used Adam for every architecture and was therefore
measuring the optimiser (CORRECTIONS C9: MLP is 69x better under L-BFGS, WavKAN
collapses under it, PirateNet is 12x better under Adam).

Protocol
  * every architecture gets BOTH optimisers; the best is its score
      - Adam, 20000 steps, resampled collocation, causal annealing
      - L-BFGS, 700 epochs, FIXED collocation, best-weight restore (rak's recipe)
  * 3 materials x 3 seeds
  * identical residual / ansatz / BC loss / metric / bandwidth for all
  * my hand-rolled SplineKAN is replaced by pykan, the reference implementation
  * checkpoints saved; results written incrementally and resumable
"""
import argparse, json, os
import numpy as np, torch
from core.problem import MATERIALS
from core.reference import fd_reference
from core.models import REGISTRY, n_params
from core.train import train, train_lbfgs

ap = argparse.ArgumentParser()
ap.add_argument("--cells", required=True)                  # material:seed,...
ap.add_argument("--archs", default="mlp,fourier,pirate,wavkan,pykan_wide")
ap.add_argument("--adam-steps", type=int, default=20000)
ap.add_argument("--lbfgs-epochs", type=int, default=700)
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--out", required=True)
ap.add_argument("--ckpt", default=None)
A = ap.parse_args()
dev = torch.device(f"cuda:{A.gpu}")

rows = json.load(open(A.out)) if os.path.exists(A.out) else []
done = {(r["material"], r["seed"], r["arch"], r["optimizer"]) for r in rows}
refs = {}

def save():
    json.dump(rows, open(A.out, "w"), indent=1)

for cell in A.cells.split(","):
    mk, sd = cell.split(":"); sd = int(sd)
    if mk not in refs:
        M = MATERIALS[mk]()
        x, t, u = fd_reference(M, nx=512, T=1.0, sigma_g=0.1)
        i = [int(np.argmin(np.abs(t - v))) for v in np.linspace(0.05, 1.0, 20)]
        refs[mk] = (M, x, t[i], u[i])
    M, x, tt, uu = refs[mk]
    for ak in A.archs.split(","):
        for opt in ("adam", "lbfgs"):
            if (mk, sd, ak, opt) in done:
                continue
            sp = None
            if A.ckpt:
                os.makedirs(A.ckpt, exist_ok=True)
                sp = os.path.join(A.ckpt, f"{mk}_s{sd}_{ak}_{opt}.pt")
            try:
                if opt == "adam":
                    mdl, m = train(REGISTRY[ak], M, ansatz_kind="legacy",
                                   steps=A.adam_steps, seed=sd, device=dev,
                                   eval_every=10**9, save_path=sp,
                                   x_ref=x, t_ref=tt, u_ref=uu)
                else:
                    mdl, m = train_lbfgs(REGISTRY[ak], M, ansatz_kind="legacy",
                                         epochs=A.lbfgs_epochs, seed=sd, device=dev,
                                         save_path=sp, x_ref=x, t_ref=tt, u_ref=uu)
            except Exception as e:
                print(f"{mk:<12}s{sd} {ak:<11}{opt:<6} FAILED: {type(e).__name__}: {e}", flush=True)
                continue
            rows.append(dict(material=mk, seed=sd, arch=ak, optimizer=opt,
                             params=n_params(mdl), rel_l2=m["rel_l2"],
                             resid=m["residual_rms_heldout"], collapse=m["trivial_collapse"],
                             min_ratio=m["min_late_norm_ratio"], wall_s=m["wall_s"]))
            print(f"{mk:<12}s{sd} {ak:<11}{opt:<6} L2 {m['rel_l2']:8.3f}%  "
                  f"resid {m['residual_rms_heldout']:.2e}  p={n_params(mdl):>7,}  "
                  f"{m['wall_s']:5.0f}s" + ("  COLLAPSE" if m["trivial_collapse"] else ""),
                  flush=True)
            save()
print("done")
