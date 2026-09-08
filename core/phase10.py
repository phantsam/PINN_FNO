"""Phase 10 -- the optimiser axis: Adam -> L-BFGS hybrid, four architectures.

Everything through Phase 9 used L-BFGS alone.  That leaves the standard PINN
recipe (Adam to find the basin, L-BFGS to polish it) untested, which is the
largest single variable still open.  This phase closes it on a clean 4 x 3 grid:

    MLP PINN      REGISTRY["mlp"]         tanh MLP on raw (x,t)
    Fourier PINN  REGISTRY["fourier"]     random Fourier features, sigma_B=10
    plain KAN     REGISTRY["pykan_wide"]  reference pykan on raw (x,t)
    PIKAN         REGISTRY["splinekan"]   B-spline KAN on a Fourier embedding
                                          -- the same architecture the main
                                          branch calls "pikan"
                                          (ML/training_code.py:688)

x {homogeneous, twolayer, multilayer}, --plain first, then --r3 with the
paper-faithful sampler (see hybrid.R3Pool).

Scoring is against nx=2048 with snapshots interpolated to the exact target
instants.  At nx=512 the reference's own discretisation error (0.11-0.18%) is
the same size as the model errors, so differences there are not measurable.

Resumable: results are keyed (material, seed, model) and appended to --out, so
the seed list can be extended without redoing work.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch

from .problem import MATERIALS
from .reference import fd_reference
from .models import REGISTRY, n_params
from .hybrid import train_hybrid
from .rescore import fine_reference, score, TARGETS
from .phase8 import build as build_rung

MODELS = ["mlp", "fourier", "pykan_wide", "splinekan"]
LABEL = {"mlp": "MLP PINN", "fourier": "Fourier PINN",
         "pykan_wide": "plain KAN", "splinekan": "PIKAN(ours)",
         "rung:fourier16": "PIKAN(pykan)", "rung:charcoords50": "tuned KAN",
         "rung:grid20": "KAN grid20"}


def make(arch, M, seed):
    """A model name is either a REGISTRY key or `rung:<phase8 rung>`.

    The rung form exists because `splinekan` is OUR hand-rolled spline layer,
    which models.py already flags as producing max|u_tt| ~ 5e5 at init.  Calling
    its failure "PIKAN fails" would be blaming the architecture for our code, so
    `rung:fourier16` provides the same idea -- splines on a Fourier embedding --
    on the AUTHORS' pykan backend.  If both fail, the Fourier embedding is the
    cause; if only ours does, our layer is.
    """
    if arch.startswith("rung:"):
        return lambda: build_rung(arch.split(":", 1)[1], M, seed=seed + 1)
    return REGISTRY[arch]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--materials", default="homogeneous,twolayer,multilayer")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--r3", action="store_true",
                    help="paper-faithful R3 during the Adam phase; the L-BFGS "
                         "phase then freezes the pool R3 converged to")
    ap.add_argument("--soft-ic", action="store_true",
                    help="enforce the IC by a loss term instead of the hard "
                         "ansatz; removes the u==0 basin every 95%% result sits in")
    ap.add_argument("--adam-steps", type=int, default=700)
    ap.add_argument("--lbfgs-epochs", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--nx", type=int, default=2048)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", required=True)
    A = ap.parse_args()

    dev = torch.device(f"cuda:{A.gpu}")
    seeds = [int(s) for s in A.seeds.split(",")]
    sfx = ("+R3" if A.r3 else "") + ("+soft" if A.soft_ic else "")
    rows = json.load(open(A.out)) if os.path.exists(A.out) else []
    done = {(r["material"], r["seed"], r["model"]) for r in rows}
    if A.ckpt:
        os.makedirs(A.ckpt, exist_ok=True)

    print(f"Phase 10: Adam({A.adam_steps}) -> L-BFGS({A.lbfgs_epochs} cap), "
          f"R3={'paper-faithful' if A.r3 else 'off'}, scored @ nx={A.nx}\n")
    print(f"{'material':<13}{'model':<18}{'seed':>5}{'rel-L2':>10}"
          f"{'adam_loss':>12}{'lbfgs_loss':>12}{'ep':>6}{'s':>7}", flush=True)

    for mk in A.materials.split(","):
        M = MATERIALS[mk]()
        xr, ur = fine_reference(M, A.nx)
        # cheap in-loop reference; the headline number comes from `score` below
        xc, tc, uc = fd_reference(M, nx=512, T=1.0, sigma_g=0.1)
        ev = [int(np.argmin(np.abs(tc - v))) for v in TARGETS]

        for arch in A.models.split(","):
            name = arch + sfx
            for sd in seeds:
                if (mk, sd, name) in done:
                    continue
                safe = arch.replace(":", "-")
                cp = (os.path.join(A.ckpt, f"{mk}_s{sd}_{safe}_"
                                   f"{'r3' if A.r3 else 'plain'}"
                                   f"{'_soft' if A.soft_ic else ''}_hybrid.pt")
                      if A.ckpt else None)
                m, mm = train_hybrid(make(arch, M, sd), M, adam_steps=A.adam_steps,
                                     ansatz_kind="none" if A.soft_ic else "legacy",
                                     soft_ic=A.soft_ic,
                                     lbfgs_epochs=A.lbfgs_epochs, seed=sd, lr=A.lr,
                                     device=dev, use_r3=A.r3, save_path=cp,
                                     x_ref=xc, t_ref=tc[ev], u_ref=uc[ev])
                v = score(m, M, xr, ur, dev,
                          ansatz_kind="none" if A.soft_ic else "legacy")
                rows.append(dict(material=mk, seed=sd, model=name, arch=arch,
                                 rel_l2=v, nx=A.nx, params=n_params(m),
                                 rel_l2_512=mm["rel_l2"], adam_loss=mm["adam_loss"],
                                 lbfgs_loss=mm["best_loss"],
                                 lbfgs_epochs_run=mm["lbfgs_epochs_run"],
                                 diverged=mm["diverged"], wall_s=mm["wall_s"],
                                 r3_retained_mean=mm["r3_retained_mean"],
                                 optimizer="adam+lbfgs"))
                flag = "  DIVERGED" if mm["diverged"] else ""
                print(f"{mk:<13}{LABEL.get(arch,arch)+sfx:<18}{sd:>5}{v:>9.4f}%"
                      f"{mm['adam_loss']:>12.3e}{mm['best_loss']:>12.3e}"
                      f"{mm['lbfgs_epochs_run']:>6}{mm['wall_s']:>7.0f}{flag}",
                      flush=True)
                json.dump(rows, open(A.out, "w"), indent=1)
                del m; torch.cuda.empty_cache()
    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
