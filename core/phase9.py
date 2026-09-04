"""Phase 9 -- does `update_grid` improve on the best KAN we have?

`charcoords50` (characteristic coords + grid=50 + k=5) reached 0.1115% +- 0.0026
on homogeneous over three seeds, tying PirateNet's 0.1114% with 4.1x fewer
parameters.  Its layer trace still shows one large, unaddressed defect:

    layer   observed range        grid span        grid used   knots in range
    acts1   [-0.567, +0.516]      [-1.20, +1.20]      45%          27.1
    acts2   [-0.213, +0.297]      [-1.20, +1.20]      21%          12.8
    acts3   [-0.075, +0.120]      [-1.20, +1.20]       8%           4.9

The deepest layer touches 8% of its grid -- 92% of its spline coefficients are
never reached by any input.  Refitting each grid to the range it actually sees
would give ~61 knots there instead of ~4.9, a 12x resolution gain concentrated
exactly where the model currently has the least.

This is the same CLASS of fix as the two changes that produced the largest gains
in the study -- the grid_range correction (+22%, zero parameter cost) and grid
5 -> 20 -- both of which worked by eliminating unreachable knots.

The risk is the schedule, not the idea.  pykan's fit() does 10 updates over the
first half of training; our loop never called it at all.  A naive port fires
every ~3 epochs on a run that converges in ~300 and destroys the optimiser
(measured: 0.606% -> 7.685% at a 60-epoch budget).  Each update also invalidates
the L-BFGS curvature, so the optimiser is rebuilt every time.  This phase
therefore sweeps the schedule rather than assuming one.

Control: grid_updates=0 must reproduce the published 0.1115% exactly.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch

from .problem import MATERIALS
from .reference import fd_reference
from .models import n_params
from .train import train_lbfgs
from .phase8 import RUNGS, build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", default="homogeneous")
    ap.add_argument("--rung", default="charcoords50")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=3000)
    # (n_updates, fraction_of_cap) pairs.  0 updates == the control.
    ap.add_argument("--schedules", default="0:0.0,3:0.10,6:0.10,3:0.25")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    A = ap.parse_args()

    dev = torch.device(f"cuda:{A.gpu}")
    M = MATERIALS[A.material]()
    x, t, u = fd_reference(M, nx=512, T=1.0, sigma_g=0.1)
    ev = [int(np.argmin(np.abs(t - v))) for v in np.linspace(0.05, 1.0, 20)]

    rows = json.load(open(A.out)) if os.path.exists(A.out) else []
    done = {(r["schedule"], r["seed"]) for r in rows}
    scheds = []
    for tok in A.schedules.split(","):
        n_up, frac = tok.split(":")
        scheds.append((int(n_up), float(frac)))

    print(f"{'schedule':<14}{'seed':>5}{'rel_L2':>10}{'epochs':>8}{'wall_s':>9}"
          f"{'params':>9}", flush=True)
    for n_up, frac in scheds:
        tag = "control" if n_up == 0 else f"{n_up}upd@{frac:g}"
        for sd in [int(s) for s in A.seeds.split(",")]:
            if (tag, sd) in done:
                continue
            model = build(A.rung, M, seed=sd + 1).to(dev)
            model.set_save_act(False)
            try:
                model, m = train_lbfgs(lambda: model, M, epochs=A.epochs, seed=sd,
                                       device=dev, use_r3=False,
                                       x_ref=x, t_ref=t[ev], u_ref=u[ev],
                                       grid_updates=n_up, grid_update_frac=frac)
            except Exception as e:
                print(f"{tag:<14}{sd:>5}  FAILED {type(e).__name__}: {e}", flush=True)
                continue
            rows.append(dict(schedule=tag, n_updates=n_up, frac=frac, seed=sd,
                             material=A.material, rung=A.rung,
                             rel_l2=m["rel_l2"], epochs_run=m["epochs_run"],
                             wall_s=m["wall_s"], params=n_params(model),
                             collapse=m["trivial_collapse"],
                             diverged=m.get("diverged")))
            print(f"{tag:<14}{sd:>5}{m['rel_l2']:>9.4f}%{m['epochs_run']:>8}"
                  f"{m['wall_s']:>9.0f}{n_params(model):>9}"
                  + ("  COLLAPSE" if m["trivial_collapse"] else "")
                  + ("  DIVERGED" if m.get("diverged") else ""), flush=True)
            json.dump(rows, open(A.out, "w"), indent=1)
            del model
            torch.cuda.empty_cache()
    print("done", flush=True)


if __name__ == "__main__":
    main()
