"""Re-score every headline model against a REFERENCE FINE ENOUGH TO RESOLVE IT.

The problem
-----------
Phases 6-9 scored against an nx=512 finite-difference solution.  With snapshot
times matched exactly, that reference converges at order 2.02/2.07/2.30 -- the
solver is correct -- but its OWN error against nx=4096 is:

    homogeneous  0.1110 %      best PINN 0.1114 %   best KAN 0.1115 %
    multilayer   0.1597 %      best PINN 0.1617 %   best KAN 0.1627 %
    twolayer     0.1829 %      best PINN 0.3944 %   best KAN 0.5563 %

On the first two materials the instrument's error EQUALS the quantity being
measured.  A model reporting 0.1114 % might be 0.11 % from the truth, or might be
nearly exact and merely disagreeing with the reference's own discretisation
error.  The reported "tie" is therefore not a measured tie -- it is two numbers
that cannot be separated.

At nx=2048 the reference error drops to 0.005-0.009 %, roughly 20x below the
model errors, which resolves everything cleanly.

Why this is cheap
-----------------
Training never uses the reference -- the loss is the PDE residual, and the
reference exists only to score.  So the model weights are IDENTICAL under either
reference and the PINN arm is a pure re-evaluation of saved checkpoints.  Only
the KAN needs retraining, because Phase 8 was run without --ckpt.

A second correction, independent of resolution
----------------------------------------------
`evaluate` picked snapshots by nearest-neighbour in time, and dt differs between
grids, so "the same" snapshot sat at a different instant on each grid.  With c=1
a time offset shifts the pulse by that amount in x; at ||u'||/||u|| ~ 14 for a
sigma=0.1 Gaussian derivative, a 4.4e-4 offset alone produces ~0.6 % apparent
error.  That artefact made the reference look FIRST order (1.09/1.04/0.74) when
it is second.  Snapshots here are interpolated linearly in time to the exact
target instants, an O(dt^2) operation.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch

from .problem import MATERIALS
from .reference import fd_reference
from .operators import make_ansatz
from .models import REGISTRY, n_params
from .metrics import spacetime_rel_l2
from .train import train_lbfgs
from .phase8 import RUNGS, build

TARGETS = np.linspace(0.05, 1.0, 20)


def fine_reference(material, nx, sigma_g=0.1, T=1.0):
    """FD solution at the EXACT target instants (linear in t, O(dt^2))."""
    x, t, u = fd_reference(material, nx=nx, T=T, sigma_g=sigma_g)
    out = []
    for tv in TARGETS:
        j = min(max(int(np.searchsorted(t, tv)), 1), len(t) - 1)
        w = (tv - t[j - 1]) / (t[j] - t[j - 1])
        out.append((1 - w) * u[j - 1] + w * u[j])
    return x, np.stack(out)


def score(model, material, x_ref, u_ref, device, sigma_g=0.1):
    X = torch.tensor(x_ref, dtype=torch.float32, device=device).reshape(-1, 1)
    tt = torch.tensor(TARGETS, dtype=torch.float32, device=device)
    xg = X.repeat(len(TARGETS), 1)
    tg = tt.repeat_interleave(len(x_ref)).reshape(-1, 1)
    ans = make_ansatz("legacy", sigma_g=sigma_g)
    with torch.no_grad():
        pred = ans(model(xg, tg), xg, tg).reshape(len(TARGETS), -1).cpu().numpy()
    return float(spacetime_rel_l2(pred, u_ref))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=2048)
    ap.add_argument("--materials", default="homogeneous,multilayer")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--pinns", default="pirate,fourier")
    ap.add_argument("--rung", default="charcoords50")
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--ckpt-dirs", nargs="+", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    A = ap.parse_args()
    dev = torch.device(f"cuda:{A.gpu}")
    seeds = [int(s) for s in A.seeds.split(",")]
    rows = json.load(open(A.out)) if os.path.exists(A.out) else []
    done = {(r["material"], r["seed"], r["model"]) for r in rows}

    print(f"re-scoring against nx={A.nx} (times matched exactly)\n")
    print(f"{'material':<13}{'model':<14}{'seed':>5}{'rel-L2 @2048':>14}", flush=True)
    for mk in A.materials.split(","):
        M = MATERIALS[mk]()
        xr, ur = fine_reference(M, A.nx)

        # PINN arm: pure re-evaluation, weights untouched
        for arch in A.pinns.split(","):
            for sd in seeds:
                if (mk, sd, arch) in done:
                    continue
                ck = None
                for d in A.ckpt_dirs:
                    p = os.path.join(d, f"{mk}_s{sd}_{arch}_plain.pt")
                    if os.path.exists(p):
                        ck = p
                if ck is None:
                    print(f"{mk:<13}{arch:<14}{sd:>5}   no checkpoint"); continue
                torch.manual_seed(0)
                m = REGISTRY[arch]().to(dev)
                sdict = torch.load(ck, map_location=dev, weights_only=False)
                m.load_state_dict(sdict["state_dict"]); m.eval()
                v = score(m, M, xr, ur, dev)
                rows.append(dict(material=mk, seed=sd, model=arch, rel_l2=v,
                                 nx=A.nx, params=n_params(m),
                                 old_rel_l2=sdict["metrics"]["rel_l2"]))
                print(f"{mk:<13}{arch:<14}{sd:>5}{v:>13.4f}%   (was "
                      f"{sdict['metrics']['rel_l2']:.4f} @512)", flush=True)
                json.dump(rows, open(A.out, "w"), indent=1)
                del m; torch.cuda.empty_cache()

        # KAN arm: retrain (Phase 8 saved no checkpoints), then score
        for sd in seeds:
            if (mk, sd, A.rung) in done:
                continue
            model = build(A.rung, M, seed=sd + 1).to(dev)
            model.set_save_act(False)
            xc, tc, uc = fd_reference(M, nx=512, T=1.0, sigma_g=0.1)
            ev = [int(np.argmin(np.abs(tc - v))) for v in TARGETS]
            model, mm = train_lbfgs(lambda: model, M, epochs=A.epochs, seed=sd,
                                    device=dev, use_r3=False,
                                    x_ref=xc, t_ref=tc[ev], u_ref=uc[ev])
            v = score(model, M, xr, ur, dev)
            rows.append(dict(material=mk, seed=sd, model=A.rung, rel_l2=v,
                             nx=A.nx, params=n_params(model),
                             old_rel_l2=mm["rel_l2"]))
            print(f"{mk:<13}{A.rung:<14}{sd:>5}{v:>13.4f}%   (was "
                  f"{mm['rel_l2']:.4f} @512)", flush=True)
            json.dump(rows, open(A.out, "w"), indent=1)
            del model; torch.cuda.empty_cache()
    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
