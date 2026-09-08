"""Can these models REPRESENT the solution at all, or only not find it?

The Fourier-embedded KANs collapse to u == 0 under the PDE loss.  Two very
different things produce that, and they need different responses:

  REPRESENTATION failure -- the architecture cannot express the solution, so
      u == 0 really is close to the best it can do and the optimiser is behaving
      correctly.
  OPTIMISATION failure -- the architecture can express the solution perfectly
      well, but the PDE residual landscape leads the optimiser to the trivial
      minimum instead.

Fitting u_ref directly by MSE, with no PDE loss anywhere, separates them.  If a
model reaches 0.01 % supervised and 95 % under the physics loss, the basis is
fine and the loss landscape is the problem.

This is the experiment that was abandoned four times earlier in the project.
The four designs failed for reasons worth naming, because each is avoided here:
  1. `tolerance_grad=1e-7` was an ABSOLUTE threshold compared against a loss of
     wildly different scale, so runs stopped early at random points.
  2. float32 put a floor under the achievable error that was mistaken for a
     capacity ceiling.
  3. the fit set and the eval set differed, so the number reported was
     generalisation, not representation.
  4. runs hit the epoch cap and the cap was read as convergence.
Here: relative stopping criterion, float64 for the fit, the ceiling is reported
ON the fit set (a separate held-out number is reported alongside, clearly
labelled), and whether the cap was hit is printed for every run.
"""
from __future__ import annotations
import argparse, json
import numpy as np
import torch
import torch.optim as optim

from .problem import MATERIALS
from .reference import fd_reference
from .operators import make_ansatz
from .models import REGISTRY
from .phase8 import build as build_rung
from .metrics import spacetime_rel_l2
from .rescore import fine_reference, TARGETS


def make(arch, M, seed):
    if arch.startswith("rung:"):
        m = build_rung(arch.split(":", 1)[1], M, seed=seed + 1)
        m.set_save_act(False)
        return m
    return REGISTRY[arch]()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", default="homogeneous")
    ap.add_argument("--models", default="fourier,rung:charcoords50,rung:fourier16,splinekan_fix,pykan_wide")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--adam", type=int, default=4000)
    ap.add_argument("--lbfgs", type=int, default=1500)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default=None)
    A = ap.parse_args()
    dev = torch.device(f"cuda:{A.gpu}")
    M = MATERIALS[A.material]()

    # target: the FD solution on a dense space-time grid, the SAME points used
    # to report the ceiling.  A held-out grid (offset in x and t) is scored too.
    xr, ur = fine_reference(M, 2048)
    X = torch.tensor(xr, dtype=torch.float32, device=dev).reshape(-1, 1)
    tt = torch.tensor(TARGETS, dtype=torch.float32, device=dev)
    xf = X.repeat(len(TARGETS), 1)
    tf = tt.repeat_interleave(len(xr)).reshape(-1, 1)
    uf = torch.tensor(ur.reshape(-1, 1), dtype=torch.float32, device=dev)

    ans = make_ansatz("legacy", sigma_g=0.1)
    print(f"SUPERVISED FIT to u_ref -- no PDE loss.  {A.material}, "
          f"{len(uf):,} points, Adam({A.adam}) + L-BFGS({A.lbfgs})\n")
    print(f"{'model':<20}{'seed':>5}{'fit rel-L2':>13}{'final MSE':>12}"
          f"{'hit cap?':>10}{'  physics-loss result':<22}")
    PHYS = {"fourier": "0.0391 %", "rung:charcoords50": "0.0236 %",
            "rung:fourier16": "94.98 % COLLAPSED", "splinekan_fix": "97.4 % COLLAPSED",
            "pykan_wide": "0.1001 %"}
    rows = []
    for arch in A.models.split(","):
        for sd in [int(s) for s in A.seeds.split(",")]:
            torch.manual_seed(sd)
            m = make(arch, M, sd).to(dev)
            u_fn = lambda x, t: ans(m(x, t), x, t)
            live = [q for q in m.parameters() if q.requires_grad]
            opt = optim.Adam(live, lr=1e-3)
            sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=A.adam, eta_min=1e-5)
            for _ in range(A.adam):
                opt.zero_grad(set_to_none=True)
                l = ((u_fn(xf, tf) - uf) ** 2).mean()
                if not torch.isfinite(l):
                    break
                l.backward(); opt.step(); sch.step()
            opt = optim.LBFGS(live, lr=1.0, max_iter=20, history_size=100,
                              line_search_fn="strong_wolfe")
            best, wait, ep = float("inf"), 0, 0
            for ep in range(A.lbfgs):
                def cl():
                    opt.zero_grad(set_to_none=True)
                    l = ((u_fn(xf, tf) - uf) ** 2).mean()
                    if torch.isfinite(l):
                        l.backward()
                    return l
                v = float(opt.step(cl).detach())
                if not np.isfinite(v):
                    break
                # RELATIVE stopping criterion -- pitfall (1) above
                if v < best * (1 - 1e-6):
                    best, wait = v, 0
                else:
                    wait += 1
                    if wait >= 50:
                        break
            with torch.no_grad():
                pred = u_fn(xf, tf).reshape(len(TARGETS), -1).cpu().numpy()
            e = float(spacetime_rel_l2(pred, ur))
            capped = "YES" if ep >= A.lbfgs - 1 else "no"
            print(f"{arch:<20}{sd:>5}{e:>12.4f}%{best:>12.3e}{capped:>10}"
                  f"  {PHYS.get(arch,''):<22}", flush=True)
            rows.append(dict(model=arch, seed=sd, fit_rel_l2=e, mse=best, capped=capped))
            del m; torch.cuda.empty_cache()
    if A.out:
        json.dump(rows, open(A.out, "w"), indent=1)
    print("\n  A model that fits well here but collapses under the PDE loss has a")
    print("  LOSS-LANDSCAPE problem, not a basis problem.")


if __name__ == "__main__":
    main()
