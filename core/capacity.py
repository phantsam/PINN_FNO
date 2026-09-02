"""Experiment 1 -- representation capacity, measured independently of the PDE.

The question this settles
-------------------------
Every number in Phases 6-8 comes from minimising a PDE residual.  A poor score
therefore admits two incompatible explanations:

  (a) REPRESENTATION limit -- the architecture cannot express u(x,t) at all;
  (b) OPTIMISATION limit  -- it can, but L-BFGS on a second-derivative residual
      cannot locate the parameters that do.

No residual-based experiment can distinguish them, because the residual is the
thing under suspicion.  So: throw the PDE away and fit the KNOWN reference
solution directly,

      min_theta  || u_theta(x,t) - u_ref(x,t) ||^2 ,

with the identical hard-IC ansatz, the identical optimiser, and a generous
budget.  The result is an upper bound on what ANY residual-based training of
that architecture could achieve.

Reading the outcome
-------------------
  supervised ~ PINN-trained      -> the architecture is representation-limited;
                                    no optimiser or loss change can help.
  supervised << PINN-trained     -> the architecture CAN express the solution and
                                    the deficit is optimisation; the residual
                                    formulation, not the basis, is at fault.

The comparison is only meaningful if both arms get the same treatment, so the
ansatz, the metric, the reference and the optimiser are shared verbatim with the
training path -- only the loss changes.
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import torch
from torch import optim

from .problem import MATERIALS
from .reference import fd_reference
from .operators import make_ansatz
from .models import REGISTRY, n_params
from .kan_variants import TunedKAN

# The plain REGISTRY entry is the UNTUNED grid=5 KAN.  Measuring only its ceiling
# would understate what a KAN can represent, so the tuned ladder configurations
# are measured too -- otherwise the capacity comparison repeats exactly the
# unfairness Phase 8 exists to remove.
def _kan(**kw):
    def f(material=None, t_max=1.0):
        return TunedKAN(width=(2, 20, 20, 20, 1), seed=1,
                        x_min=material.x_min, x_max=material.x_max, t_max=t_max, **kw)
    return f

TUNED = {
    "kan_g5":        _kan(grid=5,  k=3, normalise=False),   # == REGISTRY pykan_wide
    "kan_g5n":       _kan(grid=5,  k=3, normalise=True),
    "kan_g20":       _kan(grid=20, k=3, normalise=True),    # ladder's best rung
    "kan_g50":       _kan(grid=50, k=3, normalise=True),
    "kan_g100":      _kan(grid=100, k=3, normalise=True),   # far past the useful range
}
from .metrics import spacetime_rel_l2


def supervised_fit(arch_fn, material, *, n_snap=100, nx=512, sigma_g=0.1, T=1.0,
                   epochs=800, seed=0, device=None, ansatz_kind="legacy",
                   eval_snaps=20, log=None, dtype=torch.float64, adam_warm=4000):
    """Fit u_ref directly.  Returns (model, metrics)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)

    x_np, t_np, u_np = fd_reference(material, nx=nx, T=T, sigma_g=sigma_g)

    # TRAIN ON EXACTLY THE EVALUATION SET.
    #
    # The point of this experiment is a BOUND: the best rel-L2 the architecture
    # can possibly reach on the scored points.  Fitting a denser, different set
    # (100 snapshots) optimises a different objective from the metric (20
    # snapshots), so the result was not a bound at all -- measured, fourier
    # "ceiled" at 0.1221% while PDE training of the same model reaches 0.110%.
    #
    # Minimising MSE on precisely the scored points makes rel-L2 a monotone
    # function of the training loss, so a converged fit IS a strict upper bound
    # on what any other training procedure can achieve on that metric.
    # Generalisation is not the question here; representable accuracy is.
    ev_idx = [int(np.argmin(np.abs(t_np - v)))
              for v in np.linspace(0.05, T, eval_snaps)]
    tr_idx = ev_idx

    # FLOAT64 IS REQUIRED HERE, unlike the training path.
    # The supervised loss bottoms out around 1e-7, where float32's ~1.2e-7
    # relative precision makes the gradient indistinguishable from rounding
    # noise: traced, the fit stalled at mse=2.76e-07 with maxgrad=3.86e-07, i.e.
    # a measured "ceiling" of 0.2336% that sat ABOVE what PDE training of the
    # same model reaches (0.169%).  A bound that the thing it bounds can beat is
    # not a bound -- it was measuring float32, not the architecture.
    X = torch.tensor(x_np, dtype=dtype, device=device).reshape(-1, 1)
    def grid(idx):
        tt = torch.tensor(t_np[idx], dtype=dtype, device=device)
        xg = X.repeat(len(idx), 1)
        tg = tt.repeat_interleave(len(x_np)).reshape(-1, 1)
        ug = torch.tensor(u_np[idx], dtype=dtype, device=device).reshape(-1, 1)
        return xg, tg, ug
    xtr, ttr, utr = grid(tr_idx)
    xev, tev, uev = grid(ev_idx)

    model = arch_fn().to(device).to(dtype)
    ansatz = make_ansatz(ansatz_kind, sigma_g=sigma_g)
    u_fn = lambda a, b: ansatz(model(a, b), a, b)
    if hasattr(model, "physics_informed_init"):
        from .problem import gaussian_ic
        model.physics_informed_init(xtr[:4096], ttr[:4096], gaussian_ic(xtr[:4096], sigma_g))

    live = [q for q in model.parameters() if q.requires_grad]
    # Adam warm start.  L-BFGS from a cold init converges at very different rates
    # for different architectures -- measured at an identical 6000-epoch budget,
    # fourier reached 0.0739% while mlp was still at 0.4091%, ABOVE what PDE
    # training of the same mlp achieves (0.259%).  A "ceiling" that the thing it
    # bounds can beat is not a ceiling, and if that bias is architecture-dependent
    # the experiment measures how easily L-BFGS fits each model rather than what
    # each model can represent.  A first-order warm start puts every architecture
    # in the same basin regime before the quasi-Newton phase begins.
    if adam_warm:
        ad = optim.Adam(live, lr=1e-3)
        for _ in range(adam_warm):
            ad.zero_grad(set_to_none=True)
            l = ((u_fn(xtr, ttr) - utr) ** 2).mean()
            if not torch.isfinite(l):
                break
            l.backward(); ad.step()

    # tolerance_grad / tolerance_change are ABSOLUTE thresholds, and this loss
    # lives at ~1e-7 where torch's 1e-7 default bites immediately: measured, the
    # MSE froze bit-exactly at 3.970463e-07 from epoch 39 for 200+ steps because
    # LBFGS returned at its first inner iteration without moving.  The PDE path
    # never saw this (its loss is O(100)).  Disable both so the stopping decision
    # is made by our own patience rule, not by the objective's absolute scale.
    opt = optim.LBFGS(live, lr=1.0, max_iter=20, history_size=100,
                      line_search_fn="strong_wolfe",
                      tolerance_grad=0.0, tolerance_change=0.0)
    best, best_w, wait, t0 = float("inf"), None, 0, time.time()
    for ep in range(epochs):
        def closure():
            opt.zero_grad(set_to_none=True)
            l = ((u_fn(xtr, ttr) - utr) ** 2).mean()
            if torch.isfinite(l):
                l.backward()
            return l
        v = float(opt.step(closure))
        if not all(torch.isfinite(q).all() for q in model.parameters()) or not np.isfinite(v):
            break
        if v < best - 1e-20:
            best, wait = v, 0
            best_w = {k: c.detach().clone() for k, c in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 150:
                break
    if best_w:
        model.load_state_dict(best_w)

    with torch.no_grad():
        pred = u_fn(xev, tev).reshape(eval_snaps, -1).cpu().numpy()
    ref = u_np[ev_idx]
    return model, dict(rel_l2=float(spacetime_rel_l2(pred, ref)),
                       mse=best, epochs_run=ep + 1, wall_s=time.time() - t0,
                       params=n_params(model))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", default="fourier,mlp,pirate,pykan_wide,wavkan")
    ap.add_argument("--materials", default="homogeneous,twolayer,multilayer")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    A = ap.parse_args()
    dev = torch.device(f"cuda:{A.gpu}")
    rows = json.load(open(A.out)) if os.path.exists(A.out) else []
    # a row that ends ON the cap is not converged and must not be read as a bound
    done = {(r["material"], r["seed"], r["arch"]) for r in rows}
    print(f"{'material':<12}{'seed':>5} {'arch':<12}{'sup rel_L2':>12}{'params':>9}"
          f"{'ep':>6}{'s':>7}", flush=True)
    for mk in A.materials.split(","):
        M = MATERIALS[mk]()
        for sd in [int(s) for s in A.seeds.split(",")]:
            for ak in A.archs.split(","):
                if (mk, sd, ak) in done:
                    continue
                try:
                    fn = (lambda: TUNED[ak](M)) if ak in TUNED else REGISTRY[ak]
                    _, m = supervised_fit(fn, M, epochs=A.epochs,
                                          seed=sd, device=dev)
                except Exception as e:
                    print(f"{mk:<12}{sd:>5} {ak:<12} FAILED {type(e).__name__}: {e}", flush=True)
                    continue
                rows.append(dict(material=mk, seed=sd, arch=ak, **m))
                tag = "CAP-NOT-CONVERGED" if m["epochs_run"] >= A.epochs else ""
                print(f"{mk:<12}{sd:>5} {ak:<12}{m['rel_l2']:>11.4f}%{m['params']:>9}"
                      f"{m['epochs_run']:>6}{m['wall_s']:>7.0f}  {tag}", flush=True)
                json.dump(rows, open(A.out, "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
