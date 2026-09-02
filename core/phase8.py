"""Phase 8 -- the KAN strengthening ladder.

Purpose
-------
Phase 6 compared a KAN running library defaults against a PINN whose bandwidth
was fitted to the measured solution spectrum (DECISIONS.md D4).  That is not a
fair test of the architecture, and the unfairness was quantified:

    pykan layer-0 knot spacing 0.400  vs  Gaussian pulse FWHM 0.236
    -> the splines were coarser than the entire wave packet.
    grid_range=[-1,1] but t in [0,1] -> half the time knots unreachable.

So any claim that "KANs are worse for the 1D elastic wave equation" is currently
unsupportable.  This phase climbs a ladder of increasingly-strong KANs, ONE
controlled change at a time, so that a negative result -- if it survives -- is
attributable rather than assumed.

The literature makes this necessary, not optional: KINN (arXiv:2406.11045)
reports KAN beating MLP on heterogeneous solid mechanics, and arXiv:2602.15068
reports the same for wave propagation.  We reproduced the latter's KAN win
(0.1834% vs our 0.1834%).  Our stack is therefore not biased against KANs, which
means the Phase 6 gap needs a mechanism, and under-resolution is the candidate.

Two upstream defects had to be fixed before this ladder could run at all
(see kan_patch.py):
  * pykan's curve2coef returns NaN on CUDA for any rank-deficient design matrix,
    which every layer past the first is -- so grid refinement was silently broken.
  * the same routine is unstable at k=5 even at full rank (LSQ residual 58.65 vs
    the ridge solve's 0.0022), so the spline-order rung would have been garbage.

Staging
-------
  screen : every rung, 3 materials, 1 seed  -- ranks the rungs
  final  : selected rungs, 3 materials, 3 seeds, large budget -- the real numbers

Within `screen` a common epoch cap is fair because every entrant is a KAN of
similar cost.  `final` uses a budget large enough that convergence, not the cap,
terminates the run -- the Phase 7 lesson.
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np, torch

from .problem import MATERIALS
from .reference import fd_reference
from .models import n_params
from .train import train_lbfgs
from .kan_variants import TunedKAN
from .diagnostics import screen as health_screen, format_report

W = (2, 20, 20, 20, 1)

# Each rung changes ONE thing relative to `baseline`, except the explicit combos.
# `refine` is a schedule, not a constructor argument, so it lives beside cfg.
RUNGS = {
    #  name              constructor kwargs                                refine schedule
    "baseline":        (dict(width=W, grid=5,  k=3, base_fun="silu", normalise=False), None),
    "normalise":       (dict(width=W, grid=5,  k=3, base_fun="silu", normalise=True),  None),
    "grid20":          (dict(width=W, grid=20, k=3, base_fun="silu", normalise=True),  None),
    "grid50":          (dict(width=W, grid=50, k=3, base_fun="silu", normalise=True),  None),
    "refine":          (dict(width=W, grid=5,  k=3, base_fun="silu", normalise=True),  (10, 20, 50)),
    "k5":              (dict(width=W, grid=20, k=5, base_fun="silu", normalise=True),  None),
    "tanh":            (dict(width=W, grid=20, k=3, base_fun="tanh", normalise=True),  None),
    "sin":             (dict(width=W, grid=20, k=3, base_fun="sin",  normalise=True),  None),
    "gelu":            (dict(width=W, grid=20, k=3, base_fun="gelu", normalise=True),  None),
    "relu":            (dict(width=W, grid=20, k=3, base_fun="relu", normalise=True),  None),
    "fourier8":        (dict(width=W, grid=20, k=3, base_fun="silu", n_fourier=8),     None),
    "fourier16":       (dict(width=W, grid=20, k=3, base_fun="silu", n_fourier=16),    None),
    # Characteristic coordinates: (tau(x)-t, tau(x)+t).  For constant c the exact
    # solution is F(x-ct) + G(x+ct) -- a SUM OF UNIVARIATE FUNCTIONS, which is
    # precisely a Kolmogorov-Arnold representation.  This is the one rung with a
    # theoretical reason to favour a KAN specifically rather than every
    # architecture equally, so it is also the most informative if it fails.
    "charcoords":      (dict(width=W, grid=20, k=3, base_fun="silu", char_coords=True), None),
    "charcoords50":    (dict(width=W, grid=50, k=5, base_fun="silu", char_coords=True), None),
    # combination rung -- filled from the screen results, kept explicit for the record
    "best_combo":      (dict(width=W, grid=50, k=5, base_fun="sin",  normalise=True),  None),
}


def build(rung, material, t_max=1.0, seed=1):
    cfg, _ = RUNGS[rung]
    kw = dict(cfg)
    kw.setdefault("x_min", material.x_min)
    kw.setdefault("x_max", material.x_max)
    kw.setdefault("t_max", t_max)
    if kw.get("char_coords"):
        kw["material"] = material
    kw["seed"] = seed
    kw["ckpt"] = f"/tmp/_pykan_ckpt_{os.getpid()}"
    os.makedirs(kw["ckpt"], exist_ok=True)
    return TunedKAN(**kw)


def train_rung(rung, material, *, epochs, seed, device, x_ref, t_ref, u_ref,
               n_col=10000, t_max=1.0, full_budget_per_level=False):
    """Train one rung.  With a refine schedule this runs one L-BFGS segment per
    grid level, re-fitting the splines between segments; the model object is
    carried across segments, so `arch_fn` returns the SAME model each time and
    train_lbfgs's own seeding leaves it untouched."""
    _, sched = RUNGS[rung]
    model = build(rung, material, t_max=t_max, seed=seed + 1).to(device)
    model.set_save_act(False)                 # caching 10k activations is pure cost
    levels = [None] + list(sched or [])
    # Budget per grid level.  Dividing a fixed total across levels HANDICAPS the
    # refine rung: measured on homogeneous/s0 it converged at 757 of 800 total
    # epochs, i.e. every level ran into its own 200-epoch share, while the
    # single-level rungs had all 800.  refine then scored 0.1943% against
    # grid20's 0.1510%, which says more about the split than about refinement.
    # `full_budget_per_level` gives each level the full budget, matching the KAN
    # paper's protocol (train to convergence at each level).  Left OFF by default
    # so the screen shards already in flight stay mutually comparable; the final
    # stage turns it on for every rung.
    budget = epochs if full_budget_per_level else max(1, epochs // len(levels))
    tot_ep, tot_wall, m = 0, 0.0, None
    for i, lvl in enumerate(levels):
        if lvl is not None:
            g = torch.Generator(device=device).manual_seed(seed)
            xr = (torch.rand(4000, 1, generator=g, device=device)
                  * (material.x_max - material.x_min) + material.x_min)
            tr = torch.rand(4000, 1, generator=g, device=device) * t_max
            model.refine_(lvl, xr, tr)
            model.set_save_act(False)
        model, m = train_lbfgs(lambda: model, material, epochs=budget, seed=seed,
                               device=device, use_r3=False, n_col=n_col,
                               x_ref=x_ref, t_ref=t_ref, u_ref=u_ref)
        tot_ep += m["epochs_run"]; tot_wall += m["wall_s"]
    m["epochs_run"], m["wall_s"] = tot_ep, tot_wall
    return model, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["screen", "final"], default="screen")
    ap.add_argument("--rungs", default=None, help="comma list; default = all")
    ap.add_argument("--materials", default="homogeneous,twolayer,multilayer")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--full-budget-per-level", action="store_true",
                    dest="full_budget", help="each refine level gets the full "
                    "epoch budget instead of a share of it")
    A = ap.parse_args()

    dev = torch.device(f"cuda:{A.gpu}")
    rungs = A.rungs.split(",") if A.rungs else list(RUNGS)
    seeds = ([int(s) for s in A.seeds.split(",")] if A.seeds
             else ([0] if A.stage == "screen" else [0, 1, 2]))
    epochs = A.epochs or (1000 if A.stage == "screen" else 3000)

    jobs = [(mk, sd, rg) for mk in A.materials.split(",")
            for sd in seeds for rg in rungs][A.shard::A.nshard]
    rows = json.load(open(A.out)) if os.path.exists(A.out) else []
    done = {(r["material"], r["seed"], r["rung"]) for r in rows}
    print(f"stage={A.stage} shard={A.shard}/{A.nshard} jobs={len(jobs)} "
          f"epochs={epochs}", flush=True)

    refs = {}
    for mk, sd, rg in jobs:
        if (mk, sd, rg) in done:
            continue
        if mk not in refs:
            M = MATERIALS[mk]()
            x, t, u = fd_reference(M, nx=512, T=1.0, sigma_g=0.1)
            i = [int(np.argmin(np.abs(t - v))) for v in np.linspace(0.05, 1.0, 20)]
            refs[mk] = (M, x, t[i], u[i])
        M, x, tt, uu = refs[mk]

        # Health screen BEFORE spending the budget -- this is how the D6 gradient
        # collapse and the SplineKAN u_tt=5e5 blow-up would have been caught early.
        try:
            probe = build(rg, M, seed=sd + 1).to(dev)
            ok, warn, rep = health_screen(probe, M, name=rg, device=dev,
                                          n=2048, n_col=2048, n_bc=256)
            del probe; torch.cuda.empty_cache()
        except Exception as e:
            print(f"{mk:<12}s{sd} {rg:<11} HEALTH-SCREEN FAILED "
                  f"{type(e).__name__}: {e}", flush=True)
            continue
        if warn:
            print(format_report(rep, warn), flush=True)
        if not ok:
            rows.append(dict(material=mk, seed=sd, rung=rg, rel_l2=float("nan"),
                             params=None, epochs_run=0, wall_s=0.0,
                             health_ok=False, warnings=warn))
            json.dump(rows, open(A.out, "w"), indent=1)
            print(f"{mk:<12}s{sd} {rg:<11} SKIPPED (failed health screen)", flush=True)
            continue

        try:
            mdl, m = train_rung(rg, M, epochs=epochs, seed=sd, device=dev,
                                x_ref=x, t_ref=tt, u_ref=uu,
                                full_budget_per_level=A.full_budget)
        except Exception as e:
            print(f"{mk:<12}s{sd} {rg:<11} FAILED {type(e).__name__}: {e}", flush=True)
            continue

        rows.append(dict(material=mk, seed=sd, rung=rg, params=n_params(mdl),
                         rel_l2=m["rel_l2"], resid=m["residual_rms_heldout"],
                         collapse=m["trivial_collapse"], epochs_run=m["epochs_run"],
                         wall_s=m["wall_s"], diverged=m.get("diverged"),
                         full_budget_per_level=bool(A.full_budget),
                         health_ok=True, warnings=warn,
                         u_tt_init=rep["u_tt"], grad_init=rep["global_grad_norm"]))
        cap = "CAP" if m["epochs_run"] >= epochs else "   "
        print(f"{mk:<12}s{sd} {rg:<11} L2 {m['rel_l2']:8.4f}%  "
              f"{n_params(mdl):>6}p {m['epochs_run']:>5}ep {m['wall_s']:6.0f}s {cap}"
              + ("  DIVERGED" if m.get("diverged") else "")
              + ("  COLLAPSE" if m["trivial_collapse"] else ""), flush=True)
        json.dump(rows, open(A.out, "w"), indent=1)
        del mdl; torch.cuda.empty_cache()
    print("done", flush=True)


if __name__ == "__main__":
    main()
