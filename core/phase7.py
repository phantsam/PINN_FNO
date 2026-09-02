"""Phase 7 -- remove the epoch-budget confound from the PINN-vs-KAN verdict.

Phase 6 capped every run at 700 epochs.  That is not a neutral control: it is a
budget that truncates whichever architecture converges slowest.  Measured cap-hit
rates over the Phase 6 grid:

    pykan 57%   PirateNet 25%   MLP 22%   Fourier 12%   WavKAN 6%

So the KAN arm was being stopped mid-descent far more often than the PINN arm,
and every capped number is an upper bound on that cell's achievable error rather
than the error itself.  Since the question is explicitly "which reaches the
lowest error", with training time declared irrelevant, the cap has to go.

Design
  * Cells that early-stopped in Phase 6 are ALREADY converged -- patience=50 was
    exhausted, i.e. 50 consecutive epochs with no improvement > min_delta.  Their
    numbers stand; re-running them is wasted compute and would reproduce the same
    trajectory bit-for-bit (same seed, same Sobol set, deterministic objective).
  * Only cells that hit the cap are re-run, from scratch, with a 3000-epoch
    budget and the same patience-based stopping.  Same seed => the first 700
    epochs replay identically, then continue.
  * Runs that still hit 3000 are reported as still-capped, not silently averaged.

Also picks up the train_lbfgs divergence guard (fp32 overflow in torch's
_cubic_interpolate), so a line-search blow-up now returns best-so-far weights
instead of NaN.
"""
import argparse, json, os
import numpy as np, torch
from core.problem import MATERIALS
from core.reference import fd_reference
from core.models import REGISTRY, n_params
from core.train import train_lbfgs

ap = argparse.ArgumentParser()
ap.add_argument("--src", nargs="+", required=True, help="phase6 result jsons")
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=3000)
ap.add_argument("--cap", type=int, default=700, help="phase6 cap that defines 'truncated'")
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--ckpt", default=None)
A = ap.parse_args()
dev = torch.device(f"cuda:{A.gpu}")

src = []
for f in A.src:
    if os.path.exists(f):
        src += json.load(open(f))

# pykan cells are re-run REGARDLESS of the cap: their Phase 6 seeding was void.
# MultKAN.__init__ overrode the trainer's torch.manual_seed with its own default,
# so all three "seeds" produced the same initialisation (verified: seed0-vs-seed2
# parameter difference 1.49e-08, identical in magnitude to seed0-vs-seed0).  The
# pykan seed spread in Phase 6 therefore measured collocation-set variance only.
todo = [r for r in src if r["epochs_run"] >= A.cap
        or not np.isfinite(r["rel_l2"])
        or r["arch"] == "pykan_wide"]
todo.sort(key=lambda r: (r["material"], r["seed"], r["arch"], r["r3"]))
todo = todo[A.shard::A.nshard]

rows = json.load(open(A.out)) if os.path.exists(A.out) else []
done = {(r["material"], r["seed"], r["arch"], r["r3"]) for r in rows}
print(f"shard {A.shard}/{A.nshard}: {len(todo)} truncated cells to extend", flush=True)

refs = {}
for r in todo:
    mk, sd, ak, r3 = r["material"], r["seed"], r["arch"], r["r3"]
    if (mk, sd, ak, r3) in done:
        continue
    if mk not in refs:
        M = MATERIALS[mk]()
        x, t, u = fd_reference(M, nx=512, T=1.0, sigma_g=0.1)
        i = [int(np.argmin(np.abs(t - v))) for v in np.linspace(0.05, 1.0, 20)]
        refs[mk] = (M, x, t[i], u[i])
    M, x, tt, uu = refs[mk]
    sp = None
    if A.ckpt:
        os.makedirs(A.ckpt, exist_ok=True)
        sp = os.path.join(A.ckpt, f"{mk}_s{sd}_{ak}_{'r3' if r3 else 'plain'}.pt")
    try:
        mdl, m = train_lbfgs(REGISTRY[ak], M, epochs=A.epochs, seed=sd, device=dev,
                             use_r3=r3, save_path=sp, x_ref=x, t_ref=tt, u_ref=uu)
    except Exception as e:
        print(f"{mk:<12}s{sd} {ak:<11}{'R3' if r3 else '--':<4} FAILED {type(e).__name__}: {e}", flush=True)
        continue
    rows.append(dict(material=mk, seed=sd, arch=ak, r3=r3, params=n_params(mdl),
                     rel_l2=m["rel_l2"], resid=m["residual_rms_heldout"],
                     collapse=m["trivial_collapse"], epochs_run=m["epochs_run"],
                     wall_s=m["wall_s"], r3_retained=m.get("r3_retained"),
                     diverged=m.get("diverged"), prev_rel_l2=r["rel_l2"],
                     prev_epochs=r["epochs_run"]))
    still = "CAP" if m["epochs_run"] >= A.epochs else "   "
    print(f"{mk:<12}s{sd} {ak:<11}{'R3' if r3 else '--':<4} "
          f"{r['rel_l2']:7.3f}% -> {m['rel_l2']:7.3f}%  {m['epochs_run']:>5}ep {m['wall_s']:6.0f}s {still}"
          + ("  DIVERGED" if m.get("diverged") else ""), flush=True)
    json.dump(rows, open(A.out, "w"), indent=1)
print("done", flush=True)
