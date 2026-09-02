"""Phase 3 -- ansatz A/B.

Question: does `poly` (u = g + t^2 N) beat `legacy` (g*decay + growth*N)?

Hypothesis under test: the legacy decay term dies by t~0.25, so a model that
collapses to ~0 after the handover pays almost no residual over most of the
domain while being ~95% wrong -- a trivial basin created by the ansatz, not by
the physics.  `poly` has no dead decay term.  Its own failure mode is different
(frozen IC, N->0), which the two-sided guard now detects.

Everything else is held fixed.  Only the ansatz varies.
"""
import argparse, json, itertools
import numpy as np, torch
from core.problem import MATERIALS
from core.reference import fd_reference
from core.models import REGISTRY, n_params
from core.train import train

ap = argparse.ArgumentParser()
ap.add_argument("--materials", default="homogeneous,twolayer")
ap.add_argument("--archs", default="pirate,wavkan,fourier")
ap.add_argument("--steps", type=int, default=8000)
ap.add_argument("--seeds", default="0")
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--out", default="phase3.json")
A = ap.parse_args()
dev = torch.device(f"cuda:{A.gpu}")

rows = []
for mk in A.materials.split(","):
    M = MATERIALS[mk]()
    x, t, u = fd_reference(M, nx=512, T=1.0, sigma_g=0.1)
    idx = [int(np.argmin(np.abs(t - v))) for v in np.linspace(0.05, 1.0, 20)]
    for ak, ans, sd in itertools.product(A.archs.split(","), ("legacy", "poly"),
                                         [int(s) for s in A.seeds.split(",")]):
        mdl, m = train(REGISTRY[ak], M, ansatz_kind=ans, steps=A.steps, seed=sd,
                       device=dev, eval_every=A.steps // 4,
                       x_ref=x, t_ref=t[idx], u_ref=u[idx])
        r = dict(material=mk, arch=ak, ansatz=ans, seed=sd, params=n_params(mdl),
                 rel_l2=m["rel_l2"], resid=m["residual_rms_heldout"],
                 collapse=m["trivial_collapse"], min_ratio=m["min_late_norm_ratio"],
                 wall_s=m["wall_s"], hist=m["hist"])
        rows.append(r)
        print(f"{mk:<12}{ak:<9}{ans:<7}s{sd}  L2 {m['rel_l2']:7.2f}%  "
              f"resid {m['residual_rms_heldout']:.2e}  ratio {m['min_late_norm_ratio']:.2f}"
              f"{'  COLLAPSE' if m['trivial_collapse'] else ''}  {m['wall_s']:.0f}s", flush=True)
        json.dump(rows, open(A.out, "w"), indent=1)
print("done")
