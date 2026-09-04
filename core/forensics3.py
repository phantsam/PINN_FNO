"""What changed between the KAN that LOST and the KAN that TIED?

Phase 6/7 baseline pykan (grid=5, raw (x,t) input) reached 0.200 % on homogeneous
and lost to the best PINN by 1.7x.  The Phase 8 rung `charcoords50`
(characteristic coordinates + grid=50 + k=5) reached 0.1115 % +- 0.0026 over three
seeds and TIED PirateNet's 0.1114 % with 4.1x fewer parameters.

Both are pykan.  Same library, same trainer, same ansatz, same metric.  So the
difference is entirely in (a) what goes IN and (b) how finely the splines can
resolve it.  This module pushes the identical evaluation batch through both and
reports the same four measurements forensics.py used, so the two can be read side
by side:

    activation range vs the layer's own grid   -- is the support being used?
    effective rank                             -- how many directions are live?
    linear probe                               -- is the target linearly present?
    spline nonlinearity per edge               -- is the machinery working?

Phase 8 saved no checkpoints (--ckpt was never passed), so the winner is retrained
here with the exact rung configuration before analysis.  It early-stops around
290-320 epochs, so this is cheap.
"""
from __future__ import annotations
import argparse, os
import numpy as np
import torch

from .problem import MATERIALS
from .reference import fd_reference
from .operators import make_ansatz
from .models import REGISTRY, n_params
from .train import train_lbfgs
from .forensics import effective_rank, linear_probe, hi_freq_fraction
from .forensics2 import spline_nonlinearity
from .phase8 import RUNGS, build


def eval_batch(material, device, sigma_g=0.1, T=1.0, n_snap=20, nx=512):
    x_np, t_np, u_np = fd_reference(material, nx=nx, T=T, sigma_g=sigma_g)
    ev = [int(np.argmin(np.abs(t_np - v))) for v in np.linspace(0.05, T, n_snap)]
    X = torch.tensor(x_np, dtype=torch.float32, device=device).reshape(-1, 1)
    tt = torch.tensor(t_np[ev], dtype=torch.float32, device=device)
    xg = X.repeat(len(ev), 1)
    tg = tt.repeat_interleave(len(x_np)).reshape(-1, 1)
    ug = torch.tensor(u_np[ev], dtype=torch.float32, device=device).reshape(-1, 1)
    ans = make_ansatz("legacy", sigma_g=sigma_g)
    with torch.no_grad():
        ic = ans(torch.zeros_like(ug), xg, tg)
        gr = ans(torch.ones_like(ug), xg, tg) - ic
    keep = gr.abs().squeeze(-1) > 1e-3
    N = torch.zeros_like(ug)
    N[keep] = (ug[keep] - ic[keep]) / gr[keep]
    return xg, tg, N, keep, nx, len(ev), (x_np, t_np[ev], u_np[ev])


def report(tag, model, xg, tg, N, keep, nx, nt, rel_l2):
    """Identical measurements for any pykan-backed model."""
    print(f"\n{'='*84}\n  {tag}\n  {n_params(model):,} trainable params   rel-L2 = {rel_l2}\n{'='*84}")
    with torch.no_grad():
        model.kan.save_act = True
        inp = model._inputs(xg, tg) if hasattr(model, "_inputs") else torch.cat([xg, tg], -1)
        model.kan(inp)
        acts = list(model.kan.acts)

    # what the model is fed
    s_in = torch.linalg.svdvals((inp - inp.mean(0, keepdim=True)).double())[:4]
    print(f"\n  INPUT to the KAN: {inp.shape[1]}-vector, "
          f"range [{float(inp.min()):+.3f}, {float(inp.max()):+.3f}]")
    print(f"    singular values  {'  '.join(f'{float(v):7.2f}' for v in s_in)}")

    print(f"\n  {'layer':<8}{'width':>7}{'obs range':>22}{'grid span':>20}"
          f"{'grid used':>11}{'eff.rank':>10}{'probe':>10}{'hi-k':>8}")
    for i, a in enumerate(acts):
        a = a.detach()
        if i < len(model.kan.act_fun):
            g = model.kan.act_fun[i].grid
            glo, ghi = float(g.min()), float(g.max())
            used = 100.0 * (float(a.max()) - float(a.min())) / (ghi - glo)
            gs = f"[{glo:+.2f}, {ghi:+.2f}]"
            us = f"{used:.0f}%"
        else:
            gs, us = "-", "-"
        print(f"  {'acts'+str(i):<8}{a.shape[1]:>7}"
              f"{'[%+.3f, %+.3f]' % (float(a.min()), float(a.max())):>22}{gs:>20}{us:>11}"
              f"{effective_rank(a[keep]):>10.2f}{linear_probe(a[keep], N[keep]):>9.3f}%"
              f"{hi_freq_fraction(a, nx, nt):>8.3f}")

    print(f"\n  SPLINE NONLINEARITY per edge (over the range each edge actually sees)")
    print(f"  {'layer':<8}{'edges':>8}{'knots':>8}{'spacing':>10}{'mean':>9}{'~linear':>10}")
    for i, l in enumerate(model.kan.act_fun):
        m, md, fr = spline_nonlinearity(l, acts[i])
        g = l.grid
        sp = float(g[0][1] - g[0][0])
        print(f"  {'layer'+str(i):<8}{l.in_dim*l.out_dim:>8}{g.shape[1]:>8}{sp:>10.4f}"
              f"{m:>9.3f}{100*fr:>9.1f}%")
    model.kan.save_act = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", default="homogeneous")
    ap.add_argument("--rung", default="charcoords50")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--baseline-ckpt", default=None)
    A = ap.parse_args()
    dev = torch.device(f"cuda:{A.gpu}")
    M = MATERIALS[A.material]()
    xg, tg, N, keep, nx, nt, ref = eval_batch(M, dev)
    x_np, t_ev, u_ev = ref

    # ---- the LOSER: baseline pykan from a Phase 6/7 checkpoint -------------
    if A.baseline_ckpt and os.path.exists(A.baseline_ckpt):
        torch.manual_seed(0)
        base = REGISTRY["pykan_wide"]().to(dev)
        sd = torch.load(A.baseline_ckpt, map_location=dev, weights_only=False)
        base.load_state_dict(sd["state_dict"]); base.eval()
        report(f"BASELINE pykan  (grid=5, raw (x,t))  [{os.path.basename(A.baseline_ckpt)}]",
               base, xg, tg, N, keep, nx, nt, f"{sd['metrics']['rel_l2']:.4f} %")

    # ---- the WINNER: retrain the ladder rung (Phase 8 saved no checkpoints)
    print(f"\n\n  retraining rung '{A.rung}' on {A.material} seed {A.seed} ...", flush=True)
    _, sched = RUNGS[A.rung]
    model = build(A.rung, M, seed=A.seed + 1).to(dev)
    model.set_save_act(False)
    idx = [int(np.argmin(np.abs(t_ev - v))) for v in t_ev]
    model, m = train_lbfgs(lambda: model, M, epochs=A.epochs, seed=A.seed, device=dev,
                           use_r3=False, x_ref=x_np, t_ref=t_ev, u_ref=u_ev)
    report(f"WINNER '{A.rung}'  (characteristic coords, grid=50, k=5)",
           model, xg, tg, N, keep, nx, nt,
           f"{m['rel_l2']:.4f} %  ({m['epochs_run']} ep)")


if __name__ == "__main__":
    main()
