"""WHY is the KAN's seed variance ~5x the PINN's?

Measured at nx=2048, n=11 per arm on homogeneous:

    fourier       0.0281 +- 0.0073   (CV 26 %,  range 2.3x)
    charcoords50  0.0313 +- 0.0170   (CV 54 %,  range 5.1x)
    variance ratio 5.44x, Levene p = 0.031  -- the one significant difference

Three candidate mechanisms, each with a distinct measurable signature:

H1  COLLOCATION OVERFITTING (local vs global basis).
    Splines have LOCAL support: changing one coefficient moves the function only
    near its knot.  So a spline model can drive the residual to zero AT the 10,000
    Sobol collocation points while drifting between them.  Random Fourier features
    are GLOBAL -- every coefficient affects every point -- so fitting the
    collocation set constrains the function everywhere.
    Signature: residual on fresh points >> residual on training points, for the
    KAN but not the PINN.  And the gap should track the seed's rel-L2.

H2  DIFFERENT OPTIMISATION BASINS.
    Seeds converge to genuinely different parameter settings with different
    training losses.
    Signature: final training loss varies a lot across seeds, and correlates with
    rel-L2.

H3  THE RESIDUAL IS SATISFIED BUT THE SOLUTION IS WRONG.
    Seeds reach the SAME training loss yet different accuracy -- the discrete
    residual under-determines the solution for this model class.
    Signature: training loss nearly constant across seeds while rel-L2 varies 5x.

H1 and H3 are related but distinguishable: H1 predicts the fresh-point residual
exposes the error, H3 predicts even the fresh-point residual looks fine while the
solution is still wrong.

Everything is computed from saved checkpoints -- no retraining.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch

from .problem import MATERIALS
from .reference import fd_reference
from .operators import make_ansatz, wave_operator
from .models import REGISTRY
from .metrics import spacetime_rel_l2
from .losses import residual_scale
from .phase8 import build

TARGETS = np.linspace(0.05, 1.0, 20)


def fine_reference(M, nx=2048):
    x, t, u = fd_reference(M, nx=nx, T=1.0, sigma_g=0.1)
    out = []
    for tv in TARGETS:
        j = min(max(int(np.searchsorted(t, tv)), 1), len(t) - 1)
        w = (tv - t[j - 1]) / (t[j] - t[j - 1])
        out.append((1 - w) * u[j - 1] + w * u[j])
    return x, np.stack(out)


def residual_rms(u_fn, x, t, M, chunk=20000):
    """RMS of the PDE residual, computed in chunks.

    The wave operator needs second derivatives, so the autograd graph for 200k
    points through a grid=50 spline network does not fit alongside the running
    power studies.  Chunking is exact -- the mean of squares is accumulated, not
    approximated."""
    tot, n = 0.0, 0
    for i in range(0, x.shape[0], chunk):
        xa = x[i:i + chunk].detach().clone().requires_grad_(True)
        ta = t[i:i + chunk].detach().clone().requires_grad_(True)
        r = wave_operator(u_fn, xa, ta, M).detach()
        tot += float(r.pow(2).sum()); n += r.numel()
        del r, xa, ta
        torch.cuda.empty_cache()
    return float(np.sqrt(tot / n))


def train_collocation(seed, M, device, n_col=10000, t_max=1.0):
    """Reproduce EXACTLY the Sobol set train_lbfgs used for this seed."""
    eng = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=seed)
    pts = eng.draw(n_col)
    xc = (pts[:, 0:1] * (M.x_max - M.x_min) + M.x_min).to(device)
    tc = (pts[:, 1:2] * t_max).to(device)
    return xc.requires_grad_(True), tc.requires_grad_(True)


def fresh_points(seed, M, device, n=200000, t_max=1.0):
    """A dense INDEPENDENT sample the model never saw."""
    g = torch.Generator(device=device).manual_seed(9000 + seed)
    x = (torch.rand(n, 1, generator=g, device=device) * (M.x_max - M.x_min) + M.x_min)
    t = torch.rand(n, 1, generator=g, device=device) * t_max
    return x.requires_grad_(True), t.requires_grad_(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", default="homogeneous")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--pinn", default="fourier")
    ap.add_argument("--rung", default="charcoords50")
    ap.add_argument("--pinn-ckpt", nargs="+", required=True)
    ap.add_argument("--kan-ckpt", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    A = ap.parse_args()
    dev = torch.device(f"cuda:{A.gpu}")
    M = MATERIALS[A.material]()
    xr, ur = fine_reference(M)
    ans = make_ansatz("legacy", sigma_g=0.1)
    seeds = [int(s) for s in A.seeds.split(",")]

    print(f"{A.material}  --  residual on the TRAINING collocation set vs on 200k FRESH points")
    print(f"(H1: a LOCAL basis can satisfy the residual where it is sampled and drift between)\n")
    print(f"{'model':<14}{'seed':>5}{'rel-L2':>9}{'R_train':>11}{'R_fresh':>11}"
          f"{'R_fresh/R_train':>17}")
    out = []
    for tag, kind in [(A.pinn, "pinn"), (A.rung, "kan")]:
        for sd in seeds:
            if kind == "pinn":
                ck = None
                for d in getattr(A,"pinn_ckpt"):
                    p = os.path.join(d, f"{A.material}_s{sd}_{tag}_plain.pt")
                    if os.path.exists(p):
                        ck = p
                if ck is None:
                    continue
                torch.manual_seed(0)
                m = REGISTRY[tag]().to(dev)
                m.load_state_dict(torch.load(ck, map_location=dev,
                                             weights_only=False)["state_dict"])
            else:
                cp = os.path.join(A.kan_ckpt, f"{A.material}_s{sd}_{tag}.pt")
                if not os.path.exists(cp):
                    continue
                m = build(tag, M, seed=sd + 1).to(dev)
                m.set_save_act(False)
                m.load_state_dict(torch.load(cp, map_location=dev,
                                             weights_only=False)["state_dict"])
            m.eval()
            u_fn = lambda a, b: ans(m(a, b), a, b)

            xc, tc = train_collocation(sd, M, dev)
            xf, tf = fresh_points(sd, M, dev)
            rt = residual_rms(u_fn, xc, tc, M)
            rf = residual_rms(u_fn, xf, tf, M)

            X = torch.tensor(xr, dtype=torch.float32, device=dev).reshape(-1, 1)
            tt = torch.tensor(TARGETS, dtype=torch.float32, device=dev)
            xg = X.repeat(len(TARGETS), 1)
            tg = tt.repeat_interleave(len(xr)).reshape(-1, 1)
            with torch.no_grad():
                pred = ans(m(xg, tg), xg, tg).reshape(len(TARGETS), -1).cpu().numpy()
            e = spacetime_rel_l2(pred, ur)
            print(f"{tag:<14}{sd:>5}{e:>8.4f}%{rt:>11.4e}{rf:>11.4e}{rf/rt:>17.3f}")
            out.append((tag, sd, e, rt, rf))
            del m; torch.cuda.empty_cache()

    print()
    for tag in (A.pinn, A.rung):
        r = [(e, rf / rt) for (t_, s, e, rt, rf) in out if t_ == tag]
        if len(r) > 1:
            es = np.array([a for a, _ in r]); gs = np.array([b for _, b in r])
            print(f"  {tag:<14} mean R_fresh/R_train = {gs.mean():.3f}   "
                  f"corr(gap, rel-L2) = {np.corrcoef(gs, es)[0,1]:+.3f}")


if __name__ == "__main__":
    main()
