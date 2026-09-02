"""Step-by-step trace of one input batch through the trained PINN and KAN.

forensics.py established WHERE the two models differ (effective rank 34 vs 4.7,
linear probe 0.068% vs 26.4% at the last hidden layer).  It did not establish
WHY.  This module follows the actual vectors and asks a sharper question of each
architecture:

    how much of the nonlinear machinery each model owns is actually in use?

For a KAN every edge carries a learned univariate spline.  If those splines are
close to straight lines over the range their input actually visits, the layer is
an affine map wearing a spline costume, and the network's expressive power is
whatever survives that collapse.  Measured per edge as

    nonlinearity = || phi - best_linear_fit(phi) || / || phi - mean(phi) ||

evaluated ON THE OBSERVED ACTIVATION RANGE, not on the nominal grid -- a spline
is only as useful as the part of it the data reaches.

For a tanh network the analogous question is whether pre-activations reach the
saturating region at all: |z| << 1 everywhere means tanh(z) ~ z and the layer is
again affine.  Reported as the fraction of pre-activations with |z| > 1.

Both numbers answer the same question in the same units, so the two families can
be compared directly.
"""
from __future__ import annotations
import argparse, os
import numpy as np
import torch

from .problem import MATERIALS
from .reference import fd_reference
from .operators import make_ansatz
from .models import REGISTRY


def _target(material, device, sigma_g=0.1, T=1.0, n_snap=20, nx=512):
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
    return xg, tg, ug, N, keep, x_np, t_np[ev], u_np[ev]


def svals(F):
    X = (F - F.mean(0, keepdim=True)).double()
    return torch.linalg.svdvals(X)


def spline_nonlinearity(layer, acts_in):
    """Per-edge nonlinearity of a pykan KANLayer, over the OBSERVED input range.

    Returns (mean, median, frac_below_5pct) across all in*out edges.
    """
    from kan.spline import coef2curve
    gridpts = 200
    lo = acts_in.min(dim=0).values
    hi = acts_in.max(dim=0).values
    in_dim = acts_in.shape[1]
    out = []
    for i in range(in_dim):
        xs = torch.linspace(float(lo[i]), float(hi[i]), gridpts,
                            device=acts_in.device, dtype=acts_in.dtype).reshape(-1, 1)
        xs_full = xs.repeat(1, in_dim)                      # coef2curve wants (b,in)
        y = coef2curve(xs_full, layer.grid, layer.coef, layer.k)   # (b,in,out)
        phi = y[:, i, :]                                    # (b,out) edges from input i
        base = layer.base_fun(xs).reshape(-1, 1)
        phi = (layer.scale_base[i].reshape(1, -1) * base
               + layer.scale_sp[i].reshape(1, -1) * phi)
        s = xs.reshape(-1, 1).double()
        A = torch.cat([s, torch.ones_like(s)], dim=1)
        P = phi.double()
        w = torch.linalg.lstsq(A, P).solution
        resid = P - A @ w
        den = (P - P.mean(0, keepdim=True)).norm(dim=0).clamp_min(1e-300)
        out.append((resid.norm(dim=0) / den))
    v = torch.cat(out)
    return float(v.mean()), float(v.median()), float((v < 0.05).float().mean())


def run(arch, ckpt, material, device):
    torch.manual_seed(0)
    model = REGISTRY[arch]().to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(sd["state_dict"]); model.eval()
    xg, tg, ug, N, keep, x_np, t_ev, u_ev = _target(material, device)
    print(f"\n{'='*80}\n  {arch}   trained rel_L2 = {sd['metrics']['rel_l2']:.3f}%\n{'='*80}")

    with torch.no_grad():
        if arch == "pykan_wide":
            model.kan.save_act = True
            inp = torch.cat([xg, tg], -1)
            y = model.kan(inp)
            acts = list(model.kan.acts)
            print(f"\n  INPUT  x in [{float(xg.min()):+.2f},{float(xg.max()):+.2f}]  "
                  f"t in [{float(tg.min()):+.2f},{float(tg.max()):+.2f}]  "
                  f"-> KAN sees a {inp.shape[1]}-vector")
            print(f"\n  {'layer':<8}{'shape':>10}{'obs range':>22}{'top-5 singular values':>44}")
            for i, a in enumerate(acts):
                s = svals(a[keep])[:5]
                ss = " ".join(f"{float(v):8.2f}" for v in s)
                print(f"  {'acts'+str(i):<8}{str(tuple(a.shape[1:])):>10}"
                      f"{'[%+.2f, %+.2f]' % (float(a.min()), float(a.max())):>22}{ss:>44}")
            print(f"\n  SPLINE NONLINEARITY per edge, over the range each edge actually sees")
            print(f"  {'layer':<8}{'edges':>8}{'mean':>9}{'median':>9}{'~linear (<5%)':>16}")
            for i, l in enumerate(model.kan.act_fun):
                m, md, fr = spline_nonlinearity(l, acts[i])
                print(f"  {'layer'+str(i):<8}{l.in_dim*l.out_dim:>8}{m:>9.3f}{md:>9.3f}"
                      f"{100*fr:>15.1f}%")
        else:
            if arch == "fourier":
                h = model.embed(xg, tg)
                stages = [("embed", h)]
            else:
                h = torch.cat([xg, tg], -1)
                stages = [("input", h)]
            print(f"\n  INPUT  x in [{float(xg.min()):+.2f},{float(xg.max()):+.2f}]  "
                  f"t in [{float(tg.min()):+.2f},{float(tg.max()):+.2f}]  "
                  f"-> net sees a {stages[0][1].shape[1]}-vector")
            pre = []
            k = 0
            for m in model.net:
                hp = h
                h = m(h)
                if isinstance(m, torch.nn.Tanh):
                    k += 1
                    pre.append((f"tanh{k}", hp))
                    stages.append((f"tanh{k}", h))
            print(f"\n  {'layer':<8}{'shape':>10}{'obs range':>22}{'top-5 singular values':>44}")
            for nm, a in stages:
                s = svals(a[keep])[:5]
                ss = " ".join(f"{float(v):8.2f}" for v in s)
                print(f"  {nm:<8}{str(tuple(a.shape[1:])):>10}"
                      f"{'[%+.2f, %+.2f]' % (float(a.min()), float(a.max())):>22}{ss:>44}")
            print(f"\n  TANH SATURATION: fraction of pre-activations with |z|>1 "
                  f"(|z|<<1 => the layer is effectively affine)")
            print(f"  {'layer':<8}{'mean |z|':>10}{'frac |z|>1':>13}{'frac |z|>2':>13}")
            for nm, z in pre:
                za = z.abs()
                print(f"  {nm:<8}{float(za.mean()):>10.3f}"
                      f"{100*float((za>1).float().mean()):>12.1f}%"
                      f"{100*float((za>2).float().mean()):>12.1f}%")
    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", default="homogeneous")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ckpt-dirs", nargs="+", required=True)
    ap.add_argument("--archs", default="fourier,mlp,pykan_wide")
    A = ap.parse_args()
    dev = torch.device(f"cuda:{A.gpu}")
    M = MATERIALS[A.material]()
    for arch in A.archs.split(","):
        ck = None
        for d in A.ckpt_dirs:
            p = os.path.join(d, f"{A.material}_s0_{arch}_plain.pt")
            if os.path.exists(p):
                ck = p
        if ck is None:
            print(f"!! no checkpoint for {arch}"); continue
        run(arch, ck, M, dev)


if __name__ == "__main__":
    main()
