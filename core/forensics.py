"""Layer-by-layer forensics: WHERE does the B-spline KAN fall behind the PINN?

The Phase 6/7 tables say the Fourier PINN reaches 0.112% on homogeneous and the
B-spline KAN 0.200%.  They do not say why.  This module opens both TRAINED
networks, pushes the identical evaluation points through them, and measures each
hidden layer on four axes.  Everything is computed in float64 from saved
checkpoints -- no retraining, no randomness.

The four measurements, and what each one can prove
--------------------------------------------------
1. ACTIVATION RANGE vs the layer's own domain.
   For a KAN the splines have FIXED support, so activations that contract into a
   sliver of the grid (or overflow past its last knot) waste the layer.  An MLP
   has no analogous failure -- tanh has no finite support to fall off.

2. EFFECTIVE RANK (participation ratio of the singular values,
   (sum s_i)^2 / sum s_i^2).  How many independent directions the layer's
   representation actually occupies.  A layer with 20 units but effective rank 3
   is a rank-3 bottleneck regardless of its parameter count.

3. LINEAR PROBE -- the decisive one.
   For each layer, take its features Phi and solve min_w ||Phi w - N_target||
   exactly, by least squares.  This asks: is the information needed to express
   the solution PRESENT in this layer's representation, recoverable by a linear
   map?  It localises the deficiency to a depth.  If the PINN's features span the
   target at layer 2 while the KAN's never do, the KAN is not building the right
   features, and that is a representation statement, not an optimisation one.

4. SPECTRAL CONTENT of the features along x.
   Our solution is a travelling pulse with measured k_peak ~ 9.4.  Fourier
   features encode translation as a phase shift and carry that band natively;
   fixed-grid splines must synthesise it.  This measures the fraction of each
   layer's feature energy above k = 5, i.e. whether the layer can represent the
   frequencies the solution actually contains.

`N_target` is what the NETWORK must produce, not u itself: the trainer applies
u = g(x)decay(t) + growth(t) N(x,t), so N_target = (u_ref - g decay)/growth.
Comparing raw u would credit both models for the ansatz's hard-coded IC term.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch

from .problem import MATERIALS, gaussian_ic
from .reference import fd_reference
from .operators import make_ansatz


# ─────────────────────────── measurements ───────────────────────────
def effective_rank(F: torch.Tensor) -> float:
    """Participation ratio of the singular spectrum of the centred features."""
    X = (F - F.mean(0, keepdim=True)).double()
    s = torch.linalg.svdvals(X)
    s = s[s > 0]
    if s.numel() == 0:
        return 0.0
    return float(s.sum() ** 2 / (s ** 2).sum())


def linear_probe(F: torch.Tensor, y: torch.Tensor) -> float:
    """Best rel-L2 achievable by a LINEAR map of these features onto y.

    Ridge-stabilised normal equations rather than lstsq: on CUDA torch's only
    driver is `gels`, which assumes full rank and returns NaN silently on a
    rank-deficient design -- and these feature matrices are exactly the
    rank-deficient case (see effective_rank above).
    """
    X = torch.cat([F.double(), torch.ones(F.shape[0], 1, dtype=torch.float64,
                                          device=F.device)], dim=1)
    yy = y.double().reshape(-1, 1)
    XtX = X.T @ X
    lam = 1e-12 * float(torch.diagonal(XtX).mean().clamp_min(1e-300))
    w = torch.linalg.solve(XtX + lam * torch.eye(X.shape[1], dtype=torch.float64,
                                                 device=F.device), X.T @ yy)
    r = X @ w - yy
    return float(100.0 * r.norm() / yy.norm())


def hi_freq_fraction(F: torch.Tensor, n_x: int, n_t: int, k_cut: int = 5) -> float:
    """Fraction of feature energy above wavenumber k_cut, along x.

    F is (n_t*n_x, d) laid out with x fastest.  Returns the mean over features
    and snapshots of the high-k energy share.
    """
    d = F.shape[1]
    G = F.double().reshape(n_t, n_x, d)
    G = G - G.mean(dim=1, keepdim=True)
    S = torch.fft.rfft(G, dim=1).abs() ** 2               # (n_t, n_x//2+1, d)
    tot = S.sum(dim=1)
    hi = S[:, k_cut:, :].sum(dim=1)
    ok = tot > 0
    return float((hi[ok] / tot[ok]).mean()) if ok.any() else 0.0


# ─────────────────────────── feature extraction ───────────────────────────
def fourier_layers(model, x, t):
    """(name, features) for each stage of FourierMLP: embedding then each tanh."""
    out = [("embed", model.embed(x, t))]
    h = out[0][1]
    idx = 0
    for m in model.net:
        h = m(h)
        if isinstance(m, torch.nn.Tanh):
            idx += 1
            out.append((f"tanh{idx}", h))
    out.append(("output", h))
    return out


def mlp_layers(model, x, t):
    out = []
    h = torch.cat([x, t], dim=-1)
    idx = 0
    for m in model.net:
        h = m(h)
        if isinstance(m, torch.nn.Tanh):
            idx += 1
            out.append((f"tanh{idx}", h))
    out.append(("output", h))
    return out


def pykan_layers(model, x, t):
    """pykan caches per-layer activations in .acts when save_act is on."""
    prev = model.kan.save_act
    model.kan.save_act = True
    inp = model._inputs(x, t) if hasattr(model, "_inputs") else torch.cat([x, t], -1)
    y = model.kan(inp)
    acts = list(model.kan.acts)
    model.kan.save_act = prev
    out = [(f"acts{i}", a) for i, a in enumerate(acts)]
    out.append(("output", y))
    return out


EXTRACT = {"fourier": fourier_layers, "mlp": mlp_layers,
           "pykan_wide": pykan_layers, "pirate": None}


# ─────────────────────────── driver ───────────────────────────
def analyse(arch, ckpt, material, *, device, sigma_g=0.1, T=1.0, n_snap=20, nx=512):
    from .models import REGISTRY
    torch.manual_seed(0)
    model = REGISTRY[arch]().to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(sd["state_dict"])
    model.eval()

    x_np, t_np, u_np = fd_reference(material, nx=nx, T=T, sigma_g=sigma_g)
    ev = [int(np.argmin(np.abs(t_np - v))) for v in np.linspace(0.05, T, n_snap)]
    X = torch.tensor(x_np, dtype=torch.float32, device=device).reshape(-1, 1)
    tt = torch.tensor(t_np[ev], dtype=torch.float32, device=device)
    xg = X.repeat(len(ev), 1)
    tg = tt.repeat_interleave(len(x_np)).reshape(-1, 1)
    ug = torch.tensor(u_np[ev], dtype=torch.float32, device=device).reshape(-1, 1)

    # what the NETWORK must produce, with the ansatz's contribution removed
    ans = make_ansatz("legacy", sigma_g=sigma_g)
    with torch.no_grad():
        zero = torch.zeros_like(ug)
        ic_part = ans(zero, xg, tg)                       # g(x)*decay(t)
        one = torch.ones_like(ug)
        growth = ans(one, xg, tg) - ic_part               # growth(t)
    keep = growth.abs().squeeze(-1) > 1e-3
    N_target = torch.zeros_like(ug)
    N_target[keep] = (ug[keep] - ic_part[keep]) / growth[keep]

    with torch.no_grad():
        layers = EXTRACT[arch](model, xg, tg)
        rows = []
        for name, F in layers:
            F = F.detach()
            rows.append(dict(
                layer=name, width=int(F.shape[1]),
                lo=float(F.min()), hi=float(F.max()), std=float(F.std()),
                eff_rank=effective_rank(F[keep]),
                probe=linear_probe(F[keep], N_target[keep]),
                hi_freq=hi_freq_fraction(F, nx, len(ev)),
            ))
    return rows, dict(rel_l2=sd["metrics"]["rel_l2"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--material", default="homogeneous")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ckpt-dirs", nargs="+", required=True)
    ap.add_argument("--archs", default="fourier,mlp,pykan_wide")
    ap.add_argument("--out", default=None)
    A = ap.parse_args()
    dev = torch.device(f"cuda:{A.gpu}")
    M = MATERIALS[A.material]()
    allrows = {}
    for arch in A.archs.split(","):
        ck = None
        for d in A.ckpt_dirs:
            p = os.path.join(d, f"{A.material}_s0_{arch}_plain.pt")
            if os.path.exists(p):
                ck = p
        if ck is None:
            print(f"!! no checkpoint for {arch}"); continue
        rows, meta = analyse(arch, ck, M, device=dev)
        allrows[arch] = rows
        print(f"\n=== {arch}   (trained rel_L2 = {meta['rel_l2']:.3f}%)   [{os.path.basename(ck)}]")
        print(f"{'layer':<9}{'width':>7}{'range':>22}{'std':>9}{'eff.rank':>10}"
              f"{'linear probe':>14}{'hi-k share':>12}")
        for r in rows:
            rng = "[%+.2f, %+.2f]" % (r["lo"], r["hi"])
            print(f"{r['layer']:<9}{r['width']:>7}{rng:>22}{r['std']:>9.3f}"
                  f"{r['eff_rank']:>10.2f}{r['probe']:>13.3f}%{r['hi_freq']:>12.3f}")
    if A.out:
        json.dump(allrows, open(A.out, "w"), indent=1)
        print(f"\nwritten to {A.out}")


if __name__ == "__main__":
    main()
