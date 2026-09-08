"""Per-model error anatomy, measured rather than inferred.

Everything so far has compared SCALARS (a rel-L2 per model) and then told a story
about why they differ.  Two of those stories have now failed a check:

  * the separability argument explains homogeneous (d'Alembert residual 0.0068 %)
    but NOT multilayer (39.04 %), which is the material where the tie is solid;
  * the 233x basis-approximation advantage does not appear in the trained models,
    which differ by 1.42x -- and that 1.42x is not significant at n=3 (p = 0.237).

So this module stops comparing scalars and decomposes the error FIELD.

For each trained model, with u* the nx=2048 reference:

    e(x,t) = u_theta(x,t) - u*(x,t)

is projected onto an interpretable basis built from the reference itself:

    e  =  alpha * u*          amplitude error   (the pulse is too big/small)
       +  beta  * du*/dx      position error    (the pulse is shifted -- for a
                                                 travelling wave this is phase /
                                                 arrival-time error)
       +  r                   shape error       (everything else: the model has
                                                 the wrong waveform)

alpha and beta come from a least-squares fit per snapshot, so the split is exact
and orthogonalised.  This distinguishes three physically different failures that
a single rel-L2 number cannot:

    a model whose pulse arrives slightly early  (beta large, r small)
    a model whose pulse is the right shape but attenuated (alpha large)
    a model that cannot represent the waveform at all (r large)

Also reported per model:

  * where the error lives in x (inside the pulse vs in the far field)
  * where it lives in k (which wavenumbers are wrong, against the solution's own
    spectrum)
  * the linear probe on the final hidden layer at nx=2048 -- separating
    "the features cannot express the solution" from "the readout is suboptimal"
  * pairwise distance between seeds of the same model, which asks whether high
    seed variance means the seeds find DIFFERENT functions or the same function
    with different residuals
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
from .phase8 import build

TARGETS = np.linspace(0.05, 1.0, 20)


def fine_reference(material, nx=2048, sigma_g=0.1, T=1.0):
    x, t, u = fd_reference(material, nx=nx, T=T, sigma_g=sigma_g)
    out = []
    for tv in TARGETS:
        j = min(max(int(np.searchsorted(t, tv)), 1), len(t) - 1)
        w = (tv - t[j - 1]) / (t[j] - t[j - 1])
        out.append((1 - w) * u[j - 1] + w * u[j])
    return x, np.stack(out)


def predict(model, x_ref, device, sigma_g=0.1):
    X = torch.tensor(x_ref, dtype=torch.float32, device=device).reshape(-1, 1)
    tt = torch.tensor(TARGETS, dtype=torch.float32, device=device)
    xg = X.repeat(len(TARGETS), 1)
    tg = tt.repeat_interleave(len(x_ref)).reshape(-1, 1)
    ans = make_ansatz("legacy", sigma_g=sigma_g)
    with torch.no_grad():
        p = ans(model(xg, tg), xg, tg).reshape(len(TARGETS), -1).cpu().numpy()
    return p


def decompose(pred, uref, x):
    """Split e = pred - uref into amplitude / position / shape, per snapshot.

    Basis per snapshot k:  [ uref[k] , d uref[k]/dx ].
    Least squares gives alpha (amplitude) and beta (position, in units of x).
    Returns energy fractions plus the mean absolute shift in x.
    """
    dx = x[1] - x[0]
    amp = pos = shape = tot = 0.0
    shifts = []
    for k in range(uref.shape[0]):
        u = uref[k]
        du = np.gradient(u, dx)
        e = pred[k] - u
        A = np.stack([u, du], 1)
        c, *_ = np.linalg.lstsq(A, e, rcond=None)
        fit = A @ c
        amp += np.sum((c[0] * u) ** 2)
        pos += np.sum((c[1] * du) ** 2)
        shape += np.sum((e - fit) ** 2)
        tot += np.sum(e ** 2)
        shifts.append(c[1])
    return (amp / tot, pos / tot, shape / tot, float(np.mean(np.abs(shifts))))


def where_in_x(pred, uref, x, thresh=0.05):
    """Fraction of squared error inside the pulse support vs outside."""
    mask = np.abs(uref) > thresh * np.abs(uref).max()
    e2 = (pred - uref) ** 2
    return float(e2[mask].sum() / e2.sum()), float(mask.mean())


def where_in_k(pred, uref, x, k_split=9.4):
    """Share of error energy below / above the solution's peak wavenumber."""
    dx = x[1] - x[0]
    E = np.fft.rfft(pred - uref, axis=1)
    U = np.fft.rfft(uref, axis=1)
    k = 2 * np.pi * np.fft.rfftfreq(uref.shape[1], d=dx)
    lo = np.abs(E[:, k <= k_split]) ** 2
    hi = np.abs(E[:, k > k_split]) ** 2
    ulo = np.abs(U[:, k <= k_split]) ** 2
    uhi = np.abs(U[:, k > k_split]) ** 2
    return (float(hi.sum() / (lo.sum() + hi.sum())),
            float(uhi.sum() / (ulo.sum() + uhi.sum())))


def final_features(model, x_ref, device, arch):
    """Last hidden representation, for the linear probe."""
    X = torch.tensor(x_ref, dtype=torch.float32, device=device).reshape(-1, 1)
    tt = torch.tensor(TARGETS, dtype=torch.float32, device=device)
    xg = X.repeat(len(TARGETS), 1)
    tg = tt.repeat_interleave(len(x_ref)).reshape(-1, 1)
    with torch.no_grad():
        if hasattr(model, "kan"):
            model.kan.save_act = True
            inp = model._inputs(xg, tg) if hasattr(model, "_inputs") else torch.cat([xg, tg], -1)
            model.kan(inp)
            F = model.kan.acts[-1]
            model.kan.save_act = False
        elif arch == "pirate":
            F = model._features(xg, tg)
        else:
            h = model.embed(xg, tg) if hasattr(model, "embed") else torch.cat([xg, tg], -1)
            for m in model.net[:-1]:
                h = m(h)
            F = h
    return F


def probe(F, uref, sigma_g=0.1, x_ref=None, device=None):
    """Best rel-L2 a LINEAR readout of these features could reach.

    The target is N = (u - g(x)decay(t)) / growth(t), NOT u.  The network never
    produces u: the trainer forms u = g*decay + growth*N, so the ansatz supplies
    the IC term independently.  Probing against u asks the features to reproduce
    something they are not responsible for and reports ~13% where the true figure
    is ~0.07%.
    """
    import torch as th
    from .operators import make_ansatz
    X_ = th.tensor(x_ref, dtype=th.float32, device=F.device).reshape(-1, 1)
    tt = th.tensor(TARGETS, dtype=th.float32, device=F.device)
    xg = X_.repeat(len(TARGETS), 1)
    tg = tt.repeat_interleave(len(x_ref)).reshape(-1, 1)
    ans = make_ansatz("legacy", sigma_g=sigma_g)
    with th.no_grad():
        ug = th.tensor(uref.reshape(-1, 1), dtype=th.float32, device=F.device)
        ic = ans(th.zeros_like(ug), xg, tg)
        gr = ans(th.ones_like(ug), xg, tg) - ic
    keep = gr.abs().squeeze(-1) > 1e-3
    y = th.zeros_like(ug)
    y[keep] = (ug[keep] - ic[keep]) / gr[keep]
    F = F[keep]
    y = y[keep].double()
    X = th.cat([F.double(), th.ones(F.shape[0], 1, dtype=th.float64, device=F.device)], 1)
    y = y.reshape(-1, 1)
    XtX = X.T @ X
    lam = 1e-12 * float(th.diagonal(XtX).mean().clamp_min(1e-300))
    w = th.linalg.solve(XtX + lam * th.eye(X.shape[1], dtype=th.float64, device=F.device),
                        X.T @ y)
    return float(100.0 * (X @ w - y).norm() / y.norm())


def get_model(mk, arch, sd, M, device, ckpt_dirs, epochs, ckpt_out):
    """Load a PINN checkpoint, or retrain+save the KAN rung (Phase 8 saved none)."""
    if arch in REGISTRY:
        for d in ckpt_dirs:
            p = os.path.join(d, f"{mk}_s{sd}_{arch}_plain.pt")
            if os.path.exists(p):
                torch.manual_seed(0)
                m = REGISTRY[arch]().to(device)
                m.load_state_dict(torch.load(p, map_location=device,
                                             weights_only=False)["state_dict"])
                return m.eval()
        return None
    os.makedirs(ckpt_out, exist_ok=True)
    cp = os.path.join(ckpt_out, f"{mk}_s{sd}_{arch}.pt")
    model = build(arch, M, seed=sd + 1).to(device)
    model.set_save_act(False)
    if os.path.exists(cp):
        model.load_state_dict(torch.load(cp, map_location=device,
                                         weights_only=False)["state_dict"])
        return model.eval()
    xc, tc, uc = fd_reference(M, nx=512, T=1.0, sigma_g=0.1)
    ev = [int(np.argmin(np.abs(tc - v))) for v in TARGETS]
    model, _ = train_lbfgs(lambda: model, M, epochs=epochs, seed=sd, device=device,
                           use_r3=False, x_ref=xc, t_ref=tc[ev], u_ref=uc[ev])
    torch.save({"state_dict": model.state_dict()}, cp)
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--materials", default="homogeneous,multilayer")
    ap.add_argument("--models", default="pirate,fourier,charcoords50")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--nx", type=int, default=2048)
    ap.add_argument("--ckpt-dirs", nargs="+", required=True)
    ap.add_argument("--kan-ckpt", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    A = ap.parse_args()
    dev = torch.device(f"cuda:{A.gpu}")
    seeds = [int(s) for s in A.seeds.split(",")]
    rows = json.load(open(A.out)) if os.path.exists(A.out) else []
    done = {(r["material"], r["model"], r["seed"]) for r in rows}

    for mk in A.materials.split(","):
        M = MATERIALS[mk]()
        xr, ur = fine_reference(M, A.nx)
        preds = {}
        print(f"\n{'='*100}\n  {mk}   (reference nx={A.nx})\n{'='*100}")
        print(f"{'model':<14}{'seed':>4}{'rel-L2':>9}{'amp%':>7}{'pos%':>7}{'shape%':>8}"
              f"{'|shift|':>10}{'in-pulse':>10}{'hi-k err':>9}{'probe':>9}")
        for md in A.models.split(","):
            for sd in seeds:
                m = get_model(mk, md, sd, M, dev, A.ckpt_dirs, A.epochs, A.kan_ckpt)
                if m is None:
                    print(f"{md:<14}{sd:>4}   no checkpoint"); continue
                p = predict(m, xr, dev)
                preds[(md, sd)] = p
                e = spacetime_rel_l2(p, ur)
                a, b, s, sh = decompose(p, ur, xr)
                inp, frac = where_in_x(p, ur, xr)
                hk, uhk = where_in_k(p, ur, xr)
                F = final_features(m, xr, dev, md)
                pr = probe(F, ur, x_ref=xr)
                rows.append(dict(material=mk, model=md, seed=sd, rel_l2=e,
                                 amp=a, pos=b, shape=s, shift=sh, in_pulse=inp,
                                 pulse_frac=frac, hi_k_err=hk, hi_k_sol=uhk,
                                 probe=pr, params=n_params(m)))
                print(f"{md:<14}{sd:>4}{e:>8.4f}%{100*a:>6.1f}%{100*b:>6.1f}%"
                      f"{100*s:>7.1f}%{sh:>10.2e}{100*inp:>9.1f}%{100*hk:>8.1f}%"
                      f"{pr:>8.4f}%", flush=True)
                json.dump(rows, open(A.out, "w"), indent=1)
                del m, F; torch.cuda.empty_cache()
        # do different seeds of the same model find the same function?
        print(f"\n  seed-to-seed disagreement (rel-L2 between two seeds of one model):")
        for md in A.models.split(","):
            ds = [spacetime_rel_l2(preds[(md, i)], preds[(md, j)])
                  for i in seeds for j in seeds if i < j and (md, i) in preds and (md, j) in preds]
            if ds:
                own = np.mean([r["rel_l2"] for r in rows
                               if r["material"] == mk and r["model"] == md])
                print(f"    {md:<14} mean pairwise {np.mean(ds):.4f}%   "
                      f"vs own error {own:.4f}%   ratio {np.mean(ds)/own:.2f}x")
    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
