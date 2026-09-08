"""Does R3 help PINNs during ADAM -- the regime Daw et al. actually claim?

Phase 10 tested "R3 through Adam, then L-BFGS on the frozen pool" and scored the
FINAL model.  That is not the paper's claim.  Daw et al. (ICML 2023) train with
Adam alone and report that R3 improves the result.  So this module removes the
two things that made Phase 10 an unfair test of their claim:

  1. NO L-BFGS.  Adam only, exactly as in the official repo (which contains no
     L-BFGS anywhere).  Error is tracked THROUGH training, not just at the end,
     so if R3 helps at any point it will show.

  2. SOFT CONSTRAINTS as an option (`--soft`).  The paper enforces the initial
     condition with a loss term.  Our hard ansatz enforces it exactly, which may
     delete the concentrated early-time residual R3 is designed to chase --
     retention never dropped below 50 % in Phase 10, meaning the residual had no
     hot spot at all.  Running BOTH separates "R3 does not work here" from "our
     formulation removed what R3 works on".

Weighting: with soft constraints the PDE and (IC+BC) terms are balanced by the
repo's existing grad-norm scheme, refreshed every 500 steps -- not a hand-tuned
constant, so the comparison cannot be blamed on a weight choice.
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import torch
import torch.optim as optim

from .problem import MATERIALS
from .reference import fd_reference
from .operators import make_ansatz
from .losses import pde_loss, bc_loss, ic_loss, grad_norm_weights
from .models import REGISTRY, n_params
from .metrics import spacetime_rel_l2
from .train import sample
from .hybrid import R3Pool, sobol_pool
from .rescore import fine_reference, TARGETS


def score(u_fn, x_ref, u_ref, device):
    X = torch.tensor(x_ref, dtype=torch.float32, device=device).reshape(-1, 1)
    tt = torch.tensor(TARGETS, dtype=torch.float32, device=device)
    xg = X.repeat(len(TARGETS), 1)
    tg = tt.repeat_interleave(len(x_ref)).reshape(-1, 1)
    with torch.no_grad():
        pred = u_fn(xg, tg).reshape(len(TARGETS), -1).cpu().numpy()
    return float(spacetime_rel_l2(pred, u_ref))


def run(arch, material, *, steps, seed, lr, soft, use_r3, device,
        x_ref, u_ref, n_col=10000, n_bc=512, n_ic=512, t_max=1.0, sigma_g=0.1,
        eval_every=1000, log=None):
    torch.manual_seed(seed); np.random.seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    model = REGISTRY[arch]().to(device)
    ansatz = make_ansatz("none" if soft else "legacy", sigma_g=sigma_g)
    u_fn = lambda x, t: ansatz(model(x, t), x, t)

    live = [q for q in model.parameters() if q.requires_grad]
    _, _, tb = sample(1, n_bc, material, t_max, device, gen)
    tb = tb.detach()
    xi = (torch.rand(n_ic, 1, generator=gen, device=device)
          * (material.x_max - material.x_min) + material.x_min).detach()

    pool = R3Pool(n_col, material, t_max, device, gen, seed) if use_r3 else None
    xc, tc = (pool.x, pool.t) if pool else sobol_pool(n_col, material, t_max, device, seed)

    opt = optim.Adam(live, lr=lr)
    warm = max(10, steps // 10)
    lr_sched = optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.9 ** (s / warm)))
    w_pde = w_aux = 1.0
    curve, t0 = [], time.time()

    def terms(x, t):
        lp, _, _ = pde_loss(u_fn, x.clone(), t.clone(), material, 1.0, eps=None)
        aux = bc_loss(u_fn, tb.clone(), material, 1.0)
        if soft:
            aux = aux + ic_loss(u_fn, xi.clone(), sigma_g)
        return lp, aux

    for step in range(steps):
        if pool is not None:
            pool.update(u_fn)
            xc, tc = pool.x, pool.t
        opt.zero_grad(set_to_none=True)
        lp, aux = terms(xc, tc)
        loss = w_pde * lp + w_aux * aux
        if not torch.isfinite(loss):
            return dict(diverged=True, curve=curve, rel_l2=float("nan"))
        loss.backward(); opt.step(); lr_sched.step()
        # grad_norm_weights needs a LIVE graph; loss.backward() above has freed
        # the one `lp`/`aux` were built on, so recompute rather than reuse them.
        # Rebalance every 100 steps: at weight 1 the u==0 basin costs only 0.12,
        # which a collapsing model simply pays (measured -- see hybrid.py).
        if soft and (step + 1) % 100 == 0:
            lp2, aux2 = terms(xc, tc)
            w_pde, w_aux = grad_norm_weights(model, lp2, aux2)
        if (step + 1) % eval_every == 0:
            e = score(u_fn, x_ref, u_ref, device)
            curve.append((step + 1, e, float(loss.detach())))
            if log:
                log(f"      {step+1:>6}  rel-L2 {e:8.4f}%  loss {float(loss.detach()):.3e}"
                    + (f"  retained {pool.retained[-1]}" if pool else ""))
    return dict(diverged=False, curve=curve, rel_l2=score(u_fn, x_ref, u_ref, device),
                wall_s=time.time() - t0, params=n_params(model),
                r3_retained_mean=(float(np.mean(pool.retained)) if pool else None),
                r3_retained_first=(pool.retained[0] if pool else None),
                r3_retained_last=(pool.retained[-1] if pool else None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--materials", default="homogeneous")
    ap.add_argument("--models", default="fourier,mlp")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    A = ap.parse_args()
    dev = torch.device(f"cuda:{A.gpu}")
    rows = json.load(open(A.out)) if os.path.exists(A.out) else []
    done = {(r["material"], r["model"], r["seed"], r["soft"], r["r3"]) for r in rows}

    print(f"Adam-only ({A.steps} steps), the regime Daw et al. train in.  nx=2048\n")
    print(f"{'material':<13}{'model':<10}{'IC':<7}{'R3':<5}{'seed':>5}{'rel-L2':>10}"
          f"{'retain':>9}{'s':>7}", flush=True)
    for mk in A.materials.split(","):
        M = MATERIALS[mk]()
        xr, ur = fine_reference(M, 2048)
        for soft in (True, False):
            for arch in A.models.split(","):
                for use_r3 in (False, True):
                    for sd in [int(s) for s in A.seeds.split(",")]:
                        key = (mk, arch, sd, soft, use_r3)
                        if key in done:
                            continue
                        r = run(arch, M, steps=A.steps, seed=sd, lr=A.lr, soft=soft,
                                use_r3=use_r3, device=dev, x_ref=xr, u_ref=ur)
                        r.update(material=mk, model=arch, seed=sd, soft=soft, r3=use_r3)
                        rows.append(r)
                        ret = r.get("r3_retained_mean")
                        print(f"{mk:<13}{arch:<10}{'soft' if soft else 'hard':<7}"
                              f"{'R3' if use_r3 else '--':<5}{sd:>5}{r['rel_l2']:>9.4f}%"
                              f"{(f'{ret:.0f}' if ret else '-'):>9}"
                              f"{r.get('wall_s',0):>7.0f}", flush=True)
                        json.dump(rows, open(A.out, "w"), indent=1)
    print("\ndone", flush=True)


if __name__ == "__main__":
    main()
