"""Is the collapse the optimiser's mistake, or the loss function's?

Under our objective u == 0 has PDE residual exactly zero, while any APPROXIMATION
of the true solution has residual eps > 0.  So the trivial solution is a strictly
better minimum for every architecture.  Most models nonetheless converge to the
right answer, which means the question is not "why does the Fourier-embedded KAN
collapse" but "why does anything else avoid it".

Two candidate explanations, with opposite consequences:

  LOSS MISSPECIFIED for this architecture.  The training loss is genuinely lower
      at the collapsed solution than at a well-fitted one.  The optimiser is
      then behaving correctly and no amount of optimiser tuning will help --
      the objective has to change.
  BASIN / PATH problem.  The training loss is HIGHER at the collapsed solution,
      so the optimiser is settling for something it could beat.  Then it is an
      optimisation problem and a better initialisation or schedule can fix it.

The test: build a well-fitted model by supervised regression, evaluate the ACTUAL
training objective there, and compare against the collapsed checkpoint.  Then
restart physics training FROM the good solution and watch whether it stays.

Note on the ansatz: u == 0 is not exactly representable.  With
u = g*decay + growth*N, driving u to zero needs N = -g*decay/growth, which
diverges as t -> 0 because growth = tanh^2(25t) -> 0.  So "collapse" means u ~= 0
for t > ~0.1 with a residual that is small but not zero -- which is why the
comparison below has to be measured rather than assumed.
"""
from __future__ import annotations
import argparse, os
import numpy as np
import torch
import torch.optim as optim

from .problem import MATERIALS
from .models import REGISTRY
from .phase8 import build as build_rung
from .operators import make_ansatz
from .losses import pde_loss, bc_loss
from .hybrid import sobol_pool
from .train import sample
from .rescore import fine_reference, TARGETS
from .metrics import spacetime_rel_l2


def build(arch, M, seed=0):
    if arch.startswith("rung:"):
        m = build_rung(arch.split(":", 1)[1], M, seed=seed + 1)
        m.set_save_act(False)
        return m
    return REGISTRY[arch]()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="rung:fourier16,rung:charcoords50")
    ap.add_argument("--material", default="homogeneous")
    ap.add_argument("--sup-adam", type=int, default=6000)
    ap.add_argument("--sup-lbfgs", type=int, default=3000)
    ap.add_argument("--phys-epochs", type=int, default=400)
    ap.add_argument("--gpu", type=int, default=0)
    A = ap.parse_args()
    dev = torch.device(f"cuda:{A.gpu}")
    M = MATERIALS[A.material]()
    xr, ur = fine_reference(M, 2048)
    ans = make_ansatz("legacy", sigma_g=0.1)

    X = torch.tensor(xr, dtype=torch.float32, device=dev).reshape(-1, 1)
    tt = torch.tensor(TARGETS, dtype=torch.float32, device=dev)
    xf = X.repeat(len(TARGETS), 1); tf = tt.repeat_interleave(len(xr)).reshape(-1, 1)
    uf = torch.tensor(ur.reshape(-1, 1), dtype=torch.float32, device=dev)

    xc, tc = sobol_pool(10000, M, 1.0, dev, 0)
    g = torch.Generator(device=dev).manual_seed(0)
    _, _, tb = sample(1, 512, M, 1.0, dev, g); tb = tb.detach()

    def train_obj(u_fn):
        """The EXACT objective train_hybrid minimises."""
        l, _, _ = pde_loss(u_fn, xc.clone(), tc.clone(), M, 1.0, eps=None)
        return float((l + bc_loss(u_fn, tb.clone(), M, 1.0)).detach())

    def rel(u_fn):
        with torch.no_grad():
            p = u_fn(xf, tf).reshape(len(TARGETS), -1).cpu().numpy()
        return float(spacetime_rel_l2(p, ur))

    for arch in A.models.split(","):
        print(f"\n{'='*76}\n{arch}\n{'='*76}", flush=True)
        torch.manual_seed(0)
        m = build(arch, M).to(dev)
        u_fn = lambda x, t: ans(m(x, t), x, t)
        live = [q for q in m.parameters() if q.requires_grad]

        # ---- A: supervised regression to the reference -------------------
        opt = optim.Adam(live, lr=1e-3)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=A.sup_adam, eta_min=1e-6)
        for _ in range(A.sup_adam):
            opt.zero_grad(set_to_none=True)
            l = ((u_fn(xf, tf) - uf) ** 2).mean()
            if not torch.isfinite(l): break
            l.backward(); opt.step(); sch.step()
        opt = optim.LBFGS(live, lr=1.0, max_iter=20, history_size=100,
                          line_search_fn="strong_wolfe")
        best, wait = float("inf"), 0
        for _ in range(A.sup_lbfgs):
            def cl():
                opt.zero_grad(set_to_none=True)
                l = ((u_fn(xf, tf) - uf) ** 2).mean()
                if torch.isfinite(l): l.backward()
                return l
            v = float(opt.step(cl).detach())
            if not np.isfinite(v): break
            if v < best * (1 - 1e-9):
                best, wait = v, 0
            else:
                wait += 1
                if wait >= 200: break
        good = {k: q.detach().clone() for k, q in m.state_dict().items()}
        e_good, L_good = rel(u_fn), train_obj(u_fn)
        print(f"  A. supervised fit      rel-L2 {e_good:8.4f}%   MSE {best:.3e}")
        print(f"     TRAINING OBJECTIVE at this well-fitted solution: {L_good:.6e}")

        # ---- B: the collapsed / physics-trained solution ------------------
        S = "/tmp/claude-1010/-home-prjgnn-PINN-FNO/a765de5d-6b07-41b3-9ed9-3624831a9432/scratchpad"
        safe = arch.replace(":", "-")
        ck = os.path.join(S, "ckpt10", f"{A.material}_s0_{safe}_plain_hybrid.pt")
        if os.path.exists(ck):
            m.load_state_dict(torch.load(ck, map_location=dev, weights_only=False)["state_dict"])
            e_bad, L_bad = rel(u_fn), train_obj(u_fn)
            print(f"  B. physics-trained     rel-L2 {e_bad:8.4f}%")
            print(f"     TRAINING OBJECTIVE at the physics-trained solution: {L_bad:.6e}")
            verdict = ("LOSS PREFERS THE WRONG ANSWER -- objective is misspecified"
                       if L_bad < L_good else
                       "loss prefers the GOOD answer -- optimiser failed to find it")
            print(f"\n     ratio  L(physics)/L(good) = {L_bad/L_good:.4g}   -> {verdict}")
        else:
            print(f"  B. no checkpoint at {ck}")
            continue

        # ---- C: restart physics training FROM the good solution -----------
        m.load_state_dict(good)
        opt = optim.LBFGS(live, lr=1.0, max_iter=20, history_size=100,
                          line_search_fn="strong_wolfe")
        print(f"\n  C. physics training restarted FROM the well-fitted solution:")
        print(f"     {'epoch':>7}{'rel-L2':>11}{'objective':>14}{'||u||/||u_ref||':>18}")
        for ep in range(A.phys_epochs):
            def cl2():
                opt.zero_grad(set_to_none=True)
                l, _, _ = pde_loss(u_fn, xc.clone(), tc.clone(), M, 1.0, eps=None)
                l = l + bc_loss(u_fn, tb.clone(), M, 1.0)
                if torch.isfinite(l): l.backward()
                return l
            v = float(opt.step(cl2).detach())
            if not np.isfinite(v): break
            if ep in (0, 4, 19, 49, 99, 199, 399) or ep == A.phys_epochs - 1:
                with torch.no_grad():
                    p = u_fn(xf, tf)
                    amp = float(p.norm() / uf.norm())
                print(f"     {ep+1:>7}{rel(u_fn):>10.4f}%{v:>14.4e}{amp:>18.4f}", flush=True)
        del m; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
