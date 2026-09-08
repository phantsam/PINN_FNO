"""Adam -> L-BFGS hybrid training, plus a PAPER-FAITHFUL R3 sampler.

Why this module exists
----------------------
Phases 6-9 trained with L-BFGS alone.  That is one point in a two-dimensional
design space and the other axis was never tested, so "PINNs beat KANs" was
established only under a single optimiser.  The standard PINN recipe -- and the
one this project's own main branch uses (ML/training_code.py:666, max_epochs=700
Adam then lbfgs_epochs=200 L-BFGS) -- is Adam first, L-BFGS second: Adam finds
the basin, L-BFGS drives the loss down inside it.

What is faithful to the R3 paper, and what was not
--------------------------------------------------
Checked against the official release, arkadaw9/r3_sampling_icml2023:

  * `grep -rn "LBFGS|lbfgs"` over the whole repository returns NOTHING.  Every
    configuration trains with torch.optim.Adam (one RAR baseline uses Adamax).
    The authors never combined R3 with a quasi-Newton method.
  * `sampler.update()` is called from inside `def loss()` (pinn_model.py), so it
    runs on EVERY iteration -- not on a period, and not once.
  * The rule (sampler.py:43) is: fitness = |residual|; retain the points with
    fitness > mean(fitness); release the rest and resample them uniformly.

train.py's R3 fires ONCE, when L-BFGS first stalls, and its Adam path resamples
every 2000 steps.  Neither is the published algorithm; both were labelled "Daw
et al." in earlier write-ups.  `R3Pool` below is the published algorithm.

The middle ground
-----------------
Every-iteration resampling and L-BFGS are genuinely incompatible: L-BFGS builds
a curvature estimate from (s_k, y_k) pairs that assume ONE fixed objective, and
changing the collocation set every step invalidates it.  So R3 runs where the
paper runs it -- throughout the Adam phase -- and the L-BFGS phase inherits the
pool R3 converged to and FREEZES it.  The adaptive sampling therefore chooses
*where* L-BFGS refines, without corrupting its curvature.

Objective continuity
--------------------
Both phases minimise the identical unnormalised loss `pde_loss + bc_loss` with
eps=None (no causal weighting, no grad-norm balancing).  This is deliberate: if
the objective changed at the handoff, L-BFGS would inherit a warm start for a
function it is not minimising.  Not normalising also matters for the L-BFGS
phase specifically -- see the note in train.train_lbfgs.
"""
from __future__ import annotations
import time
import numpy as np
import torch
import torch.optim as optim

from .problem import Material
from .operators import make_ansatz, wave_operator
from .losses import (residual_scale, bc_scale, pde_loss, bc_loss, ic_loss,
                     grad_norm_weights)
from .evaluate import evaluate
from .train import sample


def sobol_pool(n, material, t_max, device, seed):
    """The same low-discrepancy collocation set train_lbfgs uses, same seed."""
    eng = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=seed)
    pts = eng.draw(n)
    x = (pts[:, 0:1] * (material.x_max - material.x_min) + material.x_min).to(device)
    t = (pts[:, 1:2] * t_max).to(device)
    return x.detach(), t.detach()


class R3Pool:
    """Retain-Resample-Release, Daw et al. ICML 2023, as published.

    Reference: arkadaw9/r3_sampling_icml2023, sampler.py::R3Sampler.get_old_new

        fitness = |residual|
        mask    = fitness > fitness.mean()
        retain  self.x[mask]
        refill  N - mask.sum() points drawn uniformly

    `update` is meant to be called every optimiser step.  The retained count is
    not fixed -- it is whatever exceeds the mean, which for a heavy-tailed
    residual is typically well under half the pool.  Tracking it is the cheapest
    diagnostic that the sampler is actually doing something: a count that sits
    at n/2 means the residual has gone uniform and R3 has nothing left to chase.
    """

    def __init__(self, n, material, t_max, device, gen, seed):
        self.n, self.m, self.t_max = n, material, t_max
        self.device, self.gen = device, gen
        self.x, self.t = sobol_pool(n, material, t_max, device, seed)
        self.retained = []

    def update(self, u_fn):
        with torch.enable_grad():
            r = wave_operator(u_fn, self.x.clone().requires_grad_(True),
                              self.t.clone().requires_grad_(True), self.m)
        fitness = r.detach().abs().squeeze(-1)
        if not torch.isfinite(fitness).all():
            return                                   # diverged; leave pool alone
        keep = fitness > fitness.mean()
        nk = int(keep.sum())
        self.retained.append(nk)
        xn, tn, _ = sample(self.n - nk, 1, self.m, self.t_max, self.device, self.gen)
        self.x = torch.cat([self.x[keep], xn.detach()]).detach()
        self.t = torch.cat([self.t[keep], tn.detach()]).detach()


def _finite(model):
    return all(torch.isfinite(q).all() for q in model.parameters())


def train_hybrid(arch_fn, material: Material, *, ansatz_kind="legacy",
                 adam_steps=700, lbfgs_epochs=3000, seed=0, lr=1e-3,
                 n_col=10000, n_bc=512, sigma_g=0.1, t_max=1.0,
                 use_r3=False, use_bc=True, soft_ic=False, n_ic=512,
                 patience=50, min_delta=1e-9,
                 device=None, x_ref=None, t_ref=None, u_ref=None,
                 save_path=None, log=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)

    model = arch_fn().to(device)
    ansatz = make_ansatz(ansatz_kind, sigma_g=sigma_g)
    u_fn = lambda x, t: ansatz(model(x, t), x, t)
    if hasattr(model, "physics_informed_init"):
        from .problem import gaussian_ic
        xi = torch.rand(4096, 1, generator=gen, device=device) * (material.x_max - material.x_min) + material.x_min
        ti = torch.rand(4096, 1, generator=gen, device=device) * t_max
        model.physics_informed_init(xi, ti, gaussian_ic(xi, sigma_g))

    rs = bs = 1.0                                    # unnormalised; see docstring
    _, _, tb = sample(1, n_bc, material, t_max, device, gen)
    tb = tb.detach()
    # SOFT initial condition.  The hard ansatz makes u == 0 a valid minimiser of
    # the residual for t > ~0.2, because decay(t) has died by then and u ~=
    # growth(t)*N; driving N to zero satisfies the PDE and the absorbing BCs
    # exactly.  Every ~95 % result in this project is that basin.  A soft IC
    # penalises u == 0 directly, so it should remove the basin -- which is worth
    # testing precisely on the models that fell into it.
    xi = ((torch.rand(n_ic, 1, generator=gen, device=device)
           * (material.x_max - material.x_min) + material.x_min).detach()
          if soft_ic else None)
    w_pde = w_aux = 1.0
    live = [q for q in model.parameters() if q.requires_grad]
    t0 = time.time()

    def terms(x, t):
        lp, _, _ = pde_loss(u_fn, x.clone(), t.clone(), material, rs, eps=None)
        aux = 0.0
        if use_bc:
            aux = aux + bc_loss(u_fn, tb.clone(), material, bs)
        if soft_ic:
            aux = aux + ic_loss(u_fn, xi.clone(), sigma_g)
        return lp, aux

    def objective(x, t):
        lp, aux = terms(x, t)
        return w_pde * lp + w_aux * aux if soft_ic else lp + aux

    # ---- phase A: Adam -----------------------------------------------------
    pool = R3Pool(n_col, material, t_max, device, gen, seed) if use_r3 else None
    xc, tc = (pool.x, pool.t) if pool else sobol_pool(n_col, material, t_max,
                                                      device, seed)
    opt = optim.Adam(live, lr=lr)
    # ML/training_code.py:695 -- warmup = decay = max_epochs//10, decay_rate 0.9
    warm = max(10, adam_steps // 10)
    lam = lambda s: min(1.0, (s + 1) / warm) * (0.9 ** (s / warm))
    lr_sched = optim.lr_scheduler.LambdaLR(opt, lam)

    # adam_steps=0 is a legitimate configuration: it runs pure L-BFGS through
    # this same code path, which is the honest control for "what did the Adam
    # phase change?".  `loss` must therefore exist even when the loop never runs.
    adam_hist, diverged, loss = [], False, None
    for step in range(adam_steps):
        if pool is not None:
            pool.update(u_fn)                        # EVERY step, per the paper
            xc, tc = pool.x, pool.t
        opt.zero_grad(set_to_none=True)
        loss = objective(xc, tc)
        if not torch.isfinite(loss):
            diverged = True; break
        loss.backward(); opt.step(); lr_sched.step()
        if not _finite(model):
            diverged = True; break
        # Rebalance often enough to matter over a 700-step Adam phase: at the
        # 500-step period used elsewhere this fires ONCE, leaving the IC term at
        # weight 1 for most of training -- and at weight 1 the u==0 basin is
        # still cheap (the whole IC penalty is only ~0.1).
        if soft_ic and (step + 1) % 100 == 0:
            lp, aux = terms(xc, tc)
            w_pde, w_aux = grad_norm_weights(model, lp, aux)
        if (step + 1) % 100 == 0:
            adam_hist.append((step + 1, float(loss.detach())))
            if log:
                log(f"      adam  {step+1:>5}  loss {float(loss):.4e}"
                    + (f"  retained {pool.retained[-1]}" if pool else ""))
    adam_loss = float(loss.detach()) if (loss is not None and not diverged) else float("nan")

    # ---- phase B: L-BFGS on a FROZEN set -----------------------------------
    # If R3 ran, the frozen set is the pool it converged to: adaptive sampling
    # chose the points, L-BFGS refines on them with a valid curvature estimate.
    xc, tc = xc.detach().clone(), tc.detach().clone()
    opt = optim.LBFGS(live, lr=1.0, max_iter=20, history_size=100,
                      line_search_fn="strong_wolfe")
    best, best_w, wait, ep = float("inf"), None, 0, -1
    if not diverged:
        for ep in range(lbfgs_epochs):
            def closure():
                opt.zero_grad(set_to_none=True)
                l = objective(xc, tc)
                if torch.isfinite(l):
                    l.backward()
                return l
            v = float(opt.step(closure).detach())
            # torch reports the loss at the START of the step, so a NaN born
            # inside _strong_wolfe is invisible in `v`.  Check the parameters.
            if not _finite(model) or not np.isfinite(v):
                diverged = True; break
            if v < best - min_delta:
                best, wait = v, 0
                best_w = {k: c.detach().clone() for k, c in model.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    break
            if log and (ep + 1) % 200 == 0:
                log(f"      lbfgs {ep+1:>5}  loss {v:.4e}")
    if best_w:
        model.load_state_dict(best_w)

    wall = time.time() - t0
    final = evaluate(u_fn, material, x_ref, t_ref, u_ref,
                     scale=residual_scale(material, sigma_g), n_residual=20000,
                     t_max=t_max, device=device, dtype=torch.float32, seed=999)
    final.pop("u_pred", None); final.pop("per_t_fixed", None); final.pop("per_t_drifting", None)
    final["norm_ratio"] = [float(v) for v in final["norm_ratio"]]
    final.update(wall_s=wall, optimizer="adam+lbfgs", adam_steps=adam_steps,
                 adam_loss=adam_loss, adam_hist=adam_hist,
                 lbfgs_epochs_run=ep + 1, best_loss=best, diverged=diverged,
                 r3=use_r3, r3_paper_faithful=use_r3, soft_ic=soft_ic,
                 r3_retained_mean=(float(np.mean(pool.retained)) if pool and pool.retained else None),
                 r3_retained_last=(pool.retained[-1] if pool and pool.retained else None))
    if save_path:
        torch.save({"state_dict": model.state_dict(),
                    "metrics": {k: v for k, v in final.items() if k != "norm_ratio"}},
                   save_path)
    return model, final
