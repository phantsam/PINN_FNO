"""Single training loop, used by EVERY architecture.

Held fixed across models (so only the network varies):
  * the residual, ansatz, BC loss, and metric from core/
  * normalised residual + Wang et al. eps-annealing (DECISIONS.md D5, D6)
  * grad-norm balancing between PDE and BC terms
  * optimiser, LR schedule, collocation counts, step budget, seed

R3 adaptive resampling is ported from the rak branch with its early-return bug
fixed: there, `if stopped: return` fired before the R3Sampler was ever built, so
runs labelled "R3" silently ran plain Sobol (3 of 6 cells in the rak retrain).
"""
from __future__ import annotations
import time
import numpy as np
import torch
import torch.optim as optim

from .problem import Material
from .operators import make_ansatz, wave_operator
from .losses import (residual_scale, bc_scale, pde_loss, bc_loss,
                     grad_norm_weights, CausalScheduler, causal_weights)
from .reference import fd_reference
from .evaluate import evaluate


def sample(n_int, n_bc, material, t_max, device, gen):
    x = torch.rand(n_int, 1, generator=gen, device=device) * (material.x_max - material.x_min) + material.x_min
    t = torch.rand(n_int, 1, generator=gen, device=device) * t_max
    tb = torch.rand(n_bc, 1, generator=gen, device=device) * t_max
    return x, t, tb


class R3Sampler:
    """Retain the highest-residual points, resample the rest (Daw et al. style)."""

    def __init__(self, n, material, t_max, device, gen):
        self.n, self.m, self.t_max = n, material, t_max
        self.device, self.gen = device, gen
        self.x, self.t, _ = sample(n, 1, material, t_max, device, gen)
        self.n_retained = 0

    def update(self, u_fn, scale):
        with torch.enable_grad():
            r = (wave_operator(u_fn, self.x.clone(), self.t.clone(), self.m) / scale) ** 2
        r = r.detach().squeeze(-1)
        keep = r > r.mean()
        nk = int(keep.sum())
        self.n_retained = nk
        xn, tn, _ = sample(self.n - nk, 1, self.m, self.t_max, self.device, self.gen)
        self.x = torch.cat([self.x[keep].detach(), xn])
        self.t = torch.cat([self.t[keep].detach(), tn])


def lbfgs_refine(model, u_fn, material, rs, bs, *, n_int, n_bc, t_max, device, gen,
                 iters=500, w_pde=1.0, w_bc=1.0, log=None):
    """Post-Adam L-BFGS refinement on a FIXED collocation set.

    Causal weighting is off here (eps=None): L-BFGS needs a deterministic
    objective, and by this stage the schedule has served its purpose.
    Note `line_search_fn="strong_wolfe"` -- spelled correctly; the misspelling
    in ML/src/train.py:307 raises RuntimeError on the first step.
    """
    x, t, tb = sample(n_int, n_bc, material, t_max, device, gen)
    x, t, tb = x.detach(), t.detach(), tb.detach()
    opt = optim.LBFGS(model.parameters(), lr=1.0, max_iter=iters, max_eval=iters,
                      history_size=50, tolerance_grad=1e-9,
                      tolerance_change=1e-12, line_search_fn="strong_wolfe")
    state = {"n": 0, "last": float("nan")}

    def closure():
        opt.zero_grad(set_to_none=True)
        lp, _, _ = pde_loss(u_fn, x.clone(), t.clone(), material, rs, eps=None)
        lb = bc_loss(u_fn, tb.clone(), material, bs)
        loss = w_pde * lp + w_bc * lb
        if torch.isfinite(loss):
            loss.backward()
        state["n"] += 1; state["last"] = float(loss)
        if log and state["n"] % 100 == 0:
            log(f"      lbfgs {state['n']:>4}  loss {float(loss):.4e}")
        return loss

    try:
        opt.step(closure)
    except RuntimeError as e:
        if log:
            log(f"      lbfgs aborted: {e}")
    return state["n"], state["last"]


def train_lbfgs(arch_fn, material: Material, *, ansatz_kind="legacy", epochs=700,
                seed=0, n_col=10000, n_bc=512, sigma_g=0.1, t_max=1.0,
                use_bc=True, patience=50, min_delta=1e-9, device=None,
                x_ref=None, t_ref=None, u_ref=None, save_path=None, log=None,
                normalise=False, sobol=True, use_r3=False, r3_frac=0.5,
                grid_updates=0):
    """L-BFGS on a FIXED collocation set -- the recipe the rak branch uses.

    Critical detail: the collocation points are sampled ONCE and reused for every
    epoch.  L-BFGS builds a quasi-Newton curvature estimate, which requires a
    deterministic objective; resampling each step (as the Adam path does) destroys
    it.  Getting this wrong made every PINN in Phase 4 look 10-70x worse than it is.

    Best-loss weights are tracked and restored, as in rak.
    """
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

    # DO NOT normalise the residual on the L-BFGS path.
    #
    # D6 normalisation exists so the Wang et al. causal-eps sequence [1e-2..1e2]
    # is meaningful -- and this path never uses causal weighting (eps=None).
    # Meanwhile torch's LBFGS picks its FIRST step as t = min(1, 1/||grad||_1)*lr.
    # Dividing the loss by residual_scale^2 (~5e4) shrinks gradients by the same
    # factor, so 1/||grad||_1 explodes and step one overshoots catastrophically.
    # Measured: WavKAN 92.63% (stuck, 6s) -> 0.88% simply by not normalising.
    # rak does not normalise, and rak never collapses.
    rs = residual_scale(material, sigma_g) if normalise else 1.0
    bs = bc_scale(material, sigma_g) if normalise else 1.0
    if sobol:                      # low-discrepancy: rak uses Sobol, not torch.rand
        eng = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=seed)
        pts = eng.draw(n_col)
        xc = (pts[:, 0:1] * (material.x_max - material.x_min) + material.x_min).to(device)
        tc = (pts[:, 1:2] * t_max).to(device)
        _, _, tb = sample(1, n_bc, material, t_max, device, gen)
    else:
        xc, tc, tb = sample(n_col, n_bc, material, t_max, device, gen)
    xc, tc, tb = xc.detach(), tc.detach(), tb.detach()          # FIXED

    # Only optimise leaves that actually receive a gradient.  pykan registers a
    # symbolic branch that is inert under symbolic_enabled=False; including it
    # padded every L-BFGS history vector with 28.6% permanent zeros.  Those slots
    # contribute exactly 0 to y.s, to the two-loop recursion and to ||g||_1, so
    # filtering them is numerically a no-op (verified against the stored Phase 6
    # value) and simply shrinks the optimiser state.
    live = [q for q in model.parameters() if q.requires_grad]
    opt = optim.LBFGS(live, lr=1.0, max_iter=20, history_size=100,
                      line_search_fn="strong_wolfe")
    best, best_w, wait, t0 = float("inf"), None, 0, time.time()
    n_retained = None
    r3_fired = False
    diverged = False
    # pykan's fit() re-fits the spline grids to the live activations 10 times
    # over the first half of training (update_grid=True, stop_grid_update_step =
    # steps/2).  Our loop never did, which left the grids mismatched to the
    # activations -- see TunedKAN.update_grid_ for the measured consequence.
    # Mirroring that schedule here; 0 keeps the previous behaviour exactly.
    if grid_updates and hasattr(model, "update_grid_"):
        half = max(1, epochs // 2)
        grid_eps_at = {int(i * half / grid_updates) for i in range(grid_updates)}
    else:
        grid_eps_at = set()

    for ep in range(epochs):
        if ep in grid_eps_at:
            model.update_grid_(xc.clone(), tc.clone())
            # the parameters have been re-fitted onto new knots, so the curvature
            # L-BFGS accumulated for the old ones is meaningless.
            opt = optim.LBFGS(live, lr=1.0, max_iter=20, history_size=100,
                              line_search_fn="strong_wolfe")
            wait = 0
        # rak's R3 schedule: train on the Sobol set, then RETAIN the high-residual
        # points and resample the rest, then continue with a FRESH optimiser
        # (stale curvature from the old point set would be invalid).
        if use_r3 and (not r3_fired) and wait >= patience:
            r3_fired = True
            with torch.enable_grad():
                r = (wave_operator(u_fn, xc.clone(), tc.clone(), material) / rs) ** 2
            r = r.detach().squeeze(-1)
            keep = r > r.mean()
            n_retained = int(keep.sum())
            xn, tn, _ = sample(n_col - n_retained, 1, material, t_max, device, gen)
            xc = torch.cat([xc[keep].detach(), xn]).detach()
            tc = torch.cat([tc[keep].detach(), tn]).detach()
            opt = optim.LBFGS(live, lr=1.0, max_iter=20,
                              history_size=100, line_search_fn="strong_wolfe")
            wait = 0
            best = float("inf")          # new point set -> old losses incomparable
        def closure():
            opt.zero_grad(set_to_none=True)
            l, _, _ = pde_loss(u_fn, xc.clone(), tc.clone(), material, rs, eps=None)
            if use_bc:
                l = l + bc_loss(u_fn, tb.clone(), material, bs)
            if torch.isfinite(l):
                l.backward()
            return l
        v = float(opt.step(closure))

        # Detect a line-search blow-up.  torch's LBFGS returns the loss at the
        # START of the step, so a NaN born inside _strong_wolfe is invisible in
        # `v`.  Mechanism (traced on twolayer/s2/wavkan): a unit trial step
        # reaches f2=6.7e18, g2=5.2e24; _cubic_interpolate then forms
        #     d1        = g1 + g2 - 3*(f1-f2)/(x1-x2)  ~ 5.2e24
        #     d2_square = d1**2 - g1*g2                ~ 2.7e49 -> inf in fp32
        # and returns (g2+d2-d1)/(g2-g1+2*d2) = inf/inf = NaN, which is written
        # into every parameter while `v` still reads a healthy 418.69.
        # So check the PARAMETERS, not the reported loss.
        if not all(torch.isfinite(q).all() for q in model.parameters()):
            diverged = True
            break
        if not np.isfinite(v):
            diverged = True
            break
        if v < best - min_delta:
            best, wait = v, 0
            best_w = {k: c.detach().clone() for k, c in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience and (r3_fired or not use_r3):
                break
    if best_w:
        model.load_state_dict(best_w)

    wall = time.time() - t0
    final = evaluate(u_fn, material, x_ref, t_ref, u_ref,
                     scale=residual_scale(material, sigma_g), n_residual=20000,
                     t_max=t_max, device=device, dtype=torch.float32, seed=999)
    final.pop("u_pred", None); final.pop("per_t_fixed", None); final.pop("per_t_drifting", None)
    final["norm_ratio"] = [float(v) for v in final["norm_ratio"]]
    final.update(wall_s=wall, epochs_run=ep + 1, best_loss=best, optimizer="lbfgs",
                 normalised=normalise, sobol=sobol, r3=use_r3, r3_retained=n_retained,
                 diverged=diverged, grid_updates=grid_updates)
    if save_path:
        torch.save({"state_dict": model.state_dict(),
                    "metrics": {k: v for k, v in final.items() if k != "norm_ratio"}}, save_path)
    return model, final


def train(arch_fn, material: Material, *, ansatz_kind="legacy", steps=20000,
          seed=0, lr=1e-3, n_int=8192, n_bc=256, n_chunks=32, sigma_g=0.1,
          t_max=1.0, use_r3=False, r3_every=2000, device=None, eval_every=2000,
          x_ref=None, t_ref=None, u_ref=None, log=None, lbfgs_iters=0,
          save_path=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)

    model = arch_fn().to(device)
    ansatz = make_ansatz(ansatz_kind, sigma_g=sigma_g)
    u_fn = lambda x, t: ansatz(model(x, t), x, t)

    # PirateNet zero-initialises its output layer; without this LSQ step it is
    # identically zero at init (u_tt == 0, dead backbone).  Paper-faithful, and
    # matches ML/src/train.py's behaviour.
    if hasattr(model, "physics_informed_init"):
        from .problem import gaussian_ic
        xi = torch.rand(4096, 1, generator=gen, device=device) * (material.x_max - material.x_min) + material.x_min
        ti = torch.rand(4096, 1, generator=gen, device=device) * t_max
        model.physics_informed_init(xi, ti, gaussian_ic(xi, sigma_g))

    rs, bs = residual_scale(material, sigma_g), bc_scale(material, sigma_g)
    sched = CausalScheduler()
    opt = optim.Adam(model.parameters(), lr=lr)
    lr_sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 1e-2)
    w_pde = w_bc = 1.0
    r3 = R3Sampler(n_int, material, t_max, device, gen) if use_r3 else None

    hist, t0 = [], time.time()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        if r3 is not None:
            if step > 0 and step % r3_every == 0:
                r3.update(u_fn, rs)                     # always runs; no early return
            x, t = r3.x.clone(), r3.t.clone()
            _, _, tb = sample(1, n_bc, material, t_max, device, gen)
        else:
            x, t, tb = sample(n_int, n_bc, material, t_max, device, gen)

        lp, chunk, w = pde_loss(u_fn, x, t, material, rs, n_chunks=n_chunks,
                                eps=sched.eps, t_max=t_max)
        lb = bc_loss(u_fn, tb, material, bs)
        loss = w_pde * lp + w_bc * lb
        if not torch.isfinite(loss):
            break
        loss.backward(); opt.step(); lr_sched.step()

        if w is not None:
            sched.step(w)
        if (step + 1) % 500 == 0:
            xa, ta, tba = sample(n_int, n_bc, material, t_max, device, gen)
            la, _, _ = pde_loss(u_fn, xa, ta, material, rs, n_chunks=n_chunks,
                                eps=sched.eps, t_max=t_max)
            lba = bc_loss(u_fn, tba, material, bs)
            w_pde, w_bc = grad_norm_weights(model, la, lba)
        if (step + 1) % eval_every == 0 and x_ref is not None:
            m = evaluate(u_fn, material, x_ref, t_ref, u_ref, scale=rs,
                         n_residual=4000, t_max=t_max, device=device,
                         dtype=torch.float32, seed=12345)
            hist.append((step + 1, m["rel_l2"], m["residual_rms_heldout"], sched.eps))
            if log:
                log(f"      step {step+1:>6}  L2 {m['rel_l2']:7.2f}%  "
                    f"resid {m['residual_rms_heldout']:.3e}  eps {sched.eps:.0e}")

    n_lb, lb_loss = (0, float("nan"))
    if lbfgs_iters > 0:
        n_lb, lb_loss = lbfgs_refine(model, u_fn, material, rs, bs, n_int=n_int,
                                     n_bc=n_bc, t_max=t_max, device=device, gen=gen,
                                     iters=lbfgs_iters, w_pde=w_pde, w_bc=w_bc, log=log)

    wall = time.time() - t0
    final = evaluate(u_fn, material, x_ref, t_ref, u_ref, scale=rs, n_residual=20000,
                     t_max=t_max, device=device, dtype=torch.float32, seed=999)
    final.pop("u_pred", None); final.pop("per_t_fixed", None); final.pop("per_t_drifting", None)
    final["norm_ratio"] = [float(v) for v in final["norm_ratio"]]
    final.update(wall_s=wall, hist=hist, final_eps=sched.eps,
                 r3_retained=(r3.n_retained if r3 else None),
                 lbfgs_evals=n_lb, lbfgs_loss=lb_loss)
    if save_path:                       # the stale-weights pitfall: always persist
        torch.save({"state_dict": model.state_dict(), "metrics":
                    {k: v for k, v in final.items() if k != "norm_ratio"}}, save_path)
    return model, final
