"""Coverage for core/models.py -- previously ZERO.

The audit's lesson was that verifying the operator and assuming everything else
is how a suite goes green while the code is wrong.  models.py sits between the
verified physics and the results, so it needs its own invariants.

Most important test here: EVERY architecture must admit a finite, non-zero
SECOND time derivative.  The PDE residual contains u_tt; an architecture whose
activations kill the second derivative (ReLU, hard-tanh, ...) would silently
produce u_tt == 0 and train against a degenerate equation.
"""
import sys
import numpy as np, torch
from core.models import REGISTRY, n_params, SIGMA_B, FourierEmbed
from core.operators import make_ansatz, wave_operator
from core.problem import TwoLayer

torch.set_default_dtype(torch.float64)
from core.problem import gaussian_ic


def build(k, seed=0):
    """Construct a model the way core.train does -- including the LSQ output-layer
    init that PirateNet's zero-initialised `out` REQUIRES.  Without it PirateNet is
    identically zero (u_tt == 0, dead backbone); see section 0."""
    torch.manual_seed(seed)
    m = REGISTRY[k]()
    if hasattr(m, "physics_informed_init"):
        xi = torch.rand(2048, 1) * 2 - 1
        ti = torch.rand(2048, 1)
        m.physics_informed_init(xi, ti, gaussian_ic(xi, 0.1))
    return m


R = []
def check(n, ok, d=""):
    R.append(ok); print(f"  [{'PASS' if ok else 'FAIL'}] {n:<50} {d}")

print("\n0. PirateNet's zero-init REQUIRES physics_informed_init (regression)")
torch.manual_seed(0); _raw = REGISTRY["pirate"]()
_x = torch.linspace(-.6, .6, 16).reshape(-1, 1); _t = torch.full_like(_x, .4)
check("uninitialised PirateNet IS degenerate (documents the requirement)",
      float(_raw(_x, _t).abs().max()) == 0.0, "max|out|=0 before LSQ init")
_init = build("pirate")
check("after physics_informed_init it is non-degenerate",
      float(_init(_x, _t).abs().max()) > 1e-6, f"max|out|={float(_init(_x,_t).abs().max()):.3e}")
import inspect as _insp
from core import train as _tr
check("core.train.train() CALLS physics_informed_init",
      "physics_informed_init" in _insp.getsource(_tr.train))

print("\n1. Interface: every model maps (x,t) -> (N,1), finite")
for k in REGISTRY:
    m = build(k)
    x = torch.randn(16, 1); t = torch.rand(16, 1)
    o = m(x, t)
    check(f"{k}: shape (16,1) and finite",
          tuple(o.shape) == (16, 1) and bool(torch.isfinite(o).all()), f"p={n_params(m):,}")

print("\n2. Fourier bandwidth is FROZEN and equals SIGMA_B (D4 confound control)")
for k in ("fourier", "pirate", "splinekan"):
    m = build(k)
    trainable = any(n.split('.')[-1] == 'B' for n, _ in m.named_parameters())
    isbuf = any(n.endswith('B') for n, _ in m.named_buffers())
    sd = float(m.embed.B.std())
    check(f"{k}: B is a buffer, not trainable", (not trainable) and isbuf)
    check(f"{k}: measured bandwidth ~ SIGMA_B={SIGMA_B:.0f}",
          abs(sd - SIGMA_B) / SIGMA_B < 0.25, f"std(B)={sd:.2f}")
m = build("wavkan")
check("wavkan: correctly has NO Fourier embedding", not hasattr(m, "embed"))

print("\n3. u_tt exists, is finite, and is NON-ZERO for every architecture")
print("   (a model with u_tt==0 would train against a degenerate PDE)")
for k in REGISTRY:
    m = build(k)
    x = torch.linspace(-0.6, 0.6, 24).reshape(-1, 1).requires_grad_(True)
    t = torch.full_like(x, 0.4).requires_grad_(True)
    u = m(x, t)
    ut = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    utt = torch.autograd.grad(ut, t, torch.ones_like(ut), create_graph=True)[0]
    ok = bool(torch.isfinite(utt).all()) and float(utt.abs().max()) > 1e-9
    check(f"{k}: u_tt finite and non-zero", ok, f"max|u_tt|={float(utt.abs().max()):.3e}")

print("\n4. No PERMANENTLY dead parameters (checked after 5 optimiser steps)")
print("   PirateNet gates blocks with alpha=0 by design, so W1-3 are dead at step 0")
print("   only; the meaningful invariant is that they unfreeze, not that they start live.")
M = TwoLayer(); ans = make_ansatz("legacy", 0.1)
for k in REGISTRY:
    m = build(k)
    u_fn = lambda x, t: ans(m(x, t), x, t)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(5):
        opt.zero_grad()
        x = torch.rand(256, 1) * 2 - 1; t = torch.rand(256, 1)
        (wave_operator(u_fn, x, t, M) ** 2).mean().backward(); opt.step()
    m.zero_grad()
    x = torch.rand(256, 1) * 2 - 1; t = torch.rand(256, 1)
    (wave_operator(u_fn, x, t, M) ** 2).mean().backward()
    dead = [n for n, p in m.named_parameters()
            if p.requires_grad and (p.grad is None or float(p.grad.abs().sum()) == 0.0)]
    check(f"{k}: all parameters live after 5 steps", len(dead) == 0,
          f"{len(dead)} dead: {dead[:2]}" if dead else "")

print("\n5. Determinism: same seed -> identical weights and output")
for k in REGISTRY:
    a = build(k, 7); b = build(k, 7)
    x = torch.randn(8, 1); t = torch.rand(8, 1)
    same = bool(torch.allclose(a(x, t), b(x, t)))
    c = build(k, 8)
    diff = not bool(torch.allclose(a(x, t), c(x, t)))
    check(f"{k}: seeded reproducibly, and seeds differ", same and diff)

print("\n6. FourierEmbed bandwidth is honoured (not silently overridden)")
for s in (1.0, 10.0, 40.0):
    torch.manual_seed(0); e = FourierEmbed(4096, s)
    check(f"FourierEmbed(sigma={s}) -> std {float(e.B.std()):.2f}",
          abs(float(e.B.std()) - s) / s < 0.05)

print("\n" + "=" * 70)
nf = len(R) - sum(R)
print(f"{sum(R)}/{len(R)} passed" + ("" if nf == 0 else f"   *** {nf} FAILURES ***"))
sys.exit(1 if nf else 0)
