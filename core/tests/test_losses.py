"""Verify the loss module and the trivial-solution guard."""
import numpy as np, torch
from core.problem import Homogeneous, TwoLayer, MultiLayer, gaussian_ic
from core.operators import make_ansatz
from core.losses import (residual_scale, causal_weights, CausalScheduler,
                         pde_loss, bc_loss, grad_norm_weights)
from core.reference import fd_reference
from core.evaluate import evaluate

torch.set_default_dtype(torch.float64)
R = []
def check(n, ok, d=""):
    R.append(ok); print(f"  [{'PASS' if ok else 'FAIL'}] {n:<52} {d}")

print("\n1. Residual scale is fixed, positive, material-dependent")
sc = {M.name: residual_scale(M) for M in (Homogeneous(), TwoLayer(), MultiLayer())}
check("all scales positive & finite", all(np.isfinite(v) and v > 0 for v in sc.values()),
      " ".join(f"{k}={v:.0f}" for k, v in sc.items()))
check("scale is deterministic (same value twice)",
      residual_scale(TwoLayer()) == residual_scale(TwoLayer()))
check("Homogeneous scale == max|g''| (analytic cross-check)",
      abs(sc["Homogeneous"] - 227.5) / 227.5 < 0.01, f"{sc['Homogeneous']:.1f} vs 227.5")

print("\n2. Causal weights match Wang et al. Eq 3.2 exactly")
L = torch.tensor([0.5, 0.3, 0.2, 0.1])
w = causal_weights(L, eps=2.0)
man = torch.tensor([1.0, np.exp(-2*0.5), np.exp(-2*0.8), np.exp(-2*1.0)])
check("w_i = exp(-eps sum_{k<i} L_k)", torch.allclose(w, man), f"max diff {(w-man).abs().max():.2e}")
check("w_1 == 1 always", float(w[0]) == 1.0)

print("\n3. Annealing follows the paper's Algorithm 1")
s = CausalScheduler()
check("default seq == paper's [1e-2,1e-1,1,10,100]", s.eps_seq == [1e-2, 1e-1, 1.0, 10.0, 100.0])
check("default delta == 0.99", s.delta == 0.99)
s.step(torch.full((8,), 0.98)); check("no advance when min_w < delta", s.eps == 1e-2)
s.step(torch.full((8,), 0.999)); check("advance when min_w > delta", s.eps == 1e-1)
for _ in range(10): s.step(torch.full((8,), 1.0))
check("clamps at the last eps", s.eps == 100.0 and s.finished)

print("\n4. Normalisation puts chunk losses in the paper's usable range")
a = make_ansatz("legacy", 0.1)
u0 = lambda x, t: a(torch.zeros_like(x), x, t)
ok = True
for M in (Homogeneous(), TwoLayer(), MultiLayer()):
    x = torch.rand(8192, 1) * (M.x_max - M.x_min) + M.x_min; t = torch.rand(8192, 1)
    _, ch, _ = pde_loss(u0, x.clone(), t.clone(), M, residual_scale(M), eps=1.0)
    wmin = float(causal_weights(ch, 100.0).min()); wmax = float(causal_weights(ch, 1e-2).min())
    ok &= (wmax > 0.9) and (wmin < 0.5)
check("eps=1e-2 permissive AND eps=1e2 strict (seq spans range)", ok)

print("\n5. grad_norm_weights balances, never amplifies")
net = torch.nn.Linear(2, 1)
xt = torch.randn(64, 2)
big = (net(xt) ** 2).mean() * 1e4
small = (net(xt) ** 2).mean() * 1e-4
wa, wb = grad_norm_weights(net, big, small)
check("smaller loss gets LARGER weight", wb > wa, f"w_big={wa:.3e} w_small={wb:.3e}")
check("weighted magnitudes comparable",
      abs(np.log10((big.item()*wa) / (small.item()*wb))) < 1e-6)

print("\n6. Trivial-collapse guard fires on a decaying model, not on a good one")
M = Homogeneous(); s_ = residual_scale(M)
x_ref, t_ref, u_ref = fd_reference(M, nx=256, T=1.0, sigma_g=0.1)
tsub = np.linspace(0.05, 1.0, 20); idx = [int(np.argmin(np.abs(t_ref - v))) for v in tsub]
good = lambda x, t: torch.tensor(np.interp(0, [0], [0])) if False else _exact(x, t)
def _exact(x, t):
    return 0.5 * (gaussian_ic(x - t, 0.1) + gaussian_ic(x + t, 0.1))
collapse = lambda x, t: gaussian_ic(x, 0.1) * torch.exp(-40.0 * t ** 2)
mg = evaluate(good, M, x_ref, t_ref[idx], u_ref[idx], scale=s_, n_residual=4000)
mc = evaluate(collapse, M, x_ref, t_ref[idx], u_ref[idx], scale=s_, n_residual=4000)
check("good model NOT flagged", not mg["trivial_collapse"],
      f"rel-L2 {mg['rel_l2']:.2f}%  min-ratio {mg['min_late_norm_ratio']:.2f}")
check("collapsing model IS flagged", mc["trivial_collapse"],
      f"rel-L2 {mc['rel_l2']:.2f}%  min-ratio {mc['min_late_norm_ratio']:.2f}")

print("\n" + "=" * 74)
print(f"{sum(R)}/{len(R)} passed" + ("" if all(R) else "   *** FAILURES ***"))
