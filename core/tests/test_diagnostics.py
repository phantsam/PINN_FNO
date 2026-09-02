"""Coverage for diagnostics.py.

A screening tool is only worth running if it FAILS on the things it claims to
catch, so each check below builds a model with a known, deliberate pathology and
asserts the screen fires -- and asserts it stays quiet on a healthy one.

The pathologies are the ones this project actually hit:
  * exploding second derivatives (hand-rolled SplineKAN, max|u_tt| ~ 5e5)
  * a constant network hidden behind the hard-IC ansatz (PirateNet zero-init
    without physics_informed_init)
  * parameters disconnected from the loss (pykan's inert symbolic branch)
"""
import sys
import torch
import torch.nn as nn

from core.problem import MATERIALS
from core.diagnostics import (screen, derivative_scale, gradient_health,
                              VANISH_RMS, EXPLODE_RMS, DEAD_OUT, UTT_HI)

R = []
def check(n, ok, d=""):
    R.append(bool(ok)); print(f"  [{'PASS' if ok else 'FAIL'}] {n:<56} {d}")

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
M = MATERIALS["twolayer"]()


class Healthy(nn.Module):
    def __init__(self, u=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, u), nn.Tanh(),
                                 nn.Linear(u, u), nn.Tanh(), nn.Linear(u, 1))
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x, t):
        return self.net(torch.cat([x, t], -1))


class Exploding(Healthy):
    """Same net, scaled so the second derivative blows past UTT_HI."""
    def forward(self, x, t):
        return 1e6 * self.net(torch.cat([x, t], -1))


class DeadOutput(Healthy):
    """Zero-initialised output layer and no warm start: constant 0 network."""
    def __init__(self):
        super().__init__()
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)


class Disconnected(Healthy):
    """Carries a trainable tensor that never enters forward()."""
    def __init__(self):
        super().__init__()
        self.orphan = nn.Parameter(torch.randn(16, 4))


print("\n1. Healthy model raises nothing")
torch.manual_seed(0)
ok, w, rep = screen(Healthy().to(DEV), M, name="healthy", device=DEV,
                    n=1024, n_col=1024, n_bc=128)
check("healthy passes", ok)
check("healthy produces no warnings", w == [], f"{w}")
check("derivatives finite", rep["finite"])
check("raw output has real spread", rep["raw_std"] > DEAD_OUT, f"std={rep['raw_std']:.3e}")
check("gradient reaches every group", rep["n_no_grad"] == 0)

print("\n2. Exploding second derivative is caught")
torch.manual_seed(0)
ok, w, rep = screen(Exploding().to(DEV), M, name="exploding", device=DEV,
                    n=1024, n_col=1024, n_bc=128)
check("u_tt exceeds the ceiling", rep["u_tt"] > UTT_HI, f"|u_tt|={rep['u_tt']:.3e}")
check("a warning names it", any("too stiff" in s for s in w), f"{len(w)} warning(s)")

print("\n3. Constant network hidden behind the ansatz is caught")
torch.manual_seed(0)
mm = DeadOutput().to(DEV)
ok, w, rep = screen(mm, M, name="dead", device=DEV, n=1024, n_col=1024, n_bc=128)
check("raw output std is ~0", rep["raw_std"] < DEAD_OUT, f"std={rep['raw_std']:.3e}")
check("DEAD OUTPUT warning fired", any("DEAD OUTPUT" in s for s in w), f"{w[:1]}")
check("but |u_tt| still looks healthy (why the raw check is needed)",
      rep["u_tt"] > 1.0, f"|u_tt|={rep['u_tt']:.3e} from the IC term alone")

print("\n4. Disconnected parameters are caught")
torch.manual_seed(0)
ok, w, rep = screen(Disconnected().to(DEV), M, name="orphan", device=DEV,
                    n=1024, n_col=1024, n_bc=128)
check("orphan tensor reported", rep["n_no_grad"] >= 1, f"n_no_grad={rep['n_no_grad']}")
check("a warning names it", any("NO gradient" in s for s in w))

print("\n5. Grouping is per-layer, not per-model")
torch.manual_seed(0)
h = gradient_health(Healthy().to(DEV), M, device=DEV, n_col=512, n_bc=64)
check("three Linear layers -> three groups", h["n_groups"] == 3, f"{sorted(h['per_group'])}")
check("depth ratio is finite and modest", h["depth_ratio"] < 1e4, f"{h['depth_ratio']:.3g}")
check("global grad norm positive", h["global_grad_norm"] > 0)

print("\n6. derivative_scale is deterministic for a fixed seed")
torch.manual_seed(0); a = Healthy().to(DEV)
d1 = derivative_scale(a, M, device=DEV, seed=7, n=512)
d2 = derivative_scale(a, M, device=DEV, seed=7, n=512)
check("same seed -> identical", all(abs(d1[k] - d2[k]) < 1e-12
                                   for k in ("u", "u_x", "u_xx", "u_t", "u_tt")))
d3 = derivative_scale(a, M, device=DEV, seed=8, n=512)
check("different seed -> different sample", d3["u_tt"] != d1["u_tt"])

print("\n7. Screen leaves no gradient state behind")
torch.manual_seed(0); a = Healthy().to(DEV)
screen(a, M, device=DEV, n=256, n_col=256, n_bc=32)
check("grads cleared after screening",
      all(p.grad is None or float(p.grad.abs().sum()) == 0 for p in a.parameters()))

print("\n" + "=" * 78)
print(f"  {sum(R)}/{len(R)} passed")
sys.exit(0 if all(R) else 1)
