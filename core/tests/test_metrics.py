"""Coverage for metrics.py (previously zero -- audit miss #15) and the
two-sided trivial-solution guard (audit A7: the one-sided version was silent on
the frozen-IC failure mode of the `poly` ansatz)."""
import sys
import numpy as np
from core.problem import Homogeneous, MultiLayer, gaussian_ic
from core.reference import fd_reference
from core.metrics import (spacetime_rel_l2, per_snapshot_rel_l2,
                          per_snapshot_fixed_den, amplitude_norm_ratio,
                          discrete_energy, RATIO_LO, RATIO_HI)

R = []
def check(n, ok, d=""):
    R.append(ok); print(f"  [{'PASS' if ok else 'FAIL'}] {n:<50} {d}")

ref = np.array([[1.0, 2.0], [3.0, 4.0], [0.5, 0.5]])

print("\n1. spacetime_rel_l2 basic properties")
check("identical -> 0%", spacetime_rel_l2(ref, ref) == 0.0)
check("zero prediction -> 100%", abs(spacetime_rel_l2(np.zeros_like(ref), ref) - 100.0) < 1e-9)
check("2x amplitude -> 100%", abs(spacetime_rel_l2(2 * ref, ref) - 100.0) < 1e-9)
check("scale invariant", abs(spacetime_rel_l2(7 * ref, 7 * ref * 1.1)
                             - spacetime_rel_l2(ref, ref * 1.1)) < 1e-9)
check("no divide-by-zero on zero reference",
      np.isfinite(spacetime_rel_l2(np.ones((2, 2)), np.zeros((2, 2)))))
check("is a MSE-type norm, not mean-abs",
      abs(spacetime_rel_l2(ref + np.array([[10., 0.], [0., 0.], [0., 0.]]), ref)
          - 100 * 10 / np.linalg.norm(ref)) < 1e-9)

print("\n2. per-snapshot variants")
# A fixed denominator must NOT return 100% at every snapshot -- that is exactly
# the drifting-denominator artefact it exists to remove.  For a zero prediction
# it must return 100 * ||u_ref[k]|| / ||u_ref[0]||.
_pf = per_snapshot_fixed_den(np.zeros_like(ref), ref)
_want = 100.0 * np.linalg.norm(ref, axis=1) / np.linalg.norm(ref[0])
check("fixed-denominator == 100*||ref[k]||/||ref[0]||", np.allclose(_pf, _want),
      f"{np.round(_pf,2)} vs {np.round(_want,2)}")
check("fixed-denominator is exactly 100% at t=0", abs(_pf[0] - 100.0) < 1e-9)
check("drifting denominator differs from fixed (the artefact)",
      not np.allclose(per_snapshot_rel_l2(np.zeros_like(ref), ref),
                      per_snapshot_fixed_den(np.zeros_like(ref), ref)))

print("\n3. Two-sided trivial-solution guard (A7)")
M = MultiLayer()
x, t, u = fd_reference(M, nx=256, T=1.0, sigma_g=0.1)
idx = [int(np.argmin(np.abs(t - v))) for v in np.linspace(0.05, 1.0, 20)]
uref = u[idx]
cases = {
    "exact":        uref.copy(),
    "collapse":     uref * np.exp(-8 * np.linspace(0, 1, len(idx)))[:, None],
    "frozen IC":    np.repeat(gaussian_ic(x, 0.1)[None, :], len(idx), axis=0),
    "2x amplitude": 2 * uref,
}
def flagged(p):
    r = amplitude_norm_ratio(p, uref)
    return bool(np.any(r < RATIO_LO) or np.any(r > RATIO_HI))
for nm, p in cases.items():
    r = amplitude_norm_ratio(p, uref)
    f = flagged(p)
    want = nm != "exact"
    check(f"{nm:<13} flagged={f}", f == want,
          f"L2={spacetime_rel_l2(p,uref):6.1f}%  ratio[{r.min():.2f},{r.max():.2f}]")

print("\n4. discrete_energy is reference-free and ~conserved before the wave exits")
x2, t2, u2 = fd_reference(Homogeneous(), nx=1024, T=0.5, sigma_g=0.1)
E = discrete_energy(u2, x2, t2, Homogeneous())
mid = E[len(E)//4: 3*len(E)//4]
drift = float((mid.max() - mid.min()) / mid.mean())
check("energy drift < 5% while wave is interior", drift < 0.05, f"drift={100*drift:.2f}%")
check("energy is positive", bool(np.all(E[1:] > 0)))

print("\n" + "=" * 72)
nf = len(R) - sum(R)
print(f"{sum(R)}/{len(R)} passed" + ("" if nf == 0 else f"   *** {nf} FAILURES ***"))
sys.exit(1 if nf else 0)
