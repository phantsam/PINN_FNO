"""Merge and summarise Phase 10 (Adam -> L-BFGS hybrid) results.

Reports per material x model: n, mean +- sd, min, and the seed-to-seed spread.
Also runs the two comparisons that matter, with the caveats attached:

  * mean difference, Welch t-test (unequal variances not assumed away)
  * variance ratio, Levene -- the KAN's excess seed variance was the ONE
    statistically significant difference found under pure L-BFGS, so whether the
    Adam warm-up removes it is a direct, pre-registered question

No selection after the fact: every model in the file is reported, and the tests
are the two named above run on the two named arms.  (Phase 8 produced a p=0.0000
by picking the three lowest values and then testing whether they were low; that
result was retracted.  See ARCHITECTURES.md S15.)
"""
from __future__ import annotations
import argparse, json, glob
import numpy as np

try:
    from scipy import stats
except ImportError:
    stats = None

ORDER = ["mlp", "fourier", "pykan_wide", "splinekan"]
LABEL = {"mlp": "MLP PINN", "fourier": "Fourier PINN",
         "pykan_wide": "plain KAN", "splinekan": "PIKAN"}
# measured under pure L-BFGS at nx=2048, n=11 -- the baseline this phase tests
LBFGS_BASELINE = {("homogeneous", "fourier"): (0.0281, 0.0073, 11),
                  ("homogeneous", "charcoords50 KAN"): (0.0313, 0.0170, 11),
                  ("multilayer", "fourier"): (0.0419, 0.0146, 11),
                  ("multilayer", "charcoords50 KAN"): (0.0574, 0.0375, 11)}


def load(patterns):
    rows = []
    for pat in patterns:
        for f in glob.glob(pat):
            rows += json.load(open(f))
    seen, out = set(), []
    for r in rows:                      # dedupe on the natural key
        k = (r["material"], r["seed"], r["model"])
        if k not in seen:
            seen.add(k); out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    A = ap.parse_args()
    rows = load(A.files)
    mats = sorted({r["material"] for r in rows},
                  key=lambda m: ["homogeneous", "twolayer", "multilayer"].index(m)
                  if m in ["homogeneous", "twolayer", "multilayer"] else 9)

    print(f"Phase 10 -- Adam(700) -> L-BFGS, scored at nx=2048   [{len(rows)} runs]\n")
    for mk in mats:
        print(f"== {mk} ==")
        print(f"{'model':<18}{'n':>3}{'mean':>10}{'sd':>9}{'min':>9}{'max':>9}"
              f"{'spread':>8}{'div':>5}")
        cells = {}
        keys = sorted({r["model"] for r in rows if r["material"] == mk},
                      key=lambda k: (ORDER.index(k.split("+")[0])
                                     if k.split("+")[0] in ORDER else 9, k))
        for k in keys:
            v = np.array([r["rel_l2"] for r in rows
                          if r["material"] == mk and r["model"] == k])
            nd = sum(1 for r in rows if r["material"] == mk
                     and r["model"] == k and r.get("diverged"))
            cells[k] = v
            base = k.split("+")[0]
            lab = LABEL.get(base, base) + ("+R3" if "+R3" in k else "")
            mark = "  <-- COLLAPSED (u~0)" if v.mean() > 50 else ""
            print(f"{lab:<18}{len(v):>3}{v.mean():>9.4f}%{v.std(ddof=1) if len(v)>1 else 0:>9.4f}"
                  f"{v.min():>8.4f}%{v.max():>8.4f}%{v.max()/max(v.min(),1e-12):>7.1f}x{nd:>5}{mark}")

        # pre-registered comparisons: best PINN vs best KAN, and plain vs +R3
        pinns = [k for k in cells if k.split("+")[0] in ("mlp", "fourier")]
        kans = [k for k in cells if k.split("+")[0] in ("pykan_wide", "splinekan")]
        if pinns and kans and stats is not None:
            bp = min(pinns, key=lambda k: cells[k].mean())
            kans = [k for k in kans if cells[k].mean() < 50] or kans
            bk = min(kans, key=lambda k: cells[k].mean())
            a, b = cells[bp], cells[bk]
            if len(a) > 1 and len(b) > 1:
                t = stats.ttest_ind(a, b, equal_var=False)
                lev = stats.levene(a, b)
                print(f"\n  best PINN {bp} {a.mean():.4f}%  vs  "
                      f"best KAN {bk} {b.mean():.4f}%")
                print(f"    ratio {b.mean()/a.mean():.3f}x   Welch p = {t.pvalue:.4f}"
                      f"   |   variance ratio {b.var(ddof=1)/a.var(ddof=1):.2f}x"
                      f"  Levene p = {lev.pvalue:.4f}")
        for k in sorted(cells):
            if "+R3" in k:
                continue
            kr = k + "+R3"
            if kr in cells and len(cells[k]) > 1 and len(cells[kr]) > 1 and stats:
                a, b = cells[k], cells[kr]
                t = stats.ttest_ind(a, b, equal_var=False)
                # A model at ~95-100 % error has collapsed to the trivial
                # solution u == 0, which satisfies the PDE and the absorbing BCs
                # exactly.  Going from "actively wrong" (>100 %) to "collapsed"
                # (~95 %) is a change of failure mode, not an improvement, so it
                # must not be reported as R3 helping.
                COLLAPSE = 50.0
                if a.mean() > COLLAPSE or b.mean() > COLLAPSE:
                    d = "BOTH FAILED - not a comparison"
                    print(f"    R3 on {LABEL.get(k, k):<14} {a.mean():.4f}% -> "
                          f"{b.mean():.4f}%  ({d})")
                    continue
                d = "helps" if b.mean() < a.mean() else "hurts"
                print(f"    R3 on {LABEL.get(k, k):<14} {a.mean():.4f}% -> "
                      f"{b.mean():.4f}%  ({b.mean()/a.mean():.2f}x, {d}, p={t.pvalue:.4f})")

        for (m, nm), (mu, sd, n) in LBFGS_BASELINE.items():
            if m == mk:
                print(f"    [pure L-BFGS baseline] {nm:<18} {mu:.4f}% +- {sd:.4f} (n={n})")
        print()


if __name__ == "__main__":
    main()
