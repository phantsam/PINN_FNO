"""Is R3 useful?  Both variants, every paired cell we have.

Two DIFFERENT algorithms have been called "R3" in this project:

  ONE-SHOT (the rak branch's, used in Phases 6-7).  Fires once, when L-BFGS
  exhausts patience: retain the above-mean-residual points, resample the rest,
  restart the optimiser.  Pure L-BFGS throughout.

  PAPER-FAITHFUL (Daw et al. ICML 2023, Phase 10).  Resamples EVERY iteration.
  Runs through an Adam phase; L-BFGS then freezes the pool it converged to.

Both are evaluated here the only way that is fair: PAIRED on
(architecture, material, seed), plain vs R3, same everything else.  A paired
design removes seed and material variance, which is what makes n=45 pairs
informative where unpaired group means were not.

Collapse handling: a run at >50 % rel-L2 has fallen to the trivial solution
u == 0, which satisfies the PDE and the absorbing BCs exactly.  Those pairs are
reported SEPARATELY as rescues/collapses rather than folded into a ratio, because
"235 % -> 95 %" is a change of failure mode, not an improvement.
"""
from __future__ import annotations
import glob, json, os
import numpy as np
from scipy import stats

S = "/tmp/claude-1010/-home-prjgnn-PINN-FNO/a765de5d-6b07-41b3-9ed9-3624831a9432/scratchpad"
COLLAPSE = 50.0


def load_oneshot():
    """Phase 6 + 7.  Phase 7 re-ran the epoch-capped cells uncapped, so it wins."""
    cells = {}
    for f in ["p6_a.json", "p6_b.json"]:
        for r in json.load(open(os.path.join(S, f))):
            cells[(r["material"], r["arch"], r["seed"], bool(r["r3"]))] = r["rel_l2"]
    n7 = 0
    for f in ["p7_a.json", "p7_b.json"]:
        for r in json.load(open(os.path.join(S, f))):
            k = (r["material"], r["arch"], r["seed"], bool(r["r3"]))
            if k in cells:
                n7 += 1
            cells[k] = r["rel_l2"]
    return cells, n7


def load_faithful():
    cells = {}
    for f in glob.glob(os.path.join(S, "p10_*.json")):
        for r in json.load(open(f)):
            arch = r.get("arch", r["model"].split("+")[0])
            cells[(r["material"], arch, r["seed"], "+R3" in r["model"])] = r["rel_l2"]
    return cells


def analyse(name, cells, note=""):
    pairs = []
    for (m, a, s, is_r3) in list(cells):
        if is_r3:
            continue
        if (m, a, s, True) in cells:
            pairs.append((m, a, s, cells[(m, a, s, False)], cells[(m, a, s, True)]))
    print(f"\n{'='*78}\n{name}   -- {len(pairs)} paired cells{note}\n{'='*78}")

    rescue = [p for p in pairs if p[3] > COLLAPSE and p[4] <= COLLAPSE]
    ruin   = [p for p in pairs if p[3] <= COLLAPSE and p[4] > COLLAPSE]
    both   = [p for p in pairs if p[3] > COLLAPSE and p[4] > COLLAPSE]
    live   = [p for p in pairs if p[3] <= COLLAPSE and p[4] <= COLLAPSE]

    print(f"\n  collapse bookkeeping (>{COLLAPSE:.0f}% = trivial solution u==0):")
    print(f"    plain collapsed, R3 RESCUED it   : {len(rescue)}")
    print(f"    plain fine,      R3 RUINED it    : {len(ruin)}")
    print(f"    both collapsed (no comparison)   : {len(both)}")
    print(f"    both alive -> the real comparison: {len(live)}")
    for m, a, s, p, q in rescue:
        print(f"       RESCUE  {m:<12}{a:<14}s{s}   {p:8.2f}% -> {q:7.3f}%")
    for m, a, s, p, q in ruin:
        print(f"       RUIN    {m:<12}{a:<14}s{s}   {p:8.3f}% -> {q:7.2f}%")

    if not live:
        return
    P = np.array([p[3] for p in live]); Q = np.array([p[4] for p in live])
    better = int((Q < P).sum())
    lr = np.log(Q / P)
    w = stats.wilcoxon(P, Q)
    sign = stats.binomtest(better, len(live), 0.5)
    print(f"\n  on the {len(live)} live pairs:")
    print(f"    R3 better in {better}/{len(live)}   (sign test p = {sign.pvalue:.4f})")
    print(f"    median ratio R3/plain = {np.exp(np.median(lr)):.3f}x   "
          f"geometric mean = {np.exp(lr.mean()):.3f}x   (>1 = R3 worse)")
    print(f"    Wilcoxon signed-rank p = {w.pvalue:.4f}")
    print(f"\n    per architecture:")
    for a in sorted({p[1] for p in live}):
        g = [(p[3], p[4]) for p in live if p[1] == a]
        gp = np.array([x for x, _ in g]); gq = np.array([y for _, y in g])
        b = int((gq < gp).sum())
        print(f"      {a:<14} n={len(g):<3} R3 better {b}/{len(g)}   "
              f"geo-mean ratio {np.exp(np.log(gq/gp).mean()):.3f}x")
    print(f"\n    per material:")
    for m in ["homogeneous", "twolayer", "multilayer"]:
        g = [(p[3], p[4]) for p in live if p[0] == m]
        if not g:
            continue
        gp = np.array([x for x, _ in g]); gq = np.array([y for _, y in g])
        b = int((gq < gp).sum())
        print(f"      {m:<14} n={len(g):<3} R3 better {b}/{len(g)}   "
              f"geo-mean ratio {np.exp(np.log(gq/gp).mean()):.3f}x")


if __name__ == "__main__":
    oc, n7 = load_oneshot()
    analyse("ONE-SHOT R3 (rak's) -- pure L-BFGS, Phases 6+7", oc,
            f"; {n7} cells superseded by Phase 7's uncapped re-runs")
    analyse("PAPER-FAITHFUL R3 (Daw et al.) -- Adam(700)->L-BFGS, Phase 10",
            load_faithful())
