"""Every result in the project, in one place, grouped by scoring resolution.

The grouping matters more than it looks.  Phases 3-7 scored against an nx=512
finite-difference reference whose OWN error (0.11-0.18 %) equals the model errors
being measured, so differences there are compressed by a factor 0.19-0.27 and a
genuine 1.42x gap displayed as 1.018x.  Those numbers are kept for the record but
cannot be used to rank models that are close.  Only the nx=2048 block supports
comparisons.
"""
from __future__ import annotations
import glob, json, os, collections
import numpy as np

S = "/tmp/claude-1010/-home-prjgnn-PINN-FNO/a765de5d-6b07-41b3-9ed9-3624831a9432/scratchpad"
MATS = ["homogeneous", "twolayer", "multilayer"]


def fmt(v):
    if len(v) == 0:
        return ""
    if len(v) == 1:
        return f"{v[0]:.4f} (1)"
    return f"{np.mean(v):.4f}±{np.std(v, ddof=1):.4f} ({len(v)})"


def block(title, cells, models, note=""):
    print(f"\n{'='*100}\n{title}\n{note}{'='*100}")
    w = max(len(m) for m in models) + 2
    print(f"{'model':<{w}}" + "".join(f"{m:>22}" for m in MATS))
    for mo in models:
        row = f"{mo:<{w}}"
        for mk in MATS:
            row += f"{fmt(cells.get((mk, mo), [])):>22}"
        print(row)



def main():
    # ------------------------------------------------------------ nx = 2048
    # Key on (material, model, seed).  An earlier version appended values and then
    # deduped with set(), which counts a seed TWICE if it was ever re-scored -- it
    # reported n=16 for an 11-seed arm.  A dict keyed on the seed cannot do that.
    lb = collections.defaultdict(dict)   # pure L-BFGS      {(mat,model): {seed: val}}
    hy = collections.defaultdict(dict)   # Adam -> L-BFGS
    for f in glob.glob(S + "/*.json"):
        if "smoke" in f:
            continue
        try:
            rows = json.load(open(f))
        except Exception:
            continue
        if not isinstance(rows, list):
            continue
        for r in rows:
            if not isinstance(r, dict) or r.get("nx") != 2048:
                continue
            name = r["model"].replace("rung:", "")
            tgt = hy if r.get("optimizer") == "adam+lbfgs" else lb
            tgt[(r["material"], name)][r["seed"]] = r["rel_l2"]

    for d in (lb, hy):
        for k in d:
            d[k] = [d[k][s] for s in sorted(d[k])]

    mods_lb = sorted({m for _, m in lb})
    mods_hy = sorted({m for _, m in hy})
    block("A.  PURE L-BFGS  --  scored at nx=2048  (rel-L2 %, mean±sd (n))", lb, mods_lb,
          "     'R3' here = ONE-SHOT resampling (rak's), fires once on patience exhaustion\n")
    block("B.  ADAM(700) -> L-BFGS  --  scored at nx=2048  (rel-L2 %, mean±sd (n))", hy, mods_hy,
          "     'R3' here = PAPER-FAITHFUL (Daw et al.), resamples every Adam step\n")

    # ---------------------------------------------------------------- nx = 512
    leg = collections.defaultdict(dict)
    for f in ["p6_a.json", "p6_b.json", "p7_a.json", "p7_b.json"]:
        p = os.path.join(S, f)
        if not os.path.exists(p):
            continue
        for r in json.load(open(p)):
            leg[(r["material"], r["arch"] + ("+R3" if r["r3"] else ""))][r["seed"]] = r["rel_l2"]
    leg = {k: [v[s] for s in sorted(v)] for k, v in leg.items()}
    block("C.  PHASES 6-7  --  pure L-BFGS, scored at nx=512  (NOT comparable across close models)",
          leg, sorted({m for _, m in leg}),
          "     the reference's own error (0.11-0.18 %) equals the model errors here\n")

    # ---------------------------------------------------------------- phase 8
    lad = collections.defaultdict(dict)
    for f in glob.glob(S + "/p8_*.json"):
        for r in json.load(open(f)):
            if "rung" in r:
                lad[(r["material"], r["rung"])][r.get("seed", 0)] = r["rel_l2"]
    if lad:
        block("D.  PHASE 8 KAN LADDER  --  pure L-BFGS, nx=512  (15-variant strengthening sweep)",
              {k: [v[s] for s in sorted(v)] for k, v in lad.items()},
              sorted({m for _, m in lad}), "")



if __name__ == "__main__":
    main()
