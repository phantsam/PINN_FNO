# Every Bug, Every Retraction, Every Blind Spot

The complete defect record for the PINN-vs-KAN study on the 1-D elastic wave
equation, from Phase 3 to the present. Nothing is removed once entered, including
entries that were themselves later corrected.

**Why this document exists.** Roughly thirty distinct defects have been found in
this project. Six of them changed a headline result and five separate findings
dissolved when the seed count went up. A study that reports only its surviving
claims is not reproducible; a reader cannot tell which numbers were stress-tested
and which were measured once. Every claim in `ARCHITECTURES.md` and
`BASIS_VS_OPTIMISER.md` should be read against this list.

**Counting.** 18 defects in the inherited repository (12 ML, 6 numerical), 15 in our
own code, 7 upstream (4 pykan, 3 torch), 8 methodology errors, 16 retracted claims,
8 refuted predictions, 1 abandoned experiment line. **72 entries.**

**A note on this document's own first draft.** It was written, checked against
`ARCHITECTURES.md` §15, and found to be missing **ten** entries — eight retractions
and two analysis defects. That is a 38 % omission rate on a document whose entire
purpose is completeness, produced by someone who had just re-read the source
material. It is recorded here because it is the same failure mode as everything
below: a claim of coverage that was asserted rather than verified.

---

## Part 0 — Defects in the inherited repository

These are why `core/` exists. All were reproduced by execution and are recorded in
`AUDIT.md`; they are summarised here so the record sits in one place. **None are
ours** — but the study cannot be read without knowing that the codebase it replaced
could not answer the question it was built to answer.

### 0.1 The ML side — 12 verified defects

| # | defect | location | consequence |
|---|---|---|---|
| A1 | **The KAN loss solves the wrong PDE.** `physics_residual = u_tt - u_xx`; `E(x)` and `rho(x)` are never referenced. | `src/ansatz_losses.py:144`, `PIKAN.py:186` | correct only for homogeneous. Assumed-wave-speed error **22.5 %** on TwoLayer, **58.1 %** on MultiLayer. Blocks every heterogeneous KAN run — which `benchmark_v2.py` is configured to attempt. |
| A2 | **The KAN's reference solution ignores the material** (`c = 1.0` hardcoded) | `src/physics.py:148` | the KAN was scored against the wrong ground truth on any heterogeneous material |
| A3 | **Latent L-BFGS typo**, `line_search_fn="strong_wolf"` | `src/train.py:307` | the constructor accepts it, then raises on the first `step()`. Dead code today; fires the moment the modular path is wired up. |
| A4 | **Loss balancing is inverted** — `w = own_loss / total` gives the **larger** loss the **larger** weight | `src/train.py:271` | measured: PDE 1e-2, BC 1e-6 gives weighted BC **9.999e-11**. The KAN effectively trained with **no boundary condition**. |
| A5 | **The KAN was trained for 10 Adam steps** (`ITERATIONS = 10`) against the PINN's 300,000 | `PIKAN.py:480` | networks also tiny — (2,7) and (3,5) hidden units against PirateNet's 150k-200k parameters |
| A6 | **The fair-comparison driver has never executed.** Both callers omit required `x_fd`/`t_fd`/`u_fd`; `training_code.py` also references an undefined `PIKAN`. | `training_code*.py:703` | `TypeError` / `NameError`. **No head-to-head comparison had ever been run.** |
| A7 | **Fourier bandwidth uncontrolled between architectures** | — | the two families saw different effective input spectra, confounding bandwidth with architecture |
| A8 | **No KAN checkpoint exists anywhere in the repository** | — | no KAN result could be reproduced or re-scored |
| A9 | **Model selection on the evaluation set.** `evaluate()` runs against the FD reference every 10 steps and the best checkpoint is reloaded at the end. | `src/train.py:168-175` | the reported "Best Mean L2" is a **best-of-30000 statistic on the same data used to report accuracy** — not a held-out number |
| A10 | **Causal weighting mis-scaled by ~2.5 orders of magnitude** — `exp(-tolerance * cumsum)` with `tolerance = 0.1`, where the method needs `tolerance * chunk_loss ~ O(1)` | `src/ansatz_losses.py:63` | weights underflow to zero; only a sliver of the time domain trains |
| A11 | **`train_ic` cannot change the initial condition** | `src/train.py:228`, `PIKAN.py:333` | 1000 iterations per run spent on a pretraining step the hard ansatz then overrides |
| A12 | **No initial-condition ablation exists or can exist.** A repo-wide search for `no_ansatz`, `use_ansatz`, `soft_ic`, `ic_loss`, `ic_weight`, `lambda_ic`, `w_ic` returns nothing. | — | the "hard vs soft IC" experiment could not be run without new code. It was finally run in the current session. |

### 0.2 The numerical suite

| # | defect | consequence |
|---|---|---|
| N1 | **The pseudo-spectral solver is wrong in heterogeneous media.** It integrates `u_tt = c^2 u_xx`, dropping the `(c^2)' u_x` term that enforces traction continuity at an interface. | **over-transmits by 34 %** — the only method that gets the interface physics wrong. Aggravated by Gibbs oscillations from a global FFT across a discontinuous `c(x)`, and no sponge layer. Trustworthy for homogeneous only. |
| N2 | **`compare.py` performs no quantitative validation.** `"Analytical"` is commented out of `METHODS_TO_PLOT`, so the entire L2-error block is dead code. | the suite produced animations and **no error numbers**; the audit's table was the first quantitative check these solvers ever received |
| N3 | `solve_fvm` is Lax-Wendroff finite *difference*, not finite volume — a frozen-coefficient local flux Jacobian rather than a conservative interface flux | second-worst at the interface (+2.31 %) |
| N4 | `solve_fd` damps only velocity, not stress | sponge less effective than intended |
| N5 | DG has no absorbing layer | relies solely on a zero-jump outflow condition |
| N6 | Latent shape bug: `solve_fd`/`solve_ps`/`solve_fem` return `X_AXIS_VIS` (800 points) alongside snapshots of length `NX_BASE` (1200) | never fires today because `solve.py` passes `X_AXIS_FULL`, but any caller using the documented default gets a length mismatch |

**Why this matters for reading the study.** A1, A4, A5, A6, A8 and A9 together mean
the inherited repository had never run a valid PINN-vs-KAN comparison: the KAN
solved the wrong equation on two of three materials, trained without a boundary
condition, for 10 Adam steps, with no saved checkpoint, selected on the test set,
through a driver that raised `TypeError`. Every number in the current study comes
from `core/`, written to replace all of it.

---

## Part I — Defects in our own code

### I.1 Found before the current session

| # | defect | mechanism | measured effect |
|---|---|---|---|
| 1 | **D6 residual normalisation** | dividing the loss by `residual_scale² ≈ 5e4` shrank gradients by the same factor; torch's L-BFGS takes its first step as `min(1, 1/‖g‖₁)·lr`, so step one overshot catastrophically | MLP/twolayer **95.59 % → 0.86 %**; WavKAN **92.63 % → 0.88 %**. Invalidated all of Phase 5. |
| 2 | **FD reference first-order** | `dt = T/nt` with a first-order start-up step | convergence order 1.01 → **2.00** |
| 3 | **Causal-weight underflow** | fixed ε = 0.1 drove the causal weights to exactly zero; only 3 % of the time domain received gradient | replaced with Wang et al.'s annealed ε ∈ [1e-2, 1e2] |
| 4 | **R3 never fired** | scheduled at epoch 350 while runs early-stopped at ~237 | plain and R3 arms returned **bit-identical** results — which is how it was caught |
| 5 | **pykan seed never reached the model** | `MultKAN.__init__` calls `torch.manual_seed` with its *own* seed, overriding the trainer's | seed-0 and seed-2 parameters differed by **1.49e-08**; all three "seeds" were one seed, so every multi-seed KAN sweep before the fix varied only the collocation set |
| 6 | **Stale checkpoints** | evaluation loaded weights from a previous configuration | 20–80× wrong, mistaken for a real result. Produced the claim "rak is terrible on heterogeneous (66–128 %)"; retraining gave **0.85 / 3.71 %**. |
| 6b | **Linear probe targeted the wrong quantity** | probed `u` rather than what the network actually produces, `N = (u − g·decay)/growth`. Probing `u` credits every architecture equally for the hard-coded IC term. | reported **13 %** where the truth was **0.07 %** — a 185× error, in the direction of making every model look worse |
| 6c | **τ-centring omitted τ(0)** | the travel-time coordinate was built without subtracting its value at the origin | a nonsensical **124 %** d'Alembert residual on a solution that is exact to 0.0068 % |

### I.2 Found during the current session

| # | defect | mechanism | measured effect |
|---|---|---|---|
| 7 | **Soft-IC runs scored with the wrong ansatz** | `rescore.score` hardcoded `make_ansatz("legacy")`. A soft-IC network **is** u, so scoring re-wrapped it as `g·decay + growth·u`. The `g·decay` term does not depend on the model. | six runs from **two unrelated architectures** all returned 20.52–20.53 % — the tell that caught it. Re-scored from checkpoints: Fourier PINN 20.54 % → **0.2140 %**. Conclusions held; every number changed. |
| 8 | **Optimiser mis-attributed via a dict default** | an inventory script used `r.get("optimizer", "lbfgs")`; Phase 4's JSON has no such key and `phase4.py:50` calls `train()`, the **Adam** path | produced the claim "PIKAN scores 95.5 % under pure L-BFGS". The real pure-L-BFGS control gives **275–626 %**. The mechanism inferred from the error — that Adam marches a bad init too far while L-BFGS is protected — was **backwards**. |
| 9 | **Representation test under-converged, and its docstring lied** | the docstring claims "float64 for the fit"; the code is float32 throughout. The L-BFGS phase stopped on a relative criterion long before the achievable minimum. | physics-trained weights sit at supervised MSEs **29–159× lower** than the supervised fit ever reached. Its numbers were reported as *ceilings*; they are upper bounds on error. (This reproduces §13.4's finding that supervised fitting lands in poorer minima — already documented, and re-derived without noticing.) |
| 10 | **`update_grid_`'s docstring measures the wrong quantity** | it reports "3e-3 function drift", which is true and irrelevant: the objective is built from second derivatives | one call moves the *training objective* by **19,000×** (see III.2). Documenting the preserved quantity rather than the optimised one hid the defect for the whole of Phase 9. |
| 11 | **`adam_steps=0` crashed** | `loss` was unbound when the Adam loop body never executed | blocked the pure-L-BFGS control, which is the honest baseline for "what did Adam change?" |
| 12 | **Graph freed before `grad_norm_weights`** | `r3_adam.py` called it on tensors whose graph `loss.backward()` had already released | crashed at step 500 of the first run; reported as "running" before being checked |

### I.3 An architectural defect, not a coding one

| # | defect | mechanism | measured effect |
|---|---|---|---|
| 13 | **`_SplineKANLayer` initialisation** | B-splines placed directly on the σ_B = 10 Fourier embedding. The chain rule compounds `(2πk)² ≈ 3.5e4` from the embedding with `(1/knot spacing)² ≈ 25` from the spline. pykan damps its spline branch by `scale_sp = 1/√fan_in` (`KANLayer.py:112`); ours used plain Xavier. | `max|u_tt|` = **1.16e6** against every other model's ~2e2; PDE loss **3.5e10** and gradients **4.4e10** at step zero. Adding the same damping (`splinekan_fix`) drops the init loss **100,000×** and converts a divergence (235 %) into a collapse (97 %) — an improvement in kind, not in outcome. |

---

## Part II — Upstream defects

### II.1 pykan

| defect | mechanism | measured effect |
|---|---|---|
| **`curve2coef` returns NaN on CUDA** | `torch.linalg.lstsq`'s only CUDA driver is `gels`, which assumes full rank and **returns NaN without raising**. KAN activations contract with depth, so layers 1–3 are rank-deficient (7/8, 4/8, 4/8). | every grid refinement past layer 0 produced NaN — the KAN paper's own accuracy mechanism was unusable on GPU |
| **`curve2coef` unstable at k = 5** | order-5 B-splines overlap more heavily; `gels` fails even at nominal full rank | LSQ residual **48.22** vs the ridge solve's **0.00197** — a 24,500× error |
| **Device bookkeeping** | `MultKAN`/`KANLayer` cache `.device` and define their own `.to()`, which `nn.Module.to()` never calls | `refine()` built grids on the CPU → device mismatch |
| **Phantom parameters** | the symbolic branch is registered `requires_grad` even under `symbolic_enabled=False` | reported 12,040 parameters against 8,600 real — 28.6 % inflation, and every L-BFGS history vector padded with permanent zeros |

Both `curve2coef` defects are fixed by a ridge-regularised normal-equation solve —
the fallback pykan itself left commented out — verified identical to pykan wherever
pykan is valid (residuals matching to 6 decimals at k = 3).

### II.2 torch

| defect | mechanism | measured effect |
|---|---|---|
| **L-BFGS fp32 overflow** | `_cubic_interpolate` forms `d1² ≈ 2.7e49`, overflowing float32 (max 3.4e38) → `inf/inf` → NaN step | WavKAN parameters NaN-ed while the *reported* loss read a healthy 418.69 |
| **L-BFGS returns the pre-step loss** | a NaN born inside the line search is invisible in the return value | the divergence guard had to check **parameters**, not the loss |
| **`tolerance_grad` is absolute** | default 1e-7 compared against a supervised loss living at ~4e-7 | L-BFGS returned at its first inner iteration; the loss froze bit-exactly for 200+ steps |

---

## Part III — Methodology errors

### III.1 The reference solution was inadequate, twice

| stage | reference own error | best model | margin | consequence |
|---|---|---|---|---|
| Phases 3–7 | nx=512 → **0.1122 %** | ~0.11 % | **~1×** | all differences compressed ~4×; a true 1.424× gap displayed as 1.018× |
| Phases 8–10 | nx=2048 → **0.0066 %** | 0.0236 % | **3.6×** | chosen when models were at 0.11 % (a 20× margin). Models improved 5× and the margin eroded silently. |
| now | **exact d'Alembert** | 0.0222 % | ∞ | closed form; zero discretisation error |

The second erosion is the more instructive failure: nothing broke, no test caught
it, and the reference was adequate *when chosen*. It decayed because the models got
better. **A resolution choice is only valid relative to the errors being measured,
and must be re-checked whenever those errors drop.** Correcting it moved the
headline from 1.69× to 1.81× — the compression had been *understating* the result.

A related, separate defect: snapshot times were selected by nearest neighbour
across grids with different `dt`, which made a second-order solver look first-order
(1.09/1.04/0.74). Linear interpolation in `t` restored 2.02/2.07/2.30.

### III.2 We ran pykan's own recommended procedure and it destroyed every run

`update_grid` is not optional — the KAN paper's `fit()` calls it 10 times by
default over the first half of training, and grid refinement is the mechanism KANs
use to gain accuracy. It collapsed our runs **8/8**, 0.109 % → 94.9–95.4 %.

One call on a trained, working model:

```
             rel-L2      training objective
before       0.0114%       1.930e-05
after        0.2272%       3.683e-01      <- 19,000x worse
```

What it preserves, measured on 4,000 collocation points:

| quantity | ‖before‖ | relative change |
|---|---|---|
| u | 1.508e+01 | 0.10 % |
| u_x | 1.863e+02 | 0.59 % |
| u_xx | 2.942e+03 | **2.28 %** |
| u_tt | 2.942e+03 | **2.29 %** |

`curve2coef` refits coefficients to match **function values**. Nothing constrains
the derivatives, and the PDE residual is built entirely from `u_xx` and `u_tt`.

Why 2.3 % is fatal: at a converged solution the residual is a near-perfect
cancellation. `‖u_tt‖ = ‖u_xx‖ = 2942` but `‖residual‖ = 0.44` — they cancel to
**1 part in 6,696**, so the solution needs its second derivatives correct to
1.5e-4 relative. `update_grid` perturbs them at 2.3e-2, **153× coarser**.
Predicted blow-up `(√2 · 0.0229 · 2942)² / N / L_before` = 47,000×; measured
19,000× — agreement within 2.5×, on the correct side, since the two perturbations
are not fully independent.

**This generalises.** Any physics-informed KAN built on pykan's default `fit()`
hits this. It is not a bug in pykan: grid adaptation is correct for regression,
where only function values matter. It is silently wrong for any loss built from
high-order derivatives.

### III.3 Parameter budgets were never matched

The unified PINN/KAN benchmark (arXiv:2602.15068) configures every architecture to
"approximately the same number of trainable parameters to control for model
capacity". Ours were not:

| model | params |
|---|---|
| Fourier PINN (headline) | 82,689 |
| tuned KAN `charcoords50` | 49,020 |

The KAN was winning with **40 % fewer parameters**, so the gap favoured us — but
the control was still missing. Adding `fourier_matched` (49,729 params, +1.45 %)
weakens the result from **1.81× to 1.50×**, and the tests then disagree:
Welch p = 0.092, Mann-Whitney p = 0.049, paired Wilcoxon p = 0.0068.

A second finding fell out of it: the *smaller* PINN is **better** (0.0332 % vs
0.0401 %). The 128-wide default was over-parameterised, so the headline arm had
been handicapped by its own capacity.

### III.4 A circular statistical test

Phase 8 selected the three lowest KAN values and then tested whether they were
low. p = 0.0000 and meaningless. Retracted. Every comparison since names its arms
before the data is examined.

### III.5 Two named algorithms conflated under one label

Phases 6–9 report an arm called "R3" and attribute it to Daw et al. (ICML 2023).
That implementation **fires once**, on patience exhaustion. The published algorithm
resamples **every iteration** — verified against the official release
`arkadaw9/r3_sampling_icml2023`, where `sampler.update()` sits inside `loss()`.
The same repository contains **no L-BFGS anywhere**; `grep -rn "LBFGS"` returns
nothing. The mislabel appeared in `ARCHITECTURES.md`, in commit `e0736bb`, and in
every summary built on them.

### III.6 The headline explanatory number measures the wrong quantity

`BASIS_VS_OPTIMISER.md` rests on "a separable spline basis fits this solution
**233× better** than isotropic random Fourier features at matched dimension". That
is a **value-space** fit, and this project's own measurements show value-fit
quality is close to irrelevant for PDE solving. At matched value-fit quality
(~0.04 %):

| model | ‖u_tt‖ | ‖u_xx‖ | ‖u_tt − u_xx‖ | cancels to |
|---|---|---|---|---|
| `fourier16` | 3.384e+04 | 4.568e+03 | 3.247e+04 | **1 : 1** |
| `charcoords50` | 2.984e+03 | 2.947e+03 | 4.269e+02 | 7 : 1 |

`fourier16`'s second derivatives are not even the same *magnitude* — off by 7.4× —
and cancel not at all. The correct statement of the KAN's advantage is that
characteristic coordinates make the cancellation **structural**, not that they fit
`u` better.

### III.7 Declaring completion before auditing

Sections were titled "the final story" and a report was published as a definitive
record, after which further checking found the reference-floor erosion, the
under-converged representation test, and the loss-landscape result that reversed
the collapse explanation. The audit belongs **before** the claim of finality.

---

## Part IV — Retracted claims

| claim | status | what was actually true |
|---|---|---|
| "The KAN beats the PINN by 1.42×" | **retracted** | three seeds. At n = 11, p = 0.577 — a tie. The KAN does win, but only under the hybrid optimiser, which is a different claim on different data. |
| "The PINN wins on two-layer by 2.20×" | **retracted** | five seeds, p = 0.056. At n = 11: 1.39×, p = 0.250. This also removes the "interface sharpness" mechanism built to explain it. |
| "PIKAN scored 95.5 % under pure L-BFGS" | **wrong** | that was Adam (defect #8). Pure L-BFGS gives 275–626 %. |
| "The Fourier embedding kills KANs" (as a statement about expressiveness) | **reframed** | the observation is right, the explanation was wrong. `fourier16` fits the solution to **0.0431 %**, better than the Fourier PINN. It is a loss-landscape failure. |
| "R3 helps on two-layer" | **not significant** | 10/14 paired, sign test p = 0.18; at n = 11, p = 0.23. The quoted 0.3944 % was the best single cell out of 45. |
| "MLP cannot represent a wavefront crossing an interface (spectral bias)" | **wrong** | it was defect #1. The MLP is in fact the best model on two-layer. |
| "KAN variance is only 1.2× the PINN's" | **wrong** | compared a settled n = 8 PINN against a 3-seed KAN. At matched n = 11 it is 5.44× (p = 0.031). The original claim was right and the retraction was premature — a retraction can be an error too. |
| "The poly ansatz has no trivial basin, so prefer it" | **wrong** | design intent, never measured. It loses 6/6 by two orders of magnitude (fourier 5.15 % vs 36.56 %; wavkan 10.18 % vs 131.83 %). |
| "KAN's L-BFGS never ran" | **wrong** | the typo was in dead code; the arm had been running all along |
| "rak is terrible on heterogeneous (66–128 %)" | **wrong** | stale checkpoints (defect #6). Retraining gave 0.85 / 3.71 %. |
| "The KAN arm is void" | **overstated** | only the hand-rolled spline arm was broken; WavKAN worked |
| "The epoch cap costs pykan 20–60 %" | **wrong** | those cells were *also* reseeded, so cap and reseed were confounded and the effect could not be attributed |
| "MLP's supervised fit is limited by spectral bias" | **wrong** | it had simply hit the epoch cap. Spectral bias was invoked twice in this project as an explanation and was wrong both times — see Part IV.b. |
| "k = 5 collapses" | **wrong** | 0.1590 % on homogeneous. It is material-dependent, not a property of the spline order. |
| "Characteristic coordinates didn't pay off" | **wrong** | `charcoords50` reached 0.1089 % and became the best KAN in the study |
| "tanh beats silu 1.9× on multilayer" | **wrong** | evaporated at 3 seeds |

**The recurring failure mode is generalising from a single seed or a single
material.** Five findings dissolved under more seeds. Nothing at n < 11 in this
project should be quoted without its n.

### IV.b — Predictions asserted before measurement, and refuted by it

Distinct from the retractions above: these were forward predictions, offered with
confidence and then contradicted. They are recorded because a wrong prediction that
is never logged looks, in hindsight, like a question nobody asked.

| prediction | outcome |
|---|---|
| "`charcoords50` will fail on multilayer" | it tied the best PINN (0.0403 % vs 0.0432 %) |
| "k = 5 collapses" | 0.1590 % on homogeneous |
| "Characteristic coordinates didn't pay off" | they produced the best KAN in the study |
| "Spectral bias explains the MLP's interface failure" | it was the D6 normalisation bug |
| "Spectral bias limits the MLP's supervised fit" | it was the epoch cap |
| "Sharpness, not interface count, explains two-layer" | the two-layer result it explained evaporated at n = 11 |
| "Adam marches PIKAN's bad init too far; L-BFGS is protected" | backwards — pure L-BFGS is the *worst* of the three (defect #8) |
| "The Fourier embedding kills KANs because they cannot express the solution" | they express it to 0.0431 %, better than the PINN |

Eight predictions, eight refuted. The pattern is that a mechanism was proposed to
explain a number *before* the mechanism itself was measured. Every explanatory
claim in this project should be checked for whether it was measured or inferred.

---

## Part V — The abandoned experiment line

The capacity probe was attempted four times and abandoned:

| attempt | flaw | symptom |
|---|---|---|
| 1 | `tolerance_grad` absolute against a differently-scaled loss | never moved; results 2× pessimistic |
| 2 | float32 precision floor | gradient 3.9e-07 = rounding noise; stalled at 0.234 % |
| 3 | fit set ≠ eval set | optimised a different objective than the metric; not a bound |
| 4 | fp64 + Adam warm-up + 20k epochs, fit == eval | still `CAP-NOT-CONVERGED` |

**Why it cannot work as designed.** `E_sup ≤ E_pde` holds by definition only if the
supervised search finds the **global** minimum. It does not: pointwise MSE on an
oscillatory target is a worse-conditioned landscape than the PDE residual, so
cold-start fitting lands in poorer minima. The experiment can prove "optimisation
headroom exists" but can **never** prove "representation-limited", which was the
direction needed.

Defect #9 is this same failure, re-derived in the current session without
recognising it as already documented. **A defect record only helps if it is read
before repeating the experiment.**

---

## Part VI — What this record implies for the results

Claims are graded by how hard they have been attacked, not by their p-values.

| claim | confidence | basis |
|---|---|---|
| KAN never worse than the PINN on homogeneous | **very strong** | 9/11 paired, every parameter budget, every reference |
| R3 fails in all four regimes | **very strong** | ~100 paired cells, mechanism measured (retention never below 50 %) |
| `update_grid` is incompatible with physics-informed training | **strong** | quantitative prediction matched measurement within 2.5× |
| The collapse is loss-driven, not representational | **strong** | measured three independent ways |
| The cancellation mechanism | **strong** | directly measured, explains three separate failures |
| KAN beats the *default* 128-wide PINN, 1.81× | **strong** | three concordant tests, exact reference |
| KAN beats a *matched-parameter* PINN, 1.50× | **moderate** | paired significant (p = 0.0068), unpaired not (p = 0.092) |
| Two-layer and multi-layer are ties | **moderate** | n = 11, but wide variance |
| The 233× basis advantage explains the win | **superseded** | measures value-space fit; see III.6 |

**The base rate to keep in mind:** roughly six defects in our own analysis code
were found in a single session, several only because someone asked for a deliberate
hunt. Results measured once should be assumed to carry defects at a similar rate.
The claims above marked "strong" have survived repeated adversarial checking; the
rest have not yet been tested that way.
