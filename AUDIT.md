# 1D Elastic Wave — Repository Audit & PINN vs KAN Assessment

**Date:** 2026-08-28  **Repo:** `PINN_FNO` @ `98ceb1f` (working tree clean, nothing modified)
**Scope:** verify the numerical-methods suite; inventory and assess every PINN / KAN result;
determine whether the repo can currently answer *"PINN or KAN — which is better for the 1D
elastic wave equation?"*

Every claim below was executed and measured, not inferred from reading code. Reproduction
scripts are listed in §8.

---

## 0. Environment

```
conda activate pinn-kan
```

Python 3.11 · torch 2.11.0+cu128 · CUDA available · 2 × NVIDIA H100 NVL (95 GB each)
Also installed: numpy, scipy, matplotlib, tqdm, pyyaml, pandas, seaborn, imageio, pykan, scikit-learn.

`pykan` additionally requires `scikit-learn` — this is **not** in either `requirements.txt`, so
`from kan import KAN` fails on a clean install of the repo's stated dependencies. Both
`ML/src/models.py` and `ML/src/loader.py` swallow that failure in a `try/except` and set
`KAN = None`, which silently disables every pykan-based architecture instead of erroring.

---

## 1. The single most important structural finding

**The two halves of the repository solve different physics problems, in different units, with
different boundary conditions, and are not comparable.** `Numerical methods/` is *not* a
reference or benchmark for `ML/`; they were never coupled.

| | `Numerical methods/` | `ML/` |
|---|---|---|
| Equation | forced: `ρ∂ₜv = ∂ₓσ + f(t)δ(x−x₀)` | free: `ρ∂ₜₜu = ∂ₓ(E∂ₓu)` |
| Driver | Ricker wavelet source, `f₀ = 5 Hz` | initial condition only (no source) |
| Unknown | particle **velocity** `v` | **displacement** `u` |
| Units | physical (m, s, kg) | non-dimensional |
| Domain | `x ∈ [0, 15000] m` | `x ∈ [−1, 1]` |
| Velocities | 3000 / 4000 m s⁻¹ | `Vp` ∈ [1.0, 1.58] non-dim |
| Duration | `T = 2.4 s` | `T = 1.0 s` physical → `T_nd ≈ 0.89` |
| Boundaries | cubic sponge damping layers | Mur absorbing / ABC residual loss |
| Reference | ray-traced analytic Green's function | in-house FD solver |

Consequence: the six-method numerical suite validates nothing about the neural results, and the
neural `fd_reference` is itself unvalidated against the six-method suite. Bridging them is a
prerequisite for any publishable claim.

---

## 2. Numerical methods — verification

### 2.1 Inventory

Six solvers exist in `Numerical methods/solvers.py`, matching your expectation:

| Method | Function | Scheme |
|---|---|---|
| Finite Difference | `solve_fd` | staggered velocity–stress, 2nd order, leapfrog |
| Pseudo-Spectral | `solve_ps` | FFT in x, 2nd-order in t, displacement form |
| Finite Element | `solve_fem` | linear P1 elements, lumped mass, explicit |
| Finite Volume | `solve_fvm` | Lax–Wendroff on the `(σ, v)` system |
| Discontinuous Galerkin | `solve_dg` | nodal LGL, `N=4`, exact Riemann flux, RK4 |
| Spectral Element | `solve_sem` | LGL `N=4`, diagonal mass, Newmark + sponge |

Supporting checks that **passed**:

- LGL nodes/weights for both DG and SEM converge to the exact values
  (`[−1, −√(3/7), 0, √(3/7), 1]`; weights `[0.1, 49/90, 32/45, 49/90, 0.1]`, sum = 2).
- Element/global assembly in SEM (`K_global[idx, idx] += …` with `idx` a `slice`) is correct
  block assembly, not the broadcasting bug it superficially resembles.
- The velocity transmission coefficient used by the analytic benchmark,
  `T_v = 2Z₁/(Z₁+Z₂)`, and the free-field amplitude `A = F/(2ρc)` are both correct.

### 2.2 Test A — free-field amplitude and waveform (homogeneous)

Left-going pulse measured 1000–2500 m from the source, clear of the source node, the interface
and the sponge. Theory: peak `|v| = 10⁶/(2·2500·3000) = 0.066667`.

| Method | peak \|v\| | vs theory | waveform rel-L2 |
|---|---|---|---|
| DG  | 0.066574 | **−0.14 %** | 5.18 % |
| FEM | 0.066526 | −0.21 % | 8.04 % |
| FD  | 0.066521 | −0.22 % | 5.68 % |
| SEM | 0.066226 | −0.66 % | 5.34 % |
| FVM | 0.064655 | −3.02 % | 14.17 % |
| PS  | 0.064427 | −3.36 % | 7.75 % |

**All six are correct in a homogeneous medium.** The residual 5–14 % waveform L2 is expected
discretisation error: the analytic reference is a true point source, whereas the solvers inject
over one grid cell. FVM is the weakest, consistent with Lax–Wendroff's known dissipation.

### 2.3 Test B — interface transmission (heterogeneous) — **the discriminating test**

Interface at `x = 9000` with `c = 3000 | 4000`, source at `x = 7500` in the slow medium.
Measured transmission = (transmitted peak) / (that solver's own free-field peak), so each method
is judged against itself and the amplitude error of Test A cancels. Theory: `T_v = 0.857143`.

| Method | `T_v` measured | error | rel-L2 in transmitted zone |
|---|---|---|---|
| DG  | 0.8576 | **+0.06 %** | 5.09 % |
| FEM | 0.8591 | +0.22 % | 8.06 % |
| SEM | 0.8591 | +0.23 % | 5.79 % |
| FD  | 0.8593 | +0.25 % | 10.56 % |
| FVM | 0.8770 | +2.31 % | 7.05 % |
| **PS** | **1.1466** | **+33.78 %** | **28.44 %** |

### 2.4 The pseudo-spectral solver is wrong in heterogeneous media

`solvers.py:44` integrates

```python
u_ps_np1 = 2*u_ps_n - u_ps_nm1 + (dt**2) * (c_ps**2 * uxx + force / rho_ps)
```

i.e. `u_tt = c(x)² u_xx`. The correct heterogeneous elastic wave equation is

```
ρ u_tt = ∂ₓ(μ ∂ₓu)   ⟹   u_tt = c² u_xx + (c²)′ uₓ      (for constant ρ)
```

The `(c²)′ uₓ` term is dropped. That term is exactly what enforces traction continuity at an
interface, so PS over-transmits by 34 % — it is the only method that gets the physics of the
interface wrong. Two aggravating factors: a global FFT across a discontinuous `c(x)` produces
Gibbs oscillations, and PS is the only solver with no sponge layer.

**PS is trustworthy for homogeneous runs only.** Fix: switch to the conservative two-step form
`u_tt = (1/ρ)·∂ₓ(μ·∂ₓu)`, applying the spectral derivative twice with `μ` in between.

### 2.5 Other numerical-side issues

1. **`compare.py` performs no quantitative validation.** `"Analytical"` is commented out of
   `METHODS_TO_PLOT` (line 12), so the entire L2-error block at lines 112–150 is dead. The
   suite currently produces animations and no error numbers — the table in §2.3 above is the
   first quantitative check these solvers have had.
2. **`solve_fvm` is Lax–Wendroff finite *difference*, not finite volume.** It uses a
   frozen-coefficient local flux Jacobian per cell rather than a conservative interface flux,
   which is why it is the second-worst at the interface (+2.31 %).
3. **`solve_fd` damps only velocity**, not stress (`v_fd *= exp(-damp*dt)`), making the sponge
   less effective than intended.
4. **DG has no absorbing layer** and relies solely on a zero-jump outflow condition.
5. **Latent shape bug.** `solve_fd` / `solve_ps` / `solve_fem` return `X_AXIS_VIS` alongside
   snapshots of length `NX_BASE`. `physics.X_AXIS_VIS` has 800 points, `NX_BASE` is 1200.
   `solve.py` happens to pass `X_AXIS_FULL` for that argument so it never fires today, but any
   caller using the documented default gets a length mismatch.
6. **Cost.** Wall time for one 2.4 s run at `nx = 1200`: FD 0.03 s, PS 0.1 s, SEM 0.1 s,
   FVM 10.2 s, FEM 10.9 s, DG 41.7 s. FEM and FVM are slow only because of Python-level loops
   over cells; both are trivially vectorisable.

### 2.6 Verdict on the numerical suite

**Five of six methods are verified correct in both homogeneous and heterogeneous media.** SEM
and DG are the most accurate overall; FD is the best accuracy-per-second by a wide margin and is
the right choice for generating ML reference data. **PS must not be used for heterogeneous
cases until §2.4 is fixed.** FVM is usable but is the least accurate and is misnamed.

---

## 3. ML side — what is actually implemented

### 3.1 Initial conditions and stress (direct answer to your question)

The ML code uses a **hard-constraint ansatz** (`src/ansatz_losses.py:18`), not an IC loss term:

```
u(x,t) = g(x)·decay(t) + growth(t)·NN(x,t)
decay  = exp(−½(15·t_phys)²)      growth = tanh(25·t_phys)²
```

where `g(x)` is the normalised Gaussian derivative. Measured with deliberately large random
network weights:

```
max |u(x,0) − g(x)| = 0.000e+00     (displacement IC)
max |u_t(x,0)|      = 0.000e+00     (velocity IC)
```

**Both initial conditions hold to machine precision for any weights.** So the answer to
"does the IC help the loss go down?" is: *there is no IC term in the loss at all.* The loss has
exactly two terms — PDE residual and absorbing-BC residual. The IC is enforced structurally, so
it can never be traded off against the physics loss and can never fail to converge. This is a
genuinely good design decision and is the strongest part of the ML codebase.

Two caveats worth knowing:

- The blend is **not a partition of unity**: at `t_nd = 0.05` (Homogeneous) `decay = 0.704` and
  `growth = 0.783`, summing to 1.49. The network must actively cancel the excess IC copy during
  the handover window (`t_nd ≈ 0.02–0.10`). Harmless but avoidable — `growth = 1 − decay` would
  be cleaner.
- The constants `15` and `25` are hard-coded and tied to neither `sigma_g` nor the wave speed.

**Stress.** The PINN residual (`src/ansatz_losses.py:31`) is built in conservative form:

```python
sigma   = material.E(x) * u_x
sigma_x = grad(sigma, x)
return u_tt - sigma_x / material.rho(x)
```

This differentiates the *stress*, not the displacement twice, which is the correct treatment for
a discontinuous `E(x)` because it keeps traction continuous across an interface. It is the right
formulation — and, as §4.1 shows, it is exactly what the KAN path discards.

### 3.2 Architectures present

`src/models.py` — `VanillaPINN`, `FourierFeaturePINN`, `PirateNet` (RWF + physics-informed
init), `PIKAN` (B-spline KAN on Fourier features), `FourierKAN` (pykan wrapper), `WavKAN`
(Mexican-hat / Morlet / DoG / Meyer / Shannon wavelet KAN).

### 3.3 Two parallel, incompatible training stacks

| | PINN stack | KAN stack |
|---|---|---|
| Entry point | `training_code.py`, `src/benchmark.py` | `PIKAN.py`, `benchmark_v2.py` |
| Loss | `physics_loss` → `compute_pde_residual` | `losses()` |
| PDE residual | `u_tt − ∂ₓ(E uₓ)/ρ` ✅ | `u_tt − u_xx` ❌ |
| Reference | `fd_reference` (material-aware) ✅ | `solve_wave_fd`/`u_sol` (c ≡ 1) ❌ |
| Loss balancing | `compute_grad_norm_weights` ✅ | `train_adam` inline rule ❌ |
| Causal weighting | yes | no |
| Adam steps | thousands | **10** |
| L-BFGS | works | **crashes** |

`training_code_pikan.py` is byte-identical to `training_code.py` plus `KANLayer`/`PIKAN`
appended at the end — it is the one file that *intends* a fair comparison. It does not run (§4.6).

---

## 4. Why the repo cannot currently answer "PINN or KAN?" — 12 verified defects

Each was reproduced by execution.

### 4.1 The KAN loss solves the wrong PDE — `src/ansatz_losses.py:144`, `PIKAN.py:186`

```python
physics_residual = u_tt - u_xx      # c ≡ 1, E(x) and ρ(x) never referenced
```

Verified: `compute_pde_residual` (PINN) references `material.E` → `True`;
`losses` (KAN) references `material.E` → `False`.

This is only correct for `HomogeneousModel`. Measured non-dimensional wave speeds:

| Material | `Vp` range | max error in assumed speed |
|---|---|---|
| Homogeneous | 1.000 | 0 % |
| TwoLayer | 1.000 → 1.225 | 22.5 % |
| MultiLayer | 1.000 → 1.581 | 58.1 % |

**Blast radius (corrected 2026-08-29).** `PIKAN.py` sets `my_material = HomogeneousModel()` and
only ever runs the homogeneous case, where `u_tt - u_xx` *is* the correct equation. So this defect
did **not** taint the numbers `PIKAN.py` actually produced. It blocks any heterogeneous KAN run —
and `benchmark_v2.py` is configured to attempt exactly that
(`EXPERIMENTS = {"exp2_twolayer": ...}`, `RUN_ARCHS = ["pikan"]`).

### 4.2 The KAN's reference solution ignores the material — `src/physics.py:148`

```python
c = 1.0  # Wave speed in non-dim units
```

Verified: `solve_wave_fd` returns bit-identical output for Homogeneous, TwoLayer and MultiLayer
at fixed `sigma_g`. Against the true heterogeneous solution from `fd_reference`:

```
rel-L2 between the c=1 reference and the true TwoLayer solution: 70.8 %
```

**Blast radius (corrected 2026-08-29).** As with §4.1: `c = 1` is *correct* for the homogeneous
model, which is the only one `PIKAN.py` runs, so the numbers it produced were measured against a
valid reference. The defect blocks heterogeneous KAN evaluation, where the reference would be
70.8 % wrong. The PINN path uses `fd_reference`, which *does* honour `E(x)` (verified:
Homogeneous vs TwoLayer differ, max |Δ| = 0.899).

### 4.3 A latent L-BFGS bug in dead code — `src/train.py:307`

> **Correction (2026-08-29).** An earlier revision of this document claimed the KAN's L-BFGS
> phase never ran. That was wrong, and is corrected here.

The typo exists in exactly one place:

```
src/train.py:307:        line_search_fn="strong_wolf"      <- broken
PIKAN.py:305:            line_search_fn="strong_wolfe"     <- correct
```

Verified against torch 2.11: the constructor accepts the misspelling, then raises
`RuntimeError: only 'strong_wolfe' is supported` on the first `optimizer.step(closure)`.

**But nothing calls the broken one.** `src/train.py:train_lbfgs` is dead code; the only caller of
any `train_lbfgs` in the repo is `PIKAN.py:519`, which uses `PIKAN.py`'s own correctly-spelled
copy. Running `PIKAN.py` unmodified, L-BFGS executes 500 iterations and does the bulk of the
training:

```
Switching to L-BFGS fine-tuning...
LBFGS ITERATION:   0 | LOSS: 1.8361e+03 | PDE: 1.8366e+03 | BC: 5.0588e-01
LBFGS ITERATION: 450 | LOSS: 2.1836e-01 | PDE: 2.1841e-01 | BC: 3.7783e-02
```

So this is a real bug that will fire the moment anyone wires up the modular training path, but it
did **not** affect any result the repo has produced. It also means the KAN *did* see the full time
domain, since `use_causal` is off during L-BFGS.

### 4.4 The KAN's loss balancing is inverted — `src/train.py:271`

```python
physics_loss_weight = p_val / total
bc_loss_weight      = b_val / total
```

This gives the **larger** loss the **larger** weight — the opposite of balancing. Measured:

| PDE loss | BC loss | w_pde | w_bc | weighted PDE | weighted BC |
|---|---|---|---|---|---|
| 1e−2 | 1e−6 | 0.9999 | 0.0001 | 9.999e−03 | **9.999e−11** |

Once one term dominates, the other is annihilated — here the absorbing boundary condition
contributes ~1e−10 and the KAN effectively trains with no boundary condition. The PINN path
instead uses `compute_grad_norm_weights`, a correct inverse-gradient-norm scheme.

### 4.5 The KAN was trained for 10 Adam steps — `PIKAN.py:480`

```python
ITERATIONS = 10
```

1000 IC-pretraining iterations, then **10** Adam steps, then an L-BFGS phase that crashes
(§4.3). The PINN default is `n_steps = 300000`. Network sizes are also tiny — `(n_hidden,
hidden_width)` ∈ {(2,7), (3,5)} — against PirateNet's 150k–200k parameters.

### 4.6 The fair-comparison driver raises `TypeError` — `training_code*.py:703`

`train_model_two_phase` requires `(model, material, x_fd, t_fd, u_fd)`. Both files call it with
2 positional arguments and no `x_fd`/`t_fd`/`u_fd` keyword. Verified by AST inspection:

```
training_code.py:        MISSING required args: ['x_fd', 't_fd', 'u_fd']  => TypeError
training_code_pikan.py:  MISSING required args: ['x_fd', 't_fd', 'u_fd']  => TypeError
```

`training_code.py` additionally references `PIKAN`, which is never defined or imported in that
file → `NameError`. **Neither head-to-head driver has ever executed.**

### 4.7 Fourier bandwidth is not controlled between the architectures

| Model | embedding | effective bandwidth |
|---|---|---|
| `PirateNet` / `FourierFeaturePINN` | `xt @ B.T`, `σ = 3.0` | ≈ 3 rad/unit |
| `PIKAN` | `2π·(xt @ B.T)`, `σ = 7.0` | ≈ 44 rad/unit |

`PIKAN` carries an extra `2π` factor *and* a larger `σ`, giving ~15× the input frequency
content. Any observed PINN-vs-KAN difference is confounded with this. §6 controls for it.

### 4.8 No KAN checkpoint exists anywhere in the repository

`ML/Models/` contains 8 files, all PINNs — `vanilla`, `fourier`, `pirate` for exp1/exp2/exp3
(`exp2_twolayer_pirate` is missing). **Zero** `pikan`, `WaveKAN` or `piann` checkpoints.

`ML/model/` holds a pykan checkpoint whose `history.txt` reads in full:

```
### Round 0 ###
init => 0.0
```

That is an **untrained network auto-saved at initialisation**. Meanwhile `src/benchmark.py`
requests `RUN_ARCHS = ["WaveKAN_3-10", "WaveKAN_5-10", "WaveKAN_7-10"]` and `benchmark_v2.py`
requests `["pikan"]` — none of which can be loaded. Both scripts print
`"! No models found … skipping"` and exit.

The `results/exp1/` figures **do** show trained WaveKAN curves (§5.2), so those models existed
at some point and their weights were lost.

### 4.9 Model selection on the evaluation set

`src/train.py:168–175` calls `evaluate(...)` against the FD reference **every 10 steps** and
saves whichever checkpoint minimises that error, then reloads it at the end. The reported
"Best Mean L2" is therefore a best-of-30000 statistic on the same data used to report accuracy,
not a held-out result. It also dominates runtime. Additionally the evaluation horizon is
inconsistent: `src/benchmark.py` uses `T_MAX_DICT["exp1_homogeneous"] = 1.0` while
`benchmark_v2.py` uses `0.75` for the same experiment.

### 4.10 The causal weighting is mis-scaled by ~2.5 orders of magnitude — `src/ansatz_losses.py:63`

```python
weights = torch.exp(-tolerance * cumsum)      # tolerance = 0.1
```

This is the Wang–Sankaran–Perdikaris causal weight, which requires `tolerance * chunk_loss ~ O(1)`
to act as a soft time-marching schedule. Measured on a freshly initialised PirateNet
(8192 collocation points, 32 chunks):

```
PDE residual RMS      = 21.0
chunk losses (0..5)   = 3126, 943, 2098, 3673, 2022, 476
causal weights (0..7) = 1.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, ... (exactly zero in float32)
```

`exp(-0.1 x 3126) = e^-312`, which underflows to zero. **Only chunk 0 — the first 3 % of the
time domain — ever receives gradient.** Chunks 1–31 are not merely down-weighted, they are
identically zero. Chunk 1 can only switch on once chunk 0's loss falls below ~10, a 300x
reduction.

Tolerance sweep at initialisation (fraction of the horizon receiving gradient):

| tolerance | 1e-1 (repo) | 1e-2 | 1e-3 | 3e-4 | 1e-4 |
|---|---|---|---|---|---|
| % of horizon active | **3 %** | 3 % | 12 % | 100 % | 100 % |

A balanced value is `tolerance ~ 1e-3` (from `1 / mean chunk loss = 2.4e-3`). The repo's `0.1`
is 40–400x too large.

**This is the direct cause of the 300 000-step training budget**, and it explains why the L-BFGS
phase matters so much: `train_model_two_phase` sets `use_causal=False` for L-BFGS, so L-BFGS is
the *only* stage that ever sees the full time domain. Which in turn compounds §4.3 — the KAN's
L-BFGS crashes, so the KAN never trains on late times at all, by either route.

**Measured cost of this one constant.** Identical PirateNet, identical seed, identical
everything except `tolerance`:

| tolerance | step 1000 | step 2000 | step 3000 | wall |
|---|---|---|---|---|
| `0.1` (repo) | 108.81 % | 110.43 % | 100.16 % | 394 s |
| `1e-3` (balanced) | 20.76 % | 30.53 % | **22.74 %** | 393 s |

At the same cost, the corrected tolerance is ~4.4x more accurate. More strikingly: **22.74 % at
3 000 steps beats the 59.75 % that `tolerance = 0.1` reached at 25 000 steps** — fixing this one
constant is worth more than 8x the compute budget. This is the single highest-leverage change
available in the repository.

Note this defect sits in the *shared* loss, so it penalises PINNs and KANs equally; it does not
bias the comparison, but it does cap the accuracy either can reach in a fixed budget.

### 4.11 `train_ic` cannot change the initial condition — `src/train.py:228`, `PIKAN.py:333`

`PIKAN.py` spends 1000 iterations on IC pretraining before every run:

```python
u_pred = model(torch.cat([x_in, t0], dim=1))
loss   = torch.mean((u_pred - target)**2)      # target = g(x)
```

It fits the **raw network** `N(x,0)` to `g(x)`. But `apply_ansatz` multiplies `N` by
`growth(0) = 0`, so `u(x,0) = g(x)` regardless. Measured:

```
BEFORE  max|u(x,0) - g(x)| = 0.000e+00      max|N(x,0) - g(x)| = 1.000e+00
AFTER   max|u(x,0) - g(x)| = 0.000e+00      max|N(x,0) - g(x)| = 3.685e-03

change in the MODEL OUTPUT u(x,0): 0.000e+00
change in the RAW NETWORK N(x,0):  1.002e+00
```

The network moves by 1.002; the model output moves by exactly zero. Worse than wasted: from §3.1,
`N(x,0)` controls `u_tt(x,0) = -225 g(x) + 1250 N(x,0)`. Forcing `N(x,0) = g(x)` sets
`u_tt(x,0) = 1025 g(x)`, whereas the correct value is `d/dx(E g')/rho`. The pretraining pushes the
second derivative toward a wrong value that the PDE loss must then undo.

### 4.12 No initial-condition ablation exists

`apply_ansatz` is applied unconditionally in all nine files that reference it. A repo-wide search
for `no_ansatz|use_ansatz|soft_ic|ic_loss|ic_weight|lambda_ic|w_ic` returns nothing. There is no
switch to disable the hard constraint, no soft-IC penalty term, and no `lambda_IC`. **The
"with IC vs without IC" experiment has never been run and cannot be run without new code.**

Worth stating for whoever runs it: with no IC at all the comparison is not merely "worse" — it is
degenerate. `u = 0` is an exact solution of the wave equation with zero PDE residual and zero
absorbing-BC residual, so it is the *global minimum* of the loss. A no-IC arm should be expected
to collapse to flat zero, not to learn a poor wave.

---

## 4b. What the repository actually does when you run it

Every entry point executed unmodified, on a copy, `MPLBACKEND=Agg`:

| Entry point | Outcome |
|---|---|
| `training_code.py` | prints `Using device: cuda` and exits — no `__main__` block, **does nothing** |
| `training_code.py::run_experiment()` | `TypeError: train_model_two_phase() missing 3 required positional arguments: 'x_fd', 't_fd', and 'u_fd'` (line 703) |
| `training_code_pikan.py::run_experiment()` | identical `TypeError`, line 703 — this is the fair-comparison driver |
| `src/benchmark.py` | `! No models found for exp1_homogeneous, skipping.` |
| `benchmark_v2.py` | `! No models found for exp2_twolayer, skipping.` |
| `PIKAN.py` | **runs to completion** |
| `Numerical methods/solve.py` | runs all six solvers, writes npz + gif |
| `Numerical methods/compare.py` | loads the npz, prints **zero error metrics** |

`PIKAN.py` unmodified produces:

```
PIKAN results:
  L=2, W=7  ->  59.90 % relative L2
  L=3, W=5  ->  79.36 % relative L2   (650 parameters)
Best config: L=2, W=7  ->  59.8964 %
checkpoint directory created: ./model
saving model version 0.0
```

That last pair of lines is the origin of `ML/model/history.txt` reading `init => 0.0` — it is
pykan's auto-save at initialisation, an artifact of exactly this script (§4.8).

Given §4.1–§4.3 as corrected, the honest summary of the KAN's actual run is: **a fair PDE, a fair
reference, a working L-BFGS — and a hopeless budget.** 59.90 % is what 10 Adam steps, 650
parameters, inverted loss weighting (§4.4) and one seed buy you.

`compare.py` output in full:

```
[File] Loading saved data...
UserWarning: Animation was deleted without rendering anything.
```

Lines matching `L2|error|METHOD`: **0**. Confirmed by execution, not by reading.

---

## 5. Measured results from what currently exists

### 5.1 Saved PINN checkpoints, re-evaluated

All 8 checkpoints evaluated against `fd_reference` (nx=512, nt=2000) at 20 times over
`t ∈ [0.05, 1.0] s`. This harness reproduces the repo's own published figures exactly
(PirateNet/exp3 → 1.06 %, matching `exp3_multilayer_metrics.png`; Fourier/exp2 → 1.90 %,
matching `exp2_twolayer_metrics.png`), so the numbers are trustworthy.

| Experiment | Arch | Params | mean L2 | early (t≤0.3) | late (t≥0.8) | PDE resid RMS |
|---|---|---|---|---|---|---|
| exp1 homogeneous | fourier | 99,329 | **57.42 %** | 15.58 % | 97.62 % | 1.17e+01 |
| exp1 homogeneous | vanilla | 66,561 | **57.43 %** | 15.58 % | 97.75 % | 1.37e+01 |
| exp1 homogeneous | pirate | 199,811 | **174.56 %** | 104.29 % | 241.22 % | 6.41e+01 |
| exp2 twolayer | fourier | 99,329 | 1.90 % | 1.11 % | 2.62 % | 5.74e+01 |
| exp2 twolayer | vanilla | 66,561 | 88.45 % | 28.94 % | 143.41 % | 1.94e+01 |
| exp3 multilayer | pirate | 149,890 | **1.06 %** | 0.58 % | 1.55 % | 2.85e+01 |
| exp3 multilayer | fourier | 99,329 | 4.27 % | 1.06 % | 6.45 % | 2.11e+01 |
| exp3 multilayer | vanilla | 66,561 | 127.24 % | 48.64 % | 181.83 % | 6.49e+01 |

**The `exp1` checkpoints are broken.** A 174 % error is worse than predicting zero everywhere
(100 %). The *easiest* case — homogeneous, constant coefficients — scores an order of magnitude
worse than the hardest. These are corrupt or badly under-trained checkpoints that were never
noticed because the exp1 figure was last regenerated with only WaveKAN enabled in `RUN_ARCHS`.

Genuine PINN capability, from the checkpoints that are sound: **≈1–4 % mean L2**, with error
growing monotonically in time (0.6 % early → 1.6 % late for the best).

### 5.2 KAN results that exist only as figures

`results/exp1/exp1_homogeneous_metrics.png` records WaveKAN on the homogeneous case:

| Model | L2 at t = 0.05 s | L2 at t = 1.0 s |
|---|---|---|
| WaveKAN 3 layers | ~0.5 % | ~22 % |
| WaveKAN 5 layers | ~0.5 % | ~20 % |
| **WaveKAN 7 layers** | ~0.5 % | **~4.2 %** |

WaveKAN-7L holds ≈1.3–2 % across most of the window — genuinely good, and achieved with very
few parameters (`WaveKAN_7-10` = 7 hidden layers of width 10). The weights no longer exist,
so this cannot be reproduced or extended from the repository as it stands.

Note these WaveKAN runs used the **PINN** loss path (`src/benchmark.py` imports
`compute_pde_residual`), not the broken `losses()`. They are the only KAN numbers in the repo
not affected by §4.1–4.5.

---

## 6. Controlled head-to-head experiment

Because nothing in the repo permits a valid comparison, I built one. Every factor is held
fixed and only the network changes:

- identical PDE residual — `compute_pde_residual`, heterogeneity-aware, for **all** models
- identical hard-IC ansatz, identical absorbing-BC loss, identical causal weighting
- identical `compute_grad_norm_weights` balancing, Adam, LR schedule, warmup/decay
- identical seed (42), collocation counts (8192 interior / 256 boundary), 8000 steps
- `kan_pikan_fair` = the repo's PIKAN with Fourier bandwidth matched to the PINNs (§4.7)

<!--RESULTS-->

All models: identical residual / ansatz / BC loss / causal weighting (corrected to `1e-3`) /
grad-norm balancing / Adam / LR schedule / seed 42 / 8192 collocation pts / 25 000 steps.
**Only the network changes.** Single seed — see the caveat below.

### Experiment 1 — Homogeneous

| Model | Params | best L2 | final L2 | PDE resid | wall |
|---|---|---|---|---|---|
| PirateNet (PINN) | 149,890 | **1.67 %** | 2.55 % | 2.21 | 17 min |
| PIKAN spline (KAN) | 171,350 | **2.40 %** | 39.52 % | 4.02 | 53 min |
| WavKAN 7x32 (KAN) | 24,960 | **3.08 %** | 4.48 % | 2.24 | 37 min |
| Fourier PINN (PINN) | 99,329 | **4.28 %** | 6.15 % | 2.87 | 13 min |

Mean L2 (%) vs Adam step — convergence stability:

| Model | 1k | 4k | 7k | 10k | 13k | 16k | 19k | 22k | 25k |
|---|---|---|---|---|---|---|---|---|---|
| PirateNet (PINN) | 20.8 | 6.1 | 6.6 | 5.3 | 4.1 | 4.0 | 1.8 | 3.9 | 2.5 |
| WavKAN 7x32 (KAN) | 115.6 | 40.1 | 9.1 | 14.0 | 7.4 | 7.5 | 3.7 | 3.9 | 4.5 |
| Fourier PINN (PINN) | 112.0 | 7.0 | 60.9 | 31.0 | 12.7 | 16.7 | 9.7 | 4.3 | 6.2 |
| PIKAN spline (KAN) | 154.0 | 90.0 | 86.7 | 77.3 | 56.6 | 30.6 | 4.3 | 3.5 | 39.5 |

### Experiment 2 — Two-layer (heterogeneous)

| Model | Params | best L2 | final L2 | PDE resid | wall |
|---|---|---|---|---|---|
| WavKAN 7x32 (KAN) | 24,960 | **4.59 %** | 4.59 % | 2.34 | 39 min |
| Fourier PINN (PINN) | 99,329 | **4.78 %** | 5.28 % | 2.75 | 15 min |
| PirateNet (PINN) | 149,890 | **7.20 %** | 9.25 % | 2.48 | 21 min |
| PIKAN spline (KAN) | 171,350 | **8.66 %** | 34.55 % | 5.26 | 55 min |

Mean L2 (%) vs Adam step — convergence stability:

| Model | 1k | 4k | 7k | 10k | 13k | 16k | 19k | 22k | 25k |
|---|---|---|---|---|---|---|---|---|---|
| PirateNet (PINN) | 20.7 | 86.9 | 91.1 | 84.2 | 60.6 | 12.2 | 13.7 | 24.4 | 9.3 |
| WavKAN 7x32 (KAN) | 98.3 | 29.6 | 29.4 | 20.1 | 10.4 | 8.7 | 6.8 | 5.0 | 4.6 |
| Fourier PINN (PINN) | 114.2 | 29.4 | 10.2 | 10.0 | 9.7 | 6.4 | 6.5 | 5.0 | 5.3 |
| PIKAN spline (KAN) | 161.0 | 92.1 | 87.7 | 75.8 | 56.2 | 41.6 | 16.5 | 32.7 | 34.6 |

### What the controlled run shows

1. **The winner depends on the regime.** PirateNet is best on the homogeneous medium
   (**1.67 %**); WavKAN is best on the heterogeneous one (**4.59 %**). Neither dominates.

2. **The KAN is dramatically more stable.** WavKAN's error falls monotonically on both problems
   (two-layer: 98.3 → 29.6 → 20.1 → 10.4 → 6.8 → 5.0 → 4.6) and its final error equals its best.
   PirateNet oscillates violently on the heterogeneous case — it *worsens* from 20.7 % at 1k to
   91.1 % at 7k before recovering, and ends 2.1 percentage points above its own best.

3. **WavKAN is the parameter-efficiency winner by a wide margin** — 24 960 parameters against
   PirateNet's 149 890 (6.0x), Fourier PINN's 99 329 (4.0x) and PIKAN's 171 350 (6.9x) — while
   placing first or second on both problems.

4. **The spline KAN (PIKAN) is not competitive.** It converges slowest, costs the most
   (53–55 min vs 13–21 min for the PINNs), and **diverges late in training on both problems**:
   3.5 % → 39.5 % on homogeneous, 16.5 % → 34.6 % on two-layer. This matters because PIKAN is the
   canonical KAN and is what `PIKAN.py` implements. The wavelet KAN, not the spline KAN, is the
   architecture worth pursuing here.

5. **PINNs are 2.5–3.5x cheaper per unit of wall-clock.** Same step count, far less time:
   Fourier PINN 13–15 min, PirateNet 17–21 min, WavKAN 37–39 min, PIKAN 53–55 min. Error-per-second
   still favours the PINNs.

**Caveat, stated plainly:** one seed per cell, one budget, two of the three material models.
The regime split (2.55 → 9.25 % for PirateNet vs 4.48 → 4.59 % for WavKAN) is far larger than
run-to-run noise usually is at this scale, but *confirming* it requires the >=3 seeds listed in
§7.4. This is a strong signal, not yet a proof.

---

## 7. Conclusions

### 7.1 Numerical methods

**Five of the six are verified correct.** DG, FEM, SEM and FD reproduce the free-field Green's
function amplitude to better than 0.7 % and the interface transmission coefficient to better than
0.25 %. FVM is correct but the least accurate (−3.0 % amplitude, +2.3 % transmission) and is
mislabelled — it is a Lax–Wendroff finite-difference scheme, not a finite-volume one.

**The pseudo-spectral solver is wrong in heterogeneous media** (§2.4): it integrates
`u_tt = c(x)²u_xx` instead of `u_tt = (1/ρ)∂ₓ(μ∂ₓu)`, dropping the `(c²)′uₓ` term that enforces
traction continuity. Measured transmission error **+33.8 %**. It is fine homogeneously.

Recommended roles: **FD** for bulk reference-data generation (0.03 s per run, 1400× faster than
DG at comparable accuracy); **SEM or DG** when you need a high-accuracy gold standard;
**PS homogeneous-only** until fixed.

Separately, `compare.py` computes no error metrics at all — `"Analytical"` is commented out of
`METHODS_TO_PLOT`, so the L2 block is dead code. The tables in §2 are the first quantitative
validation these solvers have received.

### 7.2 PINN vs KAN — the direct answer

**The repository cannot answer this question today, and no number currently in it should be
cited as evidence either way.** This is not a hedge; it is a specific, verified conclusion.

The two architectures were trained and evaluated through **two different pipelines that differ in
several ways besides the network**:

| | PINN path | KAN path |
|---|---|---|
| PDE solved | `u_tt − ∂ₓ(E uₓ)/ρ` — correct | `u_tt − u_xx` — wrong wherever `E ≠ const` (§4.1) |
| Reference solution | material-aware | `c ≡ 1`, **70.8 % wrong** on TwoLayer (§4.2) |
| Loss balancing | inverse gradient-norm | **inverted**, annihilates the BC term (§4.4) |
| Adam steps | up to 300,000 | **10** (§4.5) |
| Fourier bandwidth | ≈3 rad/unit | ≈44 rad/unit (§4.7) |

Two of these (§4.1, §4.2) are latent for the homogeneous runs `PIKAN.py` actually performed
but block the heterogeneous comparison outright. Any accuracy difference is confounded with the rest. A comparison run through these
two pipelines measures the pipelines, not the architectures.

Compounding this: **no KAN checkpoint exists in the repository** (§4.8). The only pykan artifact
present, `ML/model/`, is an untrained network saved at initialisation (`history.txt`:
`init => 0.0`). Both benchmark scripts request KAN architectures that cannot be loaded and exit
with `"No models found"`. And the two drivers that were *meant* to run the fair comparison,
`training_code.py` and `training_code_pikan.py`, both raise `TypeError` before training anything
(§4.6) — **the head-to-head has never once executed.**

### 7.3 The answer, from the controlled run

With every confound removed and one pipeline for all four architectures (§6):

| | Homogeneous | Two-layer | Params | Wall |
|---|---|---|---|---|
| PirateNet (PINN) | **1.67 %** | 7.20 % | 149,890 | 17–21 min |
| Fourier PINN | 4.28 % | 4.78 % | 99,329 | 13–15 min |
| **WavKAN (KAN)** | 3.08 % | **4.59 %** | **24,960** | 37–39 min |
| PIKAN spline (KAN) | 2.40 % → diverges | 8.66 % → diverges | 171,350 | 53–55 min |

**There is no single winner — the answer depends on what you are optimising for:**

- **Lowest error on a homogeneous medium → PINN (PirateNet).** 1.67 %, roughly 2x better than
  any KAN, and it gets there in a third of the time.
- **Robustness across media → KAN (WavKAN).** PirateNet degrades 4.3x moving from homogeneous to
  two-layer (1.67 → 7.20 %); WavKAN barely moves (3.08 → 4.59 %). For heterogeneous problems —
  which is the interesting case for elastic waves — the KAN is the better model.
- **Parameter efficiency → KAN, decisively.** WavKAN matches or beats every PINN on the
  heterogeneous problem using 4–7x fewer parameters.
- **Wall-clock efficiency → PINN, decisively.** KANs cost 2.5–3.5x more time for the same number
  of steps. Per second of compute the PINNs still win.
- **Training stability → KAN (WavKAN), with a sharp caveat.** WavKAN converges monotonically and
  ends at its best. But the *other* KAN — the canonical spline PIKAN — is the least stable model
  tested, diverging late on both problems. "KAN" is not one thing: the wavelet basis works here,
  the spline basis does not.

The earlier figure-only evidence (WavKAN-7L at ~1.3–2 % on homogeneous, §5.2) is consistent with
this: the wavelet KAN is genuinely strong, and it was always the KAN worth pursuing.

**Honest limits.** One seed per cell, one step budget, two of three material models, and none of
the models trained to the 300 000-step budget the repo's best checkpoints used. The regime split
is much larger than run-to-run noise typically is, but §7.4's multi-seed protocol is what turns
this from a strong signal into a result you can publish.

### 7.4 What certainty requires — priority-ordered

**Blocking (nothing is interpretable until these are done):**

0. **Fix the causal tolerance** — `0.1` → `1e-3` (`ansatz_losses.py:63`). Highest-leverage change in
   the repository: worth more than 8x the compute budget (§4.10). Do this one first.
1. **One loss for both.** Delete `losses()`; route KANs through `compute_pde_residual`
   (`ansatz_losses.py:31`). Removes §4.1.
2. **One reference for both.** Delete `solve_wave_fd`/`u_sol`; use `fd_reference` everywhere.
   Removes §4.2.
3. **One trainer for both.** Fix `strong_wolf` → `strong_wolfe` (`train.py:307`); replace the
   inverted rule at `train.py:271` with `compute_grad_norm_weights`. Removes §4.3, §4.4.
4. **Equal budget.** `PIKAN.py:480` `ITERATIONS = 10` → the same step count the PINNs get.
5. **Fix the drivers.** Pass `x_fd, t_fd, u_fd` into `train_model_two_phase`
   (`training_code*.py:703`); import `PIKAN` in `training_code.py`. Removes §4.6.

**Required for a defensible claim:**

6. **Control Fourier bandwidth** — sweep it per architecture, or fix it identically. Removes §4.7.
7. **Report parameters and wall-clock alongside error.** The interesting KAN claim is efficiency,
   not raw accuracy; error alone cannot show it.
8. **Stop selecting on the test metric** (`train.py:168–175`) — hold out an evaluation set, or
   report final-step error. Reconcile the `T_MAX` mismatch (1.0 vs 0.75 for exp1).
9. **Multiple seeds.** `REPETITIONS = 1` cannot separate architecture from initialisation.
10. **Retrain and commit the exp1 checkpoints**, and save the WaveKAN weights this time.

**Then, and only then:** run all architectures × {homogeneous, two-layer, multi-layer} ×
≥3 seeds under one pipeline, and the question becomes answerable.

### 7.5 One structural recommendation

`Numerical methods/` and `ML/` currently solve unrelated problems (§1). The six-method suite is
the natural gold standard for the neural work, but only if they are made to solve the *same*
problem — same units, same source or IC, same boundary treatment, same output variable. Doing
that would also let you validate `fd_reference` itself, which is presently unverified against
anything.

---

## 8. Reproduction

Scripts written during this audit (scratchpad, nothing in the repo was modified):

| Script | Purpose |
|---|---|
| `verify_numerical.py` | runs all 6 solvers, errors vs ray-traced analytic (homogeneous) |
| `verify_interface.py` | interface transmission coefficient test (heterogeneous) |
| `verify_amp.py` | clean free-field amplitude test, source node excluded |
| `eval_pinns.py` | re-evaluates every saved checkpoint in `ML/Models` |
| `headtohead.py` | controlled PINN vs KAN benchmark under one fixed loss/trainer |
