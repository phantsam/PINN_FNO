# Why the KAN Beats the PINN, and What the nx=2048 Reference Changed

Two separate questions, both answered by measurement rather than argument:

1. **What did changing the reference from nx=512 to nx=2048 actually change?**
   Nothing about the models -- the reference never enters training. It changed
   what we could *see*, and by a precisely quantifiable factor.
2. **Why does the tuned KAN beat the Fourier PINN on the homogeneous problem?**
   Because the solution's energy lives on a measure-zero set in 2-D frequency
   space, the KAN's basis sits exactly on that set by construction, and the
   PINN's random features are isotropic and mostly miss it.

Everything below is reproducible from the checkpoints and `core/rescore.py`.

---

## PART 1 -- What nx=2048 changed

### 1.1 The reference never enters training

`train_lbfgs` references `x_ref/t_ref/u_ref` in exactly two places: the function
signature, and the `evaluate()` call **after the training loop terminates**.
Early stopping tests `v < best - min_delta` where `v` is the PDE + BC residual;
weight selection (`best_w`) is driven by the same quantity.

So the trained weights are **identical** under either reference. The nx=512 ->
nx=2048 change is a change of instrument, not of experiment. All 169 training
runs stand.

### 1.2 The error decomposition

Let `u*` be the true solution, `u_h` the FD reference at resolution `h`, and
`u_θ` a trained model. Write

```
    a = u_θ − u_2048        (the model's error -- the quantity of interest)
    b = u_2048 − u_512      (the coarse reference's own error)
```

Then `u_θ − u_512 = a + b`, so what we *measured* at nx=512 was

```
    E_512² = ‖a + b‖² / ‖u‖²  =  E_2048² + ref² + 2⟨a,b⟩/‖u‖²
```

Measured, on homogeneous, with snapshot times matched exactly:

| model | E_2048 | E_512 measured | √(E_2048² + ref²) | cos(a,b) | dE_512/dE_2048 |
|---|---|---|---|---|---|
| pirate | 0.0296 % | 0.1114 % | 0.1101 % | 0.046 | 0.269 |
| fourier | 0.0292 % | 0.1118 % | 0.1100 % | 0.065 | 0.265 |
| charcoords50 | 0.0205 % | 0.1116 % | 0.1080 % | 0.181 | 0.190 |

with `ref = ‖b‖/‖u‖ = 0.1060 %`.

**The orthogonal prediction matches the measurement to ~3 %** (`cos(a,b)` between
0.05 and 0.18), so model error and discretisation error are essentially
independent directions. Therefore

```
    E_512  ≈  √(E_2048² + ref²)
```

### 1.3 Why that flattened everything

Differentiating,

```
    dE_512/dE_2048  =  E_2048 / √(E_2048² + ref²)  ≈  E_2048/ref   when E_2048 ≪ ref
```

Measured: **0.19 to 0.27**. A 1 % change in a model's true error moved the nx=512
reading by only about a quarter of a percent -- a **~4x compression of every
difference in the study**.

Concretely, on homogeneous the true gap is

```
    PINN 0.0292 %  vs  KAN 0.0205 %      →  ratio 1.424x
```

and what the nx=512 reference reported was

```
    √(0.0292² + 0.1060²) = 0.1100 %
    √(0.0205² + 0.1060²) = 0.1080 %      →  ratio 1.018x
```

**A genuine 1.42x difference was displayed as 1.018x.** That is the entire
explanation for the sequence of conclusions this study went through: the
instrument's noise floor was the same size as the signal, and it was *common
mode* -- identical for every model -- so it compressed all of them toward one
number.

### 1.4 A second, independent artefact: snapshot time misalignment

`evaluate` selected snapshots by nearest-neighbour in time. Because `dt` is
CFL-limited it differs between grids, so "the same" snapshot sat at a different
instant on each grid:

| nx | dt | t nearest 0.5 | offset |
|---|---|---|---|
| 512 | 1.761e-3 | 0.50000000 | 0 |
| 1024 | 8.795e-4 | 0.49956025 | −4.398e-4 |
| 4096 | 2.197e-4 | 0.49989013 | −1.099e-4 |

With `c = 1` a time offset `δt` translates the pulse by `δt` in `x`. For a
σ = 0.1 derivative-of-Gaussian, `‖u′‖/‖u‖ ≈ 14`, so

```
    relative error  ≈  δt · ‖u′‖/‖u‖  ≈  4.4e-4 × 14  ≈  0.6 %
```

That artefact alone produced an apparent 0.6430 % "reference error" and made the
solver look **first order** (measured orders 1.09 / 1.04 / 0.74). Interpolating
linearly in time to the exact target instants -- an O(dt²) operation -- recovers
the true behaviour:

```
    nx    rel-L2 vs nx=4096    order
    256          0.4509 %        --
    512          0.1110 %       2.02
   1024          0.0265 %       2.07
   2048          0.0054 %       2.30
```

**Second order confirmed.** The solver was always correct; the comparison was not.

### 1.5 Corrected results

| material | model | params | E_512 | **E_2048** | sd/mean @2048 |
|---|---|---|---|---|---|
| homogeneous | pirate | 199,811 | 0.1114 % | 0.0296 % | 8.2 % |
| homogeneous | fourier | 82,689 | 0.1118 % | 0.0292 % | 9.4 % |
| homogeneous | **charcoords50** | **49,020** | 0.1116 % | **0.0205 %** | 45.2 % |
| multilayer | pirate | 199,811 | 0.1671 % | 0.0563 % | 39.6 % |
| multilayer | fourier | 82,689 | 0.1617 % | 0.0388 % | 30.5 % |
| multilayer | **charcoords50** | **49,020** | 0.1628 % | **0.0393 %** | 50.8 % |

At nx=2048 the reference error is 0.0054 % (homogeneous), so these numbers sit
4-11x above the floor and are genuinely resolved.

**Every model was 3-4x more accurate than ever reported.**

### 1.6 A claim this forces me to withdraw

I reported PirateNet's seed spread as **0.6 %** and used it as evidence that
PINNs are more reliable than KANs. At nx=2048 that same arm spreads **8.2 %**.

The apparent tightness was an artefact: all three seeds were dominated by the
*same* reference error, which is common-mode and cancels in the spread. The
ordering survives -- the KAN's 45 % is still far worse -- but the magnitudes I
reported were fiction, and the KAN's variance is now the strongest single
argument against it.

---

## PART 2 -- Why the KAN wins

### 2.1 The solution is exactly separable in characteristic coordinates

For constant `c` the initial-value problem with `u_t(x,0) = 0` has the d'Alembert
solution

```
    u(x,t) = ½ [ g(x − t) + g(x + t) ]  =  F(ξ) + G(η),     ξ = x − t,  η = x + t
```

Measured against the nx=2048 solution:

```
    ‖u − ½[g(x−t) + g(x+t)]‖ / ‖u‖  =  0.0068 %
```

So on this problem the target **is** a sum of two univariate functions, to within
seven parts in 100,000. That is not an approximation; it is the structure of the
equation.

### 2.2 A KAN layer is literally that form

A Kolmogorov-Arnold layer computes

```
    out_j = Σ_i φ_ij(z_i)
```

-- a **sum of univariate functions of the individual inputs**. Feed it
`(ξ, η)` and its very first layer spans exactly `{F(ξ) + G(η)}`, which is the
solution space. Feed it `(x, t)` and it must instead synthesise a genuinely
two-dimensional function by composition through depth.

The characteristic transform supplies **no information** -- on homogeneous
`τ(x) = x/c` makes `(ξ,η)` a pure rotation of `(x,t)`, and measurement confirms it
(effective rank 1.80, linear probe 99.884 %, hi-k share 0.135 -- identical to the
digit for both inputs). It supplies **alignment**: it rotates the solution's
structure onto the axes along which a KAN is separable.

### 2.3 Where the energy actually is

`u = F(x−t) + G(x+t)` has a 2-D Fourier transform supported on the **two lines**
`k_t = ±k_x` -- a measure-zero set in the `(k_x, k_t)` plane. Measured on the
nx=1024 solution, with `sin θ` the angular distance to the nearer line:

| angular band | share of total energy |
|---|---|
| sin θ < 0.05 | 1.62 % |
| **sin θ < 0.10** | **55.68 %** |
| sin θ < 0.20 | 84.06 % |
| sin θ < 0.50 | 94.92 % |

Over half the energy lies within ~6° of the characteristic lines.

### 2.4 Where each basis puts its capacity

**Random Fourier features** draw `b_i = (b_x, b_t) ~ N(0, σ_B² I)` -- isotropic in
the plane. Measured for the 64 frequencies actually used:

| angular band | share of the 64 features | isotropic expectation |
|---|---|---|
| sin θ < 0.05 | 6.2 % | 3.2 % |
| sin θ < 0.10 | 12.5 % | 6.4 % |
| sin θ < 0.20 | 28.1 % | 12.8 % |

So roughly **12 % of the PINN's features cover the band holding 56 % of the
energy**, and the remainder is spent on directions where the solution has almost
none. The bandwidth `σ_B = 10` was tuned to the correct *radius* (k_peak ≈ 9.4)
but nothing selects the correct *angle* -- the features are isotropic by
construction.

**Separable splines on `(ξ, η)`** have every basis function depending on `ξ`
alone or `η` alone. Each therefore has a transform lying **exactly on one of the
two lines**. 100 % of the capacity sits on the support of the target, by
construction rather than by luck.

### 2.5 The consequence, measured

Least-squares fit of the nx=2048 solution in each space, as a function of
dimension:

| random Fourier (σ_B = 10) | | separable splines on (ξ,η) | |
|---|---|---|---|
| dim 32 | 79.82 % | dim 22 | 84.20 % |
| dim 64 | 71.34 % | dim 36 | 48.24 % |
| dim 128 | 44.47 % | dim 66 | **3.46 %** |
| dim 192 | 25.42 % | dim 106 | **0.1914 %** |
| dim 256 | 15.33 % | dim 154 | **0.0270 %** |

**At comparable dimension -- 128 Fourier features vs 106 spline basis functions --
the separable basis is 233x more accurate (44.47 % vs 0.1914 %).**

The random-Fourier space converges roughly like `O(m^{-1/2})`, the rate expected
for random features approximating a function whose energy they only partially
cover. The separable spline space converges like a univariate cubic-spline
approximation of `g`, i.e. `O(h⁴)` -- because after the rotation that is all it is
being asked to do.

### 2.6 Why the trained networks show a smaller gap than 233x

The function-space comparison is a statement about the *first layer* of each
model. The trained networks differ by only 1.42x (0.0292 % vs 0.0205 %) because:

- The Fourier PINN has **five tanh layers after the embedding** to repair what the
  embedding misses. Forensics shows they operate nearly linearly (`mean|z|` falls
  0.543 -> 0.300, only 1.3 % of units past `|z| > 1`), so it is close to linear
  regression in the random-feature space -- but not exactly, and the deviation
  helps.
- The KAN is not run at the resolution its basis could exploit. Its own trace
  shows `acts3` reaching only 8 % of its grid -- 4.9 of 61 knots -- so it realises
  a small fraction of the `O(h⁴)` rate available to it. Phase 9 established that
  this headroom is **unreachable under L-BFGS**: refitting the grids to recover it
  destroys the curvature estimate, and 8 of 8 runs collapsed.

So the measured 1.42x is a *lower bound* on the structural advantage. The
approximation-theoretic gap is two orders of magnitude; optimisation gives back
most of it.

### 2.7 Why this fails on twolayer

The argument depends on `u = F(ξ) + G(η)`, which requires a single wave family in
each direction. At a material interface the wave **splits** into transmitted and
reflected components, and the decomposition acquires more terms than there are
coordinates -- no pair `(ξ, η)` makes it separable.

The severity should then scale with **interface sharpness**, not interface count,
and the measurements agree:

| | twolayer | multilayer |
|---|---|---|
| interfaces | 1 | 5 |
| E contrast | 1.5x | **2.5x** |
| transition width `w` | **0.02** | 0.05 |
| max\|E′\| | **12.50** | 3.00 |
| max\|τ″\| | **4.576** | 1.228 |
| pulse FWHM / w | **11.8x** | 4.7x |
| reflection coefficient | **0.101** | 0.065 |
| `charcoords50` | 1.6642 % | **0.0393 %** |

twolayer has *fewer* interfaces and *lower* contrast, but its `E′` is **4.2x
larger** because the same jump is compressed into a transition 2.5x narrower. To
a pulse of FWHM 0.236 that interface is effectively a discontinuity.

And the transform makes it worse: `u_xx` in characteristic coordinates picks up
`τ″ · u_ξ` through the chain rule, and `τ″ = −c′/c²` is **3.7x larger** on
twolayer. The rotation that aligns the homogeneous problem *injects* the
interface spike into the residual.

This remains **correlational**. It is supported by PirateNet failing on the same
material (2.768 % vs 0.111 % elsewhere -- a 25x inversion in a different
architecture family), but the clean test is running twolayer at `w = 0.05` and
multilayer at `w = 0.02`. If sharpness is the cause, the failure follows `w`, not
the material.

---

## Summary

**What nx=2048 changed:** nothing about the models, everything about visibility.
The nx=512 reference had error 0.1060 %, essentially equal to the model errors
being measured. Because that error is common-mode and near-orthogonal to the
model errors, measurement compressed all differences by a factor
`E_2048/ref ≈ 0.19–0.27`, turning a genuine 1.42x gap into a displayed 1.018x.

**Why the KAN wins:** the solution is exactly `F(ξ) + G(η)`; a KAN layer *is* a
sum of univariate functions; and characteristic coordinates rotate the problem so
those axes coincide. The KAN's basis therefore sits entirely on the measure-zero
set where the solution's energy lives, while isotropic random Fourier features
place only ~12 % of their capacity on the band holding 56 % of it. At matched
dimension the separable basis is **233x** more accurate.

**Why the advantage is only 1.42x in practice:** the PINN's depth partially
repairs its basis, and the KAN cannot reach the resolution its own basis would
require -- a limit imposed by the optimiser, not the architecture.
