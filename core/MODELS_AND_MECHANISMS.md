# Every Model, Layer by Layer — and Why They Succeed or Fail

A complete architectural and mechanistic reference for the PINN-vs-KAN study on
the 1-D elastic wave equation. Written to be read without prior KAN knowledge:
§2 builds Kolmogorov-Arnold networks from scratch before any architecture is
described.

Companion documents: `BUGS_AND_CORRECTIONS.md` (every defect and retraction),
`ARCHITECTURES.md` (phase chronology), `DECISIONS.md` (frozen spec).

---

## 0. Reading guide

| section | what it covers |
|---|---|
| §1 | the shared pipeline every model sits inside — input tensor, ansatz, loss |
| §2 | **how a KAN actually works**, from the theorem to the tensor shapes |
| §3 | each architecture, layer by layer, with dimensions at every step |
| §4 | the best model on each material, and the numbers behind it |
| §5 | why Adam→L-BFGS beats pure L-BFGS |
| §6 | why the soft initial condition fails |
| §7 | the cancellation principle that explains every failure in one line |

---

## 1. The shared pipeline

Everything below is identical across every architecture. Only the network varies.

### 1.1 The input tensor

```
x : (N, 1)   float32   spatial coordinate, x in [-1, +1]
t : (N, 1)   float32   time,               t in [0, 1]
```

`N = 10,000` during training — a Sobol low-discrepancy sample of the space-time
rectangle, drawn **once** and frozen (L-BFGS needs a deterministic objective).
`N = 81,920` at evaluation (4096 spatial points x 20 snapshots).

Sobol rather than uniform random: a scrambled Sobol sequence fills the rectangle
with lower discrepancy, so the residual is sampled more evenly for the same count.
The draw is byte-identical between the pure-L-BFGS and hybrid code paths for a
given seed — verified, not assumed.

### 1.2 The hard-constraint ansatz

Every network outputs a raw field `N(x,t)` of shape `(N, 1)`. That is **not** the
solution. The trainer forms:

```
u(x,t) = g(x)·decay(t)  +  growth(t)·N(x,t)

  g(x)      = derivative-of-Gaussian, peak-normalised to max|g| = 1, sigma_g = 0.1
  decay(t)  = exp(-1/2 · (15t)^2)
  growth(t) = tanh^2(25t)
```

This enforces `u(x,0) = g(x)` and `u_t(x,0) = 0` **exactly**, because
`decay(0)=1`, `decay'(0)=0`, `growth(0)=0`, `growth'(0)=0`. No initial-condition
loss term is needed and the IC can never be violated.

Numbers that matter later: `decay` is 0.755 at t=0.05, 0.325 at t=0.1, and
**0.011 by t=0.2**. `growth` is 0.72 at t=0.05 and **0.974 by t=0.1**. So across
almost the whole scoring window `t in [0.05, 1.0]`, `u ≈ N`. That single fact
creates the trivial basin discussed in §6.

### 1.3 The loss

```
L  =  mean( r(x,t)^2 )  +  mean( b(t)^2 )

  r = rho(x)·u_tt - d/dx( E(x)·u_x )      the PDE residual, 10,000 points
  b = u_t -/+ c(x)·u_x                     absorbing BCs, 512 points per edge
```

Unnormalised (dividing by `residual_scale^2 ≈ 5e4` shrinks gradients by the same
factor and makes L-BFGS's first step overshoot — see `BUGS_AND_CORRECTIONS.md` #1).
Both `u_tt` and `u_x` come from `torch.autograd.grad`, so the network must be twice
differentiable and those derivatives must be *accurate*, not merely defined. §7
shows that this is the whole ballgame.

---

## 2. How a KAN actually works

### 2.1 The idea, against the MLP you already know

An **MLP** puts **fixed** nonlinearities on the *nodes* and **learned** weights on
the *edges*:

```
node output  =  sigma( sum_i  w_i · x_i  + b )
                ^^^^^        ^^^
                fixed        learned scalars
```

A **KAN** inverts this. It puts **learned univariate functions on the edges** and
plain **summation on the nodes**:

```
node output  =  sum_i  phi_i( x_i )
                       ^^^^^^
                       a learned FUNCTION of one variable
```

There is no weight matrix and no activation function in the MLP sense. Each edge
carries its own little curve, shaped during training.

The motivation is the **Kolmogorov-Arnold representation theorem**: any continuous
multivariate function can be written as a finite composition of sums of
*univariate* functions. An MLP approximates a multivariate function directly; a KAN
approximates it as compositions of one-dimensional pieces, which is what the
theorem says is always possible.

**Why this matters for our PDE specifically.** The exact solution of the
homogeneous wave equation is

```
u(x,t) = 1/2 [ g(x - t) + g(x + t) ]  =  F(xi) + G(eta)
```

a **sum of two univariate functions** of the characteristic variables
`xi = x - t` and `eta = x + t`. That is *literally* a Kolmogorov-Arnold
representation with one layer and two edges. §3.5 exploits this.

### 2.2 The edge function, concretely

Each edge carries

```
phi(x)  =  scale_base · silu(x)  +  scale_sp · spline(x)
           \______________/         \_______________/
           a fixed shape with        a fully learned curve,
           one learned amplitude     built from B-splines
```

(`KANLayer.py:161` in pykan.) Both `scale_base` and `scale_sp` are trained. The
`silu` branch is a residual path that gives the edge a sensible shape before the
spline has learned anything; the spline branch is where the expressive power lives.

### 2.3 B-splines: what the spline branch is made of

A **B-spline** is a piecewise polynomial built to be locally supported and smooth.
Two numbers define the family:

| term | meaning | our KAN (`charcoords50`) |
|---|---|---|
| **grid** | number of intervals the input range is cut into | `grid = 50` |
| **k** (order) | polynomial degree on each interval | `k = 5` |
| **grid_range** | the input interval the knots span | `[-1, +1]` (pykan default) |

From those:

```
number of basis functions   =  grid + k       =  50 + 5  =  55
number of knots stored      =  grid + 2k + 1  =  50 + 11 =  61
knot spacing                =  2 / 50         =  0.04
```

Both numbers are visible in the parameter dump: `coef` has last dimension **55**
and the `grid` buffer has **61** entries.

**Local support** is the key property. At any input `x`, only `k + 1 = 6` of the 55
basis functions are non-zero. So one coefficient only affects the curve near its
own knot. That is what makes a KAN expressive at fine scales — and, as §7 shows, it
is also why grid changes are dangerous.

**Why `k` matters for a PDE.** The spline is `C^(k-1)`. At `k = 3` the network is
`C^2`, so `u_xx` — which the residual consumes — is only `C^0`: piecewise linear
with a **kink at every knot**. At `k = 5` the network is `C^4` and `u_xx` is `C^2`,
genuinely smooth. For a second-order PDE this is a principled change, not a tweak.

The spline value is computed by the Cox–de Boor recursion (`coef2curve`): start
from indicator functions on each interval, then raise the degree `k` times, each
step blending neighbouring basis functions linearly in `x`.

### 2.4 A KAN layer, with shapes

A layer with `in_dim` inputs and `out_dim` outputs holds `in_dim x out_dim` edges,
each with its own spline. For `act_fun[1]` of `charcoords50`:

```
in_dim = 20, out_dim = 20, grid = 50, k = 5

coef        (20, 20, 55)   = 22,000   one 55-coefficient spline per edge
scale_base  (20, 20)       =    400   silu amplitude per edge
scale_sp    (20, 20)       =    400   spline amplitude per edge
grid        (20, 61)       buffer     knot positions, one row per INPUT
                                      (shared across that input's 20 output edges)
```

Forward pass, for a batch of `N` points:

```
input                 (N, 20)
  |
  |-- base branch:    silu(x)                       -> (N, 20)
  |-- spline branch:  coef2curve(x, grid, coef, k)  -> (N, 20, 20)   [in, out]
  |
  phi = scale_base * base[:,:,None] + scale_sp * spline             (N, 20, 20)
  |
  sum over the INPUT axis                                            (N, 20)
output                (N, 20)
```

The node does nothing but add. All the modelling is in the 400 edge functions.

### 2.5 Terminology, collected

| term | meaning |
|---|---|
| `width` | nodes per layer, e.g. `[2,20,20,20,1]` |
| `grid` | spline intervals per edge — the *resolution* knob |
| `k` | spline order; controls smoothness of `u` and hence of `u_xx` |
| `coef` | the trained spline coefficients, `(in, out, grid+k)` |
| `scale_base` / `scale_sp` | trained amplitudes of the two branches |
| `base_fun` | the fixed residual shape, `silu` here |
| `grid_range` | the interval the knots cover, `[-1,1]` |
| `grid_eps` | blend between uniform and quantile-adaptive knot placement (0.02) |
| `noise_scale` | initialisation noise on the spline coefficients (0.3) |
| `curve2coef` | least-squares fit of coefficients to sampled curve **values** |
| `update_grid` | re-place the knots to match observed activations, then `curve2coef` |
| `symbolic_fun` | a parallel branch for symbolic regression — **disabled** here |

Two of these are load-bearing for our results: `update_grid` (§7.2) and
`curve2coef` (which returned NaN on CUDA until we patched it — see
`BUGS_AND_CORRECTIONS.md`).

---

## 3. The architectures, layer by layer

Batch size `N` throughout. All dimensions verified by introspecting the live
models, not from memory.

### 3.1 MLP PINN — `mlp` — 66,561 parameters

The control: no embedding, raw coordinates in.

```
x (N,1), t (N,1)
  cat                                        -> (N, 2)
  Linear(2   -> 128) + tanh                  -> (N, 128)     256 + 128
  Linear(128 -> 128) + tanh                  -> (N, 128)  16,384 + 128
  Linear(128 -> 128) + tanh                  -> (N, 128)  16,384 + 128
  Linear(128 -> 128) + tanh                  -> (N, 128)  16,384 + 128
  Linear(128 -> 128) + tanh                  -> (N, 128)  16,384 + 128
  Linear(128 -> 1)                           -> (N, 1)       128 + 1
```

`tanh` is chosen because it is smooth to all orders — `relu` would give `u_xx = 0`
almost everywhere and the residual would be blind to the network.

**Best model on twolayer** (0.4318 ± 0.3070). The plainest architecture wins on the
sharpest material, which is worth sitting with: Fourier features tuned to the
*pulse* spectrum do not help with an interface discontinuity.

### 3.2 Fourier PINN — `fourier` — 82,689 parameters

Same MLP with a random Fourier feature embedding in front.

```
x (N,1), t (N,1)
  cat                                        -> (N, 2)
  @ B.T          B is (64, 2), FIXED buffer  -> (N, 64)
                 entries ~ Normal(0, sigma_B^2), sigma_B = 10
  cat[cos(p), sin(p)]                        -> (N, 128)
  Linear(128 -> 128) + tanh                  -> (N, 128)
  ... 4 more 128-wide tanh layers ...
  Linear(128 -> 1)                           -> (N, 1)
```

`B` is a **buffer, not a parameter** — the bandwidth stays at its declared value
for the whole run, so it is controlled rather than learned (`DECISIONS.md` D4).

`sigma_B = 10` was tuned to the measured spectral peak of the solution,
`k_peak ≈ 9.4`. The embedding lets a tanh MLP represent oscillations it would
otherwise need enormous depth to reach.

**The cost, which matters in §7:** every derivative brings down a factor of the
frequency. `u_x` picks up `2*pi*k ≈ 190`; `u_xx` picks up `(2*pi*k)^2 ≈ 3.5e4`.
The embedding that makes the function easy to fit makes its second derivatives
noisy.

### 3.3 Fourier PINN, parameter-matched — `fourier_matched` — 49,729 parameters

Identical except the hidden width is 96 rather than 128, chosen to match the tuned
KAN's 49,020 parameters to within 1.45 %.

```
  Linear(128 -> 96) + tanh    <- first layer narrows from the 128-d embedding
  Linear(96  -> 96) + tanh    x4
  Linear(96  -> 1)
```

**It is better than the 128-wide version** — 0.0332 % vs 0.0401 % on homogeneous.
The default was over-parameterised, so the headline PINN arm had been handicapped
by its own capacity.

### 3.4 PirateNet — `pirate` — ~150k parameters

Fourier embedding, then residual blocks with random weight factorisation
`W = diag(exp(s)) · V`, and a zero-initialised output layer fixed by a
physics-informed least-squares step at construction (without it the network is
identically zero at init, `u_tt = 0`, and the backbone is dead).

Competitive on homogeneous (0.0296 ± 0.0024, the **lowest variance of any model**)
and multilayer (0.0563), but **2.768 %** on twolayer — a 25x inversion on the sharp
interface, the same material where the plain MLP wins.

### 3.5 Plain KAN — `pykan_wide` — 8,600 parameters

The reference pykan implementation on raw `(x,t)`. Width `[2,20,20,20,1]`,
`grid = 5`, `k = 3`.

```
x (N,1), t (N,1)
  cat                                          -> (N, 2)

  act_fun[0]   2 -> 20     40 edges
      coef (2,20,8)   grid (2,12)   n_basis = 5+3 = 8
      phi = scale_base*silu + scale_sp*spline,  sum over 2 inputs
                                               -> (N, 20)
  act_fun[1]  20 -> 20    400 edges   coef (20,20,8)   grid (20,12)
                                               -> (N, 20)
  act_fun[2]  20 -> 20    400 edges   coef (20,20,8)
                                               -> (N, 20)
  act_fun[3]  20 ->  1     20 edges   coef (20,1,8)
                                               -> (N, 1)
```

860 edges total; 8 coefficients each = 6,880, plus 1,720 scale parameters.

**A real handicap worth naming.** `PyKAN.forward` is `self.kan(cat([x,t]))` with no
input scaling, and pykan's `grid_range` is `[-1,1]`. Our `x` fills that range but
`t` only covers `[0,1]` — so **roughly half the knots along the time axis are never
visited**. Phase 8's `normalise` rung tested exactly this and it mattered.

Despite that: 0.0849 % on homogeneous from 8,600 parameters — an order of magnitude
fewer than any PINN here.

### 3.6 The tuned KAN — `charcoords50` — 49,020 parameters — **the best model**

Same pykan backbone, three changes. This is the architecture that wins.

```
x (N,1), t (N,1)
  |
  |  CharCoords:  tau(x) = integral dx/c(x)      travel-time coordinate
  |               xi  = tau(x) - t
  |               eta = tau(x) + t
  |               each affinely mapped to [-1,1]
  |                                              -> (N, 2)
  |  (no affine scaler needed; the pair already fills grid_range exactly)
  |
  act_fun[0]   2 -> 20     40 edges
      coef (2,20,55)   grid (2,61)   n_basis = 50+5 = 55   knot spacing 0.04
                                               -> (N, 20)
  act_fun[1]  20 -> 20    400 edges   coef (20,20,55)  = 22,000 params
                                               -> (N, 20)
  act_fun[2]  20 -> 20    400 edges   coef (20,20,55)  = 22,000 params
                                               -> (N, 20)
  act_fun[3]  20 ->  1     20 edges   coef (20,1,55)   =  1,100 params
                                               -> (N, 1)
```

The three changes from `pykan_wide`, and why each is principled:

| change | from | to | reason |
|---|---|---|---|
| input coordinates | raw `(x,t)` | `(tau-t, tau+t)` | the exact solution is `F(xi)+G(eta)` — a literal sum of univariate functions, i.e. a Kolmogorov-Arnold representation. The coordinate change makes the KAN's own structure match the solution's. |
| `grid` | 5 | 50 | 10x resolution; knot spacing 0.04 against a pulse of FWHM 0.236 |
| `k` | 3 | 5 | `u` becomes `C^4`, so `u_xx` is `C^2` rather than kinked at every knot |

`tau(x)` is built by quintic Hermite interpolation of a Gauss-Legendre quadrature
of `1/c(x)`, in float64, accurate to `tau' = 1.5e-7` and `tau'' = 1.9e-6`. It is a
fixed buffer, not learned.

**This is the model that beats every PINN on homogeneous**, and §7 explains why in
one sentence.

### 3.7 The PIKANs — both fail

**`splinekan` (ours) — 246,208 parameters.** A hand-rolled spline layer on a 128-d
Fourier embedding.

```
x, t -> FourierEmbed(64, sigma=10)  -> (N, 128)
     -> _SplineKANLayer(128 -> 64, grid=10, order=4)  -> LayerNorm -> (N, 64)
     -> _SplineKANLayer(64 -> 64) x2                                -> (N, 64)
     -> Linear(64 -> 1)                                             -> (N, 1)
```

Each `_SplineKANLayer` computes
`F.linear(silu(x), base) + einsum("bik,oik->bo", bspline(x), coef)`.

**It is broken at initialisation.** `max|u_tt| = 1.16e6` against every other model's
~2e2; PDE loss `3.5e10` and gradients `4.4e10` at step zero. Cause: B-splines placed
directly on a `sigma_B = 10` Fourier embedding, so the chain rule compounds
`(2*pi*k)^2 ≈ 3.5e4` from the embedding with `(1/knot spacing)^2 ≈ 25` from the
spline. pykan avoids this by damping its spline branch `1/sqrt(fan_in)`
(`KANLayer.py:112`); plain Xavier does not. `splinekan_fix` applies the same
damping and drops the init loss **100,000x** — converting a divergence (235 %) into
a collapse (97 %).

**`fourier16` (pykan backend) — 36,500 parameters.** The same idea done correctly:
pykan on a 16-feature Fourier embedding, `grid = 20`, `k = 3`.

```
x, t -> FourierEmbed(16, sigma=10)   -> (N, 32)
     -> act_fun[0]  32 -> 20         -> (N, 20)
     -> act_fun[1..2]  20 -> 20      -> (N, 20)
     -> act_fun[3]  20 -> 1          -> (N, 1)
```

Initialisation is **healthy** (`max|u_tt| = 2.39e2`). It collapses anyway, on all
three materials, under both optimisers, at ~95 %. §7.3 explains it.

### 3.8 WavKAN — `wavkan`

Mexican-hat wavelets on raw `(x,t)`: `psi(z) ∝ (z^2 - 1)·exp(-z^2/2)` with learned
per-edge scale and translation. Diverges under L-BFGS via a float32 overflow in
torch's cubic line search (`d1^2 ≈ 2.7e49`), and is the only architecture that
one-shot residual resampling ever rescued — 4 times out of 4, from ~95 % to
0.8–5.2 %.

---

## 4. Results — the best model on each material

Scored against an nx=2048 finite-difference reference unless noted. Errors are
relative L2 in percent, `mean ± sd (n)`.

### 4.1 Homogeneous — the KAN wins

| rank | model | rel-L2 | params |
|---|---|---|---|
| 1 | **`charcoords50` KAN** | **0.0236 ± 0.0129 (11)** | 49,020 |
| 2 | `fourier_matched` PINN | 0.0332 ± 0.0152 (11) | 49,729 |
| 3 | `fourier` PINN | 0.0399 ± 0.0138 (11) | 82,689 |
| 4 | `pykan_wide` plain KAN | 0.0849 ± 0.0201 (5) | 8,600 |
| 5 | `mlp` PINN | 0.2555 ± 0.1232 (5) | 66,561 |
| — | PIKAN, either implementation | ~95 % COLLAPSED | — |

Against the **exact d'Alembert solution** (zero discretisation error):

```
charcoords50   0.0222 ± 0.0139       fourier   0.0401 ± 0.0175
KAN ahead 1.81x   Welch p=0.0154   MWU p=0.0126   paired 9/11   Wilcoxon p=0.0322
```

At **matched parameters** the lead narrows to **1.50x**, and the tests disagree:
Welch p = 0.092, Mann-Whitney p = 0.049, paired Wilcoxon p = 0.0068. The paired
test is the appropriate one for this design (identical seeds and collocation sets),
but this should be reported with the disagreement visible.

### 4.2 Two-layer — the plain MLP wins

| rank | model | rel-L2 |
|---|---|---|
| 1 | **`mlp` PINN** | **0.4318 ± 0.3070 (11)** |
| 2 | `charcoords50` KAN | 0.5984 ± 0.3510 (11) |
| 3 | `fourier` PINN | 0.9579 ± 0.7969 (11) |
| 4 | `pykan_wide` | 1.0340 ± 0.4055 (5) |

**No significant difference** between the best PINN and the KAN (p = 0.250 at
n = 11). An earlier reading of this cell — PINN wins 2.20x — came from five seeds
and did not survive eleven.

Twolayer is the hardest material for every architecture, by 20x. Its transition
width is `w = 0.02` against multilayer's `w = 0.05`, which makes its `E'` **4.2x
larger** despite having one interface instead of six. The characteristic transform
also injects `tau'' = -c'/c^2` into `u_xx` through the chain rule, and that term is
3.7x larger here — so the coordinate change that makes homogeneous separable
actively hurts on a sharp interface.

### 4.3 Multi-layer — a tie

| rank | model | rel-L2 |
|---|---|---|
| 1 | `charcoords50` KAN | 0.0403 ± 0.0114 (11) |
| 2 | `fourier` PINN | 0.0432 ± 0.0115 (11) |
| 3 | `pykan_wide` | 0.2320 ± 0.0984 (5) |
| 4 | `mlp` PINN | 0.6915 ± 0.2701 (11) |

p = 0.571. Six interfaces, and every model does better than on the single sharp
one — confirming that sharpness, not interface count, is what costs.

### 4.4 The same comparison under pure L-BFGS

| material | best PINN | tuned KAN | verdict |
|---|---|---|---|
| homogeneous | `fourier` 0.0281 ± 0.0073 (11) | 0.0313 ± 0.0170 (11) | **tie**, p = 0.577 |
| multilayer | `fourier` 0.0419 ± 0.0146 (11) | 0.0574 ± 0.0375 (11) | PINN nominally |
| twolayer | `fourier`+one-shot-R3 0.4928 (11) | — | — |

**Same networks, same points, different optimiser, different answer.** That is the
subject of §5.

---

## 5. Why Adam→L-BFGS beats pure L-BFGS

### 5.1 What changes

| | pure L-BFGS | Adam(700) → L-BFGS |
|---|---|---|
| Fourier PINN | 0.0281 ± 0.0073 | 0.0399 ± 0.0138 |
| tuned KAN | 0.0313 ± 0.0170 | **0.0236 ± 0.0129** |
| KAN variance vs PINN | **5.44x**, Levene p = 0.031 | **0.87x**, p = 0.73 |
| verdict | tie (p = 0.577) | KAN wins (p = 0.0098) |

The Adam warm-up **helps the KAN and slightly hurts the PINN**. It is not a global
improvement — it is an interaction between optimiser and architecture.

### 5.2 The mechanism

L-BFGS is a quasi-Newton method: it builds a curvature model from the last 100
`(s_k, y_k)` pairs and takes `p = -H·g`. Its first step is scaled as
`t = min(1, 1/||g||_1)·lr`. Two consequences:

1. **It is local.** From a cold random initialisation it descends into whatever
   basin it starts in. It cannot cross a barrier to reach a better one.
2. **It is conservative when gradients are large.** A big `||g||_1` forces a tiny
   first step, so a badly-conditioned start freezes it near the initialisation.

Adam is the opposite: its update is `lr · m/(sqrt(v)+eps)`, bounded by roughly `lr`
per parameter **regardless of gradient magnitude**. Over 700 steps at `lr = 1e-3`
each parameter can travel ~0.7. Adam therefore *explores* — it can leave the basin
it started in.

Measured on `splinekan`, from an identical initialisation:

```
                  loss@init   loss@end    ||dtheta||/||theta_0||
Adam lr=1e-3       3.39e10     5.17e5           0.1608
Adam lr=1e-5       3.39e10     2.68e6           0.0117
L-BFGS 35 epochs   3.39e10     7.89e4           0.0089
```

L-BFGS reaches a **6.5x lower loss while travelling 18x less** — it is far more
efficient locally, and that is exactly why it cannot escape.

### 5.3 Why that helps the KAN more than the PINN

The KAN's basis has a much better *potential* fit to this solution — §7 quantifies
it — but that potential lives in a region of parameter space a cold quasi-Newton
start does not reach. The 233x separable-basis advantage measured in
`BASIS_VS_OPTIMISER.md` bought **nothing** under pure L-BFGS. Add 700 Adam steps and
part of it converts.

The strongest evidence is the variance. Under pure L-BFGS the KAN's seed-to-seed
variance was 5.44x the PINN's (p = 0.031) — the single significant difference
between the families and the main argument against KANs. Under the hybrid it is
0.87x. **The Adam warm-up did not shift a mean; it removed a failure mode** —
seeds that used to land in bad basins now do not.

---

## 6. Why the soft initial condition fails

### 6.1 What was tried and why

Every catastrophic result in this study sits at ~95 %. That is not coincidence:
`u ≡ 0` satisfies the wave equation exactly **and** the absorbing boundary
conditions exactly. Under the hard ansatz, `decay(t)` is 0.011 by t = 0.2, so
driving `N -> 0` gives `u ≈ 0` across the entire scoring window. It is a genuine
global minimum of our loss, at ~95 % error, with a near-zero residual — PIKAN
reaches L-BFGS losses of 1.5e-5 while being 95 % wrong.

A **soft** initial condition should delete that basin: replace the hard ansatz
(`ansatz='none'`, so the network *is* `u`) and add
`ic_loss = mean((u(x,0) - g(x))^2) + mean(u_t(x,0)^2)`.

### 6.2 It does remove the basin — the arithmetic works

```
IC penalty for u == 0, unweighted    :  mean(g^2) = 0.1204
grad-norm assigns w_aux              :  55.6  (vs w_pde = 0.505, a 110x ratio)
IC penalty for u == 0, weighted      :  6.70
converged PDE loss for comparison    :  ~1e-3
```

So the basin goes from *cheaper than converging* to **three orders of magnitude
more expensive**. And it works: models trained this way escape and reach real
solutions.

A caution recorded in the defect log: the first attempt appeared to fail (95.52 %)
because grad-norm rebalancing fires every 500 steps and the test ran 200 — the IC
term sat at weight 1 the whole time, where the basin costs only 0.12 and a
collapsing model simply pays it. Rebalancing every 100 steps fixed it.

### 6.3 But it costs a great deal, and rescues nothing

| model | hard ansatz | soft IC | effect |
|---|---|---|---|
| Fourier PINN | 0.0399 ± 0.0138 | 0.2140 ± 0.1107 | **5.5x worse** |
| tuned KAN | 0.0236 ± 0.0129 | 0.0479 ± 0.0278 | **2.0x worse** |
| PIKAN (pykan) | 94.98 | 100.62 ± 3.02 | still dead |
| PIKAN (ours) | 235.6 | 372.9 ± 321.6 | still dead |
| MLP, Adam-only | 0.7392 | 99.63 | **fails outright, 5/6 runs** |

Three separate failures are visible:

1. **The optimisation problem gets harder.** The hard ansatz removes two constraints
   from the search entirely. Restoring them as penalty terms means the optimiser
   must now trade PDE accuracy against IC accuracy at every step, and the weighting
   between them becomes a hyperparameter with real consequences.
2. **The IC stops being exact.** A guaranteed-correct initial condition is traded
   for one that is approximately correct, adding a new error source at the exact
   place where the solution is most structured.
3. **It does not address why the PIKANs fail.** They stay dead, which is the
   evidence that their collapse was never about the trivial basin being cheap.

**One informative asymmetry.** The KAN degrades 2.0x where the PINN degrades 5.5x,
so under soft constraints the KAN's lead *widens* from 1.69x to **4.47x** (3/3
paired). The headline is not an artefact of the hard ansatz — if anything the hard
ansatz flatters the PINN.

---

## 7. The one mechanism behind every failure

### 7.1 The PDE residual is a cancellation

At a converged solution on homogeneous, measured on the trained `charcoords50`:

```
||u_tt||     = 2942
||u_xx||     = 2942
||residual|| =    0.44
```

The PDE is exactly `u_tt = u_xx`, so the residual **is** the failure of that
cancellation — and the two terms cancel to **1 part in 6,696**. Solving this
equation means getting second derivatives right to `1.5e-4` *relative*.

This reframes what a physics loss is. It is not "fit the function". Fitting `u`
to 0.04 % guarantees nothing, because differentiating twice amplifies error by
`k^2`. Everything below follows.

### 7.2 Why `update_grid` destroys a working model

pykan's `fit()` calls `update_grid` 10 times by default. On a trained model, one
call:

```
              rel-L2      training objective
before        0.0114%       1.930e-05
after         0.2272%       3.683e-01     <- 19,000x worse
```

What it preserves, on 4,000 collocation points:

| quantity | ||before|| | relative change |
|---|---|---|
| u | 1.508e+01 | 0.10 % |
| u_x | 1.863e+02 | 0.59 % |
| **u_xx** | 2.942e+03 | **2.28 %** |
| **u_tt** | 2.942e+03 | **2.29 %** |

`curve2coef` refits coefficients to match **function values**. Nothing constrains
derivatives. A 2.3 % perturbation is **153x coarser** than the 1.5e-4 the
cancellation requires. Predicted blow-up `(sqrt(2)·0.0229·2942)^2/N/L_before` =
47,000x; measured 19,000x — agreement within 2.5x, on the correct side since the
two perturbations are not independent.

L-BFGS then restarts from an objective 19,000x worse and finds the trivial
solution. **8/8 runs collapse, always to ~95 %, never gradually.**

This is not a bug in pykan. Grid adaptation is correct for regression, where only
function values matter. It is silently wrong for any loss built from high-order
derivatives — which means anyone building a PIKAN on pykan's default `fit()` hits
this.

### 7.3 Why Fourier-embedded KANs collapse

Not a representation failure. Fitted supervised to the true solution, at matched
value-fit quality:

| model | fit rel-L2 | ||u_tt|| | ||u_xx|| | ||u_tt - u_xx|| | cancels to |
|---|---|---|---|---|---|
| `fourier16` | 0.0431 % | 3.384e+04 | 4.568e+03 | 3.247e+04 | **1 : 1** |
| `charcoords50` | 0.0415 % | 2.984e+03 | 2.947e+03 | 4.269e+02 | 7 : 1 |

`fourier16` **fits `u` better than the Fourier PINN does** — its basis is excellent.
But its second derivatives are not even the same *magnitude* (off by 7.4x) and
cancel not at all.

Evaluate the real training objective at each of those correct solutions:

```
fourier16      L(correct) = 2.588e+05     L(collapsed) = 1.536e-05
charcoords50   L(correct) = 4.525e+01     L(converged) = 1.930e-05
```

For `fourier16` the **trivial answer is 17 billion times better under our loss than
the right answer**. Restart physics training from the correct solution and it
abandons it in **one epoch**, dropping to 95.29 % with amplitude 0.275.
`charcoords50` restarted the same way recovers: 3.02 % -> 0.236 % -> 0.0160 %, with
amplitude pinned at 1.000.

**The optimiser is not failing. The loss prefers the wrong answer** — because the
only way a basis with noisy second derivatives can make `u_tt - u_xx` small is to
make both small, and that means shrinking `u`.

### 7.4 Why the tuned KAN wins

In characteristic coordinates `(xi, eta) = (tau-t, tau+t)`, the wave operator
becomes `∂_xi ∂_eta`. A KAN layer computes a **sum of univariate functions** of its
inputs, so `F(xi) + G(eta)` — which is the exact solution — is produced with
`∂_xi ∂_eta [F(xi) + G(eta)] = 0` **identically**, for any `F` and `G`.

The cancellation is **structural, not learned**. That is the whole advantage, and it
is why the KAN gets 7:1 cancellation straight out of a value fit with
`||u_tt|| ≈ ||u_xx||` to 1.3 %, while the Fourier-embedded KAN gets 1:1.

It also explains twolayer, where the KAN's advantage disappears: a sharp interface
means `c(x)` varies fast, `tau'' = -c'/c^2` is large, and the chain rule injects
that term into `u_xx`. The coordinate change stops being exact and the structural
cancellation is lost.

### 7.5 The principle, stated once

> A physics-informed loss does not measure how well a network fits the solution.
> It measures how well the network's **high-order derivatives cancel**. Any
> architectural choice — a Fourier embedding, a grid refit, a coordinate change —
> must be judged in derivative space, not value space. `BASIS_VS_OPTIMISER.md`'s
> 233x advantage is a value-space number and is therefore the wrong quantity; the
> right one is the cancellation ratio in §7.3.
