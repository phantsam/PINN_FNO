# Layer-by-Layer Anatomy of Every Architecture

**Problem:** 1D elastic wave equation, homogeneous material, `ρ(x)·u_tt = ∂ₓ(E(x)·u_x)`
**Measurements:** taken from *trained* checkpoints (`homogeneous`, seed 0, plain arm),
pushed through with the identical evaluation batch: 20 snapshots × 512 spatial points
= 10,240 space-time points, `x ∈ [-1, +1]`, `t ∈ [0.05, 1.0]`. All statistics computed
in float64. No retraining, no randomness — these numbers are reproducible from the
checkpoints via `core/forensics.py` and `core/forensics2.py`.

---

## 0. What every model shares

### 0.1 The input

```
x ∈ [-1, +1]   spatial coordinate
t ∈ [0.05, 1]  time
```

As a tensor: `(N, 1)` and `(N, 1)`, with `N = 10,240`.

Singular values of the raw 2-column input matrix `[x, t]` (centred):

```
58.54   29.18
```

Two directions. That is all the information any raw-input model starts from.

### 0.2 The hard-constraint ansatz (applied outside the network, identically for all)

Every architecture outputs a raw field `N(x,t)`. The trainer then forms:

```
u(x,t) = g(x)·decay(t)  +  growth(t)·N(x,t)

  g(x)      = normalised derivative-of-Gaussian, peak-normalised to max|g| = 1,
              σ_g = 0.1, centred at x₀ = 0
  decay(t)  = exp(-½ · (15 t)²)
  growth(t) = tanh²(25 t)
```

This enforces `u(x,0) = g(x)` and `u_t(x,0) = 0` **exactly**, because
`decay(0) = 1`, `decay'(0) = 0`, `growth(0) = 0`, `growth'(0) = 0`.

**Consequence for the analysis:** the quantity the *network* must produce is not `u`,
it is

```
N_target(x,t) = ( u_ref(x,t) − g(x)·decay(t) ) / growth(t)
```

Comparing raw `u` across architectures would credit all of them equally for the
hard-coded IC term. Every "linear probe" number below targets `N_target`.

### 0.3 The two diagnostics used at every layer

**Effective rank** — participation ratio of the singular spectrum of the centred
features:

```
r_eff = (Σ sᵢ)² / Σ sᵢ²
```

`r_eff` counts how many independent directions the representation *actually*
occupies. A 128-wide layer with `r_eff = 4` is a rank-4 bottleneck regardless of
its parameter count.

**Linear probe** — the best rel-L2 a *linear* readout of that layer could achieve:

```
min_w ‖ Φ w − N_target ‖ / ‖ N_target ‖
```

solved exactly by ridge-stabilised normal equations in float64. It answers: *is the
information needed to express the solution already present here, recoverable by a
linear map?* It localises the deficiency to a depth.

---

## 1. FOURIER PINN — `FourierMLP(layers=5, units=128, n_fourier=64, σ_B=10)`

**82,689 trainable parameters. Trained rel-L2 = 0.110 %.**

### 1.1 Parameter inventory

| tensor | shape | count | trainable |
|---|---|---|---|
| `embed.B` | (64, 2) | 128 | **no — fixed buffer** |
| `net.0.weight` / `.bias` | (128,128) / (128,) | 16,384 / 128 | yes |
| `net.2.weight` / `.bias` | (128,128) / (128,) | 16,384 / 128 | yes |
| `net.4.weight` / `.bias` | (128,128) / (128,) | 16,384 / 128 | yes |
| `net.6.weight` / `.bias` | (128,128) / (128,) | 16,384 / 128 | yes |
| `net.8.weight` / `.bias` | (128,128) / (128,) | 16,384 / 128 | yes |
| `net.10.weight` / `.bias` | (1,128) / (1,) | 128 / 1 | yes |

### 1.2 The flow

```
INPUT  (N, 2)  =  [x, t]
  │
  │  ── FourierEmbed ──────────────────────────────────────────────
  │     B ~ N(0, σ_B²) with σ_B = 10, shape (64, 2), REGISTERED AS A
  │     BUFFER so the bandwidth cannot drift during training.
  │     σ_B = 10 was chosen from the measured solution spectrum
  │     (k_peak ≈ 9.4) — DECISIONS.md D4.
  │
  │         p = [x,t] @ Bᵀ                    → (N, 64)
  │         φ = concat[ cos(p), sin(p) ]      → (N, 128)
  ▼
embed  (N, 128)
    observed range     [-1.00, +1.00]      (cos/sin are bounded by construction)
    singular values    178.4  174.0  149.1  147.7  146.9      ← FLAT
    effective rank     66.91
    linear probe       36.183 %
    hi-k energy share  0.191                ← the highest of any layer anywhere
  │
  │  ── net.0: Linear(128 → 128), then tanh ─────────────────────
  ▼
tanh1  (N, 128)
    range              [-0.99, +0.99]
    singular values    169.0  144.3  139.5  135.1  123.9
    effective rank     59.41
    linear probe       29.738 %
    mean |pre-act|     0.543        fraction |z| > 1 : 14.0 %
  │
  │  ── net.2: Linear(128 → 128), then tanh ─────────────────────
  ▼
tanh2  (N, 128)
    range              [-0.98, +0.99]
    singular values    175.9  140.9  131.2  119.9  118.4
    effective rank     49.74
    linear probe       20.876 %
    mean |pre-act|     0.433        fraction |z| > 1 :  6.4 %
  │
  │  ── net.4: Linear(128 → 128), then tanh ─────────────────────
  ▼
tanh3  (N, 128)
    range              [-0.96, +0.98]
    singular values    174.4  142.4  135.3  121.1  111.7
    effective rank     41.91
    linear probe       12.778 %
    mean |pre-act|     0.381        fraction |z| > 1 :  3.6 %
  │
  │  ── net.6: Linear(128 → 128), then tanh ─────────────────────
  ▼
tanh4  (N, 128)
    range              [-0.96, +0.96]
    singular values    163.9  133.4  122.5  115.0   99.4
    effective rank     37.24
    linear probe        8.457 %
    mean |pre-act|     0.333        fraction |z| > 1 :  1.9 %
  │
  │  ── net.8: Linear(128 → 128), then tanh ─────────────────────
  ▼
tanh5  (N, 128)
    range              [-0.96, +0.97]
    singular values    148.7  128.1  120.6  108.9   94.9
    effective rank     34.07
    linear probe        0.068 %      ← features are now LINEARLY ADEQUATE
    mean |pre-act|     0.300        fraction |z| > 1 :  1.3 %
  │
  │  ── net.10: Linear(128 → 1) ───────────────────────────────────
  ▼
OUTPUT  N(x,t)  (N, 1)
    linear probe        0.113 %     ← 1.7× worse than optimal readout (0.068 %)
  │
  │  ── ansatz ──────────────────────────────────────────────────
  ▼
u(x,t) = g(x)·decay(t) + growth(t)·N(x,t)          rel-L2 = 0.110 %
```

### 1.3 What this layer stack is really doing

Two facts read together are decisive:

1. **The embedding's singular spectrum is flat** (178, 174, 149, 148, 147). Random
   Fourier features with an isotropic Gaussian `B` are near-orthogonal by
   construction, so the network is handed **~67 effective dimensions before a
   single weight acts**.

2. **The tanh layers barely bend.** `mean|z|` falls 0.543 → 0.300 through depth and
   only **1.3 %** of pre-activations reach `|z| > 1` at the last layer. In that
   regime `tanh(z) ≈ z`.

Therefore this model is operating as **near-linear regression on a fixed random
Fourier basis**. That is a *good* strategy here, because the target is a
bandlimited travelling wave and the basis was tuned (σ_B = 10) to its measured
band (k_peak ≈ 9.4). The depth is refining a solution that is already almost
linearly available at the input.

---

## 2. PLAIN MLP — `MLP(layers=5, units=128)`

**66,561 trainable parameters. Trained rel-L2 = 0.316 %.**

Identical to the Fourier PINN except the embedding is removed and `net.0` maps
`2 → 128` instead of `128 → 128`. It is the control that isolates the embedding.

### 2.1 Parameter inventory

| tensor | shape | count |
|---|---|---|
| `net.0.weight` / `.bias` | (128,2) / (128,) | 256 / 128 |
| `net.2 … net.8` (4 layers) | (128,128) / (128,) | 4 × 16,512 |
| `net.10.weight` / `.bias` | (1,128) / (1,) | 128 / 1 |

### 2.2 The flow

```
INPUT  (N, 2)  =  [x, t]  raw
    singular values    58.54   29.18            ← only 2 directions exist
  │
  │  ── net.0: Linear(2 → 128), then tanh ──────────────────────
  ▼
tanh1  (N, 128)
    range              [-0.82, +0.84]
    singular values    160.5   71.8    5.9    3.9    1.5   ← collapses after 2
    effective rank      1.96
    linear probe       86.217 %
    mean |pre-act|     0.193        fraction |z| > 1 :  0.1 %
  │
  │  ── net.2: Linear(128 → 128), then tanh ────────────────────
  ▼
tanh2  (N, 128)
    range              [-0.99, +1.00]
    singular values    332.6  151.5   39.8   33.1   12.9
    effective rank      2.64
    linear probe       59.118 %
    mean |pre-act|     0.490        fraction |z| > 1 : 11.5 %
  │
  │  ── net.4 ────────────────────────────────────────────────────
  ▼
tanh3  (N, 128)
    singular values    468.7  230.4   77.5   65.1   40.3
    effective rank      3.49
    linear probe       11.449 %
    mean |pre-act|     0.890        fraction |z| > 1 : 32.7 %
  │
  │  ── net.6 ────────────────────────────────────────────────────
  ▼
tanh4  (N, 128)
    singular values    554.4  282.3  105.2   79.8   49.5
    effective rank      4.05
    linear probe        1.177 %
    mean |pre-act|     1.139        fraction |z| > 1 : 46.7 %   ← peak nonlinearity
  │
  │  ── net.8 ────────────────────────────────────────────────────
  ▼
tanh5  (N, 128)
    singular values    517.7  271.7   99.5   71.4   56.1
    effective rank      4.30
    linear probe        0.059 %      ← linearly adequate, and BETTER than Fourier's
    mean |pre-act|     0.943        fraction |z| > 1 : 36.5 %
  │
  │  ── net.10: Linear(128 → 1) ──────────────────────────────────
  ▼
OUTPUT  N(x,t)      linear probe  0.284 %
                    ← 4.8× WORSE than its own features permit (0.059 %)
```

### 2.3 Reading it

The MLP must **manufacture** everything: it starts from 2 directions and its first
hidden layer has effective rank 1.96. It builds up to only 4.30 by layer five — a
128-wide layer using ~3 % of its width.

It compensates by **using its nonlinearity hard**: `mean|z|` climbs to 1.139 with
46.7 % of units past `|z| > 1`, the opposite of the Fourier PINN.

The most striking row is the last: the features at `tanh5` support a 0.059 %
readout, but the trained output layer delivers 0.284 %. The information is present
and the linear head is not extracting it. (Caveat: the probe targets `u` directly
while training minimises the PDE residual — different objectives, so this is not a
pure inefficiency.)

---

## 3. B-SPLINE KAN — `PyKAN(width=(2,20,20,20,1), grid=5, k=3)`

**8,600 trainable parameters. Trained rel-L2 = 0.227 %.**

### 3.1 What one KAN layer *is*

There are no weight matrices and no fixed activation. Every **edge** `i → j` carries
its own learned univariate function:

```
φ_ij(x_i) = scale_base_ij · silu(x_i)  +  scale_sp_ij · Σ_c coef_ijc · B_c(x_i)

    B_c   : cubic (k = 3) B-spline basis functions
    grid  : 5 intervals, so 5 + k = 8 coefficients per edge
    knots : 12 values spanning [-2.2, +2.2] (grid_range [-1,1] extended by k)

out_j = Σ_i φ_ij(x_i)
```

So the nonlinearity sits **on the edges**, and the node operation is a plain sum.

### 3.2 Parameter inventory

| layer | edges | `coef` | `scale_base` | `scale_sp` | trainable | frozen buffers |
|---|---|---|---|---|---|---|
| `act_fun.0` (2→20) | 40 | (2,20,8) = 320 | 40 | 40 | **400** | `grid` (2,12), `mask` (2,20) |
| `act_fun.1` (20→20) | 400 | (20,20,8) = 3,200 | 400 | 400 | **4,000** | `grid` (20,12), `mask` |
| `act_fun.2` (20→20) | 400 | (20,20,8) = 3,200 | 400 | 400 | **4,000** | `grid` (20,12), `mask` |
| `act_fun.3` (20→1) | 20 | (20,1,8) = 160 | 20 | 20 | **200** | `grid` (20,12), `mask` |
| | | | | | **8,600** | |

Additionally frozen and **not** counted:

- `node_bias_*`, `node_scale_*`, `subnode_bias_*`, `subnode_scale_*` — 164 values,
  inert because `affine_trainable=False`.
- `symbolic_fun.*.affine` — 3,440 values. The symbolic branch never enters
  `forward()` under `symbolic_enabled=False`, but pykan still registers it as
  `requires_grad` leaves. **Left alone it inflates the reported parameter count by
  28.6 % (12,040 vs 8,600) and pads every L-BFGS history vector with permanent
  zeros.** We freeze it explicitly.

### 3.3 The flow

```
INPUT  (N, 2)  =  [x, t]  raw
acts0
    singular values    58.54   29.18
    effective rank      1.80
    linear probe       99.884 %
  │
  │  ── KANLayer 0 :  2 → 20    (40 edges, 400 params) ───────────
  │     each edge: scale_base·silu(x_i) + scale_sp·Bspline(x_i)
  │     MEASURED edge nonlinearity:  mean 0.323, median 0.310
  │     only 7.5 % of edges are near-linear (<5 % deviation)
  ▼
acts1  (N, 20)
    observed range     [-2.03, +3.06]
                       ▲ EXCEEDS the last knot at +2.20.
                       Beyond the final knot the B-spline basis is dead, so those
                       inputs receive ONLY the silu base branch — the spline
                       contribution is silently lost.
    singular values    169.2   98.3   19.0   14.9   11.8   ← collapses after 2
    effective rank      2.75
    linear probe       99.223 %
    hi-k energy share  0.091
  │
  │  ── KANLayer 1 :  20 → 20   (400 edges, 4,000 params) ────────
  │     MEASURED edge nonlinearity:  mean 0.339, median 0.277
  │     9.5 % of edges near-linear
  ▼
acts2  (N, 20)
    observed range     [-1.39, +1.03]     → spans ~55 % of the [-2.2,+2.2] grid
    singular values    128.4   64.2   32.4   23.4   14.4
    effective rank      4.41
    linear probe       64.092 %
    hi-k energy share  0.076              ← LOWEST high-frequency share anywhere
  │
  │  ── KANLayer 2 :  20 → 20   (400 edges, 4,000 params) ────────
  │     MEASURED edge nonlinearity:  mean 0.341, median 0.247
  │     8.0 % of edges near-linear
  ▼
acts3  (N, 20)
    observed range     [-0.79, +0.55]     → spans ~31 % of its grid, ≈3 knots
                       ▲ the deepest hidden layer is resolution-starved: most of
                       its spline coefficients are never reached by any input.
    singular values     67.9   31.6   20.3   13.1    8.0
    effective rank      4.72
    linear probe       26.419 %           ← STILL far from linearly adequate
  │
  │  ── KANLayer 3 :  20 → 1   (20 edges, 200 params) ────────────
  │     MEASURED edge nonlinearity:  mean 0.262, median 0.111
  │     20.0 % of edges near-linear
  │     ▲ this single 200-parameter layer must carry the representation
  │       from 26.42 % down to 0.185 %.
  ▼
OUTPUT  N(x,t)  (N, 1)
    linear probe        0.185 %
  │
  ▼
u(x,t) = g(x)·decay(t) + growth(t)·N(x,t)          rel-L2 = 0.227 %
```

### 3.4 Reading it

**The KAN is not idling.** Edge nonlinearity averages ~0.30 with only 7–10 % of
edges near-linear, so the splines are genuinely bending. This is *not* a network
that has collapsed to an affine map.

**But it only reaches effective rank 4.72.** Starting from 2 directions, composing
univariate splines builds roughly 5 useful directions — against the Fourier
embedding's 67 handed over for free.

**Two concrete, measurable defects:**

- `acts1` overflows its grid (+3.06 vs a +2.20 last knot).
- `acts3` occupies 31 % of its grid — roughly 3 knots — so the deepest layer has
  almost no spline resolution where it actually operates.

Both are exactly what pykan's `update_grid` exists to fix, and our custom L-BFGS
loop never called it (pykan's own `fit()` defaults to `update_grid=True`,
10 updates over the first half of training).

**The work is unevenly distributed.** Both PINNs arrive at linear adequacy by their
last hidden layer (0.068 % / 0.059 %). The KAN's last hidden layer is still at
26.42 %, leaving its final 200-parameter layer to close the gap.

---

## 4. PIRATENET — `PirateNet(blocks=3, units=128, n_fourier=64, σ_B=10)`

**199,811 trainable parameters. Homogeneous rel-L2 = 0.111 % (best of any model).**

### 4.1 Components

**Random Weight Factorisation (RWF).** Every linear map is stored factorised:

```
_RWFLinear:   y = ( s ⊙ V ) x + b        s ~ N(μ=1.0, σ=0.1),  V ~ Xavier-uniform
```

so each output row carries its own learned scale `s_j` alongside the direction `V_j`.

**Gated residual block.** With shared encodings `U`, `V` computed once from the
embedding:

```
f  = tanh(W1 h)          z1 = f ⊙ U + (1−f) ⊙ V
g  = tanh(W2 z1)         z2 = g ⊙ U + (1−g) ⊙ V
out = α · tanh(W3 z2) + (1−α) · h          α initialised to 0
```

`α = 0` at init makes each block an **identity map**, so the network starts as a
shallow model and deepens as training moves `α` away from zero.

### 4.2 Parameter inventory

| group | tensors | count |
|---|---|---|
| `embed.B` | (64,2) buffer | 128 (fixed) |
| `encU`, `encV`, `proj` | 3 × [V (128,128), s (128), bias (128)] | 3 × 16,640 = 49,920 |
| `blocks.0/1/2` | each: W1,W2,W3 × [V, s, bias] + α | 3 × 49,921 = 149,763 |
| `out.weight` | (1,128), **no bias**, zero-initialised | 128 |
| | | **199,811** |

### 4.3 The flow

```
INPUT (N,2)
  │
  ├─ FourierEmbed (σ_B = 10, fixed)  →  φ  (N, 128)
  │
  ├──────────────┬──────────────┬───────────────
  ▼              ▼              ▼
U = tanh(encU φ)  V = tanh(encV φ)   h = tanh(proj φ)      all (N, 128)
  │              │              │
  │              │              ▼
  │              │      ┌── block 0 ──────────────────────────────┐
  └──────────────┴─────►│ f  = tanh(W1 h);  z1 = f⊙U + (1−f)⊙V    │
                        │ g  = tanh(W2 z1); z2 = g⊙U + (1−g)⊙V    │
                        │ h ← α₀·tanh(W3 z2) + (1−α₀)·h           │
                        └─────────────────────────────────────────┘
                                        ▼
                                   ── block 1 ──  (same form, α₁)
                                        ▼
                                   ── block 2 ──  (same form, α₂)
                                        ▼
                        out: Linear(128 → 1), bias-free, W initialised to ZERO
                                        ▼
                                   N(x,t)  (N, 1)
```

### 4.4 The initialisation trap

`out.weight` is zero-initialised **on purpose**, which means the raw network output
is identically zero at init — and therefore `u_tt` from the network is zero and every
backbone parameter is dead. PirateNet is only valid when followed by
`physics_informed_init`, a least-squares fit of that output layer onto `g(x)`:

```
Φ = features(x, t);   W = lstsq(Φ, g(x));   out.weight ← Wᵀ
```

Without it the model trains from a completely dead state. Our diagnostics screen
catches this case explicitly, because under the hard ansatz a dead network **still
shows a healthy `|u_tt| ≈ 215`** — that value comes entirely from the `g·decay`
term. Only the raw output's standard deviation reveals it.

### 4.5 Behaviour

Best model on homogeneous (0.111 %) and multilayer (0.167 %), with the tightest
seed spread of any architecture (sd = 0.001, i.e. 0.6 % of the mean).

**But it inverts on twolayer**: 1.70–2.77 %, the *worst* of all five architectures
on that material, ~25× its own homogeneous number. This is unexplained and is the
main reason the twolayer column is not yet reportable.

---

## 5. WAVELET KAN — `WavKAN(layers=7, units=32)`

**18,720 trainable parameters. Homogeneous rel-L2 = 0.566 % (2 of 3 seeds collapsed).**

### 5.1 What one WavKAN layer is

Mexican-hat (Ricker) wavelet on every edge, with **learnable scale and translation**:

```
z_ij      = ( x_i − trans_ij ) / scale_ij
ψ(z)      = (2 / (√3 · π^{1/4})) · (z² − 1) · exp(−z²/2)
out_j     = Σ_i w_ij · ψ(z_ij)
```

Unlike the B-spline KAN there is no fixed grid — the basis functions move and
rescale themselves. It takes raw `(x,t)`, not the Fourier embedding.

### 5.2 Parameter inventory

| layer | shape of `scale`/`trans`/`w` | params |
|---|---|---|
| `layers.0` (2→32) | (32,2) each | 192 |
| `layers.1 … layers.6` (32→32) | (32,32) each | 6 × 3,072 = 18,432 |
| `layers.7` (32→1) | (1,32) each | 96 |
| | | **18,720** |

### 5.3 The flow

```
INPUT (N,2)
  ▼  layers.0 : 2 → 32     ψ((x−trans)/scale) · w , summed over inputs
h1 (N,32)
  ▼  layers.1 : 32 → 32
h2 (N,32)
  ▼  layers.2 … layers.6   (five more 32 → 32 wavelet layers)
h7 (N,32)
  ▼  layers.7 : 32 → 1
N(x,t) (N,1)
```

### 5.4 The failure mode — traced precisely

WavKAN diverges under plain L-BFGS on several cells. The mechanism was traced to
**float32 overflow inside torch's own line search**, not to our code:

1. WavKAN sits in a near-flat, ill-conditioned region (loss 433 → 418 over 11
   epochs at ~25 objective evaluations each).
2. L-BFGS's unit trial step lands where the wavelet network explodes:
   measured `f₂ = 6.7e18`, `g₂ = 5.2e24` — the Mexican hat has a learnable `scale`
   in a **denominator**, and the PDE residual differentiates it **twice**.
3. `torch/optim/lbfgs.py::_cubic_interpolate` then forms
   `d1 = g1 + g2 − 3(f1−f2)/(x1−x2) ≈ 5.2e24` and `d1**2 ≈ 2.7e49`, which
   **overflows float32** (max 3.4e38) → `inf`.
4. `(g2 + d2 − d1)/(g2 − g1 + 2·d2)` = `inf/inf` = **NaN step size**.
5. NaN is written into all 24 parameter tensors.
6. torch reports the loss at the *start* of the step — a healthy 418.69 — so a
   guard on the returned loss never fires.

The same arithmetic in float64 returns a finite 0.667. Our fix checks the
**parameters** rather than the reported loss:

```python
if not all(torch.isfinite(q).all() for q in model.parameters()):
    diverged = True; break
```

Best-so-far weights are then restored, so a blow-up degrades to a reported number
(100.797 %, epoch 12) rather than a NaN that silently poisons the metrics.

**R3 adaptive resampling rescues WavKAN from trivial collapse 5 times out of 5**
(e.g. homogeneous s1: 95.51 % → 5.20 %; multilayer s0: 97.53 % → 1.42 %), but does
**not** rescue this divergence case.

---

## 6. Side-by-side summary

| | Fourier PINN | MLP | B-spline KAN | PirateNet | WavKAN |
|---|---|---|---|---|---|
| params | 82,689 | 66,561 | **8,600** | 199,811 | 18,720 |
| width | 128 | 128 | 20 | 128 | 32 |
| depth | 5 tanh | 5 tanh | 4 KAN layers | 3 gated blocks | 8 wavelet layers |
| input | 128-d Fourier | raw 2-d | raw 2-d | 128-d Fourier | raw 2-d |
| first-layer rank | **66.91** | 1.96 | 1.80 | — | — |
| last-hidden rank | **34.07** | 4.30 | 4.72 | — | — |
| last-hidden probe | 0.068 % | 0.059 % | 26.42 % | — | — |
| nonlinearity used | **low** (1.3 % \|z\|>1) | high (36.5 %) | ~30 % per edge | — | — |
| rel-L2 (homogeneous) | 0.112 % | 0.259 % | 0.200 % | **0.111 %** | 0.566 %* |

\* two of three seeds collapsed.

---

## 7. The mechanism, stated plainly

The target is a **travelling wave** — near-bandlimited oscillation at
k_peak ≈ 9.4, essentially `F(x − ct)`.

**Random Fourier features turn translation into a phase shift**, which is *linear*
in feature space. The Fourier PINN is handed a flat, 67-effective-dimension basis
tuned to exactly that band, and its subsequent layers barely bend
(`mean|z| = 0.30`, 1.3 % past `|z| > 1`). It succeeds by doing **near-linear
regression in a basis that was chosen for the problem**.

**The KAN gets no such basis.** From a 2-dimensional raw input it must synthesise
oscillation by composing univariate splines, and composition is a poor way to
manufacture high frequency: its interior layers carry the *lowest* high-k energy
share measured (0.076) and reach effective rank 4.72.

The comparison is therefore better described as **"with Fourier features vs
without"** than as "PINN vs KAN":

- **pykan (0.227 %) beats the plain MLP (0.316 %)** on identical raw inputs — the
  KAN is the better raw-input architecture, by 1.4×.
- **The Fourier PINN beats pykan by 2.1×**, and its advantage is the embedding.
- **The KAN cannot be given that embedding** — 3 of 3 Fourier-embedded KAN
  configurations collapse to the trivial solution (94.7 %, 95.7 %, and a
  hand-rolled variant with `max|u_tt| ≈ 5e5` at init), despite each passing the
  initialisation health screen cleanly.

That last point is the strongest available form of the argument. It is not that
KANs are weak learners on this equation — against a like-for-like opponent they
win. It is that the **one intervention that solves this problem is structurally
unavailable to them.**

---

## 8. Standing caveats

1. **Not matched on width or parameters.** Fourier is 128 wide / 82,689 params;
   pykan is 20 wide / 8,600. Effective rank as a *fraction* of width is comparable
   (27 % vs 24 %), so part of the rank gap is simply capacity. The `kan_g100`
   configuration (90,300 params, matched to Fourier's 82,689) is the correct test
   and is still running.
2. **One material, one seed.** All layer statistics here are homogeneous, seed 0,
   plain arm.
3. **`update_grid` was never used during training**, so the grid-overflow at
   `acts1` and the resolution starvation at `acts3` are unremediated in these
   checkpoints. Whether fixing them raises the ceiling is untested.
4. **The linear probe measures information, not achievability** — it says the
   features *support* a given accuracy, not that training would find it.

---

## 9. HAND-ROLLED SPLINE KAN — `SplineKAN(layers=3, units=64, grid_size=10)`

**246,208 parameters. Retained as a control; DO NOT USE for results.**

Our own B-spline KAN, written before pykan was adopted. It differs from pykan in
one structural way: it operates on the **128-d Fourier embedding**, not on raw
`(x,t)`, and it inserts `LayerNorm` + `tanh` between spline layers.

### 9.1 Layer definition

```
_SplineKANLayer(i, o, grid_size=10, order=4):
    n_basis = grid_size + order = 14
    base    (o, i)          kaiming-uniform
    coef    (o, i, 14)      xavier-uniform
    grid    (buffer)        uniform knots on [-1,1], extended by `order` on each side

    forward(x) = Linear(silu(x), base)  +  einsum("bik,oik->bo", Bspline(x), coef)
```

### 9.2 The flow

```
INPUT (N,2)
  ▼  FourierEmbed(64, σ_B=10)                  → (N, 128)
  ▼  _SplineKANLayer(128 → 64)  →  LayerNorm  →  tanh
  ▼  _SplineKANLayer( 64 → 64)  →  LayerNorm  →  tanh
  ▼  _SplineKANLayer( 64 → 64)  →  LayerNorm  →  tanh
  ▼  Linear(64 → 1, bias-free)
N(x,t)
```

### 9.3 Why it is disqualified

Measured at initialisation, against the Fourier PINN's `|u_tt| ≈ 1.8e2`:

```
max|u_tt|  = 9.44e+05        (screen threshold: 1e4)
max|u_xx|  = 7.83e+05
‖grad‖     = 1.85e+10        (screen threshold: 1e8)
loss       = 4.27e+10
```

It fails the initialisation health screen on both the stiffness and the exploding
criteria, and collapses under the physics loss. On the published benchmark
(arXiv:2602.15068) it scored **1.1015 %** where pykan scored **0.1834 %** — a 6×
gap on identical data, which is what motivated switching to the reference
implementation.

**It is kept in the registry because it is a useful negative control**: it proves
the diagnostics can detect a genuinely broken architecture, and it is the first of
the **five independent demonstrations that spline KANs cannot ingest a Fourier
embedding** (see §12.4).

---

## 10. TUNED KAN — `TunedKAN`, the Phase 8 ladder model

The configurable KAN used for every fair-tuning experiment. Same pykan core as §3,
with each knob that Phase 6 left at its library default exposed as an argument.

### 10.1 Knobs and what each fixes

| knob | default in Phase 6 | why it exists |
|---|---|---|
| `grid` | 5 | knot spacing 0.400 vs pulse FWHM 0.236 — splines coarser than the wave packet |
| `k` | 3 | `u_xx` is only C⁰ at k=3; k=5 gives C² |
| `base_fun` | silu | pykan accepts only `'silu' | 'identity' | 'zero'` as strings, so others are installed by post-construction substitution on the model **and every layer** |
| `normalise` | off | `grid_range=[-1,1]` but `t ∈ [0,1]` — half the time knots unreachable |
| `n_fourier` | 0 | gives the KAN the PINN's embedding; `cos/sin ∈ [-1,1]` matches `grid_range` natively |
| `char_coords` | off | `(τ(x)−t, τ(x)+t)`; for constant `c` the solution is `F(ξ)+G(η)`, a sum of univariate functions — natively a Kolmogorov–Arnold form |
| `seed` | hijacked | `MultKAN.__init__` calls `torch.manual_seed(its own seed)`, overriding the trainer's |

### 10.2 The input stage

```
                    ┌── char_coords ─→  (τ(x)−t, τ(x)+t), each affine-mapped to [-1,1]
INPUT (N,2) ────────┼── n_fourier   ─→  [cos(Bx), sin(Bx)]  ∈ [-1,1] natively
                    └── normalise   ─→  x ↦ 2(x−x_min)/(x_max−x_min) − 1
                                        t ↦ 2t/t_max − 1
                             │
                             ▼
                     pykan MultKAN (as §3)
```

The three are mutually exclusive; `char_coords` and `n_fourier` both already land
in `[-1,1]` so they bypass the affine scaler, and constructing with both raises.

### 10.3 The travel-time map (used by `char_coords`)

`τ(x) = ∫dx′/c(x′)`, built as a **quintic Hermite interpolant** matching value,
first and second derivative at 4,097 nodes, evaluated in float64. Node values come
from per-interval **5-point Gauss–Legendre** quadrature.

Two numerical traps had to be cleared, both of which produced plausible-looking
but wrong `τ″` — and `τ″` enters the residual through the chain rule:

- **float32 cancellation** in the quintic basis — error *grew* with node count
  (5.8e-2 at n=4097 → 1.3e-1 at n=16385), the signature of round-off. Fixed with
  an offset formulation (`H0 + H3 ≡ 1`, so the constant drops out of derivatives)
  plus float64 evaluation.
- **composite Simpson advancing odd indices by trapezoid**, leaving adjacent nodes
  inconsistent at O(h³); differentiated twice that gave **first-order** convergence
  for `τ″`.

Final accuracy: `τ′` rel-err **1.5e-7**, `τ″` abs-err **1.9e-6** against a peak of
4.58, `ξ` exactly constant along a rightgoing characteristic (std = 0).

### 10.4 `update_grid_` — implemented, never used in production

Re-fits every layer's spline grid to the activations it actually sees. This is
pykan's own default during `fit()` (`update_grid=True`, 10 updates over the first
half of training); our custom L-BFGS loop never called it. Measured effect of one
call on an untrained net: layer-2 knots contract from `[-2.20, +2.20]` to
`[-0.57, +0.49]` with 3e-3 function drift.

**Status: built and unit-verified (bit-exact no-op at `grid_updates=0`), but not
applied to any reported result.** The grid overflow at `acts1` and the resolution
starvation at `acts3` documented in §3.3 remain unremediated in every checkpoint
analysed here.

---

# PART II — THE STUDY

## 11. The problem, and why it is not the one the literature answers

### 11.1 Our equation

```
PDE   ρ(x)·u_tt = ∂ₓ( E(x)·u_x )          x ∈ [-1, 1],  t ∈ [0, 1]
IC    u(x,0) = g(x)  (derivative-of-Gaussian, σ_g = 0.1, peak-normalised)
      u_t(x,0) = 0
BC    u_t − c·u_x = 0  at x_min      (outgoing / absorbing)
      u_t + c·u_x = 0  at x_max
      c(x) = √(E/ρ)
```

Conservative (divergence) form — **not** `E·u_xx`, which drops the `E′(x)·u_x`
term. Dropping it is a real error: our mutation tests use exactly that deletion to
prove the verification suite detects a wrong equation.

### 11.2 Materials

| name | E(x) | transition width | c range |
|---|---|---|---|
| `Homogeneous` | 1.0 | — | τ_max = 2.000 |
| `TwoLayer` | 1.0 → 1.5, tanh-smoothed at x=0 | w = 0.02 | τ_max = 1.816 |
| `MultiLayer` | 6 layers, 1.0 → 2.5 | w = 0.05 | τ_max = 1.565 |
| `VariableDensity` | ρ ≠ 1 (MMS coverage only) | — | τ_max = 1.639 |

Interfaces are **tanh-smoothed, not discontinuous**. This matters when comparing
against papers claiming B-spline local support helps at material jumps — our
problem does not contain a true jump.

### 11.3 Five ways our problem differs from the KAN-favourable literature

1. **Strong form, not energy form.** [KINN](https://arxiv.org/abs/2406.11045)
   obtains its headline wins largely through the potential-energy functional,
   which contains only **first** derivatives. Our residual is built on `u_xx`. For
   a k=3 spline, `u` is C² so `u_xx` is only C⁰ — piecewise-linear with a kink at
   every knot. The energy form removes that difficulty entirely.
   **There is no energy form for our equation**: the wave action
   `∫∫[½ρu_t² − ½Eu_x²]` is a *saddle point*, not a minimum. Nothing to minimise.
2. **Hyperbolic, not elliptic or steady.** Most KAN-favourable PDE results are on
   smooth, steady, or diffusive problems.
3. **High frequency.** Measured `k_peak ≈ 9.4` (analytic 10.0). The solution is a
   localised pulse, not a smooth global field.
4. **Translation is the dominant structure.** `u ≈ F(x − ct)`. Fourier features
   encode translation as a *phase shift* — linear in feature space. Fixed-grid
   splines must relearn the pulse shape at every location.
5. **Absorbing boundaries and a hard-constraint IC ansatz**, which most benchmark
   problems do not have.

Points 1 and 3–4 are, on the evidence in §12, the reasons our answer differs from
the published ones.

## 12. Methodology

### 12.1 Reference solution

Second-order finite differences, `nx = 512`. Two bugs were fixed relative to the
inherited code:

```python
t  = np.linspace(0.0, T, nt);  dt = t[1] - t[0]       # was dt = T/nt (mislabelled axis)
u1 = u0 + 0.5*dt**2 * L(u0)                            # second-order Taylor start-up
```

Measured convergence order went **1.01 → 2.00**; error at nx=512 went
**4.1e-3 → 6.3e-4**.

### 12.2 Metric

`spacetime_rel_l2` over 20 snapshots at `t ∈ [0.05, 1.0]`, fixed denominator.
A two-sided amplitude guard (`0.30 ≤ ratio ≤ 1.50`) flags trivial collapse — a
one-sided guard was silent on the frozen-IC failure mode.

### 12.3 Training protocol

L-BFGS, `max_iter=20`, `history_size=100`, strong-Wolfe line search, on a **fixed**
Sobol collocation set of 10,000 points; **unnormalised** residual; patience-50 early
stopping; best-loss weights restored. R3 adaptive resampling fires on patience
exhaustion (retain above-mean residual, resample the rest, fresh optimiser).

Three seeds per cell. Phases 6 and 7 total **124 training runs**.

### 12.4 Phase chronology

| phase | what it was | outcome |
|---|---|---|
| 3–4 | Adam-only sweeps | superseded — optimiser is architecture-dependent (MLP 69× better under L-BFGS; WavKAN collapses under it) |
| 5 | first L-BFGS grid | **entirely invalid** — D6 residual normalisation (§13.1) |
| **6** | 5 archs × 3 materials × 3 seeds × {plain, R3} | **90/90 complete** |
| **7** | re-run every capped cell uncapped at 3000 ep, + all pykan reseeded | **34/34 complete** |
| **8** | 15-variant KAN strengthening ladder | 33/45 + 3 multi-seed confirmations |
| capacity | supervised-fit ceiling probe | **abandoned after 4 failed designs** (§13.3) |
| forensics | layer-by-layer analysis of trained models | §1–§5 above |

## 13. Every bug found, and where it came from

### 13.1 Ours

| bug | mechanism | measured effect |
|---|---|---|
| **D6 residual normalisation** | dividing the loss by `residual_scale² ≈ 5e4` shrank gradients by the same factor; torch's L-BFGS picks its first step as `min(1, 1/‖g‖₁)·lr`, so step one overshot | MLP/twolayer **95.59 % → 0.86 %**; WavKAN **92.63 % → 0.88 %** |
| FD reference first-order | `dt = T/nt` and a first-order start-up | order 1.01 → **2.00** |
| causal tolerance underflow | ε=0.1 drove weights to exactly zero; only 3 % of the time domain trained | replaced with the paper's annealed ε ∈ [1e-2, 1e2] |
| R3 never fired | scheduled at epoch 350; runs early-stopped at ~237 | detected because plain and R3 gave **bit-identical** results |
| pykan seed never passed | `MultKAN.__init__` calls `torch.manual_seed(its own seed)` | seed-0 vs seed-2 params differed by **1.49e-08** — i.e. all three "seeds" were one seed |
| stale checkpoints | 20–80× stale, mistaken for a real result | corrected by retraining |

### 13.2 Upstream — pykan

| bug | mechanism | measured effect |
|---|---|---|
| **`curve2coef` NaN on CUDA** | `torch.linalg.lstsq`'s only CUDA driver is `gels`, which assumes full rank and **returns NaN without raising**; KAN activations contract with depth so layers 1–3 are rank-deficient (7/8, 4/8, 4/8) | **every grid refinement past layer 0 produced NaN** — the KAN paper's own accuracy mechanism was unusable |
| **`curve2coef` unstable at k=5** | order-5 B-splines have heavier basis overlap; `gels` fails even at nominal full rank | LSQ residual **48.22** vs the ridge solve's **0.00197** — a 24,500× error |
| device bookkeeping | `MultKAN`/`KANLayer` cache `.device` and define their own `.to()`, which `nn.Module.to()` never calls | `refine()` built grids on CPU → device mismatch |
| phantom parameters | symbolic branch registered as `requires_grad` under `symbolic_enabled=False` | reported params **12,040 vs 8,600 real** (28.6 % inflation) |

Both `curve2coef` defects are fixed by a ridge-regularised normal-equation solve —
which is the fallback pykan itself left commented out. Verified **identical to
pykan wherever pykan is valid** (residuals matching to 6 decimals at k=3).

### 13.3 Upstream — torch

| bug | mechanism | measured effect |
|---|---|---|
| **L-BFGS fp32 overflow** | `_cubic_interpolate` forms `d1² ≈ 2.7e49`, overflowing float32 (max 3.4e38) → `inf/inf` → **NaN step size** | WavKAN parameters NaN-ed while the *reported* loss read a healthy 418.69 |
| L-BFGS reports the pre-step loss | so a NaN born in the line search is invisible in the return value | guard had to check **parameters**, not loss |
| `tolerance_grad` is absolute | default 1e-7; the supervised-fit loss lives at ~4e-7 | LBFGS returned at its first inner iteration; loss froze **bit-exactly** for 200+ steps |

### 13.4 The capacity experiment — four failed designs, then abandoned

| attempt | flaw | symptom |
|---|---|---|
| 1 | `tolerance_grad` absolute | never moved; results 2× pessimistic |
| 2 | float32 precision floor | gradient 3.9e-07 = rounding noise; stalled at 0.234 % |
| 3 | fit set ≠ eval set | optimised a different objective than the metric; not a bound |
| 4 | fp64 + Adam warm + 20k epochs, fit == eval set | **still** `CAP-NOT-CONVERGED`; MLP and KAN could not match their own PDE-trained accuracy |

**Why it was abandoned.** The invariant `E_sup ≤ E_pde` holds by definition — a
minimum cannot exceed a value already attained — but only if the supervised search
finds the **global** minimum. It doesn't; pointwise MSE on an oscillatory target is
a worse-conditioned landscape than the PDE residual, so cold-start fitting lands in
poorer minima. The experiment can therefore *prove* "optimisation headroom exists"
but can **never prove** "representation-limited" — which is the direction we
needed. The forensics (§1–§7) answer the mechanism question with direct
measurements instead.

## 14. Everything we tried to make the KAN win

Fifteen variants, 33 screen cells + 3 multi-seed confirmations.

### 14.1 Results, all cells

| variant | homogeneous | twolayer | multilayer |
|---|---|---|---|
| baseline (grid5) | 0.2176 | 1.2935 | 0.5697 ᶜ |
| normalise | 0.1692 | **0.9851** | — |
| **grid20** | **0.1510** ᶜ | 1.0957 ᶜ | 0.4915 ᶜ |
| grid50 | **66.68** ᶜ ⚠ | **66.77** ᶜ ⚠ | — |
| refine (5→10→20→50) | 0.1943 | 1.3099 | 2.7634 ᶜ |
| k5 | **0.1590** | **98.42** ⚠ | — |
| tanh | 0.1734 ᶜ | — | 0.2536 ᶜ / 0.1886 / 0.4272 |
| sin (ω₀=1) | 0.1521 ᶜ | 1.2123 ᶜ | — |
| gelu | 0.1907 ᶜ | — | 0.3562 ᶜ |
| relu | **93.64** ✗ | **96.31** ✗ | — |
| fourier8 | **94.73** ✗ | — | **96.52** ✗ |
| fourier16 | **95.14** ✗ | **95.70** ✗ | — |
| charcoords | 0.1867 | — | 0.6213 |
| **charcoords50** | **0.1089** | 1.6642 | — |
| best_combo | 0.3346 | — | 0.1936 / **9.9186** |

ᶜ = hit the epoch cap ⚠ = near-collapse ✗ = collapse to trivial

### 14.2 What worked, what didn't

**Worked:**
- **`normalise`** — 0.2176 → 0.1692, a **22 % gain at zero parameter cost**, purely
  from not wasting half the time knots.
- **`grid20`** — a further gain to 0.1510 on homogeneous. Total fair-tuning
  improvement **0.218 → 0.151, 31 %**.

**Failed, robustly (multiple materials):**
- **Fourier embedding — 5 of 5 collapses** across three materials, two feature
  widths, and two independent implementations (§9). The single most robust finding
  in the study.
- `grid50` — 66.68 / 66.77, consistent.
- `relu` — 93.6 / 96.3, consistent, and predicted: its first derivative jumps at
  every kink so `u_xx` is Dirac-singular there.

**Material-dependent (do not generalise):**
- `k5` — **0.1590 on homogeneous** (converged, one of the best KAN results) but
  **98.42 on twolayer**.
- `tanh` vs `silu` — silu wins on homogeneous (0.1510 vs 0.1734), tanh wins on
  multilayer (0.2536 vs 0.4915).
- `grid20` — a large gain on homogeneous, but *worse* than grid-5-with-normalise on
  twolayer.

**Did not survive multi-seed confirmation:**
- `best_combo` on multilayer: **0.1936 → 9.9186**, a **51× swing**.
- `tanh` on multilayer: 0.2536 / 0.1886 / 0.4272 → mean **0.290 ± 0.122** (42 %
  spread); the apparent 1.9× advantage over silu evaporated.

**Inconclusive:**
- `sin` at ω₀ = 1 ties silu (0.1521 vs 0.1510). **This is not a SIREN test** —
  SIREN needs ω₀ ≈ 30 and its own initialisation, and our k_peak ≈ 9.4 means ω₀=1
  is ~10× too low. Reported as untested, not as a negative.
- `charcoords` — the hard ansatz `u = g·decay + growth·N` means the network learns
  `N = (u − g·decay)/growth`, which is **not** additively separable in `(ξ,η)` even
  when `u` is. So this rung did not test its own hypothesis.

## 15. Corrections and retractions

Recorded because each was asserted before it was checked.

| claim | status | what actually was true |
|---|---|---|
| "KAN's L-BFGS never ran" | **wrong** | the typo was in dead code |
| "rak is terrible on heterogeneous (66–128 %)" | **wrong** | stale checkpoints; retraining gave 0.85 / 3.71 % |
| "MLP cannot represent a wavefront crossing an interface (spectral bias)" | **wrong** | it was the D6 normalisation bug; MLP reaches 0.86 % |
| "the KAN arm is void" | **overstated** | only the hand-rolled spline arm; WavKAN worked |
| "cap effect on pykan is 20–60 %" | **wrong** | those cells were *also* reseeded — cap and reseed confounded |
| "MLP's supervised fit is limited by spectral bias" | **wrong** | it had simply hit the epoch cap |
| "capacity shows the KAN is representation-limited" | **withdrawn** | the numbers were float32-limited |
| "k=5 collapses" | **wrong** | 0.1590 % on homogeneous; material-dependent |
| "characteristic coordinates didn't pay off" | **wrong** | `charcoords50` reached 0.1089 % |
| "tanh beats silu 1.9× on multilayer" | **wrong** | evaporated at 3 seeds |

The recurring failure mode is **generalising from a single seed or a single
material.** Three such findings collapsed within one hour of multi-seed checking.

## 16. Results

### 16.1 Phase 6 + 7, mean rel-L2 % over 3 seeds

| arch | params | homog plain | homog R3 | two plain | two R3 | multi plain | multi R3 |
|---|---|---|---|---|---|---|---|
| PirateNet | 199,811 | **0.111** ±0.001 | 0.117 | 2.768 | 2.964 | 0.167 | 0.176 |
| Fourier | 82,689 | **0.112** ±0.002 | 0.114 | 0.907 | **0.394** | **0.162** ±0.004 | 0.173 |
| MLP | 66,561 | 0.259 ±0.079 | 0.231 | **0.535** | 0.518 | 0.474 | 0.473 |
| pykan | 8,600 | 0.200 ±0.027 | 0.189 | 1.168 | **0.662** | 0.328 | 0.332 |
| WavKAN | 18,720 | 0.566* | 2.288 | 3.347* | 3.422* | 3.715* | 2.401 |

\* one or more seeds collapsed.

### 16.2 Verdict

| material | best PINN | best KAN | ratio |
|---|---|---|---|
| homogeneous | **0.111** | 0.189 | **1.70×** |
| twolayer | **0.394** | 0.662 | **1.68×** |
| multilayer | **0.162** | 0.328 | **2.03×** |

With the tuned ladder configurations, homogeneous narrows to ~**1.35×**
(grid20 0.1510 / sin 0.1521 / k5 0.1590 — three independent rungs clustering).

**No 3-seed comparison anywhere in this study has a KAN ahead of the best PINN.**

### 16.3 Reliability, not just accuracy

| arch | homogeneous sd / mean |
|---|---|
| PirateNet | **0.6 %** |
| Fourier | 1.5 % |
| pykan | **13.5 %** |
| MLP | 30.5 % |

pykan's seed variance is **16× the Fourier PINN's**. Any architecture gap smaller
than ~14 % is inside the KAN's own noise.

## 17. Conclusion

For the **1D elastic wave equation in strong form**, with a solution that is a
high-frequency travelling pulse (k_peak ≈ 9.4), a Fourier-featured PINN
outperforms B-spline KANs by **1.7–2.0×** across homogeneous, two-layer and
six-layer materials.

**The mechanism is the input representation, not the network family.** Random
Fourier features turn translation into a phase shift, which is *linear* in feature
space: the embedding hands the PINN a flat, **67-effective-dimension** basis before
any weight acts, and its tanh layers then barely bend (**1.3 %** of units past
|z|>1). It succeeds by doing near-linear regression in a basis tuned to the
problem's measured band.

The KAN starts from 2 directions and must synthesise oscillation by composing
univariate splines. Its edges *are* genuinely nonlinear (~30 % per edge — it is
working, not idling), yet it reaches only **effective rank 4.72** and carries the
lowest high-frequency content measured (0.076).

**Against a like-for-like opponent the KAN wins**: pykan **0.227 %** beats the plain
MLP's **0.316 %** on identical raw inputs, with 8× fewer parameters. The PINN's
advantage is the embedding — and **the KAN cannot be given that embedding**, failing
in 5 of 5 attempts.

That is the defensible form of the claim. It is scoped to this problem class, names
the mechanism, and survives the counter-examples — including our own reproduction of
arXiv:2602.15068, where the same pykan code **beats** the PINN 0.1834 % vs 0.5891 %.

## 18. Open questions

1. **`charcoords50` at 0.1089 % on homogeneous** — below every PINN, converged, but
   **one seed**, and from the grid50+k5 family whose sibling swung 51×.
   Confirmation running.
2. **`update_grid` has never been used** in a production run (§10.4). The measured
   grid overflow and resolution starvation are unremediated.
3. **PirateNet's twolayer inversion** — best on two materials (0.111, 0.167), worst
   of all five on twolayer (2.768), a 25× degradation. Unexplained.
4. **Representation vs optimisation** remains formally unresolved; the capacity
   probe cannot settle it (§13.4) and the forensics answer it only by evidence.
5. **Tier-5 KAN families** — Chebyshev, Jacobi, RBF/FastKAN, FourierKAN — untested.
6. **A proper SIREN rung** (ω₀ ≈ 10–30 with SIREN initialisation) — untested.
