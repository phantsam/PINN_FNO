# Every Model, Layer by Layer — and Why They Succeed or Fail

A complete architectural and mechanistic reference for the PINN-vs-KAN study on
the 1-D elastic wave equation.

**This document assumes no mathematics beyond secondary school.** Part 0 builds
every concept the rest of it uses — derivatives, the notation, what a differential
equation is, what a Gaussian is, what a norm is, what frequency means. Nothing is
used before it is defined. If you already know PINNs and KANs, start at §8.

Companion documents: `BUGS_AND_CORRECTIONS.md` (every defect and retraction),
`ARCHITECTURES.md` (phase chronology), `DECISIONS.md` (frozen spec).

---

## Reading guide

| part | sections | what it covers | assumes |
|---|---|---|---|
| **0** | 1–9 | **the mathematics**, from the ground up | school algebra |
| **A** | 10–12 | the physics problem — the rod, the equation, the materials | Part 0 |
| **B** | 13–16 | the shared pipeline — points, the ansatz, the loss | Part A |
| **C** | 17–19 | neural networks, then **how a KAN actually works** | Part B |
| **D** | 20 | each architecture, layer by layer, every dimension | Part C |
| **E** | 21–26 | results, the optimiser finding, the mechanism | — |

---

# PART 0 — THE MATHEMATICS, FROM THE GROUND UP

---

## 1. Functions

A **function** is a rule that turns input numbers into an output number.

`f(x) = x²` takes one number and returns its square. `f(3) = 9`.

A function can take **two** inputs: `h(x, y) = x + y²`. Then `h(2, 3) = 11`.

Our whole problem is about one particular two-input function, written `u(x, t)`.
You give it a **place** and a **time**; it gives you back **how far the rod is bent
at that place, at that moment**. Finding that function *is* the problem.

A two-input function is a **surface**. Imagine `x` running left-right, `t` running
away from you, and the function's value as height. `u(x,t)` is a landscape over the
space-time rectangle, and we are trying to work out its shape.

---

## 2. Derivatives — rate of change

### 2.1 The first derivative

The **derivative** of a function tells you **how fast it is changing**.

If `f(x)` is your position at time `x`, then `f'(x)` is your speed. If you are at
km 10 at 1pm and km 70 at 2pm, your derivative is 60 km/h.

Formally: the derivative at a point is the **slope of the curve** there. Flat
means derivative 0. Steeply rising means large positive derivative. Falling means
negative.

The two facts we use constantly:

- **at a peak or a trough, the derivative is zero** (the curve is momentarily flat)
- **the derivative of `exp(a·x)` is `a·exp(a·x)`** — exponentials differentiate into
  themselves times a constant

### 2.2 The second derivative

The **second derivative** is the derivative *of the derivative* — how fast the rate
of change is itself changing.

Position → derivative is velocity → second derivative is **acceleration**.

Geometrically it is **curvature**: how sharply the curve bends. A straight line has
second derivative 0 no matter how steep. A tight bend has a large second derivative.

**This matters enormously here, and is the single most important idea in this
document.** Our physics equation is built from second derivatives, and the entire
study turns on the fact that *a function can be very close to correct while its
second derivative is wildly wrong.*

A concrete illustration. Take the true function `f(x) = 0` and an approximation
`g(x) = 0.001·sin(1000x)`.

- The **values** differ by at most 0.001 — a superb approximation.
- The **second derivative** of `g` is `-1000·sin(1000x)`, which reaches 1000, while
  the truth is 0.

A 0.001 error in value became a 1000 error in curvature — a factor of a million.
Differentiating twice multiplies error by roughly the **square of the frequency**
of whatever wiggle you added. Hold onto this; §26 is entirely about it.

---

## 3. Partial derivatives and the `u_x`, `u_t` notation

When a function has two inputs, you can ask about the rate of change **in each
direction separately**. These are **partial derivatives**.

For `u(x, t)`:

| symbol | full name | plain meaning |
|---|---|---|
| `u_x` | ∂u/∂x | how `u` changes as you move **along the rod**, holding time still |
| `u_t` | ∂u/∂t | how `u` changes as **time passes**, holding position still |
| `u_xx` | ∂²u/∂x² | how `u_x` changes as you move along the rod — the rod's **curvature** |
| `u_tt` | ∂²u/∂t² | how `u_t` changes over time — the **acceleration** of that point |

The subscript names the variable you differentiate with respect to; repeating it
means do it twice.

Physically, for our rod:

- `u_x` is the local **stretch** — how much more the rod is displaced just to the
  right of a point than just to the left. If `u_x = 0` the rod is not stretched there.
- `u_t` is the **velocity** of that piece of rod.
- `u_tt` is its **acceleration** — which by Newton's law is what forces cause.
- `u_xx` is the **curvature** of the rod's shape.

---

## 4. Differential equations

An ordinary equation like `x² = 9` asks: *which number satisfies this?* Answer: 3
or −3.

A **differential equation** asks: *which **function** satisfies this?* — where the
equation relates the function to its own derivatives.

Example: `f'(x) = f(x)` asks for a function that is its own rate of change. The
answer is `f(x) = e^x` (times any constant). The unknown is a whole function, not a
number.

- An **ODE** (ordinary differential equation) involves a function of **one**
  variable.
- A **PDE** (partial differential equation) involves a function of **several**
  variables and its partial derivatives. Ours is a PDE, because `u` depends on both
  `x` and `t`.

**A differential equation alone is not enough.** `f'(x) = f(x)` is satisfied by
`e^x`, `5e^x`, `-2e^x` — infinitely many functions. To pick one out you need extra
information: an **initial condition** such as `f(0) = 1`, which selects exactly
`e^x`.

Our problem needs the same: the PDE, plus what the rod looked like at `t = 0`, plus
what happens at the two ends. §12 and §14 are those conditions.

---

## 5. The Gaussian, and why we use its derivative

This section exists because the previous version of this document wrote "`g(x)` =
derivative of a Gaussian, peak-normalised, `sigma_g = 0.1`" and explained none of it.

### 5.1 The Gaussian

The **Gaussian** — the bell curve — is

```
G(x)  =  exp( -x² / (2·sigma²) )
```

It equals 1 at `x = 0` and falls away smoothly and symmetrically on both sides,
never quite reaching zero but becoming negligible fast.

`sigma` (the Greek letter *sigma*) is the **width parameter**. Small `sigma` = a
narrow spike; large `sigma` = a broad hump. With our `sigma = 0.1`:

```
  G(0.00) = 1.000000
  G(0.05) = 0.882497
  G(0.10) = 0.606531      <- at x = sigma, the value is exp(-1/2)
  G(0.20) = 0.135335
  G(0.30) = 0.011109      <- essentially gone by three sigma
```

The Gaussian is the standard choice for a localised bump anywhere in physics and
engineering, for a reason that matters to us: **it is infinitely differentiable and
its derivatives stay smooth and localised.** A square pulse would have corners, and
corners have infinite curvature — which, given §2.2, would be fatal here.

### 5.2 Why the *derivative* of a Gaussian, not the Gaussian itself

Differentiate it:

```
G'(x)  =  -( x / sigma² ) · exp( -x² / (2·sigma²) )
```

This is an **odd** function: one positive lobe and one negative lobe, and it
integrates to exactly zero. Its extremes sit where the second derivative vanishes,
which works out to `x = ±sigma`.

Three reasons this shape rather than a plain bump:

1. **Zero net displacement.** The positive and negative lobes cancel exactly. A
   plain Gaussian would displace the rod's material in one direction overall — a
   net translation on top of the wave, which is not what a pluck does.
2. **It is what a physical disturbance looks like.** Push a rod at one point and
   you compress one side while stretching the other. That is one lobe up, one down.
3. **It has a clean frequency content with a well-defined peak** (§7), which is
   exactly what we need to configure the Fourier PINN honestly rather than by
   guesswork.

### 5.3 "Peak-normalised so that max|g| = 1"

`G'` is not conveniently sized. Its largest magnitude is at `x = ±sigma`:

```
|G'(-sigma)|  =  (1/sigma)·exp(-1/2)  =  10 × 0.606531  =  6.065307
```

With `sigma = 0.1` the raw derivative peaks at about **6.07**, which is an
arbitrary number that would leak into every other quantity in the study. So we
divide the whole function by it:

```
g(x)  =  G'(x) / 6.065307
```

Now `max|g| = 1` exactly. "Peak-normalised" simply means *divided by its own
largest value so the peak is 1*. It makes `u` a quantity of order 1, so a
"0.02 % error" means the same thing everywhere and no hidden scale factors appear.

One subtlety worth recording: we normalise by the **analytic** peak `(1/σ)e^{-1/2}`,
not by the largest sampled value. If you used the sampled maximum, then a network
drawing *random* batches would get a slightly different normalisation every step —
the initial condition itself would wobble from step to step.

### 5.4 The pulse, in numbers

```
  g(-0.40) = +0.002212        g(+0.00) =  0.000000
  g(-0.30) = +0.054947        g(+0.10) = -1.000000   <- negative peak
  g(-0.20) = +0.446260        g(+0.20) = -0.446260
  g(-0.10) = +1.000000  <- positive peak
                              g(+0.30) = -0.054947
                              g(+0.40) = -0.002212
```

| property | value | how it is obtained |
|---|---|---|
| peaks | `x = ±sigma = ±0.1` | where `G''= 0` |
| peak values | `+1` and `−1` | by construction, §5.3 |
| zero crossing | `x = 0` | the odd symmetry |
| width of one lobe (FWHM) | **0.16** | measured |
| where it becomes negligible | `|g| < 0.01` beyond `|x| > 0.357` | measured |

So: **a small double-lobed wiggle occupying roughly `x ∈ [-0.36, +0.36]`, and
essentially nothing outside it.** The rod is flat everywhere else.

### 5.5 Why `sigma_g = 0.1` specifically

It is a compromise between two failures:

- **Too wide** (say `sigma = 0.5`) and the pulse fills the whole rod. It never
  becomes two cleanly separated travelling waves, so the problem stops testing
  wave propagation.
- **Too narrow** (say `sigma = 0.01`) and the pulse contains very high frequencies.
  The reference solver would need an enormous grid, and every network would fail
  for reasons of resolution rather than architecture.

At `sigma = 0.1` the pulse occupies about a fifth of the rod, splits cleanly into
two halves that separate well before the run ends, and is comfortably resolved by
the reference. It is a **frozen decision** (`DECISIONS.md`), identical for every
architecture, so it cannot favour anyone.

---

## 6. Vectors, norms, and measuring error

### 6.1 Vectors

A **vector** is just an ordered list of numbers: `[3, -1, 4]`. In this study a
vector is usually the model's predictions at many points at once.

A **tensor** is the same idea generalised: a list (1-D), a table (2-D), a stack of
tables (3-D). PyTorch calls everything a tensor. A shape written `(10000, 1)` means
10,000 rows, 1 column — a tall thin column of 10,000 numbers.

**float32** means each number is stored in 32 bits, giving about **7 decimal digits**
of precision. That is normally plenty; §26 explains one place where it very nearly
was not.

### 6.2 The norm — "how big is this list?"

The **norm**, written `||v||`, is the length of a vector, by Pythagoras extended to
any number of dimensions:

```
||v||  =  sqrt( v1² + v2² + v3² + ... )
```

For `[3, 4]` it is `sqrt(9+16) = 5`. It answers: *how big is this collection of
numbers, as one number?*

### 6.3 How we score a model

Our model produces 81,920 predicted displacements; the reference provides 81,920
true ones. The error metric is the **relative L2 error**:

```
                 || u_model − u_reference ||
rel-L2  =  100 · ─────────────────────────────  percent
                     || u_reference ||
```

In words: *take the difference at every one of the 81,920 points, measure how big
that whole list of differences is, and express it as a percentage of how big the
true answer is.*

"L2" just names this particular norm (squares and a square root); an "L1" norm
would add absolute values instead.

| reading | meaning |
|---|---|
| 0.02 % | the error is one five-thousandth the size of the solution — excellent |
| 1 % | visibly wrong but recognisably the right shape |
| 95 % | the model produces essentially nothing |
| >100 % | the model is *worse than predicting zero everywhere* |

The last row is not hypothetical — one of our architectures scores 235 % (§20.8).

### 6.4 Mean and standard deviation

For a list of numbers, the **mean** is the average, and the **standard deviation**
(`sd`) measures how spread out they are around it. Written `0.0236 ± 0.0129`, the
first number is the mean and the second the sd.

Two models can have the same mean and very different sd, and **this study's most
important early finding was about sd, not mean** (§24).

---

## 7. Frequency, and what a "spectrum" is

### 7.1 The idea

A **sine wave** `sin(k·x)` wiggles regularly. The number `k` is its **frequency** —
large `k` means many wiggles packed into a short distance.

**Fourier's theorem**: essentially any function can be written as a sum of sine and
cosine waves of different frequencies. The recipe — how much of each frequency you
need — is that function's **spectrum**.

A slowly varying function needs mostly low frequencies. A function with sharp
features needs high ones. This is exactly how an audio equaliser thinks about
sound: bass is low `k`, treble is high `k`.

### 7.2 The spectrum of our pulse, and where `sigma_B = 10` comes from

Our `g(x)` is a derivative-of-Gaussian, and its spectrum has a clean closed form:

```
|ĝ(k)|  proportional to  k · exp( -k²·sigma² / 2 )
```

That expression is zero at `k = 0`, rises, then falls — so it has a single clear
**peak**, and calculus puts that peak at

```
k_peak  =  1 / sigma  =  1 / 0.1  =  10
```

Measured numerically on our actual pulse: **k_peak = 9.425**, matching the theory.

**This is where the Fourier PINN's one free parameter comes from.** That
architecture (§20.3) multiplies the input by random frequencies drawn from a
distribution of width `sigma_B`. We set `sigma_B = 10` — not by tuning against
results, but because **the pulse's own dominant frequency is 1/sigma_g = 10.**

`sigma_g = 0.1` in the initial condition and `sigma_B = 10` in the network are
therefore *the same fact stated twice*. That connection was never made explicit
before, and it is the reason the PINN's bandwidth is a principled choice rather
than a tuned one (`DECISIONS.md` D4). It is held identical for every architecture
that uses an embedding.

### 7.3 Why frequency governs everything later

From §2.2: differentiating `sin(k·x)` twice gives `-k²·sin(k·x)`. **Every
differentiation multiplies by the frequency.** With `k ≈ 10`, and remembering that
real frequency in the exponent carries a factor `2π`:

```
one derivative :  × 2πk ≈ 63
two derivatives:  × (2πk)² ≈ 3,900
```

For the highest frequencies the embedding contains (`k` up to ~30), two derivatives
multiply by **≈ 3.5 × 10⁴**.

So a Fourier embedding is a bargain with a hidden cost: it makes the *function*
easy to represent and its *second derivatives* noisy. Our loss is built entirely
from second derivatives. §26 collects the bill.

---

## 8. What a "basis" is

A **basis** is a set of building blocks you combine to make other functions.

To build any polynomial you can use `1, x, x², x³, ...` — that is a basis. To build
periodic functions you can use sines and cosines — the Fourier basis. To build a
smooth curve out of local pieces you can use B-splines (§18.4).

Choosing a basis is choosing *what kind of shapes are cheap to make*. The Fourier
basis makes waves cheap and sharp corners expensive. A spline basis makes local
bumps cheap.

**Every architecture in this study is, underneath, a choice of basis**, and the
whole question is which basis suits this particular solution. §26 shows that the
usual way of judging that — how well the basis fits the function — is the wrong
test.

---

## 9. Randomness, seeds, and why one run means nothing

Training starts from **random** numbers, and the sample points are drawn
**randomly**. A **seed** is the number that fixes those random draws, so the same
seed reproduces the same run exactly.

Change the seed and you get a genuinely different training run of the *same
architecture on the same problem* — differing only in luck. The spread across seeds
tells you how much of a result is the model and how much is chance.

**This project learned that the hard way: five separate findings dissolved when the
number of seeds was raised.** Every headline number here uses **11 seeds**, and no
number is quoted without its `n`.

---

# PART A — THE PHYSICS PROBLEM

---

## 10. The rod, and what we are trying to find

### 10.1 The physical picture

A long thin elastic rod lies along a line — think of a metal bar, or a guitar
string held straight. At one moment you deform it slightly near the middle and let
go. The deformation does not stay put: it travels outward along the rod in both
directions. That travelling deformation is a **wave**.

We want to know, for **every position** along the rod and **every moment** after
release, how far the rod is displaced from its resting position.

### 10.2 The symbols

| symbol | name | meaning | range here |
|---|---|---|---|
| `x` | position | where along the rod | `-1` to `+1` |
| `t` | time | how long since release | `0` to `1` |
| `u(x,t)` | displacement | how far the rod is bent there, then | about `-1` to `+1` |
| `E(x)` | stiffness (Young's modulus) | how hard the material resists stretching | `1.0` to `2.5` |
| `rho(x)` | density | how heavy the material is | `1.0` |
| `c(x)` | wave speed | how fast waves travel there | `1.0` to `1.58` |

`u(0.40, 0.30) = -0.500000` means: *at position 0.40 along the rod, 0.30 time units
after release, the rod is displaced by −0.5 units.* That is a real value from this
problem, derived in §12.3.

### 10.3 "Non-dimensional units"

The rod is not literally 2 metres and the run is not literally 1 second. We rescale
so that the reference stiffness and density are both 1, which makes the reference
wave speed exactly 1.

Why bother: it removes units from every equation, makes "a wave crosses the domain
in 2 time units" true by construction, and means numbers like `0.02 %` are
comparable across materials. It is standard practice and is a frozen decision
(`DECISIONS.md` D1).

---

## 11. The equation, derived rather than asserted

### 11.1 Building it from Newton's second law

Newton: **force = mass × acceleration**. Apply it to a tiny slice of the rod
between `x` and `x + dx`.

**Mass** of the slice: density times length, `rho(x)·dx`.

**Acceleration** of the slice: `u_tt` — the second derivative of displacement with
respect to time (§3).

**Force** on the slice. The rod pulls on the slice from the left and from the right.
The internal pulling force is called the **stress**, and for an elastic material it
is proportional to how stretched the material is:

```
stress(x)  =  E(x) · u_x
```

That is Hooke's law — the more you stretch it (`u_x`, §3), the harder it pulls back,
with stiffness `E(x)` as the constant of proportionality.

Now the crucial step. The slice is pulled right by the stress at `x + dx` and pulled
left by the stress at `x`. The **net** force is the difference:

```
net force  =  stress(x + dx) − stress(x)  ≈  d/dx[ stress ] · dx
```

**If the stress is the same on both sides, the two pulls cancel exactly and the
slice does not accelerate at all.** Only a *change* in stress along the rod produces
motion. That is why a derivative appears on the force side.

Put the three pieces together and cancel `dx`:

```
rho(x) · u_tt  =  d/dx ( E(x) · u_x )
```

That is the equation. It is Newton's second law written for every point of a
continuous body at once.

### 11.2 Why the "conservative" form, and not `u_tt = c²·u_xx`

Expand the right-hand side with the product rule:

```
d/dx( E·u_x )  =  E·u_xx  +  E'(x)·u_x
```

The familiar textbook form `u_tt = c²·u_xx` keeps only the first term. The second
term, `E'(x)·u_x`, vanishes **only when `E` is constant** — that is, only for a
uniform rod.

**That dropped term is exactly what makes waves reflect and transmit correctly at a
material boundary.** It is what enforces continuity of stress across a join. Drop
it and your simulation is silently wrong precisely where the physics gets
interesting.

Two consequences in this project:

- The inherited repository dropped it. Measured error in the assumed wave speed:
  **22.5 % on twolayer, 58.1 % on multilayer** (`BUGS_AND_CORRECTIONS.md`, A1).
- Our test suite **deliberately deletes that term** as one of 16 mutations, to
  confirm the tests detect a wrong equation. All 16 are caught.

### 11.3 Wave speed

Rearranged for constant coefficients, the equation reads `u_tt = (E/rho)·u_xx`,
which is the standard wave equation with speed

```
c(x)  =  sqrt( E(x) / rho(x) )
```

Stiffer → faster. Heavier → slower. With `E = rho = 1` we get `c = 1`: a wave moves
one unit of distance per unit of time, so it crosses the whole rod in 2.

---

## 12. The conditions that pin down one solution

§4 explained that a differential equation alone has infinitely many solutions. Three
extra conditions select ours.

### 12.1 The initial shape

At `t = 0` the rod has the shape `g(x)` built in §5 — the peak-normalised
derivative-of-Gaussian with `sigma_g = 0.1`. A small double-lobed wiggle near the
origin, flat elsewhere.

```
u(x, 0)  =  g(x)
```

### 12.2 Released from rest

At `t = 0` the rod is deformed but **not yet moving**:

```
u_t(x, 0)  =  0
```

This is a pluck-and-release, not a strike. Both conditions are needed because the
equation is second order in time — just as throwing a ball needs both a starting
position and a starting velocity.

### 12.3 What the physics then does — d'Alembert

For the **uniform** rod there is an exact closed-form answer, found by d'Alembert
in 1747:

```
u(x, t)  =  ½ [ g(x − t)  +  g(x + t) ]
```

Read it as: **the initial bump splits into two half-height copies of itself, one
travelling right and one travelling left, each at speed 1, neither changing shape.**

Why `x − t` means "travelling right": the value of `g(x − t)` at position `x` and
time `t` is whatever `g` was at `x − t`. As `t` grows you must increase `x` by the
same amount to keep seeing the same value — so the pattern moves right at speed 1.

Worked examples, every one verified against our code:

```
u(+0.10, 0.00) = ½[g(+0.10) + g(+0.10)] = ½[-1.0000, -1.0000] = -1.000000
u(+0.20, 0.10) = ½[g(+0.10) + g(+0.30)] = ½[-1.0000, -0.0549] = -0.527473
u(+0.40, 0.30) = ½[g(+0.10) + g(+0.70)] = ½[-1.0000, -0.0000] = -0.500000
u(-0.60, 0.50) = ½[g(-1.10) + g(-0.10)] = ½[+0.0000, +1.0000] = +0.500000
```

The third line: by `t = 0.3` the right-moving copy has travelled 0.3, so the peak
that began at `x = +0.1` now sits at `x = +0.4`, at half height. The left-moving
copy is far away and contributes nothing.

**Why this matters for the study.** On homogeneous we can score against *exact
truth* with zero discretisation error. That removes an entire category of doubt —
and §16.4 shows it mattered, because our finite-difference reference had quietly
become too coarse.

### 12.4 The ends of the rod

The rod is only `[-1, +1]`, but we want to model an *infinite* rod. Left alone, a
wave reaching an end would bounce back and pollute the answer.

So we impose **absorbing boundary conditions**:

```
at x = +1:   u_t − c·u_x = 0
at x = -1:   u_t + c·u_x = 0
```

These say "whatever arrives here is purely outgoing" and let a wave leave without
reflecting — mathematically equivalent to the rod continuing forever.

Note for later: **`u ≡ 0` satisfies both of these exactly**, and also satisfies the
PDE exactly. That fact becomes a problem in §25.

---

## 13. The three materials

Same equation, three rods. Only `E(x)` differs.

| material | `E(x)` at x = −1, −0.5, 0, +0.5, +1 | `c(x)` | interfaces | transition width `w` |
|---|---|---|---|---|
| **homogeneous** | 1.0, 1.0, 1.0, 1.0, 1.0 | 1.0 throughout | none | — |
| **twolayer** | 1.0, 1.0, 1.25, 1.5, 1.5 | 1.0 → 1.225 | **one** | **0.02** |
| **multilayer** | 1.0, 1.3, 1.75, 2.2, 2.5 | 1.0 → 1.581 | **six** | 0.05 |

An "interface" is a region where `E` changes from one value to another. The
**transition width `w`** is how gradually it does so — a small `w` means the
stiffness switches over a very short distance.

### 13.1 The counter-intuitive fact

**Twolayer, with one interface, is roughly 20× harder than multilayer, with six.**

Because difficulty is not driven by how many interfaces there are, but by how
*sharp* they are. From §11.2, the term that makes interfaces hard is `E'(x)` — the
*rate* at which stiffness changes. Twolayer's join is 2.5× narrower, so the same
jump is compressed into a shorter distance and its `E'` is **4.2× larger**.

And from §2.2, a sharper feature means higher curvature, which means the second
derivatives the loss depends on are larger and harder to get right.

**We predicted the opposite early on and were wrong**, which is logged among eight
refuted predictions in `BUGS_AND_CORRECTIONS.md`.

---

## 14. Why use a neural network at all?

A classical **finite-difference** solver chops space and time into a fine grid and
steps forward. It works, and we use one to generate reference answers.

A **Physics-Informed Neural Network (PINN)** does something different. It represents
`u(x,t)` as a neural network and trains it **by penalising violations of the
equation itself** — never by showing it a correct answer.

The appeal: no grid. Grids become punishing in high dimensions or awkward
geometries, whereas a network is just a function you can evaluate anywhere.

Whether that trade actually pays off is an open research question. This study is
one careful data point on one problem — and its main finding turns out to be about
*how you train* rather than *what you train*.

---

# PART B — THE SHARED PIPELINE

Everything in this part is **identical for every architecture**. Only the network
in the middle changes. That is deliberate: if the pipeline varied too, no difference
in results could be attributed to the architecture.

The training loop is four steps, repeated:

```
 (1) pick points in space-time                         §15
 (2) push them through the network and the ansatz      §16, Part C
 (3) measure how badly the physics is violated there   §17
 (4) nudge the network to reduce that violation        §24
```

---

## 15. Step 1 — the collocation points

### 15.1 What a collocation point is

The network must satisfy the wave equation **everywhere** in the rectangle
`x ∈ [-1,1]`, `t ∈ [0,1]`. That is infinitely many places, so we cannot check them
all. Instead we pick a finite set of sample locations and check the equation only
there. Those samples are **collocation points**.

A collocation point is nothing more exotic than **a pair of numbers `(x, t)`** — a
place and a moment. We use **10,000** of them.

The first eight our code actually generates for seed 0:

```
  point 0:  x = -0.049786   t = 0.592524
  point 1:  x = +0.157527   t = 0.037122
  point 2:  x = +0.901340   t = 0.862344
  point 3:  x = -0.759084   t = 0.261442
  point 4:  x = -0.570110   t = 0.974182
  point 5:  x = +0.707517   t = 0.403495
  point 6:  x = +0.484165   t = 0.695453
  point 7:  x = -0.371569   t = 0.172459
```

Point 3 means: *check whether the network obeys the wave equation at position
−0.759 on the rod, at time 0.261.* And so on, 10,000 times, every training step.

### 15.2 How they are stored

```
x :  shape (10000, 1)   dtype float32
t :  shape (10000, 1)   dtype float32
```

Per §6.1: 10,000 rows, 1 column. Row `i` of `x` and row `i` of `t` together are
point `i`. They are kept as **two separate tensors** rather than one `(10000, 2)`
tensor because we must differentiate with respect to `x` and `t` *separately*
(§17.2).

### 15.3 Why 10,000

- **Too few** and the network can satisfy the equation exactly at the sampled points
  while drifting wildly between them — it memorises the exam questions rather than
  learning the subject.
- **Too many** and every step becomes slow, because each point requires two rounds
  of differentiation through the entire network.

10,000 is what the field's benchmark paper (arXiv:2602.15068) uses for 2-D PDE
problems, so we match it rather than inventing our own number.

A sanity check: the rectangle has area `2 × 1 = 2`, so 10,000 points give a mean
spacing of `sqrt(2/10000) ≈ 0.014`. Our pulse has a lobe width of 0.16 (§5.4), so
the pulse is covered by roughly a dozen points across — comfortably resolved.

### 15.4 What "Sobol" means, and why not plain random

The obvious way to pick 10,000 points is uniformly at random. The problem is
**clumping**: purely random points leave gaps in some regions and pile up in others,
purely by chance. With 10,000 samples that is a real effect, and it means some parts
of the domain are barely checked.

A **Sobol sequence** is a *low-discrepancy* sequence — constructed so that every
sub-rectangle of the domain receives close to its fair share of points, far more
evenly than random. "Scrambled" means a random permutation is applied, so different
seeds give genuinely different point sets that are each still well spread.

The practical effect: for the same 10,000 points, the residual is sampled more
evenly, so the loss is a better estimate of the true violation over the whole domain.

### 15.5 "Drawn once and frozen" — and why that matters enormously

We generate the 10,000 points **once**, before training starts, and use **the same
10,000 for every single step**.

This is *not* normal deep-learning practice, where you would resample every step.
The reason is the optimiser. **L-BFGS** (§24.1) works by watching how the gradient
changes over roughly the last 100 steps to build a model of the loss surface's
curvature. That model is only meaningful if it is *the same surface* each time.
Resample the points every step and the surface moves underneath it; the curvature
estimate becomes noise and the optimiser fails.

Getting this wrong made every PINN in an earlier phase look **10–70× worse** than it
actually is.

One verified consequence: the Sobol draw is **byte-identical** between our two
training paths for a given seed. We checked this rather than assuming it, so a
difference between the two paths cannot be blamed on different data.

### 15.6 The boundary points

Separately, **512 time values** are drawn for checking the absorbing boundary
conditions (§12.4) at `x = -1` and `x = +1`. Also fixed for the whole run.

---

## 16. Step 2 — the hard-constraint ansatz

This is the piece everything later leans on, so it gets the most space.

### 16.1 The problem it solves

The network must produce a solution satisfying **three** things at once:

1. the wave equation, everywhere inside the domain
2. the initial shape, `u(x,0) = g(x)`
3. the initial stillness, `u_t(x,0) = 0`

The straightforward approach is to add penalty terms for (2) and (3) to the loss and
hope the optimiser balances all three. That is a **soft constraint**, and it is what
most of the literature does. Its known weakness: the balance between terms is a
hyperparameter, and if it is wrong the optimiser will happily sacrifice the initial
condition to reduce the PDE residual.

The alternative — ours — is to make (2) and (3) **impossible to violate**, by
construction. That is a **hard constraint**, and the construction is called an
**ansatz** (German for "setup": a chosen functional form with free parts inside it).

### 16.2 The construction

The network does **not** output `u`. Call the network's raw output `N(x,t)`. We build
the solution as

```
u(x,t)  =  g(x) · decay(t)   +   growth(t) · N(x,t)
           \_____________/       \_________________/
            term A                term B
            the known start       the learned correction
```

with two fixed helper functions, neither containing any trainable numbers:

```
decay(t)  = exp( -½ · (15t)² )      starts at 1, dies quickly
growth(t) = tanh²( 25t )            starts at 0, rises quickly
```

(`tanh` is the S-shaped function of §17.4; squaring it keeps it positive and flattens
its start.)

### 16.3 Why this forces the initial shape

At `t = 0`:

- `decay(0) = exp(0) = 1`
- `growth(0) = tanh²(0) = 0`

so

```
u(x, 0)  =  g(x)·1  +  0·N(x,0)  =  g(x)          <- condition (2), exactly
```

**The network's output is multiplied by zero at `t = 0`.** However badly trained,
however wild, `u(x,0) = g(x)` exactly. It cannot be violated — not approximately,
not usually, but never.

### 16.4 Why it also forces the initial stillness

Differentiate with respect to `t` and set `t = 0`. Both helpers have **zero slope**
at the origin:

```
decay'(t)  = -225·t·exp(-½(15t)²)          at t = 0 this is 0, because of the factor t
growth'(t) = 50·tanh(25t)·sech²(25t)       at t = 0 this is 0, because tanh(0) = 0
```

Applying the product rule to `u = g·decay + growth·N`:

```
u_t  =  g·decay'  +  growth'·N  +  growth·N_t
```

At `t = 0` the first term has `decay' = 0`, the second has `growth' = 0`, and the
third has `growth = 0`. **Every term vanishes**, so

```
u_t(x, 0) = 0                                      <- condition (3), exactly
```

Two conditions removed from the optimisation problem, permanently and for free. That
is the whole trick.

### 16.5 What the helpers do, numerically

```
     t    decay(t)   growth(t)     sum
  0.00    1.000000    0.000000   1.0000     <- pure initial condition
  0.01    0.988813    0.059985   1.0488
  0.02    0.955997    0.213552   1.1695
  0.05    0.754840    0.719585   1.4744     <- hand-over; both active
  0.08    0.486752    0.929349   1.4161
  0.10    0.324652    0.973408   1.2981
  0.15    0.079560    0.997790   1.0773
  0.20    0.011109    0.999818   1.0109     <- decay essentially dead
  0.30    0.000040    0.999999   1.0000
  0.50    0.000000    1.000000   1.0000     <- u = N exactly
  1.00    0.000000    1.000000   1.0000
```

The story in that table:

- **`t = 0`** — the answer is 100 % known initial shape, 0 % network.
- **`t ≈ 0.05`** — hand-over. Both terms contribute.
- **`t > 0.2`** — `decay` has fallen to 0.011 and `growth` has saturated at ~1. From
  here on **`u ≈ N`: the network alone is the answer.**

Our scoring window is `t ∈ [0.05, 1.0]`, so **almost everything we measure is the
raw network output.** Remember this — it is the root of the collapse in §25.

(The `sum` column peaking at 1.47 rather than 1 is a wart: the two terms overlap
rather than partitioning cleanly. A cleaner-looking alternative, `g(x) + t²·N`, was
tested and **loses 6/6 by two orders of magnitude** — design intent refuted by
measurement, logged in `BUGS_AND_CORRECTIONS.md`.)

### 16.6 A fully worked example

Take `x = +0.10`, where the initial pulse is at its negative peak `g(0.1) = -1.0000`,
and an untrained network:

```
t = 0.00 :  g=-1.0000  decay=1.0000  growth=0.0000  N=-0.8022
            u = (-1.0000)(1.0000) + (0.0000)(-0.8022) = -1.000000
            the network's -0.8022 is annihilated by growth = 0

t = 0.05 :  g=-1.0000  decay=0.7548  growth=0.7196  N=-0.2902
            u = (-1.0000)(0.7548) + (0.7196)(-0.2902) = -0.963676
            mostly still the initial condition; the network is starting to matter

t = 0.30 :  g=-1.0000  decay=0.0000  growth=1.0000  N=+0.3908
            u = (-1.0000)(0.0000) + (1.0000)(+0.3908) = +0.390720
            the initial condition has vanished; u IS the network output
```

An untrained network gives nonsense at `t = 0.30`, exactly as expected. What matters
is that at `t = 0` it is **exactly right by construction**, whatever the network does.

---

## 17. Step 3 — the loss

### 17.1 What a residual is

Rearrange the wave equation so everything is on one side:

```
r(x,t)  =  rho(x)·u_tt  −  d/dx( E(x)·u_x )
```

If `u` is the true solution, `r = 0` everywhere. If `u` is wrong, `r` is a non-zero
number measuring **how badly the physics is violated at that point**. `r` is the
**residual**.

The key property: **computing `r` never requires knowing the answer.** It needs only
the network and the equation. That is what makes this "physics-informed" rather than
supervised learning — we are not fitting to data, we are penalising disobedience.

### 17.2 How the derivatives are computed

`u_tt` and `u_x` are not finite differences. They come from **automatic
differentiation**, which computes exact derivatives by applying the chain rule
through every operation the network performed.

```
u    = ansatz(N(x,t), x, t)
u_t  = autograd.grad(u,   t, create_graph=True)     first derivative
u_tt = autograd.grad(u_t, t)                        second derivative
```

`create_graph=True` on the first call is essential: it tells PyTorch to remember
*how* it computed `u_t`, so that `u_t` can itself be differentiated. Forget it and
the second derivative silently does not exist.

This is also why `x` and `t` are separate tensors (§15.2) — you differentiate with
respect to one of them.

**This is where the study's whole difficulty lives.** The loss depends on *second*
derivatives of the network. By §2.2, a network can be an excellent approximation of
`u` and a terrible approximation of `u_xx`. §26 is entirely about the consequences.

### 17.3 The shapes

```
input   x (10000,1)   t (10000,1)
        |
        |  network + ansatz + two rounds of autograd
        v
residual  r (10000,1)      one number per collocation point
```

For an untrained network those numbers look like `[-87.5, -137.4, -15.1, +151.0,
+36.5, ...]` — large and arbitrary. For a converged model they sit around `1e-3`.

### 17.4 Assembling the loss

```
L  =  mean( r² )   +   mean( b² )
      \________/       \________/
      PDE term          boundary term
      10,000 points     512 points per edge
```

`b` is the absorbing-boundary residual `u_t ∓ c·u_x` at `x = ±1` (§12.4). Squaring
makes every violation positive so they cannot cancel; taking the mean makes the
number independent of how many points were used.

**One number.** That single scalar is what training minimises. There is no
initial-condition term, because §16 made it structurally impossible to violate.

### 17.5 Why the loss is not normalised

An earlier version divided the loss by `residual_scale² ≈ 5e4` to bring it near 1 —
normally harmless. But L-BFGS chooses its very first step size as
`min(1, 1/||gradient||)·lr`. Shrinking the loss shrinks the gradient by the same
factor, so `1/||gradient||` explodes and step one overshoots catastrophically.

Measured: MLP on twolayer went **95.59 % → 0.86 %** simply by removing the
normalisation. It is defect #1 in `BUGS_AND_CORRECTIONS.md` and it invalidated an
entire phase of the study.

---

## 18. How we score a trained model

The 10,000 collocation points are for **training**. Scoring uses a completely
separate, structured grid:

```
4096 evenly spaced positions   x = -1.0000, -0.9995, ..., +1.0000
  20 time snapshots            t = 0.05, 0.10, 0.15, ..., 1.00
------------------------------------------------------------------
4096 × 20 = 81,920 evaluation points
```

Think of it as **20 photographs of the rod**, each with 4096 pixels. We ask the model
for its displacement at all 81,920 points, ask the reference for the same, and
compare with the relative L2 error of §6.3.

Why a regular grid rather than Sobol: for scoring we want even coverage so the metric
is a fair average, and we want the grid the reference is defined on.

Why start at `t = 0.05`: before that the solution has barely evolved and every model
looks perfect, which would flatter all of them equally.

### 18.1 The reference must be finer than what it measures

The reference is not perfect either — a finite-difference solution has its own
discretisation error. **If that error is comparable to the model errors, differences
between models get compressed and become unmeasurable.**

| reference | its own error | margin vs best model (0.0222 %) | verdict |
|---|---|---|---|
| FD nx = 512 | 0.1122 % | **0.2×** | unusable; used in phases 3–7 |
| FD nx = 1024 | 0.0277 % | 0.9× | unusable |
| FD nx = 2048 | 0.0066 % | 3.6× | marginal; used in phases 8–10 |
| FD nx = 4096 | 0.0013 % | 17.5× | adequate |
| **exact d'Alembert** | **0** | infinite | used for the headline |

This bit us **twice**:

- At nx = 512 the reference error *equalled* the model errors, compressing a true
  1.424× gap into a displayed 1.018×.
- At nx = 2048 the margin was ~20× **when chosen** — and silently decayed to 3.6×
  because the models improved 5×. Nothing broke; no test caught it.

The fix for homogeneous is to use the exact solution of §12.3, which has no
discretisation error at all. Doing so moved the headline result from 1.69× to
**1.81×** — the compression had been *understating* it.

---

# PART C — THE NETWORKS

---

## 19. What a neural network is

### 19.1 A linear layer

The basic building block is a **linear layer**, written `Linear(in, out)`. It takes
`in` numbers and produces `out` numbers:

```
output_j  =  ( sum over i of  W[j,i] · input_i )  +  b[j]
```

`W` is a table of `out × in` numbers (the **weights**), `b` a list of `out` numbers
(the **biases**). Both are **trainable parameters** — the numbers training adjusts.

`Linear(2, 128)` holds `128 × 2 = 256` weights plus `128` biases = **384
parameters**. Each output is a weighted blend of the inputs, plus an offset.

### 19.2 Why you need a nonlinearity

Stacking linear layers alone is pointless: a linear map of a linear map is still
linear, so a hundred stacked linear layers can only ever produce a straight line.
Our solution is not a straight line.

So between layers you apply a fixed nonlinear function, element by element — an
**activation function**.

We use **`tanh`**, which smoothly squashes any number into `(-1, 1)` in an S-shape.
Two reasons:

1. It is **smooth to all orders**, so `u_xx` exists and is continuous. The popular
   alternative `relu` (`max(0,x)`) is a bent straight line: its second derivative is
   zero almost everywhere, which would make the PDE residual **blind to the
   network**. A KAN variant using `relu` collapses to 93.6 %.
2. It is bounded, which keeps activations from exploding through depth.

### 19.3 What "training" is

The network starts with random `W` and `b`. Training repeats:

1. compute the loss (§17)
2. compute the **gradient** — for each parameter, how much the loss would change if
   that parameter were nudged up slightly. One number per parameter, obtained by
   **backpropagation** (the chain rule applied backwards through the network).
3. move every parameter a little in the direction that reduces the loss
4. repeat

The **optimiser** is the rule for step 3. Which optimiser you use turns out to be
**the single most consequential choice in this entire study** (§24).

### 19.4 What a parameter count means

"82,689 parameters" is simply the total count of trainable numbers. It is a rough
proxy for how much the model can express.

The field's convention when comparing architectures is to **match** these counts, so
you compare architectures rather than sizes. We initially failed to follow it — the
KAN was winning with 40 % *fewer* parameters. §22.3 fixes that.

---

## 20. How a KAN actually works

### 20.1 The idea, against the MLP you now know

An **MLP** puts **fixed** nonlinearities on the *nodes* and **learned scalars** on
the *edges*:

```
node output  =  tanh( sum_i  w_i · x_i  +  b )
                ^^^^         ^^^
                fixed        learned numbers
```

Each edge can only **scale** a signal up or down. All the shape comes from the fixed
`tanh` at the node.

A **KAN** inverts this completely. It puts **learned univariate functions on the
edges** and **plain summation on the nodes**:

```
node output  =  sum_i  phi_i( x_i )
                       ^^^^^^
                       a learned FUNCTION of one variable
```

There is no weight matrix. There is no activation function in the MLP sense.
**Each edge carries its own little curve**, and training reshapes those curves.

The picture: in an MLP an edge can only make a signal bigger or smaller. In a KAN it
can **bend** it — turn a straight line into an S, a bump, a step, whatever the data
demands.

### 20.2 Why anyone thought this was a good idea

The **Kolmogorov–Arnold representation theorem** states that any continuous function
of many variables can be written as a finite composition of sums of functions of
*one* variable. An MLP approximates a multivariate function head-on; a KAN builds it
out of one-dimensional pieces, which the theorem says is always possible in principle.

**Why it matters for our PDE specifically** — this is the crux of the whole study.
From §12.3, the exact homogeneous solution is

```
u(x,t) = ½[ g(x−t) + g(x+t) ]  =  F(xi) + G(eta)     where xi = x−t, eta = x+t
```

That is **literally a sum of two univariate functions**. It is a Kolmogorov–Arnold
representation with one layer and two edges. Feed a KAN the coordinates `(xi, eta)`
instead of `(x, t)` and its very first layer can express the exact solution. §20.7
and §26.4 return to this.

### 20.3 The edge function, concretely

Each edge computes

```
phi(x)  =  scale_base · silu(x)   +   scale_sp · spline(x)
           \_________________/         \_______________/
           branch 1: a fixed shape      branch 2: a fully
           with one learned amplitude   learned curve
```

`silu(x) = x·sigmoid(x)` is a fixed smooth function, similar in shape to `relu` but
differentiable everywhere. Both `scale_base` and `scale_sp` are trained.

Branch 1 is a **residual path** — it gives the edge a sensible default shape before
the spline has learned anything, so the network is not useless at initialisation.
Branch 2 is where the expressive power lives.

### 20.4 B-splines — what branch 2 is made of

A **spline** is a curve built from polynomial pieces joined smoothly. A **B-spline**
is a particular construction with two properties we care about.

Picture the input range `[-1, +1]` cut into equal intervals by **knots** — the
joining points. Over each interval the curve is a polynomial; at each knot the pieces
are glued so that the curve and several of its derivatives match.

Two numbers define the family:

| term | meaning | our best KAN |
|---|---|---|
| **grid** | how many intervals the range is cut into | `50` |
| **k** (order) | polynomial degree on each interval | `5` |
| **grid_range** | the interval the knots span | `[-1, +1]` |

Everything else follows:

```
number of basis functions  =  grid + k       =  50 + 5  =  55
number of knots stored     =  grid + 2k + 1  =  50 + 11 =  61
knot spacing               =  2 / 50                    =  0.04
```

Both appear directly in the parameter dump: the coefficient tensor's last dimension
is **55**, and the `grid` buffer holds **61** numbers. If you ever wonder where a
shape came from, it is one of these two formulas.

The curve on one edge is a weighted sum (a **basis**, §8):

```
spline(x)  =  sum over 55 basis functions of  coef[j] · B_j(x)
```

The 55 `coef` values are the trainable numbers; training reshapes the curve by
adjusting them.

#### Local support — what makes KANs sharp, and fragile

At any given input `x`, only **`k + 1 = 6`** of the 55 basis functions are non-zero.
All the rest are exactly zero there.

**Consequence one:** a coefficient only affects the curve near its own knot. You can
add detail in one region without disturbing another — which is why a KAN can
represent fine structure that an equally-sized MLP cannot, since an MLP's every
weight affects every input.

**Consequence two, which bites us badly later:** because the coefficients are tied to
*specific knot positions*, **moving the knots invalidates the coefficients.** §26.2
is entirely about what happens when you do.

#### Why `k` matters for a PDE

A B-spline of order `k` is `C^(k−1)` — continuous with `k−1` continuous derivatives.

- At **`k = 3`**: the network is `C²`, so `u_xx` — which the loss consumes — is only
  `C⁰`: **piecewise linear, with a kink at every one of the 50 knots.**
- At **`k = 5`**: the network is `C⁴`, so `u_xx` is `C²` — genuinely smooth.

For a second-order PDE that is a principled change, not knob-twiddling.

#### How the spline is evaluated

By the **Cox–de Boor recursion**: start with indicator functions that are 1 on their
own interval and 0 elsewhere, then raise the degree `k` times, each round blending
neighbouring basis functions linearly in `x`. After `k` rounds you have smooth,
locally-supported basis functions.

### 20.5 A KAN layer, with every shape

A layer with `in_dim` inputs and `out_dim` outputs holds `in_dim × out_dim` edges —
one for every input-output pair — each with its own spline. For `act_fun[1]` of our
best KAN:

```
in_dim = 20, out_dim = 20, grid = 50, k = 5

coef        (20, 20, 55)  = 22,000 numbers   one 55-coefficient spline per edge
scale_base  (20, 20)      =    400 numbers   silu amplitude, per edge
scale_sp    (20, 20)      =    400 numbers   spline amplitude, per edge
grid        (20, 61)      buffer, not trained
                          knot positions, ONE ROW PER INPUT
                          (shared across that input's 20 outgoing edges)
```

The forward pass for a batch of `N` points:

```
input                                                      (N, 20)
  |
  |-- branch 1:  silu(input)                            -> (N, 20)
  |-- branch 2:  coef2curve(input, grid, coef, k)       -> (N, 20, 20)
  |                                                        [batch, in, out]
  |
  phi = scale_base · branch1[:,:,None] + scale_sp · branch2
                                                        -> (N, 20, 20)
  |
  sum over the INPUT axis                               -> (N, 20)
  |
output                                                     (N, 20)
```

The node does nothing but add up its 20 incoming edge values. **All the modelling
lives in the 400 edge functions.**

### 20.6 Every term, collected

| term | meaning |
|---|---|
| `width` | nodes per layer, e.g. `[2,20,20,20,1]` — 2 in, three hidden layers of 20, 1 out |
| `grid` | spline intervals per edge — the **resolution** knob |
| `k` | spline order; controls smoothness of `u` and therefore of `u_xx` |
| `coef` | trained spline coefficients, shape `(in, out, grid+k)` |
| `scale_base`, `scale_sp` | trained amplitudes of the two branches |
| `base_fun` | the fixed residual shape, `silu` here |
| `grid_range` | the interval the knots cover, `[-1,1]` |
| `grid_eps` | blend between evenly-spaced and data-adaptive knots (0.02) |
| `noise_scale` | random noise added to spline coefficients at init (0.3) |
| `curve2coef` | least-squares fit of coefficients to sampled curve **values** |
| `update_grid` | move the knots to match observed activations, then re-fit |
| `symbolic_fun` | a parallel branch for symbolic regression — **disabled** in all our runs |

Two are load-bearing. `update_grid` destroys our runs 8/8 (§26.2), and `curve2coef`
returned NaN on GPU until we patched it (`BUGS_AND_CORRECTIONS.md`).

### 20.7 Characteristic coordinates — the change that wins

This is not a KAN feature; it is a change to *what we feed the KAN*.

Define the **travel-time coordinate**

```
tau(x)  =  integral from 0 to x of  ds / c(s)
```

— how long a wave takes to reach `x`. For homogeneous material `c = 1`, so
`tau(x) = x`. Then form

```
xi  = tau(x) − t        the "right-moving" coordinate
eta = tau(x) + t        the "left-moving" coordinate
```

and rescale each to `[-1, +1]` so they land exactly on the knot range.

**Why this is the whole ball game.** In these coordinates the wave operator becomes

```
∂²/∂xi ∂eta
```

so **any** function of the form `F(xi) + G(eta)` satisfies the wave equation
*identically* — differentiate `F(xi)` with respect to `eta` and you get exactly zero,
for any `F` whatsoever, without training anything.

And a KAN layer computes precisely a sum of univariate functions. So in these
coordinates **the KAN's architecture and the solution's structure are the same
shape.** §26.4 shows what that buys, measured.

`tau(x)` is built by quintic Hermite interpolation of a Gauss–Legendre quadrature of
`1/c(x)`, in float64, accurate to `tau' = 1.5e-7` and `tau'' = 1.9e-6`. It is a fixed
buffer — computed once, never trained.

---

# PART D — THE ARCHITECTURES

`N` is the batch size — 10,000 during training, 81,920 at evaluation. Every model is
a black box taking `(x, t)` and returning one number `N(x,t)`, which then goes into
the ansatz of §16 to become `u`.

Every dimension below was read off the live models by introspection, not written from
memory.

---

## 21. The Fourier embedding (used by three models)

Two of our architectures start by transforming `(x,t)` into a much longer vector of
sines and cosines — a **random Fourier feature embedding**:

```
p      = [x, t] @ B.T         B is a FIXED 64×2 table of random numbers
                              drawn from a bell curve of width sigma_B = 10
output = [cos(p), sin(p)]     -> 128 numbers
```

Each of the 64 rows of `B` defines a plane wave in `(x,t)` with a random direction
and a random frequency of typical size 10. The embedding reports how strongly the
input point aligns with each of those 64 waves, as a cosine and a sine.

**Why do it.** From §19.2, a plain `tanh` MLP is biased toward smooth, slowly varying
functions and needs great depth to produce oscillations. Handing it 128 oscillating
features up front removes that burden.

**Where `sigma_B = 10` comes from.** §7.2: the pulse's own dominant frequency is
`1/sigma_g = 1/0.1 = 10`, measured at 9.425. The embedding's bandwidth is set to the
solution's bandwidth. It is a principled choice, not a tuned one, and it is held
identical across every architecture that uses an embedding (`DECISIONS.md` D4).

**`B` is a buffer, not a parameter** — never trained, so the bandwidth stays at its
declared value and remains a controlled variable.

**The price, which §26 collects.** From §7.3, differentiating a `cos(2πk·x)` twice
brings down `(2πk)² ≈ 3.5×10⁴`. The embedding that makes the *function* easy to fit
makes its *second derivatives* noisy — and the loss is built entirely from second
derivatives.

---

## 22. The PINNs

### 22.1 MLP PINN — `mlp` — 66,561 parameters

The control. No embedding; raw coordinates straight in.

```
x (N,1), t (N,1)
  concatenate                                -> (N, 2)
  Linear(2   -> 128) + tanh                  -> (N, 128)      256 + 128 params
  Linear(128 -> 128) + tanh                  -> (N, 128)   16,384 + 128
  Linear(128 -> 128) + tanh                  -> (N, 128)   16,384 + 128
  Linear(128 -> 128) + tanh                  -> (N, 128)   16,384 + 128
  Linear(128 -> 128) + tanh                  -> (N, 128)   16,384 + 128
  Linear(128 -> 1)                           -> (N, 1)        128 + 1
```

Five hidden layers of 128 units. **Best model on twolayer** (0.4318 ± 0.3070) — the
plainest architecture wins on the sharpest material. Fourier features tuned to the
pulse's spectrum do not help with an interface discontinuity, because a discontinuity
is not a single frequency; it needs *all* of them.

### 22.2 Fourier PINN — `fourier` — 82,689 parameters

The same MLP with the §21 embedding in front.

```
x (N,1), t (N,1)
  concatenate                                -> (N, 2)
  @ B.T        B is (64,2), FIXED            -> (N, 64)
  [cos(p), sin(p)]                           -> (N, 128)
  Linear(128 -> 128) + tanh                  -> (N, 128)   16,384 + 128
  ... four more 128-wide tanh layers ...
  Linear(128 -> 1)                           -> (N, 1)
```

The headline PINN for most of this study: 0.0399 ± 0.0138 on homogeneous.

### 22.3 Parameter-matched Fourier PINN — `fourier_matched` — 49,729 parameters

Identical, except hidden width **96** rather than 128, chosen so the parameter count
matches the tuned KAN's 49,020 to within 1.45 % (per §19.4).

```
  Linear(128 -> 96) + tanh    <- narrows from the 128-d embedding
  Linear(96  -> 96) + tanh    x4
  Linear(96  -> 1)
```

**And it is better than the 128-wide version** — 0.0332 % against 0.0401 %. The
default was over-parameterised, so the headline PINN arm had been handicapped by its
own size.

### 22.4 PirateNet — `pirate` — ~150,000 parameters

A published PINN architecture. Fourier embedding, then residual blocks using **random
weight factorisation** — each weight matrix stored as `W = diag(exp(s))·V` with both
`s` and `V` trained, decoupling a weight's scale from its direction.

Its output layer is zero-initialised, so without intervention the network is
identically zero at init, `u_tt = 0`, and the backbone receives no gradient at all. A
physics-informed least-squares step at construction fixes this.

Competitive on homogeneous (0.0296 ± 0.0024 — **the lowest seed variance of any
model**) and multilayer (0.0563), but **2.768 %** on twolayer: a 25× inversion on the
sharp interface, the same material where the plain MLP wins.

---

## 23. The KANs

### 23.1 Plain KAN — `pykan_wide` — 8,600 parameters

The reference pykan implementation on raw `(x,t)`. Width `[2,20,20,20,1]`, `grid = 5`,
`k = 3`.

```
x (N,1), t (N,1)
  concatenate                                        -> (N, 2)

  act_fun[0]   2 -> 20      40 edges
      coef (2,20,8)    grid (2,12)     n_basis = 5+3 = 8
      phi = scale_base·silu + scale_sp·spline, summed over the 2 inputs
                                                     -> (N, 20)
  act_fun[1]  20 -> 20     400 edges   coef (20,20,8)   grid (20,12)
                                                     -> (N, 20)
  act_fun[2]  20 -> 20     400 edges   coef (20,20,8)
                                                     -> (N, 20)
  act_fun[3]  20 ->  1      20 edges   coef (20,1,8)
                                                     -> (N, 1)
```

860 edges, 8 coefficients each = 6,880, plus 1,720 scale parameters = 8,600.

**A real handicap worth naming.** The forward pass is `kan(cat([x,t]))` with no input
scaling, and pykan's knot range is `[-1,1]`. Our `x` fills that range, but `t` only
covers `[0,1]` — so **roughly half the knots along the time axis are never visited by
any input.**

Despite that: 0.0849 % on homogeneous from 8,600 parameters — an order of magnitude
fewer than any PINN here.

### 23.2 The tuned KAN — `charcoords50` — 49,020 parameters — **the winner**

Same backbone, three changes.

```
x (N,1), t (N,1)
  |
  |  CharCoords (§20.7):  tau(x) = travel-time coordinate
  |                       xi  = tau(x) − t
  |                       eta = tau(x) + t
  |                       each affinely mapped to [-1,1]
  |                                                  -> (N, 2)
  |  (no scaler needed — the pair already fills the knot range exactly)
  |
  act_fun[0]   2 -> 20      40 edges
      coef (2,20,55)   grid (2,61)   n_basis = 50+5 = 55   knot spacing 0.04
                                                   -> (N, 20)
  act_fun[1]  20 -> 20     400 edges   coef (20,20,55) = 22,000 params
                                                   -> (N, 20)
  act_fun[2]  20 -> 20     400 edges   coef (20,20,55) = 22,000 params
                                                   -> (N, 20)
  act_fun[3]  20 ->  1      20 edges   coef (20,1,55)  =  1,100 params
                                                   -> (N, 1)
```

| change | from | to | why it is principled, not tuned |
|---|---|---|---|
| input coordinates | raw `(x,t)` | `(tau−t, tau+t)` | the exact solution is `F(xi)+G(eta)` — a sum of univariate functions, which is the KAN's own structure (§20.2, §20.7) |
| `grid` | 5 | 50 | 10× resolution; knot spacing 0.04 against a pulse lobe of width 0.16 (§5.4) — about four knots across the feature |
| `k` | 3 | 5 | `u` becomes `C⁴`, so `u_xx` is `C²` rather than kinked at every knot (§20.4) |

This model beats every PINN on homogeneous. §26.4 explains why in one sentence.

### 23.3 The PIKANs — both fail, for different reasons

"PIKAN" in the literature means a spline KAN on a Fourier embedding. We tried two.

**`splinekan` (ours) — 246,208 parameters.**

```
x, t -> FourierEmbed(64, sigma=10)                      -> (N, 128)
     -> SplineKANLayer(128 -> 64, grid=10, order=4) -> LayerNorm -> (N, 64)
     -> SplineKANLayer(64 -> 64)  ×2                    -> (N, 64)
     -> Linear(64 -> 1)                                  -> (N, 1)
```

**Broken before training starts.** `max|u_tt| = 1.16×10⁶` against every other model's
~2×10², PDE loss `3.5×10¹⁰` and gradients `4.4×10¹⁰` at step zero.

The cause is §21's price compounded with §20.4's local support: B-splines placed
**directly** on a `sigma_B = 10` embedding, so the chain rule multiplies
`(2πk)² ≈ 3.5×10⁴` from the embedding by `(1/knot spacing)² ≈ 25` from the spline.
pykan avoids this by damping its spline branch `1/sqrt(fan_in)`; plain initialisation
does not.

`splinekan_fix` applies the same damping and drops the init loss **100,000×** — which
converts a divergence (235 %) into a collapse (97 %). Better in kind, not in outcome.

**`fourier16` (pykan backend) — 36,500 parameters.** The same idea done correctly.

```
x, t -> FourierEmbed(16, sigma=10)   -> (N, 32)
     -> act_fun[0]  32 -> 20         -> (N, 20)
     -> act_fun[1..2]  20 -> 20      -> (N, 20)
     -> act_fun[3]  20 -> 1          -> (N, 1)
```

Initialisation is **healthy** (`max|u_tt| = 239`). It collapses anyway, on all three
materials, under both optimisers, at ~95 %. §26.3 explains why, and the answer is not
what it looks like.

### 23.4 WavKAN — `wavkan`

Mexican-hat wavelets on raw `(x,t)`: `psi(z) ∝ (z² − 1)·exp(−z²/2)`, with a learned
scale and translation per edge. A wavelet is a localised wiggle, so this is a KAN
whose edge functions are bumps rather than splines.

Diverges under L-BFGS through a float32 overflow inside torch's line search
(`d1² ≈ 2.7×10⁴⁹`, and float32 tops out at `3.4×10³⁸` — §6.1). It is the only
architecture that one-shot residual resampling ever rescued: 4 times out of 4, from
~95 % to 0.8–5.2 %.

---

# PART E — RESULTS AND MECHANISMS

---

## 24. Why the optimiser decided the result

### 24.1 What the two optimisers do

Both answer: *given the gradient, how do I move the parameters?*

**Adam** moves each parameter by roughly a fixed step `lr`, in the direction that
reduces the loss, **regardless of how large that parameter's gradient is** — it
divides by a running estimate of the gradient's own magnitude. With `lr = 1e-3` over
700 steps, each parameter can travel up to about 0.7 in total.

**L-BFGS** is a **quasi-Newton** method. Rather than just following the slope, it
watches how the gradient changed over roughly the last 100 steps to build a picture
of the loss surface's **curvature** (§2.2 again — curvature is the second
derivative), then jumps toward the estimated bottom of that local bowl. Far
better-informed steps — but its first step is scaled `min(1, 1/||gradient||)·lr`, so
a large gradient forces a tiny one.

Two consequences, and they are the whole story:

1. **L-BFGS is local.** From a cold random start it descends into whatever bowl it
   happened to land in, and cannot climb out to a better one.
2. **Adam explores.** Its step does not shrink when gradients are large, so it
   wanders across the landscape rather than settling immediately.

This is also why the collocation points must be frozen (§15.5): the curvature model
assumes the surface is not moving.

### 24.2 What changes

| | pure L-BFGS | Adam(700) → L-BFGS |
|---|---|---|
| Fourier PINN | 0.0281 ± 0.0073 | 0.0399 ± 0.0138 |
| tuned KAN | 0.0313 ± 0.0170 | **0.0236 ± 0.0129** |
| KAN variance ÷ PINN variance | **5.44×**, p = 0.031 | **0.87×**, p = 0.73 |
| verdict | tie, p = 0.577 | KAN wins, p = 0.0098 |

The warm-up **helps the KAN and slightly hurts the PINN**. It is not a global
improvement; it is an *interaction* between optimiser and architecture. A paper that
benchmarks one optimiser and reports the outcome as an architecture property is
reporting this interaction without knowing it.

### 24.3 The measurement

On `splinekan`, from an identical initialisation — how far do the parameters travel,
and how low does the loss get?

```
                  loss@init   loss@end    ||Δparameters|| / ||initial||
Adam lr=1e-3       3.39e10     5.17e5           0.1608
Adam lr=1e-5       3.39e10     2.68e6           0.0117
L-BFGS 35 epochs   3.39e10     7.89e4           0.0089
```

**L-BFGS reaches a 6.5× lower loss while travelling 18× less** — about 115× more
efficient per unit of parameter movement. It is a far better *local* optimiser, and
that is precisely why it cannot escape a poor starting basin.

### 24.4 Why the KAN benefits more

The KAN's basis fits this solution far better in principle (§26.4), but that potential
lives in a region of parameter space a cold quasi-Newton start does not reach. Under
pure L-BFGS the advantage bought **nothing** — the two families tied.

The strongest evidence that this is a basin effect rather than a mean shift is the
**variance**. Under pure L-BFGS the KAN's seed-to-seed variance was 5.44× the PINN's
(p = 0.031) — the only statistically significant difference between the families, and
the main argument against KANs. Under the hybrid it is 0.87×, not significant.
**The warm-up did not improve an average; it removed a failure mode.** Seeds that used
to land in bad basins no longer do.

---

## 25. Reading the results

### 25.1 The statistics, in plain terms

Every number is `mean ± sd (n)` where **n is the number of seeds** (§9). The tests:

| test | question it answers | why used here |
|---|---|---|
| **Welch t-test** | are the two averages different? | does not assume equal spread — ours differ |
| **Mann–Whitney U** | is one group systematically ranked higher? | assumes nothing about the shape of the distribution |
| **paired Wilcoxon** | seed by seed, does one win more often and by more? | our arms share seeds *and* collocation sets, so pairing removes shared noise and is far more sensitive |
| **Levene** | are the two *spreads* different? | the KAN's inconsistency was the main argument against it |

**`p`** is the probability of seeing a difference at least this large if there were
truly none. `p < 0.05` is convention, not law; 0.049 and 0.051 mean nearly the same
thing.

Comparison arms are **named before the data is examined**. One earlier test picked the
three lowest KAN values and then asked whether they were low (`p = 0.0000`,
meaningless). It is retracted.

### 25.2 Homogeneous — the KAN wins

Against the **exact** d'Alembert solution, n = 11 per arm:

```
Fourier PINN   0.0401 ± 0.0175        tuned KAN   0.0222 ± 0.0139
KAN ahead 1.81×   Welch p = 0.0154   Mann-Whitney p = 0.0126   paired 9/11   Wilcoxon p = 0.0322
```

At **matched parameters** (§22.3) the lead narrows to **1.50×** and the tests
disagree: Welch p = 0.092, Mann–Whitney p = 0.049, paired Wilcoxon p = 0.0068. The
paired test suits this design, but the disagreement is reported rather than the
favourable test selected.

### 25.3 All models, all materials

Adam→L-BFGS, rel-L2 %, mean ± sd (n), scored at nx = 2048:

| model | homogeneous | twolayer | multilayer |
|---|---|---|---|
| tuned KAN (`charcoords50`) | **0.0236 ± 0.0129** (11) | 0.5984 ± 0.3510 (11) | **0.0403 ± 0.0114** (11) |
| Fourier PINN | 0.0399 ± 0.0138 (11) | 0.9579 ± 0.7969 (11) | 0.0432 ± 0.0115 (11) |
| plain KAN | 0.0849 ± 0.0201 (5) | 1.0340 ± 0.4055 (5) | 0.2320 ± 0.0984 (5) |
| MLP PINN | 0.2555 ± 0.1232 (5) | **0.4318 ± 0.3070** (11) | 0.6915 ± 0.2701 (11) |
| PIKAN (pykan) | 94.98 ± 0.20 (5) | 95.45 ± 0.16 (5) | 96.46 ± 0.23 (5) |
| PIKAN (ours) | 235.6 ± 126.5 (5) | 243.6 ± 132.9 (5) | 266.9 ± 148.4 (5) |

**twolayer at n = 11: PINN ahead 1.39×, p = 0.250 — not significant.** At n = 5 it read
2.20×, p = 0.056. That is the fifth finding in this project to dissolve under more
seeds. **multilayer: p = 0.571 — a tie.**

So: the KAN wins significantly on one material of three, and is **never significantly
worse anywhere**.

---

## 26. The one mechanism behind every failure

### 26.1 The PDE residual is a cancellation

Measure the trained `charcoords50` on homogeneous, where the equation is exactly
`u_tt = u_xx`:

```
||u_tt||     = 2942
||u_xx||     = 2942
||residual|| =    0.44
```

(Norms as defined in §6.2.) The two terms are each about 2942 in size, and their
difference is 0.44. **They cancel to 1 part in 6,696.**

This reframes what a physics-informed loss actually is. It is **not** "fit the
function". It is **"produce two enormous quantities that agree with each other to
four decimal places."** Getting `u` right to 0.04 % guarantees nothing about that —
because by §2.2, differentiating twice amplifies error by the square of the frequency.

Everything below is a consequence of that one sentence.

### 26.2 Why `update_grid` destroys a working model

`update_grid` (§20.6) moves the spline knots to match where the activations actually
are, then re-fits the coefficients. **pykan's own `fit()` calls it 10 times by
default** — it is central to how KANs are meant to gain accuracy.

One call on a **trained, working** model:

```
              rel-L2      training objective
before        0.0114 %      1.930e-05
after         0.2272 %      3.683e-01     <- 19,000× worse
```

What it preserves, measured on 4,000 collocation points:

| quantity | ‖before‖ | relative change |
|---|---|---|
| u | 1.508e+01 | 0.10 % |
| u_x | 1.863e+02 | 0.59 % |
| **u_xx** | 2.942e+03 | **2.28 %** |
| **u_tt** | 2.942e+03 | **2.29 %** |

The function barely moves. The second derivatives move 23× more — exactly the
amplification §2.2 predicts.

`curve2coef` re-fits coefficients to match **function values** at sample points.
Nothing constrains the derivatives. And recall §20.4: coefficients are tied to
specific knot positions, so moving the knots means the old coefficients no longer
mean what they meant.

Why 2.3 % is fatal: the cancellation needs `1.5×10⁻⁴` relative accuracy. A 2.3 %
perturbation is **153× coarser**. Predicting the damage:

```
predicted ||residual|| after = sqrt(2) × 0.0229 × 2942 = 95.3
predicted loss               = 95.3² / 10000 = 0.908
measured loss                = 0.368
```

Agreement within 2.5×, and on the correct side — the two perturbations are not fully
independent, so some cancellation survives.

L-BFGS then restarts from an objective 19,000× worse and finds the trivial solution.
**8/8 runs collapse, always to ~95 %, never gradually.**

**This is not a bug in pykan.** Grid adaptation is exactly right for regression, where
only function values matter. It is silently wrong for any loss built from high-order
derivatives — which means anyone building a physics-informed KAN on pykan's default
`fit()` hits this without warning.

### 26.3 Why Fourier-embedded KANs collapse

It looks like the architecture cannot express the solution. **It cannot be that.**

Fit each model to the true solution by ordinary supervised regression — no PDE loss
anywhere — then measure the derivatives at that fitted solution:

| model | fit rel-L2 | ‖u_tt‖ | ‖u_xx‖ | ‖u_tt − u_xx‖ | cancels to |
|---|---|---|---|---|---|
| `fourier16` | 0.0431 % | 3.384e+04 | 4.568e+03 | 3.247e+04 | **1 : 1** |
| `charcoords50` | 0.0415 % | 2.984e+03 | 2.947e+03 | 4.269e+02 | 7 : 1 |

`fourier16` **fits `u` better than the Fourier PINN does.** Its basis is excellent.
But its two second derivatives are not even the same *magnitude* — off by 7.4× — and
cancel **not at all**.

Now evaluate the actual training objective at each of those correct solutions:

```
fourier16      L(correct) = 2.588e+05     L(collapsed) = 1.536e-05
charcoords50   L(correct) = 4.525e+01     L(converged) = 1.930e-05
```

For `fourier16`, **the trivial answer scores 17 billion times better under our loss
than the right answer does.** Restart physics training from the correct solution and
it abandons it in **one epoch**, dropping to 95.29 % with amplitude 0.275.
`charcoords50` restarted the same way *recovers*: 3.02 % → 0.236 % → 0.0160 %, with
amplitude pinned at 1.000.

**The optimiser is not failing. The loss prefers the wrong answer** — because the only
way a basis with noisy second derivatives can make `u_tt − u_xx` small is to make
*both* small, and that means shrinking `u` toward zero.

And from §12.4, `u ≡ 0` genuinely satisfies the wave equation and the absorbing
boundaries exactly. From §16.5, `decay(t)` is 0.011 by `t = 0.2`, so across almost the
entire scoring window `u ≈ N` — driving the network output to zero drives the solution
to zero. That is a real global minimum of our loss, sitting at ~95 % error.

### 26.4 Why the tuned KAN wins

From §20.7: in characteristic coordinates the wave operator is `∂xi ∂eta`, so
`F(xi) + G(eta)` satisfies the equation **identically, for any `F` and `G`
whatsoever**.

A KAN layer computes a sum of univariate functions. So the cancellation of §26.1 is
**structural — built into the coordinate system — rather than something the optimiser
must learn.** That is why `charcoords50` achieves 7:1 cancellation straight out of a
value fit, with its two second-derivative norms agreeing to 1.3 %, while the
Fourier-embedded KAN achieves 1:1.

It also explains twolayer, where the advantage disappears: a sharp interface makes
`c(x)` vary quickly, `tau'' = −c'/c²` becomes large, and the chain rule injects that
term into `u_xx`. The coordinate change stops being exact, and the structural
cancellation is lost.

### 26.5 The principle, stated once

> A physics-informed loss does not measure how well a network fits the solution.
> It measures how well the network's **high-order derivatives cancel**. Any
> architectural choice — a Fourier embedding, a grid refit, a coordinate change —
> must be judged in derivative space, not value space.

This supersedes the framing in `BASIS_VS_OPTIMISER.md`, whose headline "233×
separable-basis advantage" is a **value-space** measurement and is therefore the wrong
quantity. The right one is the cancellation ratio in §26.3.

---

## 27. Why the soft initial condition fails

### 27.1 What was tried

§16 makes the initial condition unbreakable but creates the trivial basin of §26.3. A
**soft** initial condition should remove that basin: drop the ansatz so the network
*is* `u` directly, and add a penalty

```
ic_loss = mean( (u(x,0) − g(x))² )  +  mean( u_t(x,0)² )
```

Now `u ≡ 0` is no longer free — it costs the full size of the initial pulse.

### 27.2 The arithmetic works

```
IC penalty for u ≡ 0, unweighted     :  mean(g²) = 0.1204
weight the optimiser assigns it      :  55.6   (against 0.505 for the PDE term — a 110× ratio)
IC penalty for u ≡ 0, weighted       :  6.70
converged PDE loss, for comparison   :  ~1e-3
```

The basin goes from *cheaper than converging* to **three orders of magnitude more
expensive** — and it works. Models trained this way escape and reach real solutions.

(The weight 55.6 is chosen automatically by gradient-norm balancing, independently
rediscovering the ~100× IC weight the soft-constraint literature picks by hand. A
first attempt appeared to fail at 95.52 % only because rebalancing fired every 500
steps and the test ran 200 — the IC term sat at weight 1 throughout, where the basin
costs only 0.12 and a collapsing model simply pays it.)

### 27.3 But it costs a great deal, and rescues nothing

| model | hard ansatz | soft IC | effect |
|---|---|---|---|
| Fourier PINN | 0.0399 ± 0.0138 | 0.2140 ± 0.1107 | **5.5× worse** |
| tuned KAN | 0.0236 ± 0.0129 | 0.0479 ± 0.0278 | **2.0× worse** |
| PIKAN (pykan) | 94.98 | 100.62 ± 3.02 | still dead |
| PIKAN (ours) | 235.6 | 372.9 ± 321.6 | still dead |
| MLP, Adam-only | 0.7392 | 99.63 | **fails outright, 5/6 runs** |

Three distinct failures:

1. **The optimisation problem gets harder.** The hard ansatz removes two constraints
   from the search entirely. Restoring them as penalties means the optimiser must
   trade PDE accuracy against IC accuracy at every step, and the balance becomes a
   hyperparameter with real consequences.
2. **The initial condition stops being exact.** A guaranteed-correct start is traded
   for an approximately-correct one — adding a new error source exactly where the
   solution is most structured.
3. **It does not address why the PIKANs fail.** They stay dead, which is direct
   evidence their collapse was never about the basin being cheap. It is §26.3: their
   derivatives cannot cancel, basin or no basin.

**One informative asymmetry.** The KAN degrades 2.0× where the PINN degrades 5.5×, so
under soft constraints the KAN's lead *widens* from 1.69× to **4.47×** (3/3 paired).
The headline is not an artefact of the hard ansatz — if anything the hard ansatz
flatters the PINN.

---

## 28. Summary

- **The problem** is a wave on an elastic rod; we want `u(x,t)` everywhere (§10–§12).
- **The equation** is Newton's second law for a continuum, in conservative form
  because the `E'(x)u_x` term is what makes interfaces behave correctly (§11).
- **The initial pulse** is a peak-normalised derivative-of-Gaussian with
  `sigma_g = 0.1`, giving a dominant frequency of `1/sigma_g = 10` — which is where
  the Fourier PINN's `sigma_B = 10` comes from (§5, §7).
- **We check the physics** at 10,000 frozen Sobol points and score on an 81,920-point
  grid against the exact solution (§15, §18).
- **The initial condition is enforced by construction**, not by penalty — which is
  both why training works and why a trivial solution exists (§16, §26.3).
- **The loss is built from second derivatives**, and that is the whole difficulty
  (§2.2, §17, §26.1).
- **A KAN puts learned curves on its edges** instead of scalars, and in characteristic
  coordinates its structure matches this solution's exactly (§20).
- **Under pure L-BFGS the two families tie.** Under Adam→L-BFGS the tuned KAN wins by
  1.81× on homogeneous and its variance problem disappears (§24, §25).
- **Every catastrophic failure** — `update_grid`, both PIKANs, the soft-IC MLP — is the
  same fact: a basis whose second derivatives cannot cancel to 1 part in 6,696 has no
  way to satisfy this loss except by shrinking toward zero (§26).
