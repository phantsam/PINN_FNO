# Every Model, Layer by Layer — and Why They Succeed or Fail

A complete architectural and mechanistic reference for the PINN-vs-KAN study on
the 1-D elastic wave equation.

**This document assumes no prior knowledge.** It starts from what the physics
problem is, explains every symbol before using it, and builds up to the
architectures and results. If you already know PINNs and KANs, skip to §5.

Companion documents: `BUGS_AND_CORRECTIONS.md` (every defect and retraction),
`ARCHITECTURES.md` (phase chronology), `DECISIONS.md` (frozen spec).

---

## 0. Reading guide

| section | what it covers | assumes |
|---|---|---|
| §1 | **the physics problem, from zero** — what a wave equation is, what we are solving | nothing |
| §2 | **the shared pipeline** — collocation points, the ansatz, the loss, in full detail | §1 |
| §3 | **how a KAN actually works**, from the theorem to the tensor shapes | §2 |
| §4 | each architecture, layer by layer, with dimensions at every step | §3 |
| §5 | the best model on each material, and the numbers behind it | — |
| §6 | why Adam→L-BFGS beats pure L-BFGS | §2 |
| §7 | why the soft initial condition fails | §2 |
| §8 | the cancellation principle that explains every failure in one line | §2 |

---

# PART A — THE PROBLEM

---

## 1. What are we actually solving?

### 1.1 The physical picture

Imagine a long thin elastic rod lying along a line. You pluck it somewhere in the
middle. A disturbance travels outward along the rod in both directions — a wave.

We want to predict, for **every position** along the rod and **every moment** in
time, how far the rod is displaced from rest at that place and moment.

That is the whole problem. Everything below is machinery for answering it.

### 1.2 The three symbols

| symbol | name | meaning | range here |
|---|---|---|---|
| `x` | position | where along the rod, in non-dimensional units | `-1` to `+1` |
| `t` | time | how long since the pluck | `0` to `1` |
| `u` | displacement | how far the rod is displaced at position `x`, time `t` | roughly `-1` to `+1` |

`u` is a **function of two variables**, written `u(x,t)`. Feed it a place and a
moment, it returns a number — the displacement there and then.

So `u(0.40, 0.30) = -0.500000` means: *at position 0.40 along the rod, 0.30 time
units after the pluck, the rod is displaced by −0.5 units.* That is a real value
from this problem, computed in §1.6.

**Why "non-dimensional".** The rod is not literally 2 metres long and the
simulation is not literally 1 second. We rescale so that the wave speed is 1 and
the domain is `[-1,1]`. This is standard: it removes units from the equations and
means a wave crosses the whole domain in exactly 2 time units. (`DECISIONS.md` D1.)

### 1.3 The equation

The physics says `u(x,t)` must satisfy:

```
rho(x) · u_tt  =  d/dx ( E(x) · u_x )
```

Reading it symbol by symbol:

| piece | meaning |
|---|---|
| `u_t` | how fast the displacement is changing at that point — the **velocity** |
| `u_tt` | how fast the *velocity* is changing — the **acceleration** |
| `u_x` | how much the displacement changes as you move a little along the rod — the local **stretch** or strain |
| `E(x)` | **stiffness** at position `x`. Big `E` = hard material, waves travel fast |
| `rho(x)` | **density** at position `x`. Big `rho` = heavy material, waves travel slow |
| `d/dx(...)` | how the quantity inside changes as you move along the rod |

So the equation says: **mass × acceleration = the net internal force**. It is
Newton's second law written for every point of a continuous rod at once.

The quantity `E(x)·u_x` is the **stress** — how hard the material at `x` is being
pulled. If the stress is the same on both sides of a point, the forces cancel and
that point does not accelerate. Only a *change* in stress along the rod produces
acceleration, which is why the right-hand side is `d/dx` of the stress.

**Why this "conservative" form and not `u_tt = c²u_xx`.** If you expand the right
side by the product rule you get

```
d/dx( E·u_x )  =  E·u_xx  +  E'(x)·u_x
```

The second term `E'(x)·u_x` only vanishes when `E` is constant. Drop it and your
equation is silently wrong wherever the material changes — and getting that term
right is exactly what makes waves reflect and transmit correctly at a material
boundary. The inherited repository dropped it, which is defect A1 in
`BUGS_AND_CORRECTIONS.md`. Our test suite deliberately deletes that term as a
mutation to prove the tests notice.

### 1.4 The wave speed

From `E` and `rho` you get the local wave speed:

```
c(x) = sqrt( E(x) / rho(x) )
```

Stiffer material → faster waves. In our non-dimensional units, homogeneous
material has `E = rho = 1`, so `c = 1`: a wave moves one unit of distance per unit
of time.

### 1.5 The three materials

We solve the same equation on three rods. Only `E(x)` changes.

| material | `E(x)` at x = −1, −0.5, 0, +0.5, +1 | wave speed `c(x)` | character |
|---|---|---|---|
| **homogeneous** | 1.0, 1.0, 1.0, 1.0, 1.0 | 1.0 everywhere | uniform rod, the easy case |
| **twolayer** | 1.0, 1.0, 1.25, 1.5, 1.5 | 1.0 → 1.225 | one **sharp** join, transition width `w = 0.02` |
| **multilayer** | 1.0, 1.3, 1.75, 2.2, 2.5 | 1.0 → 1.581 | six joins, but **gradual**, `w = 0.05` |

The counter-intuitive fact that runs through this whole study: **multilayer, with
six interfaces, is easier than twolayer, with one.** Because twolayer's join is
2.5× narrower, its `E'(x)` — the rate at which stiffness changes — is **4.2×
larger**. Sharpness costs, not the number of interfaces.

### 1.6 What we start with, and what happens

At `t = 0` we impose a specific shape, the **initial condition**:

```
g(x) = derivative of a Gaussian, scaled so its largest value is exactly 1
       width parameter sigma_g = 0.1, centred at x = 0
```

Numerically:

```
  g(-0.30) = +0.0549      g(+0.00) =  0.0000
  g(-0.20) = +0.4463      g(+0.05) = -0.7275
  g(-0.10) = +1.0000      g(+0.10) = -1.0000   <- the two peaks
  g(-0.05) = +0.7275      g(+0.20) = -0.4463
                          g(+0.30) = -0.0549
```

It is a small wiggle near the origin: one bump up at `x = -0.1`, one bump down at
`x = +0.1`, and essentially nothing beyond `|x| > 0.3`. We also start it **at
rest**, so `u_t(x,0) = 0` — the rod is deformed but not yet moving.

**What the physics then does.** For a uniform rod, the exact answer is known in
closed form — **d'Alembert's solution**:

```
u(x,t) = 1/2 [ g(x - t) + g(x + t) ]
```

The initial bump splits into two half-height copies of itself: one travelling
right (`x - t` stays constant as `t` grows, so it moves right at speed 1) and one
travelling left. Worked examples, all real values from this problem:

```
u(+0.10, 0.00) = 0.5·[g(+0.10) + g(+0.10)] = 0.5·[-1.0000, -1.0000] = -1.000000
u(+0.20, 0.10) = 0.5·[g(+0.10) + g(+0.30)] = 0.5·[-1.0000, -0.0549] = -0.527473
u(+0.40, 0.30) = 0.5·[g(+0.10) + g(+0.70)] = 0.5·[-1.0000, -0.0000] = -0.500000
u(-0.60, 0.50) = 0.5·[g(-1.10) + g(-0.10)] = 0.5·[+0.0000, +1.0000] = +0.500000
```

Read the third line: at time 0.3, the right-moving half-pulse has travelled 0.3
units, so the peak that started at `x = +0.1` is now at `x = +0.4`, at half height.

**This closed form is the reason homogeneous is our cleanest test**: we can score
models against the exact truth with zero discretisation error. For twolayer and
multilayer no closed form exists and we use a fine finite-difference solution
instead.

### 1.7 The boundaries

The rod is only `[-1, +1]`, but physically we want to model an *infinite* rod. If
we did nothing, waves would bounce off the ends and come back, polluting the
answer. So we impose **absorbing boundary conditions**:

```
at the right edge:  u_t - c·u_x = 0
at the left  edge:  u_t + c·u_x = 0
```

These let an outgoing wave leave without reflecting. They are the mathematical
equivalent of the rod continuing forever.

### 1.8 Why not just use a normal solver?

We could — and we do, to generate reference answers. A **finite-difference**
solver chops space and time into a fine grid and steps forward. That is the
classical approach and it works.

The question this project asks is different: **can a neural network learn the
solution directly from the equation, without ever being shown an answer?** That
is what a Physics-Informed Neural Network (PINN) is. Instead of training on
example solutions, you train on *how badly the network violates the physics*.

The reason anyone cares: classical solvers need a grid, and grids become
prohibitively expensive in high dimensions or complicated geometries. A network
is grid-free. Whether that trade actually pays off is an open research question,
and this study is one careful data point on one problem.

---

# PART B — THE SHARED PIPELINE

Everything in this part is **identical across every architecture in the study**.
Only the network in the middle changes. That is deliberate: if the pipeline varied
too, you could never attribute a difference in results to the architecture.

---

## 2. From coordinates to a loss number

The whole training loop is four steps, repeated:

```
 (1) pick points in space-time        ->  §2.1
 (2) push them through the network    ->  §2.3 (and Part C for the networks)
 (3) measure how badly the physics is violated at those points  ->  §2.4
 (4) nudge the network's numbers to reduce that violation       ->  §6
```

### 2.1 Step 1 — the collocation points

#### What a collocation point is

The network must satisfy the wave equation *everywhere* in the space-time
rectangle `x ∈ [-1,1]`, `t ∈ [0,1]`. That is infinitely many places, so we cannot
check them all. Instead we pick a finite set of sample locations and check the
equation there. Those sample locations are called **collocation points**.

A collocation point is nothing more exotic than **a pair of numbers `(x, t)`** — a
place and a moment. We use **10,000 of them**.

Here are the first eight actual points our code generates for seed 0:

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
−0.759 on the rod, at time 0.261.* And so on for all 10,000.

#### How they are stored

Two tensors — a tensor here is just a rectangular array of numbers:

```
x :  shape (10000, 1)   dtype float32
t :  shape (10000, 1)   dtype float32
```

`(10000, 1)` means 10,000 rows and 1 column: a tall thin column of numbers. Row `i`
of `x` and row `i` of `t` together are collocation point `i`. They are kept as
separate tensors rather than one `(10000, 2)` tensor because we need to
differentiate with respect to `x` and `t` *separately* (§2.4).

#### Why 10,000, and not 100 or 10,000,000

- **Too few** and the network can satisfy the equation exactly at the sample
  points while drifting wildly between them — it "memorises the exam questions".
- **Too many** and every training step becomes slow, because each point requires
  second derivatives through the whole network.

10,000 is the value the field's benchmark paper (arXiv:2602.15068) uses for 2-D
PDE problems, so we match it. It is also enough that the average spacing between
points is about `sqrt(2·1/10000) ≈ 0.014` — several times finer than the pulse
width (`sigma_g = 0.1`), so the pulse is well resolved by the sample.

#### What "Sobol" means and why not just random

The obvious way to pick 10,000 points is uniformly at random. The problem with
random is **clumping**: purely random points leave gaps in some places and pile up
in others, purely by chance.

A **Sobol sequence** is a *low-discrepancy* sequence — constructed so that any
sub-rectangle of the domain gets its fair share of points, much more evenly than
random would. "Scrambled" means a random permutation is applied so different seeds
give genuinely different (but still well-spread) point sets.

The practical effect: for the same 10,000 points, the residual is sampled more
evenly, so the loss is a better estimate of the true violation over the whole
domain.

#### "Drawn once and frozen" — and why that matters enormously

We generate the 10,000 points **one time**, before training starts, and then use
**the exact same 10,000 points for every single training step**.

This is not the usual deep-learning practice, where you would resample each step.
The reason is the optimiser. **L-BFGS** (§6) works by building a model of the
loss surface's curvature from how the gradient changed over the last ~100 steps.
That model is only meaningful if the loss surface is *the same surface* each time.
Resample the points every step and the surface moves under it; the curvature
estimate becomes noise and the optimiser fails.

Getting this wrong made every PINN in an earlier phase look 10–70× worse than it
actually is.

One consequence worth noting: the same Sobol draw is **byte-identical** between
our two training paths for a given seed. We verified this rather than assuming it,
so a difference between the two paths cannot be blamed on different data.

#### The boundary points

Separately from the 10,000 interior points, we draw **512 time values** `t_b`, used
to check the absorbing boundary conditions at `x = -1` and `x = +1`. They are also
fixed for the whole run.

### 2.2 The evaluation grid — a different set of points, for a different purpose

The 10,000 collocation points are for **training**. To *score* a trained model we
use a completely separate, structured grid:

```
4096 evenly spaced positions      x = -1.000, -0.9995, ..., +1.000
  20 specific time snapshots      t = 0.05, 0.10, 0.15, ..., 1.00
--------------------------------------------------------------
4096 × 20 = 81,920 evaluation points
```

Think of it as **20 photographs of the rod**, each with 4096 pixels. We ask the
model for its displacement at all 81,920 points, ask the reference solution for the
same, and compare.

Why a regular grid here rather than Sobol: for *scoring* we want even coverage so
the error metric is a fair average over the domain, and we want the same grid the
reference solution is defined on.

Why those 20 times and not the first instants: at `t < 0.05` the solution has barely
evolved and every model looks perfect, which would flatter everything equally.

#### The error metric

```
                ||u_model - u_reference||
rel-L2  =  100 · -------------------------  percent
                    ||u_reference||
```

where `|| · ||` means: square every one of the 81,920 differences, add them up,
take the square root. It is the ordinary Euclidean distance between two lists of
81,920 numbers, expressed as a percentage of the size of the true answer.

So **0.02 %** means the model's error is one five-thousandth of the size of the
solution itself. **95 %** means the model is essentially producing nothing.

This is the standard metric in the PINN literature, and matches the benchmark
paper's definition exactly.

### 2.3 Step 2 — the hard-constraint ansatz

This is the piece the rest of the document leans on most, so it gets the most
space.

#### The problem it solves

The network must produce a solution that satisfies **three** things at once:

1. the wave equation, everywhere inside the domain
2. the initial shape: `u(x, 0) = g(x)`
3. the initial stillness: `u_t(x, 0) = 0`

The straightforward approach is to add penalty terms to the loss for (2) and (3)
and hope the optimiser balances all three. That is called a **soft constraint**,
and it is what most of the literature does. It has a known weakness: the balance
between the terms is a hyperparameter, and if it is wrong the network trades away
the initial condition to reduce the PDE residual.

The alternative — what we do — is to make (2) and (3) **impossible to violate**, by
construction. That is a **hard constraint**, and the construction is called an
**ansatz** (German for "approach" or "setup" — a chosen functional form with free
parts inside it).

#### The construction

The network does **not** output `u`. Call the network's raw output `N(x,t)`. We
then build the solution as:

```
u(x,t)  =  g(x) · decay(t)   +   growth(t) · N(x,t)
           \_____________/       \_________________/
            term A                term B
            the known start       the learned correction
```

with two fixed helper functions:

```
decay(t)  = exp( -1/2 · (15t)^2 )     starts at 1, dies quickly
growth(t) = tanh^2( 25t )             starts at 0, rises quickly
```

Neither has any trainable parameters. They are fixed shapes chosen so the algebra
works out.

#### Why this forces the initial condition

Evaluate at `t = 0`:

- `decay(0) = exp(0) = 1`
- `growth(0) = tanh²(0) = 0`

so

```
u(x, 0)  =  g(x)·1  +  0·N(x,0)  =  g(x)          <- condition (2), exactly
```

The network's output is multiplied by zero at `t = 0`. **No matter what the network
does, no matter how badly trained, `u(x,0) = g(x)` exactly.** It cannot be violated.

For the velocity, differentiate with respect to `t` and set `t = 0`. Both helper
functions have **zero slope** at the origin:

- `decay'(t) = -225·t·exp(...)`, which is `0` at `t = 0`
- `growth'(t) = 50·tanh(25t)·sech²(25t)`, which is `0` at `t = 0` because `tanh(0) = 0`

so every term in `u_t(x,0)` carries a factor that vanishes, giving

```
u_t(x, 0) = 0                                      <- condition (3), exactly
```

That is the whole trick. Two conditions removed from the optimisation problem for
free, permanently.

#### What the helper functions actually do, numerically

```
     t    decay(t)   growth(t)     sum
  0.00    1.000000    0.000000   1.0000     <- pure initial condition
  0.01    0.988813    0.059985   1.0488
  0.02    0.955997    0.213552   1.1695
  0.05    0.754840    0.719585   1.4744     <- both active; hand-over region
  0.08    0.486752    0.929349   1.4161
  0.10    0.324652    0.973408   1.2981
  0.15    0.079560    0.997790   1.0773
  0.20    0.011109    0.999818   1.0109     <- decay essentially dead
  0.30    0.000040    0.999999   1.0000
  0.50    0.000000    1.000000   1.0000     <- u = N exactly
  1.00    0.000000    1.000000   1.0000
```

Read the story in that table:

- **`t = 0`**: the answer is 100 % the known initial shape, 0 % network.
- **`t ≈ 0.05`**: hand-over. Both terms contribute.
- **`t > 0.2`**: `decay` has died to 0.011 and `growth` has saturated at ~1. From
  here on, `u ≈ N` — **the network alone is the answer**.

Since our evaluation window is `t ∈ [0.05, 1.0]`, **almost all of what we score is
the raw network output.** Remember this: it is the root of the collapse failure in
§7 and §8.

(The `sum` column peaking at 1.47 rather than 1 is a known wart of this particular
ansatz — the two terms overlap rather than partitioning cleanly. A cleaner-looking
alternative, `g(x) + t²·N`, was tested and **loses 6/6 by two orders of
magnitude**; see `BUGS_AND_CORRECTIONS.md`.)

#### A fully worked example

Take one point, `x = +0.10`, and an untrained network. At `x = 0.1` the initial
pulse is at its negative peak, `g(0.1) = -1.0000`.

```
t = 0.00 :  g=-1.0000  decay=1.0000  growth=0.0000  N=-0.8022
            u = (-1.0000)(1.0000) + (0.0000)(-0.8022) = -1.000000
            the network's -0.8022 is annihilated by growth=0

t = 0.05 :  g=-1.0000  decay=0.7548  growth=0.7196  N=-0.2902
            u = (-1.0000)(0.7548) + (0.7196)(-0.2902) = -0.963676
            mostly still the initial condition, network starting to matter

t = 0.30 :  g=-1.0000  decay=0.0000  growth=1.0000  N=+0.3908
            u = (-1.0000)(0.0000) + (1.0000)(+0.3908) = +0.390720
            the initial condition has vanished; u IS the network output
```

The true answer at `(0.4, 0.3)` is `-0.5`, and this untrained network gives
`+0.39` at `(0.1, 0.3)` — so it is, as expected, nonsense before training. What
matters is that **the machinery is correct**: at `t = 0` it is exactly right by
construction, regardless of the network.

### 2.4 Step 3 — the loss

#### What "residual" means

Rearrange the wave equation so everything is on one side:

```
r(x,t)  =  rho(x)·u_tt  -  d/dx( E(x)·u_x )
```

If `u` is the true solution, `r = 0` everywhere. If `u` is wrong, `r` is some
non-zero number measuring **how badly the physics is violated at that point**. `r`
is called the **residual**.

Crucially, computing `r` **never requires knowing the answer.** It only requires
the network and the equation. That is what makes this "physics-informed" rather
than supervised learning.

#### How the derivatives are computed

`u_tt` and `u_x` are not finite differences. They come from **automatic
differentiation** (`torch.autograd.grad`), which computes exact derivatives of the
network's output with respect to its inputs by applying the chain rule through
every operation the network performed.

The chain for `u_tt`:

```
u = ansatz(N(x,t), x, t)
u_t  = autograd.grad(u,   t, create_graph=True)     first derivative
u_tt = autograd.grad(u_t, t)                        second derivative
```

`create_graph=True` on the first call is essential — it tells PyTorch to remember
*how* it computed `u_t`, so that `u_t` can itself be differentiated. Forget it and
the second derivative silently does not exist.

**This is where the whole study's difficulty lives.** The loss depends on *second*
derivatives of the network. A network can be an excellent approximation of `u` and
a terrible approximation of `u_xx`, because differentiating amplifies error. §8 is
entirely about the consequences.

#### The shapes

```
input   x (10000,1)  t (10000,1)
        |
        |  network + ansatz + two rounds of autograd
        v
residual  r (10000,1)      one number per collocation point
```

For an untrained network those numbers look like `[-87.5, -137.4, -15.1, +151.0,
+36.5, ...]` — large and arbitrary. For a converged model they are around `1e-3`.

#### Assembling the loss

```
L  =  mean( r^2 )   +   mean( b^2 )
      \_________/       \_________/
      PDE term          boundary term
      10,000 points     512 points per edge
```

`b` is the absorbing-boundary residual `u_t ∓ c·u_x` evaluated at `x = ±1`.
Squaring makes every violation positive so they cannot cancel; the mean makes the
number independent of how many points we used.

**One number.** That single scalar is what the optimiser minimises. There is no
initial-condition term, because §2.3 made it structurally impossible to violate.

#### Why the loss is not normalised

An earlier version divided the loss by `residual_scale² ≈ 5e4` to bring it near 1.
That is normally harmless, but L-BFGS chooses its very first step size as
`min(1, 1/||gradient||)·lr`. Shrinking the loss shrinks the gradient by the same
factor, so `1/||gradient||` explodes and step one overshoots catastrophically.

Measured consequence: MLP on twolayer went **95.59 % → 0.86 %** simply by removing
the normalisation. It is defect #1 in `BUGS_AND_CORRECTIONS.md` and it invalidated
an entire phase of the study.

---

# PART C — THE NETWORKS

---

## 3. What a neural network is, in the terms this study needs

Skip if you know this. It is here so §4 and Part D can be read without gaps.

### 3.1 A layer

The basic building block is a **linear layer**, written `Linear(in, out)`. It takes
a vector of `in` numbers and produces a vector of `out` numbers:

```
output_j  =  sum over i of ( W[j,i] · input_i )  +  b[j]
```

`W` is a matrix of `out × in` numbers, `b` a vector of `out` numbers. Both are
**trainable parameters** — the numbers training adjusts. `Linear(2, 128)` holds
`128×2 = 256` weights plus `128` biases = 384 parameters.

Stacking linear layers alone is pointless: a composition of linear maps is still
linear, and our solution is not. So between them you put a fixed nonlinear
function applied element by element — an **activation function**.

We use **`tanh`**, which squashes any real number into `(-1, 1)` in a smooth
S-shape. The choice matters here for a specific reason: `tanh` is smooth to *all*
orders, so `u_xx` exists and is continuous. The popular alternative `relu`
(`max(0,x)`) has a second derivative of zero almost everywhere, which would make
the PDE residual blind to the network. A KAN variant with `relu` collapses to
93.6 % — see §5.

### 3.2 "Training"

The network starts with random numbers in `W` and `b`. Training means:

1. compute the loss (§2.4)
2. compute the **gradient** — how much the loss would change if each parameter were
   nudged up slightly. This is one number per parameter, obtained by backpropagation.
3. move every parameter a little in the direction that reduces the loss
4. repeat

The **optimiser** is the rule for step 3. Which optimiser you use turns out to be
the single most consequential choice in this entire study (§6).

### 3.3 What "82,689 parameters" means

Just the total count of trainable numbers in `W`s and `b`s. It is a rough proxy for
model capacity. The field's convention when comparing two architectures is to match
these counts so you are comparing *architectures* and not *sizes* — a convention we
initially failed to follow (§5.1).

---

## 4. How a KAN actually works

### 4.1 The idea, against the MLP you now know

An **MLP** puts **fixed** nonlinearities on the *nodes* and **learned** weights on
the *edges*:

```
node output  =  tanh( sum_i  w_i · x_i  +  b )
                ^^^^        ^^^
                fixed       learned scalars
```

Each edge contributes just a **scalar multiply**. All the shape comes from the
fixed `tanh` at the node.

A **KAN** inverts this completely. It puts **learned univariate functions on the
edges** and **plain summation on the nodes**:

```
node output  =  sum_i  phi_i( x_i )
                       ^^^^^^
                       a learned FUNCTION of one variable
```

There is no weight matrix. There is no activation function in the MLP sense. **Each
edge carries its own little curve**, and training reshapes those curves.

The picture to hold: in an MLP, an edge can only scale a signal up or down. In a
KAN, an edge can bend it — turn a straight line into an S, a bump, a step,
whatever the data demands.

### 4.2 Why anyone thought this was a good idea

The **Kolmogorov–Arnold representation theorem** says any continuous function of
many variables can be written as a finite composition of sums of functions of *one*
variable. An MLP approximates a multivariate function head-on; a KAN builds it out
of one-dimensional pieces, which the theorem says is always possible in principle.

**Why it matters for our PDE specifically** — and this is the crux of the whole
study. The exact homogeneous solution (§1.6) is

```
u(x,t) = 1/2[ g(x-t) + g(x+t) ]  =  F(xi) + G(eta)     where xi = x-t, eta = x+t
```

That is **literally a sum of two univariate functions**. It is a Kolmogorov–Arnold
representation with one layer and two edges. If you feed a KAN the coordinates
`(xi, eta)` instead of `(x, t)`, its very first layer can express the exact
solution. §4.7 and §8.4 return to this.

### 4.3 The edge function, concretely

Each edge computes

```
phi(x)  =  scale_base · silu(x)   +   scale_sp · spline(x)
           \_________________/         \_______________/
           branch 1: a fixed shape      branch 2: a fully
           with one learned amplitude   learned curve
```

(`KANLayer.py:161` in pykan.) `silu(x) = x·sigmoid(x)` is a fixed smooth function,
similar to `relu` but differentiable everywhere. Both `scale_base` and `scale_sp`
are trained.

Branch 1 is a **residual path** — it gives the edge a sensible default shape before
the spline has learned anything, so the network is not useless at initialisation.
Branch 2 is where the expressive power lives.

### 4.4 B-splines — what branch 2 is made of

A **spline** is a curve built from polynomial pieces joined smoothly. A **B-spline**
is a particular way of building one that has two properties we care about.

Picture the input range `[-1, +1]` cut into equal intervals by **knots** — the
joining points. Over each interval the curve is a polynomial; at each knot the
pieces are glued so that the curve and several of its derivatives match.

Two numbers define the family:

| term | meaning | our best KAN (`charcoords50`) |
|---|---|---|
| **grid** | how many intervals the range is cut into | `grid = 50` |
| **k** (order) | polynomial degree on each interval | `k = 5` |
| **grid_range** | the interval the knots span | `[-1, +1]` (pykan default) |

From those two numbers everything else follows:

```
number of basis functions  =  grid + k       =  50 + 5  =  55
number of knots stored     =  grid + 2k + 1  =  50 + 11 =  61
knot spacing               =  2 / 50                    =  0.04
```

Both of those appear directly in the parameter dump: the coefficient tensor's last
dimension is **55**, and the `grid` buffer holds **61** numbers. If you ever wonder
where a shape came from, it is one of these two formulas.

The curve on one edge is then just a weighted sum:

```
spline(x)  =  sum over 55 basis functions of  coef[j] · B_j(x)
```

The 55 `coef` values are the trainable numbers. Training reshapes the curve by
adjusting them.

#### Local support — the property that makes KANs sharp, and fragile

At any given input `x`, only **`k + 1 = 6`** of the 55 basis functions are non-zero.
All the others are exactly zero there.

Consequence one: **one coefficient only affects the curve near its own knot.** You
can add detail in one region without disturbing another. This is why a KAN can
represent fine structure that an equally-sized MLP cannot — an MLP's every weight
affects every input.

Consequence two, which bites us later: because the coefficients are tied to
*specific knot positions*, **moving the knots invalidates the coefficients**. §8.2
is entirely about what happens when you do.

#### Why `k` matters for a PDE specifically

A B-spline of order `k` is `C^(k-1)` — continuous with `k-1` continuous derivatives.

- At **`k = 3`**: the network is `C²`. So `u_xx` — which the loss consumes — is only
  `C⁰`: **piecewise linear, with a kink at every one of the 50 knots.**
- At **`k = 5`**: the network is `C⁴`, so `u_xx` is `C²` — genuinely smooth.

For a second-order PDE that is a principled change, not a knob-twiddle. It is one
of the three changes that turns the plain KAN into the winning one.

#### How the spline is evaluated

By the **Cox–de Boor recursion** (`coef2curve` in pykan): start with indicator
functions that are 1 on their own interval and 0 elsewhere, then raise the degree
`k` times, each round blending neighbouring basis functions linearly in `x`. After
`k` rounds you have smooth, locally-supported basis functions.

### 4.5 A KAN layer, with every shape

A layer with `in_dim` inputs and `out_dim` outputs holds `in_dim × out_dim` edges —
one for every input-output pair — each with its own spline. Concretely, for
`act_fun[1]` of our best KAN:

```
in_dim = 20, out_dim = 20, grid = 50, k = 5

coef        (20, 20, 55)  = 22,000 numbers   one 55-coefficient spline per edge
scale_base  (20, 20)      =    400 numbers   silu amplitude, per edge
scale_sp    (20, 20)      =    400 numbers   spline amplitude, per edge
grid        (20, 61)      buffer, not trained
                          knot positions, ONE ROW PER INPUT
                          (shared across that input's 20 outgoing edges)
```

The forward pass, for a batch of `N` points:

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
  sum over the INPUT axis  (axis 1)                     -> (N, 20)
  |
output                                                     (N, 20)
```

The node does nothing but add up its 20 incoming edge values. **All the modelling
lives in the 400 edge functions.**

### 4.6 Every term, collected

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
| `update_grid` | move the knots to match observed activations, then re-fit via `curve2coef` |
| `symbolic_fun` | a parallel branch for symbolic regression — **disabled** in all our runs |

Two are load-bearing for our results. `update_grid` destroys our runs 8/8 (§8.2),
and `curve2coef` returned NaN on GPU until we patched it (`BUGS_AND_CORRECTIONS.md`).

### 4.7 Characteristic coordinates — the change that wins

This is not a KAN feature; it is a change to what we feed the KAN.

Define the **travel-time coordinate**

```
tau(x)  =  integral from 0 to x of  ds / c(s)
```

— how long a wave takes to reach `x`. For homogeneous material `c = 1` so
`tau(x) = x`. Then form

```
xi  = tau(x) - t        the "right-moving" coordinate
eta = tau(x) + t        the "left-moving" coordinate
```

and rescale each to `[-1, +1]` so they land exactly on pykan's knot range.

Why this is the whole ball game: in these coordinates the wave operator becomes

```
d²/dxi·d/eta
```

so **any** function of the form `F(xi) + G(eta)` satisfies the wave equation
*identically* — differentiate `F(xi)` with respect to `eta` and you get exactly
zero, for any `F` whatsoever.

And a KAN layer computes precisely a sum of univariate functions. So in these
coordinates, **the KAN's architecture and the solution's structure are the same
shape**. §8.4 shows what that buys, measured.

`tau(x)` is built by quintic Hermite interpolation of a Gauss–Legendre quadrature
of `1/c(x)`, in float64, accurate to `tau' = 1.5e-7` and `tau'' = 1.9e-6`. It is a
fixed buffer — computed once, never trained.

---

# PART D — THE ARCHITECTURES

`N` is the batch size throughout — 10,000 during training, 81,920 at evaluation.
Every dimension below was read off the live models by introspection, not written
from memory.

Each model is a black box that takes `(x, t)` and returns one number `N(x,t)`,
which then goes into the ansatz of §2.3 to become `u`.

---

## 5. Model by model

### 5.1 What the Fourier embedding is (needed by three models below)

Two of our architectures start by transforming `(x,t)` into a much longer vector
of sines and cosines. This is a **random Fourier feature embedding**:

```
p     = [x, t] @ B.T          B is a FIXED 64×2 matrix of random numbers
                              drawn from Normal(0, sigma_B²), sigma_B = 10
output = [cos(p), sin(p)]     -> 128 numbers
```

Each of the 64 rows of `B` defines a plane wave in `(x,t)` with a random direction
and a random frequency of typical size 10. The embedding reports how much the input
point aligns with each of those 64 waves, as a cosine and a sine.

**Why do it.** A plain `tanh` MLP is biased toward smooth, low-frequency functions
and needs great depth to produce oscillations. Handing it 128 oscillating features
up front removes that burden. `sigma_B = 10` was tuned to the measured spectral peak
of our solution, `k_peak ≈ 9.4` (`DECISIONS.md` D4).

**`B` is a buffer, not a parameter** — it is never trained, so the bandwidth stays
at its declared value for the whole run and is a controlled variable rather than a
learned one.

**The price, which §8 collects.** Differentiating a `cos(2π k · x)` brings down a
factor of `2π k ≈ 190`; differentiating twice brings down `(2π k)² ≈ 3.5e4`. The
embedding that makes the *function* easy to fit makes its *second derivatives*
noisy — and the loss is built entirely from second derivatives.

### 5.2 MLP PINN — `mlp` — 66,561 parameters

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

Five hidden layers of 128 units. **Best model on twolayer** (0.4318 ± 0.3070) —
the plainest architecture wins on the sharpest material, which is worth sitting
with. Fourier features tuned to the pulse's spectrum do not help with an interface
discontinuity, because a discontinuity is not a single frequency.

### 5.3 Fourier PINN — `fourier` — 82,689 parameters

The same MLP with the §5.1 embedding in front.

```
x (N,1), t (N,1)
  concatenate                                -> (N, 2)
  @ B.T        B is (64,2), FIXED            -> (N, 64)
  [cos(p), sin(p)]                           -> (N, 128)
  Linear(128 -> 128) + tanh                  -> (N, 128)   16,384 + 128
  ... four more 128-wide tanh layers ...
  Linear(128 -> 1)                           -> (N, 1)
```

The headline PINN for most of this study. 0.0399 ± 0.0138 on homogeneous.

### 5.4 Fourier PINN, parameter-matched — `fourier_matched` — 49,729 parameters

Identical to §5.3 except the hidden width is **96** rather than 128, chosen so the
parameter count matches the tuned KAN's 49,020 to within 1.45 %.

```
  Linear(128 -> 96) + tanh    <- narrows from the 128-d embedding
  Linear(96  -> 96) + tanh    x4
  Linear(96  -> 1)
```

**Why it exists.** The benchmark standard (arXiv:2602.15068) matches parameter
counts so you compare architectures rather than sizes. Ours did not: the KAN was
winning with 40 % *fewer* parameters, which favoured us but left the control
missing.

**And it is better than the 128-wide version** — 0.0332 % against 0.0401 %. The
default was over-parameterised, so the headline PINN arm had been handicapped by
its own size.

### 5.5 PirateNet — `pirate` — ~150,000 parameters

A published PINN architecture. Fourier embedding, then residual blocks using
**random weight factorisation** — each weight matrix is stored as
`W = diag(exp(s))·V` with both `s` and `V` trained, which decouples the scale of a
weight from its direction.

Its output layer is zero-initialised, so without intervention the network is
identically zero at init, `u_tt = 0`, and the backbone receives no gradient. A
physics-informed least-squares step at construction fixes this.

Competitive on homogeneous (0.0296 ± 0.0024 — **the lowest seed variance of any
model**) and multilayer (0.0563), but **2.768 %** on twolayer: a 25× inversion on
the sharp interface, the same material where the plain MLP wins.

### 5.6 Plain KAN — `pykan_wide` — 8,600 parameters

The reference pykan implementation on raw `(x,t)`. Width `[2,20,20,20,1]`,
`grid = 5`, `k = 3`.

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

860 edges total, 8 coefficients each = 6,880, plus 1,720 scale parameters = 8,600.

**A real handicap worth naming.** `PyKAN.forward` is `self.kan(cat([x,t]))` with no
input scaling, and pykan's knot range is `[-1,1]`. Our `x` fills that range, but
`t` only covers `[0,1]` — so **roughly half the knots along the time axis are never
visited by any input.** Phase 8's `normalise` rung tested exactly this and it
mattered.

Despite that: 0.0849 % on homogeneous from 8,600 parameters — an order of magnitude
fewer than any PINN here.

### 5.7 The tuned KAN — `charcoords50` — 49,020 parameters — **the winner**

Same pykan backbone as §5.6, three changes.

```
x (N,1), t (N,1)
  |
  |  CharCoords (§4.7):   tau(x) = travel-time coordinate
  |                       xi  = tau(x) - t
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

The three changes from `pykan_wide`, and why each is principled rather than tuned:

| change | from | to | reason |
|---|---|---|---|
| input coordinates | raw `(x,t)` | `(tau-t, tau+t)` | the exact solution is `F(xi)+G(eta)`, a literal sum of univariate functions — the KAN's own structure. §4.7. |
| `grid` | 5 | 50 | 10× resolution; knot spacing 0.04 against a pulse of width ~0.24 |
| `k` | 3 | 5 | `u` becomes `C⁴`, so `u_xx` is `C²` rather than kinked at every knot. §4.4. |

This model beats every PINN on homogeneous. §8.4 explains why in one sentence.

### 5.8 The PIKANs — both fail, for different reasons

"PIKAN" in the literature means a spline KAN on a Fourier embedding. We tried two.

**`splinekan` (ours) — 246,208 parameters.** A hand-rolled spline layer on a 128-d
Fourier embedding.

```
x, t -> FourierEmbed(64, sigma=10)                      -> (N, 128)
     -> _SplineKANLayer(128 -> 64, grid=10, order=4) -> LayerNorm -> (N, 64)
     -> _SplineKANLayer(64 -> 64)  x2                    -> (N, 64)
     -> Linear(64 -> 1)                                  -> (N, 1)
```

**It is broken before training even starts.** `max|u_tt| = 1.16e6` against every
other model's ~2e2; PDE loss `3.5e10` and gradients `4.4e10` at step zero.

The cause is exactly §5.1's price compounded with §4.4's local support: B-splines
placed **directly** on a `sigma_B = 10` Fourier embedding, so the chain rule
multiplies `(2π k)² ≈ 3.5e4` from the embedding by `(1/knot spacing)² ≈ 25` from
the spline. pykan avoids this by damping its spline branch by `1/sqrt(fan_in)`
(`KANLayer.py:112`); plain Xavier initialisation does not.

`splinekan_fix` applies the same damping and drops the init loss **100,000×** —
which converts a divergence (235 %) into a collapse (97 %). Better in kind, not in
outcome.

**`fourier16` (pykan backend) — 36,500 parameters.** The same idea done correctly:
the reference implementation on a 16-feature embedding, `grid = 20`, `k = 3`.

```
x, t -> FourierEmbed(16, sigma=10)   -> (N, 32)
     -> act_fun[0]  32 -> 20         -> (N, 20)
     -> act_fun[1..2]  20 -> 20      -> (N, 20)
     -> act_fun[3]  20 -> 1          -> (N, 1)
```

Initialisation is **healthy** (`max|u_tt| = 2.39e2`). It collapses anyway, on all
three materials, under both optimisers, at ~95 %. §8.3 explains why, and the answer
is not what it looks like.

### 5.9 WavKAN — `wavkan`

Mexican-hat wavelets on raw `(x,t)`: `psi(z) ∝ (z² - 1)·exp(-z²/2)` with a learned
scale and translation per edge. A wavelet is a localised wiggle, so this is a KAN
whose edge functions are bumps rather than splines.

Diverges under L-BFGS through a float32 overflow inside torch's cubic line search
(`d1² ≈ 2.7e49`), and is the only architecture that one-shot residual resampling
ever rescued — 4 times out of 4, from ~95 % to 0.8–5.2 %.

---

# PART E — RESULTS AND MECHANISMS

---

## 6. What each model achieves

### 6.1 Reading the statistics

Every number below is `mean ± standard deviation (n)`, where **n is the number of
seeds** — independent training runs differing only in random initialisation and the
random scramble of the Sobol points. Same architecture, same data, same optimiser;
only luck differs.

Seeds matter enormously here. **Five separate findings in this project dissolved
when the seed count went up.** A single run tells you almost nothing.

The tests used, in plain terms:

| test | question it answers | why we use it |
|---|---|---|
| **Welch t-test** | are the two *averages* different? | does not assume the two groups have equal spread, which ours do not |
| **Mann–Whitney U** | are one group's values systematically *ranked* higher? | makes no assumption that the data is bell-shaped |
| **paired Wilcoxon** | comparing *seed by seed*, does one win more often and by more? | our two arms use identical seeds and identical collocation sets, so pairing removes shared noise and is far more sensitive |
| **Levene** | are the two groups' *spreads* different? | the KAN's inconsistency was the main argument against it |

**`p`** is the probability of seeing a difference at least this large if there were
truly no difference. `p < 0.05` is the conventional threshold for "unlikely to be
chance". It is a convention, not a law, and `p = 0.049` and `p = 0.051` mean nearly
the same thing.

### 6.2 Homogeneous — the KAN wins

Scored against an nx=2048 finite-difference reference.

| rank | model | rel-L2 | params |
|---|---|---|---|
| 1 | **`charcoords50` KAN** | **0.0236 ± 0.0129 (11)** | 49,020 |
| 2 | `fourier_matched` PINN | 0.0332 ± 0.0152 (11) | 49,729 |
| 3 | `fourier` PINN | 0.0399 ± 0.0138 (11) | 82,689 |
| 4 | `pykan_wide` plain KAN | 0.0849 ± 0.0201 (5) | 8,600 |
| 5 | `mlp` PINN | 0.2555 ± 0.1232 (5) | 66,561 |
| — | PIKAN, either implementation | ~95 % COLLAPSED | — |

Against the **exact d'Alembert solution** (§1.6) — zero discretisation error:

```
charcoords50   0.0222 ± 0.0139        fourier   0.0401 ± 0.0175
KAN ahead 1.81x   Welch p=0.0154   Mann-Whitney p=0.0126   paired 9/11   Wilcoxon p=0.0322
```

At **matched parameters** (§5.4) the lead narrows to **1.50×** and the tests
disagree: Welch p = 0.092, Mann–Whitney p = 0.049, paired Wilcoxon p = 0.0068. The
paired test is the appropriate one for this design, but the disagreement should be
reported rather than the favourable test selected.

### 6.3 Two-layer — the plain MLP wins, but not significantly

| rank | model | rel-L2 |
|---|---|---|
| 1 | **`mlp` PINN** | **0.4318 ± 0.3070 (11)** |
| 2 | `charcoords50` KAN | 0.5984 ± 0.3510 (11) |
| 3 | `fourier` PINN | 0.9579 ± 0.7969 (11) |
| 4 | `pykan_wide` | 1.0340 ± 0.4055 (5) |

**p = 0.250 — no significant difference.** An earlier reading of this cell (PINN
wins 2.20×) came from five seeds and did not survive eleven.

Twolayer is the hardest material for every architecture, by roughly 20×. §1.5
explains why: its interface is 2.5× narrower, making `E'` 4.2× larger. The
characteristic transform also injects `tau'' = -c'/c²` into `u_xx` through the
chain rule, and that term is 3.7× larger here — so the coordinate change that makes
homogeneous separable actively hurts on a sharp interface.

### 6.4 Multi-layer — a tie

| rank | model | rel-L2 |
|---|---|---|
| 1 | `charcoords50` KAN | 0.0403 ± 0.0114 (11) |
| 2 | `fourier` PINN | 0.0432 ± 0.0115 (11) |
| 3 | `pykan_wide` | 0.2320 ± 0.0984 (5) |
| 4 | `mlp` PINN | 0.6915 ± 0.2701 (11) |

p = 0.571. Six interfaces, and **every model does better here than on the single
sharp one** — the clearest confirmation that sharpness costs, not interface count.

### 6.5 The same comparison under the other optimiser

| material | best PINN | tuned KAN | verdict |
|---|---|---|---|
| homogeneous | `fourier` 0.0281 ± 0.0073 (11) | 0.0313 ± 0.0170 (11) | **tie**, p = 0.577 |
| multilayer | `fourier` 0.0419 ± 0.0146 (11) | 0.0574 ± 0.0375 (11) | PINN nominally |

**Same networks, same points, different optimiser, different answer.** That is §7.

---

## 7. Why Adam→L-BFGS beats pure L-BFGS

### 7.1 What the two optimisers are

Both answer the same question — given the gradient, how do I move the parameters?

**Adam** moves each parameter by roughly a fixed step size `lr`, in the direction
that reduces the loss, *regardless of how big that parameter's gradient is*. It
divides by a running estimate of the gradient's own magnitude, which makes it
scale-invariant. With `lr = 1e-3` over 700 steps, each parameter can travel up to
about 0.7 in total.

**L-BFGS** is a **quasi-Newton** method. It watches how the gradient changes over
the last ~100 steps to build a picture of the loss surface's *curvature*, then
jumps to the estimated bottom of that local bowl. It takes far better-informed
steps — but its first step is scaled `min(1, 1/||gradient||)·lr`, so a large
gradient forces a tiny step.

Two consequences follow, and they are the whole story:

1. **L-BFGS is local.** From a cold random start it descends into whatever bowl it
   happened to land in. It cannot climb out to reach a better one.
2. **Adam explores.** Its step size does not shrink when gradients are large, so it
   wanders across the landscape rather than settling immediately.

This is also why the collocation points must be frozen (§2.1): the curvature model
assumes the surface is not moving.

### 7.2 What changes

| | pure L-BFGS | Adam(700) → L-BFGS |
|---|---|---|
| Fourier PINN | 0.0281 ± 0.0073 | 0.0399 ± 0.0138 |
| tuned KAN | 0.0313 ± 0.0170 | **0.0236 ± 0.0129** |
| KAN variance vs PINN | **5.44×**, Levene p = 0.031 | **0.87×**, p = 0.73 |
| verdict | tie (p = 0.577) | KAN wins (p = 0.0098) |

The warm-up **helps the KAN and slightly hurts the PINN**. It is not a global
improvement; it is an *interaction* between optimiser and architecture. A paper
that benchmarks one optimiser and reports the result as an architecture property is
reporting this interaction without knowing it.

### 7.3 The measurement

On `splinekan`, from an identical initialisation, how far do the parameters travel
and how low does the loss get:

```
                  loss@init   loss@end    ||change in parameters|| / ||initial||
Adam lr=1e-3       3.39e10     5.17e5           0.1608
Adam lr=1e-5       3.39e10     2.68e6           0.0117
L-BFGS 35 epochs   3.39e10     7.89e4           0.0089
```

L-BFGS reaches a **6.5× lower loss while travelling 18× less** — about 115× more
efficient per unit of parameter movement. It is a far better local optimiser, and
that is precisely why it cannot escape a poor starting basin.

### 7.4 Why the KAN benefits more

The KAN's basis fits this solution far better *in principle* — §8.4 quantifies it —
but that potential lives in a region of parameter space that a cold quasi-Newton
start does not reach. Under pure L-BFGS the advantage bought **nothing**: the two
families tied.

The strongest evidence that this is a basin effect rather than a mean shift is the
**variance**. Under pure L-BFGS the KAN's seed-to-seed variance was 5.44× the
PINN's (p = 0.031) — the only statistically significant difference between the two
families, and the main argument against KANs. Under the hybrid it is 0.87×, not
significant. **The Adam warm-up did not improve an average; it removed a failure
mode.** Seeds that used to land in bad basins no longer do.

---

## 8. The one mechanism behind every failure

### 8.1 The PDE residual is a cancellation

Measure the trained `charcoords50` on homogeneous, where the equation is exactly
`u_tt = u_xx`:

```
||u_tt||     = 2942
||u_xx||     = 2942
||residual|| =    0.44
```

The two terms are each about 2942 in size, and their difference is 0.44. **They
cancel to 1 part in 6,696.**

This reframes what a physics-informed loss actually is. It is **not** "fit the
function". It is "produce two enormous quantities that agree with each other to
four decimal places". Getting `u` right to 0.04 % guarantees nothing about that,
because differentiating twice amplifies error by `k²`.

Everything below is a consequence.

### 8.2 Why `update_grid` destroys a working model

`update_grid` (§4.6) moves the spline knots to match where the activations actually
are, then re-fits the coefficients. pykan's own `fit()` calls it **10 times by
default** — it is central to how KANs are meant to gain accuracy.

One call on a **trained, working** model:

```
              rel-L2      training objective
before        0.0114%       1.930e-05
after         0.2272%       3.683e-01     <- 19,000x worse
```

What it preserves, measured on 4,000 collocation points:

| quantity | ‖before‖ | relative change |
|---|---|---|
| u | 1.508e+01 | 0.10 % |
| u_x | 1.863e+02 | 0.59 % |
| **u_xx** | 2.942e+03 | **2.28 %** |
| **u_tt** | 2.942e+03 | **2.29 %** |

The function barely moves. The second derivatives move 23× more — and each order of
differentiation roughly quadruples the error, exactly as §8.1 predicts.

`curve2coef` re-fits coefficients to match **function values** at sample points.
Nothing constrains the derivatives. Recall from §4.4 that coefficients are tied to
specific knot positions; move the knots and the old coefficients no longer mean
what they meant.

Why 2.3 % is fatal: the cancellation needs `1.5e-4` relative accuracy. A 2.3 %
perturbation is **153× coarser**. Predicting the damage:

```
predicted ||residual|| after = sqrt(2) · 0.0229 · 2942 = 95.3
predicted loss               = 95.3² / 10000 = 0.908
measured loss                = 0.368
```

Agreement within 2.5×, and on the correct side — the two perturbations are not
fully independent, so some cancellation survives.

L-BFGS then restarts from an objective 19,000× worse and finds the trivial
solution. **8/8 runs collapse, always to ~95 %, never gradually.**

**This is not a bug in pykan.** Grid adaptation is exactly right for regression,
where only function values matter. It is silently wrong for any loss built from
high-order derivatives — which means anyone building a physics-informed KAN on
pykan's default `fit()` hits this without warning.

### 8.3 Why Fourier-embedded KANs collapse

It looks like the architecture cannot express the solution. **It is not.**

Fit each model to the true solution by ordinary supervised regression — no PDE loss
anywhere — and then measure the derivatives at that fitted solution:

| model | fit rel-L2 | ‖u_tt‖ | ‖u_xx‖ | ‖u_tt − u_xx‖ | cancels to |
|---|---|---|---|---|---|
| `fourier16` | 0.0431 % | 3.384e+04 | 4.568e+03 | 3.247e+04 | **1 : 1** |
| `charcoords50` | 0.0415 % | 2.984e+03 | 2.947e+03 | 4.269e+02 | 7 : 1 |

`fourier16` **fits `u` better than the Fourier PINN does.** Its basis is excellent.
But its two second derivatives are not even the same *magnitude* — off by 7.4× —
and cancel **not at all**.

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

**The optimiser is not failing. The loss prefers the wrong answer** — because the
only way a basis with noisy second derivatives can make `u_tt − u_xx` small is to
make *both* small, and that means shrinking `u` toward zero.

And `u ≡ 0` genuinely satisfies the wave equation and the absorbing boundaries
exactly. Recall from §2.3 that `decay(t)` is 0.011 by `t = 0.2`, so across almost
the entire scoring window `u ≈ N` — driving the network output to zero drives the
solution to zero. That is a real global minimum of our loss, sitting at ~95 % error.

### 8.4 Why the tuned KAN wins

From §4.7: in characteristic coordinates the wave operator is `∂ξ ∂η`, so
`F(ξ) + G(η)` satisfies the equation **identically, for any `F` and `G` whatsoever**.

A KAN layer computes a sum of univariate functions. So the cancellation of §8.1 is
**structural — built into the coordinate system — rather than something the
optimiser must learn.** That is why `charcoords50` achieves 7:1 cancellation
straight out of a value fit, with its two second-derivative norms agreeing to 1.3 %,
while the Fourier-embedded KAN achieves 1:1.

It also explains two-layer, where the advantage disappears: a sharp interface makes
`c(x)` vary quickly, `tau'' = -c'/c²` becomes large, and the chain rule injects that
term into `u_xx`. The coordinate change stops being exact, and the structural
cancellation is lost.

### 8.5 The principle, stated once

> A physics-informed loss does not measure how well a network fits the solution.
> It measures how well the network's **high-order derivatives cancel**. Any
> architectural choice — a Fourier embedding, a grid refit, a coordinate change —
> must be judged in derivative space, not value space.

This supersedes the framing in `BASIS_VS_OPTIMISER.md`, whose headline "233×
separable-basis advantage" is a **value-space** measurement and is therefore the
wrong quantity. The right one is the cancellation ratio in §8.3.

---

## 9. Why the soft initial condition fails

### 9.1 What was tried

§2.3 makes the initial condition unbreakable but creates the trivial basin of §8.3.
A **soft** initial condition should remove that basin: drop the ansatz so the
network *is* `u` directly, and add a penalty

```
ic_loss = mean( (u(x,0) - g(x))² )  +  mean( u_t(x,0)² )
```

Now `u ≡ 0` is no longer free — it costs the full size of the initial pulse.

### 9.2 The arithmetic works

```
IC penalty for u ≡ 0, unweighted     :  mean(g²) = 0.1204
weight the optimiser assigns to it   :  55.6   (against 0.505 for the PDE term — a 110x ratio)
IC penalty for u ≡ 0, weighted       :  6.70
converged PDE loss, for comparison   :  ~1e-3
```

The basin goes from *cheaper than converging* to **three orders of magnitude more
expensive**. And it works — models trained this way escape the basin and reach real
solutions.

(The weight 55.6 is chosen automatically by gradient-norm balancing, which
independently rediscovers the ~100× IC weight that the soft-constraint literature
picks by hand. A first attempt appeared to fail at 95.52 % only because the
rebalancing fired every 500 steps and the test ran 200 — the IC term sat at weight
1 the whole time, where the basin costs only 0.12 and a collapsing model simply
pays it.)

### 9.3 But it costs a great deal, and rescues nothing

| model | hard ansatz | soft IC | effect |
|---|---|---|---|
| Fourier PINN | 0.0399 ± 0.0138 | 0.2140 ± 0.1107 | **5.5× worse** |
| tuned KAN | 0.0236 ± 0.0129 | 0.0479 ± 0.0278 | **2.0× worse** |
| PIKAN (pykan) | 94.98 | 100.62 ± 3.02 | still dead |
| PIKAN (ours) | 235.6 | 372.9 ± 321.6 | still dead |
| MLP, Adam-only | 0.7392 | 99.63 | **fails outright, 5/6 runs** |

Three distinct failures are visible:

1. **The optimisation problem gets harder.** The hard ansatz removes two constraints
   from the search entirely. Restoring them as penalties means the optimiser must
   trade PDE accuracy against IC accuracy at every step, and the balance between
   them becomes a hyperparameter with real consequences.
2. **The initial condition stops being exact.** A guaranteed-correct start is traded
   for an approximately-correct one — adding a new error source exactly where the
   solution is most structured.
3. **It does not address why the PIKANs fail.** They stay dead, which is the direct
   evidence that their collapse was never about the trivial basin being cheap. It is
   §8.3: their derivatives cannot cancel, basin or no basin.

**One informative asymmetry.** The KAN degrades 2.0× where the PINN degrades 5.5×,
so under soft constraints the KAN's lead *widens* from 1.69× to **4.47×** (3/3
paired). The headline result is not an artefact of the hard ansatz — if anything the
hard ansatz flatters the PINN.

---

## 10. Summary

- The problem is a wave on a rod; we want `u(x,t)` everywhere (§1).
- We check the physics at 10,000 fixed Sobol points, score on an 81,920-point grid
  (§2.1–2.2).
- The initial condition is enforced by construction, not by penalty — which is both
  the reason training works and the reason a trivial solution exists (§2.3, §8.3).
- The loss is built from **second** derivatives, and that is the whole difficulty
  (§2.4, §8.1).
- A KAN puts learned curves on its edges instead of scalars, and in characteristic
  coordinates its structure matches this solution's exactly (§4).
- Under pure L-BFGS the two families tie. Under Adam→L-BFGS the tuned KAN wins by
  1.81× on homogeneous and its variance problem disappears (§6, §7).
- Every catastrophic failure in the study — `update_grid`, both PIKANs, the soft-IC
  MLP — is the same fact: a basis whose second derivatives cannot cancel to 1 part
  in 6,696 has no way to satisfy this loss except by shrinking to zero (§8).
