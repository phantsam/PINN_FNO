# Presentation Script — PINNs vs KANs on the 1-D Elastic Wave Equation

**How to use this.** Normal text is what you say. `[BRACKETS]` are stage directions
and things not to say aloud. `[IF ASKED]` blocks are answers to questions you should
expect. Timing markers assume roughly 25–30 minutes; cut the `[OPTIONAL]` blocks
first if you are running long.

**Read this before you present — two things you asked about that need correcting:**

1. You said you were under the impression we ran both optimisers on every model.
   **We did not.** The Fourier PINN is the only model run under *both* optimisers on
   *all three* materials. The tuned KAN has both on homogeneous and multilayer but
   **not** on twolayer. Everything else is one optimiser only. The exact matrix is on
   slide 4 — present that honestly and it is still a strong design, because the
   headline comparison does have both arms under both optimisers at 11 seeds.
2. **The twolayer and multilayer differences are not statistically significant.**
   Say "no measurable difference", not "the PINN wins on twolayer". The one
   significant result is homogeneous.

---

## 1. Opening — the problem  [~2 min]

Good morning. What I have been working on is a controlled comparison between two
families of neural network — physics-informed neural networks, the standard
approach, and Kolmogorov-Arnold networks, a newer alternative — on a wave
propagation problem.

The physical setup is deliberately simple, because I wanted the *comparison* to be
clean rather than the problem to be impressive. It is a one-dimensional elastic rod.
You deform it slightly in the middle at time zero and let go. A wave travels outward
in both directions. I want to predict the displacement at every position and every
moment.

The governing equation is the elastic wave equation in conservative form:

> **ρ(x) · u_tt = ∂/∂x ( E(x) · u_x )**

Reading it left to right: on the left, density times acceleration — that is mass
times acceleration. On the right, the rate of change of the internal stress along
the rod. So this is Newton's second law, written for every point of a continuous
body simultaneously.

One detail I want to flag early because it matters later. Most people write the wave
equation as `u_tt = c²·u_xx`. That is only correct for a *uniform* rod. If you expand
my right-hand side with the product rule you get `E·u_xx + E'(x)·u_x`, and the second
term only vanishes when the stiffness is constant. That term is exactly what makes
waves reflect and transmit correctly at a material boundary. I keep it. The previous
codebase for this project did not, and I will come back to what that cost.

`[IF ASKED why conservative form matters numerically]` It enforces continuity of
stress across an interface. Drop it and your simulation over-transmits energy through
joins — we measured 34 % over-transmission in one of the inherited solvers.

---

## 2. The constraints  [~2 min]

`[SLIDE: the four conditions]`

A differential equation on its own has infinitely many solutions, so four conditions
pin mine down.

**One — the equation itself**, which has to hold everywhere inside the domain. Space
runs from −1 to +1, time from 0 to 1, in non-dimensional units where the reference
wave speed is exactly 1.

**Two — the initial shape.** At time zero the rod has a specific profile, `g(x)`. It
is the derivative of a Gaussian, with width parameter 0.1, normalised so its peak is
exactly 1. That gives a small double-lobed wiggle — one bump up at x = −0.1, one bump
down at x = +0.1, and essentially nothing beyond about ±0.36.

I use the *derivative* of a Gaussian rather than a plain Gaussian for a physical
reason: the two lobes cancel, so there is zero net displacement. A plain bump would
translate the rod's material in one direction overall, which is not what a pluck does.

**Three — released from rest.** The velocity at time zero is zero everywhere. It is a
pluck-and-release, not a strike. I need both a starting position and a starting
velocity because the equation is second order in time — same reason throwing a ball
needs both.

**Four — the boundaries.** The rod is finite but I want to model an infinite one, so
I use absorbing boundary conditions at both ends. Waves leave without reflecting.

`[IF ASKED why width 0.1]` It is a compromise. Much wider and the pulse fills the
whole rod and never separates into two clean travelling waves, so the problem stops
testing propagation. Much narrower and it contains very high frequencies, so every
model fails on resolution rather than on architecture. At 0.1 the pulse occupies about
a fifth of the rod and separates cleanly well before the run ends. It is frozen
across every experiment, so it cannot favour anybody.

---

## 3. The three materials  [~1 min]

`[SLIDE: material table]`

I run the same equation on three rods. Only the stiffness profile changes.

**Homogeneous** — uniform, stiffness 1 everywhere. The easy case, and the only one
with a known exact solution, which I will use heavily.

**Twolayer** — one interface, stiffness rising from 1.0 to 1.5, over a transition
width of 0.02.

**Multilayer** — six interfaces, stiffness rising from 1.0 to 2.5, over a transition
width of 0.05.

Here is the fact that surprised me and that I want you to hold onto: **twolayer, with
one interface, is about twenty times harder than multilayer, with six.** Difficulty is
not driven by how many interfaces there are, but by how *sharp* they are. Twolayer's
join is two and a half times narrower, so the same jump is squeezed into a shorter
distance and its stiffness gradient is four point two times larger.

I predicted the opposite early on and was wrong about it.

---

## 4. What I ran — the full roster  [~3 min]

`[SLIDE: best models per material]`

Before any results, I want to be upfront about the whole space of what I tried,
including everything that failed.

**These are the models that gave the most promising results, per material:**

| material | best model | error | second | third |
|---|---|---|---|---|
| **homogeneous** | **tuned KAN — `charcoords50`** | **0.0222 %** | matched Fourier PINN 0.0332 % | Fourier PINN 0.0401 % |
| **twolayer** | **plain MLP PINN** | **0.4318 %** | tuned KAN 0.5984 % | Fourier PINN 0.9579 % |
| **multilayer** | **tuned KAN — `charcoords50`** | **0.0403 %** | Fourier PINN 0.0432 % | plain KAN 0.2320 % |

Those are relative L2 errors as a percentage — so 0.02 % means the error is about one
five-thousandth the size of the solution itself.

`[SLIDE: models that collapsed]`

**And these are the ones that failed, which I think are as informative as the
successes:**

| model | what happened | error |
|---|---|---|
| PIKAN — my hand-rolled version | diverged; broken initialisation | 235 % – 267 % |
| PIKAN — reference pykan backend | collapsed to the trivial solution | ~95 – 96 % |
| KAN with any Fourier embedding | collapsed, 6 out of 6 attempts | 94.7 – 96.7 % |
| KAN with grid = 50 but no coordinate change | collapsed | 66.7 % |
| KAN with ReLU basis | collapsed | 93.6 % |
| KAN with order k = 5 on twolayer | collapsed | 95.7 % |
| WavKAN (wavelet KAN) | diverged under L-BFGS | NaN / 63 – 66 % |
| KAN with pykan's own grid refinement | collapsed, 8 out of 8 | 94.9 – 95.4 % |

A number around 95 % means the model is producing essentially nothing. Above 100 %
means it is *worse than predicting zero everywhere*.

**So that is everything I ran.** Nine architectures, three materials, two optimisers,
roughly eight hundred training runs across ten phases.

`[SLIDE: the honest coverage matrix — IMPORTANT, do not skip]`

One thing I want to be precise about, because it affects how much weight my optimiser
claim can carry. I did **not** run both optimisers on every model:

| model | homogeneous | twolayer | multilayer |
|---|---|---|---|
| **Fourier PINN** | **both, 11 seeds each** | **both, 11 each** | **both, 11 each** |
| **tuned KAN** | **both, 11 each** | hybrid only | **both, 11 each** |
| MLP PINN | hybrid only | hybrid only | hybrid only |
| plain KAN | hybrid only | hybrid only | hybrid only |
| PirateNet | L-BFGS only | — | L-BFGS only |
| both PIKANs | hybrid only | hybrid only | hybrid only |

The Fourier PINN is the only model with both optimisers on all three materials. But
critically, **my headline comparison — the Fourier PINN against the tuned KAN on
homogeneous — has both arms under both optimisers at eleven seeds each.** That is the
like-for-like cell, and it is the one carrying the claim.

---

## 5. Two sets of experiments  [~1 min]

`[SLIDE: the two experiment families]`

I ran two distinct families of experiment, and I will take them in order.

**The first uses a hard-coded initial condition** — an ansatz, where the initial shape
and the initial stillness are built into the functional form so they are mathematically
impossible to violate.

**The second uses a soft initial condition** — the standard approach in the literature,
where those conditions become penalty terms in the loss and the optimiser has to
balance them.

The first family is where almost all my results come from. The second I ran to test
whether the hard constraint was doing something unfair, and the answer is interesting.

I will do the hard-constrained experiments first.

---

## 6. The hard-coded initial condition  [~3 min]

`[SLIDE: the ansatz]`

Here is the construction. The network does not output the solution directly. Call the
network's raw output `N(x,t)`. I build the solution as:

> **u(x,t) = g(x)·decay(t) + growth(t)·N(x,t)**

with `decay(t) = exp(−½(15t)²)` which starts at 1, and `growth(t) = tanh²(25t)` which
starts at 0.

At time zero, decay is 1 and growth is 0, so `u = g(x)` exactly — the network's output
is multiplied by zero. **However badly trained the network is, the initial shape is
exactly right.** And both helper functions have zero slope at the origin, so the
initial velocity is exactly zero too.

That removes two of my four constraints from the optimisation problem entirely, for
free and permanently.

There is one consequence I want to plant now because it comes back. Decay has fallen
to 0.011 by time 0.2, and growth has saturated at 1 by time 0.1. My scoring window
starts at 0.05. So across almost everything I measure, **the solution is essentially
just the raw network output.** That will matter in the soft-IC section.

### The optimiser question — my different approach here

`[SLIDE: two optimisers]`

For this family I took a different approach from most of the literature. Rather than
picking a training recipe and holding it fixed, **I ran two optimisers and treated the
choice as an experimental variable.**

**Pure L-BFGS.** A quasi-Newton method — it watches how the gradient changes over
about a hundred steps to build a model of the loss surface's curvature, then jumps
toward the bottom of that local bowl. Very well-informed steps.

**Adam then L-BFGS — a hybrid.** Seven hundred steps of Adam first, then hand over to
the same L-BFGS. Adam moves each parameter by roughly a fixed amount regardless of how
large its gradient is, so it explores; L-BFGS then refines.

I did this because I suspected that a comparison run under one optimiser might be
reporting an optimiser effect as an architecture effect. **That suspicion turned out
to be correct, and it is the main finding of the whole project.**

`[IF ASKED why the collocation points are frozen]` L-BFGS's curvature model assumes it
is looking at the same loss surface each step. If you resample the points every step
the surface moves underneath it and the estimate becomes noise. When I got that wrong
early on, every PINN looked ten to seventy times worse than it actually is.

---

## 7. Homogeneous — the results  [~4 min]

`[SLIDE: homogeneous results table]`

Starting with the uniform rod.

### Why these particular models

Let me explain the roster before the numbers, because each model is there to answer a
specific question.

**The plain MLP PINN** is the control. Raw position and time straight into a five-layer
tanh network, no cleverness. If anything fancier fails to beat this, the cleverness is
not earning its place.

**The Fourier PINN** adds a random Fourier feature embedding in front — the input is
projected onto sixty-four random plane waves and passed through as sines and cosines.
The reason is well established in this literature: **a plain tanh network is biased
toward smooth, slowly varying functions and needs enormous depth to produce
oscillations. Fourier features hand it oscillation for free**, so it can represent
high-frequency content that the plain MLP struggles with.

One thing I want to point out because I think it is a nice detail. The bandwidth of
that embedding is not a tuned hyperparameter. My initial pulse is a derivative of a
Gaussian with width 0.1, and the Fourier transform of that has a clean peak at exactly
one over the width — so ten. I measured it numerically at 9.425. **I set the
embedding's bandwidth to ten because that is the pulse's own dominant frequency.** The
initial condition and the network's bandwidth are the same fact stated twice.

**The plain KAN** is the KAN control — the reference implementation, raw coordinates,
eighty-six hundred parameters.

**The tuned KAN, which I call charcoords50**, is the one that wins, and I will come
back to why in a moment.

### The numbers

| model | pure L-BFGS | Adam → L-BFGS |
|---|---|---|
| **tuned KAN** | 0.0313 % ± 0.0170 (n=11) | **0.0236 % ± 0.0129 (n=11)** |
| **Fourier PINN** | **0.0281 % ± 0.0073 (n=11)** | 0.0399 % ± 0.0138 (n=11) |
| matched Fourier PINN | — | 0.0332 % ± 0.0152 (n=11) |
| plain KAN | — | 0.0849 % ± 0.0201 (n=5) |
| MLP PINN | — | 0.2555 % ± 0.1232 (n=5) |
| PirateNet | 0.0296 % ± 0.0024 (n=3) | — |
| both PIKANs | — | ~95 % and 235 % — collapsed |

**Look at the top two rows and read across.**

Under **pure L-BFGS**: Fourier PINN 0.0281, tuned KAN 0.0313. The PINN is nominally
ahead — but the p-value is 0.577. **That is a tie.** There is no difference.

Under **Adam then L-BFGS**: Fourier PINN 0.0399, tuned KAN 0.0236. **The KAN is now
ahead by a factor of 1.81, and this one is significant** — Welch p = 0.0154,
Mann-Whitney p = 0.0126, and on a seed-by-seed paired comparison the KAN wins nine
times out of eleven with Wilcoxon p = 0.032. Three different tests agree.

Same networks. Same collocation points — byte-identical for a given seed, I verified
that rather than assuming it. Same loss function. **The only thing that changed is
whether L-BFGS started cold or after seven hundred Adam steps, and the answer flipped.**

### The variance result, which I think is the strongest part

`[SLIDE: variance]`

There is a second thing in that table that I think matters more than the means.

Under pure L-BFGS the KAN's seed-to-seed variance was **5.44 times** the PINN's, with
Levene p = 0.031. That was the *only* statistically significant difference between the
two families, and it was the strongest argument against KANs — they were inconsistent.

Under the hybrid, that ratio is **0.87**, p = 0.73. It is gone. The KAN is now the
steadier of the two.

**So the Adam warm-up did not just improve an average — it removed a failure mode.**
Seeds that used to land in bad basins no longer do. That is why I think this is an
optimisation-landscape effect rather than a scaling effect.

`[IF ASKED why Adam helps]` L-BFGS is a very good *local* optimiser and that is exactly
its problem — from a cold random start it descends into whatever basin it happens to
land in and cannot climb out. Adam's step size does not shrink when gradients are
large, so it explores. I measured this directly: on one architecture, L-BFGS reached a
6.5 times lower loss while moving the parameters 18 times less. It is about a hundred
times more efficient per unit of movement — which is precisely why it cannot escape.

---

## 8. Verifying homogeneous against d'Alembert  [~4 min]

`[SLIDE: d'Alembert]`

Before I trust any of those numbers, I want to show you why you can trust the
measuring instrument. This is the part I think is the strongest evidence that the
pipeline is correct.

### What d'Alembert gives us

For the uniform rod, the wave equation has an exact closed-form solution, found by
d'Alembert in 1747:

> **u(x,t) = ½ [ g(x − t) + g(x + t) ]**

In words: the initial bump splits into two half-height copies of itself, one
travelling right, one travelling left, each at speed 1, neither changing shape.

Why `x − t` means travelling right: the value at position x and time t is whatever `g`
was at `x − t`. As time increases you have to increase x by the same amount to keep
seeing the same value, so the pattern moves right at unit speed.

So for the homogeneous case **I am not dependent on a numerical reference at all. I
have the exact answer in closed form.**

### How I used it — three separate checks

`[SLIDE: the verification chain]`

**Check one: does my finite-difference solver reproduce the exact solution, and at the
right convergence rate?**

This is the important one. A second-order solver should quarter its error every time
you double the grid. Here is what mine does, measured against d'Alembert:

| grid nx | error vs exact | ratio to previous | implied order |
|---|---|---|---|
| 256 | 0.4384 % | — | — |
| 512 | 0.1088 % | 4.03× | 2.01 |
| 1024 | 0.0271 % | 4.01× | 2.00 |
| 2048 | 0.0068 % | 4.01× | 2.00 |
| 4096 | 0.0017 % | 4.00× | 2.00 |

**Four point zero, four times running.** That is textbook second-order convergence.
This tells me the solver is correct, the time stepping is correct, the boundary
handling is not polluting the interior, and my error metric is measuring what I think
it is measuring.

`[POINT AT the ratio column]` If any of those were wrong, this column would not read
4.00.

For context — when I first built this, that column read 1.09, 1.04, 0.74. It looked
first-order. The cause turned out to be that I was picking snapshots by nearest
neighbour in time, and since the time step differs between grids, "the same" snapshot
sat at slightly different instants on each grid. Interpolating to the exact instants
recovered the true second order. **The solver had been right all along; the comparison
was wrong.**

**Check two: is my reference fine enough to measure the models?**

This is a subtler point that caught me out twice. The reference has its own error, and
if that error is comparable to the model errors, differences between models get
compressed and become unmeasurable.

Look at that table again against my best model, which is at 0.0222 %:

- At nx = 512 the reference's own error is 0.1122 % — *five times larger than the thing
  I am trying to measure.* Useless. And I used it for phases three through seven.
- At nx = 2048 the reference is at 0.0068 %, only 3.6 times below my best model. That
  is marginal. And here is the part I find instructive: **when I chose nx = 2048 it had
  a twenty-times margin. The margin decayed to 3.6× because the models got better.
  Nothing broke. No test caught it.**

**Check three — and this is the resolution: for homogeneous I score against the exact
solution directly.** No discretisation error at all. Infinite margin.

### What that did to the result

`[SLIDE: the three scorings]`

| reference | KAN ahead by | Welch p |
|---|---|---|
| FD at nx = 2048 | 1.69× | 0.0098 |
| FD at nx = 4096 | 1.80× | 0.0150 |
| **exact d'Alembert** | **1.81×** | **0.0154** |

The result **strengthens** as the reference gets better. That is exactly the direction
the compression argument predicts — the coarse reference had been *understating* the
KAN's advantage, not manufacturing it.

**So my confidence in the homogeneous result rests on three legs:** the solver
converges at exactly second order against a known analytic solution; the headline is
scored against that analytic solution with zero discretisation error; and the result
holds under three different references and three different statistical tests.

`[IF ASKED about the other two materials]` No closed form exists for a heterogeneous
rod, so there I use the finite-difference solution at nx = 2048 and I am honest that
the margin is thinner. That is one reason I lean on homogeneous for the headline claim.

---

## 9. Twolayer — one sharp interface  [~3 min]

`[SLIDE: twolayer results]`

Now the hard one. Twolayer has a single interface where stiffness jumps from 1.0 to
1.5 over a transition width of 0.02.

| model | pure L-BFGS | Adam → L-BFGS |
|---|---|---|
| **MLP PINN** | — | **0.4318 % ± 0.3070 (n=11)** |
| tuned KAN | — | 0.5984 % ± 0.3510 (n=11) |
| Fourier PINN | 0.8905 % ± 0.9874 (n=11) | 0.9579 % ± 0.7969 (n=11) |
| Fourier PINN + resampling | 0.4928 % ± 0.3512 (n=11) | — |
| tanh KAN | 0.8701 % ± 0.5035 (n=11) | — |
| plain KAN | — | 1.0340 % ± 0.4055 (n=5) |
| PirateNet | 2.768 % (n=3) | — |

**The headline for this material: the errors are twenty times larger than homogeneous
for every architecture.** That is the sharpness effect I mentioned — one interface,
but a very sharp one.

**And the plain MLP wins.** The simplest model in the study — no embedding, raw
coordinates, five tanh layers — beats the Fourier PINN by more than a factor of two
and beats the tuned KAN.

I want to be careful here: **that difference is not statistically significant.** MLP
0.4318 against tuned KAN 0.5984 gives p = 0.250. So the honest statement is **"no
measurable difference between the best PINN and the best KAN on twolayer."**

At five seeds this cell read a 2.20× PINN win with p = 0.056, and I was ready to report
it. At eleven seeds it fell to 1.39× and p = 0.250. **That is the fifth finding in this
project to dissolve when I raised the seed count**, and it is why every headline number
here is at eleven.

### Why the Fourier PINN loses here — a talking point worth making

This is the interesting physics. Fourier features help when the solution's content is
concentrated at a known frequency — which is exactly the homogeneous case, where my
embedding is tuned to the pulse's peak at k = 10.

**But an interface is not a single frequency. A sharp discontinuity needs *all* of
them.** So the embedding that was perfectly matched to the pulse is mismatched to the
interface, and the plain MLP — which has no frequency prior at all — does better.

PirateNet shows the same inversion even more dramatically: 0.0296 % on homogeneous, one
of the best numbers in the study, and 2.768 % on twolayer. **A twenty-five-fold
inversion, same architecture, different material.**

### Why the tuned KAN loses its advantage here

I will explain the KAN's mechanism properly in a moment, but the short version: it wins
on homogeneous because a coordinate change makes the problem exactly separable. **On a
sharp interface that coordinate change stops being exact** — the transform injects a
term proportional to the second derivative of the travel time, and that term is 3.7
times larger on twolayer. So the thing that makes it win elsewhere actively hurts here.

---

## 10. Multilayer — six gradual interfaces  [~2 min]

`[SLIDE: multilayer results]`

| model | pure L-BFGS | Adam → L-BFGS |
|---|---|---|
| **tuned KAN** | 0.0574 % ± 0.0375 (n=11) | **0.0403 % ± 0.0114 (n=11)** |
| **Fourier PINN** | 0.0419 % ± 0.0146 (n=11) | 0.0432 % ± 0.0115 (n=11) |
| plain KAN | — | 0.2320 % ± 0.0984 (n=5) |
| MLP PINN | — | 0.6915 % ± 0.2701 (n=11) |
| PirateNet | 0.0563 % ± 0.0223 (n=3) | — |
| both PIKANs | — | ~96 % — collapsed |

Six interfaces, stiffness going up to 2.5 — and **every model does better here than on
the single sharp interface.** That is the clearest confirmation I have that sharpness
is what costs, not the number of interfaces.

The tuned KAN and the Fourier PINN are essentially tied: 0.0403 against 0.0432,
p = 0.571. **No measurable difference.**

Notice also that the ordering is restored relative to twolayer: the plain MLP is now
the *worst* of the sensible models at 0.69 %, and the Fourier embedding is helping
again. That is consistent with the story — these interfaces are gradual enough that the
solution remains dominated by the pulse's own frequency content, which is what the
embedding is tuned for.

`[OPTIONAL — cut if short]` The tuned KAN also improves under the hybrid here, from
0.0574 to 0.0403, and its variance tightens from 0.0375 to 0.0114. Same pattern as
homogeneous, just not enough to separate it from the PINN.

---

## 11. Why the tuned KAN works — the design  [~3 min]

`[SLIDE: KAN structure]`

Let me explain what a KAN is and why I built the particular one that wins, because I
think the reason is the most satisfying part of this project.

### What a KAN is

A standard MLP puts **fixed** nonlinearities on the nodes and **learned scalars** on
the edges. Each edge can only scale a signal up or down; all the shape comes from the
fixed activation function.

A KAN inverts that. It puts **learned univariate functions on the edges** and **plain
summation on the nodes**. There is no weight matrix and no activation function in the
MLP sense — **each edge carries its own little curve**, built from B-splines, and
training reshapes those curves.

So in an MLP an edge can make a signal bigger or smaller. In a KAN it can *bend* it.

### Why that should matter for this problem specifically

Here is the connection. The exact solution on homogeneous is

> u(x,t) = ½[g(x−t) + g(x+t)] = **F(ξ) + G(η)**

where ξ = x−t and η = x+t. **That is literally a sum of two univariate functions.**

And a KAN layer computes exactly a sum of univariate functions. So this solution is
*already* in the form a KAN naturally produces — it is a Kolmogorov-Arnold
representation with one layer and two edges.

**So I feed the KAN the characteristic coordinates** ξ and η instead of position and
time. In those coordinates the wave operator becomes a single mixed derivative,
∂ξ∂η — which means **any** function of the form F(ξ) + G(η) satisfies the wave equation
*identically*, for any F and any G, without training anything at all.

That is the one design choice in this project with a theory-driven reason to favour a
KAN *specifically* rather than every architecture equally.

### Why grid = 50

`[SLIDE: the three changes]`

The tuned KAN differs from the plain one in three ways, and I want to justify each
rather than present them as tuning.

**One — characteristic coordinates**, for the reason just given.

**Two — grid 50 instead of 5.** The grid is how many intervals each edge's spline is
cut into — it is the resolution knob. At grid 50 the knot spacing is 0.04. **My pulse
has a lobe width of 0.16, so that gives me about four knots across the feature I need
to resolve.** At grid 5 the spacing is 0.4, which is more than twice the width of the
entire feature — the spline simply cannot see it.

**Three — spline order 5 instead of 3.** This one is specifically about the PDE. A
spline of order k is smooth to k−1 derivatives. At order 3 the network is twice
differentiable, which means the second derivative — the thing my loss actually consumes
— is only piecewise linear, **with a kink at every one of the fifty knots**. At order 5
the second derivative is genuinely smooth. For a second-order equation that is a
principled change, not a knob.

And critically: **none of these works alone.** Grid 50 without the coordinate change
collapses at 66.7 %. The coordinate change without the resolution gives 0.1498 %. You
need both, and that is exactly what you would predict if the mechanism is what I claim.

---

## 12. The soft initial condition — the second experiment family  [~4 min]

`[SLIDE: soft IC]`

Now the second family of experiments. Everything so far used the hard-coded initial
condition. This is what happens when you use the standard approach instead.

### Why I ran it

Two reasons. First, **soft constraints are what most of the literature does** — the
initial condition becomes a penalty term in the loss and the optimiser balances it
against the PDE term. If my hard constraint were doing something unfair, this would
expose it.

Second — and this is the real motivation — **the hard ansatz creates a specific
danger.** Remember that decay function falls to 0.011 by time 0.2. So across almost my
entire scoring window, the solution is essentially just the raw network output.

Which means: **if the network outputs zero, the solution is zero.** And `u ≡ 0`
satisfies the wave equation exactly and satisfies the absorbing boundary conditions
exactly. **It is a genuine, perfectly valid global minimum of my loss function, sitting
at about 95 % error.**

That is the trivial solution, and it is where every collapsed model in my study ended
up. A soft initial condition should remove it, because now `u ≡ 0` costs the full size
of the initial pulse.

### The arithmetic — it should work

Unweighted, the trivial solution costs 0.1204 in the IC penalty. My gradient-norm
balancing assigns that term a weight of 55.6 against 0.505 for the PDE term — a factor
of 110, which independently rediscovers the roughly hundred-fold IC weight the
soft-constraint literature picks by hand.

So weighted, the trivial solution costs **6.70**, against a converged PDE loss of about
0.001. **The basin goes from cheaper than converging to three orders of magnitude more
expensive.**

And it works, in the narrow sense: models trained this way do escape the basin and
reach real solutions.

### But it fails badly

`[SLIDE: soft IC results]`

| model | hard ansatz | soft IC | effect |
|---|---|---|---|
| Fourier PINN | 0.0399 % | 0.2140 % | **5.5× worse** |
| tuned KAN | 0.0236 % | 0.0479 % | **2.0× worse** |
| PIKAN (pykan) | 94.98 % | 100.62 % | still dead |
| PIKAN (mine) | 235.6 % | 372.9 % | still dead |
| **MLP PINN, Adam-only** | 0.7392 % | **99.63 %** | **fails outright, 5 of 6 runs** |

Three separate things go wrong.

**One — the optimisation problem gets genuinely harder.** The hard ansatz removed two
constraints from the search entirely. Putting them back as penalties means the
optimiser now has to trade PDE accuracy against initial-condition accuracy at every
single step, and the balance between them becomes a hyperparameter with real
consequences. The Fourier PINN loses a factor of 5.5 to that.

**Two — the initial condition stops being exact.** I traded a guaranteed-correct start
for an approximately-correct one, and I introduced that error precisely where the
solution is most structured.

**Three — and this is the important one — it does not rescue the collapsed models.**
Both PIKANs stay dead. The MLP under Adam alone doesn't just get worse, it **fails
completely, in five runs out of six, landing at 99.63 %**.

**That last row is the key evidence.** If the PIKANs were collapsing because the trivial
basin was cheap, removing the basin should have saved them. It didn't. **So their
collapse was never about the basin.** Something else is going on, and that is the next
section.

### One asymmetry worth pointing out

`[SLIDE: the asymmetry]`

Notice that the KAN degrades by 2.0× where the PINN degrades by 5.5×. So **under soft
constraints the KAN's lead actually widens** — from 1.69× to 4.47×, winning three
comparisons out of three.

I think that is worth stating explicitly, because the obvious objection to my headline
result is "your hard ansatz is doing the work". It isn't. **If anything the hard ansatz
flatters the PINN**, and removing it makes the KAN look better, not worse.

---

## 13. Why models collapse — the mechanism  [~4 min]

`[SLIDE: the cancellation]`

So if the trivial basin isn't the explanation, what is? This is the part I am most
pleased with, because one measurement explains every failure in the study.

### The measurement

Take my best trained model on homogeneous, where the equation is exactly `u_tt = u_xx`,
and measure the two sides separately:

> ‖u_tt‖ = **2942**
> ‖u_xx‖ = **2942**
> ‖residual‖ = **0.44**

**Those two enormous quantities cancel to one part in 6,696.**

That completely reframes what a physics-informed loss is. **It is not "fit the
function". It is "produce two huge numbers that agree with each other to four decimal
places."** Getting the solution right to 0.04 % guarantees nothing about that, because
differentiating twice amplifies error by the square of the frequency.

Here is the intuition in one line. Take the true function f = 0, and an approximation
g = 0.001·sin(1000x). The *values* differ by at most 0.001 — an excellent
approximation. The *second derivatives* differ by 1000. **A factor of a million.**

### That single fact explains all three failure modes

**One — why the Fourier-embedded KANs collapse.**

I tested this directly. I fitted them to the true solution by ordinary supervised
regression — no PDE loss at all — and then looked at their derivatives:

| model | fits u to | ‖u_tt‖ | ‖u_xx‖ | cancels to |
|---|---|---|---|---|
| PIKAN (`fourier16`) | 0.0431 % | 3.38e4 | 4.57e3 | **1 : 1** |
| tuned KAN | 0.0415 % | 2.98e3 | 2.95e3 | 7 : 1 |

**The PIKAN fits the solution *better* than the Fourier PINN does.** Its basis is
excellent. But its two second derivatives are not even the same *order of magnitude* —
off by a factor of 7.4 — and they cancel not at all.

Then I evaluated my actual training loss at those correct solutions:

> PIKAN: loss at the **correct** answer = 2.59e5, loss where it actually lands = 1.54e-5

**The trivial answer scores seventeen billion times better under my loss than the right
answer does.** I even restarted training *from* the correct solution — it abandoned it
in a single epoch.

**So the optimiser is not failing. The loss genuinely prefers the wrong answer.**
Because if your second derivatives can't cancel, the only way left to make the residual
small is to make both terms small — and that means shrinking the solution to zero.

**Two — why pykan's own grid refinement destroys my runs.**

pykan's standard training routine re-fits the spline knots ten times by default. It
collapsed my runs eight times out of eight. One call on a trained, working model:

| | rel-L2 | training loss |
|---|---|---|
| before | 0.0114 % | 1.93e-5 |
| after | 0.2272 % | 3.68e-1 — **19,000× worse** |

The function barely moved — 0.10 %. But the second derivatives moved 2.28 %. **The
refit matches function *values*; nothing constrains the derivatives.** And my
cancellation needs 1.5e-4 relative accuracy, so a 2.3 % perturbation is 153 times too
coarse. I predicted a 47,000× blow-up from that arithmetic and measured 19,000×.

This is not a bug in pykan — grid adaptation is exactly right for regression, where
only values matter. **It is silently wrong for any loss built from high-order
derivatives**, which means anyone building a physics-informed KAN on the reference
implementation's defaults will hit this without warning. I think that is a genuinely
useful thing to report.

**Three — why the tuned KAN wins.**

In characteristic coordinates the wave operator is ∂ξ∂η, so F(ξ) + G(η) satisfies it
*identically*. **The cancellation is structural — built into the coordinate system —
rather than something the optimiser has to learn.** That is why it achieves 7:1
straight out of a value fit while the Fourier-embedded KAN achieves 1:1.

And it explains twolayer: a sharp interface makes the coordinate transform inexact,
and the structural cancellation is lost.

### The principle

`[SLIDE: one sentence]`

> **A physics-informed loss does not measure how well your network fits the solution.
> It measures how well your high-order derivatives cancel. Any architectural choice —
> an embedding, a grid refit, a coordinate change — has to be judged in derivative
> space, not value space.**

---

## 14. Bringing it together  [~2 min]

`[SLIDE: summary]`

So, to summarise what I actually found.

**One. The optimiser decided the result, not the architecture.** Under pure L-BFGS my
two families tie on homogeneous, p = 0.577, and the KAN looks unstable with 5.44 times
the variance. Add seven hundred Adam steps — changing nothing else, same networks, same
points, same loss — and the KAN wins by 1.81 times, p = 0.0154, with *lower* variance.
I think that is the most important methodological point here: **a paper that benchmarks
one optimiser and reports the outcome as an architecture property is reporting an
interaction without knowing it.**

**Two. On accuracy, the honest picture is narrower than the headline.** The KAN wins
significantly on homogeneous. On twolayer and multilayer there is no measurable
difference. So the defensible claim is: **the KAN is significantly better on one
material of three, and never significantly worse anywhere.** I should also say that
under matched parameter counts the homogeneous lead narrows from 1.81× to 1.50×, and
there the tests disagree — the paired test is significant at p = 0.0068, an unpaired
test is not at p = 0.092. I would rather show you that than pick the favourable one.

**Three. The KAN's advantage is structural, and I can say exactly what it is.** In
characteristic coordinates the solution is a sum of univariate functions, which is
precisely what a KAN layer computes, so the cancellation the loss demands comes for
free rather than being learned.

**Four. Difficulty tracks interface sharpness, not interface count.** One sharp
interface is twenty times harder than six gradual ones.

**Five. The most spectacular failures are not what they look like.** Every collapsed
model can represent the solution to better than 0.05 %. They fail because the loss
function genuinely prefers the trivial answer when the second derivatives cannot cancel.

---

## 15. What I would say about limitations  [~1 min]

`[SLIDE: open questions]`

I want to be straight about what this does not establish.

- **Only one material of three separates the two families**, so a single significant
  result is carrying the accuracy claim. A fourth smooth material would test whether
  "the KAN wins where the solution is separable" is a principle or one data point.
- **I do not have a mechanism for why the residual landscape traps Fourier-embedded
  KANs specifically.** I have shown it is not a representation limit. What it *is*
  remains open, and I would rather say that than invent an explanation.
- **The decisive test of the sharpness hypothesis has not been run** — swapping the
  transition widths, so twolayer becomes gradual and multilayer becomes sharp. If
  sharpness is the cause, the failure should follow the width, not the material.
- **Five findings in this project dissolved when I raised the seed count**, including
  one I was ready to report. Everything here is at eleven seeds for that reason.

---

## Anticipated questions

**"Isn't it unfair that the KAN has fewer parameters?"**
It is unfair in the *KAN's disfavour*, which is why I flag it. The KAN has 49,020
parameters against the Fourier PINN's 82,689 — it wins with 40 % fewer. I also built a
parameter-matched PINN at 49,729 and re-ran it; the lead narrows to 1.50× and the tests
split. Worth adding: the *smaller* PINN is better than the larger one, so the default
was over-parameterised.

**"Why not just use a classical solver?"**
For this problem you absolutely should — I use one to generate my references, and it is
faster and more accurate than any network here. The interest is in whether the
grid-free approach generalises to problems where grids are punishing. This is a
controlled test on a problem where I can check every answer.

**"How do you know your PINN implementation is not just bad?"**
Three ways. My solver converges at exactly second order against a known analytic
solution. My test suite has about 190 checks and catches all 16 deliberate mutations,
including deleting the interface term from the equation. And the Fourier PINN reaches
0.028 %, which is competitive with published numbers on comparable problems.

**"Are these differences even practically meaningful?"**
At 0.02 % versus 0.04 %, arguably not for engineering purposes. What I think *is*
meaningful is the mechanism: the optimiser interaction is a factor of two and flips the
ranking, and the collapse mechanism explains failures that would otherwise look random.

**"Did the previous work on this problem not answer this?"**
No, and I want to be careful how I say this. The inherited codebase had eighteen
defects I reproduced by execution — the KAN was solving `u_tt − u_xx`, which ignores the
material entirely and is wrong on two of the three rods; its loss balancing was
inverted so the boundary condition contributed about 1e-10; it trained for ten Adam
steps against the PINN's three hundred thousand; and the head-to-head driver raised a
TypeError and had never executed. It is not that the previous answer was wrong — **no
comparison had been run.** That is why I rebuilt it.

**"What would you do next?"**
The width-swap experiment, a fourth material, and running the tuned KAN with adaptive
resampling, which I have not done in any form.

---

## Timing

| section | minutes | cut first if short |
|---|---|---|
| 1–3 problem, constraints, materials | 5 | — |
| 4–5 the roster, two experiment families | 4 | trim the collapse table |
| 6 hard IC + two optimisers | 3 | — |
| 7 homogeneous results | 4 | — |
| 8 d'Alembert verification | 4 | drop the "1.09/1.04/0.74" anecdote |
| 9–10 twolayer, multilayer | 5 | cut the multilayer OPTIONAL block |
| 11 KAN design, grid 50 | 3 | — |
| 12 soft IC | 4 | compress the arithmetic |
| 13 mechanism | 4 | — |
| 14–15 summary, limitations | 3 | — |
| **total** | **~39** | **~28 with all cuts** |
