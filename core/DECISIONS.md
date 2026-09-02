# Frozen-spec decisions, with evidence

Every choice below is fixed for **all** architectures. The model is interchangeable;
the problem is not. Each decision records why, so it can be challenged on evidence
rather than re-litigated on taste.

---

## D1. Units: non-dimensionalised (E=1, rho=1 -> c=1), NOT physical

**Decision.** Keep main's convention. `core/problem.py` defines E and rho
non-dimensionally. rak's physical units (E=80, rho=100 -> c=0.894) are not used.

**Evidence.** Non-dimensionalisation is standard preprocessing for PDE solvers and
is reported as *especially* important for gradient-based PINN training, because it
puts field components of differing magnitude on a comparable footing and stabilises
the optimisation. For 1D wave propagation specifically, a dimensionless PINN
formulation is reported at **<9% normalised error where non-dimensionless variants
gave >=40%** (Sci. Direct S0266352X25006019). The scaling is linear, so it changes
the weighting of loss terms without changing the dynamics.

**Consequence.** rak's numbers are on a different problem (c=0.894 vs c=1.0) and are
not directly comparable to anything produced under this spec.

---

## D2. Material contrast: TwoLayer E 1.0 -> 1.5 (impedance ratio 1.2247)

**Decision.** Keep main's contrast. rak's TwoLayer (E 60 -> 120, ratio 2.0) is a
*different material*, not a different scaling of the same one.

**Evidence.** This is a free modelling choice, not a physics constraint, so it is
recorded rather than derived. Main's ratio is the milder of the two and is the one
its published exp2 figures correspond to. Documented here so the two branches'
"TwoLayer" results are never again compared as though they were the same problem.

---

## D3. MultiLayer interface transition width: w = 0.05 (was 0.02)

**Decision.** Widen from main's 0.02 to 0.05.

**Evidence.** The layered-medium PINN reference (arXiv:2305.05150) smooths its
Young's-modulus steps with a transition length of **0.1** over layers of width ~0.5,
i.e. **transition/layer ~= 0.2**. Our MultiLayer has 6 layers over a domain of
length 2, so layer width = 0.333 and the equivalent transition is **0.067**.
rak independently chose 0.05. Main's 0.02 is transition/layer = 0.06, roughly
**3x sharper than the literature**, and that paper explicitly notes the derivatives
of E remain discontinuous and that this "makes it challenging ... as the PINN is a
continuous approximator."

0.05 sits between rak's choice and the literature-derived 0.067, and is still
comfortably resolved (w/dx = 51 at nx=2048).

**Consequence.** MultiLayer becomes somewhat easier than main's version. This is a
deliberate, cited change, not an accident -- and it removes a confound where the
hardest case was gratuitously hard for no stated reason.

---

## D4. Fourier feature bandwidth: sigma_B = 10, identical for every architecture

**Decision.** All Fourier-embedded architectures use sigma_B = 10 rad/unit, with no
extra 2*pi factor.

**Evidence -- derived from this problem, not borrowed.** The IC is a Gaussian
derivative, g(x) ~ x exp(-x^2/2s^2), whose transform is |G(k)| ~ k exp(-s^2k^2/2),
peaking analytically at **k = 1/s = 10.0 rad/unit** for s = 0.1. Measured on the
verified FD reference (time-averaged power spectrum, nx=2048):

| material | k_peak | k at 90% energy | k at 99% energy |
|---|---|---|---|
| Homogeneous | 9.4 | 18.8 | 37.7 |
| TwoLayer | 9.4 | 18.8 | 28.3 |
| MultiLayer | 9.4 | 18.8 | 34.5 |

Measurement confirms the analytic prediction (9.4 vs 10.0). sigma_B = 10 places the
embedding's characteristic frequency at the solution's spectral peak, with 2-sigma
covering the 90%-energy band.

**What both branches were doing:**

| model | effective bandwidth | vs k_peak = 10 |
|---|---|---|
| main PirateNet / Fourier (sigma=3) | 3.0 | 3.3x too low |
| main PIKAN (2*pi*7) | 44.0 | 4.4x too high |
| rak Fourier (sigma=1) | 1.0 | 10x too low |
| rak PirateNet (sigma=2) | 2.0 | 5x too low |

Every architecture in both branches was mis-set, in different directions and by
different factors -- so bandwidth was confounded with architecture in every
comparison made so far. Fixing it to a single derived value removes that confound.

---

## D5. Causal weighting: annealed, never a fixed epsilon

Wang, Sankaran & Perdikaris (arXiv:2203.07404) Algorithm 1: increasing sequence
eps in [1e-2, 1e-1, 1, 10, 100], advance when min_i w_i > delta = 0.99. Their eps
values presuppose an O(1) residual, which is why `residual_scale` normalisation
(D6) must be applied first.

## D6. Residual normalisation: fixed, problem-derived

Divide the residual by max|d/dx(E g')| (228 / 341 / 432 for the three materials),
computed once from the problem definition. Fixed rather than adaptive, because an
adaptive normaliser makes the objective non-stationary.
