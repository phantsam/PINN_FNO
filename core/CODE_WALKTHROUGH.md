# Code Walkthrough — How a PINN and a KAN Run, Line by Line

Every file, function and line number that a training run passes through, for both
the physics-informed MLPs and the tuned KAN. Line numbers were read from the code,
not recalled, and refer to the `phase10-hybrid-and-cancellation` branch.

Companion documents: `MODELS_AND_MECHANISMS.md` (concepts from scratch),
`BUGS_AND_CORRECTIONS.md` (defect record).

> **Known documentation error.** `MODELS_AND_MECHANISMS.md` (lines 630–631, 1010)
> and the constraints slide of `PINN_vs_KAN.pptx` state the absorbing boundary
> conditions with the signs swapped. **The code is correct** — see Step 4e:
> at `x = +1` it is `u_t + c·u_x = 0`, at `x = −1` it is `u_t − c·u_x = 0`.

---

## The map

One command runs everything:

```
python -m core.phase10 --models mlp,fourier,rung:charcoords50 --materials homogeneous --seeds 0
```

| file | job |
|---|---|
| `phase10.py` | **entry point** — loops over materials, models and seeds |
| `models.py` | the **PINNs** (MLP, Fourier MLP) |
| `phase8.py` + `kan_variants.py` | the **tuned KAN** configuration and wrapper |
| `coords.py` | the **characteristic coordinates** (KAN only) |
| pykan `MultKAN.py`, `KANLayer.py`, `spline.py` | the **KAN layers and B-splines** |
| `kan_patch.py` | fix for a pykan bug, installed on import |
| `hybrid.py` | **training** — Sobol points, the Adam and L-BFGS loops |
| `operators.py` | the **ansatz**, and the **residual** via autograd |
| `losses.py` | PDE loss and boundary loss |
| `problem.py` | the **materials** `E(x)`, `ρ(x)`, `c(x)`, and the **pulse** `g(x)` |
| `evaluate.py`, `rescore.py`, `metrics.py` | **testing** |

The call order:

```
phase10.main
  └─ make()                      → which model
  └─ train_hybrid()              → training
       ├─ sobol_pool()           → 10,000 points
       ├─ Adam loop ─┐
       └─ L-BFGS loop┴─ objective → pde_loss → wave_operator → u_fn
                                                                ├─ model(x,t)   PINN or KAN
                                                                └─ ansatz       g·decay + growth·N
                         + bc_loss → abc_operator → u_fn
  └─ score()                     → test error
```

---

## Step 1 — Entry point: `phase10.py`

```python
 92  for mk in A.materials.split(","):
 93      M = MATERIALS[mk]()                          # e.g. Homogeneous()
 94      xr, ur = fine_reference(M, A.nx)             # test reference, nx=2048
 ...
 99      for arch in A.models.split(","):             # "mlp", "fourier", "rung:charcoords50"
101          for sd in seeds:
109              m, mm = train_hybrid(make(arch, M, sd), M, adam_steps=A.adam_steps,
110                                   ansatz_kind="none" if A.soft_ic else "legacy",
...
115              v = score(m, M, xr, ur, dev, ...)    # the reported rel-L2 %
```

- **Line 93** — `MATERIALS` (`problem.py:155`) maps a name to a class; `()` builds it.
- **Line 109** — `make()` decides *which* model; `train_hybrid()` trains it.
- **Line 115** — `score()` tests the finished model. That number is what gets reported.

### Choosing the model — `phase10.py:44-56`

```python
54  if arch.startswith("rung:"):
55      return lambda: build_rung(arch.split(":", 1)[1], M, seed=seed + 1)
56  return REGISTRY[arch]
```

It returns a **recipe** — a function that builds the model when called — not the
model itself. The model is built later, inside `train_hybrid`, after the random seed
is set.

- **PINN** (`"mlp"`, `"fourier"`) → line 56 → `REGISTRY` in `models.py:278`.
- **KAN** (`"rung:charcoords50"`) → line 55 → `build_rung`, i.e. `phase8.build`.

---

## Step 2a — The PINN models: `models.py`

### Registry — `models.py:279-280`

```python
279  "mlp":     lambda: MLP(5, 128),
280  "fourier": lambda: FourierMLP(5, 128, 64, SIGMA_B),     # SIGMA_B = 10.0 (line 17)
```

### Plain MLP — `models.py:35-50`

```python
40  L = [nn.Linear(2, units), nn.Tanh()]                 # 2 → 128, then tanh
41  for _ in range(layers - 1):                          # 4 more times:
42      L += [nn.Linear(units, units), nn.Tanh()]        #   128 → 128, then tanh
43  L.append(nn.Linear(units, 1))                        # 128 → 1
44  self.net = nn.Sequential(*L)
46  if isinstance(m, nn.Linear):
47      nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)   # random start

49  def forward(self, x, t):
50      return self.net(torch.cat([x, t], dim=-1))       # (N,1)+(N,1) → (N,2) → ... → (N,1)
```

Line 50 is the whole forward pass: glue `x` and `t` side by side, then run the layer
stack.

### Fourier MLP — `models.py:20-31` and `53-68`

```python
26  self.register_buffer("B", torch.randn(64, 2) * sigma)   # FIXED random frequencies, not trained
30  p = torch.cat([x, t], dim=-1) @ self.B.T                  # (N,2) @ (2,64) → (N,64)
31  return torch.cat([torch.cos(p), torch.sin(p)], dim=-1)    # → (N,128)

58  L = [nn.Linear(self.embed.out_dim, units), nn.Tanh()]    # 128 → 128 (not 2 → 128)
67  def forward(self, x, t):
68      return self.net(self.embed(x, t))                     # embed first, then the same MLP
```

- **`register_buffer`** (line 26) stores `B` with the model but **excludes it from
  training**; the frequencies stay fixed for the whole run.
- **Line 58** — the only difference from the plain MLP: the first layer takes 128
  inputs (the sines and cosines) instead of 2.

---

## Step 2b — The KAN model

### Configuration — `phase8.py:75` and `81-92`

```python
75  "charcoords50": (dict(width=W, grid=50, k=5, base_fun="silu", char_coords=True), None),
                                                          # W = (2, 20, 20, 20, 1)
87  if kw.get("char_coords"):
88      kw["material"] = material                         # needed to compute tau(x)
89  kw["seed"] = seed
92  return TunedKAN(**kw)
```

### Wrapper — `kan_variants.py`

```python
 37  _N_PATCHED = kan_patch.install()                     # runs once, on import

115  self.chars = CharCoords(material, t_max=t_max) if char_coords else None
126  self.kan = KAN(width=list(width), grid=grid, k=k, base_fun="silu",
127                 grid_eps=grid_eps, noise_scale=noise_scale, seed=seed,
128                 symbolic_enabled=False, auto_save=False, ckpt_path=ckpt)

163  def _inputs(self, x, t):
164      if self.chars is not None:
165          return self.chars(x, t)                      # ← charcoords50 takes this branch
...
172  def forward(self, x, t):
173      return self.kan(self._inputs(x, t))              # coordinates first, then pykan
```

- **Line 37** — `kan_patch.install()` replaces pykan's `curve2coef` (which fits spline
  coefficients to curve values) with a stable version; the original silently returns
  NaN on GPU. See `kan_patch.py:56-89`.
- **Line 126** — `KAN` is the authors' own pykan library. `symbolic_enabled=False`
  turns off a symbolic-regression branch we don't use.
- **Line 173** — the whole forward pass: transform coordinates, then run the KAN.

### Characteristic coordinates — `coords.py:156-176`

```python
163  T = self.tau.tau_max                                  # 2.0 for the uniform rod
164  self.register_buffer("xi_lo",  torch.tensor(-float(t_max)))   # -1
165  self.register_buffer("xi_hi",  torch.tensor(float(T)))        #  2
166  self.register_buffer("eta_lo", torch.tensor(0.0))             #  0
167  self.register_buffer("eta_hi", torch.tensor(float(T + t_max)))#  3

170  def forward(self, x, t):
171      tau = self.tau(x)                                  # travel time, (N,1)
172      xi  = tau - t                                      # right-moving coordinate
173      eta = tau + t                                      # left-moving coordinate
174      xi  = 2 * (xi  - self.xi_lo)  / (self.xi_hi  - self.xi_lo)  - 1    # squeeze to [-1,1]
175      eta = 2 * (eta - self.eta_lo) / (self.eta_hi - self.eta_lo) - 1
176      return torch.cat([xi, eta], dim=-1)                # (N,2)
```

Lines 164–167 are the ranges `ξ ∈ [−1, 2]` and `η ∈ [0, 3]`. Lines 174–175 are the
rescaling that turns `(x, t) = (0.30, 0.10)` into `(ξ, η) = (+0.4667, −0.0667)`.

`self.tau(x)` runs `TravelTime.forward` (`coords.py:130-153`): it looks `x` up in a
precomputed table of 4,097 points (line 139) and interpolates smoothly between them
(lines 141–152), in float64 (line 138) for accuracy. It has no trainable parameters.

### Inside pykan — the network loop: `MultKAN.py:798-872`

```python
798  for l in range(self.depth):                           # 4 layers
800      x_numerical, preacts, postacts_numerical, postspline = self.act_fun[l](x)
...
867      x = self.node_scale[l][None,:] * x + self.node_bias[l][None,:]
872  return x                                              # (N,1)
```

`node_scale` and `node_bias` are fixed at 1 and 0 and are not trainable (checked), so
each iteration just runs one KAN layer.

### One KAN layer — `KANLayer.py:153-167`

```python
156  base = self.base_fun(x)                               # silu(x),        (N, in)
157  y = coef2curve(x_eval=x, grid=self.grid, coef=self.coef, k=self.k)
                                                           # spline values, (N, in, out)
161  y = self.scale_base[None,:,:] * base[:,:,None] + self.scale_sp[None,:,:] * y
                                                           # φ = scale_base·silu + scale_sp·spline
162  y = self.mask[None,:,:] * y                           # per-edge on/off switch (pruning)
166  y = torch.sum(y, dim=1)                               # NODE = SUM over inputs → (N, out)
167  return y, preacts, postacts, postspline
```

**Line 161 is every edge's curve. Line 166 is the node adding up its incoming
edges.** Those two lines are the entire KAN idea.

### The B-spline — `spline.py:51-78` and `34-47`

```python
75  b_splines = B_batch(x_eval, grid, k=k)                            # (N, in, 55) bump values
76  y_eval = torch.einsum('ijk,jlk->ijl', b_splines, coef)            # weighted sum → (N, in, out)
```

- **Line 75** — evaluates all 55 bumps at every input.
- **Line 76** — multiplies each bump by its coefficient and adds them, separately for
  every edge. `coef` has shape `(in, out, 55)`.

`B_batch` builds the bumps recursively: at order 0 they are simple on/off boxes
(line 38); each higher order blends neighbouring bumps (line 42), up to `k = 5`.

---

## Step 3 — Training setup: `hybrid.py:117-162`

From here on, **PINN and KAN run identical code.**

```python
118  torch.manual_seed(seed); np.random.seed(seed)          # fixes all randomness
119  gen = torch.Generator(device=device).manual_seed(seed)

121  model = arch_fn().to(device)                           # the model is built now, moved to GPU
122  ansatz = make_ansatz(ansatz_kind, sigma_g=sigma_g)     # "legacy" = the decay/growth ansatz
123  u_fn = lambda x, t: ansatz(model(x, t), x, t)          # u = ansatz(network output)

131  _, _, tb = sample(1, n_bc, material, t_max, device, gen)   # 512 boundary TIMES
132  tb = tb.detach()

143  live = [q for q in model.parameters() if q.requires_grad]  # the trainable numbers

161  xc, tc = ... sobol_pool(n_col, material, t_max, device, seed)   # the 10,000 points
```

**Line 123 matters most.** `u_fn` is what the rest of the code treats as "the
solution". Calling `u_fn(x, t)` runs the network and then the ansatz.

### The 10,000 points — `hybrid.py:59-65`

```python
61  eng = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=seed)
62  pts = eng.draw(n)                                               # (10000, 2), each in [0,1]
63  x = (pts[:, 0:1] * (x_max - x_min) + x_min).to(device)          # column 0 → x in [-1, 1]
64  t = (pts[:, 1:2] * t_max).to(device)                            # column 1 → t in [0, 1]
65  return x.detach(), t.detach()                                   # two (10000, 1) tensors
```

`dimension=2` means each Sobol point is an `(x, t)` pair drawn together.

### Boundary times — `train.py:27-31`

```python
30  tb = torch.rand(n_bc, 1, generator=gen, device=device) * t_max  # 512 random times
```

---

## Step 4 — One Adam step: `hybrid.py:172-196`

```python
173  for step in range(adam_steps):                         # 700 steps
177      opt.zero_grad(set_to_none=True)                    # clear previous gradients
178      loss = objective(xc, tc)                           # forward pass → ONE number
181      loss.backward(); opt.step(); lr_sched.step()       # gradients → update → adjust lr
182      if not _finite(model):                             # stop if any weight became NaN/inf
183          diverged = True; break
```

`objective` → `terms` (lines 146–157):

```python
147  lp, _, _ = pde_loss(u_fn, x.clone(), t.clone(), material, rs, eps=None)   # interior
150  aux = aux + bc_loss(u_fn, tb.clone(), material, bs)                       # boundary
157  return ... lp + aux                                                        # total loss
```

### 4a. PDE loss — `losses.py:80-85`

```python
83  R = wave_operator(u_fn, x, t, material) / scale        # residual at each point, (10000,1)
84  if eps is None or n_chunks <= 1:
85      return (R**2).mean(), None, None                   # square, average → ONE number
```

`scale = 1.0` (set at `hybrid.py:130`), so there is no normalisation.

### 4b. The residual — `operators.py:15-41`

```python
15  def _grad(y, x):
16      return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
17                                 create_graph=True)[0]

30  x = x.requires_grad_(True)                             # tell autograd to track x
31  t = t.requires_grad_(True)                             # tell autograd to track t
32  u = u_fn(x, t)                                         # ← NETWORK + ANSATZ run here
33  u_t  = _grad(u, t)                                     # ∂u/∂t
34  u_tt = _grad(u_t, t)                                   # ∂²u/∂t²
35  u_x  = _grad(u, x)                                     # ∂u/∂x
40  div  = _grad(material.E(x) * u_x, x)                   # ∂/∂x (E · u_x)
41  return material.rho(x) * u_tt - div                    # ρ·u_tt − ∂/∂x(E·u_x)
```

- **Lines 30–31** — `requires_grad_` is what makes differentiating *with respect to
  position and time* possible.
- **Line 32** — the only place in the loss where the network actually runs.
- **Line 16** — `create_graph=True` keeps each derivative itself differentiable: needed
  to take `u_tt` from `u_t`, and so `backward()` can push gradients through the
  derivatives into the weights.
- **Line 40** — differentiating `E·u_x` as one product automatically includes the
  `E'(x)·u_x` interface term.

### 4c. The ansatz — `operators.py:84-88`

```python
85  def f(nn_out, x, t):
86      g = gaussian_ic(x, sigma_g)                                  # starting shape, x only
87      return (g * torch.exp(-0.5 * (decay_rate * t) ** 2)          # g · decay(t),   decay_rate = 15
88              + torch.tanh(growth_rate * t) ** 2 * nn_out)         # + growth(t) · N, growth_rate = 25
```

Line 86 takes only `x`. Lines 87–88 are the two volume knobs.

#### The pulse — `problem.py:162-181`

```python
173  f = torch.exp(-0.5 * ((x - x0) / sigma_g) ** 2)       # Gaussian bell
176  dfdx = -(x - x0) / sigma_g**2 * f                      # its derivative
180  peak = 1.0 / (sigma_g * math.sqrt(math.e))             # = 6.065 for sigma = 0.1
181  return dfdx / peak                                     # scaled so the peak is exactly 1
```

### 4d. The materials — `problem.py`

```python
 71  def E(self, x):   return _ones_like(x, 1.0)          # Homogeneous: E = 1 everywhere
 74  def rho(self, x): return _ones_like(x, 1.0)

 88  def E(self, x):                                        # TwoLayer
 89      a = 0.5 * (1.0 + _tanh(x / self.w))                #   smooth step, width w = 0.02
 90      return self._E1 * (1.0 - a) + self._E2 * a         #   blends 1.0 → 1.5

111  self.E_vals = np.linspace(60.0, 150.0, 6) / 60.0       # MultiLayer: 1.0 … 2.5
121      a = 0.5 * (1.0 + _tanh((x - b) / self.w))          #   one smooth step per interface, w = 0.05

 38  def c(self, x):
 41      return torch.sqrt(E / r)                           # wave speed = sqrt(E/ρ)
```

### 4e. Boundary loss — `losses.py:109-116` and `operators.py:44-56`

```python
112  xL = torch.full_like(t, material.x_min)               # 512 copies of x = -1
113  xR = torch.full_like(t, material.x_max)               # 512 copies of x = +1
114  rL = abc_operator(u_fn, xL, t.clone(), material, "left")
115  rR = abc_operator(u_fn, xR, t.clone(), material, "right")
116  return ((rL**2).mean() + (rR**2).mean()) / scale**2

 53  u = u_fn(x, t)                                        # network + ansatz again, at the ends
 54  u_t, u_x = _grad(u, t), _grad(u, x)
 55  n = 1.0 if side == "right" else -1.0
 56  return u_t + n * material.c(x) * u_x                  # right: u_t + c·u_x,  left: u_t − c·u_x
```

Why these signs: a wave leaving through the right end moves right, `u = F(x − ct)`,
so `u_t = −c·F'` and `u_x = F'`, giving `u_t + c·u_x = 0`. The left end is the mirror
image.

### 4f. Back to `hybrid.py:181`

```python
181  loss.backward(); opt.step(); lr_sched.step()
```

- **`loss.backward()`** runs the chain rule backwards through everything above — the
  residual, the derivatives, the ansatz and the network (the MLP's `Linear` layers, or
  the KAN's spline coefficients) — storing a gradient in every trainable parameter.
- **`opt.step()`** — Adam uses those gradients to update the parameters, once.
- **`lr_sched.step()`** — adjusts the learning rate: warm-up, then decay (lines 165–167).

---

## Step 5 — The L-BFGS phase: `hybrid.py:198-228`

```python
201  xc, tc = xc.detach().clone(), tc.detach().clone()    # freeze the same 10,000 points
202  opt = optim.LBFGS(live, lr=1.0, max_iter=20, history_size=100,
203                    line_search_fn="strong_wolfe")
206  for ep in range(lbfgs_epochs):                         # up to 3000
207      def closure():                                     # L-BFGS calls this ~20–26 times per step
208          opt.zero_grad(set_to_none=True)
209          l = objective(xc, tc)                          # exact same forward as Adam
210          if torch.isfinite(l):
211              l.backward()
212          return l
213      v = float(opt.step(closure).detach())
216      if not _finite(model) or not np.isfinite(v):       # check WEIGHTS, not the loss
217          diverged = True; break
218      if v < best - min_delta:                           # improved?
219          best, wait = v, 0
220          best_w = {k: c.detach().clone() for k, c in model.state_dict().items()}   # remember
222      else:
223          wait += 1
224          if wait >= patience:                           # 50 steps with no improvement → stop
225              break
227  if best_w:
228      model.load_state_dict(best_w)                      # restore the best weights
```

- **`closure`** (line 207) — L-BFGS needs a function it can call repeatedly, because it
  probes the loss surface several times before committing to a step.
- **Line 216** — checks the *weights* for NaN, because L-BFGS reports the loss from
  *before* its step, which can hide a blow-up.
- **Lines 218–228** — keep the weights with the lowest *training loss*. The true answer
  is never used to choose.

---

## Step 6 — Testing

### In-loop check — `hybrid.py:231` → `evaluate.py`

Runs once, after training: a quick nx=512 check that also computes the held-out
residual on fresh random points and the collapse detector (flags a run whose late-time
amplitude falls below 30 % of the truth). It is **not** the headline number.

### The reported number — `rescore.py:66-82`

```python
75  X  = torch.tensor(x_ref, ...).reshape(-1, 1)           # 4096 x positions
76  tt = torch.tensor(TARGETS, ...)                        # 20 times, 0.05 … 1.00
77  xg = X.repeat(len(TARGETS), 1)                         # (81920, 1)
78  tg = tt.repeat_interleave(len(x_ref)).reshape(-1, 1)   # (81920, 1)
79  ans = make_ansatz(ansatz_kind, sigma_g=sigma_g)
80  with torch.no_grad():                                  # no gradients — just predict
81      pred = ans(model(xg, tg), xg, tg).reshape(20, -1)  # (20, 4096)
82  return float(spacetime_rel_l2(pred, u_ref))
```

- **Lines 77–78** — build the 81,920-point test grid. None of these points coincides
  with a training point, for any of the 11 seeds (checked).
- **Line 80** — `no_grad`, because testing needs no derivatives.
- **Line 81** — the same `model` and the same ansatz as in training.

### The reference — `rescore.py:55-63`

```python
57  x, t, u = fd_reference(material, nx=nx, T=T, sigma_g=sigma_g)   # finite-difference solver
61      w = (tv - t[j - 1]) / (t[j] - t[j - 1])                      # interpolate to exact times
62      out.append((1 - w) * u[j - 1] + w * u[j])
```

The finite-difference solver is in `reference.py`. For the exact homogeneous headline,
`reference.py:82` `dalembert()` is used instead, which has no discretisation error.

### The metric — `metrics.py:16-19`

```python
18  den = float(np.linalg.norm(u_ref))
19  return 100.0 * float(np.linalg.norm(u_pred - u_ref)) / max(den, 1e-30)
```

---

## PINN vs KAN — where the code actually differs

| stage | PINN | KAN |
|---|---|---|
| choose model | `phase10.py:56` → `REGISTRY` | `phase10.py:55` → `phase8.build` → `TunedKAN` |
| input transform | none (MLP) or `FourierEmbed` `models.py:30-31` | `CharCoords` `coords.py:170-176` |
| layers | `nn.Linear` + `nn.Tanh` `models.py:40-43` | `KANLayer.forward` `KANLayer.py:153-167` |
| what's trained | weight matrices and biases | spline `coef`, `scale_base`, `scale_sp` |
| **everything else** | **identical** — `hybrid.py`, `operators.py`, `losses.py`, `problem.py`, `rescore.py` | **identical** |

**The whole difference between the two families is `model(x, t)` inside `u_fn` on
`hybrid.py:123`.** Every other line of the pipeline is shared — which is what makes the
comparison fair.
