"""External validation against a published, open-source benchmark.

Reference: "A Unified Benchmark of Physics-Informed Neural Networks and
Kolmogorov-Arnold Networks for ODEs and PDEs", arXiv:2602.15068,
code at github.com/Salva-D/pinn-vs-pikan  (pdes/wave/problem_data.py).

Their problem, reproduced EXACTLY:
    PDE      u_tt = u_xx                    on x in [0,1], t in [0,1]
    exact    u(x,t) = cos(pi t) sin(pi x)
    IC       u(x,0) = sin(pi x),  u_t(x,0) = 0        [soft]
    BC       u(0,t) = u(1,t) = 0                      [soft, Dirichlet]
    all four constraints are SOFT penalties -- no ansatz
    100x100 uniform collocation grid, Adam lr=1e-3, 10000 iterations

Their published wave-equation result (10 repetitions):
    PINN   2 x 24, 697 params -> 0.556 % (sd 0.170)
    PIKAN  3 x  5, 650 params -> 0.180 % (sd 0.00073)

If our machinery reproduces those numbers, it is validated externally and the
divergence from our own Phase-4 result is attributable to the problem class
(broadband travelling wave in layered media) rather than to our implementation.
"""
from __future__ import annotations
import argparse, json, math
import numpy as np
import torch
import torch.nn as nn

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def u_exact(x, t):
    return torch.cos(math.pi * t) * torch.sin(math.pi * x)


def dy_dx(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


def losses(model, x, t):
    """Byte-equivalent to their pdes/wave/problem_data.py::losses."""
    xt = torch.cat((x, t), 1)
    t0 = torch.zeros_like(t).requires_grad_(True)
    xt0 = torch.cat((x.detach(), t0), 1)
    x0t = torch.cat((torch.zeros_like(x), t), 1)
    x1t = torch.cat((torch.ones_like(x), t), 1)

    u = model(xt)
    u_t = dy_dx(u, t); u_tt = dy_dx(u_t, t)
    u_x = dy_dx(u, x); u_xx = dy_dx(u_x, x)

    ut0 = model(xt0)
    ut0_t = dy_dx(ut0, t0)

    r_pde = u_tt - u_xx
    b1 = ut0 - torch.sin(math.pi * x)      # u(x,0) = sin(pi x)
    b2 = ut0_t                             # u_t(x,0) = 0
    b3 = model(x0t)                        # u(0,t) = 0
    b4 = model(x1t)                        # u(1,t) = 0
    return (r_pde**2).mean(), ((b1**2).mean() + (b2**2).mean()
                               + (b3**2).mean() + (b4**2).mean())


class PINN(nn.Module):
    def __init__(self, hidden=2, width=24):
        super().__init__()
        L = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(hidden - 1):
            L += [nn.Linear(width, width), nn.Tanh()]
        L.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*L)

    def forward(self, xt):
        return self.net(xt)


class SplineKANLayer(nn.Module):
    """B-spline KAN edge functions, grid G, order k -- their PIKAN basis."""

    def __init__(self, i, o, G=5, k=3):
        super().__init__()
        self.k, self.nb = k, G + k
        # pykan carries TWO per-edge scales (scale_base, scale_sp) plus G+k coefs,
        # i.e. (2 + G + k) params per edge -- their num_parameters_kan() formula.
        self.scale_base = nn.Parameter(torch.empty(o, i)); nn.init.kaiming_uniform_(self.scale_base, a=math.sqrt(5))
        self.scale_sp = nn.Parameter(torch.ones(o, i))
        self.coef = nn.Parameter(torch.empty(o, i, self.nb)); nn.init.xavier_uniform_(self.coef)
        g = torch.linspace(-1, 1, G + 1); s = g[1] - g[0]
        self.register_buffer("grid", torch.cat([
            torch.linspace(g[0] - k * s, g[0] - s, k), g,
            torch.linspace(g[-1] + s, g[-1] + k * s, k)]))

    def forward(self, x):
        z = x.unsqueeze(-1); g = self.grid
        b = ((z >= g[:-1]) & (z < g[1:])).to(x.dtype)
        for d in range(1, self.k + 1):
            b = ((z - g[:-d-1]) / (g[d:-1] - g[:-d-1]) * b[:, :, :-1]
                 + (g[d+1:] - z) / (g[d+1:] - g[1:-d]) * b[:, :, 1:])
        return (torch.nn.functional.silu(x) @ self.scale_base.T
                + torch.einsum("bik,oik->bo", b.contiguous(),
                               self.scale_sp.unsqueeze(-1) * self.coef))


class PIKAN(nn.Module):
    """Hand-rolled spline KAN.  RETAINED ONLY AS A CONTROL -- it reproduces the
    paper's PINN but is 6x worse than its PIKAN (1.10 % vs 0.180 %).  Use PIKANRef."""

    def __init__(self, hidden=3, width=5, G=5, k=3):
        super().__init__()
        dims = [2] + [width] * hidden + [1]
        self.layers = nn.ModuleList([SplineKANLayer(a, b, G, k)
                                     for a, b in zip(dims[:-1], dims[1:])])

    def forward(self, xt):
        h = xt
        for i, l in enumerate(self.layers):
            h = l(h)
            if i < len(self.layers) - 1:
                h = torch.tanh(h)
        return h


class PIKANRef(nn.Module):
    """The paper's actual PIKAN: pykan, width [2,5,5,5,1], G=5, k=3.

    The hand-rolled layer above matches the published PARAMETER COUNT exactly
    (650) but not the published ACCURACY.  Since the PINN arm reproduces the
    paper to within 6 %, the shared infrastructure is validated and the gap is
    isolated to the KAN layer itself -- so use the reference implementation.
    """

    def __init__(self, width=(2, 5, 5, 5, 1), G=5, k=3):
        super().__init__()
        from kan import KAN
        self.kan = KAN(width=list(width), grid=G, k=k, symbolic_enabled=False,
                       auto_save=False, ckpt_path="/tmp/_pykan_bench")

    def forward(self, xt):
        return self.kan(xt)


def run(kind, seed, iters, nx=100, nt=100, lr=1e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    model = {"pinn": lambda: PINN(2, 24),
             "pikan": lambda: PIKAN(3, 5, 5, 3),
             "pikan_ref": lambda: PIKANRef()}[kind]().to(DEV)
    npar = sum(p.numel() for p in model.parameters())

    x = torch.linspace(0, 1, nx, requires_grad=True)
    t = torch.linspace(0, 1, nt, requires_grad=True)
    x, t = torch.meshgrid(x, t, indexing="ij")
    x = x.reshape(-1, 1).to(DEV); t = t.reshape(-1, 1).to(DEV)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        lp, lb = losses(model, x, t)
        (lp + lb).backward()
        opt.step()

    # evaluate on a fresh 500x500 grid against the exact solution
    with torch.no_grad():
        xe = torch.linspace(0, 1, 500); te = torch.linspace(0, 1, 500)
        XE, TE = torch.meshgrid(xe, te, indexing="ij")
        XE = XE.reshape(-1, 1).to(DEV); TE = TE.reshape(-1, 1).to(DEV)
        pred = model(torch.cat((XE, TE), 1))
        ref = u_exact(XE, TE)
        rel = 100.0 * float(torch.linalg.norm(pred - ref) / torch.linalg.norm(ref))
    return rel, npar


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", default="bench_external.json")
    ap.add_argument("--archs", default="pinn,pikan")
    A = ap.parse_args()
    PUB = {"pinn": (0.556, 0.170, 697), "pikan": (0.180, 0.00073, 650),
           "pikan_ref": (0.180, 0.00073, 650)}
    res = {}
    for kind in A.archs.split(","):
        errs = []
        for s in range(A.reps):
            e, npar = run(kind, s, A.iters)
            errs.append(e)
            print(f"  {kind:<6} seed {s}  rel-L2 {e:8.4f}%", flush=True)
        m, sd = float(np.mean(errs)), float(np.std(errs))
        pm, psd, pp = PUB[kind]
        res[kind] = dict(mean=m, sd=sd, params=npar, errs=errs,
                         published_mean=pm, published_sd=psd, published_params=pp)
        print(f"  --> {kind}: ours {m:.4f}% (sd {sd:.4f}, p={npar}) | "
              f"published {pm}% (sd {psd}, p={pp})\n", flush=True)
        json.dump(res, open(A.out, "w"), indent=1)
