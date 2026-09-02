"""Method of Manufactured Solutions + mutation tests.

Design constraints, each closing a class of green-but-wrong found in audit:
  * runs on TwoLayer / MultiLayer / VariableDensity, not just Homogeneous
    (a homogeneous MMS is vacuous: E'==0, so a missing E'*u_x term is invisible)
  * rho is SYMBOLIC and asserted pointwise -- with rho==1 everywhere a bug that
    drops or inverts rho is undetectable by construction
  * E and rho are both asserted against the code before their derivatives are used
  * float64 with RELATIVE tolerance; a float32 row is reported so the ~1e-6
    training-precision floor is stated, not implied
  * exits non-zero on failure (a sign-flipped PDE used to exit 0)
"""
import sys
import numpy as np, sympy as sp, torch
from core.problem import Homogeneous, TwoLayer, MultiLayer, VariableDensity
from core.operators import wave_operator

torch.set_default_dtype(torch.float64)
xs, ts = sp.symbols("x t", real=True)


def sym_fields(material):
    """Symbolic (E, rho).  NOTE: this mirrors problem.py, so it cannot detect a
    wrong CONSTANT in the spec (e.g. a moved interface).  test_spec.py pins those
    against independent literals -- that is a different job and it is done there."""
    if isinstance(material, Homogeneous):
        return sp.Integer(1), sp.Integer(1)
    if isinstance(material, TwoLayer):
        a = sp.Rational(1, 2) * (1 + sp.tanh(xs / material.w))
        return material._E1 * (1 - a) + material._E2 * a, sp.Integer(1)
    if isinstance(material, VariableDensity):
        E = 1 + sp.Rational(1, 2) * (1 + sp.tanh(xs / material.w))
        rho = 1 + sp.Rational(2, 5) * sp.sin(2 * xs) + sp.Rational(1, 5) * sp.cos(3 * xs)
        return E, rho
    if isinstance(material, MultiLayer):
        val = sp.Float(float(material.E_vals[0]))
        for k, b in enumerate(material.interfaces):
            a = sp.Rational(1, 2) * (1 + sp.tanh((xs - sp.Float(float(b))) / material.w))
            val = val * (1 - a) + sp.Float(float(material.E_vals[k + 1])) * a
        return val, sp.Integer(1)
    raise TypeError(material)


U = {
    "sin_cos": (sp.sin(sp.pi * xs) * sp.cos(sp.pi * ts),
                lambda x, t: torch.sin(np.pi * x) * torch.cos(np.pi * t)),
    "gauss":   (sp.exp(-(xs**2) / sp.Float(0.02)) * sp.cos(3 * ts),
                lambda x, t: torch.exp(-(x**2) / 0.02) * torch.cos(3 * t)),
    "shifted": (sp.sin(2 * xs + sp.Rational(1, 2)) * sp.exp(-ts / 2),
                lambda x, t: torch.sin(2 * x + 0.5) * torch.exp(-t / 2)),
}


def run_mms(material, name, *, drop_E_prime=False, n=4000, seed=0, dtype=torch.float64):
    E_s, rho_s = sym_fields(material)
    u_s, u_t = U[name]
    rng = np.random.default_rng(seed)

    # -- assert BOTH symbolic fields match the code before differentiating them --
    xq = rng.uniform(material.x_min, material.x_max, 400)
    for sym, fn, lbl in ((E_s, material.E, "E"), (rho_s, material.rho, "rho")):
        num = np.array([float(sym.subs(xs, v)) for v in xq])
        assert np.max(np.abs(num - np.asarray(fn(xq), float))) < 1e-12, \
            f"symbolic {lbl} != material.{lbl} for {material.name}"

    f_s = rho_s * sp.diff(u_s, ts, 2) - sp.diff(E_s * sp.diff(u_s, xs), xs)
    f_f = sp.lambdify((xs, ts), f_s, "numpy")
    utt_f = sp.lambdify((xs, ts), rho_s * sp.diff(u_s, ts, 2), "numpy")
    div_f = sp.lambdify((xs, ts), sp.diff(E_s * sp.diff(u_s, xs), xs), "numpy")

    x = torch.tensor(rng.uniform(material.x_min, material.x_max, (n, 1)), dtype=dtype)
    t = torch.tensor(rng.uniform(0.0, 1.0, (n, 1)), dtype=dtype)
    R = wave_operator(u_t, x.clone(), t.clone(), material,
                      drop_E_prime=drop_E_prime).detach().double().numpy()
    xn, tn = x.double().numpy(), t.double().numpy()
    F = np.broadcast_to(np.asarray(f_f(xn, tn), float), R.shape)
    a = np.abs(np.broadcast_to(np.asarray(utt_f(xn, tn), float), R.shape))
    b = np.abs(np.broadcast_to(np.asarray(div_f(xn, tn), float), R.shape))
    scale = max(np.max(a), np.max(b), 1e-30)
    return float(np.max(np.abs(R - F)) / scale)


TOL = 1e-11

if __name__ == "__main__":
    mats = [Homogeneous(), TwoLayer(), MultiLayer(), VariableDensity()]
    print(f"{'material':<16}{'u*':<10}{'rel (correct)':>15}{'rel (E'' dropped)':>19}  verdict")
    print("-" * 82)
    fails = []
    for M in mats:
        homo = isinstance(M, Homogeneous)
        for name in U:
            ok = run_mms(M, name)
            bad = run_mms(M, name, drop_E_prime=True)
            if not ok < TOL:
                fails.append(f"{M.name}/{name}: correct operator rel={ok:.2e}")
                v = "*** FAIL (operator wrong) ***"
            elif homo:
                v = "PASS (mutation invisible - EXPECTED, E'=0)"
            elif bad > 1e-6:
                v = "PASS"
            else:
                fails.append(f"{M.name}/{name}: mutation NOT detected (bad={bad:.2e})")
                v = "*** FAIL (mutation undetected) ***"
            print(f"{M.name:<16}{name:<10}{ok:>15.3e}{bad:>19.3e}  {v}")

    print("-" * 82)
    print("float32 floor (training precision) -- bugs below this are INVISIBLE at fp32:")
    for M in (TwoLayer(), MultiLayer()):
        f32 = run_mms(M, "shifted", dtype=torch.float32)
        f64 = run_mms(M, "shifted", dtype=torch.float64)
        print(f"   {M.name:<16} fp64 {f64:.2e}    fp32 {f32:.2e}")

    print("-" * 82)
    if fails:
        print("FAILURES:"); [print("  -", f) for f in fails]; sys.exit(1)
    print("ALL CHECKS PASSED"); sys.exit(0)
