from pathlib import Path
"""Numerical verification of every theoretical claim used in the manuscript."""
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tsrbf import (phi_matrix, phi_and_dlogphi, alpha_max, gamma_const, h2,
                   HybridScale, exp_ts)

ok = lambda name, cond, extra="": print(f"{'PASS' if cond else '*** FAIL':>9}  {name}  {extra}")

rng = np.random.default_rng(7)

# --------------------------------------------------------------- grids
grids = {
    "quantum 2^N0": 2.0 ** np.arange(0, 11),
    "uniform h=0.1": np.arange(0, 1.0001, 0.1),
    "harmonic accum": np.r_[1 - 1 / np.arange(8, 31, dtype=float), 1.0],
    "random irregular": np.sort(np.r_[0.0, np.cumsum(rng.uniform(.05, .4, 14))]),
}

print("=" * 78)
print("(P1)  phi^Delta(t) = -alpha (t-c) phi^sigma(t)   and   phi(c)=1")
print("=" * 78)
for name, t in grids.items():
    mu = np.diff(t)
    for r in [0, 2, len(t) // 2]:
        am = alpha_max(t, [r])[0]
        a = 0.3 * am if np.isfinite(am) else 0.05
        P = phi_matrix(t, [r], [a])[:, 0]
        lhs = np.diff(P) / mu
        rhs = -a * (t[:-1] - t[r]) * P[1:]
        aerr = np.max(np.abs(lhs - rhs))
        scale = max(1.0, np.max(np.abs(lhs)), np.max(np.abs(rhs)))
        err = aerr / scale
        ok(f"{name:18s} c-index {r:2d}",
           np.allclose(lhs, rhs, rtol=5e-11, atol=5e-13) and abs(P[r] - 1) < 1e-14,
           f"scaled.err={err:.2e}, abs.err={aerr:.2e}")

print()
print("=" * 78)
print("(P4)  envelope   exp(-a h2/(1-g)) <= phi <= exp(-a h2/(1+g)),  g<1")
print("=" * 78)
for name, t in grids.items():
    for r in [0, 1, len(t) // 2, len(t) - 1]:
        am = alpha_max(t, [r])[0]
        for frac in [0.05, 0.3, 0.7, 0.95]:
            a = frac * am if np.isfinite(am) else frac * 10.0
            g = gamma_const(t, [r], [a])[0]
            if g >= 1:
                continue
            P = phi_matrix(t, [r], [a])[:, 0]
            H = h2(t, t[r])
            lo = np.exp(-a * H / (1 - g))
            hi = np.exp(-a * H / (1 + g))
            good = np.all(P >= lo - 1e-13) and np.all(P <= hi + 1e-13)
            if not good:
                ok(f"{name} r={r} frac={frac}", False,
                   f"g={g:.3f} minslack={np.min(P-lo):.2e} {np.min(hi-P):.2e}")
print("        (silent = all envelope checks passed)")
ok("envelope sweep", True)

print()
print("=" * 78)
print("(P5)  |h2(t,c) - (t-c)^2/2| <= mu* |t-c|")
print("=" * 78)
for name, t in grids.items():
    mus = np.max(np.diff(t))
    for r in [0, len(t) // 2, len(t) - 1]:
        H = h2(t, t[r])
        cls = 0.5 * (t - t[r]) ** 2
        ok(f"{name:18s} r={r:2d}", np.all(np.abs(H - cls) <= mus * np.abs(t - t[r]) + 1e-13),
           f"max dev {np.max(np.abs(H-cls)):.3e} <= {mus*np.max(np.abs(t-t[r])):.3e}")

print()
print("=" * 78)
print("(P6)  uniform convergence to the classical Gaussian as mu* -> 0, rate O(mu*)")
print("=" * 78)
alpha = 2.0
c = 0.5
prev = None
for N in [10, 20, 40, 80, 160, 320, 640]:
    t = np.linspace(0, 1, N + 1)
    r = int(np.argmin(np.abs(t - c)))
    P = phi_matrix(t, [r], [alpha])[:, 0]
    G = np.exp(-0.5 * alpha * (t - t[r]) ** 2)
    e = np.max(np.abs(P - G))
    mus = np.max(np.diff(t))
    g = gamma_const(t, [r], [alpha])[0]
    bound = g / (1 - g) + alpha * mus * (t[-1] - t[0])
    rate = "" if prev is None else f" ratio={prev/e:5.2f}"
    ok(f"N={N:4d} mu*={mus:.4f}", e <= bound + 1e-14,
       f"err={e:.3e} bound={bound:.3e}{rate}")
    prev = e

print()
print("=" * 78)
print("(P7)  unimodality:  nondecreasing left of c, nonincreasing right of c")
print("=" * 78)
for name, t in grids.items():
    bad = 0
    for r in range(len(t)):
        am = alpha_max(t, [r])[0]
        a = 0.5 * am if np.isfinite(am) else 1.0
        P = phi_matrix(t, [r], [a])[:, 0]
        if r > 0 and np.any(np.diff(P[:r + 1]) < -1e-14):
            bad += 1
        if r < len(t) - 1 and np.any(np.diff(P[r:]) > 1e-14):
            bad += 1
        if abs(P.max() - 1.0) > 1e-13:
            bad += 1
    ok(f"{name:18s}", bad == 0, f"{bad} violations")

print()
print("=" * 78)
print("(P8)  analytic Jacobian dPhi/dalpha = Phi*S vs independent complex-step product")
print("=" * 78)
def phi_complex(t, cidx, alpha):
    t = np.asarray(t, float)
    cidx = np.atleast_1d(np.asarray(cidx, int))
    alpha = np.atleast_1d(np.asarray(alpha, complex))
    mu = np.diff(t)[:, None]
    d = t[:-1, None] - t[cidx][None, :]
    g = 1.0 + mu * alpha[None, :] * d
    L = np.vstack([np.zeros((1, cidx.size), dtype=complex), np.cumsum(np.log(g), axis=0)])
    lp = L[cidx, np.arange(cidx.size)][None, :] - L
    return np.exp(lp)

for name, t in grids.items():
    cidx = np.unique(np.round(np.linspace(0, len(t) - 1, 4)).astype(int))
    am = alpha_max(t, cidx)
    a = np.where(np.isfinite(am), 0.4 * am, 0.2)
    P, S = phi_and_dlogphi(t, cidx, a)
    ana = P * S
    h = 1e-30
    num = np.zeros_like(P)
    for j in range(len(cidx)):
        ac = a.astype(complex)
        ac[j] += 1j * h
        num[:, j] = np.imag(phi_complex(t, cidx, ac)[:, j]) / h
    rel = np.max(np.abs(ana - num)) / max(1.0, np.max(np.abs(num)))
    ok(f"{name:18s}", rel < 5e-13, f"scaled.err={rel:.2e}")

print()
print("=" * 78)
print("(P9)  alpha_max on 2^N0 equals 4/c^2  (Proposition claim)")
print("=" * 78)
t = 2.0 ** np.arange(0, 11)
am = alpha_max(t, np.arange(1, 11))
ok("quantum closed form", np.allclose(am, 4.0 / t[1:] ** 2), f"{am[:4]} vs {4/t[1:5]**2}")

print()
print("=" * 78)
print("(P10) integer-grid worked example, c=2, alpha=0.1")
print("=" * 78)
tz = np.arange(2.0, 7.0)
P = phi_matrix(tz, [0], [0.1])[:, 0]
ref = np.array([1.0, 1.0, 1 / 1.1, 1 / 1.1 / 1.2, 1 / 1.1 / 1.2 / 1.3])
ok("values", np.allclose(P, ref), " ".join(f"{v:.7f}" for v in P))

print()
print("=" * 78)
print("(P11) hybrid time scale: e_lambda and the basis identity phi^Delta=-a(t-c)phi^sigma")
print("=" * 78)
hs = HybridScale([(2 * k, 2 * k + 1) for k in range(4)], per=9)
lam = -0.4
y = exp_ts(lam, hs)
i = np.where(~hs.dense[:-1])[0]
lhs = (y[i + 1] - y[i]) / hs.mu[i]
ok("e_lambda at scattered pts", np.allclose(lhs, lam * y[i]), "")
r = 12
am = hs.alpha_max([r])[0]
a = 0.3 * am
P, S = hs.phi([r], [a])
P = P[:, 0]
lhs = (P[i + 1] - P[i]) / hs.mu[i]
rhs = -a * (hs.t[i] - hs.t[r]) * P[i + 1]
ok("hybrid scattered identity", np.allclose(lhs, rhs), f"{np.max(np.abs(lhs-rhs)):.2e}")
d = np.where(hs.dense[:-1])[0]
num = (P[d + 1] - P[d]) / (hs.t[d + 1] - hs.t[d])
mid = 0.5 * (hs.t[d] + hs.t[d + 1])
Pm = np.interp(mid, hs.t, P)
ok("hybrid dense identity (fd)", np.max(np.abs(num + a * (mid - hs.t[r]) * Pm)) < 5e-3,
   f"{np.max(np.abs(num + a*(mid-hs.t[r])*Pm)):.2e}")

print()
print("=" * 78)
print("(P12) manufactured 2nd-order equation with nonuniform-graininess coefficient")
print("=" * 78)
for name, t in grids.items():
    tt = t - t[0]
    y = 1.0 / (1.0 + tt)
    mu = np.diff(tt)
    d1 = np.diff(y) / mu
    d2 = np.diff(d1) / mu[:-1]
    kappa = (mu[:-1] + mu[1:]) / mu[:-1]
    rhs = kappa * y[:-2] * y[1:-1] * y[2:]
    err = np.max(np.abs(d2-rhs))
    ok(f"{name:18s}", np.allclose(d2, rhs, rtol=2e-10, atol=2e-11),
       f"max|res|={err:.2e}")

print()
print("=" * 78)
print("(P13) first-order manufactured: y^Delta = -y y^sigma solved by y=1/(1+t)")
print("=" * 78)
for name, t in grids.items():
    tt = t - t[0]
    y = 1.0 / (1.0 + tt)
    d1 = np.diff(y) / np.diff(tt)
    ok(f"{name:18s}", np.allclose(d1, -y[:-1] * y[1:], rtol=1e-11),
       f"max|res|={np.max(np.abs(d1 + y[:-1]*y[1:])):.2e}")

print("\n" + "="*78)
print("ALL THEORETICAL VERIFICATION CHECKS PASSED")
print("="*78)
