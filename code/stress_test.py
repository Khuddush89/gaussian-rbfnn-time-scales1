"""
Stress test: every claim of the theory checked against the implementation.

Run:  python code/stress_test.py

Sections
  A  the activation function and its delta derivative
  B  the closed forms of Section 2
  C  admissibility (Proposition: admissible widths)
  D  the width gradient
  E  trial-solution delta derivatives, orders 1, 2 and n
  F  each example's exact solution really solves its dynamic equation
  G  the a-posteriori certificate
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsrbf import (TimeScale, uniform, quantum, real_grid, hybrid, phi_ts,
                   dphi_dalpha, TrialSolution, train, certified_bound,
                   alpha_admissible_max)

FAIL = []


def check(name, val, tol):
    bad = not (abs(val) <= tol)
    if bad:
        FAIL.append(name)
    print(f"  [{'FAIL' if bad else 'PASS'}]  {name:56s} {val:.3e}  (tol {tol:.0e})")


def q_of(ts, c_idx, alpha):
    """q = ominus p,  p(tau) = alpha (tau - c)."""
    p = alpha * (ts.t - ts.t[c_idx])
    return -p / (1.0 + ts.mu * p)


def banner(s):
    print("\n" + "=" * 74 + f"\n{s}\n" + "=" * 74)


# ----------------------------------------------------------------- A
banner("A.  activation function and its delta derivative")
for ts, nm, al in [(quantum(2., 0, 7), "2^N0", 2e-3),
                   (uniform(0, 3, 0.25), "0.25Z", 0.7),
                   (uniform(0, 2, 0.1), "0.1Z", 1.3),
                   (hybrid([("interval", 2 * k, 2 * k + 1) for k in range(3)],
                           n_dense=15), "hybrid", 0.9)]:
    c = 0                                    # left endpoint: always admissible
    ph = phi_ts(ts, [c], [al])[:, 0]
    q = q_of(ts, c, al)
    # phi^Delta = q phi   (the identity used throughout Section 4)
    lhs = ts.delta(ph)[:-1]
    rhs = (q * ph)[:-1]
    check(f"{nm}: phi^Delta - (ominus p) phi", np.abs(lhs - rhs).max(), 1e-11)
    check(f"{nm}: 0 < phi <= 1", max(0.0, ph.max() - 1.0), 1e-14)
    check(f"{nm}: phi(c) = 1", abs(ph[c] - 1.0), 1e-14)
    # monotone decay away from the centre
    check(f"{nm}: phi nonincreasing for t >= c",
          max(0.0, np.diff(ph).max()), 1e-14)

# ----------------------------------------------------------------- B
banner("B.  closed forms of Section 2")
tsR = real_grid(-3, 3, 20001)
cR = 10000
check("T=R: phi - exp(-a(t-c)^2/2)",
      np.abs(phi_ts(tsR, [cR], [1.0])[:, 0] - np.exp(-tsR.t ** 2 / 2)).max(), 5e-4)
sig = 0.8
check("T=R, a=1/sigma^2: classical Gaussian",
      np.abs(phi_ts(tsR, [cR], [1 / sig ** 2])[:, 0]
             - np.exp(-tsR.t ** 2 / (2 * sig ** 2))).max(), 5e-4)
# Z  and  hZ :  phi = prod_{k=0}^{n-1} (1 + a h^2 k)^{-1}
for h in [1.0, 0.5, 0.25, 0.1]:
    ts = uniform(0, 4, h)
    al = 1.3
    ph = phi_ts(ts, [0], [al])[:, 0]
    worst = 0.0
    for n in range(ts.N):
        closed = np.prod([1.0 / (1.0 + al * h * h * k) for k in range(n)])
        worst = max(worst, abs(ph[n] - closed))
    check(f"hZ h={h}: phi - prod (1 + a h^2 k)^-1", worst, 1e-13)
# quantum:  phi(2^n c, c) = prod_{k=0}^{n-1} (1 + a c^2 2^k (2^k - 1))^{-1}
tsq = quantum(2., 0, 7)
al = 1e-3
for mIdx in [0, 1, 2]:
    c = tsq.t[mIdx]
    ph = phi_ts(tsq, [mIdx], [al])[:, 0]
    worst = 0.0
    for n in range(tsq.N - mIdx):
        closed = np.prod([1.0 / (1.0 + al * c * c * (2.0 ** k) * (2.0 ** k - 1))
                          for k in range(n)])
        worst = max(worst, abs(ph[mIdx + n] - closed))
    check(f"2^N0 c=2^{mIdx}: phi - prod (1+a c^2 2^k(2^k-1))^-1", worst, 1e-12)

# ----------------------------------------------------------------- C
banner("C.  admissibility:  alpha_max(c) = 1/W(c),  W(c)=sup_{tau<c} mu(tau)(c-tau)")
worst = 0.0
for i in range(1, 7):
    worst = max(worst, abs(alpha_admissible_max(tsq, i) - 4.0 / (tsq.t[i] ** 2)))
check("2^N0: alpha_max - 4/((q-1)c^2)", worst, 1e-12)
for h in [1.0, 0.5, 0.25]:
    ts = uniform(0, 3, h)
    k = ts.N - 1
    check(f"hZ h={h}: alpha_max(t_max) - 1/(h(t_max-t_0))",
          abs(alpha_admissible_max(ts, k) - 1.0 / (h * (ts.t[k] - ts.t[0]))), 1e-12)
# just above alpha_max the kernel must be rejected
ts = uniform(0, 3, 0.5)
amax = alpha_admissible_max(ts, ts.N - 1)
try:
    phi_ts(ts, [ts.N - 1], [amax * 1.001])
    check("hZ: alpha > alpha_max is rejected", 1.0, 0.0)
except ValueError:
    check("hZ: alpha > alpha_max is rejected", 0.0, 0.0)

# ----------------------------------------------------------------- D
banner("D.  width gradient  d phi/d alpha")
for ts, nm, al, c in [(quantum(2., 0, 6), "2^N0", 2e-3, 0),
                      (uniform(0, 3, 0.25), "0.25Z", 0.7, 0),
                      (uniform(0, 2, 0.1), "0.1Z", 1.3, 5)]:
    eps = al * 1e-6
    an = dphi_dalpha(ts, [c], [al])[:, 0]
    fd = (phi_ts(ts, [c], [al + eps])[:, 0]
          - phi_ts(ts, [c], [al - eps])[:, 0]) / (2 * eps)
    scale = max(1.0, np.abs(an).max())
    check(f"{nm}: analytic vs central difference", np.abs(an - fd).max() / scale, 1e-6)
# closed form on R:  -(t-c)^2/2 * phi
an = dphi_dalpha(tsR, [cR], [1.0])[:, 0]
check("T=R: d phi/d alpha + (t-c)^2/2 phi",
      np.abs(an + 0.5 * tsR.t ** 2 * phi_ts(tsR, [cR], [1.0])[:, 0]).max(), 5e-4)

# ----------------------------------------------------------------- E
banner("E.  trial-solution delta derivatives")


def h_apply(ts, arr, word):
    """Apply a string of sigma/Delta to a sampled function, left to right."""
    out = arr.copy()
    for ch in word:
        out = ts.delta(out) if ch == "D" else ts.shift(out)
    return out


for ts, nm in [(quantum(2., 0, 7), "2^N0"), (uniform(0, 3, 0.25), "0.25Z")]:
    rng = np.random.default_rng(1)
    N = rng.standard_normal(ts.N)
    h1 = ts.h(1, 0)
    h2 = ts.h(2, 0)
    h3 = ts.h(3, 0)
    mu, muS = ts.mu, ts.shift(ts.mu)
    muD = ts.delta(ts.mu)
    h1S, h2S = ts.shift(h1), ts.shift(h2)
    h2SS = ts.shift(h2S)

    # order 1:  y_a = y0 + h1 N
    ya = 3.0 + h1 * N
    check(f"{nm} n=1: y_a^Delta = N + h1^sigma N^Delta",
          np.abs(ts.delta(ya) - (N + h1S * ts.delta(N)))[:-1].max(), 1e-10)

    # order 2:  y_a = y0 + h1 y1 + h2 N
    ya2 = 3.0 + h1 * 2.0 + h2 * N
    check(f"{nm} n=2: y_a^Delta = y1 + h1 N + h2^sigma N^Delta",
          np.abs(ts.delta(ya2) - (2.0 + h1 * N + h2S * ts.delta(N)))[:-1].max(), 1e-10)
    # the ERRONEOUS version, with h1^sigma in place of h2^sigma
    wrong = np.abs(ts.delta(ya2) - (2.0 + h1 * N + h1S * ts.delta(N)))[:-1].max()
    print(f"         (with h1^sigma instead: error {wrong:.3e} -- must be large)")
    # second delta derivative
    coef1 = h1S + (1.0 + muD) * h1 + muS
    pred = N + coef1 * ts.delta(N) + h2SS * ts.delta(ts.delta(N))
    check(f"{nm} n=2: y_a^DD = N + [h1^s+(1+mu^D)h1+mu^s] N^D + h2^ss N^DD",
          np.abs(ts.delta(ts.delta(ya2)) - pred)[:-2].max(), 1e-9)

    # order n=3, general Leibniz formula with S_l^{(k)}
    for k in [1, 2, 3]:
        acc = np.zeros(ts.N)
        for l in range(k + 1):
            words = [w for w in itertools.product("DS", repeat=k)
                     if w.count("S") == l]
            coef = np.zeros(ts.N)
            for w in words:
                coef = coef + h_apply(ts, h3, "".join(w))
            acc = acc + coef * h_apply(ts, N, "D" * l)
        direct = h_apply(ts, h3 * N, "D" * k)
        check(f"{nm} n=3, k={k}: Leibniz sum_l (sum_S_l h_n^Lambda) N^(Delta^l)",
              np.abs(acc - direct)[:-k - 1].max(), 1e-8)

# ----------------------------------------------------------------- F
banner("F.  does each example's exact solution solve its dynamic equation?")
tsq = quantum(2., 0, 8)
y = 1.0 / tsq.t
check("Ex1  y=1/t  on 2^N0 :  y^Delta + y/(2t)",
      np.abs(tsq.delta(y) + y / (2 * tsq.t))[:-1].max(), 1e-14)

h = 0.1
ts2 = uniform(0, 2, h)
y = (1 - h) ** (ts2.t / h)
check("Ex2  y=(1-h)^{t/h}  on 0.1Z :  y^Delta + y",
      np.abs(ts2.delta(y) + y)[:-1].max(), 1e-14)

hh = 0.2
ts3 = uniform(0, 2, hh)
y = (1 + hh) ** (ts3.t / hh)
check("Ex3  y=(1+h)^{t/h}  on 0.2Z :  y^DD - 3y^D + 2y",
      np.abs(ts3.delta(ts3.delta(y)) - 3 * ts3.delta(y) + 2 * y)[:-2].max(), 1e-13)
check("Ex3  initial conditions y(0)=1, y^Delta(0)=1",
      abs(y[0] - 1.0) + abs(ts3.delta(y)[0] - 1.0), 1e-13)

ts4 = hybrid([("interval", 2 * k, 2 * k + 1) for k in range(3)], n_dense=21)
y = 1.0 / (ts4.t + 1.0)
check("Ex4  y=1/(t+1)  on hybrid :  y^Delta + y y^sigma",
      np.abs(ts4.delta(y) + y * ts4.shift(y))[:-1].max(), 1e-14)
check("Ex4  explicit form  y^Delta + y^2/(1+mu y)",
      np.abs(ts4.delta(y) + y ** 2 / (1 + ts4.mu * y))[:-1].max(), 1e-14)

# ----------------------------------------------------------------- G
banner("G.  a-posteriori certificate  e(t) <= B(t)")
cases = [
    ("Ex1", tsq, 1, [1.0], np.array([0, 1, 2, 3, 4]),
     lambda t, ya: ya[1] + ya[0] / (2 * t.t), 1.0 / tsq.t, 1.0 / tsq.t, 1e-3,
     (1e-3, 1e-2, 1e-1, 1.0)),
    ("Ex2", ts2, 1, [1.0], np.linspace(0, ts2.N - 2, 5).round().astype(int),
     lambda t, ya: ya[1] + ya[0], np.full(ts2.N, 1.0), (1 - h) ** (ts2.t / h), 1.0,
     (1e-2, 1e-1, 1.0, 1e1)),
]
for nm, ts, order, iv, cs, rf, L, exact, a0, dec in cases:
    tr = TrialSolution(ts, order, iv, np.unique(cs))
    o = train(tr, rf, a0, decades=dec)
    ya = tr.evaluate(o["v"], o["alpha"])
    e = np.abs(ya[0] - exact)
    B = certified_bound(ts, rf(ts, ya), L)
    check(f"{nm}: max(e - B)  (must be <= 0)", max(0.0, (e - B).max()), 1e-12)
    print(f"         ||e||={e.max():.4e}  ||B||={B.max():.4e}  "
          f"theta={B.max()/max(e.max(),1e-300):.2f}  E={o['E']:.3e}")

print("\n" + "=" * 74)
if FAIL:
    print(f"{len(FAIL)} FAILURE(S):")
    for f in FAIL:
        print("   -", f)
else:
    print("ALL CONSISTENCY CHECKS PASSED")
print("=" * 74)
sys.exit(1 if FAIL else 0)
