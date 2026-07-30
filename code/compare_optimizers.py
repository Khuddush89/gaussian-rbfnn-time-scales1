"""Which training method should be used?

Self-contained test, independent of the examples in the numerical section:

    y^Delta = -y,  y(0) = 1,  T = 0.1 Z cap [0,2],
    exact solution  y(t) = e_{-1}(t,0) = (1-h)^{t/h},

five neurons at equally spaced lattice points.  Compares
  (a) gradient descent on (v, alpha)        -- as usually written
  (b) gradient descent on (v, log alpha)    -- keeps alpha > 0
  (c) trust-region reflective
  (d) Levenberg-Marquardt
"""
import os, sys, time
import numpy as np, pandas as pd
from scipy.optimize import least_squares
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tsrbf import uniform, TrialSolution

H = 0.1
ts = uniform(0, 2, H)
exact = (1 - H) ** (ts.t / H)
CS = np.unique(np.linspace(0, ts.N - 2, 5).round().astype(int))
m = CS.size
trial = TrialSolution(ts, 1, [1.0], CS)
coll = trial.coll

def resid(v, al): return (lambda ya: ya[1] + ya[0])(trial.evaluate(v, al))[coll]
def E(v, al):
    r = resid(v, al); return 0.5 * float(r @ r)
def err(v, al): return float(np.abs(trial.evaluate(v, al)[0] - exact).max())
def grad(v, al, eps=1e-7):
    g = np.zeros(2 * m)
    for k in range(m):
        d = np.zeros(m); d[k] = eps
        g[k] = (E(v + d, al) - E(v - d, al)) / (2 * eps)
    for k in range(m):
        d = np.zeros(m); d[k] = eps * al[k]
        g[m + k] = (E(v, al + d) - E(v, al - d)) / (2 * eps * al[k])
    return g

# every method gets the SAME restart budget: 5 width decades x 2 weight seeds
STARTS = [(0.05 * np.random.default_rng(sd).standard_normal(m),
           np.full(m, 1.0 * 10.0 ** k))
          for k in (-2, -1, 0, 1, 2) for sd in (0, 1)]
v0, a0 = STARTS[4]
print(f"T = {H}Z cap [0,2]:  N={ts.N}, collocation M={coll.size}, m={m}, 2m={2*m}\n")
rows = []
for eta in [1e-1, 1e-2, 1e-4]:
    best = None; nfail = 0; t0 = time.perf_counter()
    for vs, as_ in STARTS:
        v, al = vs.copy(), as_.copy(); ok = True
        for it in range(1, 4001):
            try: g = grad(v, al)
            except ValueError: ok = False; break
            v = v - eta * g[:m]; al = al - eta * g[m:]
            if np.any(al <= 0) or not np.all(np.isfinite(al)): ok = False; break
        if not ok: nfail += 1; continue
        c = E(v, al)
        if best is None or c < best[0]: best = (c, err(v, al))
    dt = time.perf_counter() - t0
    rows.append({"method": f"gradient descent, $\\eta=10^{{{int(np.log10(eta))}}}$",
                 "evaluations": f"{10*4001*(2*m+1):.0e}",
                 "$E(p)$": best[0] if best else np.nan,
                 "$\\|e\\|_\\infty$": best[1] if best else np.nan,
                 "outcome": f"{nfail}/10 starts left the admissible set"})
    print(f"  GD eta={eta:g}: fails {nfail}/10  " +
          (f"best E={best[0]:.3e} err={best[1]:.3e}" if best else "all failed") + f" [{dt:.0f}s]")
for eta in [1e-2]:
    best = None; t0 = time.perf_counter()
    for vs, as_ in STARTS:
        v, th = vs.copy(), np.log(as_); ok = True
        for it in range(1, 4001):
            al = np.exp(th)
            try: g = grad(v, al)
            except ValueError: ok = False; break
            v = v - eta * g[:m]; th = th - eta * g[m:] * al
            if not np.all(np.isfinite(th)): ok = False; break
        if not ok: continue
        al = np.exp(th); c = E(v, al)
        if best is None or c < best[0]: best = (c, err(v, al))
    dt = time.perf_counter() - t0
    rows.append({"method": f"GD on $(v,\\log\\alpha)$, $\\eta=10^{{{int(np.log10(eta))}}}$",
                 "evaluations": f"{10*4001*(2*m+1):.0e}",
                 "$E(p)$": best[0] if best else np.nan,
                 "$\\|e\\|_\\infty$": best[1] if best else np.nan,
                 "outcome": "best of 10 starts"})
    print(f"  GDlog eta={eta:g}: " + (f"best E={best[0]:.3e} err={best[1]:.3e}" if best else "all failed") + f" [{dt:.0f}s]")
def F(p):
    v, al = p[:m], np.exp(p[m:])
    try: return resid(v, al)
    except ValueError: return np.full(coll.size, 1e6)
for meth, nm in [("trf", "trust-region reflective"), ("lm", "Levenberg--Marquardt")]:
    t0 = time.perf_counter(); best = None; tot = 0
    for vs, as_ in STARTS:
        try:
            sol = least_squares(F, np.concatenate([vs, np.log(as_)]), method=meth,
                                xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=4000)
        except Exception:
            continue
        tot += sol.nfev
        if best is None or sol.cost < best.cost: best = sol
    dt = time.perf_counter() - t0
    v, al = best.x[:m], np.exp(best.x[m:])
    lab = f"\\textbf{{{nm}}}"
    rows.append({"method": lab, "evaluations": f"{tot:d}", "$E(p)$": best.cost,
                 "$\\|e\\|_\\infty$": err(v, al),
                 "outcome": "\\textbf{converged}"})
    print(f"  {nm}: total nfev={tot} best E={best.cost:.3e} err={err(v,al):.3e} [{dt:.1f}s]")
df = pd.DataFrame(rows)
TB = os.path.join(os.path.dirname(__file__), "..", "results", "tables")
df.to_csv(os.path.join(TB, "tab4_optimizers.csv"), index=False)


def _sci(x, d=3):
    """LaTeX scientific notation; em-dash for a missing entry."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "---"
    if x == 0:
        return "$0$"
    e = int(np.floor(np.log10(abs(x))))
    mant = x / 10.0 ** e
    if e == 0:
        return f"${mant:.{d}f}$"
    return f"${mant:.{d}f}\\times10^{{{e}}}$"


lines = [r"\begin{tabular}{@{}l r r r l@{}}", r"\toprule",
         r"method & evaluations & $E(p)$ & $\|e\|_\infty$ & outcome \\",
         r"\midrule"]
for _, r in df.iterrows():
    lines.append(" & ".join([str(r["method"]), str(r["evaluations"]),
                             _sci(r["$E(p)$"]), _sci(r["$\\|e\\|_\\infty$"]),
                             str(r["outcome"])]) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(TB, "tab4_optimizers.tex"), "w").write("\n".join(lines) + "\n")
print("\nwrote tab4_optimizers")
