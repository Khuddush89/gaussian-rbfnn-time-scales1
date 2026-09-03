"""Stage B: hybrid (mixed continuous/discrete) time scale and BVPs."""
import json
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings('ignore')
from solver import BVP, multistart, summarise, SEED0
from hybrid import HybridScale, HybridIVP, multistart_hybrid

OUT = Path(__file__).resolve().parents[1] / 'data'
OUT.mkdir(exist_ok=True)
res = {}


class ClassicHybridIVP(HybridIVP):
    """Same hybrid problem with the classical Gaussian and a generous width box."""

    def __init__(self, hs, y0, f, dfdy, m, cidx):
        super().__init__(hs, y0, f, dfdy, m, cidx)
        d = np.diff(self.t)
        dmin = float(np.min(d[d > 0]))
        # The classical Gaussian carries no regressivity restriction, so it is
        # given exactly the resolution cap 1/(min spacing)^2 that the proposed
        # basis receives when its mathematical bound is infinite. This is the
        # most generous cap available to either basis and avoids both an unfair
        # restriction and a degenerate underflow regime.
        self.hi = np.full(m, 1.0 / dmin ** 2)
        self.lo = np.full(m, 1e-12 / (self.t[-1] - self.t[0]) ** 2)

    def _blocks(self, a):
        t = self.t
        d = t[:, None] - t[self.cidx][None, :]
        P = np.exp(-0.5 * a[None, :] * d ** 2)
        dP = -0.5 * d ** 2 * P
        Pp = -a[None, :] * d * P
        dPp = -d * P * (1 - 0.5 * a[None, :] * d ** 2)
        M = self.D[:, None] * P
        Q = self.D[:, None] * dP
        Md = np.empty((self.nres, self.m))
        Qd = np.empty((self.nres, self.m))
        sc, dn = self.sc, ~self.sc
        Md[sc] = (M[1:] - M[:-1])[sc] / self.mu[:-1][sc, None]
        Md[dn] = (P + self.D[:, None] * Pp)[:-1][dn]
        Qd[sc] = (Q[1:] - Q[:-1])[sc] / self.mu[:-1][sc, None]
        Qd[dn] = (dP + self.D[:, None] * dPp)[:-1][dn]
        return P, M, Md, Q, Qd


# =====================================================================
# B1. hybrid time scale   T = [0,1] u [2,3] u [4,5],   y^Delta = lambda y
# =====================================================================
hs = HybridScale([(0, 1), (2, 3), (4, 5)], per=11)
lam = -0.35
yex = hs.exp_ts(lam)
f = lambda t, y: lam * y
dfdy = lambda t, y: np.full_like(t, lam)

# localisation-admissibility trade-off along the scale
am = hs.alpha_max(np.arange(hs.t.size))
pd.DataFrame(dict(t=hs.t, right_dense=hs.dense, graininess=hs.mu,
                  alpha_max=am)).to_csv(OUT / 'hybrid_admissibility.csv', index=False)

hyb_rows = []
best_h = None
for m in [3, 4, 6, 8, 10, 12]:
    c = np.round(np.linspace(0, hs.t.size - 1, m)).astype(int)
    if len(set(c)) < m:
        continue
    for cls, lab in [(HybridIVP, 'time-scale Gaussian'),
                     (ClassicHybridIVP, 'classical Gaussian')]:
        mk = lambda cls=cls, m=m, c=c: cls(hs, 1.0, f, dfdy, m, c)
        df, runs, p = multistart_hybrid(mk, yex, nstarts=12, max_nfev=1500,
                                        mode='varpro', seed0=SEED0)
        s = summarise(df, 1e-6)
        s.update(m=m, basis=lab, residual_equations=p.nres,
                 finite=float(np.mean(np.isfinite(df.max_error))))
        hyb_rows.append(s)
        if lab.startswith('time') and m == 10:
            best_h = runs[int(np.nanargmin(df.max_error.values))]
pd.DataFrame(hyb_rows).to_csv(OUT / 'hybrid_neuron_study.csv', index=False)

eh = np.abs(best_h['y'] - yex)
pd.DataFrame(dict(t=hs.t, right_dense=hs.dense, exact=yex, ts_rbfnn=best_h['y'],
                  abs_error=eh)).to_csv(OUT / 'hybrid_solution.csv', index=False)
res['hybrid'] = dict(max_error=float(eh.max()), rmse=float(np.sqrt(np.mean(eh ** 2))),
                     sample_nodes=int(hs.t.size), residual_equations=int(hs.t.size - 1),
                     dense_residual_nodes=int(hs.dense[:-1].sum()),
                     scattered_residual_nodes=int((~hs.dense[:-1]).sum()),
                     terminal_sample=float(hs.t[-1]), nfev=best_h['nfev'],
                     alpha_max_right_end=float(am[-1]))

# =====================================================================
# B2. linear BVP   y^{DD} = lambda^2 y,  y(a)=A, y(b)=B
# =====================================================================
tb = np.array([0, .04, .10, .18, .28, .40, .52, .62, .72, .80, .87, .93, .97, 1.0])
mub = np.diff(tb)
lb_ = 1.8


def e_const(l, t):
    y = np.ones(t.size)
    mu = np.diff(t)
    for i in range(1, t.size):
        y[i] = y[i - 1] * (1 + mu[i - 1] * l)
    return y


ep, em = e_const(lb_, tb), e_const(-lb_, tb)
A0, B0 = 1.0, 0.35
c1, c2 = np.linalg.solve(np.array([[ep[0], em[0]], [ep[-1], em[-1]]]), [A0, B0])
ybvp = c1 * ep + c2 * em
res['bvp_linear_reference_residual'] = float(np.max(np.abs(
    np.diff(np.diff(ybvp) / mub) / mub[:-1] - lb_ ** 2 * ybvp[:-2])))

Fl = lambda tt, y, ys, yss: lb_ ** 2 * y
dFl = lambda tt, y, ys, yss: (np.full_like(y, lb_ ** 2), np.zeros_like(y),
                              np.zeros_like(y))

bvp_rows, best_b = [], None
for m in range(3, 13):
    c = np.round(np.linspace(0, tb.size - 1, m)).astype(int)
    if len(set(c)) < m:
        continue
    mk = lambda m=m, c=c: BVP(tb, A0, B0, Fl, dFl, m, c, 'ts')
    df, runs, p = multistart(mk, ybvp, nstarts=15, max_nfev=2000, mode='varpro')
    s = summarise(df, 1e-10)
    s.update(m=m, parameters=2 * m, residual_equations=p.nres)
    bvp_rows.append(s)
    if m == 8:
        best_b = runs[int(df.max_error.argmin())]
pd.DataFrame(bvp_rows).to_csv(OUT / 'bvp_linear_neuron_study.csv', index=False)

eb = np.abs(best_b['y'] - ybvp)
pd.DataFrame(dict(t=tb, graininess=np.r_[mub, 0.0], exact=ybvp,
                  ts_rbfnn=best_b['y'], abs_error=eb)).to_csv(
    OUT / 'bvp_linear_solution.csv', index=False)
res['bvp_linear'] = dict(max_error=float(eb.max()),
                         rmse=float(np.sqrt(np.mean(eb ** 2))),
                         res_inf=float(np.max(np.abs(best_b['R']))),
                         nfev=best_b['nfev'])

# =====================================================================
# B3. nonlinear BVP  y^{DD} = ((mu+mu^sigma)/mu) y y^sigma y^{sigma^2}
#                    exact solution  y = 1/(1+t)
# =====================================================================
kk = (mub[:-1] + mub[1:]) / mub[:-1]
ynl = 1.0 / (1.0 + tb)
res['bvp_nonlinear_reference_residual'] = float(np.max(np.abs(
    np.diff(np.diff(ynl) / mub) / mub[:-1] - kk * ynl[:-2] * ynl[1:-1] * ynl[2:])))

Fn = lambda tt, y, ys, yss: kk * y * ys * yss
dFn = lambda tt, y, ys, yss: (kk * ys * yss, kk * y * yss, kk * y * ys)

nl_rows, best_n = [], None
for m in range(3, 13):
    c = np.round(np.linspace(0, tb.size - 1, m)).astype(int)
    if len(set(c)) < m:
        continue
    mk = lambda m=m, c=c: BVP(tb, ynl[0], ynl[-1], Fn, dFn, m, c, 'ts')
    df, runs, p = multistart(mk, ynl, nstarts=15, max_nfev=3000, mode='analytic')
    s = summarise(df, 1e-10)
    s.update(m=m, parameters=2 * m, residual_equations=p.nres)
    nl_rows.append(s)
    if m == 8:
        best_n = runs[int(df.max_error.argmin())]
pd.DataFrame(nl_rows).to_csv(OUT / 'bvp_nonlinear_neuron_study.csv', index=False)

en = np.abs(best_n['y'] - ynl)
pd.DataFrame(dict(t=tb, graininess=np.r_[mub, 0.0], exact=ynl,
                  ts_rbfnn=best_n['y'], abs_error=en)).to_csv(
    OUT / 'bvp_nonlinear_solution.csv', index=False)
res['bvp_nonlinear'] = dict(max_error=float(en.max()),
                            rmse=float(np.sqrt(np.mean(en ** 2))),
                            res_inf=float(np.max(np.abs(best_n['R']))),
                            nfev=best_n['nfev'])

(OUT / 'stageB.json').write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
pd.set_option('display.width', 200)
print('\nhybrid\n', pd.DataFrame(hyb_rows)[
    ['m', 'basis', 'best_max_error', 'median_max_error', 'median_nfev']].to_string(index=False))
print('\nBVP linear\n', pd.DataFrame(bvp_rows)[
    ['m', 'residual_equations', 'best_max_error', 'median_max_error',
     'success_rate', 'median_nfev']].to_string(index=False))
print('\nBVP nonlinear\n', pd.DataFrame(nl_rows)[
    ['m', 'residual_equations', 'best_max_error', 'median_max_error',
     'success_rate', 'median_nfev']].to_string(index=False))
