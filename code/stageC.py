"""Stage C: matched basis ablation, optimizer study, recurrence baseline,
irregular logistic model and joint parameter identification."""
import json
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings('ignore')
from solver import (IVP, multistart, summarise, certificate, SEED0, TOL, TRF,
                    _lstsq, BIG)
from tsrbf import phi_and_dlogphi, width_box

OUT = Path(__file__).resolve().parents[1] / 'data'
res = {}

# =====================================================================
# C1. matched basis ablation  (identical centres, m, optimiser, seeds)
# =====================================================================
tq = 2.0 ** np.arange(0, 11)
yq = 1.0 / tq
fq = lambda t, y: -y / (2 * t)
dfq = lambda t, y: -1.0 / (2 * t)

th = np.r_[1 - 1 / np.arange(8, 31, dtype=float), 1.0]
yh = 1.0 / (1.0 + th)
muh = np.diff(th)
fh = lambda t, y: -y * y / (1 + muh * y)
dfh = lambda t, y: -(y * (2 + muh * y)) / (1 + muh * y) ** 2

tl = np.array([0, 1, 2, 4, 7, 11, 16, 22, 29, 37, 46, 56], float)
r_true, K, z0 = 0.03, 1000.0, 0.08
mul = np.diff(tl)


def logistic_ref(r, t=tl, z0=z0):
    z = np.empty(t.size)
    z[0] = z0
    for k, h in enumerate(np.diff(t)):
        z[k + 1] = z[k] + h * r * z[k] * (1 - z[k])
    return z


zl = logistic_ref(r_true)
fl = lambda t, z: r_true * z * (1 - z)
dfl = lambda t, z: r_true * (1 - 2 * z)

CASES = [
    ('quantum', tq, 1.0, fq, dfq, yq, 5, np.arange(5), 1e-8, 800, 'hybrid'),
    ('accumulation', th, yh[0], fh, dfh, yh, 6,
     np.round(np.linspace(0, th.size - 1, 6)).astype(int), 1e-10, 6000, 'analytic'),
    ('logistic', tl, z0, fl, dfl, zl, 6,
     np.round(np.linspace(0, tl.size - 1, 6)).astype(int), 1e-7, 3000, 'analytic'),
]

abl = []
best_log = {}
for nm, t, y0, f, dfy, ytr, m, c, thr, nf, mode in CASES:
    variants = [
        ('ts', 'time-scale Gaussian', 'ts'),
        ('classic_same', 'classical Gaussian, same width box', 'classic'),
        ('classic_generous', 'classical Gaussian, generous width box', 'classic'),
    ]
    for tag, lab, kind in variants:
        def mk(t=t, y0=y0, f=f, dfy=dfy, m=m, c=c, kind=kind, tag=tag):
            prob = IVP(t, y0, f, dfy, m, c, kind)
            if tag == 'classic_same':
                # Strict Reviewer-2 control: only the basis formula changes.
                # Centres, neuron count, optimiser, seeds, trial solution,
                # initialisation rule, and the alpha box are identical.
                prob.lo, prob.hi = width_box(t, c)
            return prob
        df, runs, p = multistart(mk, ytr, nstarts=20, max_nfev=nf, mode=mode)
        df.to_csv(OUT / f'ablation_{nm}_{tag}_runs.csv', index=False)
        s = summarise(df, thr)
        s.update(example=nm, basis=lab, threshold=thr, m=m)
        abl.append(s)
        if nm == 'logistic' and tag == 'ts':
            best_log['ts'] = runs[int(df.max_error.argmin())]
pd.DataFrame(abl).to_csv(OUT / 'basis_ablation.csv', index=False)

bl = best_log['ts']
el = np.abs(bl['y'] - zl)
Bl = certificate(tl, bl['R'], np.full(tl.size - 1, r_true))
pd.DataFrame(dict(day=tl, reference_population=K * zl, ts_rbfnn_population=K * bl['y'],
                  abs_error_population=K * el,
                  certificate_population=K * Bl)).to_csv(OUT / 'logistic_solution.csv',
                                                         index=False)
res['logistic'] = dict(max_error_normalised=float(el.max()),
                       max_error_population=float(K * el.max()),
                       rmse_population=float(K * np.sqrt(np.mean(el ** 2))),
                       cert_population=float(K * Bl.max()), nfev=bl['nfev'])

# =====================================================================
# C2. optimiser study on an independent problem, budgets large enough to converge
# =====================================================================
# The comparison is run on the quantum benchmark itself, at the neuron count
# used in Section 6, so that it justifies the solver actually employed. With
# m = 5 the system is square (10 residuals, 10 variables), which is also the
# condition Levenberg-Marquardt requires.
to, yo = tq, yq
fo, dfo = fq, dfq
MO = 5
co = np.arange(5)

opt = []
for method in ['trf', 'dogbox', 'lm', 'lbfgsb']:
    rows = []
    for s in range(20):
        p = IVP(to, 1.0, fo, dfo, MO, co, 'ts')
        x0 = p.init_x(SEED0 + s)
        lb = np.r_[np.full(MO, -np.inf), np.log(p.lo)]
        ub = np.r_[np.full(MO, np.inf), np.log(p.hi)]
        fun = lambda x: p.residual_jac(x, False)[0]
        jfn = lambda x: p.residual_jac(x, True)[1]
        tic = time.perf_counter()
        if method in ('trf', 'dogbox'):
            sol = least_squares(fun, x0, jac=jfn, bounds=(lb, ub), method=method,
                                x_scale='jac', max_nfev=5000, **TOL)
            x, nfev, conv = sol.x, sol.nfev, sol.status > 0
        elif method == 'lm':
            sol = least_squares(fun, x0, jac=jfn, method='lm', max_nfev=5000, **TOL)
            x, nfev, conv = sol.x, sol.nfev, sol.status > 0
        else:
            obj = lambda x: 0.5 * float(fun(x) @ fun(x))
            grd = lambda x: p.residual_jac(x, True)[1].T @ fun(x)
            mm = minimize(obj, x0, jac=grd, method='L-BFGS-B',
                          bounds=list(zip(lb, ub)),
                          options=dict(maxfun=5000, maxiter=5000, ftol=1e-16,
                                       gtol=1e-14))
            x, nfev, conv = mm.x, mm.nfev, bool(mm.success)
        wall = time.perf_counter() - tic
        a = np.exp(x[MO:])
        Phi = phi_and_dlogphi(to, co, a)[0]
        y = p.trial(Phi, x[:MO])
        adm = bool(np.all(a >= p.lo * (1 - 1e-9)) and np.all(a <= p.hi * (1 + 1e-9)))
        rows.append(dict(start=s + 1, max_error=float(np.max(np.abs(y - yo))),
                         nfev=int(nfev), wall_s=wall, admissible=adm, converged=conv))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / f'optimizer_{method}_runs.csv', index=False)
    opt.append(dict(method=method, median_max_error=float(d.max_error.median()),
                    best_max_error=float(d.max_error.min()),
                    success_rate=float(np.mean((d.max_error < 1e-8) & d.admissible)),
                    admissible_rate=float(d.admissible.mean()),
                    converged_rate=float(d.converged.mean()),
                    median_nfev=float(d.nfev.median()),
                    median_wall_s=float(d.wall_s.median())))
pd.DataFrame(opt).to_csv(OUT / 'optimizer_summary.csv', index=False)

# =====================================================================
# C3. direct delta-recurrence baseline
# =====================================================================
def recur(t, y0, f, reps=2000):
    mu = np.diff(t)
    best = None
    ts = []
    for _ in range(reps):
        tic = time.perf_counter_ns()
        y = np.empty(t.size)
        y[0] = y0
        for k, h in enumerate(mu):
            y[k + 1] = y[k] + h * f(t[k], y[k])
        ts.append((time.perf_counter_ns() - tic) * 1e-9)
        best = y
    return best, float(np.median(ts))


base = []
for nm, t, y0, f, ytr in [('quantum', tq, 1.0, lambda t, y: -y / (2 * t), yq),
                          ('accumulation', th, yh[0],
                           lambda t, y: -y * y / (1 + np.diff(th)[
                               np.searchsorted(th[:-1], t)] * y), yh),
                          ('logistic', tl, z0, lambda t, z: r_true * z * (1 - z), zl)]:
    y, tm = recur(t, y0, f)
    e = np.abs(y - ytr)
    base.append(dict(example=nm, applicable='yes', max_error=float(e.max()),
                     rmse=float(np.sqrt(np.mean(e ** 2))),
                     rhs_evaluations=int(t.size - 1), median_wall_s=tm))
base.append(dict(example='hybrid time scale', applicable='no', max_error=np.nan,
                 rmse=np.nan, rhs_evaluations=np.nan, median_wall_s=np.nan))
base.append(dict(example='two-point BVP', applicable='no', max_error=np.nan,
                 rmse=np.nan, rhs_evaluations=np.nan, median_wall_s=np.nan))
pd.DataFrame(base).to_csv(OUT / 'recurrence_baseline.csv', index=False)

# =====================================================================
# C4. joint parameter identification from noisy irregular observations
# =====================================================================
rng = np.random.default_rng(SEED0)
noise = 2.0 / K                       # 2 individuals s.d. on the normalised scale
obs = zl + rng.normal(0, noise, zl.size)
m_id, c_id = 6, np.round(np.linspace(0, tl.size - 1, 6)).astype(int)
w_data = 1.0                          # data weight relative to residual weight

id_rows = []
for s in range(15):
    p = IVP(tl, z0, lambda t, z: r_true * z * (1 - z), dfl, m_id, c_id, 'ts')
    x0 = p.init_x(SEED0 + s)
    r0 = 0.01 + 0.04 * ((s % 5) / 4.0)     # spread of initial guesses for r

    def joint(u):
        v, thh, rr = u[:m_id], u[m_id:2 * m_id], u[-1]
        Phi, _ = phi_and_dlogphi(tl, c_id, np.exp(thh))
        y = p.trial(Phi, v)
        R = (y[1:] - y[:-1]) / mul - rr * y[:-1] * (1 - y[:-1])
        return np.r_[R, w_data * (y - obs)]

    lb = np.r_[np.full(m_id, -np.inf), np.log(p.lo), 1e-4]
    ub = np.r_[np.full(m_id, np.inf), np.log(p.hi), 1.0]
    sol = least_squares(joint, np.r_[x0, r0], bounds=(lb, ub), max_nfev=6000,
                        **TOL, **TRF)
    rhat = float(sol.x[-1])
    Phi, _ = phi_and_dlogphi(tl, c_id, np.exp(sol.x[m_id:2 * m_id]))
    y = p.trial(Phi, sol.x[:m_id])
    id_rows.append(dict(start=s + 1, r0=r0, r_hat=rhat,
                        rel_error_r=abs(rhat - r_true) / r_true,
                        traj_max_error_population=float(K * np.max(np.abs(y - zl)))))
idf = pd.DataFrame(id_rows)
idf.to_csv(OUT / 'inverse_identification.csv', index=False)


def recur_fit():
    """Least-squares fit of r through the exact delta recurrence (baseline)."""
    def g(u):
        return logistic_ref(u[0]) - obs
    out = []
    for s in range(15):
        r0 = 0.01 + 0.04 * ((s % 5) / 4.0)
        so = least_squares(g, [r0], bounds=([1e-4], [1.0]), **TOL)
        out.append(float(so.x[0]))
    return np.array(out)


rr = recur_fit()
res['inverse'] = dict(r_true=r_true, noise_individuals=2.0,
                      rbfnn_r_median=float(idf.r_hat.median()),
                      rbfnn_r_iqr=float(idf.r_hat.quantile(.75) - idf.r_hat.quantile(.25)),
                      rbfnn_rel_error_median=float(idf.rel_error_r.median()),
                      recurrence_r_median=float(np.median(rr)),
                      recurrence_rel_error_median=float(np.median(np.abs(rr - r_true)) / r_true))

(OUT / 'stageC.json').write_text(json.dumps(res, indent=2))
pd.set_option('display.width', 220)
print(json.dumps(res, indent=2))
print('\nablation\n', pd.DataFrame(abl)[
    ['example', 'basis', 'best_max_error', 'median_max_error', 'success_rate',
     'median_nfev']].to_string(index=False))
print('\noptimiser\n', pd.DataFrame(opt).to_string(index=False))
print('\nbaseline\n', pd.DataFrame(base).to_string(index=False))
