"""Stage A: basis properties, quantum grid, accumulation-point grid."""
import json
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings('ignore')
from tsrbf import phi_matrix, alpha_max, gamma_const, h2, classical_matrix
from solver import IVP, multistart, summarise, certificate, SEED0

OUT = Path(__file__).resolve().parents[1] / 'data'
OUT.mkdir(exist_ok=True)
res = {}

# =====================================================================
# A1. basis properties
# =====================================================================
tq = 2.0 ** np.arange(0, 11)

# admissible width limits on the quantum grid
rows = []
for k in range(0, 6):
    c = 2.0 ** k
    am = alpha_max(tq, [k])[0]
    rows.append(dict(center=c, alpha_max=am,
                     computational_cap=(1.0 if not np.isfinite(am) else 0.95 * am)))
pd.DataFrame(rows).to_csv(OUT / 'width_limits.csv', index=False)

# envelope theorem, sampled
env = []
for name, t in [('quantum', tq), ('uniform h=0.05', np.arange(0, 1.0001, 0.05)),
                ('accumulation', np.r_[1 - 1 / np.arange(8, 31, dtype=float), 1.0])]:
    for r in [0, len(t) // 2, len(t) - 1]:
        am = alpha_max(t, [r])[0]
        a = 0.5 * am if np.isfinite(am) else 5.0
        g = gamma_const(t, [r], [a])[0]
        if g >= 1:
            continue
        P = phi_matrix(t, [r], [a])[:, 0]
        H = h2(t, t[r])
        env.append(dict(grid=name, center=t[r], alpha=a, gamma=g,
                        lower_ok=bool(np.all(P >= np.exp(-a * H / (1 - g)) - 1e-13)),
                        upper_ok=bool(np.all(P <= np.exp(-a * H / (1 + g)) + 1e-13)),
                        max_rel_gap=float(np.max(np.exp(-a*H/(1+g)) - np.exp(-a*H/(1-g))))))
pd.DataFrame(env).to_csv(OUT / 'envelope_check.csv', index=False)

# O(mu*) convergence to the classical Gaussian
conv = []
alpha, c = 2.0, 0.5
for N in [10, 20, 40, 80, 160, 320, 640, 1280]:
    t = np.linspace(0, 1, N + 1)
    r = int(np.argmin(np.abs(t - c)))
    P = phi_matrix(t, [r], [alpha])[:, 0]
    G = np.exp(-0.5 * alpha * (t - t[r]) ** 2)
    mus = float(np.max(np.diff(t)))
    g = gamma_const(t, [r], [alpha])[0]
    conv.append(dict(N=N, mu_star=mus, sup_error=float(np.max(np.abs(P - G))),
                     bound=float(g / (1 - g) + alpha * mus * 1.0)))
cv = pd.DataFrame(conv)
cv['ratio'] = cv.sup_error.shift(1) / cv.sup_error
cv.to_csv(OUT / 'gaussian_convergence.csv', index=False)
res['gaussian_convergence_last_ratio'] = float(cv.ratio.iloc[-1])

# =====================================================================
# A2. quantum time scale:  y^Delta = -y/(2t),  y(1)=1,  exact 1/t
# =====================================================================
yq = 1.0 / tq
fq = lambda t, y: -y / (2 * t)
dfq = lambda t, y: -1.0 / (2 * t)
Lq = 1.0 / (2 * tq[:-1])

neuron = []
best_by_m = {}
for m in range(2, 8):
    mk = lambda m=m: IVP(tq, 1.0, fq, dfq, m, np.arange(m), 'ts')
    df, runs, p = multistart(mk, yq, nstarts=20, max_nfev=800, mode='hybrid')
    df.to_csv(OUT / f'quantum_m{m}_runs.csv', index=False)
    s = summarise(df, 1e-8)
    s.update(m=m, parameters=2 * m, residual_equations=p.nres)
    neuron.append(s)
    best_by_m[m] = runs[int(df.max_error.argmin())]
pd.DataFrame(neuron).to_csv(OUT / 'quantum_neuron_study.csv', index=False)

# training-mode comparison at m = 5
modes = []
for mode in ['fd', 'analytic', 'varpro', 'hybrid']:
    mk = lambda: IVP(tq, 1.0, fq, dfq, 5, np.arange(5), 'ts')
    df, _, _ = multistart(mk, yq, nstarts=20, max_nfev=800, mode=mode)
    s = summarise(df, 1e-8)
    s['mode'] = mode
    modes.append(s)
pd.DataFrame(modes).to_csv(OUT / 'quantum_training_modes.csv', index=False)

q5 = best_by_m[5]
eq = np.abs(q5['y'] - yq)
Bq = certificate(tq, q5['R'], Lq)
pd.DataFrame(dict(t=tq, exact=yq, ts_rbfnn=q5['y'], abs_error=eq,
                  certificate=Bq)).to_csv(OUT / 'quantum_solution.csv', index=False)
res['quantum'] = dict(max_error=float(eq.max()), rmse=float(np.sqrt(np.mean(eq ** 2))),
                      cert_max=float(Bq.max()), res_inf=float(np.max(np.abs(q5['R']))),
                      nfev=q5['nfev'], alphas=q5['alpha'].tolist())

# centre-placement ablation
cs = {'early geometric': np.arange(5), 'index quantiles': np.array([0, 2, 5, 8, 10]),
      'intermediate': np.array([0, 1, 3, 6, 10]), 'late geometric': np.array([6, 7, 8, 9, 10])}
crows = []
for nm, c in cs.items():
    mk = lambda c=c: IVP(tq, 1.0, fq, dfq, 5, c, 'ts')
    df, _, _ = multistart(mk, yq, nstarts=10, max_nfev=800, mode='hybrid')
    s = summarise(df, 1e-8)
    s.update(strategy=nm, centres=';'.join(f'{tq[i]:g}' for i in c),
             alpha_max_min=float(np.min(np.where(np.isfinite(alpha_max(tq, c)),
                                                 alpha_max(tq, c), 1e9))))
    crows.append(s)
pd.DataFrame(crows).to_csv(OUT / 'quantum_centre_ablation.csv', index=False)

# =====================================================================
# A3. accumulation-point grid:  y^Delta = -y y^sigma,  exact 1/(1+t)
# =====================================================================
def harmonic_grid(nmax):
    return np.r_[1 - 1 / np.arange(8, nmax + 1, dtype=float), 1.0]

th = harmonic_grid(30)
yh = 1.0 / (1.0 + th)
muh = np.diff(th)
fh = lambda t, y: -y * y / (1 + muh * y)
dfh = lambda t, y: -(y * (2 + muh * y)) / (1 + muh * y) ** 2

hs_rows, hbest = [], None
for m in [4, 6, 8, 10, 12]:
    ch = np.round(np.linspace(0, th.size - 1, m)).astype(int)
    mk = lambda m=m, ch=ch: IVP(th, yh[0], fh, dfh, m, ch, 'ts')
    df, runs, p = multistart(mk, yh, nstarts=15, max_nfev=6000, mode='analytic')
    df.to_csv(OUT / f'harmonic_m{m}_runs.csv', index=False)
    s = summarise(df, 1e-10)
    s.update(m=m, residual_equations=p.nres)
    hs_rows.append(s)
    if m == 6:
        hbest = runs[int(df.max_error.argmin())]
pd.DataFrame(hs_rows).to_csv(OUT / 'harmonic_neuron_study.csv', index=False)

eh = np.abs(hbest['y'] - yh)
Bh = certificate(th, hbest['R'], np.full(th.size - 1, 2.0))
pd.DataFrame(dict(t=th, graininess=np.r_[muh, 0.0], exact=yh, ts_rbfnn=hbest['y'],
                  abs_error=eh, certificate=Bh)).to_csv(OUT / 'harmonic_solution.csv',
                                                        index=False)
res['harmonic'] = dict(max_error=float(eh.max()), rmse=float(np.sqrt(np.mean(eh ** 2))),
                       cert_max=float(Bh.max()), nfev=hbest['nfev'])

# refinement toward the accumulation point
ref = []
for nmax in [30, 60, 120, 240, 480]:
    t = harmonic_grid(nmax)
    y = 1.0 / (1.0 + t)
    mu = np.diff(t)
    f = lambda tt, yy, mu=mu: -yy * yy / (1 + mu * yy)
    dd = lambda tt, yy, mu=mu: -(yy * (2 + mu * yy)) / (1 + mu * yy) ** 2
    m = 6
    ch = np.round(np.linspace(0, t.size - 1, m)).astype(int)
    mk = lambda t=t, f=f, dd=dd, ch=ch, y=y: IVP(t, y[0], f, dd, m, ch, 'ts')
    df, runs, p = multistart(mk, y, nstarts=10, max_nfev=6000, mode='analytic')
    s = summarise(df, 1e-8)
    s.update(n_max=nmax, nodes=t.size, mu_min=float(mu.min()),
             residual_equations=p.nres)
    ref.append(s)
pd.DataFrame(ref).to_csv(OUT / 'harmonic_refinement.csv', index=False)

(OUT / 'stageA.json').write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
print('\nquantum neuron study\n', pd.DataFrame(neuron)[
    ['m', 'best_max_error', 'median_max_error', 'success_rate', 'median_nfev']])
print('\ntraining modes\n', pd.DataFrame(modes)[
    ['mode', 'best_max_error', 'median_max_error', 'success_rate', 'median_nfev',
     'median_wall_s']])
print('\nharmonic refinement\n', pd.DataFrame(ref)[
    ['n_max', 'nodes', 'mu_min', 'best_max_error', 'median_max_error', 'success_rate']])
