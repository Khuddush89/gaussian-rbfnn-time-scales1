"""Generate every manuscript figure from the stored result files."""
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings('ignore')
from tsrbf import phi_matrix, alpha_max, gamma_const, h2

R = Path(__file__).resolve().parents[1] / 'data'
F = Path(__file__).resolve().parents[1] / 'figures'
F.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9.5,
    'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.5,
    'axes.linewidth': 0.7, 'lines.linewidth': 1.4, 'lines.markersize': 4.2,
    'figure.dpi': 120, 'savefig.bbox': 'tight', 'legend.frameon': True,
    'legend.framealpha': 0.92, 'legend.edgecolor': '0.8',
})
C = dict(ts='#1f4e79', cls='#c0392b', ref='#2b2b2b', cert='#e67e22',
         acc='#2e8b57', alt='#7d3c98', grey='#8a8a8a')


def save(fig, name):
    fig.savefig(F / f'{name}.png', dpi=600)
    fig.savefig(F / f'{name}.pdf')
    plt.close(fig)
    print('  wrote', name)


# =====================================================================
# Figure 1: the basis itself
# =====================================================================
fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.9))

# (a) shape on a uniform scale, interior centre, against the classical Gaussian
t = np.linspace(0, 1, 41)
r = 20
tf = np.linspace(0, 1, 601)
for a, ls in [(20.0, '-'), (60.0, '--'), (150.0, ':')]:
    ax[0].plot(t, phi_matrix(t, [r], [a])[:, 0], ls, color=C['ts'],
               marker='o', ms=2.6, label=rf'$\phi_{{\alpha,c}}$, $\alpha={a:g}$')
    ax[0].plot(tf, np.exp(-0.5 * a * (tf - 0.5) ** 2), ls, color=C['cls'],
               lw=0.9, alpha=0.85)
ax[0].set_xlabel(r'$t$')
ax[0].set_ylabel(r'$\phi_{\alpha,c}(t)$')
ax[0].set_title(r'(a) $\mathbb{T}=0.025\,\mathbb{Z}$, $c=0.5$')
ax[0].plot([], [], color=C['cls'], lw=0.9, label='classical Gaussian')
ax[0].legend(loc='upper left', fontsize=6.6)

# (b) envelope theorem, on a grid where the hypothesis gamma < 1 holds
ta = np.r_[1 - 1 / np.arange(8, 31, dtype=float), 1.0]
rr = 12
a = 0.5 * alpha_max(ta, [rr])[0]
g = gamma_const(ta, [rr], [a])[0]
P = phi_matrix(ta, [rr], [a])[:, 0]
H = h2(ta, ta[rr])
lo_e, hi_e = np.exp(-a * H / (1 - g)), np.exp(-a * H / (1 + g))
ax[1].fill_between(ta, lo_e, hi_e, color=C['grey'], alpha=0.22,
                   label='envelope band')
ax[1].plot(ta, lo_e, '--', color=C['grey'], lw=0.9)
ax[1].plot(ta, hi_e, ':', color=C['grey'], lw=0.9)
ax[1].plot(ta, P, 'o-', color=C['ts'], ms=3, label=r'$\phi_{\alpha,c}$')
ax[1].plot(ta, np.exp(-a * H), '-', color=C['cls'], lw=0.9,
           label=r'$e^{-\alpha h_2(t,c)}$')
ax[1].set_xlabel(r'$t$ on the accumulation grid')
ax[1].set_ylabel('value')
ax[1].set_title(rf'(b) envelope, $\gamma={g:.2f}$')
ax[1].legend(loc='lower left', fontsize=6.8)

# (c) O(mu*) convergence to the classical Gaussian
cv = pd.read_csv(R / 'gaussian_convergence.csv')
ax[2].loglog(cv.mu_star, cv.sup_error, 'o-', color=C['ts'],
             label=r'$\|\phi_{\alpha,c}-G_{\alpha,c}\|_\infty$')
ax[2].loglog(cv.mu_star, cv.bound, 's--', color=C['grey'], ms=3,
             label='theoretical bound')
ref = cv.sup_error.iloc[0] * (cv.mu_star / cv.mu_star.iloc[0])
ax[2].loglog(cv.mu_star, ref, ':', color=C['cls'], label=r'slope $1$')
ax[2].set_xlabel(r'$\mu^{*}$')
ax[2].set_ylabel('sup error')
ax[2].set_title(r'(c) convergence to $G_{\alpha,c}$')
ax[2].legend(loc='lower right', fontsize=7)
fig.tight_layout()
save(fig, 'fig1_basis')

# =====================================================================
# Figure 2: admissible width and the localisation trade-off
# =====================================================================
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
for name, t, col, mk in [(r'$2^{\mathbb{N}_0}$', 2.0 ** np.arange(0, 11), C['ts'], 'o'),
                         (r'$0.1\,\mathbb{Z}\cap[0,1]$',
                          np.round(np.arange(0, 1.001, .1), 10), C['acc'], 's'),
                         ('accumulation grid',
                          np.r_[1 - 1 / np.arange(8, 31, dtype=float), 1.0],
                          C['alt'], '^')]:
    am = alpha_max(t, np.arange(t.size))
    x = (t - t[0]) / (t[-1] - t[0])
    fin = np.isfinite(am)
    ax[0].semilogy(x[fin], am[fin], mk + '-', color=col, ms=3, label=name)
ax[0].set_xlabel(r'normalised centre position in $[a,b]_{\mathbb{T}}$')
ax[0].set_ylabel(r'$\alpha_{\max}(c)$')
ax[0].set_title('(a) admissible width limit')
ax[0].legend(fontsize=7)

hy = pd.read_csv(R / 'hybrid_admissibility.csv')
ax[1].semilogy(hy.t, hy.alpha_max.replace(np.inf, np.nan), 'o-', color=C['ts'],
               ms=3, label=r'$\alpha_{\max}(c)$')
sep = np.median(np.diff(hy.t[hy.right_dense])) * 4
ax[1].axhline(1 / sep ** 2, color=C['cls'], ls='--',
              label=r'width needed to localise at $\delta$')
for a, b in [(0, 1), (2, 3), (4, 5)]:
    ax[1].axvspan(a, b, color=C['acc'], alpha=0.08)
ax[1].set_xlabel(r'$t$')
ax[1].set_ylabel(r'$\alpha_{\max}(c)$')
ax[1].set_title(r'(b) hybrid $\mathbb{T}=[0,1]\cup[2,3]\cup[4,5]$')
ax[1].legend(fontsize=7, loc='upper right')
fig.tight_layout()
save(fig, 'fig2_admissibility')

# =====================================================================
# Figure 3: quantum benchmark
# =====================================================================
q = pd.read_csv(R / 'quantum_solution.csv')
nq = pd.read_csv(R / 'quantum_neuron_study.csv')
fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.9))
ax[0].loglog(q.t, q.exact, 'o-', color=C['ref'], label=r'exact $1/t$')
ax[0].loglog(q.t, q.ts_rbfnn, 's--', color=C['ts'], ms=3.4, label='TS-RBFNN')
ax[0].set_xscale('log', base=2)
ax[0].set_xlabel(r'$t$')
ax[0].set_ylabel(r'$y(t)$')
ax[0].set_title(r'(a) solution on $2^{\mathbb{N}_0}$')
ax[0].legend(fontsize=7)

fl = 1e-18
ax[1].loglog(q.t, np.maximum(q.abs_error, fl), 'o-', color=C['ts'],
             label=r'$|y_a-y|$')
ax[1].loglog(q.t, np.maximum(q.certificate, fl), 's--', color=C['cert'],
             label=r'certificate $B(t)$')
ax[1].axhline(np.finfo(float).eps, color=C['grey'], ls=':',
              label='machine epsilon')
ax[1].set_xscale('log', base=2)
ax[1].set_xlabel(r'$t$')
ax[1].set_ylabel('magnitude')
ax[1].set_title('(b) error and certificate')
ax[1].legend(fontsize=7, loc='upper left')

ax[2].semilogy(nq.m, np.maximum(nq.best_max_error, fl), 'o-', color=C['ts'],
               label='best of 20 starts')
ax[2].semilogy(nq.m, np.maximum(nq.median_max_error, fl), 's--', color=C['alt'],
               label='median of 20 starts')
axb = ax[2].twinx()
axb.bar(nq.m, nq.success_rate, color=C['acc'], alpha=0.18, width=0.55)
axb.set_ylabel('success rate', color=C['acc'])
axb.set_ylim(0, 1.05)
axb.grid(False)
ax[2].set_xlabel('neurons $m$')
ax[2].set_ylabel(r'$\|e\|_\infty$')
ax[2].set_title('(c) neuron count')
ax[2].legend(fontsize=7, loc='lower left')
fig.tight_layout()
save(fig, 'fig3_quantum')

# =====================================================================
# Figure 4: accumulation-point benchmark
# =====================================================================
h = pd.read_csv(R / 'harmonic_solution.csv')
hr = pd.read_csv(R / 'harmonic_refinement.csv')
fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.9))
ax[0].plot(h.t, h.exact, 'o-', color=C['ref'], ms=3, label=r'exact $1/(1+t)$')
ax[0].plot(h.t, h.ts_rbfnn, 's--', color=C['ts'], ms=3, label='TS-RBFNN')
axg = ax[0].twinx()
axg.semilogy(h.t[:-1], h.graininess[:-1], '^:', color=C['grey'], ms=3)
axg.set_ylabel(r'$\mu(t)$', color=C['grey'])
axg.grid(False)
ax[0].set_xlabel(r'$t$')
ax[0].set_ylabel(r'$y(t)$')
ax[0].set_title('(a) solution and graininess')
ax[0].legend(fontsize=7, loc='lower left')

ax[1].semilogy(h.t, np.maximum(h.abs_error, fl), 'o-', color=C['ts'],
               label=r'$|y_a-y|$')
ax[1].semilogy(h.t, np.maximum(h.certificate, fl), 's--', color=C['cert'],
               label=r'certificate $B(t)$')
ax[1].set_xlabel(r'$t$')
ax[1].set_ylabel('magnitude')
ax[1].set_title('(b) error and certificate')
ax[1].legend(fontsize=7, loc='lower right')

ax[2].loglog(hr.mu_min, hr.best_max_error, 'o-', color=C['ts'], label='best')
ax[2].loglog(hr.mu_min, hr.median_max_error, 's--', color=C['alt'], label='median')
ax[2].fill_between(hr.mu_min, hr.q1, hr.q3, color=C['alt'], alpha=0.15)
ax[2].invert_xaxis()
ax[2].set_xlabel(r'smallest graininess $\min\mu$')
ax[2].set_ylabel(r'$\|e\|_\infty$')
ax[2].set_title('(c) refinement toward the limit point')
ax[2].legend(fontsize=7)
fig.tight_layout()
save(fig, 'fig4_accumulation')

# =====================================================================
# Figure 5: hybrid time scale
# =====================================================================
hs = pd.read_csv(R / 'hybrid_solution.csv')
hn = pd.read_csv(R / 'hybrid_neuron_study.csv')
fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.9))
for a, b in [(0, 1), (2, 3), (4, 5)]:
    ax[0].axvspan(a, b, color=C['acc'], alpha=0.09)
    sel = (hs.t >= a) & (hs.t <= b)
    ax[0].plot(hs.t[sel], hs.exact[sel], '-', color=C['ref'], lw=1.6)
    ax[0].plot(hs.t[sel], hs.ts_rbfnn[sel], '--', color=C['ts'], lw=1.4)
sc = ~hs.right_dense.astype(bool)
ax[0].plot(hs.t[sc], hs.exact[sc], 'o', color=C['ref'], ms=4.4,
           label='right-scattered nodes')
ax[0].plot([], [], '-', color=C['ref'], label=r'exact $e_\lambda(t,0)$')
ax[0].plot([], [], '--', color=C['ts'], label='TS-RBFNN')
ax[0].set_xlabel(r'$t$')
ax[0].set_ylabel(r'$y(t)$')
ax[0].set_title(r'(a) $\mathbb{T}=[0,1]\cup[2,3]\cup[4,5]$')
ax[0].legend(fontsize=6.8, loc='upper right')

ax[1].semilogy(hs.t, np.maximum(hs.abs_error, 1e-16), 'o-', color=C['ts'], ms=3)
for a, b in [(0, 1), (2, 3), (4, 5)]:
    ax[1].axvspan(a, b, color=C['acc'], alpha=0.09)
ax[1].set_xlabel(r'$t$')
ax[1].set_ylabel(r'$|y_a-y|$')
ax[1].set_title('(b) pointwise error')

for lab, col, mk in [('time-scale Gaussian', C['ts'], 'o'),
                     ('classical Gaussian', C['cls'], 's')]:
    d = hn[hn.basis == lab].sort_values('m')
    ax[2].semilogy(d.m, d.best_max_error, mk + '-', color=col, label=lab + ' (best)')
    ax[2].semilogy(d.m, d.median_max_error, mk + ':', color=col, alpha=0.6, ms=3)
ax[2].set_xlabel('neurons $m$')
ax[2].set_ylabel(r'$\|e\|_\infty$')
ax[2].set_title('(c) matched basis comparison')
ax[2].legend(fontsize=6.8)
fig.tight_layout()
save(fig, 'fig5_hybrid')

# =====================================================================
# Figure 6: boundary value problems
# =====================================================================
bl = pd.read_csv(R / 'bvp_linear_solution.csv')
bn = pd.read_csv(R / 'bvp_nonlinear_solution.csv')
sl = pd.read_csv(R / 'bvp_linear_neuron_study.csv')
sn = pd.read_csv(R / 'bvp_nonlinear_neuron_study.csv')
fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.9))
ax[0].plot(bl.t, bl.exact, 'o-', color=C['ref'], ms=3.4, label='exact (linear)')
ax[0].plot(bl.t, bl.ts_rbfnn, 's--', color=C['ts'], ms=3, label='TS-RBFNN')
ax[0].plot(bn.t, bn.exact, '^-', color=C['acc'], ms=3.4, label='exact (nonlinear)')
ax[0].plot(bn.t, bn.ts_rbfnn, 'v--', color=C['alt'], ms=3, label='TS-RBFNN')
ax[0].scatter(bl.t.iloc[[0, -1]], bl.exact.iloc[[0, -1]], s=55,
              facecolors='none', edgecolors=C['cls'], zorder=5,
              label='imposed boundary data')
ax[0].set_xlabel(r'$t$')
ax[0].set_ylabel(r'$y(t)$')
ax[0].set_title('(a) two-point BVPs')
ax[0].legend(fontsize=6.3, loc='upper right')

ax[1].semilogy(sl.m, np.maximum(sl.best_max_error, fl), 'o-', color=C['ts'],
               label='linear, best')
ax[1].semilogy(sl.m, np.maximum(sl.median_max_error, fl), 'o:', color=C['ts'],
               alpha=.55, ms=3, label='linear, median')
ax[1].semilogy(sn.m, np.maximum(sn.best_max_error, fl), '^-', color=C['alt'],
               label='nonlinear, best')
ax[1].semilogy(sn.m, np.maximum(sn.median_max_error, fl), '^:', color=C['alt'],
               alpha=.55, ms=3, label='nonlinear, median')
ax[1].axvline(8, color=C['cls'], ls='--', lw=1.0)
ax[1].text(8.12, 3e-6, 'observed\nthreshold', color=C['cls'], fontsize=6.4,
           va='center')
ax[1].set_xlabel('neurons $m$')
ax[1].set_ylabel(r'$\|e\|_\infty$')
ax[1].set_title('(b) exactness threshold')
ax[1].legend(fontsize=6.3, loc='lower left')

w = 0.38
x = np.arange(len(sl))
ax[2].bar(x - w / 2, sl.success_rate, w, color=C['ts'], label='linear')
ax[2].bar(x + w / 2, sn.success_rate, w, color=C['alt'], label='nonlinear')
ax[2].set_xticks(x)
ax[2].set_xticklabels(sl.m)
ax[2].set_xlabel('neurons $m$')
ax[2].set_ylabel(r'success rate ($\|e\|_\infty<10^{-10}$)')
ax[2].set_title('(c) restart reliability')
ax[2].set_ylim(0, 1.05)
ax[2].legend(fontsize=7)
fig.tight_layout()
save(fig, 'fig6_bvp')

# =====================================================================
# Figure 7: basis ablation and optimiser study
# =====================================================================
ab = pd.read_csv(R / 'basis_ablation.csv')
op = pd.read_csv(R / 'optimizer_summary.csv')
fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.9))
ex = ['quantum', 'accumulation', 'logistic']
x = np.arange(len(ex))
for k, (lab, col) in enumerate([('time-scale Gaussian', C['ts']),
                                ('classical Gaussian, same width box', C['cls'])]):
    d = ab[ab.basis == lab].set_index('example').loc[ex]
    ax[0].bar(x + (k - .5) * .38, np.maximum(d.median_max_error, fl), .38,
              color=col, label=lab)
ax[0].set_yscale('log')
ax[0].set_xticks(x)
ax[0].set_xticklabels(ex, fontsize=7)
ax[0].set_ylabel(r'median $\|e\|_\infty$ (20 starts)')
ax[0].set_title('(a) matched ablation, median')
ax[0].legend(fontsize=6.6)

for k, (lab, col) in enumerate([('time-scale Gaussian', C['ts']),
                                ('classical Gaussian, same width box', C['cls'])]):
    d = ab[ab.basis == lab].set_index('example').loc[ex]
    ax[1].bar(x + (k - .5) * .38, d.success_rate, .38, color=col, label=lab)
ax[1].set_xticks(x)
ax[1].set_xticklabels(ex, fontsize=7)
ax[1].set_ylabel('restart success rate')
ax[1].set_ylim(0, 1.05)
ax[1].set_title('(b) matched ablation, reliability')
ax[1].legend(fontsize=6.6)

names = {'trf': 'TRF', 'dogbox': 'dogbox', 'lm': 'LM', 'lbfgsb': 'L-BFGS-B'}
op['label'] = op.method.map(names)
xx = np.arange(len(op))
ax[2].bar(xx - .2, op.success_rate, .4, color=C['ts'], label='success rate')
ax[2].bar(xx + .2, op.admissible_rate, .4, color=C['cert'], label='admissible rate')
ax[2].set_xticks(xx)
ax[2].set_xticklabels(op.label, fontsize=7)
ax[2].set_ylim(0, 1.08)
ax[2].set_ylabel('fraction of 20 starts')
ax[2].set_title('(c) optimiser comparison')
ax[2].legend(fontsize=6.6, loc='lower left')
fig.tight_layout()
save(fig, 'fig7_ablation_optimizer')

# =====================================================================
# Figure 8: logistic surrogate and parameter identification
# =====================================================================
lg = pd.read_csv(R / 'logistic_solution.csv')
iv = pd.read_csv(R / 'inverse_identification.csv')
fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.9))
ax[0].plot(lg.day, lg.reference_population, 'o-', color=C['ref'], ms=3.6,
           label=r'exact $\Delta$-recurrence')
ax[0].plot(lg.day, lg.ts_rbfnn_population, 's--', color=C['ts'], ms=3,
           label='TS-RBFNN surrogate')
ax[0].set_xlabel('monitoring day')
ax[0].set_ylabel('population')
ax[0].set_title('(a) irregular logistic model')
ax[0].legend(fontsize=7)

ax[1].semilogy(lg.day, np.maximum(lg.abs_error_population, 1e-16), 'o-',
               color=C['ts'], label='absolute error')
ax[1].semilogy(lg.day, np.maximum(lg.certificate_population, 1e-16), 's--',
               color=C['cert'], label='certificate')
ax[1].set_xlabel('monitoring day')
ax[1].set_ylabel('individuals')
ax[1].set_title('(b) error and certificate')
ax[1].legend(fontsize=7)

cp = pd.read_csv(R / 'inverse_comparison.csv')
ax[2].plot(iv.r0, iv.r_hat, 'o', color=C['ts'], ms=5,
           label=r'TS-RBFNN $\hat r$ (15 starts)')
ax[2].axhline(cp.r_recurrence.iloc[0] * 0 + 0.0298234, color=C['acc'], ls='-.',
              lw=1.2, label=r'$\Delta$-recurrence fit')
ax[2].axhline(0.03, color=C['cls'], ls='--', label=r'true $r=0.03$')
ax[2].set_xlabel(r'initial guess $r^{(0)}$')
ax[2].set_ylabel(r'$\hat r$')
ax[2].set_ylim(0.02975, 0.03005)
ax[2].set_title('(c) joint parameter identification')
ax[2].legend(fontsize=6.5, loc='lower right')
fig.tight_layout()
save(fig, 'fig8_logistic_inverse')

print('all figures written to', F)
