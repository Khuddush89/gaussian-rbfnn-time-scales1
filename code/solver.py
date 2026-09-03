from pathlib import Path
"""Residual assembly, analytic Jacobians, variable projection and multistart.

Training variables:  x = (v_1..v_m, theta_1..theta_m),  alpha_j = exp(theta_j).
"""
import time
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tsrbf import (phi_and_dlogphi, classical_and_dlog, width_box,
                   width_box_classical)

SEED0 = 20260901
RCOND = 1e-12          # truncation level for every inner linear solve
BIG = 1e12             # finite penalty returned when a trial point is unusable


def _lstsq(A, b):
    """Truncated least squares, applied identically to every basis."""
    if not (np.all(np.isfinite(A)) and np.all(np.isfinite(b))):
        return None
    v, *_ = np.linalg.lstsq(A, b, rcond=RCOND)
    return v if np.all(np.isfinite(v)) else None
TOL = dict(ftol=1e-15, xtol=1e-15, gtol=1e-15)
TRF = dict(method='trf', x_scale='jac', tr_solver='exact')


def basis_pair(kind):
    return phi_and_dlogphi if kind == 'ts' else classical_and_dlog


def box(kind, t, cidx):
    return width_box(t, cidx) if kind == 'ts' else width_box_classical(t, cidx)


class _Base:
    def sep(self):
        """Separation distance delta_j = min_{k!=j} |c_j - c_k|."""
        cc = self.t[self.cidx]
        if cc.size == 1:
            return np.array([self.t[-1] - self.t[0]])
        return np.array([np.min(np.abs(cc[j] - np.delete(cc, j)))
                         for j in range(cc.size)])

    def init_theta(self, seed):
        """alpha_j^0 = clip(eta_j / delta_j^2),  eta_j ~ 10^U[-1.5, 1]."""
        rng = np.random.default_rng(seed)
        eta = 10.0 ** rng.uniform(-1.5, 1.0, self.m)
        return np.log(np.clip(eta / self.sep() ** 2, self.lo, self.hi))

    def init_x(self, seed):
        th = self.init_theta(seed)
        try:
            A, b, _ = self.linear_AB(np.exp(th))
            v = _lstsq(A, b)
        except Exception:
            v = None
        if v is None:
            v = np.zeros(self.m)
        return np.r_[v, th]


class IVP(_Base):
    """y^Delta = f(t,y),  y(t0)=y0  on a finite right-scattered grid.

    Trial:     y_a = y0 + h1(t,t0) N(t),   N = sum_j v_j phi_j.
    Residual:  R_i = (y_a(t_{i+1}) - y_a(t_i))/mu_i - f(t_i, y_a(t_i)).
    """
    kindname = 'IVP'

    def __init__(self, t, y0, f, dfdy, m, cidx, kind='ts'):
        self.t = np.asarray(t, float)
        self.mu = np.diff(self.t)
        self.y0, self.f, self.dfdy = y0, f, dfdy
        self.m, self.cidx, self.kind = m, np.asarray(cidx, int), kind
        self.D = self.t - self.t[0]
        self.lo, self.hi = box(kind, self.t, self.cidx)
        self.bfun = basis_pair(kind)
        self.nres = self.t.size - 1

    def trial(self, Phi, v):
        return self.y0 + self.D * (Phi @ v)

    def residual_jac(self, x, want_jac=True):
        v, a = x[:self.m], np.exp(x[self.m:])
        Phi, S = self.bfun(self.t, self.cidx, a)
        y = self.trial(Phi, v)
        R = (y[1:] - y[:-1]) / self.mu - self.f(self.t[:-1], y[:-1])
        if not want_jac:
            return R, None
        dy = np.empty((self.t.size, 2 * self.m))
        dy[:, :self.m] = self.D[:, None] * Phi
        dy[:, self.m:] = self.D[:, None] * (v[None, :] * a[None, :] * Phi * S)
        fy = self.dfdy(self.t[:-1], y[:-1])[:, None]
        return R, (dy[1:] - dy[:-1]) / self.mu[:, None] - fy * dy[:-1]

    def linear_AB(self, a):
        """R = A(theta) v - b, exact when f is affine in y."""
        Phi, _ = self.bfun(self.t, self.cidx, a)
        M = self.D[:, None] * Phi
        z = np.zeros(self.nres)
        lam = self.dfdy(self.t[:-1], z)
        A = (M[1:] - M[:-1]) / self.mu[:, None] - lam[:, None] * M[:-1]
        b = self.f(self.t[:-1], z) + lam * self.y0
        return A, b, Phi


class BVP(_Base):
    """y^{DeltaDelta} = F(t, y, y^sigma, y^{sigma^2}),  y(a)=A, y(b)=B.

    Trial:  y_a = A + (B-A)(t-a)/(b-a) + (t-a)(t-b) N(t).
    """
    kindname = 'BVP'

    def __init__(self, t, A, B, F, dF, m, cidx, kind='ts'):
        self.t = np.asarray(t, float)
        self.mu = np.diff(self.t)
        self.A, self.B, self.F, self.dF = A, B, F, dF
        self.m, self.cidx, self.kind = m, np.asarray(cidx, int), kind
        a0, b0 = self.t[0], self.t[-1]
        self.lin = A + (B - A) * (self.t - a0) / (b0 - a0)
        self.D = (self.t - a0) * (self.t - b0)
        self.lo, self.hi = box(kind, self.t, self.cidx)
        self.bfun = basis_pair(kind)
        self.nres = self.t.size - 2

    def trial(self, Phi, v):
        return self.lin + self.D * (Phi @ v)

    def _d2(self, y):
        d1 = (y[1:] - y[:-1]) / self.mu
        return (d1[1:] - d1[:-1]) / self.mu[:-1]

    def residual_jac(self, x, want_jac=True):
        v, a = x[:self.m], np.exp(x[self.m:])
        Phi, S = self.bfun(self.t, self.cidx, a)
        y = self.trial(Phi, v)
        tt = self.t[:-2]
        R = self._d2(y) - self.F(tt, y[:-2], y[1:-1], y[2:])
        if not want_jac:
            return R, None
        dy = np.empty((self.t.size, 2 * self.m))
        dy[:, :self.m] = self.D[:, None] * Phi
        dy[:, self.m:] = self.D[:, None] * (v[None, :] * a[None, :] * Phi * S)
        d1 = (dy[1:] - dy[:-1]) / self.mu[:, None]
        d2 = (d1[1:] - d1[:-1]) / self.mu[:-1, None]
        g0, g1, g2 = self.dF(tt, y[:-2], y[1:-1], y[2:])
        return R, d2 - (g0[:, None] * dy[:-2] + g1[:, None] * dy[1:-1]
                        + g2[:, None] * dy[2:])

    def linear_AB(self, a):
        Phi, _ = self.bfun(self.t, self.cidx, a)
        M = self.D[:, None] * Phi
        tt, z = self.t[:-2], np.zeros(self.nres)
        g0, g1, g2 = self.dF(tt, z, z, z)
        d1 = (M[1:] - M[:-1]) / self.mu[:, None]
        d2 = (d1[1:] - d1[:-1]) / self.mu[:-1, None]
        A = d2 - (g0[:, None] * M[:-2] + g1[:, None] * M[1:-1] + g2[:, None] * M[2:])
        L = self.lin
        d1L = (L[1:] - L[:-1]) / self.mu
        b = -(d1L[1:] - d1L[:-1]) / self.mu[:-1] + self.F(tt, L[:-2], L[1:-1], L[2:])
        return A, b, Phi


# ------------------------------------------------------------------ solvers
def _pack(prob, x, nfev, wall, status):
    R = prob.residual_jac(x, False)[0]
    a = np.exp(x[prob.m:])
    Phi = prob.bfun(prob.t, prob.cidx, a)[0]
    return dict(x=x, y=prob.trial(Phi, x[:prob.m]), R=R, cost=0.5 * float(R @ R),
                nfev=int(nfev), wall=float(wall), alpha=a, status=int(status),
                converged=bool(status > 0),
                admissible=bool(np.all(a >= prob.lo * (1 - 1e-9))
                                and np.all(a <= prob.hi * (1 + 1e-9))))


def train(prob, max_nfev=600, x0=None, seed=SEED0, jac='analytic'):
    if x0 is None:
        x0 = prob.init_x(seed)
    lb = np.r_[np.full(prob.m, -np.inf), np.log(prob.lo)]
    ub = np.r_[np.full(prob.m, np.inf), np.log(prob.hi)]
    fun = lambda x: prob.residual_jac(x, False)[0]
    jfn = (lambda x: prob.residual_jac(x, True)[1]) if jac == 'analytic' else '2-point'
    tic = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sol = least_squares(fun, x0, jac=jfn, bounds=(lb, ub),
                            max_nfev=max_nfev, **TOL, **TRF)
    return _pack(prob, sol.x, sol.nfev, time.perf_counter() - tic, sol.status)


def train_varpro(prob, max_nfev=600, x0=None, seed=SEED0):
    """Golub-Pereyra variable projection: eliminate the linear weights v."""
    if x0 is None:
        x0 = prob.init_x(seed)
    th0 = x0[prob.m:]

    def red(th):
        A, b, _ = prob.linear_AB(np.exp(th))
        v = _lstsq(A, b)
        if v is None:
            return np.full(prob.nres, BIG)
        r = A @ v - b
        return r if np.all(np.isfinite(r)) else np.full(prob.nres, BIG)

    tic = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sol = least_squares(red, th0, bounds=(np.log(prob.lo), np.log(prob.hi)),
                            max_nfev=max_nfev, **TOL, **TRF)
    A, b, _ = prob.linear_AB(np.exp(sol.x))
    v = _lstsq(A, b)
    if v is None:
        v = np.zeros(prob.m)
    out = _pack(prob, np.r_[v, sol.x], sol.nfev, time.perf_counter() - tic, sol.status)
    out['cond'] = float(np.linalg.cond(A))
    return out


def train_hybrid(prob, max_nfev=600, x0=None, seed=SEED0):
    """Variable projection, then a full analytic-Jacobian polish in (v, theta)."""
    r1 = train_varpro(prob, max_nfev=max_nfev // 2, x0=x0, seed=seed)
    r2 = train(prob, max_nfev=max_nfev // 2, x0=r1['x'], jac='analytic')
    out = dict(r2 if r2['cost'] <= r1['cost'] else r1)
    out['nfev'] = r1['nfev'] + r2['nfev']
    out['wall'] = r1['wall'] + r2['wall']
    return out


_MODES = {'analytic': lambda p, **k: train(p, jac='analytic', **k),
          'fd': lambda p, **k: train(p, jac='2-point', **k),
          'varpro': train_varpro,
          'hybrid': train_hybrid}


def multistart(make_prob, ytrue, nstarts=20, max_nfev=600, mode='hybrid',
               seed0=SEED0):
    prob = make_prob()
    rows, runs = [], []
    for s in range(nstarts):
        x0 = prob.init_x(seed0 + s)
        r = _MODES[mode](prob, max_nfev=max_nfev, x0=x0)
        e = np.abs(r['y'] - ytrue)
        rows.append(dict(start=s + 1, cost=r['cost'], max_error=float(e.max()),
                         rmse=float(np.sqrt(np.mean(e * e))), nfev=r['nfev'],
                         wall_s=r['wall'], admissible=r['admissible'],
                         converged=r['converged'],
                         res_inf=float(np.max(np.abs(r['R'])))))
        runs.append(r)
    return pd.DataFrame(rows), runs, prob


def summarise(df, threshold):
    return dict(best_max_error=float(df.max_error.min()),
                median_max_error=float(df.max_error.median()),
                q1=float(df.max_error.quantile(.25)),
                q3=float(df.max_error.quantile(.75)),
                success_rate=float(np.mean((df.max_error < threshold) & df.admissible)),
                median_nfev=float(df.nfev.median()),
                median_wall_s=float(df.wall_s.median()))


def certificate(t, R, L):
    """B_{k+1} = (1 + mu_k L_k) B_k + mu_k |R_k|,  B_0 = 0."""
    t = np.asarray(t, float)
    mu = np.diff(t)
    L = np.broadcast_to(np.asarray(L, float), (len(R),))
    B = np.zeros(len(R) + 1)
    for k in range(len(R)):
        B[k + 1] = (1.0 + mu[k] * L[k]) * B[k] + mu[k] * abs(R[k])
    if B.size < t.size:
        B = np.r_[B, np.full(t.size - B.size, B[-1])]
    return B
