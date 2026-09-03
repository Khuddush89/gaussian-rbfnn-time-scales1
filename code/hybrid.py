from pathlib import Path
"""IVP on a hybrid time scale T = union_k [a_k, b_k].

At right-dense points the residual uses the analytic derivative
    phi'(t) = -alpha (t-c) phi(t),
at right-scattered points the exact delta quotient. Both are instances of
    phi^Delta(t) = -alpha (t-c) phi^sigma(t).
"""
import time
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from solver import TOL, TRF, _pack, summarise, _lstsq, BIG


class HybridScale:
    """T = union_k [a_k,b_k] sampled with `per` nodes per block."""

    def __init__(self, blocks, per=9):
        t, dense = [], []
        for a, b in blocks:
            xs = np.linspace(a, b, per)
            t.append(xs)
            d = np.ones(per, bool)
            d[-1] = False
            dense.append(d)
        self.t = np.concatenate(t)
        self.dense = np.concatenate(dense)
        self.dense[-1] = False
        self.mu = np.r_[np.diff(self.t), 0.0]
        self.mu[self.dense] = 0.0
        self.blocks, self.per = blocks, per

    def phi(self, cidx, alpha):
        """phi, dphi/dalpha, phi', d(phi')/dalpha at every node."""
        t, dense = self.t, self.dense
        n = t.size
        cidx = np.atleast_1d(np.asarray(cidx, int))
        alpha = np.atleast_1d(np.asarray(alpha, float))
        step = np.zeros((n - 1, cidx.size))     # d log phi contribution per edge
        dstep = np.zeros((n - 1, cidx.size))    # its derivative wrt alpha
        for j, (r, a) in enumerate(zip(cidx, alpha)):
            c = t[r]
            for i in range(n - 1):
                if dense[i]:
                    s = -0.5 * ((t[i + 1] - c) ** 2 - (t[i] - c) ** 2)
                    step[i, j], dstep[i, j] = a * s, s
                else:
                    m = t[i + 1] - t[i]
                    g = 1.0 + m * a * (t[i] - c)
                    step[i, j] = -np.log(g)
                    dstep[i, j] = -m * (t[i] - c) / g
        L = np.vstack([np.zeros((1, cidx.size)), np.cumsum(step, 0)])
        T = np.vstack([np.zeros((1, cidx.size)), np.cumsum(dstep, 0)])
        idx = (cidx, np.arange(cidx.size))
        P = np.exp(L - L[idx][None, :])
        S = T - T[idx][None, :]                 # d log phi / d alpha
        d = t[:, None] - t[cidx][None, :]
        Pp = -alpha[None, :] * d * P            # phi'  (valid at dense nodes)
        Spp = -d * P * (1.0 + alpha[None, :] * S)   # d phi' / d alpha
        return P, P * S, Pp, Spp

    def alpha_max(self, cidx):
        t, mu = self.t, self.mu
        out = []
        for r in np.atleast_1d(np.asarray(cidx, int)):
            w = mu[:r] * (t[r] - t[:r]) if r > 0 else np.array([0.0])
            W = float(np.max(w)) if w.size else 0.0
            out.append(1.0 / W if W > 0 else np.inf)
        return np.asarray(out, float)

    def exp_ts(self, lam):
        y = np.ones(self.t.size)
        for i in range(1, self.t.size):
            y[i] = (y[i - 1] * np.exp(lam * (self.t[i] - self.t[i - 1]))
                    if self.dense[i - 1] else y[i - 1] * (1 + self.mu[i - 1] * lam))
        return y


class HybridIVP:
    """y^Delta = f(t,y), y(t0)=y0 on a hybrid time scale."""
    kindname = 'hybrid IVP'

    def __init__(self, hs, y0, f, dfdy, m, cidx, safety=0.95):
        self.hs, self.y0, self.f, self.dfdy = hs, y0, f, dfdy
        self.m, self.cidx = m, np.asarray(cidx, int)
        self.t, self.mu, self.dense = hs.t, hs.mu, hs.dense
        self.D = self.t - self.t[0]
        am = hs.alpha_max(self.cidx)
        d = np.diff(self.t)
        cap = 1.0 / float(np.min(d[d > 0])) ** 2
        self.hi = np.minimum(np.where(np.isfinite(am), safety * am, cap), cap)
        self.lo = np.maximum(1e-12 * self.hi, 1e-16)
        self.sc = ~self.dense[:-1]              # right-scattered residual nodes
        self.nres = self.t.size - 1

    def sep(self):
        cc = self.t[self.cidx]
        return np.array([np.min(np.abs(cc[j] - np.delete(cc, j)))
                         for j in range(cc.size)])

    def init_x(self, seed):
        rng = np.random.default_rng(seed)
        eta = 10.0 ** rng.uniform(-1.5, 1.0, self.m)
        th = np.log(np.clip(eta / self.sep() ** 2, self.lo, self.hi))
        A, b, _ = self.linear_AB(np.exp(th))
        v = _lstsq(A, b)
        return np.r_[np.zeros(self.m) if v is None else v, th]

    def trial(self, P, v):
        return self.y0 + self.D * (P @ v)

    def _blocks(self, a):
        P, dP, Pp, dPp = self.hs.phi(self.cidx, a)
        M = self.D[:, None] * P                            # d y_a / d v
        # d y_a^Delta / d v   (row per residual node)
        Md = np.empty((self.nres, self.m))
        Md[self.sc] = (M[1:] - M[:-1])[self.sc] / self.mu[:-1][self.sc, None]
        d = ~self.sc
        Md[d] = (P + self.D[:, None] * Pp)[:-1][d]
        Q = self.D[:, None] * dP
        Qd = np.empty((self.nres, self.m))
        Qd[self.sc] = (Q[1:] - Q[:-1])[self.sc] / self.mu[:-1][self.sc, None]
        Qd[d] = (dP + self.D[:, None] * dPp)[:-1][d]
        return P, M, Md, Q, Qd

    def residual_jac(self, x, want_jac=True):
        v, a = x[:self.m], np.exp(x[self.m:])
        P, M, Md, Q, Qd = self._blocks(a)
        y = self.trial(P, v)
        R = Md @ v - self.f(self.t[:-1], y[:-1])
        if not want_jac:
            return R, None
        J = np.empty((self.nres, 2 * self.m))
        fy = self.dfdy(self.t[:-1], y[:-1])[:, None]
        J[:, :self.m] = Md - fy * M[:-1]
        J[:, self.m:] = (Qd - fy * Q[:-1]) * (v * a)[None, :]
        return R, J

    def linear_AB(self, a):
        P, M, Md, _, _ = self._blocks(a)
        z = np.zeros(self.nres)
        lam = self.dfdy(self.t[:-1], z)
        return Md - lam[:, None] * M[:-1], self.f(self.t[:-1], z) + lam * self.y0, P


def train_hybrid_scale(prob, max_nfev=600, x0=None, seed=0, mode='hybrid'):
    if x0 is None:
        x0 = prob.init_x(seed)
    lb = np.r_[np.full(prob.m, -np.inf), np.log(prob.lo)]
    ub = np.r_[np.full(prob.m, np.inf), np.log(prob.hi)]

    def run_full(start):
        fun = lambda x: prob.residual_jac(x, False)[0]
        jfn = lambda x: prob.residual_jac(x, True)[1]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return least_squares(fun, start, jac=jfn, bounds=(lb, ub),
                                 max_nfev=max_nfev, **TOL, **TRF)

    def run_vp(th0):
        def red(th):
            A, b, _ = prob.linear_AB(np.exp(th))
            v = _lstsq(A, b)
            if v is None:
                return np.full(prob.nres, BIG)
            r = A @ v - b
            return r if np.all(np.isfinite(r)) else np.full(prob.nres, BIG)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return least_squares(red, th0, bounds=(np.log(prob.lo), np.log(prob.hi)),
                                 max_nfev=max_nfev, **TOL, **TRF)

    tic = time.perf_counter()
    if mode in ('varpro', 'hybrid'):
        s = run_vp(x0[prob.m:])
        A, b, _ = prob.linear_AB(np.exp(s.x))
        v = _lstsq(A, b)
        if v is None:
            v = np.zeros(prob.m)
        x, nfev, st = np.r_[v, s.x], s.nfev, s.status
        if mode == 'hybrid':
            s2 = run_full(x)
            if 0.5 * float(s2.fun @ s2.fun) <= 0.5 * float(s.fun @ s.fun):
                x, st = s2.x, s2.status
            nfev += s2.nfev
    else:
        s = run_full(x0)
        x, nfev, st = s.x, s.nfev, s.status
    wall = time.perf_counter() - tic
    R = prob.residual_jac(x, False)[0]
    a = np.exp(x[prob.m:])
    P = prob._blocks(a)[0]          # use the problem's own activation, not the default
    return dict(x=x, y=prob.trial(P, x[:prob.m]), R=R, cost=0.5 * float(R @ R),
                nfev=int(nfev), wall=wall, alpha=a, status=int(st),
                converged=bool(st > 0),
                admissible=bool(np.all(a >= prob.lo * (1 - 1e-9))
                                and np.all(a <= prob.hi * (1 + 1e-9))))


def multistart_hybrid(make, ytrue, nstarts=20, max_nfev=600, mode='hybrid', seed0=0):
    prob = make()
    rows, runs = [], []
    for s in range(nstarts):
        r = train_hybrid_scale(prob, max_nfev, prob.init_x(seed0 + s), mode=mode)
        e = np.abs(r['y'] - ytrue)
        rows.append(dict(start=s + 1, cost=r['cost'], max_error=float(e.max()),
                         rmse=float(np.sqrt(np.mean(e * e))), nfev=r['nfev'],
                         wall_s=r['wall'], admissible=r['admissible'],
                         converged=r['converged'],
                         res_inf=float(np.max(np.abs(r['R'])))))
        runs.append(r)
    return pd.DataFrame(rows), runs, prob
