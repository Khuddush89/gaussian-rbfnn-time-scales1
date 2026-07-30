"""
Gaussian radial basis function neural networks on time scales.

Implements exactly the construction of the manuscript:

    p_{alpha,c}(tau) = alpha * h_1(tau, c) = alpha (tau - c)
    phi(t,c)         = e_{ominus p_{alpha,c}}(t, c)

On T = R this equals exp(-alpha (t-c)^2 / 2), so alpha = 1/sigma^2 recovers the
classical Gaussian exp(-(t-c)^2/(2 sigma^2)).

Trial solution for an n-th order IVP posed at t_0 = min T:

    y_a(t) = sum_{k=0}^{n-1} h_k(t,t_0) y_k + h_n(t,t_0) N(t)

License: MIT
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

__all__ = ["TimeScale", "uniform", "quantum", "real_grid", "hybrid",
           "cantor", "harmonic", "phi_ts", "dphi_dalpha", "TrialSolution", "train",
           "train_lm", "certified_bound", "alpha_admissible_max"]


# --------------------------------------------------------------------------
#  time scale
# --------------------------------------------------------------------------
class TimeScale:
    """Finite model of a time scale: points, sigma, mu, Delta, generalised h_k."""

    def __init__(self, points, name="T"):
        t = np.unique(np.asarray(points, dtype=float).ravel())
        if t.size < 2:
            raise ValueError("need at least two points")
        self.t, self.N, self.name = t, t.size, name
        self.mu = np.empty(self.N)
        self.mu[:-1] = np.diff(t)
        self.mu[-1] = 0.0
        self.sigma_idx = np.minimum(np.arange(self.N) + 1, self.N - 1)

    def __repr__(self):
        return (f"TimeScale({self.name}, N={self.N}, "
                f"[{self.t[0]:g},{self.t[-1]:g}], max mu={self.mu.max():g})")

    def delta(self, y):
        """Delta derivative along axis 0; y may be (N,) or (N, k)."""
        y = np.asarray(y, float)
        out = np.empty_like(y)
        mu = self.mu[:-1]
        if y.ndim == 1:
            out[:-1] = (y[1:] - y[:-1]) / mu
        else:
            out[:-1] = (y[1:] - y[:-1]) / mu[:, None]
        out[-1] = out[-2]
        return out

    def delta_n(self, y, n):
        for _ in range(n):
            y = self.delta(y)
        return y

    def shift(self, y):
        return np.asarray(y)[self.sigma_idx]

    def cum_integral(self, f):
        f = np.asarray(f, float)
        out = np.zeros(self.N)
        out[1:] = np.cumsum(f[:-1] * self.mu[:-1])
        return out

    def h(self, k, s_idx=0):
        """Generalised monomial h_k(., t_{s_idx})."""
        cur = np.ones(self.N)
        for _ in range(k):
            F = self.cum_integral(cur)
            cur = F - F[s_idx]
        return cur


def uniform(a, b, h, name=None):
    if not h > 0:
        raise ValueError("h must be positive")
    n = int(round((b - a) / h))
    return TimeScale(a + h * np.arange(n + 1), name or f"{h:g}Z")


def quantum(q, k_min, k_max, name=None):
    if not q > 1:
        raise ValueError("quantum scales need q > 1")
    return TimeScale(q ** np.arange(k_min, k_max + 1, dtype=float),
                     name or f"{q:g}^N0")


def cantor(level, name=None):
    """Endpoints of the level-`level` middle-thirds Cantor set.

    Level K has 2**K closed intervals of length 3**-K; taking both endpoints
    gives 2**(K+1) points.  The graininess takes the K+1 distinct values
    3**-K (inside an interval) and 3**-j, j = 1..K (across the removed gaps),
    so max(mu)/min(mu) = 3**(K-1): an extremely nonuniform time scale.
    """
    iv = [(0.0, 1.0)]
    for _ in range(level):
        nxt = []
        for a, b in iv:
            t = (b - a) / 3.0
            nxt += [(a, a + t), (b - t, b)]
        iv = nxt
    pts = np.unique(np.array([x for ab in iv for x in ab]))
    return TimeScale(pts, name or f"Cantor_{level}")


def harmonic(n_max, n_min=1, name=None):
    """T = {1 - 1/n : n = n_min..n_max} u {1}.

    The point t = 1 is an accumulation point approached from the left, and
    mu(1 - 1/n) = 1/(n(n+1)) tends to zero, so the graininess spans several
    orders of magnitude on a bounded interval.
    """
    pts = np.array([1.0 - 1.0 / n for n in range(n_min, n_max + 1)] + [1.0])
    return TimeScale(np.unique(pts), name or f"harmonic_{n_min}_{n_max}")


def real_grid(a, b, n, name="R"):
    return TimeScale(np.linspace(a, b, n), name)


def hybrid(blocks, n_dense=41, name="T_hyb"):
    """blocks: ('interval', a, b) or ('points', [...])."""
    pts = []
    for blk in blocks:
        if blk[0] == "interval":
            pts.append(np.linspace(blk[1], blk[2], n_dense))
        else:
            pts.append(np.asarray(blk[1], float))
    return TimeScale(np.concatenate(pts), name)


# --------------------------------------------------------------------------
#  the activation function of the manuscript
# --------------------------------------------------------------------------
def phi_ts(ts: TimeScale, centers, alphas):
    r"""Phi[i,j] = e_{ominus alpha_j h_1(., c_j)}(t_i, c_j).

    log phi(t_i, c_j) = -( G_j(i) - G_j(i_c) ),
        G_j(i) = sum_{k<i} log( 1 + mu_k alpha_j (t_k - c_j) ),
    which is the delta integral of xi_mu(ominus p) between c_j and t_i.
    """
    cidx = np.asarray(centers, dtype=int).ravel()
    al = np.broadcast_to(np.atleast_1d(np.asarray(alphas, float)),
                         (cidx.size,))
    if np.any(al <= 0) or np.any(~np.isfinite(al)):
        raise ValueError("alphas must be finite and positive")
    t, mu, N = ts.t, ts.mu, ts.N
    out = np.empty((N, cidx.size))
    for j, (c, a) in enumerate(zip(cidx, al)):
        fac = 1.0 + mu[:-1] * a * (t[:-1] - t[c])
        if np.any(fac <= 0):
            raise ValueError(
                f"ominus p is not positively regressive for centre index {c}, "
                f"alpha={a:g}: need mu(tau) alpha (c-tau) != 1 for tau < c")
        G = np.zeros(N)
        G[1:] = np.cumsum(np.log(fac))
        out[:, j] = np.exp(-(G - G[c]))
    return out


def dphi_dalpha(ts: TimeScale, centers, alphas, Phi=None):
    """d phi / d alpha = -phi * int_c^t (tau-c)/(1+mu alpha (tau-c)) Delta tau."""
    cidx = np.asarray(centers, dtype=int).ravel()
    al = np.broadcast_to(np.atleast_1d(np.asarray(alphas, float)),
                         (cidx.size,))
    if Phi is None:
        Phi = phi_ts(ts, cidx, al)
    t, mu, N = ts.t, ts.mu, ts.N
    out = np.empty((N, cidx.size))
    for j, (c, a) in enumerate(zip(cidx, al)):
        d = t[:-1] - t[c]
        g = np.zeros(N)
        g[1:] = np.cumsum(mu[:-1] * d / (1.0 + mu[:-1] * a * d))
        out[:, j] = -Phi[:, j] * (g - g[c])
    return out


def alpha_admissible_max(ts: TimeScale, c_idx):
    """Largest alpha for which ominus (alpha h_1(.,c)) stays positively
    regressive on all of T, i.e.  sup_{tau<c} mu(tau) alpha (c-tau) < 1.

    Returns +inf when c is the left endpoint (then tau >= c throughout).
    """
    c = ts.t[c_idx]
    lo = ts.t[:c_idx]
    if lo.size == 0:
        return np.inf
    w = ts.mu[:c_idx] * (c - lo)
    wmax = float(np.max(w))
    return np.inf if wmax <= 0 else 1.0 / wmax


# --------------------------------------------------------------------------
#  trial solution and training
# --------------------------------------------------------------------------
class TrialSolution:
    def __init__(self, ts, order, init_values, centers):
        self.ts, self.n = ts, int(order)
        self.y0 = np.asarray(init_values, float).ravel()
        if self.y0.size != self.n:
            raise ValueError("need `order` initial values")
        self.centers = np.asarray(centers, dtype=int).ravel()
        self.m = self.centers.size
        A = np.zeros(ts.N)
        for k in range(self.n):
            A = A + self.y0[k] * ts.h(k, 0)
        self.A = np.stack([ts.delta_n(A, j) for j in range(self.n + 1)])
        self.hn = ts.h(self.n, 0)
        self.coll = np.arange(ts.N - self.n)

    def design(self, alphas):
        G = self.hn[:, None] * phi_ts(self.ts, self.centers, alphas)
        return np.stack([self.ts.delta_n(G, j) for j in range(self.n + 1)])

    def evaluate(self, v, alphas, B=None):
        if B is None:
            B = self.design(alphas)
        v = np.asarray(v, float).ravel()
        return [self.A[j] + B[j] @ v for j in range(self.n + 1)]


def train_lm(trial, residual_fn, alpha0, max_nfev=4000, tol=1e-14,
             decades=(1e-2, 1e-1, 1.0, 1e1, 1e2), seeds=(0, 1)):
    """Levenberg-Marquardt on (v, log alpha).

    E(p) = 1/2 sum R^2 is a sum of squares, so Gauss-Newton uses J^T J as a
    Hessian surrogate.  Requires M >= 2m; falls back to the trust-region
    reflective method otherwise.
    """
    ts, m, coll = trial.ts, trial.m, trial.coll
    meth = "lm" if coll.size >= 2 * m else "trf"
    best, costs = None, []

    def F(p):
        v, al = p[:m], np.exp(p[m:])
        try:
            return residual_fn(ts, trial.evaluate(v, al))[coll]
        except (ValueError, FloatingPointError):
            return np.full(coll.size, 1e6)

    for k in decades:
        for sd in seeds:
            rng = np.random.default_rng(sd)
            p0 = np.concatenate([0.05 * rng.standard_normal(m),
                                 np.log(np.full(m, alpha0 * k))])
            try:
                sol = least_squares(F, p0, method=meth, xtol=tol, ftol=tol,
                                    gtol=tol, max_nfev=max_nfev)
            except Exception:
                continue
            costs.append(float(sol.cost))
            if best is None or sol.cost < best["E"]:
                best = {"v": sol.x[:m], "alpha": np.exp(sol.x[m:]),
                        "E": float(sol.cost), "nfev": int(sol.nfev),
                        "method": meth}
    if best is not None:
        c = np.array(costs)
        best.update(n_starts=len(decades) * len(seeds),
                    E_median=float(np.median(c)))
    return best


def train(trial, residual_fn, alpha0, v0=None, max_nfev=4000, tol=1e-14,
          decades=(1e-2, 1e-1, 1.0, 1e1, 1e2), seeds=(0, 1)):
    """Trust-region nonlinear least squares on (v, log alpha).

    Multi-start over |decades| x |seeds| initialisations; returns the best run
    together with the median objective so that variability is visible.
    """
    ts, m, coll = trial.ts, trial.m, trial.coll
    best, costs = None, []

    def make_resid(al_scale):
        cache = {}

        def resid(p):
            v, al = p[:m], np.exp(p[m:])
            key = al.tobytes()
            if key not in cache:
                cache.clear()
                try:
                    cache[key] = trial.design(al)
                except ValueError:
                    # alpha has left the regressive range: return a large but
                    # finite residual so the trust region backs off instead of
                    # the whole run aborting
                    cache[key] = None
            if cache[key] is None:
                return np.full(coll.size, 1e6)
            return residual_fn(ts, trial.evaluate(v, al, cache[key]))[coll]
        return resid

    # Proposition (admissible widths): clip every start to alpha < alpha_max(c_j)
    amax = np.array([min(alpha_admissible_max(ts, c), 1e8)
                     for c in trial.centers])
    for k in decades:
        for sd in seeds:
            rng = np.random.default_rng(sd)
            a = np.minimum(np.full(m, alpha0 * k, dtype=float), 0.9 * amax)
            v = 0.05 * rng.standard_normal(m) if v0 is None else np.asarray(v0)
            try:
                sol = least_squares(make_resid(k),
                                    np.concatenate([v, np.log(a)]),
                                    method="trf", xtol=tol, ftol=tol,
                                    gtol=tol, max_nfev=max_nfev)
            except Exception:
                continue
            costs.append(float(sol.cost))
            if best is None or sol.cost < best["E"]:
                best = {"v": sol.x[:m], "alpha": np.exp(sol.x[m:]),
                        "E": float(sol.cost), "nfev": int(sol.nfev)}
    if best is not None:
        c = np.array(costs)
        best.update(n_starts=len(decades) * len(seeds),
                    E_median=float(np.median(c)))
    return best


# --------------------------------------------------------------------------
#  a-posteriori certificate
# --------------------------------------------------------------------------
def certified_bound(ts: TimeScale, residual, L):
    """|y_a - y|(t) <= int_{t_0}^t e_L(t, sigma(s)) |R(s)| Delta s."""
    R = np.abs(np.asarray(residual, float))
    Larr = np.broadcast_to(np.atleast_1d(np.asarray(L, float)), (ts.N,))
    fac = 1.0 + ts.mu[:-1] * Larr[:-1]
    if np.any(fac <= 0):
        raise ValueError("L must be positively regressive")
    logc = np.zeros(ts.N)
    logc[1:] = np.cumsum(np.log(fac))
    lj = np.concatenate([logc[1:], logc[-1:]])
    E = np.exp(logc[:, None] - lj[None, :])
    idx = np.arange(ts.N)
    E[idx[:, None] <= idx[None, :]] = 0.0
    w = ts.mu.copy()
    w[-1] = 0.0
    return E @ (R * w)
