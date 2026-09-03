"""
Core library: Gaussian-type radial basis functions on time scales.

phi_{alpha,c}(t) = e_{ominus p}(t,c),   p(tau) = alpha*(tau-c)

Key identities implemented and verified here:
  (P1)  phi^Delta(t) = -alpha*(t-c)*phi^sigma(t),  phi(c)=1
  (P2)  forward product  phi(t_i) = prod_{k=r}^{i-1} 1/(1+mu_k*alpha*(t_k-c))
  (P3)  backward product phi(t_i) = prod_{k=i}^{r-1} (1+mu_k*alpha*(t_k-c))
  (P4)  envelope   exp(-a h2/(1-g)) <= phi <= exp(-a h2/(1+g)),
                   g = alpha*sup mu(tau)|tau-c| < 1
"""
import numpy as np

# ----------------------------------------------------------------------
# discrete (purely right-scattered) time-scale grids
# ----------------------------------------------------------------------

def graininess(t):
    t = np.asarray(t, float)
    return np.diff(t)


def phi_matrix(t, cidx, alpha):
    """Phi[i,j] = phi_{alpha_j, t[cidx_j]}(t_i).

    Both product formulas collapse to a single ratio of cumulative products,
        phi(t_i) = G_r / G_i,   G_i = prod_{k<i} (1 + mu_k alpha (t_k - c)),
    which is evaluated in logarithms for stability (every factor is positive
    under the positive-regressivity condition).
    """
    return np.exp(_logphi(t, cidx, alpha)[0])


def _logphi(t, cidx, alpha):
    t = np.asarray(t, float)
    mu = np.diff(t)[:, None]
    cidx = np.atleast_1d(np.asarray(cidx, int))
    alpha = np.atleast_1d(np.asarray(alpha, float))[None, :]
    d = (t[:-1, None] - t[cidx][None, :])
    g = 1.0 + mu * alpha * d
    with np.errstate(divide='ignore', invalid='ignore'):
        L = np.vstack([np.zeros((1, cidx.size)), np.cumsum(np.log(np.abs(g)), 0)])
    return L[cidx, np.arange(cidx.size)][None, :] - L, (g, mu, d)


def phi_and_dlogphi(t, cidx, alpha):
    """Return Phi and S = d(log Phi)/d(alpha_j) = -(T_i - T_r)."""
    cidx = np.atleast_1d(np.asarray(cidx, int))
    lp, (g, mu, d) = _logphi(t, cidx, alpha)
    P = np.exp(lp)
    T = np.vstack([np.zeros((1, cidx.size)), np.cumsum(mu * d / g, 0)])
    S = -(T - T[cidx, np.arange(cidx.size)][None, :])
    return P, S


def alpha_max(t, cidx):
    """Exact positive-regressivity limit alpha_max(c) = 1/sup_{tau<c} mu(tau)(c-tau)."""
    t = np.asarray(t, float)
    mu = np.diff(t)
    out = []
    for r in np.atleast_1d(np.asarray(cidx, int)):
        if r == 0:
            out.append(np.inf)
        else:
            W = float(np.max(mu[:r] * (t[r] - t[:r])))
            out.append(1.0 / W if W > 0 else np.inf)
    return np.asarray(out, float)


def gamma_const(t, cidx, alpha):
    """gamma = alpha * sup_{tau in I} mu(tau)|tau - c| (envelope theorem constant)."""
    t = np.asarray(t, float)
    mu = np.diff(t)
    out = []
    for r, a in zip(np.atleast_1d(np.asarray(cidx, int)), np.atleast_1d(alpha)):
        out.append(a * float(np.max(mu * np.abs(t[:-1] - t[r]))))
    return np.asarray(out, float)


def h2(t, c):
    """Time-scale monomial h_2(t,c) = int_c^t (s-c) Delta s, on a discrete grid."""
    t = np.asarray(t, float)
    mu = np.diff(t)
    r = int(np.argmin(np.abs(t - c)))
    w = mu * (t[:-1] - c)
    C = np.concatenate(([0.0], np.cumsum(w)))
    return C - C[r]


# ----------------------------------------------------------------------
# width box on a finite interval
# ----------------------------------------------------------------------

def width_box(t, cidx, safety=0.95, rel_floor=1e-12):
    """Return (lo, hi) admissible width box for the time-scale basis.

    hi_j = safety * alpha_max(c_j) when finite; otherwise a resolution cap
    tied to the grid, 1/(min mu)^2, which only bounds the numerical search.
    """
    t = np.asarray(t, float)
    mu = np.diff(t)
    cap = 1.0 / float(np.min(mu[mu > 0])) ** 2
    am = alpha_max(t, cidx)
    hi = np.where(np.isfinite(am), safety * am, cap)
    hi = np.minimum(hi, cap)
    lo = np.maximum(rel_floor * hi, 1e-16)
    return lo, hi


def width_box_classical(t, cidx, span_factor=1e3, rel_floor=1e-12):
    """Fair, generous width box for the CLASSICAL Gaussian.

    The classical Gaussian has no regressivity restriction, so it is given a
    scale-based range covering everything from essentially flat to
    essentially a delta at the nearest neighbouring node.
    """
    t = np.asarray(t, float)
    mu = np.diff(t)
    d = float(t[-1] - t[0])
    hi = span_factor / float(np.min(mu[mu > 0])) ** 2
    lo = rel_floor / d ** 2
    n = np.atleast_1d(np.asarray(cidx, int)).size
    return np.full(n, lo), np.full(n, hi)


def classical_matrix(t, cidx, alpha):
    t = np.asarray(t, float)
    C = t[np.atleast_1d(np.asarray(cidx, int))][None, :]
    return np.exp(-0.5 * alpha[None, :] * (t[:, None] - C) ** 2)


def classical_and_dlog(t, cidx, alpha):
    t = np.asarray(t, float)
    C = t[np.atleast_1d(np.asarray(cidx, int))][None, :]
    D2 = (t[:, None] - C) ** 2
    return np.exp(-0.5 * alpha[None, :] * D2), -0.5 * D2


# ----------------------------------------------------------------------
# hybrid time scales: union of closed intervals
# ----------------------------------------------------------------------

class HybridScale:
    """T = union_k [a_k, b_k], sampled with `per` points per interval.

    Stores nodes, graininess (0 inside an interval, gap at each right endpoint)
    and a flag marking right-dense nodes.
    """

    def __init__(self, intervals, per=9):
        nodes, dense = [], []
        for k, (a, b) in enumerate(intervals):
            xs = np.linspace(a, b, per)
            nodes.append(xs)
            d = np.ones(per, bool)
            d[-1] = False              # right endpoint of a block is scattered
            if k == len(intervals) - 1:
                d[-1] = False          # maximum of T
            dense.append(d)
        self.t = np.concatenate(nodes)
        self.dense = np.concatenate(dense)
        self.mu = np.concatenate([np.diff(self.t), [0.0]])
        self.mu[:-1] = np.diff(self.t)
        self.dense[-1] = False
        self.intervals = intervals
        self.per = per

    def phi(self, cidx, alpha):
        """phi and d phi/d alpha on a hybrid scale.

        Inside an interval phi solves phi' = -alpha (t-c) phi, giving
        exp(-alpha[(t-c)^2-(t0-c)^2]/2); across a gap the scattered factor
        1/(1+mu alpha (t-c)) applies.
        """
        t, mu = self.t, self.mu
        n = t.size
        cidx = np.atleast_1d(np.asarray(cidx, int))
        alpha = np.atleast_1d(np.asarray(alpha, float))
        P = np.ones((n, cidx.size))
        S = np.zeros((n, cidx.size))          # d log phi / d alpha
        for j, (r, a) in enumerate(zip(cidx, alpha)):
            c = t[r]
            for i in range(r + 1, n):
                if self.dense[i - 1]:
                    step = -0.5 * ((t[i] - c) ** 2 - (t[i - 1] - c) ** 2)
                    P[i, j] = P[i - 1, j] * np.exp(a * step)
                    S[i, j] = S[i - 1, j] + step
                else:
                    m = mu[i - 1]
                    f = 1.0 + m * a * (t[i - 1] - c)
                    P[i, j] = P[i - 1, j] / f
                    S[i, j] = S[i - 1, j] - m * (t[i - 1] - c) / f
            for i in range(r - 1, -1, -1):
                if self.dense[i]:
                    step = -0.5 * ((t[i] - c) ** 2 - (t[i + 1] - c) ** 2)
                    P[i, j] = P[i + 1, j] * np.exp(a * step)
                    S[i, j] = S[i + 1, j] + step
                else:
                    m = mu[i]
                    f = 1.0 + m * a * (t[i] - c)
                    P[i, j] = P[i + 1, j] * f
                    S[i, j] = S[i + 1, j] + m * (t[i] - c) / f
        return P, S

    def alpha_max(self, cidx):
        t, mu = self.t, self.mu
        out = []
        for r in np.atleast_1d(np.asarray(cidx, int)):
            m = mu[:r]
            w = m * (t[r] - t[:r])
            W = float(np.max(w)) if r > 0 else 0.0
            out.append(1.0 / W if W > 0 else np.inf)
        return np.asarray(out, float)


def exp_ts(lmbda, hs):
    """Time-scale exponential e_lambda(t,t0) on a HybridScale (lambda constant)."""
    t, mu, dense = hs.t, hs.mu, hs.dense
    y = np.ones(t.size)
    for i in range(1, t.size):
        if dense[i - 1]:
            y[i] = y[i - 1] * np.exp(lmbda * (t[i] - t[i - 1]))
        else:
            y[i] = y[i - 1] * (1.0 + mu[i - 1] * lmbda)
    return y
