# Gaussian Radial Basis Function Neural Networks on Time Scales

Reference implementation, numerical results and figures.

> **Note.** The manuscript itself (LaTeX source and PDF) is deliberately **not**
> included: several journals treat a publicly posted paper as prior
> publication. This repository holds only code, generated data and figures.
> `.gitignore` is set up to keep it that way.

## Quick start

```bash
pip install -r requirements.txt

python code/stress_test.py          # 40 theory-vs-code consistency checks
python code/make_figures.py         # regenerates every figure and table
python code/make_architecture.py    # regenerates the architecture diagram
python code/compare_optimizers.py   # gradient descent vs Gauss-Newton
```

Everything runs on one CPU core in a couple of minutes. No GPU.

## The activation function

```
p_{a,c}(tau) = a h_1(tau, c) = a (tau - c)
phi(t,c)     = e_{ominus p_{a,c}}(t, c)
```

The subscript is a **function** of the integration variable, not a constant.
On `T = R` this gives exactly `exp(-a (t-c)^2 / 2)`, so `a = 1/sigma^2`
recovers the classical Gaussian.

| time scale | `phi(t,c)`, `t >= c` |
|---|---|
| `R` | `exp(-a (t-c)^2 / 2)` |
| `Z`, `r = t-c` | `prod_{k=0}^{r-1} (1 + a k)^{-1}` |
| `hZ`, `t = c + n h` | `prod_{k=0}^{n-1} (1 + a h^2 k)^{-1}` |
| `q^N0`, `c = q^M` | `prod_{k=0}^{n-1} (1 + a c^2 q^k (q^k - 1))^{-1}` |

**Admissible widths.** With `W(c) = sup_{tau<c} mu(tau)(c-tau)`, the parameter
`ominus p_{a,c}` is positively regressive on all of `T` iff `0 < a < 1/W(c)`.
On `q^N0` with `c = q^M` this is `a_max = 4/((q-1) c^2)`; on `hZ` regressivity
fails exactly at `a = 1/(k h^2)`. Every training start is clipped to
`0.9 a_max(c_j)`.

## Error measures

| symbol | meaning | computable without the exact solution? |
|---|---|---|
| `R(t,p)` | residual | yes |
| `E(p) = 1/2 sum R^2` | error functional minimised | yes |
| `e(t) = abs(y_a - y)` | pointwise error | no |
| `norm(e,inf)`, `RMSE` | uniform, root-mean-square | no |
| `B(t) = int e_L(t,sigma(s)) abs(R(s)) Delta s` | a-posteriori certificate | **yes** |
| `theta = norm(B,inf)/norm(e,inf) >= 1` | effectivity index | no |

`e(t) <= B(t)` follows from the integral form of Gronwall's inequality on time
scales; `B` remains available when the exact solution is unknown.

## Training method

Measured on `T = 0.1Z`, `y' = -y`, `m = 5`, identical restart budget
(5 width decades x 2 weight seeds):

| method | evaluations | `E(p)` | `norm(e,inf)` | outcome |
|---|---|---|---|---|
| GD, `eta=1e-1` | 4.4e5 | - | - | all 10 starts inadmissible |
| GD, `eta=1e-2` | 4.4e5 | 1.108e-03 | 3.224e-03 | 4/10 failed |
| GD, `eta=1e-4` | 4.4e5 | 1.497e-01 | 7.984e-02 | 6/10 failed |
| GD on `(v, log a)` | 4.4e5 | 8.466e-03 | 8.763e-03 | best of 10 |
| Levenberg-Marquardt | 453 | 5.536 | 6.436e-01 | left admissible set |
| **trust-region reflective** | 23932 | **1.340e-11** | **1.977e-07** | **converged** |

`E(p)` is a sum of squares, so Gauss-Newton methods use `J^T J` as a Hessian
surrogate. Gradient descent cannot work: admissibility forces the widths
several decades below the output weights, so no single `eta` serves both.
Levenberg-Marquardt has no trust region on the parameters and steps out of the
admissible set. Trust-region reflective is the only method both accurate and
reliable, and is used throughout.

## Results

| Example | `T` | `N` | `m` | `norm(e,inf)` | RMSE | `norm(B,inf)` | `theta` |
|---|---|---|---|---|---|---|---|
| 1: `y' = -y/(2t)` | `2^N0`, `{1..1024}` | 11 | 5 | 6.66e-16 | 2.48e-16 | 2.24e-14 | 33.6 |
| 2: `y' = -y y^sigma` | harmonic `{1-1/n} u {1}` | 24 | 8 | 1.30e-11 | 3.83e-12 | 9.11e-11 | 7.0 |

Example 2 uses a time scale with an **accumulation point** at `t = 1`, where
`mu(1-1/n) = 1/(n(n+1))` spans more than an order of magnitude on a bounded
interval. Its exact solution `y = 1/(t+1)` solves `y' = -y y^sigma` on *every*
time scale, since `y' = -[(t+1)(sigma(t)+1)]^{-1}`.

## Stress test

`python code/stress_test.py` — 40 checks in seven groups:

A. `phi^Delta = (ominus p) phi`, range, decay, on four time scales
B. closed forms on `R`, `hZ`, `q^N0`
C. admissibility, including rejection above `a_max`
D. the width gradient against central differences
E. trial-solution delta derivatives at orders 1, 2 and `n = 3`
F. that each example's exact solution really solves its equation
G. `e(t) <= B(t)`

Group E pins down a correction to the second-order Leibniz expansion: the
corrected formula holds to `7e-15`, the erroneous one is off by `91`.

## Layout

```
code/tsrbf.py               time scale, activation, trial solution,
                            trust-region training, Gronwall certificate
code/make_figures.py        regenerates every figure and table
code/make_architecture.py   the architecture diagram
code/stress_test.py         40 theory-vs-code consistency checks
code/compare_optimizers.py  training-method comparison
results/figures/            PNG
results/tables/             CSV and TeX fragments
```

## License

MIT — see `LICENSE`.
