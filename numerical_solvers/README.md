# numerical_solvers

Generic numerical methods for the cases of this repository. The package knows nothing about shafts,
bearings, gears or contact: the caller gives a function, the solver drives the method, keeps the history
and says why it stopped. It imports only NumPy, SciPy and the standard library (layer rules in
`tests/test_imports.py`).

```text
numerical_solvers/
  __init__.py          public API (eager re-exports)
  newton_raphson/      newton, NewtonOptions, NewtonError, NewtonResult, IterationRecord
  richardson_extrapolation/
                       richardson, richardson_triple, RichardsonOptions, RichardsonError,
                       RichardsonResult, LevelRecord, MetricConvergence
```
A subpackage is named after its method, not after its entry function (`newton_raphson/newton`,
`richardson_extrapolation/richardson`): a subpackage named like the function it re-exports is shadowed by
it. Names in the top-level `__all__` are unique across the solvers.

## Solvers

| Solver | Solves | Call | Returns | Status |
|---|---|---|---|---|
| `newton` | `F(x) = 0`, `x` in R^n | `newton(residual, x0, *, jacobian=None, options=None, callback=None, name="", label="")` | `NewtonResult` | ported from AxisForge d34ec7f; logic under José's line-by-line review; NR-xx open (below) |
| `richardson` | observed order, extrapolated value and GCI of metrics over refinement levels | `richardson(evaluate, levels, *, options=None, callback=None, name="", label="")` | `RichardsonResult` | new; formulas of AxisForge `RichardsonGCI` (cross-checked); R3 OPEN points |

Conventions shared by the solvers: frozen options checked at construction (`validate()`,
`validate_or_raise()`); a frozen result with `record()` (summary, one flat dict) and `records()` (history,
one flat dict per step) for `axisforge.outputs._format.write_records_csv`; `name` and `label` copied into
every row; `on_failure="return"` (check `converged` and `reason`) or `"raise"` (an error carrying the
result); `callback(record)` after each step, returning True stops the run.

## newton_raphson

Newton-Raphson for a nonlinear system, with Armijo line search or damping, analytic or finite-difference
(forward, central) Jacobian, modified Newton, scaling of unknowns and residuals, four stopping tests and
an optional SciPy `root` fallback. The algorithm and every option are in the docstrings of
`newton_raphson/newton.py`.

- Source: AxisForge `axisforge/solvers/numerics/newton.py` and
  `axisforge/results/numerics/newton_raphson/newton_results.py` at d34ec7f (branch
  `feature/newton-solver`). Only imports, module paths and docstring references differ; the 35 tests are
  ported and `test_port_matches_axisforge` checks, when AxisForge has the module, that both solvers call
  the residual at the same points and return the same result and history.
- Open points (`Tese/80_Inbox/2026-10-27/axisforge_newton/issues.csv`): the port carries NR-02..NR-09,
  NR-11, NR-13..NR-15 and NR-17 unchanged; NR-01 (one-file results folder) does not arise here (the
  results sit next to the solver); NR-10, NR-12 and NR-16 concern AxisForge only.
- Two copies exist while AxisForge `feature/newton-solver` is unmerged: this one is the copy the cases of
  this repository use; what happens to the AxisForge branch is José's decision (R3 OPEN).
- Method sources (not in the library): Dennis and Schnabel (1996), Nocedal and Wright (2006), Kelley
  (2003).

## richardson_extrapolation

The caller gives `evaluate(level) -> float | {name: float}` (its model at one refinement level) and the
levels, coarse to fine: a sequence, or `refine(k) -> level` with `max_levels` (adaptive). The solver calls
`evaluate` once per level, in order, and from the third level on judges every metric on the last triple
with `richardson_triple`; it stops when every judged metric is converged (`stop_when_converged`), when the
levels run out, or when the callback returns True.

- What a level is: `RichardsonOptions.kind`, `"size"` (h, a time step: `r = level[k-1] / level[k]`) or
  `"count"` (elements or intervals along one direction: `r = level[k] / level[k-1]`; not grid points,
  not a total count in 2-D: pass sizes then). Levels must be positive and refine (ratio > 1), else
  `ValueError`.
- One triple, ratio `r` > 1 equal for both intervals: `p = ln((f_c - f_m) / (f_m - f_f)) / ln r` (clamped
  to `[min_p, max_p]`), `f_h0 = f_f + (f_f - f_m) / (r^p - 1)`, `e_mc = (f_m - f_c) / f_c`,
  `e_fm = (f_f - f_m) / f_m`, `GCI = Fs |e| / (r^p - 1)` per interval, converged if both GCI
  < `threshold`. Formulas, checks and their order are those of AxisForge `RichardsonGCI`
  (`axisforge/solvers/mesh/convergence_solver.py`); `test_triple_matches_axisforge` checks equal numbers.
- Status per metric and triple, never silent: `"monotonic"`, `"flat"` (both differences below
  `max(flat_atol, flat_rtol max|f|)`, defaults 1e-8 = AxisForge's
  `max(1e-9, 1e-8 max(|f|, 1))`; `flat_atol` is in the units of the metric: GCI 0, converged), `"oscillatory"` (differences change sign),
  `"non-convergent"` (`p <= 0` or `r^p - 1 ~ 0`), `"invalid"` (a ratio not > 1, unequal ratios beyond
  `ratio_rtol`, `f_m == f_f`, a zero coarser value, a non-finite value). The last three carry NaN numbers,
  `converged=False` and a `reason`; AxisForge returns a `_DummyGCI` without one.
- Not implemented (R3 OPEN): unequal ratios (Celik et al. 2008 iteration for `p`), a treatment of
  oscillatory convergence, the relative error divided by the finer value.
- Sources: Richardson (1911), Roache (1998), Celik et al. (2008); none is in the library (source to
  find), so the tests use analytic sequences `f0 + C h^p` and AxisForge, not values quoted from them.

## How a study uses it

**newton**, a 2x2 system: a rigid body on two hardening springs, displacement `u` [mm] and rotation `th` [rad], a
force [N] and a moment [N mm] to balance.

```python
import numpy as np
from axisforge.outputs._format import write_records_csv
from numerical_solvers import NewtonOptions, newton

def residual(x):
    u, th = x
    d1, d2 = u - 50.0 * th, u + 50.0 * th                     # spring deflections [mm]
    q1, q2 = 1e4 * d1 + 1e7 * d1**3, 1e4 * d2 + 1e7 * d2**3   # spring forces [N]
    return np.array([q1 + q2 - 2000.0, 50.0 * (q2 - q1) - 1.0e4])

opts = NewtonOptions(jacobian_fd="central", x_scale=[1e-2, 1e-4], f_scale=[1e3, 1e5])
r = newton(residual, [0.0, 0.0], options=opts, name="rigid_body", label="F 2 kN, M 10 N m")
write_records_csv("newton_history.csv", r.records())   # in a case: its results/ folder
```

**richardson**, a BEM grid series: the smooth ball-on-flat contact of v104, load control, spatial kernel on a
domain of half-length 2 semi-axes (the v104 rule; the frequency kernel at span 4 has an approach error of
about 1 % that does not change with h, which the GCI cannot see: the GCI bounds the discretisation error
only). `Grid.around` rounds h down to a power of two, so the levels are powers of two (b/h in [8, 16),
doubling). A failed solve returns NaN, which makes its triples "invalid".

```python
import math
from slippy_bridge.contact import Grid, hertz, solve
from numerical_solvers import RichardsonOptions, richardson

contact = ball_on_flat()                       # the case's construction, unsolved, load control
hz = hertz(contact)
h0 = 2.0 ** math.floor(math.log2(hz.b / 8))    # [m]

def evaluate(h):                               # h [m], the BEM grid spacing
    grid = Grid.around(contact, hz, h=h, span_x=2.0, span_y=2.0)
    res = solve(contact, grid, results / "slippy", hertz_result=hz, use_frequency_domain=False)
    if not (res.converged and res.edge_clear and res.resolution_ok):
        return {"p0": math.nan, "a": math.nan, "delta": math.nan}
    return {"p0": res.p0, "a": res.a, "delta": res.delta}

opts = RichardsonOptions(kind="size", threshold=5e-3, max_p=2.0, min_levels=4, max_levels=6,
                         flat_atol=0.0)        # SI metrics much smaller than 1: no absolute floor
r = richardson(evaluate, lambda k: h0 / 2**k, options=opts, name="bem_grid", label="ball, 500 N")
write_records_csv(results / "bem_grid.csv", r.records())     # one row per level and metric
print(r.converged, r.reason, {m: (c.f_extrapolated, c.status) for m, c in r.final.items()})
```
`max_p` keeps a plateau from passing as a high-order convergence; `min_levels=4` keeps a spurious flat first
triple from stopping the run at three levels. The `max_p = 2` here is an illustration, not a known order of the
BEM pressure (R3 OPEN).
