# numerical_solvers

Generic numerical methods for the cases of this repository. The package knows nothing about shafts,
bearings, gears or contact: the caller gives a function, the solver drives the method, keeps the history
and says why it stopped. It imports only NumPy, SciPy and the standard library (layer rules in
`tests/test_imports.py`).

```text
numerical_solvers/
  __init__.py          public API (eager re-exports)
  newton_raphson/      newton, NewtonOptions, NewtonError, NewtonResult, IterationRecord
```

## Solvers

| Solver | Solves | Call | Returns | Status |
|---|---|---|---|---|
| `newton` | `F(x) = 0`, `x` in R^n | `newton(residual, x0, *, jacobian=None, options=None, callback=None, name="", label="")` | `NewtonResult` | ported from AxisForge d34ec7f; logic under José's line-by-line review; NR-xx open (below) |

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

## How a study uses it

A 2x2 system: a rigid body on two hardening springs, displacement `u` [mm] and rotation `th` [rad], a
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
