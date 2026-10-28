"""Generic Newton-Raphson solver for a nonlinear system ``F(x) = 0``, ``x`` in R^n (n >= 1).

Knows nothing about shafts, bearings or gears: any design-studies study passes a residual
(and optionally its Jacobian) and reads a ``NewtonResult``
(``numerical_solvers.newton_raphson``). Every choice a caller may want to make is a field of
``NewtonOptions``; the defaults are a safe start (Armijo line search, residual and relative
step tests, Euclidean norm, no scaling).

Ported from AxisForge ``axisforge/solvers/numerics/newton.py`` at d34ec7f: only imports, module
paths and docstring references differ; the logic is that of d34ec7f.

Algorithm (one iteration ``k``; the vectors are scaled, ``Fs = F / f_scale``,
``s = dx / x_scale``):

1. Jacobian ``J(x_k)``: the analytic callable, or finite differences (forward or central),
   evaluated every ``jacobian_update`` iterations (``1`` = Newton, ``> 1`` = modified Newton).
2. Newton step: ``p = linear_solver(J, -F)``. A singular Jacobian (``LinAlgError``, non-finite
   ``p``, or scaled condition number above ``max_condition``) stops the solve.
3. Cap: if ``||p / x_scale|| > max_step``, ``p`` is multiplied by ``t = max_step / ||p / x_scale||``
   (``t = 1`` otherwise).
4. Globalisation: ``"none"`` takes ``alpha = 1``; ``"damped"`` takes ``alpha = damping``;
   ``"line_search"`` backtracks ``alpha <- shrink * alpha`` from 1 until the Armijo condition
   on ``f = 0.5 ||Fs||_2^2`` holds, ``f(x + alpha t p) <= (1 - 2 c1 alpha t) f(x)``.
   ``x_{k+1} = project(x_k + alpha t p)``.
5. Stopping test on ``F(x_{k+1})`` and the step taken (see ``NewtonOptions``).

References
----------
Dennis and Schnabel (1996), Numerical Methods for Unconstrained Optimization and Nonlinear
Equations, SIAM: Newton's method for nonlinear equations, finite-difference Jacobian, backtracking
line search on ``0.5 ||F||^2``, scaling and stopping tests. Nocedal and Wright (2006), Numerical
Optimization, 2nd ed., Springer: Armijo sufficient-decrease condition and backtracking. Kelley
(2003), Solving Nonlinear Equations with Newton's Method, SIAM: chord and Shamanskii (modified
Newton) methods, termination on the residual. None is in the library (source to find).

Examples
--------
Scalar (``n = 1``), finite-difference Jacobian, default options:

>>> import numpy as np
>>> from numerical_solvers import NewtonOptions, newton
>>> r = newton(lambda x: x**2 - 2.0, [1.0], name="sqrt2", label="demo")
>>> bool(r.converged), round(float(r.x[0]), 10), r.reason
(True, 1.4142135624, 'residual_abs')

Small system with an analytic Jacobian (circle ``x^2 + y^2 = 4`` and parabola ``y = x^2 - 1``):

>>> F = lambda v: np.array([v[0]**2 + v[1]**2 - 4.0, v[1] - v[0]**2 + 1.0])
>>> J = lambda v: np.array([[2*v[0], 2*v[1]], [-2*v[0], 1.0]])
>>> r = newton(F, [1.0, 1.0], jacobian=J, name="circle_parabola")
>>> np.round(r.x, 6)
array([1.51749 , 1.302776])

Central finite-difference Jacobian, line search and mixed units: a displacement ``u`` [mm] and
a rotation ``th`` [rad] of a rigid body on two hardening springs, residuals a force [N] and a
moment [N mm]:

>>> def residual(x):
...     u, th = x
...     d1, d2 = u - 50.0 * th, u + 50.0 * th                 # spring deflections [mm]
...     q1, q2 = 1e4 * d1 + 1e7 * d1**3, 1e4 * d2 + 1e7 * d2**3   # spring forces [N]
...     return np.array([q1 + q2 - 2000.0, 50.0 * (q2 - q1) - 1.0e4])
>>> opts = NewtonOptions(jacobian_fd="central", x_scale=[1e-2, 1e-4], f_scale=[1e3, 1e5])
>>> r = newton(residual, [0.0, 0.0], options=opts, name="rigid_body", label="F 2 kN, M 10 N m")
>>> bool(r.converged), float(np.max(np.abs(r.F / [1e3, 1e5]))) < 1e-10
(True, True)
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import root
from scipy.sparse import issparse

from .newton_results import IterationRecord, NewtonResult

__all__ = ["NewtonOptions", "NewtonError", "newton"]

_EPS = np.finfo(float).eps
_GLOBALISATIONS = ("none", "damped", "line_search")
_CRITERIA = ("residual_abs", "residual_rel", "step_abs", "step_rel")
_COND_MAX_N = 200          # condition number estimated (SVD) only up to this size


@dataclass(frozen=True, kw_only=True)
class NewtonOptions:
    """Options of ``newton``. Frozen; built once and stored on the ``NewtonResult``.

    Parameters
    ----------
    residual_abs : float or None
        Stop when ``||F / f_scale|| <= residual_abs`` [-]. None: not used. Default 1e-10. In
        the units of ``F / f_scale``: with ``f_scale = 1`` and residuals of order 1e6 (N mm),
        round-off alone keeps ``||F||`` near 1e-9, so set ``f_scale`` for such problems.
    residual_rel : float or None
        Stop when ``||F / f_scale|| <= residual_rel * ||F(x0) / f_scale||`` [-]. Default None.
    step_abs : float or None
        Stop when the scaled step taken ``||dx / x_scale|| <= step_abs`` [-]. Default None.
    step_rel : float or None
        Stop when ``||dx / x_scale|| <= step_rel * ||x / x_scale||`` [-]. Default 1e-12.
    combine : {"any", "all"}
        ``"any"``: stop when one of the criteria in use holds; ``"all"``: when all of them hold.
        The step criteria never hold at ``k = 0``, nor for a step shortened by the line search
        (``alpha < 1``), so a stalled line search is not taken for convergence. Default
        ``"any"``.
    norm : {"inf", 2} or callable
        Norm of the scaled vectors in the stopping tests, ``max_step`` and the history; a
        callable takes a 1-D array and returns a float. The line search always uses the 2-norm.
        Default 2.
    x_scale : float or array_like of shape (n,)
        Typical magnitude of each unknown [units of x] (for example 1e-2 mm and 1e-4 rad).
        Positive. Default 1.0.
    f_scale : float or array_like of shape (n,)
        Typical magnitude of each residual [units of F] (for example 1e3 N and 1e5 N mm).
        Every norm is taken of ``x / x_scale`` and ``F / f_scale``, so mixed units are compared
        as numbers of typical magnitudes. Positive. Default 1.0.
    max_iter : int
        Maximum number of Newton steps [-]. Default 50.
    max_residual_evaluations : int or None
        Maximum residual evaluations by Newton, finite differences included; the fallback is
        not counted against it [-]. None: no limit. Default 1000.
    globalisation : {"none", "damped", "line_search"}
        Full Newton step, fixed fraction ``damping`` of it, or backtracking Armijo line search
        on ``0.5 ||F / f_scale||_2^2``. Default ``"line_search"``.
    damping : float
        Step fraction of ``"damped"``, in (0, 1] [-]. Default 0.5.
    c1 : float
        Armijo constant, in (0, 0.5) [-]. Default 1e-4.
    shrink : float
        Backtracking factor ``alpha <- shrink * alpha``, in (0, 1) [-]. Default 0.5.
    min_step : float
        Smallest ``alpha`` tried; below it the line search fails [-]. Default 1e-10.
    max_step : float or None
        Cap on the scaled norm ``||p / x_scale||`` of the Newton step [-]. None: no cap.
        Default None.
    jacobian_fd : {"forward", "central"}
        Finite-difference formula when no analytic Jacobian is given. Default ``"forward"``.
    fd_rel_step : float or None
        Relative step ``h_j = fd_rel_step * max(|x_j|, x_scale_j)`` [-]. None: ``sqrt(eps)``
        (forward) or ``eps**(1/3)`` (central). Default None.
    jacobian_update : int
        Evaluate the Jacobian when it is ``jacobian_update`` iterations old; 1 is Newton, more
        is modified Newton [-]. A step from an old Jacobian that is singular or finds no
        acceptable point is retried once with a new Jacobian. Default 1.
    linear_solver : callable
        ``linear_solver(J, rhs) -> dx``, solving ``J dx = rhs`` with ``rhs = -F``. Must raise
        ``numpy.linalg.LinAlgError`` or return non-finite values for a singular ``J``. Default
        ``numpy.linalg.solve``; a sparse solver may be passed together with a sparse ``J``.
    max_condition : float or None
        A scaled Jacobian ``diag(1/f_scale) J diag(x_scale)`` with a 2-norm condition number
        above this is treated as singular [-]. Estimated (SVD) only for a dense ``J`` with
        ``n <= 200``. None: not estimated. Default ``1 / eps`` (about 4.5e15).
    project : callable or None
        ``project(x) -> x`` applied to ``x0`` and to every trial point (bounds, sign
        constraints). Not applied to the finite-difference points. With an active bound the
        Armijo test is a heuristic and may stall; a bound that is part of the physics (contact
        only for ``delta > 0``) is better written into the residual. Default None.
    fallback : {None, "hybr", "lm"}
        SciPy ``root`` method run, in the scaled variables, from the last iterate when Newton
        fails (not after a callback stop). Default None.
    on_failure : {"return", "raise"}
        Return the result with ``converged=False`` (also after a callback stop), or raise
        ``NewtonError`` carrying it. Default ``"return"``.

    Raises
    ------
    ValueError
        At construction, if an option is out of range (see ``validate``).

    Notes
    -----
    A result whose options hold a lambda (``project``, ``norm``, ``linear_solver``) cannot be
    pickled; use module-level functions when results cross processes.
    """

    residual_abs: float | None = 1e-10
    residual_rel: float | None = None
    step_abs: float | None = None
    step_rel: float | None = 1e-12
    combine: str = "any"
    norm: str | int | Callable[[np.ndarray], float] = 2
    x_scale: float | Sequence[float] = 1.0
    f_scale: float | Sequence[float] = 1.0
    max_iter: int = 50
    max_residual_evaluations: int | None = 1000
    globalisation: str = "line_search"
    damping: float = 0.5
    c1: float = 1e-4
    shrink: float = 0.5
    min_step: float = 1e-10
    max_step: float | None = None
    jacobian_fd: str = "forward"
    fd_rel_step: float | None = None
    jacobian_update: int = 1
    linear_solver: Callable = field(default=np.linalg.solve)
    max_condition: float | None = 1.0 / _EPS
    project: Callable[[np.ndarray], np.ndarray] | None = None
    fallback: str | None = None
    on_failure: str = "return"

    def __post_init__(self) -> None:
        """Fail fast: see ``validate``."""
        self.validate_or_raise()

    def validate(self) -> list[str]:
        """Return the list of error messages; empty if the options are consistent."""
        e: list[str] = []
        tols = {c: getattr(self, c) for c in _CRITERIA}
        if all(v is None for v in tols.values()):
            e.append("NewtonOptions: at least one stopping tolerance must be set.")
        e += [f"NewtonOptions: {c} must be >= 0, got {v}"
              for c, v in tols.items() if v is not None and not v >= 0]
        if self.combine not in ("any", "all"):
            e.append(f"NewtonOptions: combine must be 'any' or 'all', got {self.combine!r}")
        if not (callable(self.norm) or self.norm in ("inf", 2)):
            e.append(f"NewtonOptions: norm must be 'inf', 2 or a callable, got {self.norm!r}")
        for nm in ("x_scale", "f_scale"):
            s = np.asarray(getattr(self, nm), dtype=float)
            if s.ndim > 1 or not np.all(np.isfinite(s)) or not np.all(s > 0):
                e.append(f"NewtonOptions: {nm} must be positive and finite (scalar or 1-D).")
        if self.max_iter < 1:
            e.append(f"NewtonOptions: max_iter must be >= 1, got {self.max_iter}")
        if self.max_residual_evaluations is not None and self.max_residual_evaluations < 1:
            e.append("NewtonOptions: max_residual_evaluations must be >= 1 or None.")
        if self.globalisation not in _GLOBALISATIONS:
            e.append(f"NewtonOptions: globalisation must be one of {_GLOBALISATIONS}, "
                     f"got {self.globalisation!r}")
        if not 0.0 < self.damping <= 1.0:
            e.append(f"NewtonOptions: damping must be in (0, 1], got {self.damping}")
        if not 0.0 < self.c1 < 0.5:
            e.append(f"NewtonOptions: c1 must be in (0, 0.5), got {self.c1}")
        if not 0.0 < self.shrink < 1.0:
            e.append(f"NewtonOptions: shrink must be in (0, 1), got {self.shrink}")
        if not 0.0 < self.min_step <= 1.0:
            e.append(f"NewtonOptions: min_step must be in (0, 1], got {self.min_step}")
        if self.max_step is not None and not self.max_step > 0:
            e.append(f"NewtonOptions: max_step must be > 0 or None, got {self.max_step}")
        if self.jacobian_fd not in ("forward", "central"):
            e.append(f"NewtonOptions: jacobian_fd must be 'forward' or 'central', "
                     f"got {self.jacobian_fd!r}")
        if self.fd_rel_step is not None and not self.fd_rel_step > 0:
            e.append(f"NewtonOptions: fd_rel_step must be > 0 or None, got {self.fd_rel_step}")
        if self.jacobian_update < 1:
            e.append(f"NewtonOptions: jacobian_update must be >= 1, got {self.jacobian_update}")
        if self.max_condition is not None and not self.max_condition > 1:
            e.append("NewtonOptions: max_condition must be > 1 or None.")
        if self.fallback not in (None, "hybr", "lm"):
            e.append(f"NewtonOptions: fallback must be None, 'hybr' or 'lm', "
                     f"got {self.fallback!r}")
        if self.on_failure not in ("return", "raise"):
            e.append(f"NewtonOptions: on_failure must be 'return' or 'raise', "
                     f"got {self.on_failure!r}")
        return e

    def validate_or_raise(self) -> None:
        """Raise ValueError with all messages of ``validate()`` if there are any."""
        errors = self.validate()
        if errors:
            raise ValueError("\n".join(errors))


class NewtonError(RuntimeError):
    """Raised by ``newton`` with ``on_failure="raise"`` when the solve does not converge.

    Parameters
    ----------
    result : NewtonResult
        The unconverged result, available as ``error.result``.
    """

    def __init__(self, result: NewtonResult):
        """Store the result; the message names the analysis and the reason."""
        super().__init__(f"newton {result.name!r} {result.label!r} did not converge: "
                         f"{result.reason}")
        self.result = result


class _Budget(Exception):
    """Internal: the residual-evaluation budget is spent."""


def newton(residual: Callable[[np.ndarray], np.ndarray], x0: Sequence[float], *,
           jacobian: Callable[[np.ndarray], np.ndarray] | None = None,
           options: NewtonOptions | None = None,
           callback: Callable[[IterationRecord], bool | None] | None = None,
           name: str = "", label: str = "") -> NewtonResult:
    """Solve ``F(x) = 0`` by Newton-Raphson with optional globalisation and fallback.

    Parameters
    ----------
    residual : callable
        ``residual(x) -> array_like`` of shape (n,) [units of F]; zero at the solution. Called
        with a float array of shape (n,) (``n = 1`` for a scalar problem).
    x0 : array_like of shape (n,) or float
        Initial point [units of x]; flattened to shape (n,).
    jacobian : callable or None
        ``jacobian(x) -> (n, n)`` array, ``J_ij = dF_i / dx_j`` [units of F per unit of x], or a
        sparse matrix accepted by ``options.linear_solver``. A dense result is converted with
        ``numpy.atleast_2d`` (a scalar is accepted for ``n = 1``). None: finite differences per
        ``options.jacobian_fd``.
    options : NewtonOptions or None
        None: ``NewtonOptions()``.
    callback : callable or None
        ``callback(record)`` after each iteration ``k >= 1`` with its ``IterationRecord``;
        returning True stops the solve with reason ``"callback"`` (unless it has converged),
        which counts as not converged.
    name, label : str
        Identification of the analysis, stored on the result and copied into every row of
        ``NewtonResult.records()``.

    Returns
    -------
    NewtonResult
        Last iterate, residual, convergence flag and reason, counts and the history.

    Raises
    ------
    ValueError
        If a scale does not have shape (n,) or ``()``, or if ``residual`` or ``jacobian``
        returns the wrong size.
    NewtonError
        If the solve does not converge (callback stop included) and
        ``options.on_failure == "raise"``.

    Notes
    -----
    Scaling enters every norm, the finite-difference step, the condition estimate and
    ``max_step``; the Newton step itself does not depend on it. A non-finite residual at a
    trial point is a failed Armijo test (the step is shortened) with ``"line_search"`` and
    stops the solve otherwise. The step tests use the step actually taken,
    ``x_{k+1} - x_k``. When no fallback runs, the last call of ``residual`` is at the returned
    ``x``, so a caller may cache the state of an expensive residual (a FEM solve) in a closure.
    """
    opt = NewtonOptions() if options is None else options
    x = np.array(x0, dtype=float).reshape(-1)
    n = x.size
    xs = _scale(opt.x_scale, n, "x_scale")
    fs = _scale(opt.f_scale, n, "f_scale")
    nrm = _norm_function(opt.norm)
    count = {"F": 0, "J": 0}

    def F(v: np.ndarray, budget: bool = True) -> np.ndarray:
        if budget and opt.max_residual_evaluations is not None \
                and count["F"] >= opt.max_residual_evaluations:
            raise _Budget
        count["F"] += 1
        out = np.asarray(residual(v), dtype=float).reshape(-1)
        if out.size != n:
            raise ValueError(f"newton {name!r}: residual returned {out.size} values, "
                             f"expected {n}.")
        return out

    def J(v: np.ndarray, Fv: np.ndarray):
        count["J"] += 1
        if jacobian is None:
            return _fd_jacobian(F, v, Fv, xs, opt.jacobian_fd, opt.fd_rel_step)
        Jv = jacobian(v.copy())
        if issparse(Jv):
            return Jv
        Jv = np.atleast_2d(np.asarray(Jv, dtype=float))
        if Jv.shape != (n, n):
            raise ValueError(f"newton {name!r}: jacobian returned shape {Jv.shape}, "
                             f"expected {(n, n)}.")
        return Jv

    if opt.project is not None:
        x = np.asarray(opt.project(x), dtype=float).reshape(-1)
    Fx = F(x)
    r0 = nrm(Fx / fs)
    history = [IterationRecord(k=0, residual_norm=r0, relative_residual=_ratio(r0, r0),
                               n_residual_evaluations=count["F"])]
    converged = False
    reason = "non_finite_residual" if not np.all(np.isfinite(Fx)) else None
    if reason is None:
        converged, reason = _stop_test(opt, nrm, Fx / fs, r0, None, x / xs)
    Jx, age, k = None, 0, 0
    while reason is None:
        if k >= opt.max_iter:
            reason = "max_iter"
            break
        fresh = Jx is None or age >= opt.jacobian_update
        step = alpha = None
        try:
            while True:                      # at most twice: an old Jacobian is retried fresh
                if fresh:
                    Jx, age = J(x, Fx), 0
                step = _newton_step(opt, Jx, Fx, xs, fs, nrm)
                if step is not None:
                    p, t, cond = step
                    alpha, x_new, F_new = _globalise(opt, F, x, Fx, p, t, fs)
                if alpha is not None or fresh:
                    break
                fresh = True
        except _Budget:
            reason = "max_residual_evaluations"
            break
        if step is None:
            reason = "singular_jacobian"
            break
        if alpha is None:
            reason = "line_search_failed" if opt.globalisation == "line_search" \
                else "non_finite_residual"
            break
        k, age = k + 1, age + 1
        dx = x_new - x
        x, Fx = x_new, F_new
        rk = nrm(Fx / fs)
        rec = IterationRecord(k=k, residual_norm=rk, relative_residual=_ratio(rk, r0),
                              step_norm=nrm(dx / xs), step_length=alpha * t, condition=cond,
                              jacobian_evaluated=fresh, n_residual_evaluations=count["F"])
        history.append(rec)
        shortened = opt.globalisation == "line_search" and alpha < 1.0
        converged, reason = _stop_test(opt, nrm, Fx / fs, r0, None if shortened else dx / xs,
                                       x / xs)
        if reason is None and callback is not None and callback(rec):
            reason = "callback"

    newton_reason = reason
    used = None
    if not converged and reason != "callback" and opt.fallback is not None:
        used = opt.fallback
        jac = None if jacobian is None else \
            (lambda z: _dense(jacobian(z * xs)) * xs[None, :] / fs[:, None])
        sol = root(lambda z: F(z * xs, budget=False) / fs, x / xs, method=used, jac=jac)
        x = np.atleast_1d(np.asarray(sol.x, dtype=float)) * xs
        Fx = np.atleast_1d(np.asarray(sol.fun, dtype=float)) * fs
        converged = bool(sol.success) and bool(np.all(np.isfinite(Fx)))
        if converged and (opt.residual_abs is not None or opt.residual_rel is not None):
            converged = _stop_test(opt, nrm, Fx / fs, r0, None, x / xs)[0]
        reason = f"fallback_{used}" if converged else f"fallback_{used}_failed"

    result = NewtonResult(name=name, label=label, x=x, F=Fx, converged=converged,
                          reason=reason, newton_reason=newton_reason, fallback=used,
                          residual_norm=nrm(Fx / fs), n_iter=k,
                          n_residual_evaluations=count["F"],
                          n_jacobian_evaluations=count["J"], history=tuple(history),
                          options=opt)
    if not converged and opt.on_failure == "raise":
        raise NewtonError(result)
    return result


def _scale(s, n: int, nm: str) -> np.ndarray:
    """Broadcast a scale (scalar or (n,)) to shape (n,)."""
    a = np.asarray(s, dtype=float)
    if a.ndim == 0:
        return np.full(n, float(a))
    if a.shape != (n,):
        raise ValueError(f"newton: {nm} has shape {a.shape}, expected ({n},).")
    return a


def _dense(Jv) -> np.ndarray:
    """A Jacobian as a dense 2-D float array (for SciPy ``root``)."""
    return np.atleast_2d(np.asarray(Jv.toarray() if issparse(Jv) else Jv, dtype=float))


def _norm_function(norm) -> Callable[[np.ndarray], float]:
    """Return the norm of the options as a function of a 1-D array."""
    if callable(norm):
        return lambda v: float(norm(v))
    if norm == "inf":
        return lambda v: float(np.max(np.abs(v)))

    def norm2(v: np.ndarray) -> float:             # scaled first: no overflow of sum(v**2)
        m = float(np.max(np.abs(v)))
        return m * float(np.linalg.norm(v / m)) if 0.0 < m < np.inf else m
    return norm2


def _ratio(a: float, b: float) -> float:
    """``a / b``, NaN when ``b`` is 0 or not finite."""
    return a / b if b > 0 and np.isfinite(b) else float("nan")


def _stop_test(opt: NewtonOptions, nrm, Fs: np.ndarray, r0: float,
               s: np.ndarray | None, xsc: np.ndarray) -> tuple[bool, str | None]:
    """Apply the stopping criteria in use.

    ``Fs`` scaled residual, ``r0`` initial scaled residual norm, ``s`` scaled step taken (None
    when the step tests cannot hold: ``k = 0`` or a shortened line-search step), ``xsc``
    scaled iterate. Returns ``(True, criteria met joined by "+")`` or ``(False, None)``.
    """
    r = nrm(Fs)
    held = {
        "residual_abs": r <= opt.residual_abs if opt.residual_abs is not None else None,
        "residual_rel": r <= opt.residual_rel * r0 if opt.residual_rel is not None else None,
        "step_abs": (s is not None and nrm(s) <= opt.step_abs)
        if opt.step_abs is not None else None,
        "step_rel": (s is not None and nrm(s) <= opt.step_rel * nrm(xsc) < np.inf)
        if opt.step_rel is not None else None,
    }
    used = {c: h for c, h in held.items() if h is not None}
    met = [c for c, h in used.items() if h]
    done = (len(met) > 0) if opt.combine == "any" else (len(met) == len(used))
    return (True, "+".join(met)) if done else (False, None)


def _fd_jacobian(F, x: np.ndarray, Fx: np.ndarray, xs: np.ndarray, kind: str,
                 rel: float | None) -> np.ndarray:
    """Finite-difference Jacobian, column by column.

    ``h_j = rel * max(|x_j|, xs_j)`` with the sign of ``x_j``, and ``rel`` = ``sqrt(eps)``
    (forward, error O(h)) or ``eps**(1/3)`` (central, error O(h^2)) by default, as in Dennis
    and Schnabel (1996) (not in the library, source to find). ``h_j`` is rounded so that
    ``x_j + h_j`` is exactly representable. Written here because SciPy's public
    ``approx_fprime`` has forward differences and absolute steps only.
    """
    n = x.size
    if rel is None:
        rel = np.sqrt(_EPS) if kind == "forward" else _EPS ** (1.0 / 3.0)
    Jm = np.empty((n, n))
    for j in range(n):
        h = rel * max(abs(x[j]), xs[j]) * (1.0 if x[j] >= 0 else -1.0)
        xp = x.copy()
        xp[j] += h
        h = xp[j] - x[j]
        if kind == "forward":
            Jm[:, j] = (F(xp) - Fx) / h
        else:
            xm = x.copy()
            xm[j] -= h
            Jm[:, j] = (F(xp) - F(xm)) / (2.0 * h)
    return Jm


def _newton_step(opt: NewtonOptions, Jx, Fx: np.ndarray, xs: np.ndarray, fs: np.ndarray,
                 nrm) -> tuple[np.ndarray, float, float] | None:
    """Solve ``J p = -F``, check conditioning and apply ``max_step``.

    Returns ``(p, t, cond)``: the Newton step ``p`` already multiplied by the cap factor ``t``
    (1 if not capped), and the condition number of the scaled Jacobian
    ``diag(1/f_scale) J diag(x_scale)`` (NaN if not estimated), which does not depend on the
    units of the caller. None if ``J`` is singular.
    """
    cond = float("nan")
    if isinstance(Jx, np.ndarray) and opt.max_condition is not None and Jx.shape[0] <= _COND_MAX_N:
        if not np.all(np.isfinite(Jx)):
            return None
        cond = float(np.linalg.cond(Jx * xs[None, :] / fs[:, None]))
        if not cond <= opt.max_condition:
            return None
    try:
        p = np.asarray(opt.linear_solver(Jx, -Fx), dtype=float).reshape(-1)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(p)):
        return None
    t = 1.0
    if opt.max_step is not None:
        ps = nrm(p / xs)
        if ps > opt.max_step:
            t = opt.max_step / ps
    return t * p, t, cond


def _globalise(opt: NewtonOptions, F, x: np.ndarray, Fx: np.ndarray, p: np.ndarray, t: float,
               fs: np.ndarray) -> tuple[float | None, np.ndarray | None, np.ndarray | None]:
    """Choose the step length ``alpha`` along ``p`` and return ``(alpha, x_new, F_new)``.

    ``alpha`` is None if no acceptable point was found (line search below ``min_step``, or a
    non-finite residual without line search). ``p`` is the Newton step times the cap factor
    ``t``. Armijo condition on the merit function ``f(x) = 0.5 ||F(x) / f_scale||_2^2``: along
    the Newton direction the directional derivative of ``f`` is ``-2 f(x)``, so along ``p`` it
    is ``-2 t f(x)`` and the test is ``f(x + alpha p) <= (1 - 2 c1 alpha t) f(x)`` (Nocedal and
    Wright (2006), Dennis and Schnabel (1996); not in the library, source to find). A
    modified-Newton or projected step is not exactly the Newton direction, so the test is then
    a heuristic.
    """
    def trial(a: float) -> tuple[np.ndarray, np.ndarray]:
        xt = x + a * p
        if opt.project is not None:
            xt = np.asarray(opt.project(xt), dtype=float).reshape(-1)
        return xt, F(xt)

    if opt.globalisation != "line_search":
        a = 1.0 if opt.globalisation == "none" else opt.damping
        xt, Ft = trial(a)
        return (a, xt, Ft) if np.all(np.isfinite(Ft)) else (None, None, None)

    f0 = 0.5 * float(np.sum((Fx / fs) ** 2))
    a = 1.0
    while a >= opt.min_step:
        xt, Ft = trial(a)
        if np.all(np.isfinite(Ft)) and 0.5 * float(np.sum((Ft / fs) ** 2)) <= \
                (1.0 - 2.0 * opt.c1 * a * t) * f0:
            return a, xt, Ft
        a *= opt.shrink
    return None, None, None
