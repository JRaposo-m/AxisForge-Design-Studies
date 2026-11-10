"""Tests of the generic Newton-Raphson solver ``numerical_solvers.newton_raphson.newton``.

Ported from AxisForge ``axisforge/solvers/numerics/tests/test_newton.py`` at d34ec7f (35 tests,
unchanged in substance; ``test_reexports`` checks this package's rollups), plus
``test_port_matches_axisforge``, which runs the same problems through both solvers.

Groups (José's list, N0): 1 scalar order of convergence; 2 linear system in one step; 3 2x2
system, analytic and finite-difference Jacobians; 4 arctan, full step against line search;
5 stopping criteria alone and combined; 6 singular Jacobian; 7 scaling (mm against m, N against
kN); 8 project, callback, max_step, modified Newton, damping, stalled line search; 9 fallback
to SciPy ``root``; 10 two Hertz-type contacts in series with a closed form. Then the reasons
not reached above, the evaluation contract and the package re-exports.

Every solve goes through ``_solve``, which passes a ``name`` and a ``label`` and checks that
they reach the result, its summary and every history row. Units are stated per test; the
tolerances are in scaled units (``F / f_scale``, ``dx / x_scale``) [-]. Method sources:
Dennis and Schnabel (1996), Nocedal and Wright (2006), Kelley (2003); none is in the library
(source to find).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import numerical_solvers
from numerical_solvers import IterationRecord, NewtonError, NewtonOptions, NewtonResult, newton

_SUMMARY_KEYS = {"name", "label", "converged", "reason", "newton_reason", "fallback", "n_iter",
                 "n_residual_evaluations", "n_jacobian_evaluations", "residual_norm"}
_ROW_KEYS = {"name", "label", "k", "residual_norm", "relative_residual", "step_norm",
             "step_length", "condition", "jacobian_evaluated", "n_residual_evaluations"}
_NONE = dict(residual_abs=None, residual_rel=None, step_abs=None, step_rel=None)
"""All four stopping tests off; a test switches on the ones it needs (at least one)."""


def _check_ids(r: NewtonResult, name: str, label: str) -> None:
    """Name and label on the result, its summary and every row; the row columns are fixed."""
    assert isinstance(r, NewtonResult)
    assert (r.name, r.label) == (name, label)
    summary = r.record()
    assert set(summary) == _SUMMARY_KEYS
    assert (summary["name"], summary["label"]) == (name, label)
    rows = r.records()
    assert len(rows) == len(r.history) == r.n_iter + 1
    assert all(set(row) == _ROW_KEYS for row in rows)
    assert all(row["name"] == name and row["label"] == label for row in rows)
    assert rows[0]["k"] == 0 and math.isnan(rows[0]["step_norm"])
    counts = [row["n_residual_evaluations"] for row in rows]
    assert counts == sorted(counts)


def _solve(residual, x0, name: str, label: str, **kwargs) -> NewtonResult:
    """Run ``newton`` with a name and a label and check that they reach the records."""
    r = newton(residual, x0, name=name, label=label, **kwargs)
    _check_ids(r, name, label)
    return r


# ---- 1. scalar x^2 - 2 ------------------------------------------------------------------
@pytest.mark.parametrize("jac", [lambda x: np.array([[2.0 * x[0]]]), lambda x: 2.0 * x[0]],
                         ids=["array J", "scalar J"])
def test_scalar_sqrt2_quadratic_order(jac):
    """Newton on x^2 - 2 from 1 reaches sqrt(2) with order 2 (a scalar J is accepted too)."""
    r = _solve(lambda x: x**2 - 2.0, 1.0, "sqrt2", "t1", jacobian=jac,
               options=NewtonOptions(residual_abs=1e-14, step_rel=None))
    assert r.converged and r.reason == "residual_abs"
    assert r.x.shape == (1,) and r.x[0] == pytest.approx(math.sqrt(2.0), abs=1e-14)
    # Near the root |F| = 2 sqrt(2) |e|, so the residual norms stand for the error. Observed
    # order p = log(r3/r2) / log(r2/r1) on the last three residuals above round-off [-].
    res = [h.residual_norm for h in r.history if h.residual_norm > 1e-13]
    assert math.log(res[-1] / res[-2]) / math.log(res[-2] / res[-3]) == pytest.approx(2.0, abs=0.1)
    assert all(h.step_length == 1.0 for h in r.history[1:])


# ---- 2. linear system --------------------------------------------------------------------
def test_linear_system_one_iteration():
    """With the exact Jacobian A, one Newton step solves A x = b."""
    A = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, -1.0], [0.0, -1.0, 2.0]])
    b = np.array([1.0, 2.0, 3.0])
    r = _solve(lambda x: A @ x - b, np.zeros(3), "linear", "t2", jacobian=lambda x: A)
    assert r.converged and r.n_iter == 1 and r.n_jacobian_evaluations == 1
    np.testing.assert_allclose(r.x, np.linalg.solve(A, b), rtol=1e-14)
    assert r.history[1].condition == pytest.approx(np.linalg.cond(A))


# ---- 3. 2x2 nonlinear: circle x^2 + y^2 = 4 and parabola y = x^2 - 1 ---------------------
_U = (1.0 + math.sqrt(13.0)) / 2.0            # x^2 at the intersection: u^2 - u - 3 = 0
ROOT_3 = np.array([math.sqrt(_U), _U - 1.0])


def _F3(v):
    return np.array([v[0]**2 + v[1]**2 - 4.0, v[1] - v[0]**2 + 1.0])


def _J3(v):
    return np.array([[2.0 * v[0], 2.0 * v[1]], [-2.0 * v[0], 1.0]])


@pytest.mark.parametrize("jac, fd, per_iter", [(_J3, "forward", 1), (None, "forward", 3),
                                               (None, "central", 5)],
                         ids=["analytic", "forward", "central"])
def test_circle_parabola_analytic_and_fd(jac, fd, per_iter):
    """Analytic and finite-difference Jacobians reach the same root; FD costs n or 2n calls."""
    r = _solve(_F3, [1.0, 1.0], "circle_parabola", f"t3 {fd}", jacobian=jac,
               options=NewtonOptions(jacobian_fd=fd))
    assert r.converged
    np.testing.assert_allclose(r.x, ROOT_3, rtol=1e-11)
    # every step is a full step (one trial): per iteration 1 trial + n (forward) or 2n
    # (central) FD evaluations, here n = 2
    assert all(h.step_length == 1.0 for h in r.history[1:])
    assert r.n_residual_evaluations == 1 + r.n_iter * per_iter


def test_fd_central_more_accurate_than_forward():
    """On a non-polynomial residual the first step with central FD is closer to the exact one."""
    F = lambda v: np.array([np.exp(v[0]) - 2.0 + np.sin(v[1]), v[0] * v[1] - 0.5])
    J = lambda v: np.array([[np.exp(v[0]), np.cos(v[1])], [v[1], v[0]]])
    one = dict(max_iter=1, globalisation="none")
    exact = _solve(F, [0.3, 0.8], "fd_accuracy", "t3 analytic", jacobian=J,
                   options=NewtonOptions(**one)).x
    err = {fd: np.max(np.abs(_solve(F, [0.3, 0.8], "fd_accuracy", f"t3 {fd}",
                                    options=NewtonOptions(**one, jacobian_fd=fd)).x - exact))
           for fd in ("forward", "central")}
    assert err["central"] < 1e-9 < err["forward"] < 1e-6


# ---- 4. arctan from 1.5: full Newton diverges, line search converges ---------------------
def test_arctan_none_fails_line_search_converges():
    """From 1.5 the full Newton step overshoots (x1 = -1.69) and diverges; Armijo saves it."""
    J = lambda x: np.array([[1.0 / (1.0 + x[0] ** 2)]])
    with np.errstate(over="ignore"):              # x**2 overflows once the iterate is huge
        bad = _solve(np.arctan, [1.5], "arctan", "t4 none", jacobian=J,
                     options=NewtonOptions(globalisation="none"))
    assert not bad.converged
    assert bad.reason == "singular_jacobian"           # J = 1/(1 + x^2) becomes 0
    assert abs(bad.x[0]) > 1e100
    res = [h.residual_norm for h in bad.history]
    assert res[1] > res[0] and res[2] > res[1]        # the residual grows: divergence

    good = _solve(np.arctan, [1.5], "arctan", "t4 line_search", jacobian=J,
                  options=NewtonOptions(globalisation="line_search"))
    assert good.converged and abs(good.x[0]) < 1e-10
    assert good.history[1].step_length == 0.5          # the first step was halved
    assert all(h.step_length == 1.0 for h in good.history[2:])   # full steps near the root


# ---- 5. each stopping criterion alone, and "any" / "all" ---------------------------------
# Tolerances chosen so that each criterion stops at its own iteration on _F3 from (1, 1).
@pytest.mark.parametrize("crit, tol, k_stop", [("residual_rel", 1e-3, 3), ("residual_abs", 1e-8, 4),
                                               ("step_rel", 1e-3, 4), ("step_abs", 1e-8, 5)])
def test_each_criterion_alone(crit, tol, k_stop):
    """Each criterion alone stops the run at the first iterate that meets it."""
    r = _solve(_F3, [1.0, 1.0], "criterion", f"t5 {crit}", jacobian=_J3,
               options=NewtonOptions(**{**_NONE, crit: tol}))
    assert r.converged and r.reason == crit and r.n_iter == k_stop

    def value(h):
        return {"residual_abs": h.residual_norm, "residual_rel": h.relative_residual,
                "step_abs": h.step_norm,
                "step_rel": h.step_norm / np.linalg.norm(ROOT_3)}[crit]
    assert value(r.history[-1]) <= tol
    assert not value(r.history[-2]) <= tol              # not met one iteration earlier


def test_any_and_all():
    """"any" stops at the first criterion met, "all" when both are met."""
    both = dict(_NONE, residual_abs=1e-14, step_abs=1e-2)
    r_any = _solve(_F3, [1.0, 1.0], "combine", "t5 any", jacobian=_J3,
                   options=NewtonOptions(**both, combine="any"))
    r_all = _solve(_F3, [1.0, 1.0], "combine", "t5 all", jacobian=_J3,
                   options=NewtonOptions(**both, combine="all"))
    assert r_any.reason == "step_abs" and r_all.reason == "residual_abs+step_abs"
    assert r_any.n_iter < r_all.n_iter


def test_converged_at_start_takes_no_step():
    """At the root (|F| about 1e-16 < 1e-6) the run stops at k = 0 with no Jacobian."""
    r = _solve(_F3, ROOT_3, "start", "t5 k0", jacobian=_J3,
               options=NewtonOptions(residual_abs=1e-6))
    assert r.converged and r.n_iter == 0 and r.n_jacobian_evaluations == 0


# ---- 6. singular Jacobian ----------------------------------------------------------------
@pytest.mark.parametrize("max_condition", [1.0 / np.finfo(float).eps, None],
                         ids=["condition test", "LinAlgError"])
@pytest.mark.parametrize("F, J, x0", [
    (lambda x: x**2 + 1.0, lambda x: np.array([[2.0 * x[0]]]), [0.0]),          # J(0) = 0
    # rank 1: second row = 2 x first row, right sides 1 and 3 inconsistent (no solution)
    (lambda v: np.array([v[0] + v[1] - 1.0, 2 * v[0] + 2 * v[1] - 3.0]),
     lambda v: np.array([[1.0, 1.0], [2.0, 2.0]]), [0.0, 0.0]),
], ids=["J=0", "rank 1"])
def test_singular_jacobian(F, J, x0, max_condition):
    """A singular J is reported, not raised, unless on_failure="raise"."""
    opts = dict(max_condition=max_condition)
    r = _solve(F, x0, "singular", "t6 return", jacobian=J, options=NewtonOptions(**opts))
    assert not r.converged and r.reason == "singular_jacobian" and r.n_iter == 0
    with pytest.raises(NewtonError, match="'singular' 't6 raise'") as err:
        newton(F, x0, jacobian=J, options=NewtonOptions(**opts, on_failure="raise"),
               name="singular", label="t6 raise")
    _check_ids(err.value.result, "singular", "t6 raise")
    assert err.value.result.reason == "singular_jacobian"


# ---- 7. scaling: same problem in mm and in m, and in N and kN ----------------------------
def _spring(unit: float, force: float = 1.0):
    """Hardening spring k x + c x^3 = P (k 2e5 N/m, c 4e13 N/m^3, P 1 kN).

    ``unit``: metres per unit of x (1e-3 for mm); ``force``: newtons per unit of F (1e3 for
    kN). The residual is in units of ``force``.
    """
    k, c, P = 2.0e5, 4.0e13, 1.0e3
    return lambda x: np.array([(k * (x[0] * unit) + c * (x[0] * unit) ** 3 - P) / force])


_ROOT_7_MM = 1e3 * float(np.real(np.roots([4.0e13, 0.0, 2.0e5, -1.0e3])[-1]))   # [mm]


def test_scaling_mm_vs_m_same_iterations():
    """With x_scale = 1e-3 m the test in m is the test in mm; without it, it is 1000x looser."""
    common = dict(_NONE, step_abs=1e-4)               # scaled step [-]
    r_mm = _solve(_spring(1e-3), [1.0], "spring", "t7 mm",
                  options=NewtonOptions(**common, x_scale=1.0))
    r_m = _solve(_spring(1.0), [1e-3], "spring", "t7 m",
                 options=NewtonOptions(**common, x_scale=1e-3))
    r_m_unscaled = _solve(_spring(1.0), [1e-3], "spring", "t7 m unscaled",
                          options=NewtonOptions(**common))
    assert r_mm.converged and r_m.converged and r_mm.n_iter == r_m.n_iter
    assert r_mm.x[0] == pytest.approx(_ROOT_7_MM, rel=1e-8)
    assert r_m.x[0] * 1e3 == pytest.approx(r_mm.x[0], rel=1e-12)
    # 1e-4 m = 0.1 mm: the unscaled run stops early, with a residual of about 10 % of P [N]
    assert r_m_unscaled.n_iter < r_m.n_iter and abs(r_m_unscaled.F[0]) > 10.0


def test_f_scale_kn_vs_n_same_history():
    """The residual in kN with f_scale = 1e-3 gives the same history as in N."""
    opts = dict(residual_abs=1e-9, x_scale=1.0)
    r_n = _solve(_spring(1e-3), [1.0], "spring", "t7 N", options=NewtonOptions(**opts))
    r_kn = _solve(_spring(1e-3, 1e3), [1.0], "spring", "t7 kN",
                  options=NewtonOptions(**opts, f_scale=1e-3))
    assert r_n.n_iter == r_kn.n_iter
    np.testing.assert_allclose([h.residual_norm for h in r_kn.history],
                               [h.residual_norm for h in r_n.history], rtol=1e-6, atol=1e-10)
    # equal up to round-off of the forward-difference Jacobian


# ---- 8. project, callback, max_step, modified Newton, damping -----------------------------
def test_project_keeps_bound():
    """ln x from 3: x1 = 3 - 3 ln 3 = -0.30 makes ln x NaN; projection on x >= 1e-3 converges."""
    seen: list[float] = []

    def F(x):
        seen.append(float(x[0]))
        return np.log(x)

    J = lambda x: np.array([[1.0 / x[0]]])
    with np.errstate(invalid="ignore"):
        free = _solve(F, [3.0], "log", "t8 free", jacobian=J,
                      options=NewtonOptions(globalisation="none"))
    assert not free.converged and free.reason == "non_finite_residual"
    seen.clear()
    r = _solve(F, [3.0], "log", "t8 project", jacobian=J,
               options=NewtonOptions(globalisation="none", project=lambda x: np.maximum(x, 1e-3)))
    assert r.converged and r.x[0] == pytest.approx(1.0, abs=1e-10)
    assert min(seen) >= 1e-3


def test_callback_stops():
    """A callback returning True stops the run, which counts as not converged."""
    seen: list[IterationRecord] = []

    def cb(rec):
        seen.append(rec)
        return rec.k == 2

    r = _solve(np.arctan, [1.5], "callback", "t8 callback", callback=cb,
               options=NewtonOptions(fallback="hybr"))
    assert not r.converged and r.reason == "callback" and r.n_iter == 2
    assert r.fallback is None                          # no fallback after a callback stop
    assert [s.k for s in seen] == [1, 2] and seen[-1] is r.history[-1]
    with pytest.raises(NewtonError, match="'callback' 't8 raise'"):
        newton(np.arctan, [1.5], callback=cb, options=NewtonOptions(on_failure="raise"),
               name="callback", label="t8 raise")


def test_max_step_caps_the_step():
    """x - 100 from 0 with max_step = 10 (x_scale = 1): ten steps of 10."""
    r = _solve(lambda x: x - 100.0, [0.0], "cap", "t8 max_step", jacobian=lambda x: np.eye(1),
               options=NewtonOptions(globalisation="none", max_step=10.0))
    assert r.converged and r.n_iter == 10
    assert all(h.step_norm == pytest.approx(10.0) for h in r.history[1:])


def test_max_step_with_line_search():
    """The cap shortens the Newton step by t; the Armijo slope must use t as well."""
    r = _solve(lambda x: x - 1.0e6, [0.0], "cap", "t8 max_step line_search",
               jacobian=lambda x: np.eye(1), options=NewtonOptions(max_step=10.0, max_iter=3))
    # t = 1e-5 < c1: without t in the Armijo test no step would be accepted
    assert r.reason == "max_iter" and r.n_iter == 3 and r.x[0] == pytest.approx(30.0)
    assert all(h.step_length == pytest.approx(10.0 / (1e6 - 10.0 * (h.k - 1)))
               for h in r.history[1:])


def test_damped_steps():
    """A fixed damping 0.5 converges linearly with every step at half length."""
    r = _solve(lambda x: x**2 - 2.0, [1.0], "damped", "t8 damped",
               options=NewtonOptions(globalisation="damped", damping=0.5, step_rel=None,
                                     residual_abs=1e-8))
    assert r.converged and r.x[0] == pytest.approx(math.sqrt(2.0), abs=1e-8)
    assert all(h.step_length == 0.5 for h in r.history[1:]) and r.n_iter > 20


def test_stalled_line_search_is_not_convergence():
    """x^2 + 1 has no real root: Armijo stalls near 0 and a shortened step is not convergence."""
    r = _solve(lambda x: x**2 + 1.0, [0.3], "stall", "t8 stall",
               jacobian=lambda x: np.array([[2.0 * x[0]]]),
               options=NewtonOptions(max_condition=None))
    assert not r.converged and r.reason == "line_search_failed"


def test_modified_newton():
    """With jacobian_update = 3 the Jacobian is kept for three steps (chord method)."""
    full = _solve(_F3, [1.0, 1.0], "modified", "t8 k=1", jacobian=_J3)
    mod = _solve(_F3, [1.0, 1.0], "modified", "t8 k=3", jacobian=_J3,
                 options=NewtonOptions(jacobian_update=3))
    assert mod.converged
    np.testing.assert_allclose(mod.x, ROOT_3, rtol=1e-11)
    assert full.n_jacobian_evaluations == full.n_iter
    assert mod.n_jacobian_evaluations < mod.n_iter
    assert [h.jacobian_evaluated for h in mod.history[1:5]] == [True, False, False, True]


# ---- 9. fallback -------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["hybr", "lm"])
def test_fallback_only_on_failure(method):
    """Full Newton from 1.5 on arctan fails at max_iter = 2; SciPy root finishes the solve."""
    r = _solve(np.arctan, [1.5], "fallback", f"t9 {method}",
               options=NewtonOptions(globalisation="none", max_iter=2, fallback=method))
    assert r.fallback == method and r.newton_reason == "max_iter"
    assert r.converged and r.reason == f"fallback_{method}" and abs(r.x[0]) < 1e-10
    assert r.n_residual_evaluations > r.history[-1].n_residual_evaluations
    assert r.residual_norm == pytest.approx(abs(np.arctan(r.x[0])))     # final point
    assert r.record()["residual_norm"] == r.residual_norm

    ok = _solve(_F3, [1.0, 1.0], "fallback", "t9 unused", options=NewtonOptions(fallback=method))
    assert ok.converged and ok.fallback is None and ok.reason == ok.newton_reason


def test_fallback_checked_against_caller_tolerance():
    """SciPy success is not enough: the caller's residual test is applied to its point."""
    # x^2 - 2 is never exactly 0 in floating point (|F| >= 4.4e-16 > 1e-20)
    r = _solve(lambda x: x**2 - 2.0, [1.0], "fallback", "t9 failed",
               options=NewtonOptions(residual_abs=1e-20, step_rel=None, max_iter=1,
                                     fallback="hybr"))
    assert abs(r.F[0]) < 1e-12                          # SciPy found the root ...
    assert not r.converged and r.reason == "fallback_hybr_failed"   # ... but not to 1e-20
    assert r.newton_reason == "max_iter" and r.fallback == "hybr"


# ---- 10. two Hertz-type point contacts in series, Q = K delta^1.5 ------------------------
def test_hertz_chain_closed_form():
    """Ball between two raceways: each contact carries P, delta_i = (P/K_i)^(2/3).

    Hertz-type point-contact law Q = K delta^1.5 (source to find). K_i, K_o [N/mm^1.5] are
    orders of magnitude for a steel ball of about 10 mm in conforming grooves, not from a
    standard; P [N]. Unknowns u1, u2 [mm]: approach of the inner contact and total approach.
    """
    Ki, Ko, P = 5.0e5, 7.0e5, 1.0e3

    def F(u):                                 # residual [N]
        d1, d2 = u[0], u[1] - u[0]            # contact approaches [mm]
        q1 = Ki * max(d1, 0.0) ** 1.5
        q2 = Ko * max(d2, 0.0) ** 1.5
        return np.array([q1 - q2, q2 - P])

    def J(u):                                 # [N/mm]
        d1, d2 = u[0], u[1] - u[0]
        k1 = 1.5 * Ki * max(d1, 0.0) ** 0.5   # contact stiffness dQ/d delta
        k2 = 1.5 * Ko * max(d2, 0.0) ** 0.5
        return np.array([[k1 + k2, -k2], [-k2, k2]])

    d_i, d_o = (P / Ki) ** (2.0 / 3.0), (P / Ko) ** (2.0 / 3.0)
    r = _solve(F, [1e-3, 2e-3], "hertz_chain", "t10 P=1kN", jacobian=J,
               options=NewtonOptions(x_scale=1e-2, f_scale=P, residual_abs=1e-12))
    assert r.converged
    np.testing.assert_allclose(r.x, [d_i, d_i + d_o], rtol=1e-10)
    Kn = (Ki ** (-2.0 / 3.0) + Ko ** (-2.0 / 3.0)) ** -1.5      # series constant
    assert Kn * r.x[1] ** 1.5 == pytest.approx(P, rel=1e-10)
    res = [h.residual_norm for h in r.history if h.residual_norm > 1e-13]
    # order about 2 only if J is the exact derivative of F
    assert math.log(res[-1] / res[-2]) / math.log(res[-2] / res[-3]) > 1.8


# ---- reasons, evaluation contract, options, exports --------------------------------------
def test_residual_budget():
    """The residual-evaluation budget stops the run (FD evaluations count)."""
    r = _solve(_F3, [1.0, 1.0], "budget", "budget",
               options=NewtonOptions(max_residual_evaluations=5))
    assert not r.converged and r.reason == "max_residual_evaluations"
    assert r.n_residual_evaluations == 5


def test_last_residual_call_is_at_result():
    """Without fallback, residual is last called at r.x (a caller may cache its state)."""
    calls: list[np.ndarray] = []

    def F(v):
        calls.append(v.copy())
        return _F3(v)

    r = _solve(F, [1.0, 1.0], "cache", "contract")
    np.testing.assert_array_equal(calls[-1], r.x)


def test_options_validation():
    """Inconsistent options fail at construction; a wrong scale shape fails in newton."""
    with pytest.raises(ValueError, match="at least one stopping tolerance"):
        NewtonOptions(**_NONE)
    with pytest.raises(ValueError, match="globalisation"):
        NewtonOptions(globalisation="trust")
    assert NewtonOptions().validate() == []
    with pytest.raises(ValueError, match="x_scale"):
        newton(_F3, [1.0, 1.0], options=NewtonOptions(x_scale=[1.0, 1.0, 1.0]),
               name="options", label="bad x_scale")


def test_reexports():
    """The public names are reachable from the package rollups."""
    assert numerical_solvers.newton_raphson.newton is newton and numerical_solvers.newton is newton
    assert numerical_solvers.newton_raphson.NewtonOptions is NewtonOptions
    assert numerical_solvers.newton_raphson.NewtonResult is NewtonResult
    assert numerical_solvers.newton_raphson.IterationRecord is IterationRecord


# ---- port check: the same problems through AxisForge d34ec7f ------------------------------
def _hertz_F(u):
    d1, d2 = u[0], u[1] - u[0]
    return np.array([5.0e5 * max(d1, 0.0) ** 1.5 - 7.0e5 * max(d2, 0.0) ** 1.5,
                     7.0e5 * max(d2, 0.0) ** 1.5 - 1.0e3])


_PORT_CASES = {   # name: (residual, x0, jacobian, options); the problems of the tests above
    "t1 sqrt2": (lambda x: x**2 - 2.0, 1.0, lambda x: np.array([[2.0 * x[0]]]),
                 dict(residual_abs=1e-14, step_rel=None)),
    "t3 analytic": (_F3, [1.0, 1.0], _J3, {}),
    "t3 forward": (_F3, [1.0, 1.0], None, dict(jacobian_fd="forward")),
    "t3 central": (_F3, [1.0, 1.0], None, dict(jacobian_fd="central")),
    "t4 none": (np.arctan, [1.5], lambda x: np.array([[1.0 / (1.0 + x[0] ** 2)]]),
                dict(globalisation="none")),
    "t4 line_search": (np.arctan, [1.5], lambda x: np.array([[1.0 / (1.0 + x[0] ** 2)]]), {}),
    "t5 all": (_F3, [1.0, 1.0], _J3, dict(_NONE, residual_abs=1e-14, step_abs=1e-2, combine="all")),
    "t6 rank 1": (lambda v: np.array([v[0] + v[1] - 1.0, 2 * v[0] + 2 * v[1] - 3.0]), [0.0, 0.0],
                  lambda v: np.array([[1.0, 1.0], [2.0, 2.0]]), {}),
    "t7 m scaled": (_spring(1.0), [1e-3], None, dict(_NONE, step_abs=1e-4, x_scale=1e-3)),
    "t8 max_step": (lambda x: x - 1.0e6, [0.0], lambda x: np.eye(1), dict(max_step=10.0, max_iter=3)),
    "t8 damped": (lambda x: x**2 - 2.0, [1.0], None,
                  dict(globalisation="damped", damping=0.5, step_rel=None, residual_abs=1e-8)),
    "t8 stall": (lambda x: x**2 + 1.0, [0.3], lambda x: np.array([[2.0 * x[0]]]),
                 dict(max_condition=None)),
    "t8 modified": (_F3, [1.0, 1.0], _J3, dict(jacobian_update=3)),
    "t9 lm": (np.arctan, [1.5], None, dict(globalisation="none", max_iter=2, fallback="lm")),
    "t10 hertz": (_hertz_F, [1e-3, 2e-3], None, dict(x_scale=1e-2, f_scale=1e3, residual_abs=1e-12)),
    "budget": (_F3, [1.0, 1.0], None, dict(max_residual_evaluations=5)),
    "t5 k0": (_F3, ROOT_3, _J3, dict(residual_abs=1e-6)),
    "t5 residual_rel": (_F3, [1.0, 1.0], _J3, dict(_NONE, residual_rel=1e-3)),
    "t6 LinAlgError": (lambda v: np.array([v[0] + v[1] - 1.0, 2 * v[0] + 2 * v[1] - 3.0]), [0.0, 0.0],
                       lambda v: np.array([[1.0, 1.0], [2.0, 2.0]]), dict(max_condition=None)),
    "t8 project": (np.log, [3.0], lambda x: np.array([[1.0 / x[0]]]),
                   dict(globalisation="none", project=lambda x: np.maximum(x, 1e-3))),
    "t8 non_finite": (np.log, [3.0], lambda x: np.array([[1.0 / x[0]]]), dict(globalisation="none")),
    "t9 hybr analytic J": (np.arctan, [1.5], lambda x: np.array([[1.0 / (1.0 + x[0] ** 2)]]),
                           dict(globalisation="none", max_iter=2, fallback="hybr", x_scale=2.0)),
    "norm inf, vector scales": (_F3, [1.0, 1.0], None,
                                dict(norm="inf", x_scale=[1.0, 0.5], f_scale=[4.0, 1.0],
                                     fd_rel_step=1e-7)),
}


@pytest.mark.parametrize("case", list(_PORT_CASES))
def test_port_matches_axisforge(case):
    """The port and ``axisforge.solvers.numerics.newton`` call the residual at the same points and
    return the same result and history, bit for bit. It compares with the AxisForge that is installed,
    not with d34ec7f itself: a failure after a change of AxisForge's Newton means AxisForge moved; it
    is skipped when the installed AxisForge has no ``axisforge.solvers.numerics``."""
    af = pytest.importorskip("axisforge.solvers.numerics",
                             reason="installed AxisForge has no numerics (not on feature/newton-solver)")
    residual, x0, jac, kw = _PORT_CASES[case]
    out = {}
    for tag, solve, Options in (("port", newton, NewtonOptions),
                                ("axisforge", af.newton, af.NewtonOptions)):
        calls: list[np.ndarray] = []

        def F(v):
            calls.append(np.array(v, dtype=float))
            return residual(v)

        with np.errstate(over="ignore", invalid="ignore"):
            r = solve(F, x0, jacobian=jac, options=Options(**kw), name="port", label=case)
        out[tag] = (r, calls)
    (rp, cp), (ra, ca) = out["port"], out["axisforge"]
    assert len(cp) == len(ca) and all(np.array_equal(a, b) for a, b in zip(cp, ca))
    np.testing.assert_array_equal(rp.x, ra.x)
    np.testing.assert_array_equal(rp.F, ra.F)
    assert rp.record() == ra.record()
    assert len(rp.records()) == len(ra.records())
    for a, b in zip(rp.records(), ra.records()):
        assert a.keys() == b.keys()
        assert all(a[k] == b[k] or (isinstance(a[k], float) and math.isnan(a[k]) and math.isnan(b[k]))
                   for k in a)
