"""Fast tests of s003: the ball law against solve_contact, the linear, power-law and stiff limits, lift-off, statics,
bad input."""

import numpy as np
import pytest
from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281.contact_solver import ISO16281BallSolver

from studies.s000_axisforge.s003_hyperstatic_shaft import ball_loads, bearings, shaft, solve_shaft
from studies.s000_axisforge.s003_hyperstatic_shaft.construction import P_N, SUPPORTS_MM, X_GEAR_MM, X_MIDDLE_MM
from studies.s000_axisforge.s003_hyperstatic_shaft.study import (EI_NMM2, TOL_HERTZ, hertz_judge, hertz_K, law_check,
                                                                 linear_limit, stiff_limit)
from studies.v000_axisforge.v005_shaft_three_supports.study import EB, hertz_problem

N = 27                                     # coarse mesh: supports (10, 110, 260) and gear (60) on 10 mm nodes


@pytest.fixture(scope="module")
def problem():
    return hertz_problem(shaft(), EB, N)


def test_ball_sums_at_the_solve_contact_solution_give_the_load():
    assert all(r["verdict"] == "PASS" for r in law_check())


@pytest.mark.parametrize("s_mm", [0.0, 0.02])
def test_no_force_inside_the_half_clearance_and_cos_1_5_at_zero_clearance(s_mm):
    b = bearings(s_mm)[SUPPORTS_MM[0]]
    assert ball_loads(b, 0.999 * s_mm / 2.0)[0] == 0.0
    if s_mm == 0.0:
        Fr, Q = ball_loads(b, 0.01)
        phi = np.asarray(b.phi_j)
        expected = Q.max() * np.clip(np.cos(phi), 0.0, None) ** 1.5    # Stribeck shape, delta_j = delta_r cos(phi)
        assert Q == pytest.approx(expected, rel=1e-12, abs=1e-12)
        sol = ISO16281BallSolver(tol=1e-12).solve_contact(b, Fr, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        assert sol.delta_r == pytest.approx(0.01, rel=1e-9)


def test_a_negative_displacement_is_refused():
    with pytest.raises(ValueError):
        ball_loads(bearings(0.0)[SUPPORTS_MM[0]], -1e-3)


def test_bearings_must_sit_at_the_shaft_bearings(problem):
    with pytest.raises(ValueError, match="bearings at"):
        solve_shaft(problem, {10.0: bearings(0.0)[10.0]})


def test_linear_law_equals_v005():
    assert all(r["verdict"] == "PASS" for r in linear_limit(hertz_problem(shaft(), EB, 54)))


@pytest.mark.parametrize("s_mm, e_um", [(0.0, 0.0), (0.03, -50.0), (0.03, 0.0)])
def test_coupled_solve_converges_and_balances_the_load(problem, s_mm, e_um):
    c = solve_shaft(problem, bearings(s_mm), seat_mm={X_MIDDLE_MM: e_um * 1e-3})
    assert c["newton"].converged
    F, x = c["F_N"], np.asarray(c["positions"])
    assert F.sum() == pytest.approx(P_N, rel=1e-8)
    assert F @ (x - x[0]) == pytest.approx(P_N * (X_GEAR_MM - x[0]), rel=1e-8)
    # the bearing force opposes the ring displacement (the balls push the shaft back toward its seat)
    assert all(Fk * dk <= 0.0 for Fk, dk in zip(F, c["d_mm"]))
    if (s_mm, e_um) == (0.03, 0.0):
        # S8: bearing 3 inside its clearance: two-support statics and the end slope of span 1 along the overhang
        assert F == pytest.approx([P_N / 2, P_N / 2, 0.0], abs=1e-8 * P_N)
        D = P_N * (x[1] - x[0]) ** 2 * (x[2] - x[1]) / (16.0 * EI_NMM2)
        assert c["d_mm"][2] - c["d_mm"][0] == pytest.approx(D, rel=1e-8)
        assert abs(c["d_mm"][2]) < s_mm / 2.0


@pytest.mark.parametrize("scale", [1.0, 1e4])
def test_zero_clearance_equals_the_v005_power_law(problem, scale):
    # s = 0: F = K |d|^1.5, K = scale c_p sum cos^2.5 phi_j (physics review MUST 1), with the seat at +50 um
    B, seat = bearings(0.0), {X_MIDDLE_MM: 0.05}
    c = solve_shaft(problem, B, seat_mm=seat, scale=scale)
    F_j, d_j = hertz_judge(problem, seat, hertz_K(B[SUPPORTS_MM[0]], scale))
    assert np.max(np.abs(c["F_N"] - F_j)) / P_N < TOL_HERTZ
    assert np.max(np.abs(c["d_mm"] - d_j)) / np.max(np.abs(d_j)) < TOL_HERTZ


def test_stiff_limit_records_agree_with_their_verdicts(problem):
    rows = stiff_limit(problem)
    judged = [r for r in rows if r["id"] in ("S6", "S6v")]
    assert judged and all(r["verdict"] == "PASS" and abs(r["rel_err"]) <= r["tol"] for r in judged)
    # S6r: RECORD only (R5-167), the distance to the rigid FEM falling as scale^(-2/3)
    slopes = [r["slope_vs_previous"] for r in rows if r["id"] == "S6r" and r["scale"] >= 1e4]
    assert all(r["verdict"] == "RECORD" for r in rows if r["id"] == "S6r")
    assert slopes == pytest.approx([-2.0 / 3.0] * len(slopes), abs=0.02)
