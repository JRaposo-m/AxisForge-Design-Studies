"""Fast tests of s005: the moment probe, the flexibility, the sign of the bearing moment, the Newton solve, its limits."""
import math

import numpy as np
import pytest

from studies.s000_axisforge.s004_bearing_compliance import coupled_tilt, ring_displacement
from studies.s000_axisforge.s005_bearing_moment_feedback import (axial_loads, build, moment_balance, moment_on_shaft,
                                                                 pin_flexibility, solve_feedback, with_moments)
from studies.s000_axisforge.s005_bearing_moment_feedback.model import BEARING_TOL, EB, N_ELEMENTS
from studies.v000_axisforge.v002_shaft_stepped_gear.study import solve


@pytest.fixture(scope="module")
def ball():
    return build(span_mm=200.0, floating="ball", s_key="C3-like", Fa_over_Fr=0.5)


def _e(result):
    R = np.array([result.R_xy[0], result.R_xz[0]])
    return R / np.linalg.norm(R)


def test_moment_probe_and_flexibility_are_the_simply_supported_beam(ball):
    """A moment M along e at a pin of a uniform two-pin beam: slope L/(3EI) M there, -L/(6EI) M at the other pin,
    reactions +-M/L along e (closed form; the sign shows how the FEM consumes ExternalMoment)."""
    ss = ball["system"]
    rigid = solve(ss, EB, N_ELEMENTS)
    e = _e(rigid)
    f = pin_flexibility(ss, rigid, [0.0, 200.0], e)
    EI = 210e3 * math.pi * 40.0**4 / 64.0            # S355 E 210 GPa (s006 shaft), d 40 mm
    L = 200.0
    assert f["F"][0, 0] == pytest.approx(L / (3 * EI), rel=1e-6)
    assert f["F"][1, 0] == pytest.approx(-L / (6 * EI), rel=1e-6)
    assert f["C"][0, 0] == pytest.approx(e / L, rel=1e-6) and f["C"][1, 1] == pytest.approx(-e / L, rel=1e-6)
    assert np.max(np.abs(f["theta_n"])) < 1e-18
    neg = with_moments(ss, [0.0, 200.0], [-5.0, 0.0], e).external_moments[-1]
    assert neg.magnitude == 5.0 and neg.My * e[0] + neg.Mz * e[1] == pytest.approx(-5.0)


@pytest.mark.parametrize("floating, k", [("ball", 0), ("roller", 1)])
def test_moment_on_the_shaft_restores_the_tilt(floating, k):
    """Virtual work: ball -Mz18, roller +Mz46 (Eq. (40)) give a moment on the shaft opposing the tilt (Fa = 0)."""
    c = build(span_mm=200.0, floating=floating, s_key="C3-like", Fa_over_Fr=0.0)
    rigid = solve(c["system"], EB, N_ELEMENTS)
    R = np.array([rigid.R_xy[k], rigid.R_xz[k]])
    b = c["bearings"][k]
    for psi in (5e-4, -5e-4):
        ring = ring_displacement(b, R, psi * _e(rigid), tol=BEARING_TOL)
        assert float(moment_on_shaft(b, ring, R) @ _e(rigid)) * psi * float(_e(rigid) @ R) < 0.0


def test_the_axial_sense_sets_the_sign_of_the_moment_at_zero_tilt(ball):
    """Symmetric deep groove at psi = 0: reversing Fa reverses the moment, the same |Q_max| (the other zone reading of
    the first run is this sense, review of s005); Mz0 / (Fa Dpw/2) in (0, 1] (Eqs (17)-(18))."""
    rigid = solve(ball["system"], EB, N_ELEMENTS)
    R = np.array([rigid.R_xy[0], rigid.R_xz[0]])
    b = ball["bearings"][0]
    p, m = (ring_displacement(b, R, (0.0, 0.0), Fa_N=f, tol=BEARING_TOL) for f in (2000.0, -2000.0))
    assert m["Mz_Nmm"] == pytest.approx(-p["Mz_Nmm"], rel=1e-9) and m["Q_max_N"] == pytest.approx(p["Q_max_N"])
    assert 0.0 < p["Mz_Nmm"] / (2000.0 * b.Dpw / 2.0) <= 1.0


def test_no_feedback_is_s004_and_feedback_closes_the_moment_balance(ball):
    """Limit Mz -> 0 = s004 coupled_tilt (Fa = 0); with feedback, Newton converges and the shaft balances incl. Mz."""
    c = build(span_mm=200.0, floating="ball", s_key="C3-like", Fa_over_Fr=0.0)
    o = solve_feedback(c["system"], c["bearings"], feedback=0.0)
    ct = coupled_tilt(c["system"], o["rigid"], c["bearings"])
    assert o["newton"].converged
    assert np.allclose([p @ o["e_yz"] for p in ct["psi_yz"]], o["psi_rad"], rtol=0, atol=1e-10)
    Fa = axial_loads(solve(ball["system"], EB, N_ELEMENTS), 0.5)
    o = solve_feedback(ball["system"], ball["bearings"], Fa_N=Fa, feedback=1.0)
    assert o["newton"].converged and o["newton"].n_iter <= 6
    assert moment_balance(o["final_system"], o["final"], o["x_mm"], o["G_fed_yz"])["rel"] < 1e-8
    assert o["superposition_theta_err_rad"] < 1e-10
    # the fed-back moment is the bearing's reaction: m_k = G_k . e = -Mz_k (R_hat . e) for the ball
    assert o["m_Nmm"][0] == pytest.approx(-o["bearing"][0]["Mz_Nmm"] * float(o["R_yz"][0] @ o["e_yz"])
                                          / np.linalg.norm(o["R_yz"][0]), rel=1e-9)


def test_bad_inputs():
    with pytest.raises(ValueError):
        build(span_mm=200.0, floating="ball", s_key="C9", Fa_over_Fr=0.0)
    c = build(span_mm=200.0, floating="ball", s_key="0", Fa_over_Fr=0.0)
    with pytest.raises(ValueError):
        solve_feedback(c["system"], c["bearings"][:1])
