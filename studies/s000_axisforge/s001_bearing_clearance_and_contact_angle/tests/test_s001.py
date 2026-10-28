"""Fast tests of s001: the bearings, the model (node, solve, zone, records), one point of each criterion."""
import math

import pytest

from studies.s000_axisforge.s001_bearing_clearance_and_contact_angle import (BALANCE_RTOL, angular_contact,
                                                                             cylindrical_roller, deep_groove, load_node,
                                                                             loaded_zone, records, solve_bearing)


def _solve(bearing, Fr, Fa=0.0):
    analysis, post, error = solve_bearing(bearing, load_node(bearing, Fr, Fa))
    return records(bearing, analysis, postprocessed=post, error=error, kind=bearing.label)


def test_bearings_are_unsolved_with_their_clearance():
    assert deep_groove(0.02).s == pytest.approx(0.02) and cylindrical_roller(0.03).s == pytest.approx(0.03)
    b = angular_contact(25.0)
    assert math.degrees(b.alpha_0) == pytest.approx(25.0)
    assert b.s == pytest.approx(2.0 * b.A * (1.0 - math.cos(b.alpha_0)))     # s derived from alpha_0


def test_load_node_points_the_radial_load():
    node = load_node(deep_groove(), 1000.0, 200.0, phi_Fr_rad=math.pi / 2)
    assert (node.Fr_xz, node.Fr_xy, node.Fa) == pytest.approx((0.0, 1000.0, 200.0), abs=1e-12)


def test_zero_clearance_radial_ball_has_half_ring_zone():
    """s = 0, Fa = 0: psi_l = 90 deg, eps = 0.5 (Harris Eq. 7.12), balanced; Stribeck Z Q_max / Fr near 4.37."""
    case, rows = _solve(deep_groove(0.0), 5000.0)
    assert case["ok"] and case["balanced"] and case["psi_l_deg"] == pytest.approx(90.0)
    assert case["epsilon"] == pytest.approx(0.5)
    assert 4.0 < 9 * case["Q_max_N"] / 5000.0 < 4.6
    assert [r["phi_local_deg"] for r in rows][0] == 0.0


def test_clearance_shrinks_the_zone_and_raises_q_max():
    a, b = _solve(deep_groove(0.0), 5000.0)[0], _solve(deep_groove(0.04), 5000.0)[0]
    assert b["psi_l_deg"] < a["psi_l_deg"] and b["Q_max_N"] > a["Q_max_N"]
    c = _solve(cylindrical_roller(0.04), 5000.0)[0]
    assert math.cos(math.radians(c["psi_l_deg"])) == pytest.approx(0.04 / (2 * c["delta_r_mm"]))


def test_loaded_zone_edge_matches_the_element_loads():
    """Every ball inside psi_l carries load, every ball outside none (deep groove, s 20 um)."""
    b = deep_groove(0.02)
    case, rows = _solve(b, 5000.0)
    for r in rows:
        inside = abs((r["phi_local_deg"] + 180.0) % 360.0 - 180.0) < case["psi_l_deg"]
        assert (r["Q_N"] > 0.0) == inside


def test_angular_contact_guard_and_r137_case():
    loaded = _solve(angular_contact(40.0), 5000.0, 5000.0)[0]
    assert loaded["alpha_guard"] is True and loaded["alpha_max_deg"] > 30.0
    radial = _solve(angular_contact(40.0), 5000.0, 0.0)[0]
    assert radial["alpha_guard"] is None          # no axial load: guard not applicable; the study flags R-137


def test_balance_rule_and_bad_input():
    assert BALANCE_RTOL == 1e-6
    assert math.isnan(loaded_zone(deep_groove(), 0.0)[0])
    with pytest.raises(ValueError):
        deep_groove(-0.01)


def test_study_criteria_pass(tmp_path):
    """The whole sweep (about 5 s): S1-S4 PASS, validated through v003.check, figures written to tmp_path."""
    from studies.s000_axisforge.s001_bearing_clearance_and_contact_angle import study
    out = study.run(tmp_path)
    assert {c["verdict"] for c in out["criteria"]} == {"PASS"}, out["criteria"]
    assert all(p.is_file() for p in out["figures"]) and len(out["figures"]) == 6


def test_guard_reference_is_fa_for_a_symmetric_groove():
    """R5-44: a deep groove bearing with clearance under Fa < 0 runs at alpha < 0 legitimately; no false SB-45."""
    case = _solve(deep_groove(0.02), 5000.0, -2500.0)[0]
    assert case["alpha_guard"] is True and case["alpha_max_deg"] < 0.0
