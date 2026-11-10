"""Fast tests of v002: the resolved system, the comparator with couples and settlements, the gear identities, Harris."""
import math

import pytest

from studies.v000_axisforge.v002_shaft_stepped_gear import (THEORIES, check, gear_identities, gear_system, literature,
                                                            mesh_judge)
from studies.v000_axisforge.v002_shaft_stepped_gear.construction import STEPS_MM, SUPPORTS_MM
from studies.v000_axisforge.v002_shaft_stepped_gear.study import SETTLEMENTS_MM, literature_series, solve

EB, TIM = THEORIES["euler_bernoulli"], THEORIES["timoshenko/cowper/single_point"]


@pytest.fixture(scope="module")
def base():
    return gear_system()


def test_the_system_is_resolved_unsolved_with_a_couple_on_each_shaft(base):
    for ss in base.shafts:
        assert [b.position for b in ss.bearings] == list(SUPPORTS_MM)
        assert [ss.shaft.axial_end(i) for i in range(4)] == list(STEPS_MM)
        (m,) = ss.external_moments
        assert m.source == "gear_mesh" and m.magnitude > 0.0


@pytest.mark.parametrize("settings, expected", [(EB, {"C1", "C2", "C3", "C7"}), (TIM, {"C3", "C4", "C7"})])
def test_check_passes_with_couples_and_settlements_on_the_coarsest_mesh(base, settings, expected):
    for ss in base.shafts:
        rows = check(ss, solve(ss, settings, 40, settlements_mm=SETTLEMENTS_MM), settings, settlements_mm=SETTLEMENTS_MM)
        judged = [r for r in rows if r["verdict"] != "RECORD"]
        assert {r["id"] for r in judged} == expected and {r["plane"] for r in judged} == {"XY", "XZ", "X"}
        assert all(r["verdict"] == "PASS" for r in judged), judged
        assert all(r["tag"] == "convention-dependent (J-16)" for r in rows)


def test_a_settlement_moves_the_shaft_without_changing_the_reactions(base):
    ss = base.shafts[0]
    rigid, settled = solve(ss, EB, 40), solve(ss, EB, 40, settlements_mm=SETTLEMENTS_MM)
    assert settled.R_xy == pytest.approx(rigid.R_xy, rel=1e-9) and settled.R_xz == pytest.approx(rigid.R_xz, rel=1e-9)
    i_A = list(settled.x_nodes).index(SUPPORTS_MM[0])
    assert settled.v_xy[i_A] == pytest.approx(SETTLEMENTS_MM["XY"][0], abs=1e-15)


def test_ignoring_the_settlement_in_the_judge_is_caught(base):
    ss = base.shafts[0]
    rows = check(ss, solve(ss, EB, 40, settlements_mm=SETTLEMENTS_MM), EB)
    assert any(r["verdict"] == "MISS" for r in rows if r["id"] == "C1")


@pytest.mark.parametrize("beta, hand, phi", [(0.0, 1, 30.0), (20.0, 1, 30.0), (20.0, -1, 120.0)])
def test_gear_identities(beta, hand, phi):
    rows = gear_identities(gear_system(beta_deg=beta, hand=hand, phi_deg=phi), _judge(beta, hand), theta_c_deg=phi)
    assert all(r["verdict"] == "PASS" for r in rows if r["verdict"] != "RECORD"), rows
    assert {r["id"] for r in rows} == {"I1", "I2", "I3", "I3s", "I4", "I5", "I6", "I7", "I8", "T5"}


def _judge(beta, hand):
    return mesh_judge(torque_Nm=100.0, beta_deg=beta, mn_mm=2.0, z1=20, z2=40, alpha_n_deg=20.0, b_mm=20.0,
                      hand1=hand if beta else 0, rot1=1)


def test_a_wrong_hand_in_the_judge_is_caught():
    """I3s / I5 / I8 judge the SIGN of the axial force from the inputs, not from AxisForge's MeshForces."""
    rows = gear_identities(gear_system(beta_deg=20.0, hand=1, phi_deg=30.0), _judge(20.0, -1), theta_c_deg=30.0)
    assert {r["id"] for r in rows if r["verdict"] == "MISS"} >= {"I3s", "I5", "I8"}


def test_axial_reaction_at_the_locating_pin(base):
    """C7: the locating pin takes -F_x of the helical mesh, the other none."""
    ss = base.shafts[0]
    rows = [r for r in check(ss, solve(ss, EB, 40), EB) if r["id"] == "C7"]
    assert len(rows) == 1 and rows[0]["verdict"] == "PASS"


def test_spur_f_tw_is_2000_t_over_d():
    """T = 100 N.m on d = 2 x 20 = 40 mm: F_tw = 5000 N, F_r = 5000 tan 20 deg, no axial force (spur)."""
    mesh = gear_system(beta_deg=0.0).meshes[0]
    assert mesh.F_tw_N == pytest.approx(5000.0, rel=1e-12)   # d_w = d_b / cos(alpha_wt): 3e-14 round trip
    assert mesh.F_r_N == pytest.approx(5000.0 * math.tan(math.radians(20.0)), rel=1e-12) and mesh.F_a_N == 0.0


def test_the_named_reading_of_the_couple_is_wrong_cw01(base):
    """T5: the ExternalMoment read as My about +Y, Mz about +Z is not r x F (the FEM consumes it 90 deg on)."""
    rows = gear_identities(base, _judge(20.0, 1), theta_c_deg=30.0)
    assert all(r["rel_err"] > 0.1 for r in rows if r["id"] == "T5")


def test_harris_eqs_9_24_9_27():
    rows = [r for r in literature_series() if r["id"].startswith("L")]
    assert all(r["verdict"] == "PASS" for r in rows if r["id"] != "L6"), rows
    assert all(r["rel_err"] > 0.05 for r in rows if r["id"] == "L6")          # the printed 1/l^3 misprint shows


def test_harris_pin_slopes_reproduce_statics():
    """Pins (M_1 = M_2 = 0): F_1 from Eq. 9.24 equals the statics of the loads, whatever the settlements."""
    EI, l, loads = 6.3e9, 180.0, literature.LOADS
    th = literature.harris_pin_slopes(EI, l, loads, 0.1, -0.2)
    F = literature.harris_eqs_9_24_9_27(EI, l, loads, *th, 0.1, -0.2)
    statics = sum(P * (l - a) / l - T / l for a, P, T in loads)
    assert F["F_1"] == pytest.approx(statics, rel=1e-12)
