"""slippy_bridge.contact: AxisForge bearing -> Contact -> surfaces -> static step."""
import math
from dataclasses import asdict, replace

import numpy as np
import pytest

af_c = pytest.importorskip("axisforge.core")
pytest.importorskip("slippy")

from axisforge.core.machine_elements.bearings.contact_models.iso16281 import PointContactStiffness  # noqa: E402
from axisforge.core.machine_elements.bearings.families import available_families  # noqa: E402
from axisforge.results import (BallBearingResult, BallLoadDistributionResult,  # noqa: E402
                               BearingAnalysisResult, RollerLoadDistributionResult)
from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281.contact_solver import (  # noqa: E402
    ISO16281BallSolver)

from references.contact.ehl import e_prime, film_scalings, groups, moes_inverse  # noqa: E402
from references.contact.hertz import hertz_point  # noqa: E402
from slippy_bridge.contact import (BearingLocation, Contact, ContactCase, Grid, Kinematics,  # noqa: E402
                                   bearing_contacts, contact_case, gear_contacts, hertz, measure, pair,
                                   rough, semi_axes, static_step)
from slippy_bridge.contact.elements.bearings import _entrainment_speed  # noqa: E402
from slippy_bridge.contact.reference import relative_radii  # noqa: E402
from slippy_bridge.lubrication import ehl  # noqa: E402

CONTACT = dict(e1=210000.0, e2=210000.0, nu1=0.3, nu2=0.3)


def _bearing(family, arrangement, contact: bool = True, **geometry):
    """Assembled AxisForge bearing; ``contact`` adds the ISO/TS 16281 contact constants (steel)."""
    catalog = af_c.BearingCatalog(d=20.0, D=42.0, b=12.0, designation="test", position=0.0,
                                  arrangement=arrangement, label="b")
    if contact:
        geometry = {**geometry, "contact": af_c.ContactAnalysis.ISO16281, **CONTACT}
    return af_c.Bearing.assemble(family=family, catalog=catalog, geometry=geometry)


@pytest.fixture(scope="module")
def ball():
    return _bearing(af_c.DeepGrooveBallFamily(), "locating", Dw=7.0, Dpw=31.0, Z=9, s=0.010)


@pytest.fixture(params=[0.0, 0.015])
def ball_6210(request, load_case):
    """The 6210-like ball bearing of ``validation/bearings/iso16281_ball_contact`` (Dw 12.7,
    Dpw 70 mm, Z 9, grooves 0.52 Dw), at radial clearance s = 0 (alpha_0 = 0) and s = 0.015 mm
    (alpha_0 = 9.86 deg). Built by the case, so these tests gate the bearing its README names."""
    return load_case("iso16281_ball_contact").build_bearing(request.param)


@pytest.fixture(scope="module")
def roller():
    return _bearing(af_c.CylindricalRollerFamily(), "non-locating",
                    Dwe=5.0, Lwe=5.0, Dpw=31.0, Z=14, n_s=50, s=0.010)


@pytest.fixture(scope="module")
def ball_row():
    Z = 9
    Q = np.array([300.0, 200.0, 50.0, 0.0, 0.0, 0.0, 0.0, 50.0, 200.0])
    return BallLoadDistributionResult(phi_j=np.linspace(0, 2 * np.pi, Z, endpoint=False), delta_j=np.zeros(Z),
                                      alpha_j=np.full(Z, 0.2), Q_j=Q, Fr_row=0.0, Fa_row=0.0, Mz=0.0)


@pytest.fixture(scope="module")
def roller_row():
    Z, n_s = 14, 50
    q = np.zeros((Z, n_s))
    q[0], q[1] = 10.0, 4.0
    return RollerLoadDistributionResult(phi_j=np.linspace(0, 2 * np.pi, Z, endpoint=False), delta_j=np.zeros(Z),
                                        alpha_j=np.zeros(Z), Q_j=q.sum(axis=1), Fr_row=0.0, Fa_row=0.0, Mz=0.0,
                                        x_k=np.linspace(-2.45, 2.45, n_s), psi_j=np.zeros(Z),
                                        delta_jk=np.zeros((Z, n_s)), q_jk=q)


# ---- every AxisForge bearing family --------------------------------------------------------

# Single-row geometry of each AxisForge family, the values of AxisForge's own family tests
# (axisforge/core/machine_elements/bearings/families/tests/conftest.py); multi-row families take
# a list of rows. A new AxisForge family fails the test below until it has an entry here.
_FAMILY_GEOMETRY = {
    "DeepGrooveBallFamily": dict(Dw=7.0, Dpw=31.0, Z=9, s=0.010),
    "AngularContactFamily": dict(Dw=8.0, Dpw=40.0, Z=12, alpha_0_deg=25.0),
    "SelfAligningBallFamily": dict(Dw=8.0, Dpw=40.0, Z=12, alpha_0_deg=25.0),
    "ThrustBallSingleRowFamily": dict(Dw=8.0, Dpw=40.0, Z=12, alpha_0_deg=90.0),
    "ThrustBallMultiRowFamily": dict(rows=[dict(Dw=8.0, Dpw=40.0, Z=12, alpha_0_deg=90.0)] * 2),
    "CylindricalRollerFamily": dict(Dwe=5.0, Lwe=5.0, Dpw=31.0, Z=14, n_s=50, s=0.010),
    "ThrustCylindricalRollerFamily": dict(Dwe=8.0, Lwe=8.0, Dpw=72.5, Z=18, s=0.01, n_s=40, alpha_0_deg=90.0),
    "RollerThrustMultiRowFamily": dict(rows=[dict(Dwe=8.0, Lwe=8.0, Dpw=72.5, Z=18, s=0.01, n_s=40,
                                                  alpha_0_deg=90.0)] * 2),
}


@pytest.mark.parametrize("name", sorted(available_families()))
def test_every_axisforge_family_translates_or_raises_clearly(name):
    """Each family in AxisForge ``available_families()`` either gives contacts or a NotImplementedError
    that names it. A family AxisForge itself cannot assemble is skipped (an AxisForge bug, not the bridge)."""
    geometry = _FAMILY_GEOMETRY[name]
    contact = "rows" not in geometry and "ThrustCylindrical" not in name     # these take no contact constants
    arrangement = "thrust" if "Thrust" in name else "non-locating" if "Cylindrical" in name else "locating"
    try:
        bearing = _bearing(available_families()[name](), arrangement, contact=contact, **geometry)
    except Exception as exc:                                              # noqa: BLE001
        pytest.skip(f"AxisForge cannot assemble {name}: {type(exc).__name__}: {exc}")
    try:
        cs = bearing_contacts(bearing, load=100.0)
    except NotImplementedError as exc:
        assert name in str(exc)
    else:
        assert [c.location.race for c in cs] == ["inner", "outer"] and cs[0].subtype == name


def _gear_pair(material_id="steel"):
    """A 20/40 spur pair, module 2 mm, face width 20 mm, as the old stub test built it."""
    from axisforge.core.machine_elements.gears import CylindricalFamily, CylindricalGearData, Gear, GearPair
    family = CylindricalFamily()
    return GearPair.assemble(
        family, gear1=Gear.assemble(family, CylindricalGearData(2.0, 20, b=20.0, material_id=material_id)),
        gear2=Gear.assemble(family, CylindricalGearData(2.0, 40, b=20.0, material_id=material_id)))


def _gear_inputs(pair):
    """AxisForge contact_inputs at A, C and E (gear 1 driving, 1000 rpm, 1 kN normal force).

    The module exists only on AxisForge refactor/gears-mechanical-system (GL-01): skip elsewhere."""
    contact = pytest.importorskip("axisforge.solvers.machine_elements.gears.contact")
    return contact.contact_inputs(pair, "gear1", 1000.0, 1000.0, n_points=3)


def test_gear_adapter_scaffold_raises():
    pair = _gear_pair()
    with pytest.raises(NotImplementedError, match="OPEN R-023"):
        gear_contacts(pair, _gear_inputs(pair), points=["A", "C", "E"])


@pytest.mark.parametrize("change, match", [
    ("points_length", "labels for 3 steps"),
    ("points_repeated", "not unique"),
    ("points_unknown", "not in"),
    ("rho1_not_positive", "rho1_n_mm must be > 0"),
    ("rho2_wrong_sign", "has not the sign of gear2.z"),
    ("material_empty", "material_id is empty"),
])
def test_gear_adapter_guards(change, match):
    pair = _gear_pair("" if change == "material_empty" else "steel")
    inputs = dict(_gear_inputs(pair))
    points = {"points_length": ["A", "E"], "points_repeated": ["A", "A", "E"],
              "points_unknown": ["A", "X", "E"]}.get(change, ["A", "C", "E"])
    if change == "rho1_not_positive":
        inputs["rho1_n_mm"] = np.array([0.0, 10.0, 20.0])   # what a check=False pair can give at A (R-030)
    if change == "rho2_wrong_sign":
        inputs["rho2_n_mm"] = np.array([30.0, 20.0, -1.0])  # concave gear 2 on an external pair (z2 > 0)
    with pytest.raises(ValueError, match=match):
        gear_contacts(pair, inputs, points=points)


def test_gear_material_given_only_for_a_gear_without_one():
    pair = _gear_pair()
    with pytest.raises(ValueError, match="given only for a gear without one"):
        gear_contacts(pair, _gear_inputs(pair), points=["A", "C", "E"], E=(206000.0, None), nu=(0.3, None))
    bare = _gear_pair("")
    with pytest.raises(NotImplementedError, match="OPEN R-023"):        # explicit E, nu pass the guard
        gear_contacts(bare, _gear_inputs(bare), points=["A", "C", "E"], E=(206000.0, 206000.0), nu=(0.3, 0.3))
    for bad in (206000.0, (206000.0,), "ab"):
        with pytest.raises(ValueError, match="must be"):
            gear_contacts(bare, _gear_inputs(bare), points=["A", "C", "E"], E=bad, nu=(0.3, 0.3))


# ---- bearing geometry ----------------------------------------------------------------------

def _curvature_sum(case):
    return sum(0.0 if math.isinf(r) else 1.0 / r for r in (*case.r1, *case.r2))


@pytest.mark.parametrize("race, sign", [("inner", +1.0), ("outer", -1.0)])
def test_ball_curvature_sum_matches_iso(ball, race, sign):
    """Signs and units: sum of curvatures = (1/Dw)(4 - 1/f +/- 2 gamma/(1 -/+ gamma)) (ISO/TS 16281)."""
    case = contact_case(ball, race, load=100.0)
    f = ball.ri / ball.Dw if race == "inner" else ball.re / ball.Dw
    g = ball.Dw * math.cos(ball.alpha_0) / ball.Dpw
    iso = (4.0 - 1.0 / f + sign * 2.0 * g / (1.0 - sign * g)) / (ball.Dw * 1e-3)
    assert _curvature_sum(case) == pytest.approx(iso, rel=1e-12)
    assert case.r2[0] * sign > 0 and case.r2[1] < 0


def test_alpha_default_is_family_angle(ball):
    assert contact_case(ball, "outer", 100.0, alpha=ball.alpha_0).r2 == contact_case(ball, "outer", 100.0).r2


def test_alpha_j_changes_only_the_rolling_radius(ball):
    a0, aj = contact_case(ball, "inner", 100.0), contact_case(ball, "inner", 100.0, alpha=0.4)
    assert aj.r1 == a0.r1 and aj.r2[1] == a0.r2[1]
    assert aj.r2[0] == pytest.approx((ball.Dpw - ball.Dw * math.cos(0.4)) / (2 * math.cos(0.4)) * 1e-3)


def test_default_label_is_the_bearing_label(ball):
    assert contact_case(ball, "inner", 1.0).label == f"{ball.label}_inner"


@pytest.mark.parametrize("kwargs", [dict(race="middle", load=1.0), dict(race="inner", load=0.0),
                                    dict(race="inner", load=1.0, alpha=math.pi / 2),
                                    dict(race="inner", load=1.0, length=2.0)])
def test_bad_input_raises(ball, kwargs):
    with pytest.raises(ValueError):
        contact_case(ball, **kwargs)


# ---- bearing contacts from the ISO/TS 16281 result -----------------------------------------

def test_ball_contacts_from_result(ball, ball_row):
    cs = bearing_contacts(ball, ball_row, races=("inner", "outer"), elements="loaded")
    assert len(cs) == 2 * 5                                         # 5 loaded balls x 2 races
    c0 = cs[0]
    assert isinstance(c0, Contact) and (c0.element, c0.subtype) == ("bearing", "DeepGrooveBallFamily")
    assert isinstance(c0.location, BearingLocation)
    assert asdict(c0.location) == asdict(BearingLocation(race="inner", alpha_rad=0.2, row=0, j=0, phi_rad=0.0))
    assert c0.case.load == 300.0
    assert c0.case.r2[0] == pytest.approx(contact_case(ball, "inner", 300.0, alpha=0.2).r2[0])
    assert bearing_contacts(ball, ball_row, elements="max", races=("outer",))[0].location.j == 0


def test_contacts_of_a_real_iso16281_solve(ball_6210, load_case):
    """Solver -> bridge on a solved bearing: both raceways of an element carry its Q_j and alpha_j.

    AxisForge has no centrifugal term, so the inner and the outer contact of a ball share the load
    and the operating angle; they differ only in the raceway radii. The combined load of the
    case (radial + axial), so that alpha_j differs from element to element and from alpha_0.
    """
    load = load_case("iso16281_ball_contact").LOADS["combined"]                                 # [N]
    result = ISO16281BallSolver().solve_contact(ball_6210, Fr_xz=load["Fr"], Fr_xy=0.0, Fa=load["Fa"],
                                                delta_r_init=0.0, delta_a_init=0.0, psi=0.0, phi_Fr=0.0)
    assert result.ok
    row = result.rows[0]
    loaded = np.flatnonzero(row.Q_j > 0.0)
    cs = bearing_contacts(ball_6210, result)
    assert len(loaded) > 0 and len(cs) == 2 * len(loaded)
    assert not np.allclose(row.alpha_j[loaded], ball_6210.alpha_0, rtol=0.01)
    for j, inner, outer in zip(loaded, cs[0::2], cs[1::2]):
        for race, c in (("inner", inner), ("outer", outer)):
            assert (c.location.race, c.location.j) == (race, j)
            assert c.case.load == row.Q_j[j] and c.location.alpha_rad == row.alpha_j[j]


def test_phi_is_wrapped_as_the_axisforge_element_record(ball, ball_result):
    """``phi_rad`` is AxisForge's own element record value: local angle wrapped to [-pi, pi)."""
    from axisforge.outputs.solvers.bearings.load_distribution.load_distribution_results import element_records
    by_j = {e["j"]: e["phi_rad"] for e in element_records(ball_result)}
    cs = bearing_contacts(ball, ball_result, races=("inner",))
    assert any(ball_result.rows[0].phi_j[c.location.j] >= math.pi for c in cs)     # some angles do wrap
    for c in cs:
        assert -math.pi <= c.location.phi_rad < math.pi and c.location.phi_rad == by_j[c.location.j]


def test_roller_laminae_from_result(roller, roller_row):
    whole = bearing_contacts(roller, roller_row, races=("inner",))
    assert [c.case.load for c in whole] == pytest.approx([500.0 / 5e-3, 200.0 / 5e-3])     # Q_j / Lwe
    lam = bearing_contacts(roller, roller_row, races=("inner",), laminae=True, elements=[1])
    assert len(lam) == 50 and lam[0].case.load == pytest.approx(4.0 / 0.1e-3)          # q_jk / (Lwe/n_s)
    assert lam[3].location.k == 3 and lam[0].case.line and lam[3].case.label == "b_inner_e1_l3"
    assert lam[3].location.x_mm == pytest.approx(roller_row.x_k[3])


def test_listed_unloaded_element_is_skipped_and_row_checked(ball, ball_row):
    assert [c.location.j for c in bearing_contacts(ball, ball_row, races=("inner",), elements=[0, 4])] == [0]
    with pytest.raises(ValueError):
        bearing_contacts(ball, ball_row, row=1)


@pytest.fixture(scope="module")
def ball_result(ball_row):
    """Single-row AxisForge ``BallBearingResult`` of the ``ball`` fixture (label "b"), converged."""
    return BallBearingResult.single(ball_row, label="b", delta_r=0.0, delta_a=0.0, psi=0.0, phi_Fr=0.5,
                                    Fr_xz=0.0, Fr_xy=0.0, Fa=0.0, n_iter=1, residual=0.0, ok=True)


def test_bearing_result_gives_global_angle(ball, ball_result):
    loc = bearing_contacts(ball, ball_result, races=("inner",), elements="max")[0].location
    assert loc.phi_global_rad == pytest.approx(ball_result.phi_j_global(0)[0])
    assert bearing_contacts(ball, ball_result.rows[0], races=("inner",))[0].location.phi_global_rad is None


def test_analysis_result_of_the_solver_is_accepted(ball, ball_result):
    """``SolverBase.solve`` returns a ``BearingAnalysisResult`` per label; the bridge reads its
    load distribution."""
    analysis = BearingAnalysisResult(label="b", load_distribution=ball_result)
    got, expected = bearing_contacts(ball, analysis), bearing_contacts(ball, ball_result)
    assert [(c.record(), c.case) for c in got] == [(c.record(), c.case) for c in expected]


def test_unconverged_result_is_refused(ball, ball_result):
    with pytest.raises(ValueError, match="not converged"):
        bearing_contacts(ball, replace(ball_result, ok=False, residual=12.0))


def test_result_of_another_bearing_is_refused(ball, ball_result):
    with pytest.raises(ValueError, match="not of bearing"):
        bearing_contacts(ball, replace(ball_result, label="other"))


def test_alpha_with_result_is_refused(ball, ball_row):
    with pytest.raises(ValueError, match="alpha"):
        bearing_contacts(ball, ball_row, alpha=0.1)


def test_load_path_records_the_angle_used(ball):
    loc = bearing_contacts(ball, load=100.0, races=("inner",))[0].location
    assert asdict(loc) == asdict(BearingLocation(race="inner", alpha_rad=ball.alpha_0))
    assert bearing_contacts(ball, load=100.0, alpha=0.3, races=("inner",))[0].location.alpha_rad == 0.3


def test_record_of_a_bearing_contact(roller, roller_row):
    """The record keys: AxisForge element and lamina record names, every location field always present."""
    c = bearing_contacts(roller, roller_row, races=("outer",), laminae=True, elements=[0], n_inner_rpm=1000.0)[0]
    assert list(c.record()) == ["element", "subtype", "race", "alpha_rad", "row", "j", "phi_rad",
                                "phi_global_rad", "k", "x_mm", "label", "case_hash", "line",
                                "ue_m_per_s", "vg_m_per_s", "u1_m_per_s", "u2_m_per_s"]
    rec = bearing_contacts(roller, load=100.0, races=("inner",))[0].record()
    assert rec["j"] is None and "ue_m_per_s" not in rec


def test_load_or_result_not_both(ball, ball_row):
    with pytest.raises(ValueError):
        bearing_contacts(ball, ball_row, load=1.0)
    with pytest.raises(ValueError):
        bearing_contacts(ball)


def test_kinematics_only_when_speeds_are_given(ball):
    assert bearing_contacts(ball, load=100.0)[0].kinematics is None
    k = bearing_contacts(ball, load=100.0, n_inner_rpm=3000.0)[0].kinematics
    g = ball.Dw * math.cos(ball.alpha_0) / ball.Dpw
    assert isinstance(k, Kinematics) and (k.vg_m_per_s, k.u1_m_per_s, k.u2_m_per_s) == (0.0, None, None)
    with pytest.raises(TypeError):                                      # no silent pure rolling
        Kinematics(ue_m_per_s=1.0)
    assert k.ue_m_per_s == pytest.approx(31e-3 / 4 * (1 - g**2) * 3000 * 2 * math.pi / 60)
    assert k.ue_m_per_s == _entrainment_speed(ball.Dpw, ball.Dw, ball.alpha_0, 3000.0, 0.0)


# ---- reference, grid, measurements ----------------------------------------------------------

def test_hertz_matches_the_closed_forms():
    """Circular point contact and line contact against Johnson (1985), ch. 4.

    Contact modulus ``1/E* = (1 - nu1^2)/E1 + (1 - nu2^2)/E2``. Sphere of radius R on a flat, load Q:
    ``a = (3 Q R / (4 E*))**(1/3)``, ``p0 = 3 Q / (2 pi a^2)``, ``delta = a^2 / R``. Cylinder of
    radius R on a flat, load per length w: ``b = sqrt(4 w R / (pi E*))``, ``p0 = 2 w / (pi b)``; the
    contact is infinite along the cylinder axis (y) and the approach is not defined (plane strain).
    """
    R, E, nu = 6.35e-3, (2.1e11, 0.7e11), (0.3, 0.33)        # [m], [Pa], [-]; two materials
    e_star = 1.0 / ((1 - nu[0]**2) / E[0] + (1 - nu[1]**2) / E[1])

    Q = 500.0                                                 # [N]
    ref = hertz(ContactCase("sphere", r1=R, r2=math.inf, E=E, nu=nu, load=Q))
    a = (3 * Q * R / (4 * e_star)) ** (1 / 3)
    assert (ref["s_x_m"], ref["s_y_m"]) == pytest.approx((a, a), rel=1e-6)
    assert ref["p0_Pa"] == pytest.approx(3 * Q / (2 * math.pi * a**2), rel=1e-6)
    assert ref["delta_m"] == pytest.approx(a**2 / R, rel=1e-6)

    w = 5.0e4                                                 # [N/m]
    ref = hertz(ContactCase("cylinder", r1=(R, math.inf), r2=math.inf, E=E, nu=nu, load=w, line=True))
    b = math.sqrt(4 * w * R / (math.pi * e_star))
    assert ref["s_x_m"] == pytest.approx(b, rel=1e-9) and math.isinf(ref["s_y_m"])
    assert ref["p0_Pa"] == pytest.approx(2 * w / (math.pi * b), rel=1e-9)
    assert math.isnan(ref["delta_m"])


@pytest.mark.parametrize("race", ["inner", "outer"])
def test_independent_hertz_agrees_with_slippy(ball_6210, race):
    """The judge (``references.contact.hertz_point``, Johnson ch. 4) against SlipPY's ``hertz_full``
    on the bridge axes, for the elliptical contact of a ball with each raceway (the outer one has a
    negative rolling radius). Two independent implementations of the same theory."""
    case = contact_case(ball_6210, race, load=2436.0)         # [N], Q_max of the bearing at Fr = 5 kN
    judge, ref = hertz_point(case.r1, case.r2, case.E, case.nu, case.load), hertz(case)
    for key in ("p0_Pa", "s_x_m", "s_y_m", "delta_m"):
        assert judge[key] == pytest.approx(ref[key], rel=1e-6), key


def test_axisforge_cp_is_hertz_with_a_rounded_constant(ball_6210):
    """Implementation check, same theory: AxisForge's ``cp`` against the judge's Hertz approach.

    This does not test whether Hertz holds; it tests that AxisForge codes the Hertz load-deflection
    constant of a ball (``Q = cp delta**1.5``, delta = approach of the two raceways, ISO/TS 16281)
    as the theory gives it, and that the bridge hands the judge the radii AxisForge used. With the
    contact cases at the free angle alpha_0 (where AxisForge evaluates the curvatures), the judge
    gives ``cp_judge = Q / (delta_inner + delta_outer)**1.5`` [N/mm^1.5], independent of Q.

    Finding (flagged, not fixed): the Hertz approach rearranges to
    ``Q = (pi sqrt(2) / 3) E' (T_i + T_e)**(-3/2) delta**1.5`` with pi sqrt(2)/3 = 1.48096, and
    AxisForge has the constant 1.48 (``contact_models/iso16281.py:104``): cp is 0.065 % low, the
    approach 0.043 % high. Whether ISO/TS 16281 prints 1.48 has not been checked. The test removes
    the constant and requires the rest to agree to 1e-6.

    Per raceway, ``PointContactStiffness`` is rebuilt from the bearing's public attributes (as
    ``families/family.py`` does) and its curvature sum [1/mm] and ellipticity ``chi`` are compared
    with the curvature sum of the bridge radii and the judge's semi-axis ratio.
    """
    b, Q = ball_6210, 2436.0                                  # [N]
    stiffness = PointContactStiffness(b.Dw, b.Dpw, b.gamma, b.ri, b.re, b.alpha_0, b.e1, b.e2, b.nu1, b.nu2)
    approach_mm = 0.0
    for race in ("inner", "outer"):
        case = contact_case(b, race, Q)                       # radii at alpha_0
        judge = hertz_point(case.r1, case.r2, case.E, case.nu, case.load)
        approach_mm += judge["delta_m"] * 1e3                 # [m] -> [mm], AxisForge length unit
        assert _curvature_sum(case) * 1e-3 == pytest.approx(getattr(stiffness, f"curvature_sum_{race}"), rel=1e-6)
        # a groove contact is long across rolling: chi = s_y / s_x >= 1
        assert judge["s_y_m"] / judge["s_x_m"] == pytest.approx(getattr(stiffness, f"chi_{race}"), rel=1e-6)
    cp_judge = Q / approach_mm**1.5                           # [N/mm^1.5]
    assert b.cp * (math.pi * math.sqrt(2.0) / 3.0) / 1.48 / cp_judge == pytest.approx(1.0, rel=1e-6)


@pytest.mark.parametrize("r1, x_longer", [((50e-3, 5e-3), True), ((5e-3, 50e-3), False)])
def test_semi_axes_follow_the_relative_radii(r1, x_longer):
    s_x, s_y = semi_axes(ContactCase("e", r1=r1, r2=math.inf, E=2.1e11, nu=0.3, load=100.0))
    assert (s_x > s_y) is x_longer


@pytest.mark.parametrize("r1, line_axis", [((5e-3, math.inf), 1), ((math.inf, 5e-3), 0)])
def test_line_axis_follows_the_cylinder(r1, line_axis):
    s = semi_axes(ContactCase("c", r1=r1, r2=math.inf, E=2.1e11, nu=0.3, load=1e5, line=True))
    assert math.isinf(s[line_axis]) and math.isfinite(s[1 - line_axis])


@pytest.mark.parametrize("r1, line", [((5e-3, 5e-3), True),           # sphere flagged as a line
                                      ((5e-3, math.inf), False)])      # cylinder not flagged
def test_line_flag_inconsistent_with_radii_raises(r1, line):
    from slippy_bridge.contact.reference import _oriented
    case = ContactCase("c", r1=r1, r2=math.inf, E=2.1e11, nu=0.3, load=1e5, line=line)
    with pytest.raises(ValueError):                           # SlipPY's hertz_full refuses it first
        semi_axes(case)
    with pytest.raises(ValueError, match="case.line"):        # the bridge's own guard, on any reference
        _oriented(case, {"contact_radii": (1e-4, 2e-4)})


def test_grid_rolling_axis_and_dyadic_spacing(ball):
    g = Grid.around(contact_case(ball, "inner", 300.0))
    assert math.log2(g.h) == round(math.log2(g.h))
    assert np.all(np.diff(g.X[:, 0]) > 0) and np.all(g.X[:, 0] == g.X[:, -1])
    assert g.X[g.X.shape[0] // 2, 0] == 0.0 and g.Y[0, g.Y.shape[1] // 2] == 0.0


def test_hand_built_grid_is_checked(ball):
    g = Grid.around(contact_case(ball, "inner", 300.0))
    assert Grid(X=g.X, Y=g.Y, h=g.h * (1 + 1e-12)).h > g.h            # not dyadic, within tolerance
    for bad in (dict(X=g.X, Y=g.Y, h=2 * g.h),                        # wrong spacing
                dict(X=g.Y, Y=g.X, h=g.h),                            # rolling along axis 1
                dict(X=g.X, Y=g.Y[:, :-1], h=g.h),                    # shapes differ
                dict(X=g.X[::-1], Y=g.Y, h=g.h),                      # rolling direction reversed
                dict(X=g.X[::-1], Y=g.Y[:, ::-1], h=-g.h)):           # negative spacing
        with pytest.raises(ValueError):
            Grid(**bad)


def test_measure_finds_hertz_semi_axes_below_one_node(ball):
    case = contact_case(ball, "inner", 300.0)
    g = Grid.around(case)
    ref = hertz(case)
    m = measure(ref["pressure"](g.X, g.Y), g.h)
    assert abs(m["s_x_m"] - ref["s_x_m"]) < 0.5 * g.h and abs(m["s_y_m"] - ref["s_y_m"]) < 0.5 * g.h
    assert m["apex_offset_nodes"] == 0.0


def test_measure_gap_scaffold_raises(ball):
    case = contact_case(ball, "inner", 300.0)
    g = Grid.around(case)
    p = hertz(case)["pressure"](g.X, g.Y)
    with pytest.raises(NotImplementedError, match="OPEN R-033"):
        measure(p, g.h, gap=np.zeros_like(p), X=g.X, Y=g.Y)


def test_around_span_tuple_scaffold_raises(ball):
    with pytest.raises(NotImplementedError, match="OPEN R-048"):
        Grid.around(contact_case(ball, "inner", 300.0), span_x=(6.0, 2.0))


def test_grid_beyond_the_body_raises():
    case = ContactCase("tiny", r1=1e-4, r2=math.inf, E=2.1e11, nu=0.3, load=1e-3)
    g = Grid.around(case, span_x=200.0, span_y=200.0)
    with pytest.raises(ValueError):
        pair(case, g)


# ---- roughness -----------------------------------------------------------------------------

def test_fractal_rms_and_seed():
    a = rough.fractal((64, 32), 1e-6, sigma=0.2e-6, hurst=0.8, seed=1)
    assert a.shape == (64, 32) and a.std() == pytest.approx(0.2e-6) and abs(a.mean()) < 1e-15
    assert np.array_equal(a, rough.fractal((64, 32), 1e-6, sigma=0.2e-6, hurst=0.8, seed=1))


def test_roughness_composite_and_two_profiles(ball):
    import slippy.surface as s
    case = contact_case(ball, "inner", 300.0)
    g = Grid.around(case)
    z_rough_1, z_rough_2 = np.full(g.X.shape, 1e-7), np.full(g.X.shape, 2e-7)
    smooth = pair(case, g)
    s1, s2 = pair(case, g, rough=(z_rough_1, z_rough_2))
    assert np.allclose(s1.profile - smooth[0].profile, 3e-7) and np.allclose(s2.profile, 0.0)
    t1, t2 = pair(case, g, build="two_profiles", rough=(z_rough_1, z_rough_2))
    two = pair(case, g, build="two_profiles")
    assert np.allclose(t1.profile - two[0].profile, 1e-7) and np.allclose(t2.profile - two[1].profile, 2e-7)
    # every call makes new Elastic objects, so the properties are compared, not the objects;
    # two different materials, so that a swap of the bodies would show
    mixed = replace(case, E=(2.1e11, 0.7e11), nu=(0.3, 0.33))
    for build in ("composite", "two_profiles"):
        m1, m2 = (surface.material for surface in pair(mixed, g, build=build, rough=(z_rough_1, z_rough_2)))
        assert (m1.E, m1.v, m2.E, m2.v) == (2.1e11, 0.3, 0.7e11, 0.33)
    as_surface = pair(case, g, rough=(s.Surface(profile=z_rough_1, grid_spacing=g.h), z_rough_2))
    assert np.array_equal(as_surface[0].profile, s1.profile)
    for bad in (dict(rough=(np.zeros((3, 3)), None)),
                dict(rough=(s.Surface(profile=z_rough_1, grid_spacing=2 * g.h), None)),
                dict(build="equivalent")):
        with pytest.raises(ValueError):
            pair(case, g, **bad)


@pytest.mark.parametrize("build", ["composite", "two_profiles"])
def test_kernel_flag_reaches_both_materials(ball, build):
    """SlipPY stores ``use_frequency_domain`` as ``Material.use_frequency`` (``slippy/core/materials.py:37-40``)."""
    case = contact_case(ball, "inner", 300.0)
    g = Grid.around(case)
    assert all(s.material.use_frequency for s in pair(case, g, build=build))          # SlipPY default kept
    assert not any(s.material.use_frequency for s in pair(case, g, build=build, use_frequency_domain=False))


def test_line_contact_refuses_the_spatial_kernel():
    """On the periodic line axis the spatial kernel spans one period: a finite strip, not plane strain."""
    case = ContactCase("roller", r1=(5e-3, math.inf), r2=(30e-3, math.inf), E=210e9, nu=0.3, load=5e4, line=True)
    with pytest.raises(ValueError, match="use_frequency_domain"):
        pair(case, Grid.around(case), use_frequency_domain=False)


# ---- static step, end to end ---------------------------------------------------------------

@pytest.mark.parametrize("race", ["inner", "outer"])
def test_static_step_end_to_end(ball, race, tmp_path):
    """AxisForge bearing -> both builds -> SlipPY static BEM: p0 vs Hertz, builds agree, checks true."""
    pytest.importorskip("pyfftw")
    c = bearing_contacts(ball, load=300.0, races=(race,))[0]
    g = Grid.around(c.case)
    recs = {b: static_step.run(c, g, tmp_path, build=b)[0] for b in ("composite", "two_profiles")}
    rc = recs["composite"]
    assert rc["element"] == "bearing" and rc["race"] == race
    assert (rc["build"], rc["use_frequency_domain"]) == ("composite", True)
    assert (rc["method"], rc["tolerance"]) == ("pk", 1e-9) and math.isnan(rc["length_m"])
    assert (rc["R_x_m"], rc["R_y_m"]) == relative_radii(c.case)
    assert math.isfinite(rc["delta_m"])
    assert abs(rc["err_p0"]) < 0.02
    assert all(rc[k] for k in ("converged", "load_ok", "edge_clear", "margin_ok", "resolution_ok",
                               "within_reference"))
    assert recs["two_profiles"]["p0_Pa"] == pytest.approx(rc["p0_Pa"], rel=0.01)


def test_static_step_line_contact_has_no_approach(roller, tmp_path):
    """Line contact: the approach needs a datum in each body (Johnson1985 §4.2(c), Eq. (4.37)), so
    ``delta_m`` is nan; the load is the load per length times the periodic length ``n_y h``."""
    pytest.importorskip("pyfftw")
    c = bearing_contacts(roller, load=500.0, races=("inner",))[0]
    g = Grid.around(c.case)
    rec, _ = static_step.run(c, g, tmp_path)
    assert math.isnan(rec["delta_m"]) and math.isnan(rec["delta_ref_m"]) and math.isnan(rec["err_delta"])
    assert rec["length_m"] == pytest.approx(rec["n_y"] * rec["h_m"])
    assert rec["load_N_applied"] == pytest.approx(c.case.load * rec["length_m"])
    assert rec["converged"] and rec["within_reference"]


def test_static_step_flags_a_short_domain(ball, tmp_path):
    pytest.importorskip("pyfftw")
    case = contact_case(ball, "inner", 300.0)
    g = Grid.around(case, span_x=2.0)
    rec, _ = static_step.run(case, g, tmp_path)
    assert rec["edge_clear"] and not rec["margin_ok"]


def test_step_args_line_load_and_periodic_flags():
    """Line contact: the step gets w n_y h [N] and the strip periodic along the roller axis only."""
    case = ContactCase("roller", r1=(5e-3, math.inf), r2=(30e-3, math.inf), E=210e9, nu=0.3, load=5e4, line=True)
    g = Grid.around(case)
    args = g.step_args(case, unloading=False)
    assert args == {"normal_load": pytest.approx(5e4 * g.X.shape[1] * g.h), "periodic_geometry": True,
                    "periodic_axes": (False, True), "unloading": False}
    assert g.length_y_m == g.X.shape[1] * g.h
    assert g.step_args(case, normal_load=3.0)["normal_load"] == 3.0
    point = ContactCase("ball", r1=6.35e-3, r2=math.inf, E=210e9, nu=0.3, load=500.0)
    assert Grid.around(point).step_args(point) == {"normal_load": 500.0}


@pytest.mark.parametrize("contrary", [{"periodic_geometry": False}, {"periodic_axes": (False, False)}])
def test_step_args_refuses_a_non_periodic_line(contrary):
    case = ContactCase("roller", r1=(5e-3, math.inf), r2=(30e-3, math.inf), E=210e9, nu=0.3, load=5e4, line=True)
    with pytest.raises(ValueError, match="line contact needs"):
        Grid.around(case).step_args(case, **contrary)


def test_static_step_refuses_surfaces_off_the_grid(ball, tmp_path):
    """SlipPY solves on the spacing of the surfaces; surfaces of another grid would be solved silently."""
    case = contact_case(ball, "inner", 300.0)
    g = Grid.around(case)
    with pytest.raises(ValueError, match="grid_spacing"):
        static_step.run(case, g, tmp_path, surfaces=pair(case, Grid.around(case, h=g.h / 2)))
    with pytest.raises(ValueError, match="cannot be given with surfaces"):
        static_step.run(case, g, tmp_path, surfaces=pair(case, g), build="two_profiles")
    import slippy.surface as ss
    bare = [ss.Surface(profile=surface.profile, grid_spacing=g.h) for surface in pair(case, g)]   # material None
    with pytest.raises(ValueError, match="no material"):
        static_step.run(case, g, tmp_path, surfaces=bare)


def test_given_spatial_kernel_surfaces_refused_for_a_line(tmp_path):
    """Same rule as ``pair``: on the periodic line axis the spatial kernel is a finite strip."""
    line = ContactCase("roller", r1=(5e-3, math.inf), r2=(30e-3, math.inf), E=210e9, nu=0.3, load=5e4, line=True)
    g = Grid.around(line)
    point = replace(line, r1=(5e-3, 5e-3), line=False)          # spatial-kernel surfaces on the same grid
    with pytest.raises(ValueError, match="use_frequency_domain=True"):
        static_step.run(line, g, tmp_path, surfaces=pair(point, g, use_frequency_domain=False))


def test_static_step_records_given_surfaces_and_their_kernel(ball, tmp_path):
    pytest.importorskip("pyfftw")
    case = contact_case(ball, "inner", 300.0)
    g = Grid.around(case, h=semi_axes(case)[0] / 4.0)       # coarse: a fast solve
    rec, _ = static_step.run(case, g, tmp_path, surfaces=pair(case, g, use_frequency_domain=False))
    assert (rec["build"], rec["use_frequency_domain"]) == ("given", False)


def test_contact_case_hash_ignores_label():
    a = ContactCase("x", r1=1e-3, r2=math.inf, E=2e11, nu=0.3, load=1.0)
    assert a.case_hash() == ContactCase("y", r1=1e-3, r2=math.inf, E=2e11, nu=0.3, load=1.0).case_hash()


def test_contact_case_keeps_slippy_keys_and_the_saved_hash():
    """E and nu go out under SlipPY's names, so the hash of the saved results does not change."""
    # the ball_flat case of validation/contact/hertz_point_grid_domain: 12.7 mm steel ball on a steel flat, 500 N
    case = ContactCase("ball_flat", r1=6.35e-3, r2=math.inf, E=210e9, nu=0.3, load=500.0)
    assert {"moduli", "v"} <= set(case.to_dict()) and {"moduli", "v"} <= set(case.hertz_args())
    assert case.case_hash() == "9ccc2ddabd085997"

# ---- lubrication: ehl.run guards (the solve itself is a scaffold) --------------------------

# 12.7 mm steel ball on a steel flat, 500 N; an oil of 0.1 Pa s with the Roelands constants of
# SlipPY's own EHL test (z 0.68, p0 = 1/5.1e-9 Pa), rolling at a mean 2 m/s
_EHL_CASE = ContactCase("ehl_ball_flat", r1=6.35e-3, r2=math.inf, E=210e9, nu=0.3, load=500.0)
_EHL_OK = dict(eta0_Pa_s=0.1, rho0_kg_per_m3=870.0, viscosity=("roelands", 0.68, 1.0 / 5.1e-9),
               ue_m_per_s=2.0)


@pytest.mark.parametrize("change, exc, match", [
    ({}, NotImplementedError, "OPEN R-076"),                                  # valid inputs reach the scaffold
    ({"output_dir": None}, ValueError, "output_dir"),
    ({"viscosity": ("roelands", 0.68)}, ValueError, "viscosity must be"),
    ({"viscosity": ("roelands", 0.68, 196.0)}, ValueError, "in MPa"),        # p0 given in MPa
    ({"ue_m_per_s": 0.0}, ValueError, "> 0"),
    ({"interference": 1e-6}, ValueError, "load-controlled"),
    ({"number_of_steps": 2}, ValueError, "steady"),
    ({"surfaces": "other_h"}, ValueError, "grid_spacing"),
])
def test_ehl_run_guards(change, exc, match, tmp_path):
    g = Grid.around(_EHL_CASE)
    kwargs = {**_EHL_OK, **change}
    output_dir = kwargs.pop("output_dir", tmp_path)
    if kwargs.get("surfaces") == "other_h":
        # surfaces sampled at half the solve spacing (R-085)
        kwargs["surfaces"] = pair(_EHL_CASE, Grid.around(_EHL_CASE, h=g.h / 2))
    with pytest.raises(exc, match=match):
        ehl.run(_EHL_CASE, g, output_dir, **kwargs)


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="scaffold OPEN GL-28 GL-16 GL-34 R-091")
def test_ehl_run_line_smoke(load_case, tmp_path):
    """Smoke of a line EHL solve: Venner & Bos M = 20, L = 10, incompressible, H_c = 9.923e-2 +-10 %.

    The reference is the finest-level central film of VennerBos1994 Table 1 (``line_contact`` CSV L45), in
    Venner's scaling H = h R / b^2. The tolerance is a smoke band, not the case criterion N1 (2 %): SlipPY's
    line contact is a finite periodic strip (GL-16). The solver settings are pinned to keep the run short
    (GL-28), and the record must say on what grid it ran (n_x, n_y, h) and how long it took.
    """
    cc = load_case("line_contact")
    E = e_prime(cc.E_PA, cc.NU, cc.E_PA, cc.NU)
    point = moes_inverse(M=cc.M_LOW, L=cc.L_MOES, line=True, E_prime_Pa=E["E_prime_Pa"], R_x_m=cc.R_X_M,
                         u_mean_m_per_s=cc.U_MEAN_M_PER_S, roelands_z=cc.ROELANDS_Z, roelands_p0_Pa=cc.ROELANDS_P0_PA)
    case = ContactCase("vb_line_M20_smoke", r1=(cc.R_X_M, math.inf), r2=math.inf, E=cc.E_PA, nu=cc.NU,
                       load=point["w_N_per_m"], line=True)
    grid = Grid.around(case, span_x=cc.SPAN_X_LOW)                   # (inlet, outlet) 4 b and 1.5 b (CSV L56)
    rec, p, gap = ehl.run(case, grid, tmp_path, eta0_Pa_s=point["eta0_Pa_s"], rho0_kg_per_m3=cc.RHO0_KG_PER_M3,
                          viscosity=("roelands", cc.ROELANDS_Z, cc.ROELANDS_P0_PA), density="incompressible",
                          ue_m_per_s=cc.U_MEAN_M_PER_S, rtol_pressure=1e-4, relaxation_factor=0.05)
    assert {"n_x", "n_y", "h_m", "solve_time_s"} <= rec.keys()      # grid, domain (n h) and run time recorded
    U_mean = groups(u_mean_m_per_s=cc.U_MEAN_M_PER_S, eta0_Pa_s=point["eta0_Pa_s"], E_prime_Pa=E["E_prime_Pa"],
                    R_x_m=cc.R_X_M, alpha_per_Pa=point["alpha_initial_per_Pa"], w_N_per_m=point["w_N_per_m"])["U_mean"]
    H_c = film_scalings(h_m=rec["h_c_m"], R_x_m=rec["R_x_m"], s_x_ref_m=rec["s_x_ref_m"], U_mean=U_mean)["H_venner"]
    assert H_c == pytest.approx(9.923e-2, rel=0.10)                 # line_contact CSV L45


# ---- FZG C40: AxisForge -> gear adapter -> Hertz (validation/gears/fzg_geometry_ph) -------------

@pytest.fixture
def fzg_c40(load_case, monkeypatch):
    """The FZG C40 pair of the case, with a steel registered for this test only.

    E 206 GPa and nu 0.3 are the values of the R-117 hypothesis (no held source, R-118); the density is not
    read by AxisForge. monkeypatch removes the registry entry after the test, so no other test sees it.
    """
    pytest.importorskip("axisforge.solvers.machine_elements.gears.contact")   # c328999 only (GL-01)
    from axisforge.core.materials import IsotropicElastic, Material
    from axisforge.core.materials import base as materials_base
    cc = load_case("fzg_geometry_ph")
    steel = Material(material_id=cc.STEEL_ID, density=7.85e3, elastic=IsotropicElastic(E=206000.0, poisson_ratio=0.3))
    # AxisForge has no public unregister (materials/base.py:298-320); private _REGISTRY patched, restored by monkeypatch
    monkeypatch.setitem(materials_base._REGISTRY, cc.STEEL_ID, steel)
    return cc, cc.build_pair("C40")


def _fzg_p_H_MPa(cc, pair, stage, points):
    """Hertz p_H [MPa] of C40 at ``points`` for FZG ``stage``: wheel torque, speed to gear 1 (R-124); gear 1 drives."""
    from axisforge.solvers.machine_elements.gears.contact import contact_inputs
    z1, z2 = pair.gear1.z, pair.gear2.z
    n1_rpm = cc.WHEEL_SPEED_RPM[0] * z2 / z1                         # 200 rpm on the wheel (CSV L198)
    F_bn_N = cc.FAMILY.mesh_forces(pair, cc.WHEEL_TORQUE_NM[stage] * z1 / z2, rot1=+1).F_bn_N
    times = cc.path_times(pair, n1_rpm)
    inputs = contact_inputs(pair, cc.DRIVER, n1_rpm, F_bn_N, t_s=[times[k] for k in points])
    return [hertz(c.case)["p0_Pa"] / 1e6 for c in gear_contacts(pair, inputs, points=points)]   # Pa -> MPa


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="scaffold OPEN R-023 R-031 (gear adapter)")
def test_fzg_ph_load_ratios(fzg_c40):
    """V-GEAR-04 criterion P1: p_H(K7)/p_H(K5) = 685/498 = 1.3755, p_H(K9)/p_H(K5) = 874/498 = 1.7550, +-0.2 %.

    Martins2015 Table 4 (``fzg_geometry_ph`` CSV L174, L184, L194). Hertz p_H grows as the square root of the
    load whatever E and the point, so the ratio checks the stage torques and p ~ sqrt(w) only; 0.2 % is the
    rounding of the printed integers (0.5/498 + 0.5/685 = 0.17 %).
    """
    cc, pair = fzg_c40
    points = ["A", "B+", "C"]
    p5, p7, p9 = (_fzg_p_H_MPa(cc, pair, stage, points) for stage in ("K5", "K7", "K9"))
    for a, b, c in zip(p5, p7, p9):
        assert b / a == pytest.approx(685.0 / 498.0, rel=2e-3)
        assert c / a == pytest.approx(874.0 / 498.0, rel=2e-3)


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="scaffold OPEN R-023 R-031 (gear adapter)")
def test_hypothesis_R117_ph_at_A_half_load(fzg_c40):
    """Hypothesis R-117, not a pass criterion; the tolerance depends on E (R-118).

    FEUP's p_H 498 MPa at K5 (Martins2015 Table 4, ``fzg_geometry_ph`` CSV L174) does not state its point. The
    hypothesis: it is the Hertz pressure at A, where two tooth pairs share the load (share 0.5), with the
    pinion driving (GL-30). E 206 GPa, nu 0.3 and the wheel d_a 118.54 (CSV L110, AxisForge from x) are
    assumed; 210 GPa would give about +0.8 %.
    """
    cc, pair = fzg_c40
    (p_A,) = _fzg_p_H_MPa(cc, pair, "K5", ["A"])
    assert p_A == pytest.approx(498.0, rel=0.01)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="known gap GL-10: AxisForge 1.4624 vs printed 1.44")
def test_fzg_c40_eps_alpha_known_gap(load_case):
    """C40 transverse contact ratio: printed 1.44 (Fernandes2015b CSV L13 = Martins2015 L111), AxisForge 1.4624.

    A known gap (GL-10) kept visible: it fails today with AssertionError; an XPASS would mean it closed.
    """
    pytest.importorskip("axisforge.solvers.machine_elements.gears.contact")   # the case needs c328999 (GL-01)
    pair = load_case("fzg_geometry_ph").build_pair("C40")
    assert pair.epslon_alpha == pytest.approx(1.44, abs=0.01)


@pytest.mark.parametrize("stage, F_r_printed_N", [
    ("K5", 790.0),                  # fzg_geometry_ph CSV L172
    pytest.param("K7", 1495.0,      # CSV L182
                 marks=pytest.mark.xfail(strict=True, raises=AssertionError,
                                         reason="GL-41: 1494.50 vs 1495, 0.004 N over the print rounding")),
    ("K9", 2432.0),                 # CSV L192
])
def test_fzg_c40_radial_force(load_case, stage, F_r_printed_N):
    """P0 (GL-41, GL-45): C40 radial force F_r = F_tw tan(alpha_wt) against Martins2015 Table 4, within 0.5 N.

    0.5 N is the rounding of the printed integer. The check covers the wheel -> pinion torque (T1 = T2 z1/z2,
    R-124), the working pitch diameter and the working pressure angle; it needs no gear adapter.
    """
    pytest.importorskip("axisforge.solvers.machine_elements.gears.contact")   # the case needs c328999 (GL-01)
    cc = load_case("fzg_geometry_ph")
    pair = cc.build_pair("C40")
    T1_Nm = cc.WHEEL_TORQUE_NM[stage] * pair.gear1.z / pair.gear2.z            # CSV L168/178/188, on the wheel
    F_r_N = cc.FAMILY.mesh_forces(pair, T1_Nm, rot1=+1).F_r_N
    assert F_r_N == pytest.approx(F_r_printed_N, abs=0.5)
