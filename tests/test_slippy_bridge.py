"""slippy_bridge.contact: AxisForge bearing -> Contact -> surfaces -> static step."""
import math
from dataclasses import replace

import numpy as np
import pytest

af_c = pytest.importorskip("axisforge.core")
pytest.importorskip("slippy")

from axisforge.core.machine_elements.bearings.contact_models.iso16281 import PointContactStiffness  # noqa: E402
from axisforge.results import BallLoadDistributionResult, RollerLoadDistributionResult  # noqa: E402
from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281.contact_solver import (  # noqa: E402
    ISO16281BallSolver)

from references.contact.hertz import hertz_point  # noqa: E402
from slippy_bridge.contact import (BearingContacts, Contact, ContactCase, Grid, contact_case,  # noqa: E402
                                   contacts, entrainment_speed, hertz, measure, pair, register,
                                   registered, rough, semi_axes, static_step)

CONTACT = dict(e1=210000.0, e2=210000.0, nu1=0.3, nu2=0.3)


def _bearing(family, arrangement, **geometry):
    catalog = af_c.BearingCatalog(d=20.0, D=42.0, b=12.0, designation="test", position=0.0,
                                  arrangement=arrangement, label="b")
    return af_c.Bearing.assemble(family=family, catalog=catalog,
                                 geometry={**geometry, "contact": af_c.ContactAnalysis.ISO16281, **CONTACT})


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


# ---- registry ------------------------------------------------------------------------------

def test_every_axisforge_bearing_family_is_registered():
    from axisforge.core.machine_elements.bearings.families.family import _FAMILY_REGISTRY
    reg = registered()
    assert set(_FAMILY_REGISTRY) == set(reg["bearing"]) and len(reg["bearing"]) >= 8
    assert "CylindricalFamily" in reg["gear"]


def test_gear_adapter_not_written_yet_raises():
    from axisforge.core.machine_elements.gears import CylindricalFamily, CylindricalGearData, Gear, GearPair
    family = CylindricalFamily()
    pair = GearPair.assemble(family, gear1=Gear.assemble(family, CylindricalGearData(2.0, 20)),
                             gear2=Gear.assemble(family, CylindricalGearData(2.0, 40)))
    with pytest.raises(NotImplementedError, match="adapter not written yet"):
        contacts(pair)


def test_unregistered_subtype_raises():
    with pytest.raises(NotImplementedError, match="no SlipPY adapter"):
        contacts(object())


def test_register_refuses_what_is_not_an_adapter():
    class Dummy:
        pass

    before = registered()
    for not_an_adapter in (type("Plain", (), {}), lambda obj: []):
        with pytest.raises(TypeError):
            register(Dummy)(not_an_adapter)
    assert registered() == before
    with pytest.raises(NotImplementedError):
        contacts(Dummy())


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
    cs = BearingContacts.contacts(ball, ball_row, races=("inner", "outer"), elements="loaded")
    assert len(cs) == 2 * 5                                         # 5 loaded balls x 2 races
    c0 = cs[0]
    assert isinstance(c0, Contact) and (c0.element, c0.subtype) == ("bearing", "DeepGrooveBallFamily")
    assert c0.location == {"race": "inner", "row": 0, "element_j": 0, "phi_local_rad": 0.0, "alpha_rad": 0.2}
    assert c0.case.load == 300.0
    assert c0.case.r2[0] == pytest.approx(contact_case(ball, "inner", 300.0, alpha=0.2).r2[0])
    assert contacts(ball, ball_row, elements="max", races=("outer",))[0].location["element_j"] == 0


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
    cs = contacts(ball_6210, result)
    assert len(loaded) > 0 and len(cs) == 2 * len(loaded)
    assert not np.allclose(row.alpha_j[loaded], ball_6210.alpha_0, rtol=0.01)
    for j, inner, outer in zip(loaded, cs[0::2], cs[1::2]):
        for race, c in (("inner", inner), ("outer", outer)):
            assert (c.location["race"], c.location["element_j"]) == (race, j)
            assert c.case.load == row.Q_j[j] and c.location["alpha_rad"] == row.alpha_j[j]


def test_roller_laminae_from_result(roller, roller_row):
    whole = contacts(roller, roller_row, races=("inner",))
    assert [c.case.load for c in whole] == pytest.approx([500.0 / 5e-3, 200.0 / 5e-3])     # Q_j / Lwe
    lam = contacts(roller, roller_row, races=("inner",), laminae=True, elements=[1])
    assert len(lam) == 50 and lam[0].case.load == pytest.approx(4.0 / 0.1e-3)          # q_jk / (Lwe/n_s)
    assert lam[3].location["lamina_k"] == 3 and lam[0].case.line
    assert lam[3].location["x_k_mm"] == pytest.approx(roller_row.x_k[3])


def test_listed_unloaded_element_is_skipped_and_row_checked(ball, ball_row):
    assert [c.location["element_j"] for c in contacts(ball, ball_row, races=("inner",), elements=[0, 4])] == [0]
    with pytest.raises(ValueError):
        contacts(ball, ball_row, row=1)


def test_multirow_result_gives_global_angle(ball, ball_row):
    from axisforge.results import BallBearingResult
    res = BallBearingResult.single(ball_row, label="b", delta_r=0.0, delta_a=0.0, psi=0.0, phi_Fr=0.5,
                                   Fr_xz=0.0, Fr_xy=0.0, Fa=0.0, n_iter=1, residual=0.0, ok=True)
    loc = contacts(ball, res, races=("inner",), elements="max")[0].location
    assert loc["phi_global_rad"] == pytest.approx(res.phi_j_global(0)[0])


def test_record_keys_must_not_overlap(ball):
    c = contacts(ball, load=1.0, races=("inner",))[0]
    assert {"element", "subtype", "race", "label", "case_hash"} <= set(c.record())
    bad = Contact("bearing", "X", {"label": "clash"}, c.case)
    with pytest.raises(ValueError, match="overlap"):
        bad.record()


def test_load_or_result_not_both(ball, ball_row):
    with pytest.raises(ValueError):
        contacts(ball, ball_row, load=1.0)
    with pytest.raises(ValueError):
        contacts(ball)


def test_kinematics_only_when_speeds_are_given(ball):
    assert contacts(ball, load=100.0)[0].kinematics == {}
    k = contacts(ball, load=100.0, n_inner_rpm=3000.0)[0].kinematics
    g = ball.Dw * math.cos(ball.alpha_0) / ball.Dpw
    assert k["u_e_m_per_s"] == pytest.approx(31e-3 / 4 * (1 - g**2) * 3000 * 2 * math.pi / 60)
    assert k["u_e_m_per_s"] == entrainment_speed(ball.Dpw, ball.Dw, ball.alpha_0, 3000.0, 0.0)


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
    c = contacts(ball, load=300.0, races=(race,))[0]
    g = Grid.around(c.case)
    recs = {b: static_step.run(c, pair(c.case, g, build=b), g, tmp_path)[0]
            for b in ("composite", "two_profiles")}
    rc = recs["composite"]
    assert rc["element"] == "bearing" and rc["race"] == race
    assert abs(rc["err_p0"]) < 0.02
    assert all(rc[k] for k in ("converged", "load_ok", "edge_clear", "margin_ok", "resolution_ok",
                               "within_reference"))
    assert recs["two_profiles"]["p0_Pa"] == pytest.approx(rc["p0_Pa"], rel=0.01)


def test_static_step_flags_a_short_domain(ball, tmp_path):
    pytest.importorskip("pyfftw")
    case = contact_case(ball, "inner", 300.0)
    g = Grid.around(case, span_x=2.0)
    rec, _ = static_step.run(case, pair(case, g), g, tmp_path)
    assert rec["edge_clear"] and not rec["margin_ok"]


def test_contact_case_hash_ignores_label():
    a = ContactCase("x", r1=1e-3, r2=math.inf, E=2e11, nu=0.3, load=1.0)
    assert a.case_hash() == ContactCase("y", r1=1e-3, r2=math.inf, E=2e11, nu=0.3, load=1.0).case_hash()


def test_contact_case_keeps_slippy_keys_and_the_saved_hash():
    """E and nu go out under SlipPY's names, so the hash of the saved results does not change."""
    # the ball_flat case of validation/contact/hertz_point_grid_domain: 12.7 mm steel ball on a steel flat, 500 N
    case = ContactCase("ball_flat", r1=6.35e-3, r2=math.inf, E=210e9, nu=0.3, load=500.0)
    assert {"moduli", "v"} <= set(case.to_dict()) and {"moduli", "v"} <= set(case.hertz_args())
    assert case.case_hash() == "9ccc2ddabd085997"
