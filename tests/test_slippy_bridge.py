"""slippy_bridge.contact: AxisForge bearing -> Contact -> surfaces -> static step."""
import math

import numpy as np
import pytest

af_c = pytest.importorskip("axisforge.core")
pytest.importorskip("slippy")

from axisforge.results import BallLoadDistributionResult, RollerLoadDistributionResult  # noqa: E402

from slippy_bridge.contact import (Contact, ContactCase, bearing_contacts, composite, contact_case,  # noqa: E402
                                   contacts, entrainment_speed, grid, hertz, measure, registered,
                                   rough, semi_axes, static_step, two_profiles)

CONTACT = dict(e1=210000.0, e2=210000.0, nu1=0.3, nu2=0.3)


def _bearing(family, arrangement, **geometry):
    catalog = af_c.BearingCatalog(d=20.0, D=42.0, b=12.0, designation="test", position=0.0,
                                  arrangement=arrangement, label="b")
    return af_c.Bearing.assemble(family=family, catalog=catalog,
                                 geometry={**geometry, "contact": af_c.ContactAnalysis.ISO16281, **CONTACT})


@pytest.fixture(scope="module")
def ball():
    return _bearing(af_c.DeepGrooveBallFamily(), "locating", Dw=7.0, Dpw=31.0, Z=9, s=0.010)


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
    assert "SpurHelicalGearMeshing" in reg["gear"]


def test_gear_adapter_not_written_yet_raises():
    from axisforge.core.machine_elements.gears.parallel_axis import SpurHelicalGearMeshing
    with pytest.raises(NotImplementedError):
        contacts(SpurHelicalGearMeshing.__new__(SpurHelicalGearMeshing))


def test_unregistered_subtype_raises():
    with pytest.raises(NotImplementedError, match="no SlipPY adapter"):
        contacts(object())


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
    assert c0.location == {"race": "inner", "row": 0, "element_j": 0, "phi_local_rad": 0.0, "alpha_rad": 0.2}
    assert c0.case.load == 300.0
    assert c0.case.r2[0] == pytest.approx(contact_case(ball, "inner", 300.0, alpha=0.2).r2[0])
    assert contacts(ball, ball_row, elements="max", races=("outer",))[0].location["element_j"] == 0


def test_roller_laminae_from_result(roller, roller_row):
    whole = bearing_contacts(roller, roller_row, races=("inner",))
    assert [c.case.load for c in whole] == pytest.approx([500.0 / 5e-3, 200.0 / 5e-3])     # Q_j / Lwe
    lam = bearing_contacts(roller, roller_row, races=("inner",), laminae=True, elements=[1])
    assert len(lam) == 50 and lam[0].case.load == pytest.approx(4.0 / 0.1e-3)          # q_jk / (Lwe/n_s)
    assert lam[3].location["lamina_k"] == 3 and lam[0].case.line


def test_listed_unloaded_element_is_skipped_and_row_checked(ball, ball_row):
    assert [c.location["element_j"] for c in bearing_contacts(ball, ball_row, races=("inner",), elements=[0, 4])] == [0]
    with pytest.raises(ValueError):
        bearing_contacts(ball, ball_row, row=1)


def test_multirow_result_gives_global_angle(ball, ball_row):
    from axisforge.results import BallBearingResult
    res = BallBearingResult.single(ball_row, label="b", delta_r=0.0, delta_a=0.0, psi=0.0, phi_Fr=0.5,
                                   Fr_xz=0.0, Fr_xy=0.0, Fa=0.0, n_iter=1, residual=0.0, ok=True)
    loc = bearing_contacts(ball, res, races=("inner",), elements="max")[0].location
    assert loc["phi_global_rad"] == pytest.approx(res.phi_j_global(0)[0])


def test_record_keys_must_not_overlap(ball):
    c = bearing_contacts(ball, load=1.0, races=("inner",))[0]
    assert {"element", "subtype", "race", "label", "case_hash"} <= set(c.record())
    bad = Contact("bearing", "X", {"label": "clash"}, c.case)
    with pytest.raises(ValueError, match="overlap"):
        bad.record()


def test_load_or_result_not_both(ball, ball_row):
    with pytest.raises(ValueError):
        bearing_contacts(ball, ball_row, load=1.0)
    with pytest.raises(ValueError):
        bearing_contacts(ball)


def test_kinematics_only_when_speeds_are_given(ball):
    assert bearing_contacts(ball, load=100.0)[0].kinematics == {}
    k = bearing_contacts(ball, load=100.0, n_inner_rpm=3000.0)[0].kinematics
    g = ball.Dw * math.cos(ball.alpha_0) / ball.Dpw
    assert k["u_e_m_per_s"] == pytest.approx(31e-3 / 4 * (1 - g**2) * 3000 * 2 * math.pi / 60)
    assert k["u_e_m_per_s"] == entrainment_speed(ball.Dpw, ball.Dw, ball.alpha_0, 3000.0, 0.0)


# ---- reference, grid, measurements ----------------------------------------------------------

@pytest.mark.parametrize("r1, x_longer", [((50e-3, 5e-3), True), ((5e-3, 50e-3), False)])
def test_semi_axes_follow_the_relative_radii(r1, x_longer):
    s_x, s_y = semi_axes(ContactCase("e", r1=r1, r2=math.inf, moduli=2.1e11, v=0.3, load=100.0))
    assert (s_x > s_y) is x_longer


def test_line_axis_follows_the_cylinder():
    s = semi_axes(ContactCase("c", r1=(5e-3, math.inf), r2=math.inf, moduli=2.1e11, v=0.3, load=1e5, line=True))
    assert math.isinf(s[1]) and math.isfinite(s[0])


def test_grid_rolling_axis_and_dyadic_spacing(ball):
    X, Y, h = grid(contact_case(ball, "inner", 300.0))
    assert math.log2(h) == round(math.log2(h))
    assert np.all(np.diff(X[:, 0]) > 0) and np.all(X[:, 0] == X[:, -1])
    assert X[X.shape[0] // 2, 0] == 0.0 and Y[0, Y.shape[1] // 2] == 0.0


def test_measure_finds_hertz_semi_axes_below_one_node(ball):
    case = contact_case(ball, "inner", 300.0)
    X, Y, h = grid(case)
    ref = hertz(case)
    m = measure(ref["pressure"](X, Y), h)
    assert abs(m["s_x_m"] - ref["s_x_m"]) < 0.5 * h and abs(m["s_y_m"] - ref["s_y_m"]) < 0.5 * h
    assert m["apex_offset_nodes"] == 0.0


def test_grid_beyond_the_body_raises():
    case = ContactCase("tiny", r1=1e-4, r2=math.inf, moduli=2.1e11, v=0.3, load=1e-3)
    X, Y, h = grid(case, half_x=200.0, half_y=200.0)
    with pytest.raises(ValueError):
        composite(case, X, Y, h)


# ---- roughness -----------------------------------------------------------------------------

def test_fractal_rms_and_seed():
    a = rough.fractal((64, 32), 1e-6, sigma=0.2e-6, hurst=0.8, seed=1)
    assert a.shape == (64, 32) and a.std() == pytest.approx(0.2e-6) and abs(a.mean()) < 1e-15
    assert np.array_equal(a, rough.fractal((64, 32), 1e-6, sigma=0.2e-6, hurst=0.8, seed=1))


def test_add_roughness_composite_and_two_profiles(ball):
    case = contact_case(ball, "inner", 300.0)
    X, Y, h = grid(case)
    r1, r2 = np.full(X.shape, 1e-7), np.full(X.shape, 2e-7)
    smooth = composite(case, X, Y, h)
    s1, s2 = rough.add_roughness(smooth, r1, r2, composite=True)
    assert np.allclose(s1.profile - smooth[0].profile, 3e-7) and np.allclose(s2.profile, 0.0)
    assert s1.material is smooth[0].material and s2.material is smooth[1].material
    t1, t2 = rough.add_roughness(two_profiles(case, X, Y, h), r1, r2, composite=False)
    assert np.allclose(t2.profile - two_profiles(case, X, Y, h)[1].profile, 2e-7)
    with pytest.raises(ValueError):
        rough.add_roughness(smooth, np.zeros((3, 3)))


# ---- static step, end to end ---------------------------------------------------------------

@pytest.mark.parametrize("race", ["inner", "outer"])
def test_static_step_end_to_end(ball, race, tmp_path):
    """AxisForge bearing -> both builds -> SlipPY static BEM: p0 vs Hertz, builds agree, checks true."""
    pytest.importorskip("pyfftw")
    c = bearing_contacts(ball, load=300.0, races=(race,))[0]
    X, Y, h = grid(c.case)
    recs = {b.__name__: static_step.run(c, b(c.case, X, Y, h), X, h, tmp_path)[0]
            for b in (composite, two_profiles)}
    rc = recs["composite"]
    assert rc["element"] == "bearing" and rc["race"] == race
    assert abs(rc["err_p0"]) < 0.02
    assert all(rc[k] for k in ("converged", "load_ok", "edge_clear", "margin_ok", "resolution_ok",
                               "within_reference"))
    assert recs["two_profiles"]["p0_Pa"] == pytest.approx(rc["p0_Pa"], rel=0.01)


def test_static_step_flags_a_short_domain(ball, tmp_path):
    pytest.importorskip("pyfftw")
    case = contact_case(ball, "inner", 300.0)
    X, Y, h = grid(case, half_x=2.0)
    rec, _ = static_step.run(case, composite(case, X, Y, h), X, h, tmp_path)
    assert rec["edge_clear"] and not rec["margin_ok"]


def test_contact_case_hash_ignores_label():
    a = ContactCase("x", r1=1e-3, r2=math.inf, moduli=2e11, v=0.3, load=1.0)
    assert a.case_hash() == ContactCase("y", r1=1e-3, r2=math.inf, moduli=2e11, v=0.3, load=1.0).case_hash()
