"""slippy_bridge: AxisForge bearing -> contact case -> SlipPY surfaces."""
import math

import numpy as np
import pytest

af_c = pytest.importorskip("axisforge.core")
pytest.importorskip("slippy")

from slippy_bridge import ContactCase, composite, contact_case, grid, semi_axes, two_profiles  # noqa: E402

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
    assert case.r2[0] * sign > 0 and case.r2[1] < 0          # outer concave in x, grooves concave in y


def test_alpha_default_is_family_angle(ball):
    assert contact_case(ball, "outer", 100.0, alpha=ball.alpha_0) == contact_case(ball, "outer", 100.0)


def test_alpha_j_changes_only_the_rolling_radius(ball):
    a0, aj = contact_case(ball, "inner", 100.0), contact_case(ball, "inner", 100.0, alpha=0.4)
    assert aj.r1 == a0.r1 and aj.r2[1] == a0.r2[1] and aj.r2[0] != a0.r2[0]
    expected = (ball.Dpw - ball.Dw * math.cos(0.4)) / (2 * math.cos(0.4)) * 1e-3
    assert aj.r2[0] == pytest.approx(expected)


def test_roller_is_a_line_contact_with_load_per_length(roller):
    case = contact_case(roller, "outer", load=500.0)
    assert case.line and math.isinf(case.r1[1]) and case.r2[0] < 0
    assert case.load == pytest.approx(500.0 / 5.0e-3)                    # Q / Lwe [N/m]
    lamina = contact_case(roller, "inner", load=10.0, length=roller.Lwe / roller.n_s)
    assert lamina.load == pytest.approx(10.0 / (0.1e-3))


@pytest.mark.parametrize("kwargs", [dict(race="middle", load=1.0), dict(race="inner", load=0.0),
                                    dict(race="inner", load=1.0, alpha=math.pi / 2)])
def test_bad_input_raises(ball, kwargs):
    with pytest.raises(ValueError):
        contact_case(ball, **kwargs)


def test_length_only_for_line_contacts(ball):
    with pytest.raises(ValueError):
        contact_case(ball, "inner", 1.0, length=2.0)


@pytest.mark.parametrize("race", ["inner", "outer"])
def test_grid_rolling_axis_and_dyadic_spacing(ball, race):
    case = contact_case(ball, race, 300.0)
    X, Y, h = grid(case)
    s_x, s_y = semi_axes(case)
    assert s_x < s_y                                         # ball on raceway: short axis along rolling
    assert math.log2(h) == round(math.log2(h))
    assert np.all(np.diff(X[:, 0]) > 0) and np.all(X[:, 0] == X[:, -1])  # X varies along axis 0 only
    assert X[X.shape[0] // 2, 0] == 0.0 and Y[0, Y.shape[1] // 2] == 0.0


def _bem_p0(case, build, out_dir):
    import slippy.contact as c
    X, Y, h = grid(case)                         # defaults: h = s_x/10, +-4 s_x along rolling
    s1, s2 = build(case, X, Y, h)
    model = c.ContactModel(f"bridge_{case.label}_{build.__name__}", s1, s2, output_dir=str(out_dir))
    model.add_step(c.StaticStep("load", normal_load=case.load, method="pk", tolerance=1e-9))
    p = np.asarray(model.solve()["loads_z"], dtype=float)       # SlipPY a4fbb44: a pressure [Pa]
    assert p.sum() * h**2 == pytest.approx(case.load, rel=1e-3)
    return p.max()


@pytest.mark.parametrize("race", ["inner", "outer"])
def test_dry_bem_from_axisforge_matches_hertz(ball, race, tmp_path):
    """End to end: AxisForge bearing -> both builds -> SlipPY static BEM; p0 vs Hertz and vs each other."""
    pytest.importorskip("pyfftw")
    import slippy.contact as c
    case = contact_case(ball, race, 300.0, label=f"ball_{race}")
    p0_ref = c.hertz_full(**case.hertz_args())["max_pressure"]
    p0 = {b.__name__: _bem_p0(case, b, tmp_path) for b in (composite, two_profiles)}
    assert p0["composite"] == pytest.approx(p0_ref, rel=0.02)
    assert p0["two_profiles"] == pytest.approx(p0["composite"], rel=0.01)


def test_default_label_is_the_bearing_label(ball):
    assert contact_case(ball, "inner", 1.0).label == f"{ball.label}_inner"


@pytest.mark.parametrize("r1, expect_x_longer", [((50e-3, 5e-3), True), ((5e-3, 50e-3), False)])
def test_semi_axes_follow_the_relative_radii(r1, expect_x_longer):
    case = ContactCase("ellipse", r1=r1, r2=math.inf, moduli=2.1e11, v=0.3, load=100.0)
    s_x, s_y = semi_axes(case)
    assert (s_x > s_y) is expect_x_longer


def test_line_axis_follows_the_cylinder():
    along_y = ContactCase("cyl_y", r1=(5e-3, math.inf), r2=math.inf, moduli=2.1e11, v=0.3, load=1e5, line=True)
    assert math.isinf(semi_axes(along_y)[1]) and math.isfinite(semi_axes(along_y)[0])


def test_grid_beyond_the_body_raises():
    case = ContactCase("tiny", r1=1e-4, r2=math.inf, moduli=2.1e11, v=0.3, load=1e-3)
    X, Y, h = grid(case, half_x=200.0, half_y=200.0)
    with pytest.raises(ValueError):
        composite(case, X, Y, h)


def test_contact_case_is_hashable_data():
    a = ContactCase("x", r1=1e-3, r2=math.inf, moduli=2e11, v=0.3, load=1.0)
    assert a.case_hash() == ContactCase("y", r1=1e-3, r2=math.inf, moduli=2e11, v=0.3, load=1.0).case_hash()
