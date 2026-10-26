"""slippy_bridge.contact: bodies and contacts (core), then the FZG C40 checks that need no bridge."""
import math
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

af_c = pytest.importorskip("axisforge.core")

from axisforge.core.machine_elements.bearings.families import available_families  # noqa: E402
from axisforge.results import (BallBearingResult, BallLoadDistributionResult,  # noqa: E402
                               BearingAnalysisResult, RollerLoadDistributionResult)
from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281.contact_solver import (  # noqa: E402
    ISO16281BallSolver)

from slippy_bridge.contact.core.contact import Body, Contact  # noqa: E402
from slippy_bridge.contact.core.machine_elements.bearings import (BallRacewayContact, BearingContact,  # noqa: E402
                                                                  RollerRacewayContact, bearing_contacts)
from slippy_bridge.contact.mesh.grid import SPAN_X_0, Grid  # noqa: E402
from slippy_bridge.contact.results.hertz import HertzResult  # noqa: E402
from slippy_bridge.contact.results.static import StaticResult  # noqa: E402

STEEL = dict(E=210e9, nu=0.3)                      # [Pa], [-]
MM = 1e-3                                          # [m/mm]


class _Point(Contact):
    line = False


class _Line(Contact):
    line = True


# ---- Body.profile against the closed forms ----------------------------------------------------

# sample nodes [m]: X along axis 0, Y along axis 1, both sides of the apex
_X, _Y = np.meshgrid(np.linspace(-2.0, 2.0, 5) * MM, np.linspace(-1.5, 1.5, 4) * MM, indexing="ij")
_INNER, _OUTER = (28.65 * MM, -6.604 * MM), (-41.35 * MM, -6.604 * MM)       # 6210 raceways (BR1 section 5)


@pytest.mark.parametrize("r, expected", [
    (6.35 * MM, lambda x, y: -(6.35 * MM - np.sqrt((6.35 * MM)**2 - x**2 - y**2))),           # sphere
    ((10 * MM, math.inf), lambda x, y: -(10 * MM - np.sqrt((10 * MM)**2 - x**2))),           # cylinder
    ((math.inf, 10 * MM), lambda x, y: -(10 * MM - np.sqrt((10 * MM)**2 - y**2))),           # cylinder along x
])
def test_profile_matches_the_closed_form(r, expected):
    body = Body(r=r, **STEEL)
    np.testing.assert_allclose(body.profile(_X, _Y), expected(_X, _Y), rtol=1e-12, atol=1e-18)
    assert body.profile(np.zeros((1, 1)), np.zeros((1, 1)))[0, 0] == 0.0                   # apex at 0


@pytest.mark.parametrize("r", [_INNER, _OUTER, (20 * MM, 8 * MM)], ids=["inner", "outer", "convex"])
def test_torus_profile_lies_on_the_implicit_torus(r):
    """Implicit torus: depth w = -z; the axis of revolution is the line (w = r_x, parallel to y_b); the
    signed distance to it s = sign(r_x) sqrt(x^2 + (w - r_x)^2) and y satisfy (s - (r_x - r_y))^2 + y^2 = r_y^2
    (generating circle of radius |r_y| centred at r_x - r_y from the axis)."""
    r_x, r_y = r
    w = -Body(r=r, **STEEL).profile(_X, _Y)
    s = math.copysign(1.0, r_x) * np.sqrt(_X**2 + (w - r_x)**2)
    np.testing.assert_allclose((s - (r_x - r_y))**2 + _Y**2, r_y**2, rtol=1e-12)


@pytest.mark.parametrize("r", [_INNER, _OUTER, (20 * MM, 8 * MM)], ids=["inner", "outer", "convex"])
def test_torus_profile_is_the_hertz_paraboloid_near_the_apex(r):
    """Johnson 1985 Eq. (4.2a): -z -> x^2/(2 r_x) + y^2/(2 r_y); relative error O((x/r)^2) < 1e-4 at 0.05 mm."""
    r_x, r_y = r
    x, y = _X / 40, _Y / 30                                                     # |x|, |y| <= 0.05 mm
    np.testing.assert_allclose(-Body(r=r, **STEEL).profile(x, y), x**2 / (2 * r_x) + y**2 / (2 * r_y),
                               rtol=1e-3, atol=1e-20)


def test_profile_signs_convex_and_concave():
    """Convex body falls away from the contact plane (-sag < 0), a concave one rises (> 0)."""
    assert Body(r=5 * MM, **STEEL).profile(_X, _Y)[0, 0] < 0
    assert Body(r=-50 * MM, **STEEL).profile(_X, _Y)[0, 0] > 0
    assert Body(r=math.inf, **STEEL).profile(_X, _Y).tolist() == np.zeros_like(_X).tolist()


@pytest.mark.parametrize("angle", [0.3, -1.1, math.pi / 2])
def test_profile_rotation_is_the_body_frame(angle):
    body = Body(r=_INNER, **STEEL)
    x_b = _X * math.cos(angle) + _Y * math.sin(angle)
    y_b = -_X * math.sin(angle) + _Y * math.cos(angle)
    np.testing.assert_allclose(body.profile(_X, _Y, angle), body.profile(x_b, y_b), rtol=1e-12, atol=1e-18)


def test_cylinder_turned_by_90_deg_lies_along_x():
    turned = Body(r=(10 * MM, math.inf), **STEEL).profile(_X, _Y, math.pi / 2)
    np.testing.assert_allclose(turned, Body(r=(math.inf, 10 * MM), **STEEL).profile(_X, _Y), rtol=1e-9, atol=1e-15)


@pytest.mark.parametrize("r, x, y", [
    (1.0 * MM, _X, _Y),                                  # |x| beyond the sphere (NaN)
    ((-3.0 * MM, -1.6 * MM), _X, _Y),                    # |x_b| >= |rho|, outside the swept circle
    # at y = 1.5 mm sag_y = -1.043 mm, so rho = r_x - sag_y = +0.043 mm has crossed the axis of
    # revolution: the formula gives a finite but wrong height, only the sign check catches it
    ((-1.0 * MM, -1.6 * MM), np.zeros((1, 1)), np.full((1, 1), 1.5 * MM)),
])
def test_grid_beyond_the_body_raises(r, x, y):
    with pytest.raises(ValueError, match="beyond the body"):
        Body(r=r, **STEEL).profile(x, y)


# ---- relative curvatures, psi, E* (Johnson 1985 section 4.1) ----------------------------------

def test_curvatures_of_the_6210_inner_contact():
    """Ball 12.7 mm in the inner groove of a 6210 (Dpw 70, f_i 0.52), alpha 0 (BR1 section 5, printed to 5 digits
    [1/mm]), and the unrounded hand sums 1/6.35 + 1/28.65, 1/6.35 - 1/6.604."""
    c = _Point(body1=Body(r=6.35 * MM, **STEEL), body2=Body(r=_INNER, **STEEL), load=1000.0)
    assert c.kappa_max == pytest.approx(0.19238 / MM, abs=0.000005 / MM)
    assert c.kappa_min == pytest.approx(0.00606 / MM, abs=0.000005 / MM)
    assert c.kappa_max == pytest.approx((1 / 6.35 + 1 / 28.65) / MM, rel=1e-12)
    assert c.kappa_min == pytest.approx((1 / 6.35 - 1 / 6.604) / MM, rel=1e-12)
    assert c.psi == 0.0


def test_crossed_cylinders_at_90_deg_are_a_sphere():
    R = 10 * MM
    cyl = Body(r=(R, math.inf), **STEEL)
    c = _Point(body1=cyl, body2=cyl, load=1.0, angle=math.pi / 2)
    assert c.kappa_max == pytest.approx(1 / R, rel=1e-12)
    assert c.kappa_min == pytest.approx(1 / R, rel=1e-12)


@pytest.mark.parametrize("sign", [+1.0, -1.0])
def test_crossed_cylinders_at_45_deg(sign):
    """Johnson 1985 p. 86: A, B = (1 -+ 1/sqrt 2)/(2R), so kappa = 2A, 2B; kappa_max bisects the two axes."""
    R = 10 * MM
    cyl = Body(r=(R, math.inf), **STEEL)
    c = _Point(body1=cyl, body2=cyl, load=1.0, angle=sign * math.pi / 4)
    assert c.kappa_max == pytest.approx((1 + math.sqrt(2) / 2) / R, rel=1e-12)
    assert c.kappa_min == pytest.approx((1 - math.sqrt(2) / 2) / R, rel=1e-12)
    assert c.psi == pytest.approx(sign * math.radians(22.5), rel=1e-12)


# two convex tori at 0.3 rad, both more curved along x_b than across (outside the SlipPY R-534 domain)
_T1, _T2, _ANGLE = (5 * MM, 20 * MM), (8 * MM, 30 * MM), 0.3


def test_two_tori_at_an_angle_match_johnson():
    """Johnson 1985 Eqs. (4.4)-(4.5), p. 85: A + B and |B - A| from the body radii and alpha; kappa = 2A, 2B.
    psi: the eigenvector of the largest eigenvalue of K = sum R diag(1/r_x, 1/r_y) R^T (numpy eigh)."""
    c = _Point(body1=Body(r=_T1, **STEEL), body2=Body(r=_T2, **STEEL), load=1.0, angle=_ANGLE)
    (k1x, k1y), (k2x, k2y) = (1 / r for r in _T1), (1 / r for r in _T2)
    A_plus_B = 0.5 * (k1x + k1y + k2x + k2y)
    B_minus_A = 0.5 * math.sqrt((k1x - k1y)**2 + (k2x - k2y)**2
                                + 2 * (k1x - k1y) * (k2x - k2y) * math.cos(2 * _ANGLE))
    assert c.kappa_max == pytest.approx(A_plus_B + B_minus_A, rel=1e-12)          # 2B = 1/R''
    assert c.kappa_min == pytest.approx(A_plus_B - B_minus_A, rel=1e-12)          # 2A = 1/R'
    rot = np.array([[math.cos(_ANGLE), -math.sin(_ANGLE)], [math.sin(_ANGLE), math.cos(_ANGLE)]])
    w, v = np.linalg.eigh(np.diag([k1x, k1y]) + rot @ np.diag([k2x, k2y]) @ rot.T)
    assert w[1] == pytest.approx(c.kappa_max, rel=1e-12)
    # psi is a direction: compare mod pi with the eigenvector of the largest eigenvalue
    assert abs(math.remainder(math.atan2(v[1, 1], v[0, 1]) - c.psi, math.pi)) < 1e-12


_GAP_CASES = [((_T1, _T2), _ANGLE), ((_T1, _T2), -1.2), ((6.35 * MM, _INNER), 0.0),
              (((10 * MM, math.inf), (10 * MM, math.inf)), math.pi / 4)]


@pytest.mark.parametrize("radii, angle", _GAP_CASES)
def test_profiles_give_the_gap_curvatures(radii, angle):
    """The gap -(z1 + z2) of the exact profiles is kappa t^2 / 2 at a small distance t along psi (kappa_max) and
    across it (kappa_min): profiles() and the curvature tensor describe the same two bodies."""
    c = _Point(body1=Body(r=radii[0], **STEEL), body2=Body(r=radii[1], **STEEL), load=1.0, angle=angle)
    t = 1e-5                                                                      # [m], (t / r)^2 <= 1e-5
    cp, sp = math.cos(c.psi), math.sin(c.psi)
    grid = SimpleNamespace(X=np.array([[t * cp, -t * sp]]), Y=np.array([[t * sp, t * cp]]))
    z1, z2 = c.profiles(grid)
    gap = -(z1 + z2)[0]
    assert gap[0] == pytest.approx(c.kappa_max * t**2 / 2, rel=1e-4)
    assert gap[1] == pytest.approx(c.kappa_min * t**2 / 2, rel=1e-4)


@pytest.mark.parametrize("radii, angle", _GAP_CASES)
def test_curvatures_and_psi_agree_with_hertz_full(radii, angle):
    """SlipPY hertz_full: kappa_max, kappa_min = 1/relative_radii[0], [1]; psi = alpha (mod pi, no pi/2 shift);
    E* = e_star. Cases outside the SlipPY R-534 domain (both bodies more curved across than along)."""
    hertz_full = pytest.importorskip("slippy.contact").hertz_full
    c = _Point(body1=Body(r=radii[0], **STEEL), body2=Body(r=radii[1], **STEEL), load=100.0, angle=angle)
    ref = hertz_full(r1=radii[0], r2=radii[1], moduli=(STEEL["E"],) * 2, v=(STEEL["nu"],) * 2, load=100.0,
                     angle=angle)
    assert c.kappa_max == pytest.approx(1 / ref["relative_radii"][0], rel=1e-12)
    assert c.kappa_min == pytest.approx(1 / ref["relative_radii"][1], rel=1e-12)
    assert abs(math.remainder(c.psi - ref["alpha"], math.pi)) < 1e-12
    assert c.E_star == pytest.approx(ref["e_star"], rel=1e-14)


def test_line_contact_has_zero_kappa_min_along_y():
    c = _Line(body1=Body(r=(10 * MM, math.inf), **STEEL), body2=Body(r=math.inf, **STEEL), load=1e5)
    assert (c.kappa_max, c.kappa_min, c.psi) == (pytest.approx(100.0), 0.0, 0.0)


def test_parallel_cylinders_turned_by_90_deg_are_a_line_along_y():
    """Body 2 r = (inf, R) turned by pi/2 is the cylinder (R, inf): cos(pi/2) = 6e-17 must not tilt the line."""
    R = 10 * MM
    c = _Line(body1=Body(r=(R, math.inf), **STEEL), body2=Body(r=(math.inf, R), **STEEL), load=1e5,
              angle=math.pi / 2)
    assert (c.kappa_max, c.kappa_min, c.psi) == (pytest.approx(2 / R, rel=1e-12), 0.0, 0.0)


def test_E_star():
    c = _Point(body1=Body(r=5 * MM, E=210e9, nu=0.3), body2=Body(r=math.inf, E=70e9, nu=0.33), load=1.0)
    assert c.E_star == pytest.approx(1 / ((1 - 0.3**2) / 210e9 + (1 - 0.33**2) / 70e9), rel=1e-14)


# ---- input checks (D-0034 item 6) ---------------------------------------------------------------

_BALL, _FLAT = dict(r=5 * MM, **STEEL), dict(r=math.inf, **STEEL)


@pytest.mark.parametrize("body", [
    dict(r=0.0, **STEEL),
    dict(r=(math.nan, 5 * MM), **STEEL),
    dict(r=(5 * MM, -math.inf), **STEEL),
    dict(r=(1.0, 2.0, 3.0), **STEEL),
])
def test_bad_radius_raises(body):
    with pytest.raises(ValueError, match="Body: r"):
        Body(**body)


@pytest.mark.parametrize("E, nu, match", [
    (0.0, 0.3, "E must be positive"), (-1.0, 0.3, "E must be positive"), (math.inf, 0.3, "E must be positive"),
    (210e9, 0.0, r"nu must be in \(0, 0.5\)"), (210e9, 0.5, r"nu must be in \(0, 0.5\)"),
])
def test_bad_material_raises(E, nu, match):
    with pytest.raises(ValueError, match=match):
        Body(r=5 * MM, E=E, nu=nu)


@pytest.mark.parametrize("cls, body1, body2, kwargs, match", [
    (_Point, _BALL, _FLAT, dict(load=0.0), "load must be positive"),
    (_Point, _BALL, _FLAT, dict(load=math.nan), "load must be positive"),
    (_Point, _BALL, _FLAT, dict(load=1.0, angle=-math.pi / 2), r"angle must be in \(-pi/2, pi/2\]"),
    (_Point, _BALL, _FLAT, dict(load=1.0, angle=2.0), r"angle must be in \(-pi/2, pi/2\]"),
    (_Point, _BALL, dict(r=-4 * MM, **STEEL), dict(load=1.0), "not a Hertz contact"),       # ball in a smaller cup
    (_Point, dict(r=(5 * MM, math.inf), **STEEL), dict(r=(math.inf, -5 * MM), **STEEL), dict(load=1.0),
     "saddle"),                                                                              # kappa_min < 0
    (_Point, dict(r=(5 * MM, math.inf), **STEEL), _FLAT, dict(load=1.0), "point contact needs kappa_min > 0"),
    (_Line, _BALL, _FLAT, dict(load=1.0), "line contact needs kappa_min = 0"),
    (_Line, dict(r=(math.inf, 5 * MM), **STEEL), _FLAT, dict(load=1.0), "along bridge y"),
    (_Point, dict(r=(5 * MM, 5e3), **STEEL), _FLAT, dict(load=1.0), "neither a point"),      # ratio 1e-6
    (_Point, dict(r=(5 * MM, 5 * MM / 1.2e-5), **STEEL), _FLAT, dict(load=1.0), "neither a point"),  # dead band
    (_Line, dict(r=(5 * MM, 5 * MM / 1.2e-5), **STEEL), _FLAT, dict(load=1.0), "neither a point"),
])
def test_bad_contact_raises(cls, body1, body2, kwargs, match):
    with pytest.raises(ValueError, match=match):
        cls(body1=Body(**body1), body2=Body(**body2), label="c1", **kwargs)


def test_contact_names_its_label():
    with pytest.raises(ValueError, match="Contact 'ball_7': load"):
        _Point(body1=Body(**_BALL), body2=Body(**_FLAT), load=-1.0, label="ball_7")


def test_contact_is_abstract():
    with pytest.raises(TypeError, match="abstract"):
        Contact(body1=Body(**_BALL), body2=Body(**_FLAT), load=1.0)


def test_profiles_turn_body_2_only():
    body1, body2 = Body(r=(10 * MM, math.inf), **STEEL), Body(r=(10 * MM, math.inf), **STEEL)
    grid = SimpleNamespace(X=_X, Y=_Y)
    z1, z2 = _Point(body1=body1, body2=body2, load=1.0, angle=0.4).profiles(grid)
    np.testing.assert_array_equal(z1, body1.profile(_X, _Y))
    np.testing.assert_array_equal(z2, body2.profile(_X, _Y, 0.4))


# ---- bearings: AxisForge bearing -> BearingContact (core/machine_elements/bearings.py) ------------

CONTACT = dict(e1=210000.0, e2=210000.0, nu1=0.3, nu2=0.3)          # AxisForge steel [MPa], [-]


def _bearing(family, arrangement, contact: bool = True, catalog=dict(d=20.0, D=42.0, b=12.0), **geometry):
    """Assembled AxisForge bearing; ``contact`` adds the ISO/TS 16281 contact constants (steel)."""
    catalog = af_c.BearingCatalog(**catalog, designation="test", position=0.0, arrangement=arrangement, label="b")
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
def roller_10():
    """Cylindrical roller 10 x 10 mm on Dpw 70 mm (BR1 section 5)."""
    return _bearing(af_c.CylindricalRollerFamily(), "non-locating", catalog=dict(d=50.0, D=90.0, b=20.0),
                    Dwe=10.0, Lwe=10.0, Dpw=70.0, Z=14, n_s=50, s=0.010)


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


@pytest.fixture(scope="module")
def ball_result(ball_row):
    """Single-row AxisForge ``BallBearingResult`` of the ``ball`` fixture (label "b"), converged."""
    return BallBearingResult.single(ball_row, label="b", delta_r=0.0, delta_a=0.0, psi=0.0, phi_Fr=0.5,
                                    Fr_xz=0.0, Fr_xy=0.0, Fa=0.0, n_iter=1, residual=0.0, ok=True)


def _key(c):
    """Everything a bearing contact holds, for comparing two lists of contacts."""
    return (type(c), c.label, c.load, c.body1.r, c.body2.r, c.body1.E, c.body2.nu, c.race, c.alpha_rad, c.row,
            c.j, c.phi_rad, c.phi_global_rad)


def test_6210_hand_check(load_case):
    """BR1 section 5: 6210-like (Dw 12.7, Dpw 70, f 0.52, s = 0 so alpha_0 = 0), Q 1000 N given by hand.
    R_x = 1/(1/6.35 + 1/28.65) = 5.198 mm inner, 1/(1/6.35 - 1/41.35) = 7.502 mm outer; R_y =
    1/(1/6.35 - 1/6.604) = 165.1 mm both; printed to 4 digits."""
    bearing = load_case("iso16281_ball_contact").build_bearing(0.0)
    inner, outer = bearing_contacts(bearing, load=1000.0)
    assert all(isinstance(c, BallRacewayContact) and c.load == 1000.0 and c.psi == 0.0 for c in (inner, outer))
    assert (inner.race, outer.race, inner.label, outer.label) == ("inner", "outer", "6210_inner", "6210_outer")
    assert 1 / inner.kappa_max / MM == pytest.approx(5.198, abs=0.0005)
    assert 1 / outer.kappa_max / MM == pytest.approx(7.502, abs=0.0005)
    for c in (inner, outer):
        assert 1 / c.kappa_min / MM == pytest.approx(165.1, abs=0.05)
    assert inner.body2.r[0] > 0 and outer.body2.r[0] < 0 and outer.body2.r[1] < 0          # concave outer race
    assert inner.E_star == pytest.approx(210e9 / (2 * (1 - 0.3**2)), rel=1e-12)            # MPa -> Pa once


def test_roller_hand_check(roller_10):
    """Cylindrical roller 10 x 10 mm on Dpw 70 mm, Q 1000 N, inner race: line load Q/Lwe = 1000 N / 0.010 m =
    100 000 N/m; R = 1/(1/5 + 1/30) = 4.2857 mm (raceway (Dpw - Dwe)/2 = 30 mm)."""
    (c,) = bearing_contacts(roller_10, load=1000.0, races=("inner",))
    assert isinstance(c, RollerRacewayContact) and c.line and (c.k, c.x_mm) == (None, None)
    assert c.load == pytest.approx(1e5, rel=1e-12)
    assert 1 / c.kappa_max / MM == pytest.approx(4.2857, abs=0.00005)
    assert (c.kappa_min, c.psi) == (0.0, 0.0)


@pytest.mark.parametrize("race, sign", [("inner", +1.0), ("outer", -1.0)])
def test_ball_curvature_sum_matches_iso(ball, race, sign):
    """Signs and units: sum of curvatures = (1/Dw)(4 - 1/f +/- 2 gamma/(1 -/+ gamma)) (ISO/TS 16281); the
    trace of the gap tensor is kappa_max + kappa_min."""
    (c,) = bearing_contacts(ball, load=100.0, races=(race,))
    f = ball.ri / ball.Dw if race == "inner" else ball.re / ball.Dw
    g = ball.Dw * math.cos(ball.alpha_0) / ball.Dpw
    iso = (4.0 - 1.0 / f + sign * 2.0 * g / (1.0 - sign * g)) / (ball.Dw * MM)
    assert c.kappa_max + c.kappa_min == pytest.approx(iso, rel=1e-12)
    assert c.body2.r[0] * sign > 0 and c.body2.r[1] < 0


def test_alpha_default_is_family_angle(ball):
    (a0,), (given,) = (bearing_contacts(ball, load=100.0, races=("outer",), **kw)
                       for kw in ({}, {"alpha": ball.alpha_0}))
    assert given.body2.r == pytest.approx(a0.body2.r, rel=1e-12) and a0.alpha_rad == ball.alpha_0


def test_alpha_changes_only_the_rolling_radius(ball):
    (a0,), (aj,) = (bearing_contacts(ball, load=100.0, races=("inner",), **kw) for kw in ({}, {"alpha": 0.4}))
    assert aj.body1.r == a0.body1.r and aj.body2.r[1] == a0.body2.r[1] and aj.alpha_rad == 0.4
    assert aj.body2.r[0] == pytest.approx((ball.Dpw - ball.Dw * math.cos(0.4)) / (2 * math.cos(0.4)) * MM)


def test_ball_contacts_from_result(ball, ball_row):
    cs = bearing_contacts(ball, ball_row, races=("inner", "outer"), elements="loaded")
    assert len(cs) == 2 * 5                                         # 5 loaded balls x 2 races
    c0 = cs[0]
    assert isinstance(c0, BallRacewayContact) and c0.label == "b_inner_e0"
    assert (c0.race, c0.alpha_rad, c0.row, c0.j, c0.phi_rad, c0.phi_global_rad) == ("inner", 0.2, 0, 0, 0.0, None)
    assert c0.load == 300.0
    (by_hand,) = bearing_contacts(ball, load=300.0, alpha=0.2, races=("inner",))
    assert c0.body2.r == by_hand.body2.r
    assert bearing_contacts(ball, ball_row, elements="max", races=("outer",))[0].j == 0


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
            assert (c.race, c.j) == (race, j)
            assert c.load == row.Q_j[j] and c.alpha_rad == row.alpha_j[j]
        # ISO/TS 16281 curvature sum at the operating angle: (1/Dw)(4 - 1/f +/- 2 g_j/(1 -/+ g_j)), inner +, outer -
        g_j = ball_6210.Dw * math.cos(row.alpha_j[j]) / ball_6210.Dpw
        for sign, c in ((+1.0, inner), (-1.0, outer)):
            f = 0.52
            iso = (4.0 - 1.0 / f + sign * 2.0 * g_j / (1.0 - sign * g_j)) / (ball_6210.Dw * MM)   # [1/m]
            assert c.kappa_max + c.kappa_min == pytest.approx(iso, rel=1e-9)


def test_phi_is_wrapped_as_the_axisforge_element_record(ball, ball_result):
    """``phi_rad`` is AxisForge's own element record value: local angle wrapped to [-pi, pi)."""
    from axisforge.outputs.solvers.bearings.load_distribution.load_distribution_results import element_records
    by_j = {e["j"]: e["phi_rad"] for e in element_records(ball_result)}
    cs = bearing_contacts(ball, ball_result, races=("inner",))
    assert any(ball_result.rows[0].phi_j[c.j] >= math.pi for c in cs)                    # some angles do wrap
    for c in cs:
        assert -math.pi <= c.phi_rad < math.pi and c.phi_rad == by_j[c.j]


def test_roller_laminae_from_result(roller, roller_row):
    whole = bearing_contacts(roller, roller_row, races=("inner",))
    assert [c.load for c in whole] == pytest.approx([500.0 / 5e-3, 200.0 / 5e-3])          # Q_j / Lwe
    lam = bearing_contacts(roller, roller_row, races=("inner",), laminae=True, elements=[1])
    assert len(lam) == 50 and lam[0].load == pytest.approx(4.0 / 0.1e-3)                    # q_jk / (Lwe/n_s)
    assert lam[3].k == 3 and lam[0].line and lam[3].label == "b_inner_e1_l3"
    assert lam[3].x_mm == pytest.approx(roller_row.x_k[3])
    assert whole[0].k is None and whole[0].x_mm is None


def test_listed_unloaded_element_is_skipped_and_row_checked(ball, ball_row):
    assert [c.j for c in bearing_contacts(ball, ball_row, races=("inner",), elements=[0, 4])] == [0]
    with pytest.raises(ValueError, match="row applies"):
        bearing_contacts(ball, ball_row, row=1)


def test_bearing_result_gives_global_angle(ball, ball_result):
    c = bearing_contacts(ball, ball_result, races=("inner",), elements="max")[0]
    assert c.phi_global_rad == pytest.approx(ball_result.phi_j_global(0)[0])
    assert bearing_contacts(ball, ball_result.rows[0], races=("inner",))[0].phi_global_rad is None


def test_analysis_result_of_the_solver_is_accepted(ball, ball_result):
    """``SolverBase.solve`` returns a ``BearingAnalysisResult`` per label; the bridge reads its load distribution."""
    analysis = BearingAnalysisResult(label="b", load_distribution=ball_result)
    assert ([_key(c) for c in bearing_contacts(ball, analysis)]
            == [_key(c) for c in bearing_contacts(ball, ball_result)])


@pytest.mark.parametrize("call, match", [
    (lambda ball, res: bearing_contacts(ball, replace(res, ok=False, residual=12.0)), "not converged"),
    (lambda ball, res: bearing_contacts(ball, replace(res, label="other")), "not of bearing"),
    (lambda ball, res: bearing_contacts(ball, res, alpha=0.1), "alpha applies to load= only"),
    (lambda ball, res: bearing_contacts(ball, res, load=1.0), "either result"),
    (lambda ball, res: bearing_contacts(ball), "either result"),
    (lambda ball, res: bearing_contacts(ball, res, laminae=True), "laminae applies"),
    (lambda ball, res: bearing_contacts(ball, load=1.0, row=1), "laminae and row apply to result= only"),
    (lambda ball, res: bearing_contacts(ball, replace(res, rows=(replace(res.rows[0], Q_j=0.0 * res.rows[0].Q_j),)),
                                        elements="max"), "no loaded element"),
    (lambda ball, res: bearing_contacts(ball, replace(res.rows[0], Q_j=0.0 * res.rows[0].Q_j)), "no loaded element"),
])
def test_bearing_contacts_refuses(ball, ball_result, call, match):
    with pytest.raises(ValueError, match=match):
        call(ball, ball_result)


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
        assert [c.race for c in cs] == ["inner", "outer"] and isinstance(cs[0], BearingContact)


def _thrust_ball(alpha_0_deg=90.0):
    return _bearing(af_c.ThrustBallSingleRowFamily(), "thrust", Dw=8.0, Dpw=40.0, Z=12, alpha_0_deg=alpha_0_deg)


@pytest.mark.parametrize("Fr", [0.0, 200.0])
def test_contacts_of_a_solved_thrust_ball(Fr):
    """ISO16281BallSolver on a thrust ball at 90 deg, Fa 2 kN (+ Fr): alpha_j at or past 90 deg is accepted
    and the flat raceways stay flat."""
    bearing = _thrust_ball()
    result = ISO16281BallSolver().solve_contact(bearing, Fr_xz=Fr, Fr_xy=0.0, Fa=2000.0, delta_r_init=0.0,
                                                delta_a_init=0.0, psi=0.0, phi_Fr=0.0)
    assert result.ok
    cs = bearing_contacts(bearing, result)
    assert len(cs) == 2 * np.count_nonzero(result.rows[0].Q_j > 0.0)
    assert all(math.isinf(c.body2.r[0]) and c.psi == 0.0 for c in cs)


def test_thrust_ball_away_from_90_deg_is_refused():
    with pytest.raises(NotImplementedError, match="ThrustBallSingleRowFamily at alpha_0 = 60"):
        bearing_contacts(_thrust_ball(60.0), load=100.0)


def test_alpha_is_refused_where_it_changes_nothing(roller):
    for bearing in (roller, _thrust_ball()):
        with pytest.raises(ValueError, match="alpha has no effect"):
            bearing_contacts(bearing, load=100.0, alpha=0.1)


def test_roller_raceway_is_not_re_evaluated_at_alpha_j(roller, roller_row):
    """(Dpw -/+ Dwe)/2 has no angle: a roller row with alpha_j != 0 leaves the raceway radii as stored."""
    tilted = replace(roller_row, alpha_j=np.full_like(roller_row.alpha_j, 0.3))
    c = bearing_contacts(roller, tilted, races=("inner",), elements="max")[0]
    assert c.body2.r[0] == pytest.approx(roller.r_inner[0] * MM, rel=1e-15) and c.alpha_rad == 0.3


def test_load_mode_refuses_laminae(roller):
    with pytest.raises(ValueError, match="laminae and row apply to result= only"):
        bearing_contacts(roller, load=100.0, laminae=True)


def test_row_of_the_other_kind_is_refused(ball, roller_row):
    with pytest.raises(ValueError, match="line contact row, the bearing a point contact family"):
        bearing_contacts(ball, roller_row)


def test_bearing_without_elastic_constants_names_them():
    bare = _bearing(af_c.DeepGrooveBallFamily(), "locating", contact=False, Dw=7.0, Dpw=31.0, Z=9, s=0.010)
    with pytest.raises(ValueError, match="contact=ContactAnalysis.ISO16281 and e1, e2, nu1, nu2"):
        bearing_contacts(bare, load=100.0)


# ---- BearingContact checks -----------------------------------------------------------------------

_RACE = dict(r=(28.65 * MM, -6.604 * MM), **STEEL)            # 6210 inner raceway, curved along rolling
_THRUST = dict(r=(math.inf, -6.604 * MM), **STEEL)            # thrust raceway, flat along rolling


@pytest.mark.parametrize("race, kwargs, match", [
    (_RACE, dict(race="middle", alpha_rad=0.0), "race must be"),
    (_RACE, dict(race="inner", alpha_rad=math.pi / 2), r"\|alpha_rad\| must be"),              # finite r_x
    (_RACE, dict(race="inner", alpha_rad=-1.6), r"\|alpha_rad\| must be"),
    (_RACE, dict(race="inner", alpha_rad=0.0, angle=0.1), "angle must be 0"),
    (dict(r=(-50 * MM, 20 * MM), **STEEL), dict(race="inner", alpha_rad=0.0), "psi = 0 expected"),
])
def test_bearing_contact_checks(race, kwargs, match):
    with pytest.raises(ValueError, match=match):
        BallRacewayContact(body1=Body(r=6.35 * MM, **STEEL), body2=Body(**race), load=1.0, **kwargs)


@pytest.mark.parametrize("alpha", [math.pi / 2, math.pi / 2 + 4.7e-12, math.radians(98.7)])
def test_thrust_raceway_accepts_any_angle(alpha):
    """A raceway flat along rolling does not depend on alpha; the solver gives a thrust ball alpha_j at and
    beyond 90 deg (pure axial: pi/2 + 4.7e-12; with some Fr up to 98.7 deg)."""
    c = BallRacewayContact(body1=Body(r=6.35 * MM, **STEEL), body2=Body(**_THRUST), load=1.0, race="inner",
                           alpha_rad=alpha)
    assert c.psi == 0.0 and c.kappa_max == pytest.approx(1 / (6.35 * MM))


def test_bearing_contact_is_abstract():
    with pytest.raises(TypeError, match="abstract"):
        BearingContact(body1=Body(r=6.35 * MM, **STEEL), body2=Body(**_RACE), load=1.0, race="inner", alpha_rad=0.0)


# ---- mesh: Grid around a contact and the step arguments (mesh/grid.py) ----------------------------

def _ellipse(a_mm, b_mm, psi=0.0):
    """A HertzResult with only the ellipse that sizes the grid set (mm -> m; a = inf for a line); the other
    values are placeholders, Grid.around does not read them."""
    line = math.isinf(a_mm)
    return HertzResult(p0=1.0, a=a_mm * MM, b=b_mm * MM, psi=psi, delta=math.nan, b_over_a=0.0 if line else b_mm / a_mm,
                       E_star=1.0, size_ratio=0.0, valid=True, line=line)


@pytest.mark.parametrize("race, a_mm, b_mm, n_x, n_y", [("inner", 1.5307, 0.1666, 90, 264),
                                                         ("outer", 1.4877, 0.2025, 110, 256)])
def test_grid_of_the_6210_contacts(load_case, race, a_mm, b_mm, n_x, n_y):
    """BR3 section 5: 6210, Q 1000 N; h = b/10 rounded down to 2^-16 m, n = 2 ceil(span s / h) + 2."""
    bearing = load_case("iso16281_ball_contact").build_bearing(0.0)
    (c,) = bearing_contacts(bearing, load=1000.0, races=(race,))
    grid = Grid.around(c, _ellipse(a_mm, b_mm))
    assert grid.h == 2.0**-16 and grid.X.shape == (n_x, n_y)
    assert grid.step_args(c) == {"normal_load": 1000.0}


def test_grid_origin_on_the_centre_node(load_case):
    """The contact centre is node (n_x//2, n_y//2); the +x side is the short one and still reaches span_x b."""
    (c,) = bearing_contacts(load_case("iso16281_ball_contact").build_bearing(0.0), load=1000.0, races=("inner",))
    grid = Grid.around(c, _ellipse(1.5307, 0.1666))
    n_x, n_y = grid.X.shape
    assert grid.X[n_x // 2, 0] == 0.0 and grid.Y[0, n_y // 2] == 0.0
    assert grid.X.max() >= SPAN_X_0 * 0.1666 * MM and -grid.X.min() > grid.X.max()


def test_grid_of_a_roller_line_contact(roller_10):
    """BR3 section 5: roller line contact b 0.0688 mm -> h 2^-18 m, n_x 148, n_y 16; period 16 h = 61.04 um;
    load on the strip 100 000 N/m x period = 6.10 N."""
    (c,) = bearing_contacts(roller_10, load=1000.0, races=("inner",))
    grid = Grid.around(c, _ellipse(math.inf, 0.0688))
    assert grid.h == 2.0**-18 and grid.X.shape == (148, 16)
    assert grid.length_y_m == 16 * 2.0**-18
    assert grid.step_args(c, max_it=50) == {"normal_load": pytest.approx(1e5 * 16 * 2.0**-18, rel=1e-12),
                                            "periodic_geometry": True, "periodic_axes": (False, True),
                                            "max_it": 50}


def test_grid_of_the_fzg_h501_line_contact():
    """BR2: FZG H501 at C, b 0.08519 mm -> h 2^-17, n_x 92, n_y 16; 75 697.52 N/m on 16 h = 9.24 N."""
    c = _Line(body1=Body(r=(14.20546 * MM, math.inf), **STEEL), body2=Body(r=(21.30819 * MM, math.inf), **STEEL),
              load=75697.52)
    grid = Grid.around(c, _ellipse(math.inf, 0.08519))
    assert grid.h == 2.0**-17 and grid.X.shape == (92, 16)
    assert grid.step_args(c)["normal_load"] == pytest.approx(75697.52 * 16 * 2.0**-17, rel=1e-12)


def test_grid_covers_a_rotated_ellipse():
    """Two equal crossed cylinders at 0.8 rad: psi = 0.4 (the bisector). The domain covers the bounding box of
    the ellipse (b along psi, a across it), found here by sampling its boundary."""
    cyl = Body(r=(10 * MM, math.inf), **STEEL)
    c = _Point(body1=cyl, body2=cyl, load=1.0, angle=0.8)
    assert c.psi == pytest.approx(0.4, rel=1e-12)
    a, b = 1.0 * MM, 0.2 * MM
    t = np.linspace(0.0, 2 * math.pi, 100001)
    x = b * np.cos(t) * math.cos(c.psi) - a * np.sin(t) * math.sin(c.psi)
    y = b * np.cos(t) * math.sin(c.psi) + a * np.sin(t) * math.cos(c.psi)
    grid = Grid.around(c, _ellipse(a / MM, b / MM, psi=c.psi))
    for coord, extent, span in ((grid.X, x.max(), SPAN_X_0), (grid.Y, y.max(), 1.3)):
        assert coord.max() >= span * extent and coord.max() - grid.h < span * extent + 1e-12   # tight on +side
    assert grid.X.max() > 2 * SPAN_X_0 * b                                    # wider than the unrotated b


def test_grid_spacing_is_dyadic_and_never_coarser():
    c = _Point(body1=Body(**_BALL), body2=Body(**_FLAT), load=1.0)
    assert Grid.around(c, _ellipse(0.2, 0.1), h=3e-6).h == 2.0**-19          # 1.9 um <= 3 um
    assert Grid.around(c, _ellipse(0.2, 0.1), h=2.0**-19).h == 2.0**-19


@pytest.mark.parametrize("ellipse, h, match", [
    (_ellipse(0.2, 0.0), None, "minor semi-axis b"),
    (_ellipse(0.2, math.nan), None, "minor semi-axis b"),
    (_ellipse(0.2, 0.1), -1e-6, "h must be positive"),
    (_ellipse(0.2, 0.1), math.inf, "h must be positive"),
    (_ellipse(math.inf, 0.1), None, "finite major semi-axis"),
])
def test_grid_around_refuses(ellipse, h, match):
    c = _Point(body1=Body(**_BALL), body2=Body(**_FLAT), load=1.0)
    with pytest.raises(ValueError, match=match):
        Grid.around(c, ellipse, h=h)


_h = 2.0**-20
_x, _y = np.meshgrid(np.arange(4) * _h, np.arange(3) * _h, indexing="ij")


@pytest.mark.parametrize("X, Y, h, match", [
    (_x, _y[:, :2], _h, "same shape"),
    (_x, _y, 0.0, "h must be positive"),
    (_x / _h * 1e-6, _y / _h * 1e-6, 1e-6, "power of two"),                    # consistent but not dyadic
    (_x.T, _y.T, _h, "X must vary only along axis 0"),                         # axes swapped
    (_x, _y, 2 * _h, "step of X"),
    (_x, 2 * _y, _h, "step of Y"),
])
def test_hand_built_grid_is_checked(X, Y, h, match):
    with pytest.raises(ValueError, match=match):
        Grid(X=X, Y=Y, h=h)


@pytest.mark.parametrize("contrary", [{"periodic_geometry": False}, {"periodic_axes": (False, False)},
                                      {"periodic_axes": True}])
def test_step_args_refuses_a_non_periodic_line(roller_10, contrary):
    (c,) = bearing_contacts(roller_10, load=1000.0, races=("inner",))
    grid = Grid.around(c, _ellipse(math.inf, 0.0688))
    with pytest.raises(ValueError, match="a line contact needs"):
        grid.step_args(c, **contrary)


@pytest.mark.parametrize("key", ["normal_load", "interference", "mean_gap"])
def test_step_args_refuses_another_load(key):
    c = _Point(body1=Body(**_BALL), body2=Body(**_FLAT), load=1.0)
    with pytest.raises(ValueError, match=f"{key} not accepted"):
        Grid.around(c, _ellipse(0.2, 0.1)).step_args(c, **{key: 1.0})


def test_step_args_point_with_periodic_flags_keeps_the_full_load():
    c = _Point(body1=Body(**_BALL), body2=Body(**_FLAT), load=7.0)
    args = Grid.around(c, _ellipse(0.2, 0.1)).step_args(c, periodic_geometry=True, periodic_axes=(True, True))
    assert args == {"normal_load": 7.0, "periodic_geometry": True, "periodic_axes": (True, True)}


# ---- results: HertzResult and StaticResult, data only (results/) ---------------------------------

def _hertz(a, b, psi=0.0, line=False, p0=1e9):
    return HertzResult(p0=p0, a=a, b=b, psi=psi, delta=math.nan if line else 1e-6, b_over_a=0.0 if line else b / a,
                       E_star=115e9, size_ratio=0.05, valid=True, line=line)


def test_hertz_pressure_on_a_rotated_ellipse():
    """Johnson 1985 p. 95 (elliptical pressure, from Eq. (3.58)): p0 at the centre, 0 at b along psi and at a across it, p0 sqrt(3)/2 at b/2."""
    a, b, psi = 1.0 * MM, 0.2 * MM, 0.4
    hz = _hertz(a, b, psi)
    along, across = np.array([math.cos(psi), math.sin(psi)]), np.array([-math.sin(psi), math.cos(psi)])
    pts = np.array([[0.0, 0.0], b * along, -b * along, a * across, 0.5 * b * along, 1.01 * b * along])
    p = hz.pressure(pts[:, 0], pts[:, 1])
    assert p[0] == hz.p0
    assert p[1:4] == pytest.approx(0.0, abs=1e-6 * hz.p0)
    assert p[4] == pytest.approx(hz.p0 * math.sqrt(0.75), rel=1e-12) and p[5] == 0.0


def test_hertz_pressure_of_a_line_is_a_strip():
    """Johnson 1985 Eq. (4.44): p0 sqrt(1 - (x/b)^2), independent of y (a = inf)."""
    hz = _hertz(math.inf, 0.1 * MM, line=True)
    X, Y = np.meshgrid(np.linspace(-0.15, 0.15, 7) * MM, np.linspace(-5.0, 5.0, 4) * MM, indexing="ij")
    p = hz.pressure(X, Y)
    np.testing.assert_array_equal(p, p[:, :1] * np.ones_like(p))
    np.testing.assert_allclose(p[:, 0], hz.p0 * np.sqrt(np.clip(1 - (X[:, 0] / (0.1 * MM))**2, 0, None)), rtol=1e-12)


def test_hertz_pressure_integrates_to_the_load():
    """Johnson 1985 Eq. (4.27): P = (2/3) p0 pi a b, summed on a fine grid over an ellipse turned by 0.4 rad."""
    hz = _hertz(1.0 * MM, 0.2 * MM, psi=0.4)
    h = 2e-6
    x = np.arange(-1.1 * MM, 1.1 * MM, h)
    X, Y = np.meshgrid(x, x, indexing="ij")
    assert hz.pressure(X, Y).sum() * h**2 == pytest.approx(2 / 3 * hz.p0 * math.pi * hz.a * hz.b, rel=1e-3)


def test_hertz_half_extents():
    hz = _hertz(1.0 * MM, 0.2 * MM, psi=0.4)
    assert hz.half_extents() == (math.hypot(0.2 * MM * math.cos(0.4), 1.0 * MM * math.sin(0.4)),
                                 math.hypot(0.2 * MM * math.sin(0.4), 1.0 * MM * math.cos(0.4)))
    assert _hertz(math.inf, 0.1 * MM, line=True).half_extents() == (0.1 * MM, math.inf)


def test_hertz_record_keys():
    assert list(_hertz(1.0 * MM, 0.2 * MM).record()) == ["p0_Pa", "a_m", "b_m", "psi_rad", "delta_m", "b_over_a",
                                                        "E_star_Pa", "size_ratio", "valid", "line"]
    assert list(_hertz(1.0 * MM, 0.2 * MM).record("_ref"))[:5] == ["p0_ref_Pa", "a_ref_m", "b_ref_m", "psi_ref_rad",
                                                                   "delta_ref_m"]


def _static_result(contact, grid, hz):
    """A StaticResult filled by hand (the solver fills it in C6): Hertz pressure as the 'solution'."""
    z1, z2 = contact.profiles(grid)
    p = hz.pressure(grid.X, grid.Y)
    return StaticResult(contact=contact, grid=grid, z1=z1, z2=z2, gap0=-(z1 + z2), load_applied=contact.load,
                        period=math.nan, use_frequency_domain=True, method="pk", tolerance=1e-9, converged=True,
                        solve_time=0.1, p=p, delta=hz.delta, p0=float(p.max()), a=hz.a, b=hz.b,
                        load=float(p.sum() * grid.h**2), apex_offset_nodes=0.0, hertz=hz, err_p0=0.0, err_a=0.0,
                        err_b=0.0, err_delta=0.0, err_load=0.0, load_ok=True, edge_clear=True, margin_ok=True,
                        resolution_ok=True, within_reference=True,
                        commits={"axisforge": "abc", "slippy": "def"})


def test_static_record_of_a_6210_contact(load_case):
    """record(): every scalar, no arrays; identification from the contact's own fields; R_x = 1/kappa_max =
    5.198 mm (BR1 section 5), R_y 165.1 mm; the Hertz values with _ref; missing commits 'unknown'."""
    (c,) = bearing_contacts(load_case("iso16281_ball_contact").build_bearing(0.0), load=1000.0, races=("inner",))
    hz = _hertz(1.5307 * MM, 0.1666 * MM, p0=1.88e9)
    rec = _static_result(c, Grid.around(c, hz), hz).record()
    assert all(v is None or isinstance(v, (bool, int, float, str)) for v in rec.values())
    assert (rec["contact"], rec["label"], rec["line"], rec["race"], rec["alpha_rad"], rec["j"]) == (
        "BallRacewayContact", "6210_inner", False, "inner", 0.0, None)
    assert rec["R_x_m"] / MM == pytest.approx(5.198, abs=0.0005) and rec["R_y_m"] / MM == pytest.approx(165.1, abs=0.05)
    assert rec["r2_x_m"] == pytest.approx(28.65 * MM) and rec["r2_y_m"] == pytest.approx(-6.604 * MM)
    assert rec["E1_Pa"] == 210e9 and rec["Q_N"] == 1000.0 and math.isnan(rec["w_N_per_m"])
    assert math.isnan(rec["period_m"]) and rec["hertz_valid"] is True
    assert (rec["n_x"], rec["n_y"], rec["h_m"]) == (90, 264, 2.0**-16)
    assert rec["span_x"] >= SPAN_X_0 and rec["span_y"] >= 1.3
    assert rec["p0_ref_Pa"] == 1.88e9 and rec["b_ref_m"] == 0.1666 * MM and rec["p0_Pa"] == pytest.approx(1.88e9)
    assert (rec["commit_axisforge"], rec["commit_slippy"], rec["commit_design_studies"]) == ("abc", "def", "unknown")
    for key in ("body1", "body2", "kappa_max", "p", "z1", "grid", "hertz", "psi_ref_rad", "E_star_ref_Pa",
                "line_ref", "valid_ref", "load_contact"):
        assert key not in rec


def test_static_record_of_a_line_contact(roller_10):
    (c,) = bearing_contacts(roller_10, load=1000.0, races=("inner",))
    hz = _hertz(math.inf, 0.0688 * MM, line=True)
    rec = _static_result(c, Grid.around(c, hz), hz).record()
    assert rec["R_y_m"] == math.inf and math.isnan(rec["span_y"]) and rec["span_x"] >= SPAN_X_0
    assert rec["k"] is None and rec["x_mm"] is None and rec["contact"] == "RollerRacewayContact"
    assert rec["w_N_per_m"] == pytest.approx(1e5) and math.isnan(rec["Q_N"])


# ---- FZG C40 checks that do not use the bridge (validation/gears/fzg_geometry_ph) -------------

@pytest.mark.xfail(strict=True, raises=AssertionError, reason="known gap GL-10: AxisForge 1.4624 vs printed 1.44")
def test_fzg_c40_eps_alpha_known_gap(load_case):
    """C40 transverse contact ratio: printed 1.44 (Fernandes2015b CSV L13 = Martins2015 L111), AxisForge 1.4624.

    A known gap (GL-10) kept visible: it fails today with AssertionError; an XPASS would mean it closed.
    """
    pytest.importorskip("axisforge.solvers.machine_elements.gears.contact")
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
    pytest.importorskip("axisforge.solvers.machine_elements.gears.contact")
    cc = load_case("fzg_geometry_ph")
    pair = cc.build_pair("C40")
    T1_Nm = cc.WHEEL_TORQUE_NM[stage] * pair.gear1.z / pair.gear2.z            # CSV L168/178/188, on the wheel
    F_r_N = cc.FAMILY.mesh_forces(pair, T1_Nm, rot1=+1).F_r_N
    assert F_r_N == pytest.approx(F_r_printed_N, abs=0.5)
