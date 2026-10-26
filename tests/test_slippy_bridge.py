"""slippy_bridge.contact: bodies and contacts (core), then the FZG C40 checks that need no bridge."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from slippy_bridge.contact.core.contact import Body, Contact

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
