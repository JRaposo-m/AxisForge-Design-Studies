"""slippy_bridge.contact: bodies and contacts (core), then the FZG C40 checks that need no bridge."""
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

af_c = pytest.importorskip("axisforge.core")

from axisforge.core.machine_elements.bearings.families import available_families  # noqa: E402
from axisforge.core.machine_elements.gears.cylindrical import CylindricalFamily  # noqa: E402
from axisforge.results import (BallBearingResult, BallLoadDistributionResult,  # noqa: E402
                               BearingAnalysisResult, RollerLoadDistributionResult)
from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281.contact_solver import (  # noqa: E402
    ISO16281BallSolver)
from axisforge.solvers.machine_elements.gears.contact import contact_inputs  # noqa: E402

from studies.r00_analytical_references.contact.hertz import hertz_point  # noqa: E402

from slippy_bridge.contact.core.contact import Body, Contact, SweptCircleBody  # noqa: E402
from slippy_bridge.contact.core.machine_elements.bearings import (BallRacewayContact, BearingContact,  # noqa: E402
                                                                  DeepGrooveBallContact,
                                                                  RollerRacewayContact, bearing_contacts)
from slippy_bridge.contact.core.machine_elements.gears import (ToothContact, gear_contacts,  # noqa: E402
                                                               path_diameters)
from slippy_bridge.contact.mesh.grid import SPAN_X_0, Grid  # noqa: E402
from slippy_bridge.contact.results.hertz import HertzResult  # noqa: E402
from slippy_bridge.contact.results.static import StaticResult  # noqa: E402
from slippy_bridge.contact.solvers.hertz import hertz  # noqa: E402
from slippy_bridge.contact.solvers.static import _measure, solve  # noqa: E402
from slippy_bridge.contact.solvers.surfaces import surface_pair  # noqa: E402
from slippy.contact import hertz_full  # noqa: E402

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
    body = SweptCircleBody(r=r, **STEEL)
    np.testing.assert_allclose(body.profile(_X, _Y), expected(_X, _Y), rtol=1e-12, atol=1e-18)
    assert body.profile(np.zeros((1, 1)), np.zeros((1, 1)))[0, 0] == 0.0                   # apex at 0


@pytest.mark.parametrize("r", [_INNER, _OUTER, (20 * MM, 8 * MM)], ids=["inner", "outer", "convex"])
def test_torus_profile_lies_on_the_implicit_torus(r):
    """Implicit torus: depth w = -z; the axis of revolution is the line (w = r_x, parallel to y_b); the
    signed distance to it s = sign(r_x) sqrt(x^2 + (w - r_x)^2) and y satisfy (s - (r_x - r_y))^2 + y^2 = r_y^2
    (generating circle of radius |r_y| centred at r_x - r_y from the axis)."""
    r_x, r_y = r
    w = -SweptCircleBody(r=r, **STEEL).profile(_X, _Y)
    s = math.copysign(1.0, r_x) * np.sqrt(_X**2 + (w - r_x)**2)
    np.testing.assert_allclose((s - (r_x - r_y))**2 + _Y**2, r_y**2, rtol=1e-12)


@pytest.mark.parametrize("r", [_INNER, _OUTER, (20 * MM, 8 * MM)], ids=["inner", "outer", "convex"])
def test_torus_profile_is_the_hertz_paraboloid_near_the_apex(r):
    """Johnson 1985 Eq. (4.2a): -z -> x^2/(2 r_x) + y^2/(2 r_y); relative error O((x/r)^2) < 1e-4 at 0.05 mm."""
    r_x, r_y = r
    x, y = _X / 40, _Y / 30                                                     # |x|, |y| <= 0.05 mm
    np.testing.assert_allclose(-SweptCircleBody(r=r, **STEEL).profile(x, y), x**2 / (2 * r_x) + y**2 / (2 * r_y),
                               rtol=1e-3, atol=1e-20)


def test_profile_signs_convex_and_concave():
    """Convex body falls away from the contact plane (-sag < 0), a concave one rises (> 0)."""
    assert SweptCircleBody(r=5 * MM, **STEEL).profile(_X, _Y)[0, 0] < 0
    assert SweptCircleBody(r=-50 * MM, **STEEL).profile(_X, _Y)[0, 0] > 0
    assert SweptCircleBody(r=math.inf, **STEEL).profile(_X, _Y).tolist() == np.zeros_like(_X).tolist()


@pytest.mark.parametrize("angle", [0.3, -1.1, math.pi / 2])
def test_profile_rotation_is_the_body_frame(angle):
    body = SweptCircleBody(r=_INNER, **STEEL)
    x_b = _X * math.cos(angle) + _Y * math.sin(angle)
    y_b = -_X * math.sin(angle) + _Y * math.cos(angle)
    np.testing.assert_allclose(body.profile(_X, _Y, angle), body.profile(x_b, y_b), rtol=1e-12, atol=1e-18)


def test_cylinder_turned_by_90_deg_lies_along_x():
    turned = SweptCircleBody(r=(10 * MM, math.inf), **STEEL).profile(_X, _Y, math.pi / 2)
    np.testing.assert_allclose(turned, SweptCircleBody(r=(math.inf, 10 * MM), **STEEL).profile(_X, _Y),
                               rtol=1e-9, atol=1e-15)


@pytest.mark.parametrize("r, x, y", [
    (1.0 * MM, _X, _Y),                                  # |x| beyond the sphere (NaN)
    ((-3.0 * MM, -1.6 * MM), _X, _Y),                    # |x_b| >= |rho|, outside the swept circle
    # at y = 1.5 mm sag_y = -1.043 mm, so rho = r_x - sag_y = +0.043 mm has crossed the axis of
    # revolution: the formula gives a finite but wrong height, only the sign check catches it
    ((-1.0 * MM, -1.6 * MM), np.zeros((1, 1)), np.full((1, 1), 1.5 * MM)),
])
def test_grid_beyond_the_body_raises(r, x, y):
    with pytest.raises(ValueError, match="beyond the body"):
        SweptCircleBody(r=r, **STEEL).profile(x, y)


# ---- relative curvatures, psi, E* (Johnson 1985 section 4.1) ----------------------------------

def test_curvatures_of_the_6210_inner_contact():
    """Ball 12.7 mm in the inner groove of a 6210 (Dpw 70, f_i 0.52), alpha 0 (BR1 section 5, printed to 5 digits
    [1/mm]), and the unrounded hand sums 1/6.35 + 1/28.65, 1/6.35 - 1/6.604."""
    c = _Point(body1=SweptCircleBody(r=6.35 * MM, **STEEL), body2=SweptCircleBody(r=_INNER, **STEEL), load=1000.0)
    assert c.kappa_max == pytest.approx(0.19238 / MM, abs=0.000005 / MM)
    assert c.kappa_min == pytest.approx(0.00606 / MM, abs=0.000005 / MM)
    assert c.kappa_max == pytest.approx((1 / 6.35 + 1 / 28.65) / MM, rel=1e-12)
    assert c.kappa_min == pytest.approx((1 / 6.35 - 1 / 6.604) / MM, rel=1e-12)
    assert c.psi == 0.0


def test_crossed_cylinders_at_90_deg_are_a_sphere():
    R = 10 * MM
    cyl = SweptCircleBody(r=(R, math.inf), **STEEL)
    c = _Point(body1=cyl, body2=cyl, load=1.0, angle=math.pi / 2)
    assert c.kappa_max == pytest.approx(1 / R, rel=1e-12)
    assert c.kappa_min == pytest.approx(1 / R, rel=1e-12)


@pytest.mark.parametrize("sign", [+1.0, -1.0])
def test_crossed_cylinders_at_45_deg(sign):
    """Johnson 1985 p. 86: A, B = (1 -+ 1/sqrt 2)/(2R), so kappa = 2A, 2B; kappa_max bisects the two axes."""
    R = 10 * MM
    cyl = SweptCircleBody(r=(R, math.inf), **STEEL)
    c = _Point(body1=cyl, body2=cyl, load=1.0, angle=sign * math.pi / 4)
    assert c.kappa_max == pytest.approx((1 + math.sqrt(2) / 2) / R, rel=1e-12)
    assert c.kappa_min == pytest.approx((1 - math.sqrt(2) / 2) / R, rel=1e-12)
    assert c.psi == pytest.approx(sign * math.radians(22.5), rel=1e-12)


# two convex tori at 0.3 rad, both more curved along x_b than across (outside the SlipPY R-534 domain)
_T1, _T2, _ANGLE = (5 * MM, 20 * MM), (8 * MM, 30 * MM), 0.3


def test_two_tori_at_an_angle_match_johnson():
    """Johnson 1985 Eqs. (4.4)-(4.5), p. 85: A + B and |B - A| from the body radii and alpha; kappa = 2A, 2B.
    psi: the eigenvector of the largest eigenvalue of K = sum R diag(1/r_x, 1/r_y) R^T (numpy eigh)."""
    c = _Point(body1=SweptCircleBody(r=_T1, **STEEL), body2=SweptCircleBody(r=_T2, **STEEL), load=1.0, angle=_ANGLE)
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
              (((10 * MM, math.inf), (10 * MM, math.inf)), math.pi / 4),
              (((10 * MM, math.inf), (12 * MM, math.inf)), 0.3)]


@pytest.mark.parametrize("radii, angle", _GAP_CASES)
def test_profiles_give_the_gap_curvatures(radii, angle):
    """The gap -(z1 + z2) of the exact profiles is kappa t^2 / 2 at a small distance t along psi (kappa_max) and
    across it (kappa_min): profiles() and the curvature tensor describe the same two bodies."""
    c = _Point(body1=SweptCircleBody(r=radii[0], **STEEL),
               body2=SweptCircleBody(r=radii[1], **STEEL), load=1.0, angle=angle)
    t = 1e-5                                                                      # [m], (t / r)^2 <= 1e-5
    cp, sp = math.cos(c.psi), math.sin(c.psi)
    grid = SimpleNamespace(X=np.array([[t * cp, -t * sp]]), Y=np.array([[t * sp, t * cp]]))
    z1, z2 = c.profiles(grid)
    gap = -(z1 + z2)[0]
    assert gap[0] == pytest.approx(c.kappa_max * t**2 / 2, rel=1e-4)
    assert gap[1] == pytest.approx(c.kappa_min * t**2 / 2, rel=1e-4)


@pytest.mark.parametrize("radii, angle", _GAP_CASES)
def test_contact_and_hertz_agree_with_hertz_full_on_the_raw_radii(radii, angle):
    """SlipPY hertz_full on the per-body radii with its own angle: kappa_max, kappa_min = 1/relative_radii[0], [1];
    psi = alpha (mod pi, no pi/2 shift); E* = e_star; and the bridge's hertz() (equivalent body at angle 0) gives
    the same p0, sorted semi-axes and approach, so the psi rotation and SlipPY's angle agree. Needs SlipPY
    local-fixes (the omega factor 2). Cases outside the SlipPY R-534 domain (both bodies more curved across than
    along)."""
    c = _Point(body1=SweptCircleBody(r=radii[0], **STEEL),
               body2=SweptCircleBody(r=radii[1], **STEEL), load=100.0, angle=angle)
    sol = hertz_full(r1=radii[0], r2=radii[1], moduli=(STEEL["E"],) * 2, v=(STEEL["nu"],) * 2, load=100.0,
                     angle=angle)
    assert c.kappa_max == pytest.approx(1 / sol["relative_radii"][0], rel=1e-12)
    assert c.kappa_min == pytest.approx(1 / sol["relative_radii"][1], rel=1e-12)
    assert abs(math.remainder(c.psi - sol["alpha"], math.pi)) < 1e-12
    assert c.E_star == pytest.approx(sol["e_star"], rel=1e-14)
    hz = hertz(c)
    assert hz.p0 == pytest.approx(sol["max_pressure"], rel=1e-6)
    assert [hz.b, hz.a] == pytest.approx(sorted(sol["contact_radii"]), rel=1e-6)
    assert hz.delta == pytest.approx(sol["total_deflection"], rel=1e-6)


def test_line_contact_has_zero_kappa_min_along_y():
    c = _Line(body1=SweptCircleBody(r=(10 * MM, math.inf), **STEEL),
              body2=SweptCircleBody(r=math.inf, **STEEL), load=1e5)
    assert (c.kappa_max, c.kappa_min, c.psi) == (pytest.approx(100.0), 0.0, 0.0)


def test_parallel_cylinders_turned_by_90_deg_are_a_line_along_y():
    """Body 2 r = (inf, R) turned by pi/2 is the cylinder (R, inf): cos(pi/2) = 6e-17 must not tilt the line."""
    R = 10 * MM
    c = _Line(body1=SweptCircleBody(r=(R, math.inf), **STEEL),
              body2=SweptCircleBody(r=(math.inf, R), **STEEL), load=1e5,
              angle=math.pi / 2)
    assert (c.kappa_max, c.kappa_min, c.psi) == (pytest.approx(2 / R, rel=1e-12), 0.0, 0.0)


def test_E_star():
    c = _Point(body1=SweptCircleBody(r=5 * MM, E=210e9, nu=0.3),
               body2=SweptCircleBody(r=math.inf, E=70e9, nu=0.33), load=1.0)
    assert c.E_star == pytest.approx(1 / ((1 - 0.3**2) / 210e9 + (1 - 0.33**2) / 70e9), rel=1e-14)


# ---- input checks (D-0034 item 6) ---------------------------------------------------------------

_BALL, _FLAT = dict(r=5 * MM, **STEEL), dict(r=math.inf, **STEEL)


@pytest.mark.parametrize("body", [
    dict(r=0.0, **STEEL),
    dict(r=(math.nan, 5 * MM), **STEEL),
    dict(r=(5 * MM, -math.inf), **STEEL),
    dict(r=(1.0, 2.0, 3.0), **STEEL),
    dict(**STEEL),                                                              # r missing
])
def test_bad_radius_raises(body):
    with pytest.raises(ValueError, match="SweptCircleBody: r"):
        SweptCircleBody(**body)


def test_body_is_abstract():
    """Body is the contract (E, nu, r, profile); only a subclass that gives r and profile is created."""
    with pytest.raises(TypeError, match="abstract"):
        Body(**STEEL)

    class _NoProfile(Body):
        r = (1.0, 1.0)
    with pytest.raises(TypeError, match="abstract"):
        _NoProfile(**STEEL)


@pytest.mark.parametrize("E, nu, match", [
    (0.0, 0.3, "E must be positive"), (-1.0, 0.3, "E must be positive"), (math.inf, 0.3, "E must be positive"),
    (210e9, 0.0, r"nu must be in \(0, 0.5\)"), (210e9, 0.5, r"nu must be in \(0, 0.5\)"),
])
def test_bad_material_raises(E, nu, match):
    with pytest.raises(ValueError, match=match):
        SweptCircleBody(r=5 * MM, E=E, nu=nu)


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
        cls(body1=SweptCircleBody(**body1), body2=SweptCircleBody(**body2), label="c1", **kwargs)


def test_contact_names_its_label():
    with pytest.raises(ValueError, match="Contact 'ball_7': load"):
        _Point(body1=SweptCircleBody(**_BALL), body2=SweptCircleBody(**_FLAT), load=-1.0, label="ball_7")


def test_contact_is_abstract():
    with pytest.raises(TypeError, match="abstract"):
        Contact(body1=SweptCircleBody(**_BALL), body2=SweptCircleBody(**_FLAT), load=1.0)


def test_profiles_turn_body_2_only():
    body1, body2 = SweptCircleBody(r=(10 * MM, math.inf), **STEEL), SweptCircleBody(r=(10 * MM, math.inf), **STEEL)
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


def _6210_contacts(load_case, **kw):
    """Contacts of the 6210 of the iso16281_ball_contact case at s = 0 (alpha_0 = 0), Q 1000 N given by hand."""
    return DeepGrooveBallContact.from_bearing(load_case("iso16281_ball_contact").build_bearing(0.0), load=1000.0, **kw)


def test_6210_hand_check(load_case):
    """BR1 section 5: 6210-like (Dw 12.7, Dpw 70, f 0.52, s = 0 so alpha_0 = 0), Q 1000 N given by hand.
    R_x = 1/(1/6.35 + 1/28.65) = 5.198 mm inner, 1/(1/6.35 - 1/41.35) = 7.502 mm outer; R_y =
    1/(1/6.35 - 1/6.604) = 165.1 mm both; printed to 4 digits."""
    inner, outer = _6210_contacts(load_case)
    assert all(isinstance(c, DeepGrooveBallContact) and c.load == 1000.0 and c.psi == 0.0 for c in (inner, outer))
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
    (c,) = DeepGrooveBallContact.from_bearing(ball, load=100.0, races=(race,))
    f = ball.ri / ball.Dw if race == "inner" else ball.re / ball.Dw
    g = ball.Dw * math.cos(ball.alpha_0) / ball.Dpw
    iso = (4.0 - 1.0 / f + sign * 2.0 * g / (1.0 - sign * g)) / (ball.Dw * MM)
    assert c.kappa_max + c.kappa_min == pytest.approx(iso, rel=1e-12)
    assert c.body2.r[0] * sign > 0 and c.body2.r[1] < 0


def test_alpha_default_is_family_angle(ball):
    (a0,), (given,) = (DeepGrooveBallContact.from_bearing(ball, load=100.0, races=("outer",), **kw)
                       for kw in ({}, {"alpha": ball.alpha_0}))
    assert given.body2.r == pytest.approx(a0.body2.r, rel=1e-12) and a0.alpha_rad == ball.alpha_0


def test_alpha_changes_only_the_rolling_radius(ball):
    (a0,), (aj,) = (DeepGrooveBallContact.from_bearing(ball, load=100.0, races=("inner",), **kw)
                    for kw in ({}, {"alpha": 0.4}))
    assert aj.body1.r == a0.body1.r and aj.body2.r[1] == a0.body2.r[1] and aj.alpha_rad == 0.4
    assert aj.body2.r[0] == pytest.approx((ball.Dpw - ball.Dw * math.cos(0.4)) / (2 * math.cos(0.4)) * MM)


def test_ball_contacts_from_result(ball, ball_row):
    cs = DeepGrooveBallContact.from_bearing(ball, ball_row, races=("inner", "outer"), elements="loaded")
    assert len(cs) == 2 * 5                                         # 5 loaded balls x 2 races
    c0 = cs[0]
    assert isinstance(c0, DeepGrooveBallContact) and c0.label == "b_inner_e0"
    assert (c0.race, c0.alpha_rad, c0.row, c0.j, c0.phi_rad, c0.phi_global_rad) == ("inner", 0.2, 0, 0, 0.0, None)
    assert c0.load == 300.0
    (by_hand,) = DeepGrooveBallContact.from_bearing(ball, load=300.0, alpha=0.2, races=("inner",))
    assert c0.body2.r == by_hand.body2.r
    assert DeepGrooveBallContact.from_bearing(ball, ball_row, elements="max", races=("outer",))[0].j == 0


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
    cs = DeepGrooveBallContact.from_bearing(ball_6210, result)
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
    cs = DeepGrooveBallContact.from_bearing(ball, ball_result, races=("inner",))
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
    assert [c.j for c in DeepGrooveBallContact.from_bearing(ball, ball_row, races=("inner",), elements=[0, 4])] == [0]
    with pytest.raises(ValueError, match="row applies"):
        DeepGrooveBallContact.from_bearing(ball, ball_row, row=1)


def test_bearing_result_gives_global_angle(ball, ball_result):
    c = DeepGrooveBallContact.from_bearing(ball, ball_result, races=("inner",), elements="max")[0]
    assert c.phi_global_rad == pytest.approx(ball_result.phi_j_global(0)[0])
    assert DeepGrooveBallContact.from_bearing(ball, ball_result.rows[0], races=("inner",))[0].phi_global_rad is None


def test_analysis_result_of_the_solver_is_accepted(ball, ball_result):
    """``SolverBase.solve`` returns a ``BearingAnalysisResult`` per label; the bridge reads its load distribution."""
    analysis = BearingAnalysisResult(label="b", load_distribution=ball_result)
    assert ([_key(c) for c in DeepGrooveBallContact.from_bearing(ball, analysis)]
            == [_key(c) for c in DeepGrooveBallContact.from_bearing(ball, ball_result)])


@pytest.mark.parametrize("call, match", [
    (lambda ball, res: DeepGrooveBallContact.from_bearing(ball, replace(res, ok=False, residual=12.0)),
     "not converged"),
    (lambda ball, res: DeepGrooveBallContact.from_bearing(ball, replace(res, label="other")), "not of bearing"),
    (lambda ball, res: DeepGrooveBallContact.from_bearing(ball, res, alpha=0.1), "alpha applies to load= only"),
    (lambda ball, res: DeepGrooveBallContact.from_bearing(ball, res, load=1.0), "either result"),
    (lambda ball, res: DeepGrooveBallContact.from_bearing(ball), "either result"),
    (lambda ball, res: bearing_contacts(ball, res, laminae=True), "laminae applies"),
    (lambda ball, res: DeepGrooveBallContact.from_bearing(ball, load=1.0, row=1), "row applies to result= only"),
    (lambda ball, res: DeepGrooveBallContact.from_bearing(
        ball, replace(res, rows=(replace(res.rows[0], Q_j=0.0 * res.rows[0].Q_j),)),
        elements="max"), "no loaded element"),
    (lambda ball, res: DeepGrooveBallContact.from_bearing(ball, replace(res.rows[0], Q_j=0.0 * res.rows[0].Q_j)),
     "no loaded element"),
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
    """Each family in AxisForge ``available_families()`` either gives contacts with a solvable, valid Hertz
    solution or a NotImplementedError that names it. A family AxisForge itself cannot assemble is skipped (an
    AxisForge bug, not the bridge)."""
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
        for c in cs:
            hz = hertz(c)
            assert hz.p0 > 0 and 0 < hz.b <= hz.a and hz.valid


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
    with pytest.raises(ValueError, match="line contact row, DeepGrooveBallContact a point contact"):
        DeepGrooveBallContact.from_bearing(ball, roller_row)


def test_bearing_without_elastic_constants_names_them():
    bare = _bearing(af_c.DeepGrooveBallFamily(), "locating", contact=False, Dw=7.0, Dpw=31.0, Z=9, s=0.010)
    with pytest.raises(ValueError, match="contact=ContactAnalysis.ISO16281 and e1, e2, nu1, nu2"):
        DeepGrooveBallContact.from_bearing(bare, load=100.0)


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
        BallRacewayContact(body1=SweptCircleBody(r=6.35 * MM, **STEEL),
                           body2=SweptCircleBody(**race), load=1.0, **kwargs)


@pytest.mark.parametrize("alpha", [math.pi / 2, math.pi / 2 + 4.7e-12, math.radians(98.7)])
def test_thrust_raceway_accepts_any_angle(alpha):
    """A raceway flat along rolling does not depend on alpha; the solver gives a thrust ball alpha_j at and
    beyond 90 deg (pure axial: pi/2 + 4.7e-12; with some Fr up to 98.7 deg)."""
    c = BallRacewayContact(body1=SweptCircleBody(r=6.35 * MM, **STEEL),
                           body2=SweptCircleBody(**_THRUST), load=1.0, race="inner",
                           alpha_rad=alpha)
    assert c.psi == 0.0 and c.kappa_max == pytest.approx(1 / (6.35 * MM))


def test_bearing_contact_is_abstract():
    with pytest.raises(TypeError, match="abstract"):
        BearingContact(body1=SweptCircleBody(r=6.35 * MM, **STEEL),
                       body2=SweptCircleBody(**_RACE), load=1.0, race="inner", alpha_rad=0.0)


# ---- true ball raceway: the groove swept about the bearing axis (DeepGrooveBallContact) ------------

# Dw 12.7, Dpw 70, groove 0.52 Dw [m] (D-0034 item 10; cowork's independent check of 2026-10-26 19:36)
_DW, _DPW, _RG = 12.7 * MM, 70.0 * MM, 0.52 * 12.7 * MM


def _raceway(race, alpha_deg):
    """True raceway with the contact point at R_P = (Dpw -/+ Dw cos(alpha))/2 (ball centre on the pitch circle)."""
    alpha = math.radians(alpha_deg)
    sign = 1.0 if race == "inner" else -1.0
    return DeepGrooveBallContact.Raceway(race=race, R_P=(_DPW - sign * _DW * math.cos(alpha)) / 2, r_g=_RG,
                                         alpha=alpha, **STEEL)


def _local_torus(body):
    """The SweptCircleBody with the same principal radii: the local torus."""
    return SweptCircleBody(r=body.r, **STEEL)


@pytest.mark.parametrize("race", ["inner", "outer"])
def test_true_raceway_at_alpha_0_is_the_torus(race):
    """At alpha = 0 the groove is swept about an axis at r_x = +/-R_P: the torus is exact. Equal at every node to
    rel 1e-12; atol 1e-17 m is the rounding of the closed form (r - sqrt(r^2 - x^2) at r ~ 0.03 m)."""
    X, Y = np.meshgrid(np.linspace(-0.3, 0.3, 7) * MM, np.linspace(-3.0, 3.0, 13) * MM, indexing="ij")
    body = _raceway(race, 0.0)
    np.testing.assert_allclose(body.profile(X, Y), _local_torus(body).profile(X, Y), rtol=1e-12, atol=1e-17)


@pytest.mark.parametrize("race", ["inner", "outer"])
@pytest.mark.parametrize("alpha_deg", [25.0, 40.0])
def test_true_raceway_on_the_axes_lines(race, alpha_deg):
    """On x = 0 the section is the groove circle itself: equal to the local torus to 1e-12 m up to |y| = 3 mm.
    OPEN SB-42: on y = 0 the true section is a plane section of the raceway inclined by alpha, not a circle: equal to
    fourth order only (x^4); inner 40 deg: 1.0e-12 m at 0.1 mm, 5.1e-12 m at 0.15 mm (b of the 6210 contacts is
    0.17-0.20 mm), so 6e-12 m up to |x| = 0.15 mm."""
    body = _raceway(race, alpha_deg)
    y = np.linspace(-3.0, 3.0, 13)[None, :] * MM
    np.testing.assert_allclose(body.profile(0 * y, y), _local_torus(body).profile(0 * y, y), rtol=0, atol=1e-12)
    x = np.linspace(-0.15, 0.15, 9)[:, None] * MM
    np.testing.assert_allclose(body.profile(x, 0 * x), _local_torus(body).profile(x, 0 * x), rtol=0, atol=6e-12)


@pytest.mark.parametrize("race, x_mm, y_mm, diff_um", [
    ("inner", 0.15, 2.0, 0.0915), ("inner", 0.3, 3.0, 0.588), ("outer", 0.15, 2.0, 0.0526),
])
def test_true_raceway_against_the_local_torus_off_the_axes(race, x_mm, y_mm, diff_um):
    """Cowork's independent check (2026-10-26 19:36, D-0034 item 10), alpha 40 deg: z_model - z_true (both SlipPY
    sign) off the x and y axes, within 2 % of each value. +y_b is the meridian tangent cos(alpha) e_a - sin(alpha) e_r
    on both raceways (DeepGrooveBallContact.Raceway); the difference is odd in y (an x^2 y term)."""
    body = _raceway(race, 40.0)
    x, y = np.full((1, 1), x_mm * MM), np.full((1, 1), y_mm * MM)
    diff = (_local_torus(body).profile(x, y) - body.profile(x, y))[0, 0]
    assert diff / 1e-6 == pytest.approx(diff_um, rel=0.02)


@pytest.mark.parametrize("body", [DeepGrooveBallContact.Ball(Dw=_DW, **STEEL), _raceway("inner", 40.0),
                                  _raceway("outer", 40.0), _raceway("outer", -25.0)],
                         ids=["ball", "inner_40", "outer_40", "outer_-25"])
def test_profile_agrees_with_r_near_the_contact_point(body):
    """Body contract (Johnson 1985 Eq. (4.2a)): -profile = x^2/(2 r_x) + y^2/(2 r_y) + O(|x|^3) near the contact
    point; at |x|, |y| <= 0.05 mm the third-order term is below 3e-3 of the two terms' magnitudes (they cancel where
    r_x and r_y have opposite signs, so not of their sum; true raceway at 40 deg: 1.5e-3, of order y/r_g). Also on a
    turned frame."""
    x, y = _X / 40, _Y / 30
    r_x, r_y = body.r
    paraboloid, size = x**2 / (2 * r_x) + y**2 / (2 * r_y), x**2 / (2 * abs(r_x)) + y**2 / (2 * abs(r_y))
    assert np.all(np.abs(-body.profile(x, y) - paraboloid) <= 3e-3 * size + 1e-20)
    xb, yb = x * math.cos(0.7) + y * math.sin(0.7), -x * math.sin(0.7) + y * math.cos(0.7)
    np.testing.assert_allclose(body.profile(x, y, 0.7), body.profile(xb, yb), rtol=1e-12, atol=1e-20)


@pytest.mark.parametrize("y_mm", [7.0, -7.0])
def test_node_off_the_true_raceway_raises(y_mm):
    """Beyond the groove (|y| > r_g = 6.604 mm) the node has no root on the raceway."""
    with pytest.raises(ValueError, match="beyond the inner raceway"):
        _raceway("inner", 25.0).profile(np.zeros((1, 1)), np.full((1, 1), y_mm * MM))


@pytest.mark.parametrize("race", ["inner", "outer"])
@pytest.mark.parametrize("alpha_deg", [25.0, 40.0])
def test_true_raceway_mirror_in_alpha(race, alpha_deg):
    """profile(x, y) at -alpha equals profile(x, -y) at +alpha (Raceway docstring): guards the sense of y_b and of
    the axial normal component."""
    plus, minus = _raceway(race, alpha_deg), _raceway(race, -alpha_deg)
    np.testing.assert_allclose(minus.profile(_X / 10, _Y), plus.profile(_X / 10, -_Y), rtol=1e-12, atol=1e-18)


@pytest.mark.parametrize("race", ["inner", "outer"])
@pytest.mark.parametrize("which", ["alpha_0", "alpha_j"])
def test_raceway_radii_equal_the_old_bodies(ball_6210, race, which):
    """The principal radii of the true raceway equal the bodies of C9 (the formula of bearings._bodies): r_x =
    +/-(Dpw -/+ Dw cos a) / (2 cos a) [mm -> m], outer negated; r_y = -ri or -re; at alpha_0 (also AxisForge
    r_inner/r_outer as stored) and at an alpha_j (0.3 rad), rel 1e-12."""
    a = ball_6210.alpha_0 if which == "alpha_0" else 0.3
    (c,) = DeepGrooveBallContact.from_bearing(ball_6210, load=100.0, alpha=a, races=(race,))
    s = 1.0 if race == "inner" else -1.0
    r_g = ball_6210.ri if race == "inner" else ball_6210.re
    old = (s * (ball_6210.Dpw - s * ball_6210.Dw * math.cos(a)) / (2 * math.cos(a)) * MM, -r_g * MM)
    assert c.body2.r == pytest.approx(old, rel=1e-12) and c.body1.r == (ball_6210.Dw / 2 * MM,) * 2
    if which == "alpha_0":
        stored = (ball_6210.r_inner if race == "inner" else ball_6210.r_outer)
        assert c.body2.r == pytest.approx((s * stored[0] * MM, stored[1] * MM), rel=1e-12)


@pytest.mark.parametrize("race, sign", [("inner", +1.0), ("outer", -1.0)])
def test_contact_point_with_the_ball_centre_on_the_pitch_circle(ball_6210, race, sign):
    """OPEN SB-41, option (i): R_P = (Dpw -/+ Dw cos a)/2. Option (ii), the groove fixed in the ring (ISO/TS 16281
    Eq. (13)-(15)), differs by (r_g - Dw/2)(cos a - cos a_0): numbers in the vault note, not here."""
    alpha = math.radians(20.0)
    (c,) = DeepGrooveBallContact.from_bearing(ball_6210, load=100.0, alpha=alpha, races=(race,))
    assert c.body2.R_P == pytest.approx((ball_6210.Dpw - sign * ball_6210.Dw * math.cos(alpha)) / 2 * MM, rel=1e-15)


@pytest.mark.parametrize("family, geometry", [
    (af_c.AngularContactFamily, dict(Dw=8.0, Dpw=40.0, Z=12, alpha_0_deg=25.0)),
    (af_c.CylindricalRollerFamily, dict(Dwe=5.0, Lwe=5.0, Dpw=31.0, Z=14, n_s=50, s=0.010)),
])
def test_deep_groove_class_refuses_another_family(family, geometry):
    arrangement = "non-locating" if family is af_c.CylindricalRollerFamily else "locating"
    bearing = _bearing(family(), arrangement, **geometry)
    with pytest.raises(TypeError, match=f"DeepGrooveBallContact takes a DeepGrooveBallFamily bearing, got "
                                        f"{family.__name__}"):
        DeepGrooveBallContact.from_bearing(bearing, load=100.0)


def test_deep_groove_class_refuses_two_rows_and_bad_indices(ball, ball_result):
    """AxisForge's ISO solver returns one row of Z balls for i = 2 (Q_j of both rows): refused. A row or an element
    index outside the result is named."""
    two_rows = _bearing(af_c.DeepGrooveBallFamily(), "locating", Dw=7.0, Dpw=31.0, Z=9, s=0.010, i=2)
    with pytest.raises(NotImplementedError, match="2-row bearing"):
        DeepGrooveBallContact.from_bearing(two_rows, load=100.0)
    with pytest.raises(ValueError, match="row 1 is not a row of result"):
        DeepGrooveBallContact.from_bearing(ball, ball_result, row=1)
    with pytest.raises(ValueError, match=r"must be indices in \[0, 9\)"):
        DeepGrooveBallContact.from_bearing(ball, ball_result, elements=[-1])


def test_deep_groove_contact_checks_its_bodies(ball):
    (c,) = DeepGrooveBallContact.from_bearing(ball, load=100.0, races=("inner",))
    for kw in (dict(body1=SweptCircleBody(r=c.body1.r, **STEEL)), dict(race="outer"), dict(alpha_rad=0.1)):
        with pytest.raises(ValueError, match="body1 must be its Ball and body2 its Raceway"):
            replace(c, **kw)


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
    (c,) = _6210_contacts(load_case, races=(race,))
    grid = Grid.around(c, _ellipse(a_mm, b_mm))
    assert grid.h == 2.0**-16 and grid.X.shape == (n_x, n_y)
    assert grid.step_args(c) == {"normal_load": 1000.0}


def test_grid_origin_on_the_centre_node(load_case):
    """The contact centre is node (n_x//2, n_y//2); the +x side is the short one and still reaches span_x b."""
    (c,) = _6210_contacts(load_case, races=("inner",))
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
    c = _Line(body1=SweptCircleBody(r=(14.20546 * MM, math.inf), **STEEL),
              body2=SweptCircleBody(r=(21.30819 * MM, math.inf), **STEEL),
              load=75697.52)
    grid = Grid.around(c, _ellipse(math.inf, 0.08519))
    assert grid.h == 2.0**-17 and grid.X.shape == (92, 16)
    assert grid.step_args(c)["normal_load"] == pytest.approx(75697.52 * 16 * 2.0**-17, rel=1e-12)


def test_grid_covers_a_rotated_ellipse():
    """Two equal crossed cylinders at 0.8 rad: psi = 0.4 (the bisector). The domain covers the bounding box of
    the ellipse (b along psi, a across it), found here by sampling its boundary."""
    cyl = SweptCircleBody(r=(10 * MM, math.inf), **STEEL)
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
    c = _Point(body1=SweptCircleBody(**_BALL), body2=SweptCircleBody(**_FLAT), load=1.0)
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
    c = _Point(body1=SweptCircleBody(**_BALL), body2=SweptCircleBody(**_FLAT), load=1.0)
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
    c = _Point(body1=SweptCircleBody(**_BALL), body2=SweptCircleBody(**_FLAT), load=1.0)
    with pytest.raises(ValueError, match=f"{key} not accepted"):
        Grid.around(c, _ellipse(0.2, 0.1)).step_args(c, **{key: 1.0})


def test_step_args_point_with_periodic_flags_keeps_the_full_load():
    c = _Point(body1=SweptCircleBody(**_BALL), body2=SweptCircleBody(**_FLAT), load=7.0)
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
    (c,) = _6210_contacts(load_case, races=("inner",))
    hz = _hertz(1.5307 * MM, 0.1666 * MM, p0=1.88e9)
    rec = _static_result(c, Grid.around(c, hz), hz).record()
    assert all(v is None or isinstance(v, (bool, int, float, str)) for v in rec.values())
    assert (rec["contact"], rec["label"], rec["line"], rec["race"], rec["alpha_rad"], rec["j"]) == (
        "DeepGrooveBallContact", "6210_inner", False, "inner", 0.0, None)
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


# ---- solvers: Hertz from SlipPY hertz_full on the bridge axes (solvers/hertz.py) --------------------

def test_hertz_of_the_6210_contacts(load_case):
    """BR1 section 5, Q 1000 N, tolerances of the printed digits: inner / outer b 0.1666 / 0.2025 mm, a 1.5307 /
    1.4877 mm, p0 1873 / 1585 MPa, delta 9.76 / 9.43 um; delta_i + delta_o = 19.20 um."""
    inner, outer = (hertz(c) for c in _6210_contacts(load_case))
    for hz, b, a, p0, delta in ((inner, 0.1666, 1.5307, 1873, 9.76), (outer, 0.2025, 1.4877, 1585, 9.43)):
        assert hz.b / MM == pytest.approx(b, abs=0.00005) and hz.a / MM == pytest.approx(a, abs=0.00005)
        assert hz.p0 / 1e6 == pytest.approx(p0, abs=0.5) and hz.delta / 1e-6 == pytest.approx(delta, abs=0.005)
        assert hz.psi == 0.0 and not hz.line and hz.valid and hz.b_over_a == pytest.approx(b / a, abs=1e-4)
    assert (inner.delta + outer.delta) / 1e-6 == pytest.approx(19.20, abs=0.005)


def test_hertz_agrees_with_the_independent_judge(load_case):
    """``references.contact.hertz.hertz_point`` (D-0021, no SlipPY, no bridge) on the 6210 contacts."""
    for c in _6210_contacts(load_case):
        hz, ref = hertz(c), hertz_point(c.body1.r, c.body2.r, (c.body1.E, c.body2.E), (c.body1.nu, c.body2.nu), c.load)
        assert hz.p0 == pytest.approx(ref["p0_Pa"], rel=1e-6) and hz.delta == pytest.approx(ref["delta_m"], rel=1e-6)
        assert (hz.b, hz.a) == (pytest.approx(ref["s_x_m"], rel=1e-6), pytest.approx(ref["s_y_m"], rel=1e-6))


def test_hertz_of_a_roller_line_contact(roller_10):
    """BR1 section 5: roller 10 x 10 on Dpw 70, Q 1000 N inner: b 0.0688 mm, p0 925.7 MPa; no approach."""
    hz = hertz(bearing_contacts(roller_10, load=1000.0, races=("inner",))[0])
    assert hz.b / MM == pytest.approx(0.0688, abs=0.00005) and hz.p0 / 1e6 == pytest.approx(925.7, abs=0.05)
    assert hz.line and hz.a == math.inf and math.isnan(hz.delta) and hz.b_over_a == 0.0


def test_hertz_of_the_fzg_h501_line_contact():
    """BR2 section 6: FZG H501 at C, rho 14.20546 / 21.30819 mm, E 206 000 MPa, nu 0.3, w 75 697.52 N/m:
    p0 565.67 MPa."""
    steel = dict(E=206e9, nu=0.3)
    c = _Line(body1=SweptCircleBody(r=(14.20546 * MM, math.inf), **steel),
              body2=SweptCircleBody(r=(21.30819 * MM, math.inf), **steel),
              load=75697.52)
    assert hertz(c).p0 / 1e6 == pytest.approx(565.67, abs=0.005)


def test_hertz_of_the_6210_inner_integrates_to_Q(load_case):
    """The Hertz pressure of the 6210 inner contact summed on a fine grid gives back Q = 1000 N."""
    (c,) = _6210_contacts(load_case, races=("inner",))
    hz = hertz(c)
    h = hz.b / 100
    X, Y = np.meshgrid(np.arange(-1.05 * hz.b, 1.05 * hz.b, h), np.arange(-1.05 * hz.a, 1.05 * hz.a, h),
                       indexing="ij")
    assert hz.pressure(X, Y).sum() * h**2 == pytest.approx(1000.0, rel=1e-3)


# ---- solvers: surfaces and the static BEM solve (solvers/surfaces.py, solvers/static.py) ---------------

def _bem(contact, tmp_path, *, use_frequency_domain=True, solve_kwargs=None, **around):
    """Hertz once, the grid around it, and the static BEM solve with that Hertz (SlipPY files to tmp_path)."""
    pytest.importorskip("pyfftw", reason="SlipPY plan_convolve needs pyfftw (R-128)")
    hz = hertz(contact)
    return solve(contact, Grid.around(contact, hz, **around), tmp_path, hertz_result=hz,
                 use_frequency_domain=use_frequency_domain, **(solve_kwargs or {}))


def _all_checks(res):
    return all((res.converged, res.load_ok, res.edge_clear, res.margin_ok, res.resolution_ok, res.within_reference,
                res.hertz_valid))


@pytest.fixture
def inner_6210(load_case):
    """Ball on the 6210 inner raceway, Q 1000 N (BR1 section 5)."""
    return _6210_contacts(load_case, races=("inner",))[0]


def _tori():
    """Two convex tori at 0.3 rad (psi = 0.113 rad), 100 N."""
    return _Point(body1=SweptCircleBody(r=_T1, **STEEL),
                  body2=SweptCircleBody(r=_T2, **STEEL), load=100.0, angle=_ANGLE, label="tori")


def test_bem_of_the_6210_inner_contact(inner_6210, tmp_path):
    """End to end on the default (frequency-domain) kernel: p0 within 1 %, b, a and the approach within 3 % of
    Hertz; every check passes; the record carries the commits. The approach holds at the default span_y 1.3
    only: the image error of the frequency kernel is not monotonic in the domain (R-046, R-509; span_y 2.0
    gives +4.7 %)."""
    res = _bem(inner_6210, tmp_path)
    assert abs(res.err_p0) <= 0.01
    for err in (res.err_b, res.err_a, res.err_delta):
        assert abs(err) <= 0.03
    assert _all_checks(res) and res.total_displacement_z.shape == res.grid.X.shape
    rec = res.record()
    assert all(isinstance(rec[f"commit_{k}"], str) for k in ("axisforge", "slippy", "design_studies"))


def test_bem_of_the_true_raceway_against_the_local_torus(load_case, tmp_path):
    """6210 inner at alpha 40 deg (given), Q 1000 N: BEM on the true raceway within 3 % of Hertz (p0 1 %), and within
    0.1 % of the BEM on the local torus of the same radii (p0, approach, load). The odd x^2 y term of the true
    raceway shifts and skews the pressure in y but changes p0 and the approach only to second order (C10b
    physics-reviewer; slippy-expert measured +0.184 / +0.176 % p0 error, +0.958 % approach for both)."""
    (true,) = _6210_contacts(load_case, alpha=math.radians(40.0), races=("inner",))
    torus = _Point(body1=SweptCircleBody(r=true.body1.r, **STEEL), body2=SweptCircleBody(r=true.body2.r, **STEEL),
                   load=1000.0)
    res, ref = _bem(true, tmp_path / "true"), _bem(torus, tmp_path / "torus")
    assert abs(res.err_p0) <= 0.01 and all(abs(e) <= 0.03 for e in (res.err_a, res.err_b, res.err_delta))
    assert res.converged and res.load_ok and res.edge_clear
    for key in ("p0", "delta", "load"):
        assert getattr(res, key) == pytest.approx(getattr(ref, key), rel=1e-3)
    rec = res.record()
    assert (rec["body1_class"], rec["body2_class"]) == ("Ball", "Raceway")


@pytest.mark.parametrize("make", ["inner_6210", "tori"])
def test_bem_of_a_point_contact_on_the_spatial_kernel(make, inner_6210, tmp_path):
    """The spatial (Boussinesq) kernel has no images, so the approach does not depend on the domain: p0 within
    1 %, b, a and the approach within 3 % of Hertz."""
    c = inner_6210 if make == "inner_6210" else _tori()
    res = _bem(c, tmp_path, use_frequency_domain=False)
    assert abs(res.err_p0) <= 0.01
    for err in (res.err_b, res.err_a, res.err_delta):
        assert abs(err) <= 0.03
    assert _all_checks(res) and not res.use_frequency_domain


def test_bem_of_a_roller_line_contact(roller_10, tmp_path):
    """Roller 10 x 10 on Dpw 70, Q 1000 N inner (periodic strip): p0 within 1 %, b within 3 % of Hertz."""
    res = _bem(bearing_contacts(roller_10, load=1000.0, races=("inner",))[0], tmp_path)
    assert abs(res.err_p0) <= 0.01 and abs(res.err_b) <= 0.03 and _all_checks(res)
    assert math.isnan(res.delta) and res.a == math.inf and res.period == res.grid.length_y_m
    assert res.load_applied == pytest.approx(1e5 * res.period, rel=1e-12)


def test_bem_of_the_fzg_h501_tooth_contact(h501_C, tmp_path):
    """FZG H501 at C (ToothContact from AxisForge, BR2 section 6): p0 within 1 %, b within 3 % of Hertz."""
    res = _bem(h501_C, tmp_path)
    assert abs(res.err_p0) <= 0.01 and abs(res.err_b) <= 0.03 and _all_checks(res)


def test_bem_of_a_turned_point_contact(tmp_path):
    """Two tori at 0.3 rad: the measured b along psi and a across it, p0 and the approach against Hertz (default
    kernel and span_y; see the 6210 test for the approach)."""
    res = _bem(_tori(), tmp_path)
    assert abs(res.err_p0) <= 0.01
    for err in (res.err_b, res.err_a, res.err_delta):
        assert abs(err) <= 0.03
    assert _all_checks(res)


@pytest.mark.parametrize("angle", [0.3, -1.2, 0.8])
def test_measure_on_the_hertz_field_along_psi(angle):
    """Closed form: the measurement on the exact Hertz pressure of two tori turned by ``angle`` (psi != 0,
    b/h >= 10) gives p0, b along psi and a across it within 1 %."""
    c = _Point(body1=SweptCircleBody(r=_T1, **STEEL), body2=SweptCircleBody(r=_T2, **STEEL), load=100.0, angle=angle)
    hz = hertz(c)
    grid = Grid.around(c, hz)
    assert hz.b / grid.h >= 10 and c.psi != 0.0
    p0, b, a, _ = _measure(hz.pressure(grid.X, grid.Y), grid.h, c.psi, False)
    assert p0 == pytest.approx(hz.p0, rel=0.01) and b == pytest.approx(hz.b, rel=0.01)
    assert a == pytest.approx(hz.a, rel=0.01)


def test_short_domain_is_flagged(inner_6210, tmp_path):
    """span_x = 2 (half the default): margin_ok is False."""
    assert not _bem(inner_6210, tmp_path, span_x=2.0).margin_ok


def test_narrow_domain_is_flagged(inner_6210, tmp_path):
    """span_y = 0.8: the contact is cut across rolling, pressure on the boundary: edge_clear is False."""
    assert not _bem(inner_6210, tmp_path, span_y=0.8).edge_clear


def test_tight_reference_tolerance_is_flagged(inner_6210, tmp_path):
    assert not _bem(inner_6210, tmp_path, solve_kwargs={"rtol_reference": 1e-6}).within_reference


def test_offset_of_the_surfaces_is_refused(inner_6210):
    grid = Grid.around(inner_6210, hertz(inner_6210))
    with pytest.raises(ValueError, match="off_set_x not accepted"):
        grid.step_args(inner_6210, off_set_x=1e-6)


@pytest.mark.parametrize("flag", [True, False])
def test_kernel_flag_reaches_both_materials(inner_6210, flag):
    s1, s2 = surface_pair(inner_6210, Grid.around(inner_6210, hertz(inner_6210)), name=f"kernel_{flag}",
                          use_frequency_domain=flag)
    assert s1.material.use_frequency is flag and s2.material.use_frequency is flag
    assert s1.profile.shape == s2.profile.shape and s1.grid_spacing == 2.0**-16


def test_line_contact_refuses_the_spatial_kernel(roller_10):
    (c,) = bearing_contacts(roller_10, load=1000.0, races=("inner",))
    with pytest.raises(ValueError, match="finite strip"):
        surface_pair(c, Grid.around(c, hertz(c)), name="line_spatial", use_frequency_domain=False)


# ---- grid angle: a grid turned by gamma from the bridge axes (mesh/grid.py, C9) -------------------------------

_R_CYL = 10 * MM                                   # [m] crossed cylinders, Johnson 1985 p. 86


def _crossed(angle):
    """Two equal steel cylinders R = 10 mm (axes along y), body 2 at ``angle``, 100 N: b/h = 14.8 on the default
    grid, size_ratio 0.019."""
    cyl = SweptCircleBody(r=(_R_CYL, math.inf), **STEEL)
    return _Point(body1=cyl, body2=cyl, load=100.0, angle=angle, label="crossed")


@pytest.mark.parametrize("use_frequency_domain", [True, False])
def test_crossed_cylinders_at_45_deg_on_the_bridge_and_principal_grids(use_frequency_domain, tmp_path):
    """Two equal cylinders crossed at 45 deg: kappa_max,min = (1 +- sqrt(2)/2)/R (Johnson 1985 p. 86: A, B =
    (1 -+ 1/sqrt 2)/2R, kappa = 2A, 2B); psi = 22.5 deg, the bisector of the cylinder axes, derived from Eqs.
    (4.1)-(4.3) with K = (1/R) [[3/2, 1/2], [1/2, 1/2]] (Mohr circle) and Fig. 4.1, p. 86.

    Solved on the bridge grid (gamma = 0, measured on interpolated cuts at psi) and on the principal grid
    (gamma = psi, measured on grid lines): p0, a, b and the load agree within 1 % between the two and each is
    within 3 % of Hertz; psi (from bridge x) is recorded the same, only grid_angle_rad differs. The 1 % is a
    discretisation bound at b/h 14.8 (b: 0.73 % frequency kernel, 0.44 % spatial, C9 review); with the frequency
    kernel it also holds the images, whose lattice turns with the grid (R-509). A wrong rotation shows as tens of %.
    The approach is global and insensitive to the edge: the same on both grids within 1e-3 on the spatial kernel."""
    c = _crossed(math.pi / 4)
    assert c.kappa_max == pytest.approx((1 + math.sqrt(2) / 2) / _R_CYL, rel=1e-12)
    assert c.kappa_min == pytest.approx((1 - math.sqrt(2) / 2) / _R_CYL, rel=1e-12)
    assert c.psi == pytest.approx(math.radians(22.5), rel=1e-12)
    bridge, principal = (_bem(c, tmp_path / str(k), use_frequency_domain=use_frequency_domain, angle=angle)
                         for k, angle in enumerate((0.0, "principal")))
    assert (bridge.grid.angle, principal.grid.angle) == (0.0, c.psi)
    assert bridge.hertz.b / bridge.grid.h >= 10 and principal.grid.h == bridge.grid.h
    for key in ("p0", "a", "b", "load"):
        assert getattr(principal, key) == pytest.approx(getattr(bridge, key), rel=0.01), key
    if not use_frequency_domain:
        assert principal.delta == pytest.approx(bridge.delta, rel=1e-3)
    for res in (bridge, principal):
        assert all(abs(err) <= 0.03 for err in (res.err_p0, res.err_a, res.err_b)) and _all_checks(res)
    rec_b, rec_p = bridge.record(), principal.record()
    assert rec_b["psi_rad"] == rec_p["psi_rad"] == c.psi
    assert (rec_b["grid_angle_rad"], rec_p["grid_angle_rad"]) == (0.0, c.psi)


def test_crossed_cylinders_at_minus_45_deg(tmp_path):
    """Body 2 at -45 deg gives Kxy < 0, psi = -22.5 deg. The sign of psi is the bridge convention (positive from x
    toward y, as Contact.angle); Johnson Eq. (4.5) holds only cos 2 alpha and does not fix it. The solve on the
    principal grid at -22.5 deg checks that Contact.psi and Body.profile agree: a mismatch would put the cuts 45 deg
    off the ellipse."""
    c = _crossed(-math.pi / 4)
    assert c.psi == pytest.approx(math.radians(-22.5), rel=1e-12)
    res = _bem(c, tmp_path, use_frequency_domain=False, angle="principal")
    assert res.grid.angle == c.psi
    assert all(abs(err) <= 0.03 for err in (res.err_p0, res.err_a, res.err_b)) and _all_checks(res)


def test_principal_grid_of_the_6210_inner_contact_is_the_bridge_grid(inner_6210, tmp_path):
    """psi = 0 (bearing): angle="principal" gives gamma = 0, the same grid and the same solution (rel 1e-12: two
    SlipPY solves, pyfftw may plan them differently)."""
    bridge, principal = (_bem(inner_6210, tmp_path / str(k), angle=angle)
                         for k, angle in enumerate((0.0, "principal")))
    assert principal.grid.angle == 0.0
    np.testing.assert_array_equal(principal.grid.X, bridge.grid.X)
    np.testing.assert_allclose(principal.p, bridge.p, rtol=1e-12, atol=0.0)
    for key in ("p0", "a", "b", "delta"):
        assert getattr(principal, key) == pytest.approx(getattr(bridge, key), rel=1e-12, abs=0.0), key


@pytest.mark.parametrize("gamma", [0.5, -1.2, math.pi / 2])
def test_profiles_and_hertz_on_a_turned_grid(gamma):
    """A node at grid coordinates (Xg, Yg) lies at bridge (Xg cos g - Yg sin g, Xg sin g + Yg cos g): there each
    body's profile is Body.profile at its own angle on the bridge axes, and the Hertz pressure is the bridge one
    (rel 1e-12)."""
    c = _tori()
    hz = hertz(c)
    grid = Grid.around(c, hz, angle=gamma)
    cg, sg = math.cos(gamma), math.sin(gamma)
    X_b, Y_b = grid.X * cg - grid.Y * sg, grid.X * sg + grid.Y * cg
    z1, z2 = c.profiles(grid)
    np.testing.assert_allclose(z1, c.body1.profile(X_b, Y_b), rtol=1e-12, atol=1e-24)
    np.testing.assert_allclose(z2, c.body2.profile(X_b, Y_b, c.angle), rtol=1e-12, atol=1e-24)
    np.testing.assert_allclose(hz.pressure(grid.X, grid.Y, grid.angle), hz.pressure(X_b, Y_b), rtol=1e-12, atol=1e-6)


def test_principal_grid_half_extents_are_the_semi_axes():
    assert _hertz(1.0 * MM, 0.2 * MM, psi=0.4).half_extents(0.4) == (0.2 * MM, 1.0 * MM)


def test_line_contact_on_a_turned_grid_is_refused(roller_10):
    """The strip of a line contact is periodic along grid y: its line must lie there (gamma = psi = 0)."""
    (c,) = bearing_contacts(roller_10, load=1000.0, races=("inner",))
    hz = hertz(c)
    assert Grid.around(c, hz, angle="principal").angle == 0.0
    with pytest.raises(ValueError, match="line along grid y"):
        Grid.around(c, hz, angle=0.1).step_args(c)


@pytest.mark.parametrize("angle", [-math.pi / 2, 2.0])
def test_grid_angle_is_checked(angle):
    g = Grid.around(_tori(), hertz(_tori()))
    with pytest.raises(ValueError, match="angle must be in"):
        Grid(X=g.X, Y=g.Y, h=g.h, angle=angle)


def test_grid_around_refuses_an_unknown_angle_name():
    with pytest.raises(ValueError, match="'principal'"):
        Grid.around(_tori(), hertz(_tori()), angle="major")


# ---- gears: AxisForge cylindrical pair -> ToothContact (core/machine_elements/gears.py) ------------------

_TEST_STEEL = "test_steel_206GPa"     # an AxisForge material registered for the gear tests


def _register_test_steel():
    """Register ``_TEST_STEEL`` (E 206 000 MPa, nu 0.3) once per process (AxisForge has no unregister)."""
    from axisforge.core.materials import IsotropicElastic, Material, available_materials, register
    if _TEST_STEEL not in available_materials():
        register(Material(material_id=_TEST_STEEL, density=7.85e3,
                          elastic=IsotropicElastic(E=206000.0, poisson_ratio=0.3)))


def _gear_pair(material_id=_TEST_STEEL, z2=40):
    """A 20/z2 spur pair, module 2 mm, face width 20 mm (z2 < 0: an internal ring); ``_TEST_STEEL`` is
    registered, other ids are not."""
    _register_test_steel()
    from axisforge.core.machine_elements.gears import CylindricalFamily, CylindricalGearData, Gear, GearPair
    family = CylindricalFamily()
    z1 = 20 if z2 > 0 else 24
    return GearPair.assemble(
        family, gear1=Gear.assemble(family, CylindricalGearData(2.0, z1, b=20.0, material_id=material_id)),
        gear2=Gear.assemble(family, CylindricalGearData(2.0, z2, b=20.0, material_id=material_id)),
        label="internal" if z2 < 0 else "")


@pytest.fixture
def fzg_h501(load_case):
    """FZG H501 pair of ``validation/gears/fzg_geometry_ph``, T1 60.75 N m: (case, pair, F_bn [N])."""
    cc = load_case("fzg_geometry_ph")
    pair = cc.build_pair("H501")
    return cc, pair, cc.FAMILY.mesh_forces(pair, 60.75, rot1=+1).F_bn_N


@pytest.fixture
def h501_C(fzg_h501):
    """The H501 tooth contact at the pitch point C, gear 1 driving, E 206 000 MPa, nu 0.3 (the case's)."""
    cc, pair, F_bn_N = fzg_h501
    (c,) = gear_contacts(pair, {"C": path_diameters(pair, "gear1")["C"]}, F_bn_N, driver="gear1",
                         E=cc.GEAR_E_MPA, nu=cc.GEAR_NU)
    return c


def test_fzg_h501_at_C_against_the_hand_check(fzg_h501, h501_C):
    """BR2 section 6, independent hand check (ISO 21771 geometry, ISO 6336-2 forces, Johnson 1985 Eq. (4.43),
    (4.45)): F_bn 1847.109 N, rho1_n 14.20546 mm, rho2_n 21.30819 mm, w = F_bn / l_sum = 75 697.52 N/m,
    3 tooth pairs, share 0.9717558, Hertz p0 565.67 MPa (the local equivalent cylinders under AxisForge's equal
    line load, not the ISO 6336-2 sigma_H)."""
    cc, pair, F_bn_N = fzg_h501
    c = h501_C
    assert F_bn_N == pytest.approx(1847.109, abs=0.0005)
    assert isinstance(c, ToothContact) and c.line and (c.label, c.point, c.driver) == ("H501_C", "C", "gear1")
    assert c.body1.r == pytest.approx((14.20546 * MM, math.inf), rel=1e-6)
    assert c.body2.r == pytest.approx((21.30819 * MM, math.inf), rel=1e-6)
    assert (c.body1.E, c.body2.E, c.body1.nu) == (206000.0e6, 206000.0e6, 0.3)
    assert c.load == pytest.approx(75697.52, rel=1e-6)                          # [N/m]
    assert (c.n_pairs, c.share) == (3, pytest.approx(0.9717558, rel=1e-6)) and type(c.n_pairs) is int
    assert c.l_sum_mm == pytest.approx(24.40118, rel=1e-6) and c.load == pytest.approx(F_bn_N / c.l_sum_mm / MM)
    assert c.g_y_mm == pytest.approx(0.0, abs=1e-9) and c.dy1_mm == pytest.approx(pair.dl1, rel=1e-12)
    given = f"given:{cc.STEEL_ID}"                 # an id that AxisForge does not have, kept visible
    assert (c.material1, c.material2, c.z2) == (given, given, pair.gear2.z)
    assert hertz(c).p0 / 1e6 == pytest.approx(565.67, abs=0.005)


def test_path_diameters_of_both_drivers(fzg_h501):
    """Gear 2 driving swaps A <-> E and B <-> D, C stays; with gear 1 driving A is the active root 2 rA1
    (d_Nf1), E the tip da1, C the operating pitch diameter dl1."""
    _, pair, _ = fzg_h501
    d1, d2 = path_diameters(pair, "gear1"), path_diameters(pair, "gear2")
    assert (d2["A"], d2["B"], d2["C"], d2["D"], d2["E"]) == (d1["E"], d1["D"], d1["C"], d1["B"], d1["A"])
    assert d1["A"] == pytest.approx(2 * pair.rA1, rel=1e-12) and d1["E"] == pytest.approx(pair.gear1.da, rel=1e-12)
    assert d1["C"] == pytest.approx(pair.dl1, rel=1e-12)
    assert d1["A"] < d1["C"] < d1["E"]           # B and D need not bracket C (internal 24/-60)


@pytest.mark.parametrize("driver", ["gear1", "gear2"])
@pytest.mark.parametrize("which", ["H501", "internal"])
def test_gear_contacts_agree_with_contact_inputs(fzg_h501, driver, which):
    """At each step of AxisForge ``contact_inputs`` (9 steps A..E for the driver), dy1 = 2 sqrt((T1C + g_y)^2
    + rb1^2) gives the same rho1_n, rho2_n and w through ``pair_values_at`` and ``load_sharing``: no hidden
    'pinion = gear 1' assumption in AxisForge. FZG H501 (helical) and an internal spur pair 24/-60."""
    cc, pair, F_bn_N = fzg_h501
    material = dict(E=cc.GEAR_E_MPA, nu=cc.GEAR_NU)
    if which == "internal":
        pair, material = _gear_pair(z2=-60), {}
    inputs = contact_inputs(pair, driver, 1500.0, F_bn_N, n_points=9)
    dy1 = 2.0 * np.sqrt((pair.T1C + inputs["g_y_mm"]) ** 2 + pair.gear1.rb ** 2)
    cs = gear_contacts(pair, list(dy1), F_bn_N, driver=driver, **material)
    np.testing.assert_allclose([c.body1.r[0] / MM for c in cs], inputs["rho1_n_mm"], rtol=1e-9)
    np.testing.assert_allclose([c.body2.r[0] / MM for c in cs], inputs["rho2_n_mm"], rtol=1e-9)
    np.testing.assert_allclose([c.load * MM for c in cs], inputs["w_N_per_mm"], rtol=1e-9)
    np.testing.assert_allclose([c.g_y_mm for c in cs], inputs["g_y_mm"], rtol=1e-9, atol=1e-12)
    assert [c.n_pairs for c in cs] == [int(n) for n in inputs["n_pairs"]]


def test_internal_pair_contacts():
    """Internal spur pair z 24/-60, mn 2 mm: the ring flank is concave (rho2 < 0), the relative curvature stays
    positive and Hertz solves along the whole path."""
    pair = _gear_pair(z2=-60)
    cs = gear_contacts(pair, path_diameters(pair, "gear1"), 1000.0, driver="gear1")
    for c in cs:
        assert c.body2.r[0] < 0.0 < c.kappa_max and c.z2 == -60 and math.isfinite(hertz(c).p0)
    # closed form at C (spur, x = 0, so d_w = m z): rho_C = (d_w / 2) sin(alpha_wt), alpha_wt = 20 deg
    c = cs[2]
    assert c.body1.r[0] / MM == pytest.approx(24.0 * math.sin(math.radians(20.0)), rel=1e-9)     # 8.2085 mm
    assert c.body2.r[0] / MM == pytest.approx(-60.0 * math.sin(math.radians(20.0)), rel=1e-9)    # -20.521 mm
    assert [c.label for c in cs] == ["internal_A", "internal_B", "internal_C", "internal_D", "internal_E"]


def test_gear_material_source_and_default_labels():
    """E, nu from the AxisForge material (the contact names the material_id); a scalar or sequence dy1 gets the
    label ``dy1=<value>``."""
    pair = _gear_pair()
    d = path_diameters(pair, "gear1")["C"]
    (c,) = gear_contacts(pair, d, 1000.0, driver="gear1")
    assert (c.material1, c.material2, c.body1.E) == (_TEST_STEEL, _TEST_STEEL, 206000.0e6)
    assert c.point == f"dy1={d:.6g}" and c.label == f"gear_dy1={d:.6g}"
    for material_id, recorded in (("", "given"), ("typo_steel", "given:typo_steel")):
        (c,) = gear_contacts(_gear_pair(material_id), [d], 1000.0, driver="gear1", E=(206000.0, 210000.0),
                             nu=(0.3, 0.29))
        assert (c.material1, c.material2, c.body2.E, c.body2.nu) == (recorded, recorded, 210000.0e6, 0.29)


@pytest.mark.parametrize("change, match", [
    ("off_path_low", "off the path"),
    ("off_path_high", "off the path"),
    ("driver", "driver must be"),
    ("colon", "no ':'"),
    ("F_bn", "F_bn_N must be positive"),
    ("material_given", "given only for a gear without one"),
    ("material_missing", "has no AxisForge material"),
    ("material_format", "must be"),
])
def test_gear_contacts_refuses(change, match):
    pair = _gear_pair("" if change in ("material_missing", "material_format") else _TEST_STEEL)
    d = path_diameters(pair, "gear1")
    kwargs = dict(dy1=d["C"], F_bn_N=1000.0, driver="gear1")
    kwargs.update({"off_path_low": dict(dy1=d["A"] * (1 - 1e-6)), "off_path_high": dict(dy1=d["E"] * (1 + 1e-6)),
                   "driver": dict(driver="pinion"), "colon": dict(dy1={"t:1": d["C"]}),
                   "F_bn": dict(F_bn_N=0.0), "material_given": dict(E=(206000.0, None), nu=(0.3, None)),
                   "material_format": dict(E=206000.0, nu=(0.3, 0.3))}.get(change, {}))
    with pytest.raises(ValueError, match=match):
        gear_contacts(pair, kwargs.pop("dy1"), kwargs.pop("F_bn_N"), **kwargs)


def test_pair_through_a_base_tangent_point_is_refused():
    """A pair assembled with check=False whose path passes T2 (external 10/10, x +0.6/-0.6: T2E < 0): there
    pair_values_at would return rho2 with the wrong sign, so the pair is refused (SB-20)."""
    from axisforge.core.machine_elements.gears import CylindricalGearData, Gear, GearPair
    family = CylindricalFamily()
    gears = [Gear.assemble(family, CylindricalGearData(2.0, 10, x=x, b=20.0, material_id=""), check=False)
             for x in (0.6, -0.6)]
    pair = GearPair.assemble(family, gear1=gears[0], gear2=gears[1], check=False)
    assert pair.T2E < 0
    with pytest.raises(ValueError, match="passes a base-tangent point"):
        gear_contacts(pair, 2.0 * pair.rA1, 1000.0, driver="gear1", E=(206000.0, 206000.0), nu=(0.3, 0.3))


def test_tooth_contact_checks():
    """rho1 > 0, sign of rho2 = sign of z2, straight along y: a hand-built ToothContact."""
    flank = dict(E=206e9, nu=0.3)
    fields = dict(load=1e5, point="C", dy1_mm=40.0, dy2_mm=80.0, g_y_mm=0.0, share=1.0, l_mm=20.0, l_sum_mm=20.0,
                  n_pairs=1,
                  driver="gear1", material1="given", material2="given")
    convex, concave = SweptCircleBody(r=(10 * MM, math.inf), **flank), SweptCircleBody(r=(-60 * MM, math.inf), **flank)
    with pytest.raises(ValueError, match="has not the sign of z2"):
        ToothContact(body1=convex, body2=convex, z2=-60, **fields)
    with pytest.raises(ValueError, match="rho1_n must be > 0"):
        ToothContact(body1=concave, body2=convex, z2=40, **fields)
    with pytest.raises(ValueError, match="straight along the face width"):
        ToothContact(body1=SweptCircleBody(r=10 * MM, **flank), body2=convex, z2=40, **fields)
    assert ToothContact(body1=convex, body2=concave, z2=-60, **fields).kappa_max > 0


# ---- FZG C40 on the bridge (validation/gears/fzg_geometry_ph) -------------------------------------

@pytest.fixture
def fzg_c40(load_case):
    """The FZG C40 pair of the case; E 206 000 MPa and nu 0.3 are the case's assumption (``GEAR_E_MPA``,
    ``GEAR_NU``: ISO 6336-2:2006 Table 1 generic steel, R-118 open), given to the bridge."""
    cc = load_case("fzg_geometry_ph")
    return cc, cc.build_pair("C40")


def _fzg_p_H_MPa(cc, pair, stage, points):
    """Hertz p_H [MPa] of C40 at ``points`` for FZG ``stage``: wheel torque (R-124), the case's driver. "B+" is
    the one-sided point DELTA_S_MM past B toward D (single contact for a spur pair)."""
    z1, z2 = pair.gear1.z, pair.gear2.z
    F_bn_N = cc.FAMILY.mesh_forces(pair, cc.WHEEL_TORQUE_NM[stage] * z1 / z2, rot1=+1).F_bn_N
    dy1 = path_diameters(pair, cc.DRIVER)
    g = CylindricalFamily.path_points(pair, cc.DRIVER)
    g_b = g["B"] + math.copysign(cc.DELTA_S_MM, g["E"] - g["A"])            # past B in the direction of travel
    dy1 = {**dy1, "B+": 2.0 * math.hypot(pair.T1C + g_b, pair.gear1.rb)}
    dy1 = {k: dy1[k] for k in points}
    contacts = gear_contacts(pair, dy1, F_bn_N, driver=cc.DRIVER, E=cc.GEAR_E_MPA, nu=cc.GEAR_NU)
    return [hertz(c).p0 / 1e6 for c in contacts]                              # Pa -> MPa


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


@pytest.mark.xfail(strict=False, reason="Q-09 / R-117 not ruled")
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


# ---- package: exports, sub-model data and the layer order ---------------------------------------------

_LEVELS = ("core", "mesh", "results", "solvers", "outputs")   # import order (D-0034 item 1; outputs later)


def _imports(path: Path, module: str) -> set[str]:
    """Absolute names of every module imported by the file ``path`` (module name ``module``), including
    the imports under ``if TYPE_CHECKING:``."""
    import ast
    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = module.split(".") if path.name == "__init__.py" else module.split(".")[:-1]
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = package[:len(package) - node.level + 1] if node.level else []
            target = ".".join(base + ([node.module] if node.module else []))
            names.add(target)
            if node.module is None:                     # "from . import x": x is a module
                names.update(f"{target}.{alias.name}" for alias in node.names)
    return names


def test_layer_order_of_the_contact_package():
    """core imports nothing of mesh/results/solvers and no SlipPY; mesh no results/solvers/SlipPY; results no
    solvers/SlipPY. Imports under TYPE_CHECKING are held to the same rule (they name earlier levels only)."""
    import slippy_bridge.contact as pkg
    root = Path(pkg.__file__).parent
    files = [p for p in sorted(root.rglob("*.py")) if len(p.relative_to(root).parts) > 1]
    checked = 0
    for path in files:                                   # contact/__init__.py (the exports) excluded
        rel = path.relative_to(root)
        level = rel.parts[0]
        module = ".".join(("slippy_bridge", "contact", *rel.with_suffix("").parts))
        later = _LEVELS[_LEVELS.index(level) + 1:]
        forbidden = [f"slippy_bridge.contact.{name}" for name in later]
        for name in _imports(path, module):
            assert not any(name == f or name.startswith(f + ".") for f in forbidden), f"{module} imports {name}"
            if level != "solvers":
                assert name.split(".")[0] != "slippy", f"{module} imports SlipPY ({name})"
        checked += 1
    assert checked == len(files) >= 14


def test_public_names_import_from_the_package():
    import slippy_bridge.contact as pkg
    for name in ("Body", "SweptCircleBody", "Contact", "Kinematics", "BearingContact", "BallRacewayContact",
                 "RollerRacewayContact",
                 "bearing_contacts", "GearContact", "ToothContact", "gear_contacts", "path_diameters", "Grid",
                 "SPAN_X_0", "HertzResult", "StaticResult", "hertz", "surface_pair", "solve"):
        assert name in pkg.__all__ and getattr(pkg, name) is not None
    assert pkg.solve is solve and pkg.ToothContact is ToothContact


def test_kinematics_data():
    """ue is the MEAN entrainment speed (u1 + u2)/2, vg = u1 - u2; data only, finite real numbers; with u1 and
    u2 given, the sum given as ue is refused (R-075)."""
    from slippy_bridge.contact import Kinematics
    k = Kinematics(ue=(3.0 + 1.0) / 2, vg=3.0 - 1.0, u1=3.0, u2=1.0)
    assert (k.ue, k.vg, k.u1, k.u2) == (2.0, 2.0, 3.0, 1.0)
    assert Kinematics(ue=1.0, vg=0.0).u1 is None
    assert Kinematics(ue=np.float32(1.5), vg=np.float64(0.0)).ue == 1.5
    for bad in (dict(ue=math.nan, vg=0.0), dict(ue=1.0, vg=math.inf), dict(ue=1.0, vg=0.0, u1="1"),
                dict(ue=True, vg=0.0)):
        with pytest.raises(ValueError, match="finite real number"):
            Kinematics(**bad)
    with pytest.raises(ValueError, match="ue is the MEAN"):
        Kinematics(ue=3.0 + 1.0, vg=2.0, u1=3.0, u2=1.0)              # the sum given as ue
    with pytest.raises(ValueError, match="vg = u1 - u2"):
        Kinematics(ue=2.0, vg=-2.0, u1=3.0, u2=1.0)


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
