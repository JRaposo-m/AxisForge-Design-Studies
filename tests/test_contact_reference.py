"""references.contact.hertz: the judge alone (no AxisForge, no SlipPY)."""
import math

import pytest
from scipy.integrate import dblquad

from references.contact.hertz import hertz_point

# 12.7 mm ball in a groove of radius 0.52 Dw (concave across rolling), rolling radii of a raceway
# on a 70 mm pitch diameter at zero contact angle: (70 -/+ 12.7) / 2 mm. Radii [m], convex > 0.
R_BALL = 6.35e-3
INNER = (28.65e-3, -0.52 * 12.7e-3)       # convex along rolling
OUTER = (-41.35e-3, -0.52 * 12.7e-3)      # concave along rolling
STEEL = dict(E=210e9, nu=0.3)
# two materials: pairing nu of one body with E of the other, or dropping a body, would show (a swap
# of whole bodies cannot: E* is symmetric)
MIXED = dict(E=(210e9, 70e9), nu=(0.3, 0.33))


def _e_star(E, nu):
    """Contact modulus [Pa], ``1/E* = (1 - nu1^2)/E1 + (1 - nu2^2)/E2``; scalars apply to both bodies."""
    (E1, E2), (nu1, nu2) = (E if isinstance(E, tuple) else (E, E)), (nu if isinstance(nu, tuple) else (nu, nu))
    return 1.0 / ((1.0 - nu1**2) / E1 + (1.0 - nu2**2) / E2)


@pytest.mark.parametrize("r2, material, load", [(INNER, STEEL, 2436.0), (OUTER, MIXED, 800.0)])
def test_hertz_pressure_closes_the_gap(r2, material, load):
    """Boussinesq self-check: the judge's pressure deforms the surfaces into contact.

    A point load P on an elastic half-space lowers the surface by ``(1 - nu^2) P / (pi E r)`` at
    the distance r (Boussinesq; Johnson, Contact Mechanics (1985), ch. 3). Summed over both bodies
    and over the pressure, ``u(x, y) = 1/(pi E*) * integral p / r dA``. Inside the contact the two
    surfaces coincide, so ``u = delta - k_x x^2/2 - k_y y^2/2`` (approach minus the initial gap). The
    curvatures and ``E*`` are built here from the raw inputs, so the check does not depend on how
    the judge computes them.

    The integral is taken in polar coordinates centred on the evaluation point, where
    ``dA / r = dr dtheta`` and the 1/r singularity cancels. Along a ray the pressure is
    ``p0 sqrt(A (r_max - r)(r - r_min))``; with ``r = r_max (1 - t^2)`` the square root at the edge
    of the contact becomes smooth in t.
    """
    ref = hertz_point(R_BALL, r2, load=load, **material)
    p0, s_x, s_y, delta = ref["p0_Pa"], ref["s_x_m"], ref["s_y_m"], ref["delta_m"]
    k_x, k_y = 1.0 / R_BALL + 1.0 / r2[0], 1.0 / R_BALL + 1.0 / r2[1]      # relative curvatures [1/m]

    def displacement(x0, y0):
        C = 1.0 - (x0 / s_x) ** 2 - (y0 / s_y) ** 2

        def integrand(t, theta):
            c, s = math.cos(theta), math.sin(theta)
            A = (c / s_x) ** 2 + (s / s_y) ** 2
            B = x0 * c / s_x**2 + y0 * s / s_y**2
            root = math.sqrt(B * B + A * C)
            r_max, r_min = (-B + root) / A, (-B - root) / A     # the ray meets the ellipse at r_max
            r = r_max * (1.0 - t * t)
            return p0 * math.sqrt(A * r_max * (r - r_min)) * 2.0 * r_max * t * t   # p dr, dr = 2 r_max t dt

        return dblquad(integrand, 0.0, 2.0 * math.pi, 0.0, 1.0, epsabs=0.0, epsrel=1e-11)[0] / (
            math.pi * _e_star(**material))

    # centre, one point on each axis, one off both axes
    for fx, fy in [(0.0, 0.0), (0.5, 0.0), (0.0, 0.6), (0.4, 0.5)]:
        x0, y0 = fx * s_x, fy * s_y
        assert displacement(x0, y0) == pytest.approx(delta - k_x * x0**2 / 2 - k_y * y0**2 / 2, rel=1e-8)


@pytest.mark.parametrize("r2, material", [(INNER, STEEL), (OUTER, MIXED), (math.inf, MIXED)])
def test_pressure_carries_the_load(r2, material):
    """Load tie: the ellipsoidal pressure integrates to ``(2/3) pi s_x s_y p0`` (Johnson, Contact
    Mechanics (1985), ch. 4)."""
    ref = hertz_point(R_BALL, r2, load=1234.5, **material)
    assert 2.0 / 3.0 * math.pi * ref["s_x_m"] * ref["s_y_m"] * ref["p0_Pa"] == pytest.approx(1234.5, rel=1e-12)


def test_circular_limit_matches_johnson():
    """Sphere of radius R on a flat (Johnson, Contact Mechanics (1985), ch. 4):
    ``a = (3 Q R / (4 E*))**(1/3)``, ``p0 = 3 Q / (2 pi a^2)``, ``delta = a^2 / R``."""
    Q = 500.0                                                 # [N]
    ref = hertz_point(R_BALL, math.inf, load=Q, **MIXED)
    a = (3.0 * Q * R_BALL / (4.0 * _e_star(**MIXED))) ** (1.0 / 3.0)
    assert (ref["s_x_m"], ref["s_y_m"]) == pytest.approx((a, a), rel=1e-9)
    assert ref["p0_Pa"] == pytest.approx(3.0 * Q / (2.0 * math.pi * a**2), rel=1e-9)
    assert ref["delta_m"] == pytest.approx(a**2 / R_BALL, rel=1e-9)


def test_long_axis_lies_across_rolling_in_a_groove_and_follows_the_axes():
    ref = hertz_point(R_BALL, INNER, load=100.0, **STEEL)
    assert ref["s_y_m"] > ref["s_x_m"]                        # the groove conforms to the ball across rolling
    swapped = hertz_point(R_BALL, INNER[::-1], load=100.0, **STEEL)
    assert (swapped["s_x_m"], swapped["s_y_m"]) == (ref["s_y_m"], ref["s_x_m"])


@pytest.mark.parametrize("r1, r2", [((R_BALL, math.inf), math.inf),          # cylinder on flat: line contact
                                    (R_BALL, (28.65e-3, -6.0e-3))])          # groove tighter than the ball
def test_not_a_point_contact_raises(r1, r2):
    with pytest.raises(ValueError):
        hertz_point(r1, r2, load=100.0, **STEEL)
