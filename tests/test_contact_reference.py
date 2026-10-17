"""references.contact: the judges alone (no AxisForge; SlipPY only in the Dowson-Higginson unit check)."""
import csv
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import dblquad

from references.contact.ehl import canzi_moes, e_prime, groups, moes
from references.contact.hertz import (hertz_line, hertz_line_stress, hertz_point, hertz_point_axis_stress,
                                      hertz_pressure)

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


# roller of 10 mm on the inner raceway of a 70 mm pitch diameter: (70 - 10) / 2 mm. Radii [m].
ROLLER = (5e-3, math.inf)
ROLLER_RACEWAY = (30e-3, math.inf)


@pytest.mark.parametrize("material", [STEEL, MIXED])
def test_line_contact_matches_johnson(material):
    """Line contact (Johnson, Contact Mechanics (1985), Eqs (4.43)-(4.45)):
    ``a = (4 P R / (pi E*))**(1/2)``, ``p0 = 2 P / (pi a) = (P E* / (pi R))**(1/2)``, P per unit length."""
    P = 5e4                                                   # [N/m]
    R = 1.0 / (1.0 / ROLLER[0] + 1.0 / ROLLER_RACEWAY[0])     # relative radius along rolling [m]
    e_star = _e_star(**material)
    ref = hertz_line(ROLLER, ROLLER_RACEWAY, w=P, **material)
    a = math.sqrt(4.0 * P * R / (math.pi * e_star))
    assert ref["s_x_m"] == pytest.approx(a, rel=1e-12)
    assert ref["p0_Pa"] == pytest.approx(2.0 * P / (math.pi * a), rel=1e-12)
    assert ref["p0_Pa"] == pytest.approx(math.sqrt(P * e_star / (math.pi * R)), rel=1e-12)


@pytest.mark.parametrize("r1, r2, w", [(R_BALL, math.inf, 5e4),             # ball on flat: point contact
                                       (ROLLER, (30e-3, 0.1), 5e4),         # raceway crowned along y
                                       (ROLLER, (-4e-3, math.inf), 5e4),    # bore tighter than the roller
                                       (ROLLER, ROLLER_RACEWAY, 0.0)])      # no load
def test_not_a_line_contact_raises(r1, r2, w):
    with pytest.raises(ValueError):
        hertz_line(r1, r2, w=w, **STEEL)


# ---- references.contact.ehl ------------------------------------------------------------------

def test_e_prime_definition():
    """E' = 2 / [(1 - nu1^2)/E1 + (1 - nu2^2)/E2] = 2 E*; for two equal steels E' = E / (1 - nu^2)."""
    out = e_prime(210e9, 0.3, 210e9, 0.3)
    assert out["E_prime_Pa"] == pytest.approx(210e9 / (1.0 - 0.3**2), rel=1e-15)
    assert out["E_prime_Pa"] == pytest.approx(2.0 * out["E_star_Pa"], rel=1e-15)
    assert e_prime(**{"E1_Pa": 210e9, "nu1": 0.3, "E2_Pa": 70e9, "nu2": 0.33})["E_star_Pa"] == pytest.approx(
        _e_star(**MIXED), rel=1e-15)                                         # E* is the Hertz modulus


@pytest.mark.parametrize("loads", [{}, {"load_N": 500.0, "w_N_per_m": 1e5}])
def test_groups_needs_one_load(loads):
    with pytest.raises(ValueError, match="exactly one"):
        groups(u_mean_m_per_s=2.0, eta0_Pa_s=0.1, E_prime_Pa=2.3e11, R_x_m=6.35e-3, alpha_per_Pa=2e-8, **loads)


@pytest.mark.parametrize("bad", [{"E1_Pa": 0.0}, {"nu2": 0.0}, {"nu1": 0.5}])
def test_e_prime_bad_input_raises(bad):
    with pytest.raises(ValueError):
        e_prime(**{"E1_Pa": 210e9, "nu1": 0.3, "E2_Pa": 210e9, "nu2": 0.3, **bad})


@pytest.mark.parametrize("key", ["load_N", "u_mean_m_per_s", "R_x_m", "eta0_Pa_s", "alpha_per_Pa"])
def test_groups_refuses_non_positive_input(key):
    kwargs = dict(u_mean_m_per_s=2.0, eta0_Pa_s=0.1, E_prime_Pa=2.3e11, R_x_m=6.35e-3, alpha_per_Pa=2e-8,
                  load_N=500.0)
    with pytest.raises(ValueError, match=key):
        groups(**{**kwargs, key: 0.0})


def test_moes_refuses_p_h_in_MPa():
    with pytest.raises(ValueError, match="in MPa"):
        moes(M=20.0, L=10.0, line=True, compressible=True, p_h_Pa=500.0)    # 500 MPa given as 500


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="scaffold OPEN R-114 GL-13 GL-12")
@pytest.mark.parametrize("line, alpha_bar, M_canzi, H_c_printed", [
    (True, 20.0, 17.97, 4.77),      # Canzi2010 Table 2 "Hc Moes formula": line_contact CSV L166 (L160, L163)
    (False, 20.0, 100.0, 4.05),     # Canzi2010 Table 3 "Hc Moes formula": canzi_circular_extra CSV L10 (L3, L6)
], ids=["line", "circular"])
def test_moes_canzi_formula(line, alpha_bar, M_canzi, H_c_printed):
    """Moes' film formula, compressible-corrected (Canzi2010 Eq. 4 line, Eq. 15 circular, Eq. 11), as printed.

    The point is given in Canzi's (alpha_bar, M) and mapped to Moes (M, L) by Eq. 3 / Eq. 14. The
    compressibility correction needs a dimensional p_h that Canzi does not print (R-114, GL-13).
    """
    p_h_assumed_Pa = None           # OPEN R-114: the assumed p_h is written before the body is filled
    point = canzi_moes(alpha_bar=alpha_bar, M_canzi=M_canzi, line=line)
    H = moes(M=point["M_moes"], L=point["L_moes"], line=line, compressible=True, p_h_Pa=p_h_assumed_Pa)
    assert H["H_c"] == pytest.approx(H_c_printed, abs=0.005)


def test_dowson_higginson_asymptote_1p34():
    """SlipPY's Dowson-Higginson law, rho/rho0 = (5.9e8 + 1.34 p)/(5.9e8 + p) with p in Pa, tends to 1.34.

    The constant 5.9e8 is in Pa inside ``nd_dowson_higginson`` (lubricant_models.py:216), so the Hertz
    pressure that scales P = p/p_h must be given in Pa: a value in MPa leaves the oil almost
    incompressible (R-077; ruled note 2026-10-16).
    """
    lm = pytest.importorskip("slippy.contact.lubricant_models")
    rho = lm.nd_dowson_higginson(1.0e9)                                # p_h = 1 GPa, in Pa
    assert float(rho(np.array([0.0]))[0]) == pytest.approx(1.0, rel=1e-12)            # ambient pressure
    assert float(rho(np.array([1.0e6]))[0]) == pytest.approx(1.34, rel=1e-5)          # p = 1e15 Pa: the asymptote
    assert float(rho(np.array([1.0]))[0]) == pytest.approx((5.9e8 + 1.34e9) / (5.9e8 + 1.0e9), rel=1e-12)
    # the same 1 GPa written in MPa (p_h = 1000) gives rho/rho0 ~ 1 at P = 1: compressibility lost
    assert float(lm.nd_dowson_higginson(1.0e3)(np.array([1.0]))[0]) == pytest.approx(1.0, abs=1e-6)


# --- surface pressure and subsurface stresses ----------------------------------------------------

DEPTH = np.linspace(0.0, 2.0, 200001)                     # z / s_min, step 1e-5 [-]


@pytest.mark.parametrize("judge", [hertz_point(R_BALL, INNER, load=2436.0, **STEEL),
                                   hertz_line((5e-3, math.inf), (30e-3, math.inf), w=5e4, **STEEL)])
def test_hertz_pressure_peak_and_edge(judge):
    """p0 at the centre, the ellipse value inside, zero on and outside the edge (point and line)."""
    s_x = judge["s_x_m"]
    assert float(hertz_pressure(judge, 0.0)) == judge["p0_Pa"]
    assert float(hertz_pressure(judge, 0.6 * s_x)) == pytest.approx(0.8 * judge["p0_Pa"], rel=1e-12)
    assert np.all(hertz_pressure(judge, np.array([1.0, 1.5, -3.0]) * s_x) == 0.0)


def test_line_stress_maximum_shear_on_the_axis():
    """Johnson (1985) Sec. 4.2(c), Fig. 4.5: tau1 max = 0.30 p0 at z = 0.78 a (printed to 2-3 digits)."""
    tau1 = hertz_line_stress(0.0 * DEPTH, DEPTH)["tau1_over_p0"]
    assert tau1.max() == pytest.approx(0.300, abs=1e-3)
    assert DEPTH[tau1.argmax()] == pytest.approx(0.786, abs=5e-3)


def test_line_stress_on_the_axis_matches_the_axis_formulas():
    """On x = 0: sigma_z = -(1 + zeta^2)^(-1/2) (Eq. (4.46b), corrected by Appendix 4) and
    tau1 = zeta - zeta^2 (1 + zeta^2)^(-1/2) (Johnson PDF 112), at every depth; sigma_z(0) = -1."""
    s = hertz_line_stress(0.0 * DEPTH, DEPTH)
    np.testing.assert_allclose(s["sigma_z_over_p0"], -1.0 / np.sqrt(1.0 + DEPTH**2), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(s["tau1_over_p0"], DEPTH - DEPTH**2 / np.sqrt(1.0 + DEPTH**2), atol=1e-12)
    assert s["sigma_z_over_p0"][0] == -1.0 and s["sigma_x_over_p0"][0] == -1.0


def test_line_stress_shear_sign_opposite_to_x():
    """Johnson Appendix 4 table 3 prints tau_xz as -/+ against +-x/a: negative for x > 0."""
    assert float(hertz_line_stress(0.5, 0.3)["tau_xz_over_p0"]) < 0.0
    assert float(hertz_line_stress(-0.5, 0.3)["tau_xz_over_p0"]) > 0.0


@pytest.mark.parametrize("x, z", [(0.3, 0.4), (-0.7, 0.8), (1.2, 0.5)])
def test_line_stress_satisfies_equilibrium(x, z):
    """No body force, plane strain: d sigma_x/dx + d tau_xz/dz = 0 and d tau_xz/dx + d sigma_z/dz = 0
    (z into the body), by central differences at interior points; this fixes the sign of tau_xz."""
    d = 1e-5                                                  # finite-difference step [-]

    def s(key, dx=0.0, dz=0.0):
        return float(hertz_line_stress(x + dx, z + dz)[key])

    def ddx(key):
        return (s(key, dx=d) - s(key, dx=-d)) / (2.0 * d)

    def ddz(key):
        return (s(key, dz=d) - s(key, dz=-d)) / (2.0 * d)

    assert abs(ddx("sigma_x_over_p0") + ddz("tau_xz_over_p0")) < 1e-6
    assert abs(ddx("tau_xz_over_p0") + ddz("sigma_z_over_p0")) < 1e-6


def test_line_stress_matches_johnson_appendix_4():
    """McEwen's field against the 192 printed values of Johnson's Appendix 4, tables 1-3 (PDF 439-440),
    within their print resolution 0.001: -sigma_x/p0, -sigma_z/p0 and |tau_xz|/p0 (printed as -/+
    magnitudes). Rows copied unchanged in validation/contact/hertz_line_subsurface/reference_values.csv."""
    path = Path(__file__).resolve().parents[1] / "validation" / "contact" / "hertz_line_subsurface" / "reference_values.csv"
    tables = {"Appendix 4, table 1": lambda s: -s["sigma_x_over_p0"], "Appendix 4, table 2": lambda s: -s["sigma_z_over_p0"],
              "Appendix 4, table 3": lambda s: abs(s["tau_xz_over_p0"])}
    with path.open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["location"] in tables]
    assert len(rows) == 192
    for r in rows:
        z_tag, x_tag = r["condition"].split(";")              # "z/a=0.2", "x/a=+-0.4"
        x, z = float(x_tag.split("=+-")[1]), float(z_tag.split("=")[1])
        assert float(tables[r["location"]](hertz_line_stress(x, z))) == pytest.approx(float(r["value"]), abs=1e-3), r


def test_point_axis_stress_surface_and_maximum():
    """Johnson Eq. (3.45), nu = 0.3: sigma_z(0) = -p0, sigma_r(0) = -(1 + 2 nu)/2 p0 = -0.8 p0;
    tau1 max about 0.31 p0 at z = 0.48 a (Johnson Sec. 3.4, printed to 2 digits)."""
    surface = hertz_point_axis_stress(0.0, 0.3)
    assert float(surface["sigma_z_over_p0"]) == -1.0
    assert float(surface["sigma_r_over_p0"]) == pytest.approx(-(1.0 + 2.0 * 0.3) / 2.0, abs=1e-12)
    tau1 = hertz_point_axis_stress(DEPTH, 0.3)["tau1_over_p0"]
    assert tau1.max() == pytest.approx(0.31, abs=5e-3)
    assert DEPTH[tau1.argmax()] == pytest.approx(0.48, abs=1e-2)


@pytest.mark.parametrize("function, args", [(hertz_line_stress, (0.0, -0.1)), (hertz_point_axis_stress, (-0.1, 0.3))])
def test_subsurface_stress_rejects_a_negative_depth(function, args):
    with pytest.raises(ValueError, match="depth"):
        function(*args)
