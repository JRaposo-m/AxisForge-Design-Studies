"""r02_hertz_contact (SlipPY ``hertz_full``) against the exact Hertz solution of this file, Hamrock-Brewe 1983
Tables 1-2, closed forms and identities."""
import csv
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import dblquad
from scipy.optimize import brentq
from scipy.special import ellipe, ellipk, elliprd

from studies.r00_analytical_references.r02_hertz_contact import (Validity, hamrock_brewe_fit, hertz,
                                                                 hertz_point)

PUBLISHED = Path(__file__).resolve().parents[1] / "published" / "hamrock_brewe_1983_t1_t2.csv"
CM, N_PER_CM2 = 1e-2, 1e4                       # table units -> m, Pa

# 12.7 mm ball in a groove of radius 0.52 Dw (concave across rolling), rolling radii of a raceway
# on a 70 mm pitch diameter at zero contact angle: (70 -/+ 12.7) / 2 mm. Radii (R_x, R_y) [m], x along
# rolling, convex > 0.
R_BALL = 6.35e-3
INNER = (28.65e-3, -0.52 * 12.7e-3)       # convex along rolling
OUTER = (-41.35e-3, -0.52 * 12.7e-3)      # concave along rolling
STEEL = ((210e9, 210e9), (0.3, 0.3))      # E_Pa, nu of (body 1, body 2)
# two materials: pairing nu of one body with E of the other, or dropping a body, would show
MIXED = ((210e9, 70e9), (0.3, 0.33))
INF = math.inf


def _e_star(material):
    """Contact modulus [Pa], ``1/E* = (1 - nu1^2)/E1 + (1 - nu2^2)/E2``, built here from the raw inputs."""
    (E1, E2), (nu1, nu2) = material
    return 1.0 / ((1.0 - nu1**2) / E1 + (1.0 - nu2**2) / E2)


def _ball(r2, material, F_N):
    """The ball (body 1) against the raceway ``r2 = (R_x, R_y)`` (body 2), axes in common."""
    return hertz(R_BALL, r2, *material, F_N)


# ---- the check of hertz_full: the exact Hertz solution (R1b), a test helper, not public API ------

def _relative_curvatures(r1, r2, angle):
    """Principal relative curvatures ``(k_min, k_max)`` [1/m] and their unit directions in the axes of body 1.

    Near the first contact each surface is ``z = (x, y) K (x, y)^T / 2`` with the curvature tensor ``K`` of
    the body; the gap is the sum of the two tensors (Johnson 1985, Sec. 4.1, Eq. (4.4)). Body 1 is
    ``diag(1/R1x, 1/R1y)``; body 2, whose x axis lies at ``+angle`` from that of body 1 (counter-clockwise
    in the axes of body 1), is ``Q diag(1/R2x, 1/R2y) Q^T`` with ``Q`` the rotation by ``angle``. The
    eigenvalues of the sum are the principal relative curvatures, the eigenvectors their directions.
    Written without hertz_full's ``alpha``, ``beta``.
    """
    c, s = math.cos(angle), math.sin(angle)
    Q = np.array([[c, -s], [s, c]])
    K = np.diag([1.0 / r1[0], 1.0 / r1[1]]) + Q @ np.diag([1.0 / r2[0], 1.0 / r2[1]]) @ Q.T
    k, vectors = np.linalg.eigh(K)                              # ascending: k_min, k_max
    return (float(k[0]), float(k[1])), vectors


def _exact_point(k_min, k_max, e_star, F_N):
    """Exact elliptical Hertz contact, ``(a, b, p0, delta)`` with ``a >= b``.

    With ``A >= B`` the semi-axes (Johnson's ``a >= b``), ``m = e^2 = 1 - (B/A)^2`` and ``K(m)``,
    ``E(m)`` the complete elliptic integrals, the ellipsoidal pressure displaces the surfaces by
    ``u = (L - M s^2 - N t^2) / (pi E*)``, ``s`` along ``A`` (Johnson 1985, Sec. 3.5(b), Eqs (3.61)-(3.63);
    Sec. 4.2(b), Eqs (4.25)-(4.28))::

        M = pi p0 B (K - E) / (m A^2)
        N = pi p0 B ((A/B)^2 E - K) / (m A^2)
        L = pi p0 B K

    Contact requires ``u = delta - h`` inside the ellipse, hence ``M = pi E* k_min / 2``,
    ``N = pi E* k_max / 2`` and ``delta = L / (pi E*) = p0 B K / E*``: ``A`` lies along the direction of
    ``k_min``. ``N / M = k_max / k_min`` fixes ``m`` (one scalar root; Hamrock-Brewe Eqs (8)-(12) solve the
    same equation by iteration) and the load ``F = (2/3) pi A B p0`` closes the system. ``(K - E) / m`` and
    ``(E - (1 - m) K) / m`` are 0/0 at ``m = 0``, so they are Carlson's ``R_D`` (DLMF 19.25.1),
    regular there (both tend to ``pi / 4``: the circle is the root ``m = 0``)::

        (K - E) / m                     = R_D(0, 1 - m, 1) / 3
        (E - (1 - m) K) / (m (1 - m))   = R_D(0, 1, 1 - m) / 3
    """
    # left minus right is <= 0 at m = 0 and grows as 1/(1 - m) towards m = 1 (K infinite): A/B up to 1e6
    m = brentq(lambda m: k_min * elliprd(0.0, 1.0, 1.0 - m) - k_max * elliprd(0.0, 1.0 - m, 1.0),
               0.0, 1.0 - 1e-12, xtol=1e-16)
    D = float(elliprd(0.0, 1.0 - m, 1.0)) / 3.0                          # (K - E) / m
    A = (3.0 * F_N * D / (math.pi * k_min * e_star)) ** (1.0 / 3.0)    # major semi-axis, from M and F
    B = A * math.sqrt(1.0 - m)
    p0 = 3.0 * F_N / (2.0 * math.pi * A * B)
    return A, B, p0, p0 * B * float(ellipk(m)) / e_star


HF_VS_EXACT_REL = 1e-10      # reached: 6.0e-12 (a, crossed_45_reduced) over these 19 cases (R1d)
DEG = math.pi / 180.0
GROOVE = (INF, -0.52 * 12.7e-3)         # straight groove, concave across: the ball conforms in y
EXACT_CASES = {               # (r1, r2, angle) [m, m, rad], material, F [N]
    # (a) the R1c cases, unchanged in value (angle 0)
    "ball_inner": (((R_BALL, R_BALL), INNER, 0.0), STEEL, 2436.0),
    "ball_outer_mixed": (((R_BALL, R_BALL), OUTER, 0.0), MIXED, 800.0),
    "ball_flat_mixed": (((R_BALL, R_BALL), (INF, INF), 0.0), MIXED, 500.0),
    "crossed_45_reduced": (((10e-3 / (1.0 + math.sqrt(0.5)), 10e-3 / (1.0 - math.sqrt(0.5))), (INF, INF), 0.0),
                           STEEL, 100.0),
    "y_flatter": (((5e-3, 2e-3), (INF, INF), 0.0), STEEL, 50.0),
    **{f"alpha_{alpha:g}": (((1e-2, alpha * 1e-2), (INF, INF), 0.0), STEEL, 4.448)
       for alpha in (1.0001, 1.25, 3.0, 10.0, 100.0, 1e3, 1e4)},
    # (b) two bodies at an angle
    "crossed_45_bodies": (((10e-3, INF), (10e-3, INF), 45 * DEG), STEEL, 100.0),
    "crossed_30_unequal": (((10e-3, INF), (20e-3, INF), 30 * DEG), MIXED, 100.0),
    "ellipsoid_on_cylinder_60": (((5e-3, 20e-3), (8e-3, INF), 60 * DEG), STEEL, 200.0),
    "ellipsoid_on_cylinder_120": (((5e-3, 20e-3), (8e-3, INF), 120 * DEG), STEEL, 200.0),
    "two_ellipsoids_-70": (((4e-3, 9e-3), (12e-3, 6e-3), -70 * DEG), MIXED, 300.0),
    # (d) ball in a conforming groove: rolling along x, the groove radius (concave) across
    "ball_groove_inner_turned_25": (((R_BALL, R_BALL), INNER, 25 * DEG), STEEL, 1000.0),
    "ball_straight_groove": (((R_BALL, R_BALL), GROOVE, 0.0), STEEL, 1000.0),
}


@pytest.mark.parametrize("name", EXACT_CASES)
def test_hertz_full_matches_the_exact_solution(name):
    """r02 (hertz_full) against the exact solution: a, b, p0, delta, and E* from the raw inputs."""
    (r1, r2, angle), material, F_N = EXACT_CASES[name]
    r = hertz(r1, r2, *material, F_N, angle_rad=angle)
    (k_min, k_max), _ = _relative_curvatures(r1, r2, angle)
    assert (r.a_m, r.b_m, r.p0_Pa, r.delta_m) == pytest.approx(_exact_point(k_min, k_max, _e_star(material), F_N),
                                                                rel=HF_VS_EXACT_REL)
    assert r.E_star_Pa == pytest.approx(_e_star(material), rel=1e-15)


@pytest.mark.parametrize("name", EXACT_CASES)
def test_relative_radii_are_the_eigenvalues(name):
    """``Rx_rel_m`` <= ``Ry_rel_m`` (hertz_full, Johnson Eq. (4.4) in its alpha / beta form) against the
    eigenvalues of the curvature sum; ``R_e`` their geometric mean."""
    (r1, r2, angle), _, _ = EXACT_CASES[name]
    r = hertz(r1, r2, *STEEL, 100.0, angle_rad=angle)
    (k_min, k_max), _ = _relative_curvatures(r1, r2, angle)
    assert (r.Rx_rel_m, r.Ry_rel_m) == pytest.approx((1.0 / k_max, 1.0 / k_min), rel=1e-12)
    assert r.R_e_m == pytest.approx((k_min * k_max) ** -0.5, rel=1e-12)


# ---- published: Hamrock-Brewe 1983, Tables 1-2 -------------------------------------------------

def _published():
    """Rows of the two tables: {(location, symbol, alpha): (value as printed, unit)} and the parameters."""
    rows = list(csv.DictReader(PUBLISHED.open(encoding="utf-8")))
    table = {}
    for r in rows:
        if r["condition"].startswith("alpha="):
            alpha = float(r["condition"].split(";")[0].removeprefix("alpha="))
            table[(r["location"], r["symbol"], alpha)] = (r["value"], r["unit"])
    params = {r["symbol"]: float(r["value"]) for r in rows if not r["condition"].startswith("alpha=")}
    return table, params


TABLE, PARAMS = _published()
ALPHAS = sorted({alpha for _, _, alpha in TABLE})
TO_SI = {"-": 1.0, "cm": CM, "N/cm^2": N_PER_CM2}
# Printed cells the formula misses by more than half a unit of the last digit, checked to one unit:
# Eqs (13)-(17) with the fits give 0.439499e+05 N/cm^2 at alpha 1.25, printed 0.440e+05 (0.0003 % below
# the rounding boundary 0.4395e+05: the authors' rounding, not a formula difference).
ONE_UNIT = {("Table 2", "sigma_max_bar", 1.25)}


def _printed(location, symbol, alpha):
    """(value in SI, half a unit of the last printed digit in SI) of one published cell."""
    text, unit = TABLE[(location, symbol, alpha)]
    mantissa, _, exponent = text.lower().partition("e")
    half = 0.5 * 10.0 ** (int(exponent or 0) - len(mantissa.partition(".")[2]))
    if (location, symbol, alpha) in ONE_UNIT:
        half *= 2.0
    return float(text) * TO_SI[unit], half * TO_SI[unit]


def _table2_body():
    """The Table 2 contact: R_x = 1 cm, F = 4.448 N, E' = 2 E* = 2.1972e7 N/cm^2 (nu 0.3 on both)."""
    E_prime = PARAMS["E'"] * N_PER_CM2
    E = E_prime * (1.0 - 0.3**2)                 # E' = E / (1 - nu^2) for two equal bodies
    return PARAMS["R_x"] * CM, PARAMS["F"], (E, 0.3), E_prime


def _hb_eqs_13_17(k, E_int, F_int, alpha):
    """Hamrock-Brewe Eqs (13)-(15), (17) for the Table 2 contact, given k, E and F (the integrals)."""
    R_x, F, _, E_prime = _table2_body()
    R = 1.0 / (1.0 / R_x + 1.0 / (alpha * R_x))                                   # Eq. (2)
    D_y = 2.0 * (6.0 * k**2 * E_int * F * R / (math.pi * E_prime)) ** (1.0 / 3.0)
    D_x = 2.0 * (6.0 * E_int * F * R / (math.pi * k * E_prime)) ** (1.0 / 3.0)
    delta = F_int * (9.0 / (2.0 * E_int * R) * (F / (math.pi * k * E_prime)) ** 2) ** (1.0 / 3.0)
    return {"D_y": D_y, "D_x": D_x, "delta": delta, "sigma_max": 6.0 * F / (math.pi * D_x * D_y)}


def _table2_r02(alpha):
    """The Table 2 equivalent body (R_x, alpha R_x) on a flat; alpha >= 1, so D_y = 2 a and D_x = 2 b."""
    R_x, F, material, _ = _table2_body()
    return hertz((R_x, alpha * R_x), INF, *material, F)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_table_1_exact_columns(alpha):
    """k = D_y / D_x of r02 and K(m), E(m) at m = 1 - 1/k^2, to the printed digits."""
    r = _table2_r02(alpha)
    k = r.a_m / r.b_m
    m = 1.0 - 1.0 / k**2
    for symbol, value in (("k", k), ("F_int", float(ellipk(m))), ("E", float(ellipe(m)))):
        printed, half = _printed("Table 1", symbol, alpha)
        assert value == pytest.approx(printed, abs=half), symbol


@pytest.mark.parametrize("alpha", ALPHAS)
def test_table_1_fit_columns(alpha):
    fit = hamrock_brewe_fit(alpha)
    for symbol, key in (("k_bar", "k"), ("F_int_bar", "F"), ("E_bar", "E")):
        printed, half = _printed("Table 1", symbol, alpha)
        assert fit[key] == pytest.approx(printed, abs=half), symbol
    assert fit["validity"] == Validity(True, "alpha", alpha)


def _table2_ball_in_groove(alpha):
    """(d) The Table 2 contact as two real bodies: a ball of radius R_x in a groove straight along x and
    concave across, of radius ``r_g = alpha R_x / (alpha - 1)`` (``1/R_x - 1/r_g = 1/(alpha R_x)``)."""
    R_x, F, material, _ = _table2_body()
    return hertz(R_x, (INF, -alpha * R_x / (alpha - 1.0)), *material, F)


@pytest.mark.parametrize("alpha, body", [(a, _table2_r02) for a in ALPHAS]
                         + [(a, _table2_ball_in_groove) for a in ALPHAS if a > 1.0])
def test_table_2_exact_columns(alpha, body):
    r = body(alpha)
    for symbol, value in (("D_y", 2.0 * r.a_m), ("D_x", 2.0 * r.b_m), ("sigma_max", r.p0_Pa), ("delta", r.delta_m)):
        printed, half = _printed("Table 2", symbol, alpha)
        assert value == pytest.approx(printed, abs=half), symbol


@pytest.mark.parametrize("alpha", ALPHAS)
def test_table_2_fit_columns(alpha):
    """Eqs (13)-(17) with the fitted k, E, F (Table 4, alpha >= 1)."""
    fit = hamrock_brewe_fit(alpha)
    simplified = _hb_eqs_13_17(fit["k"], fit["E"], fit["F"], alpha)
    for symbol in ("D_y", "D_x", "sigma_max", "delta"):
        printed, half = _printed("Table 2", f"{symbol}_bar", alpha)
        assert simplified[symbol] == pytest.approx(printed, abs=half), symbol


@pytest.mark.parametrize("alpha", [1.0, 1.25, 7.0, 100.0])
def test_r02_satisfies_hamrock_brewe_eqs_13_17(alpha):
    """Identity: Eqs (13)-(17) with r02's own k, E, F give back its a, b, p0, delta."""
    r = _table2_r02(alpha)
    k = r.a_m / r.b_m
    m = 1.0 - 1.0 / k**2
    eqs = _hb_eqs_13_17(k, float(ellipe(m)), float(ellipk(m)), alpha)
    assert (eqs["D_y"], eqs["D_x"]) == (pytest.approx(2.0 * r.a_m, rel=1e-10), pytest.approx(2.0 * r.b_m, rel=1e-10))
    assert (eqs["sigma_max"], eqs["delta"]) == (pytest.approx(r.p0_Pa, rel=1e-10), pytest.approx(r.delta_m, rel=1e-10))


# ---- cowork artifact A10 (load control in the bridge, C11) ---------------------------------------

@pytest.mark.parametrize("bodies", [((10e-3, INF), (10e-3, INF), math.pi / 4),
                                    ((10e-3 / (1.0 + math.sqrt(0.5)), 10e-3 / (1.0 - math.sqrt(0.5))), INF, 0.0)])
def test_crossed_cylinders_at_45_deg(bodies):
    """(b) Two steel cylinders R 10 mm, axes 45 deg apart, 100 N, as two bodies at 45 deg and (R1c) as their
    principal relative radii (1 +/- cos 45) / R against a flat (Johnson 1985, Sec. 4.1). A10: p0 1178.488 MPa,
    b 0.112690 mm, delta 2.97689 um."""
    r1, r2, angle = bodies
    r = hertz(r1, r2, *STEEL, 100.0, angle_rad=angle)
    assert r.shape == "elliptical"
    assert r.p0_Pa / 1e6 == pytest.approx(1178.488, abs=0.0005)
    assert r.b_m / 1e-3 == pytest.approx(0.112690, abs=0.0000005)
    assert r.delta_m / 1e-6 == pytest.approx(2.97689, abs=0.000005)


def test_crossed_equal_cylinders_at_90_deg_are_a_sphere_on_a_flat():
    """(c) Curvature sum diag(1/R, 0) + diag(0, 1/R): the gap of a sphere of radius R on a flat."""
    crossed = hertz((10e-3, INF), (10e-3, INF), *STEEL, 100.0, angle_rad=math.pi / 2)
    sphere = hertz(10e-3, INF, *STEEL, 100.0)
    assert crossed.shape == sphere.shape == "sphere"
    assert (crossed.a_m, crossed.b_m, crossed.p0_Pa, crossed.delta_m) == pytest.approx(
        (sphere.a_m, sphere.b_m, sphere.p0_Pa, sphere.delta_m), rel=1e-12)


def test_line_10_20_mm():
    """(f) Two steel cylinders R 10 and 20 mm, 1e5 N/m. A10: p0 742.240 MPa."""
    r = hertz((10e-3, INF), (20e-3, INF), *STEEL, 1e5, line=True)
    assert r.shape == "line" and r.a_m == INF
    assert (r.delta_m, r.area_m2, r.k_n_N_per_m) == (None, None, None)
    assert r.p0_Pa / 1e6 == pytest.approx(742.240, abs=0.0005)


@pytest.mark.parametrize("r1", [(INF, 15e-3), (15e-3, 1e4)])
def test_line_along_x_or_above_the_ratio_has_an_infinite_a(r1):
    """A line along x (hertz_full leaves ~1e29 for the infinite relative radius) and a finite radius ratio
    above hertz_full's 1e5: still a line, ``a_m = inf``, the half-width as for a cylinder of radius 15 mm."""
    r = hertz(r1, INF, *STEEL, 1e5, line=True)
    ref = hertz((15e-3, INF), INF, *STEEL, 1e5, line=True)
    assert r.shape == "line" and r.a_m == INF
    assert r.b_m == pytest.approx(ref.b_m, rel=1e-6) and r.validity.governing == "b/r1"


def test_line_contact_without_the_flag_raises():
    """(f) hertz_full refuses to guess a line contact (the load would change meaning), and the reverse."""
    with pytest.raises(ValueError, match="Line contact detected"):
        hertz((10e-3, INF), (20e-3, INF), *STEEL, 1e5)
    with pytest.raises(ValueError, match="elliptical contact"):
        hertz(R_BALL, INNER, *STEEL, 100.0, line=True)


# ---- (e) invariances -----------------------------------------------------------------------------

INVARIANCE_CASES = ["crossed_30_unequal", "ellipsoid_on_cylinder_60", "two_ellipsoids_-70",
                    "ball_groove_inner_turned_25", "ball_outer_mixed"]


@pytest.mark.parametrize("name", INVARIANCE_CASES)
def test_invariances(name):
    """Swapping the bodies (materials with them), angle -> -angle and angle -> angle + pi leave the
    contact unchanged: the eigenvalues of the curvature sum do not depend on them."""
    (r1, r2, angle), ((E1, E2), (nu1, nu2)), F_N = EXACT_CASES[name]
    ref = hertz(r1, r2, (E1, E2), (nu1, nu2), F_N, angle_rad=angle)
    for other in (hertz(r2, r1, (E2, E1), (nu2, nu1), F_N, angle_rad=angle),
                  hertz(r1, r2, (E1, E2), (nu1, nu2), F_N, angle_rad=-angle),
                  hertz(r1, r2, (E1, E2), (nu1, nu2), F_N, angle_rad=angle + math.pi)):
        assert (other.a_m, other.b_m, other.p0_Pa, other.delta_m) == pytest.approx(
            (ref.a_m, ref.b_m, ref.p0_Pa, ref.delta_m), rel=1e-12)


def test_groove_axis_order():
    """(d) The groove given as (R_x, R_y) or as (R_y, R_x) with the ball: same contact, the major semi-axis
    across rolling (alpha_rad = pi/2) or along the new x (alpha_rad = 0) of the raceway's axes. The concave
    groove radius is negative in either order."""
    r, s = _ball(INNER, STEEL, 100.0), _ball(INNER[::-1], STEEL, 100.0)
    assert r.a_m > r.b_m
    assert (s.a_m, s.b_m, s.p0_Pa, s.delta_m) == pytest.approx((r.a_m, r.b_m, r.p0_Pa, r.delta_m), rel=1e-14)
    assert math.cos(r.alpha_rad) == pytest.approx(0.0, abs=1e-15)
    assert math.sin(s.alpha_rad) == pytest.approx(0.0, abs=1e-15)


# ---- (g) orientation of the contact ellipse ------------------------------------------------------

ORIENTATION_CASES = ["crossed_45_bodies", "crossed_30_unequal", "ellipsoid_on_cylinder_60",
                     "ellipsoid_on_cylinder_120", "two_ellipsoids_-70", "ball_groove_inner_turned_25",
                     "ball_inner", "y_flatter"]


def _same_axis(theta, vector):
    """|sin| of the angle between the direction ``theta`` and ``vector``: 0 when they are one axis."""
    return abs(math.cos(theta) * vector[1] - math.sin(theta) * vector[0])


@pytest.mark.parametrize("name", ORIENTATION_CASES)
def test_alpha_beta_point_along_the_major_semi_axis(name):
    """alpha_rad from the x axis of body 1, -beta_rad from the x axis of body 2 (at +angle in the axes of
    body 1): both on the eigenvector of the smaller relative curvature, the direction of ``a``."""
    (r1, r2, angle), material, F_N = EXACT_CASES[name]
    r = hertz(r1, r2, *material, F_N, angle_rad=angle)
    _, vectors = _relative_curvatures(r1, r2, angle)
    major = vectors[:, 0]                                       # k_min: the flatter gap
    assert _same_axis(r.alpha_rad, major) == pytest.approx(0.0, abs=1e-12)
    assert _same_axis(angle - r.beta_rad, major) == pytest.approx(0.0, abs=1e-12)


def _major_direction(pressure, a_m, transform):
    """Direction in [-pi/2, pi/2) of the point at 0.999 a where ``pressure`` is largest (the major axis)."""
    theta = np.linspace(-math.pi / 2, math.pi / 2, 200001, endpoint=False)
    p = pressure(0.999 * a_m * np.cos(theta), 0.999 * a_m * np.sin(theta), transform=transform)
    return float(theta[np.argmax(p)])


@pytest.mark.parametrize("name", ["crossed_30_unequal", "ellipsoid_on_cylinder_60", "two_ellipsoids_-70"])
def test_pressure_f_axes(name):
    """The raw pressure closure: ``transform=0`` has ``a`` along x, ``transform=2`` puts it at -beta_rad in the
    axes of body 2 (as the eigenvectors); ``transform=1`` puts it at -alpha_rad in the axes of body 1, the
    mirror image (y -> -y) of the eigenvectors (OPEN RD-04, SlipPY)."""
    (r1, r2, angle), material, F_N = EXACT_CASES[name]
    r = hertz(r1, r2, *material, F_N, angle_rad=angle)
    p = r.raw["pressure_f"]
    for transform, expected in ((0, 0.0), (2, -r.beta_rad), (1, -r.alpha_rad)):
        found = _major_direction(p, r.a_m, transform)
        assert abs(math.sin(found - expected)) < 1e-4, transform


# ---- limits, symmetries and the sign convention ------------------------------------------------

def test_sphere_on_a_flat():
    """Sphere of radius R on a flat (Johnson 1985, Sec. 4.2(a)):
    ``a = (3 F R / (4 E*))**(1/3)``, ``p0 = 3 F / (2 pi a^2)``, ``delta = a^2 / R``."""
    F = 500.0
    r = hertz(R_BALL, INF, *MIXED, F)
    a = (3.0 * F * R_BALL / (4.0 * _e_star(MIXED))) ** (1.0 / 3.0)
    assert r.shape == "sphere"
    assert (r.a_m, r.b_m) == (pytest.approx(a, rel=1e-12), pytest.approx(a, rel=1e-12))
    assert r.p0_Pa == pytest.approx(3.0 * F / (2.0 * math.pi * a**2), rel=1e-12)
    assert r.delta_m == pytest.approx(a**2 / R_BALL, rel=1e-12)
    assert r.area_m2 == pytest.approx(math.pi * a**2, rel=1e-12)


def test_concave_radius_is_negative():
    """A concave raceway (R < 0) is the equivalent convex body of curvature 1/R_ball - 1/|R|: the outer
    contact is wider and lower in pressure than the inner one at the same load."""
    r = _ball(OUTER, STEEL, 1000.0)
    eq = hertz((1.0 / (1.0 / R_BALL + 1.0 / OUTER[0]), 1.0 / (1.0 / R_BALL + 1.0 / OUTER[1])), INF, *STEEL, 1000.0)
    assert (r.a_m, r.b_m, r.p0_Pa, r.delta_m) == pytest.approx((eq.a_m, eq.b_m, eq.p0_Pa, eq.delta_m), rel=1e-14)
    inner = _ball(INNER, STEEL, 1000.0)
    assert r.b_m > inner.b_m and r.p0_Pa < inner.p0_Pa


def test_hertz_point_is_hertz_at_angle_0():
    """The pre-R1d call kept for tests/test_slippy_bridge.py."""
    (E1, E2), (nu1, nu2) = MIXED
    r, s = hertz_point(R_BALL, R_BALL, *OUTER, E1, nu1, E2, nu2, 800.0), _ball(OUTER, MIXED, 800.0)
    assert (r.a_m, r.b_m, r.p0_Pa, r.delta_m) == (s.a_m, s.b_m, s.p0_Pa, s.delta_m)


@pytest.mark.parametrize("angle", [0.0, 0.5, 1.2])
def test_both_bodies_more_curved_along_y(angle):
    """OPEN RD-03 (SlipPY): hertz_full raises its 'bug' message at any angle; one body given with its axes
    swapped and angle + pi/2 is the same pair, and matches the exact helper."""
    r1, r2 = (20e-3, 5e-3), (30e-3, 8e-3)
    with pytest.raises(ValueError, match="bug in this program"):
        hertz(r1, r2, *STEEL, 100.0, angle_rad=angle)
    r = hertz(r1, r2[::-1], *STEEL, 100.0, angle_rad=angle + math.pi / 2)
    (k_min, k_max), _ = _relative_curvatures(r1, r2, angle)
    assert (r.a_m, r.b_m, r.p0_Pa, r.delta_m) == pytest.approx(_exact_point(k_min, k_max, _e_star(STEEL), 100.0),
                                                                rel=HF_VS_EXACT_REL)


@pytest.mark.parametrize("r2, material", [(INNER, STEEL), (OUTER, MIXED), ((INF, INF), MIXED)])
def test_pressure_carries_the_load_and_the_derived_fields(r2, material):
    """Load tie ``F = (2/3) pi a b p0``; pm = 2 p0 / 3; k_n = 3 F / (2 delta) (F ~ delta^(3/2))."""
    r = _ball(r2, material, 1234.5)
    assert 2.0 / 3.0 * math.pi * r.a_m * r.b_m * r.p0_Pa == pytest.approx(1234.5, rel=1e-12)
    assert r.pm_Pa == pytest.approx(2.0 * r.p0_Pa / 3.0, rel=1e-15)
    dF = 1e-4 * 1234.5
    up, down = _ball(r2, material, 1234.5 + dF), _ball(r2, material, 1234.5 - dF)
    assert r.k_n_N_per_m == pytest.approx(2.0 * dF / (up.delta_m - down.delta_m), rel=1e-7)


@pytest.mark.parametrize("r2, material, F_N", [(INNER, STEEL, 2436.0), (OUTER, MIXED, 800.0)])
def test_hertz_pressure_closes_the_gap(r2, material, F_N):
    """Boussinesq self-check: r02's pressure deforms the surfaces into contact.

    A point load P on an elastic half-space lowers the surface by ``(1 - nu^2) P / (pi E r)`` at
    the distance r (Boussinesq; Johnson 1985, Sec. 3.2, Eq. (3.22b)). Summed over both bodies and over the pressure,
    ``u(x, y) = 1/(pi E*) * integral p / r dA``. Inside the contact the two surfaces coincide, so
    ``u = delta - k_x x^2/2 - k_y y^2/2`` (approach minus the initial gap). The curvatures and
    ``E*`` are built here from the raw inputs, so the check does not depend on how r02
    computes them.

    The integral is taken in polar coordinates centred on the evaluation point, where
    ``dA / r = dr dtheta`` and the 1/r singularity cancels. Along a ray the pressure is
    ``p0 sqrt(A (r_max - r)(r - r_min))``; with ``r = r_max (1 - t^2)`` the square root at the edge
    of the contact becomes smooth in t.
    """
    r = _ball(r2, material, F_N)
    assert math.cos(r.alpha_rad) == pytest.approx(0.0, abs=1e-15)     # the groove conforms: a along y
    p0, s_x, s_y, delta = r.p0_Pa, r.b_m, r.a_m, r.delta_m
    k_x, k_y = 1.0 / R_BALL + 1.0 / r2[0], 1.0 / R_BALL + 1.0 / r2[1]

    def displacement(x0, y0):
        C = 1.0 - (x0 / s_x) ** 2 - (y0 / s_y) ** 2

        def integrand(t, theta):
            c, s = math.cos(theta), math.sin(theta)
            A = (c / s_x) ** 2 + (s / s_y) ** 2
            B = x0 * c / s_x**2 + y0 * s / s_y**2
            root = math.sqrt(B * B + A * C)
            r_max, r_min = (-B + root) / A, (-B - root) / A     # the ray meets the ellipse at r_max
            rr = r_max * (1.0 - t * t)
            return p0 * math.sqrt(A * r_max * (rr - r_min)) * 2.0 * r_max * t * t   # p dr, dr = 2 r_max t dt

        return dblquad(integrand, 0.0, 2.0 * math.pi, 0.0, 1.0, epsabs=0.0, epsrel=1e-11)[0] / (
            math.pi * _e_star(material))

    # centre, one point on each axis, one off both axes
    for fx, fy in [(0.0, 0.0), (0.5, 0.0), (0.0, 0.6), (0.4, 0.5)]:
        x0, y0 = fx * s_x, fy * s_y
        assert displacement(x0, y0) == pytest.approx(delta - k_x * x0**2 / 2 - k_y * y0**2 / 2, rel=1e-8)


@pytest.mark.parametrize("material", [STEEL, MIXED])
def test_line_contact_matches_johnson(material):
    """Line contact (Johnson 1985, Eqs (4.43)-(4.45)):
    ``b = (4 w R / (pi E*))**(1/2)``, ``p0 = 2 w / (pi b) = (w E* / (pi R))**(1/2)``."""
    w, R1, R2 = 5e4, 5e-3, 30e-3              # 10 mm roller on the inner raceway of a 70 mm pitch diameter
    R = 1.0 / (1.0 / R1 + 1.0 / R2)
    e_star = _e_star(material)
    r = hertz((R1, INF), (R2, INF), *material, w, line=True)
    b = math.sqrt(4.0 * w * R / (math.pi * e_star))
    assert r.b_m == pytest.approx(b, rel=1e-12)
    assert r.p0_Pa == pytest.approx(2.0 * w / (math.pi * b), rel=1e-12)
    assert r.p0_Pa == pytest.approx(math.sqrt(w * e_star / (math.pi * R)), rel=1e-12)
    assert r.pm_Pa == pytest.approx(w / (2.0 * b), rel=1e-12)


# ---- validity (P-02): soft range in a record, domain errors raise --------------------------------

def test_size_ratio_is_per_body_and_a_record_not_an_error():
    """Largest semi-axis over the smallest finite |radius| of each body; flats are skipped, concave radii
    count. The ball (6.35 mm) is tighter than its groove (6.604 mm): a / R_ball passes 0.1 near 70 N."""
    small, large = _ball(INNER, STEEL, 10.0), _ball(INNER, STEEL, 100.0)
    assert small.validity.in_range and small.validity.governing == "a/r1"
    assert small.validity.ratio == pytest.approx(small.a_m / R_BALL, rel=1e-15)
    assert not large.validity.in_range and large.validity.ratio > 0.1 and not large.validity.clamp_acted
    swapped = hertz(INNER, R_BALL, *STEEL, 10.0)                                # the ball as body 2
    assert swapped.validity.governing == "a/r2" and swapped.validity.ratio == small.validity.ratio
    # the concave groove radius counts (|r| = 6.604 mm) but cannot govern: a positive relative curvature
    # along it needs a tighter convex radius on the other body
    outer = _ball(OUTER, STEEL, 10.0)
    assert outer.validity.governing == "a/r1" and outer.a_m / abs(OUTER[1]) < outer.validity.ratio
    line = hertz((5e-3, INF), (30e-3, INF), *STEEL, 5e4, line=True).validity
    assert line.in_range and line.governing == "b/r1"
    on_flat = hertz(R_BALL, INF, *STEEL, 100.0).validity                        # the flat is skipped
    assert on_flat.governing == "a/r1"
    assert not hamrock_brewe_fit(150.0)["validity"].in_range


@pytest.mark.parametrize("r1, r2", [(R_BALL, (INF, 0.0)),                    # zero radius
                                    ((R_BALL, INF), INF),                     # cylinder on flat: line
                                    (R_BALL, (28.65e-3, -6.0e-3)),            # groove tighter than ball
                                    (INF, INF)])                              # two flats: conformal
@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")   # hertz_full's r_e of a negative radius
def test_not_a_point_contact_raises(r1, r2):
    """hertz_full's errors pass through."""
    with pytest.raises(ValueError):
        hertz(r1, r2, *STEEL, 100.0)


@pytest.mark.parametrize("E, nu, F_N", [((0.0, 210e9), 0.3, 100.0), (210e9, (0.5, 0.3), 100.0),
                                        (210e9, 0.3, 0.0), (210e9, 0.3, -1.0), (210e9, 0.3, math.nan)])
def test_bad_material_or_load_raises(E, nu, F_N):
    with pytest.raises(ValueError):
        hertz(R_BALL, INF, E, nu, F_N)


@pytest.mark.parametrize("r1", [(1e-3, 2e-3, 3e-3), "6.35e-3", None, True, np.array([1e-3, 2e-3])])
def test_bad_type_raises(r1):
    with pytest.raises(TypeError):
        hertz(r1, INF, *STEEL, 100.0)


def test_numpy_scalars_are_numbers():
    r = hertz(np.float32(6.35e-3), INF, np.int64(210_000_000_000), 0.3, np.float64(500.0))
    assert r.p0_Pa == pytest.approx(hertz(float(np.float32(6.35e-3)), INF, 210e9, 0.3, 500.0).p0_Pa, rel=1e-15)


def test_fit_below_alpha_1_raises():
    with pytest.raises(ValueError):
        hamrock_brewe_fit(0.5)
