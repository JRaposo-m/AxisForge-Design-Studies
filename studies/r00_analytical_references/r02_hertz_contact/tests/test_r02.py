"""r02_hertz_contact: the judge alone, against Hamrock-Brewe 1983 Tables 1-2, closed forms and identities."""
import csv
import math
from pathlib import Path

import pytest
from scipy.integrate import dblquad
from scipy.special import ellipe, ellipk

from studies.r00_analytical_references.r02_hertz_contact import Validity, hamrock_brewe_fit, hertz_line, hertz_point

PUBLISHED = Path(__file__).resolve().parents[1] / "published" / "hamrock_brewe_1983_t1_t2.csv"
CM, N_PER_CM2 = 1e-2, 1e4                       # table units -> m, Pa

# 12.7 mm ball in a groove of radius 0.52 Dw (concave across rolling), rolling radii of a raceway
# on a 70 mm pitch diameter at zero contact angle: (70 -/+ 12.7) / 2 mm. Radii [m], convex > 0.
R_BALL = 6.35e-3
INNER = (28.65e-3, -0.52 * 12.7e-3)       # (R_x, R_y): convex along rolling
OUTER = (-41.35e-3, -0.52 * 12.7e-3)      # concave along rolling
STEEL = (210e9, 0.3, 210e9, 0.3)          # E1_Pa, nu1, E2_Pa, nu2
# two materials: pairing nu of one body with E of the other, or dropping a body, would show
MIXED = (210e9, 0.3, 70e9, 0.33)


def _e_star(E1, nu1, E2, nu2):
    """Contact modulus [Pa], ``1/E* = (1 - nu1^2)/E1 + (1 - nu2^2)/E2``, built here from the raw inputs."""
    return 1.0 / ((1.0 - nu1**2) / E1 + (1.0 - nu2**2) / E2)


def _ball(r2, material, F_N):
    """The ball (body 1) against the raceway ``r2 = (R_x, R_y)`` (body 2)."""
    return hertz_point(R_BALL, R_BALL, *r2, *material, F_N)


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
    return PARAMS["R_x"] * CM, PARAMS["F"], (E, 0.3, E, 0.3), E_prime


def _hb_eqs_13_17(k, E_int, F_int, alpha):
    """Hamrock-Brewe Eqs (13)-(15), (17) for the Table 2 contact, given k, E and F (the integrals)."""
    R_x, F, _, E_prime = _table2_body()
    R = 1.0 / (1.0 / R_x + 1.0 / (alpha * R_x))                                   # Eq. (2)
    D_y = 2.0 * (6.0 * k**2 * E_int * F * R / (math.pi * E_prime)) ** (1.0 / 3.0)
    D_x = 2.0 * (6.0 * E_int * F * R / (math.pi * k * E_prime)) ** (1.0 / 3.0)
    delta = F_int * (9.0 / (2.0 * E_int * R) * (F / (math.pi * k * E_prime)) ** 2) ** (1.0 / 3.0)
    return {"D_y": D_y, "D_x": D_x, "delta": delta, "sigma_max": 6.0 * F / (math.pi * D_x * D_y)}


def _table2_judge(alpha):
    R_x, F, material, _ = _table2_body()
    return hertz_point(R_x, alpha * R_x, math.inf, math.inf, *material, F)


@pytest.mark.parametrize("alpha", ALPHAS)
def test_table_1_exact_columns(alpha):
    """k = D_y / D_x of the judge and K(m), E(m) at m = 1 - 1/k^2, to the printed digits."""
    r = _table2_judge(alpha)
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


@pytest.mark.parametrize("alpha", ALPHAS)
def test_table_2_exact_columns(alpha):
    r = _table2_judge(alpha)
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
def test_judge_satisfies_hamrock_brewe_eqs_13_17(alpha):
    """Identity: Eqs (13)-(17) with the judge's own k, E, F give back its a, b, p0, delta."""
    r = _table2_judge(alpha)
    k = r.a_m / r.b_m
    m = 1.0 - 1.0 / k**2
    eqs = _hb_eqs_13_17(k, float(ellipe(m)), float(ellipk(m)), alpha)
    assert (eqs["D_y"], eqs["D_x"]) == (pytest.approx(2.0 * r.a_m, rel=1e-10), pytest.approx(2.0 * r.b_m, rel=1e-10))
    assert (eqs["sigma_max"], eqs["delta"]) == (pytest.approx(r.p0_Pa, rel=1e-10), pytest.approx(r.delta_m, rel=1e-10))


# ---- cowork artifact A10 (load control in the bridge, C11) ---------------------------------------

def test_crossed_cylinders_at_45_deg():
    """Two steel cylinders R 10 mm, axes 45 deg apart, 100 N. Principal relative curvatures of two
    equal cylinders crossed at theta (Johnson 1985, Sec. 4.1): (1 +/- cos theta) / R, given as body 1
    against a flat. A10: p0 1178.488 MPa, b 0.112690 mm, delta 2.97689 um."""
    R = 10e-3
    k_max, k_min = (1.0 + math.sqrt(2.0) / 2.0) / R, (1.0 - math.sqrt(2.0) / 2.0) / R
    r = hertz_point(1.0 / k_max, 1.0 / k_min, math.inf, math.inf, *STEEL, 100.0)
    assert r.p0_Pa / 1e6 == pytest.approx(1178.488, abs=0.0005)
    assert r.b_m / 1e-3 == pytest.approx(0.112690, abs=0.0000005)
    assert r.delta_m / 1e-6 == pytest.approx(2.97689, abs=0.000005)


def test_line_10_20_mm():
    """Two steel cylinders R 10 and 20 mm, 1e5 N/m. A10: p0 742.240 MPa."""
    assert hertz_line(10e-3, 20e-3, *STEEL, 1e5).p0_Pa / 1e6 == pytest.approx(742.240, abs=0.0005)


# ---- limits, symmetries and the sign convention ------------------------------------------------

def test_alpha_1_is_the_sphere():
    """Sphere of radius R on a flat (Johnson 1985, Sec. 4.2(a)):
    ``a = (3 F R / (4 E*))**(1/3)``, ``p0 = 3 F / (2 pi a^2)``, ``delta = a^2 / R``."""
    F = 500.0
    r = hertz_point(R_BALL, R_BALL, math.inf, math.inf, *MIXED, F)
    a = (3.0 * F * R_BALL / (4.0 * _e_star(*MIXED))) ** (1.0 / 3.0)
    assert (r.a_m, r.b_m) == (pytest.approx(a, rel=1e-12), pytest.approx(a, rel=1e-12))
    assert r.p0_Pa == pytest.approx(3.0 * F / (2.0 * math.pi * a**2), rel=1e-12)
    assert r.delta_m == pytest.approx(a**2 / R_BALL, rel=1e-12)
    assert r.E_star_Pa == pytest.approx(_e_star(*MIXED), rel=1e-15)


def test_x_y_swap_swaps_the_semi_axes():
    r = _ball(INNER, STEEL, 100.0)
    s = hertz_point(R_BALL, R_BALL, INNER[1], INNER[0], *STEEL, 100.0)
    assert r.a_m > r.b_m                                       # the groove conforms across rolling
    assert (s.a_m, s.b_m) == (r.b_m, r.a_m)
    assert (s.p0_Pa, s.delta_m) == (r.p0_Pa, r.delta_m)


@pytest.mark.parametrize("r2", [INNER, OUTER])
def test_body_swap_gives_the_same_contact(r2):
    """Inner / outer body swap: the concave raceway as body 1 and the ball as body 2."""
    E1, nu1, E2, nu2 = MIXED
    r = _ball(r2, MIXED, 800.0)
    s = hertz_point(*r2, R_BALL, R_BALL, E2, nu2, E1, nu1, 800.0)
    assert (s.a_m, s.b_m, s.p0_Pa, s.delta_m) == pytest.approx((r.a_m, r.b_m, r.p0_Pa, r.delta_m), rel=1e-14)


def test_concave_radius_is_negative():
    """A concave raceway (R < 0) is the equivalent convex body of curvature 1/R_ball - 1/|R|: the outer
    contact is wider and lower in pressure than the inner one at the same load."""
    r = _ball(OUTER, STEEL, 1000.0)
    eq = hertz_point(1.0 / (1.0 / R_BALL + 1.0 / OUTER[0]), 1.0 / (1.0 / R_BALL + 1.0 / OUTER[1]),
                     math.inf, math.inf, *STEEL, 1000.0)
    assert (r.a_m, r.b_m, r.p0_Pa, r.delta_m) == pytest.approx((eq.a_m, eq.b_m, eq.p0_Pa, eq.delta_m), rel=1e-14)
    inner = _ball(INNER, STEEL, 1000.0)
    assert r.b_m > inner.b_m and r.p0_Pa < inner.p0_Pa


@pytest.mark.parametrize("r2, material", [(INNER, STEEL), (OUTER, MIXED), ((math.inf, math.inf), MIXED)])
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
    """Boussinesq self-check: the judge's pressure deforms the surfaces into contact.

    A point load P on an elastic half-space lowers the surface by ``(1 - nu^2) P / (pi E r)`` at
    the distance r (Boussinesq; Johnson 1985, Sec. 3.2, Eq. (3.22b)). Summed over both bodies and over the pressure,
    ``u(x, y) = 1/(pi E*) * integral p / r dA``. Inside the contact the two surfaces coincide, so
    ``u = delta - k_x x^2/2 - k_y y^2/2`` (approach minus the initial gap). The curvatures and
    ``E*`` are built here from the raw inputs, so the check does not depend on how the judge
    computes them.

    The integral is taken in polar coordinates centred on the evaluation point, where
    ``dA / r = dr dtheta`` and the 1/r singularity cancels. Along a ray the pressure is
    ``p0 sqrt(A (r_max - r)(r - r_min))``; with ``r = r_max (1 - t^2)`` the square root at the edge
    of the contact becomes smooth in t.
    """
    r = _ball(r2, material, F_N)
    p0, s_x, s_y, delta = r.p0_Pa, r.b_m, r.a_m, r.delta_m          # b along x, a along y
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
            math.pi * _e_star(*material))

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
    e_star = _e_star(*material)
    r = hertz_line(R1, R2, *material, w)
    b = math.sqrt(4.0 * w * R / (math.pi * e_star))
    assert r.b_m == pytest.approx(b, rel=1e-12)
    assert r.p0_Pa == pytest.approx(2.0 * w / (math.pi * b), rel=1e-12)
    assert r.p0_Pa == pytest.approx(math.sqrt(w * e_star / (math.pi * R)), rel=1e-12)
    assert r.w_N_per_m == w


# ---- validity (P-02): soft range in a record, domain errors raise --------------------------------

def test_size_ratio_is_a_record_not_an_error():
    """The groove conforms across rolling: a / R_ball passes 0.1 near 70 N for this 6.35 mm ball."""
    small, large = _ball(INNER, STEEL, 10.0), _ball(INNER, STEEL, 100.0)
    assert small.validity.in_range and small.validity.governing == "a/R1y"     # the ball radius governs
    assert small.validity.ratio == pytest.approx(small.a_m / R_BALL, rel=1e-15)
    assert not large.validity.in_range and large.validity.ratio > 0.1 and not large.validity.clamp_acted
    line = hertz_line(5e-3, 30e-3, *STEEL, 5e4).validity
    assert line.in_range and line.governing == "b/R_rel"                       # 1/R = 1/R1 + 1/R2
    spheres = hertz_point(R_BALL, R_BALL, R_BALL, R_BALL, *STEEL, 100.0)        # relative radius R/2 governs
    assert spheres.validity.governing in ("b/Rx_rel", "a/Ry_rel")
    assert spheres.validity.ratio == pytest.approx(2.0 * spheres.a_m / R_BALL, rel=1e-12)
    assert not hamrock_brewe_fit(150.0)["validity"].in_range


@pytest.mark.parametrize("radii", [(R_BALL, R_BALL, math.inf, 0.0),                # zero radius
                                   (R_BALL, math.inf, math.inf, math.inf),         # cylinder on flat: line
                                   (R_BALL, R_BALL, 28.65e-3, -6.0e-3)])           # groove tighter than ball
def test_not_a_point_contact_raises(radii):
    with pytest.raises(ValueError):
        hertz_point(*radii, *STEEL, 100.0)


@pytest.mark.parametrize("material, F_N", [((0.0, 0.3, 210e9, 0.3), 100.0), ((210e9, 0.5, 210e9, 0.3), 100.0),
                                           (STEEL, 0.0)])
def test_bad_material_or_load_raises(material, F_N):
    with pytest.raises(ValueError):
        hertz_point(R_BALL, R_BALL, math.inf, math.inf, *material, F_N)


@pytest.mark.parametrize("R1, R2, w", [(5e-3, -4e-3, 5e4),      # bore tighter than the roller
                                       (5e-3, 30e-3, 0.0),      # no load
                                       (0.0, 30e-3, 5e4)])      # zero radius
def test_not_a_line_contact_raises(R1, R2, w):
    with pytest.raises(ValueError):
        hertz_line(R1, R2, *STEEL, w)


def test_fit_below_alpha_1_raises():
    with pytest.raises(ValueError):
        hamrock_brewe_fit(0.5)
