"""r03_load_zone against Harris (2007) Tables 7.1 and 7.4, closed forms and limits."""
import csv
import math
from pathlib import Path

import numpy as np
import pytest

from studies.r00_analytical_references.r03_load_zone import (N_LINE, N_POINT, ball_deflection, ring_load_distribution,
                                                             sjovall_integrals, solve_ring_displacement,
                                                             stribeck_constant)

PUBLISHED = Path(__file__).resolve().parents[1] / "published"
N_OF = {"point": N_POINT, "line": N_LINE}


def _half_unit(printed: str) -> float:
    """Half a unit of the last printed digit of a decimal string."""
    return 0.5 * 10.0 ** -(len(printed.split(".")[1]) if "." in printed else 0)


def _eps(printed: str) -> float:
    """The eps of a printed row: '1.67' is 5/3 (OPEN R5-03), the infinity sign is inf."""
    return {"1.67": 5.0 / 3.0, "inf": math.inf}.get(printed, None) or float(printed)


# printed values the integral does not reproduce to half a unit (OPEN R5-04); never tuned, listed in the README
_TRUNC = "printed 0.8964, the integral gives 0.896470: last digit truncated, not rounded"
_MISPRINT = "misprint: printed 0.4338, J_r/J_a = 0.2289/0.5044 = 0.4538 (the integral gives 0.453778)"
_RATIO = "Harris prints the line ratio as the ratio of its rounded J_r, J_a (checked in test_line_ratio_column)"
_LINE_J = ("Harris's line integral differs from Eq. 7.67/7.69 with n = 1.1 by 0.6-1.6 units of the last digit; no "
           "single n in 1.09-1.12 reproduces all line values")
KNOWN = {("7_4", "point", "0.3", "Fr_tan_alpha_over_Fa"): _TRUNC,
         ("7_4", "point", "1.25", "Fr_tan_alpha_over_Fa"): _MISPRINT,
         **{("7_4", "line", e, "Fr_tan_alpha_over_Fa"): _RATIO
            for e in ("0.2", "0.3", "0.4", "0.5", "0.6", "0.8", "0.9", "1.25")},
         ("7_4", "line", "0.2", "J_r"): _LINE_J, ("7_4", "line", "0.8", "J_a"): _LINE_J,
         ("7_4", "line", "0.9", "J_a"): _LINE_J, ("7_4", "line", "1.25", "J_r"): _LINE_J,
         ("7_1", "line", "0.2", "J_r"): _LINE_J, ("7_1", "line", "1.25", "J_r"): _LINE_J}


def _table_cases():
    cases = []
    for table, columns in (("7_4", ("Fr_tan_alpha_over_Fa", "J_r", "J_a")), ("7_1", ("J_r",))):
        with open(PUBLISHED / f"harris2007_vol1_table{table}.csv", newline="") as f:
            for row in csv.DictReader(f):
                if row["eps_printed"] == "0":
                    continue                    # "1/Z": the discrete limit, test_one_element_limit
                for col in columns:
                    key = (table, row["contact"], row["eps_printed"], col)
                    marks = [pytest.mark.xfail(reason=KNOWN[key], strict=True)] if key in KNOWN else []
                    cases.append(pytest.param(row["contact"], row["eps_printed"], col, row[col],
                                              id="-".join(key), marks=marks))
    return cases


# ---- C1, C2: Harris Tables 7.4 and 7.1 --------------------------------------------------------------------

@pytest.mark.parametrize("contact, eps, column, printed", _table_cases())
def test_harris_tables(contact, eps, column, printed):
    """Every printed J_r, J_a and F_r tan(alpha)/F_a to half a unit of its last digit."""
    J = sjovall_integrals(_eps(eps), n=N_OF[contact])
    assert abs(J[column] - float(printed)) <= _half_unit(printed)


def test_line_ratio_column():
    """The line ratio column of Table 7.4 is the ratio of the printed (rounded) J_r and J_a (OPEN R5-04)."""
    with open(PUBLISHED / "harris2007_vol1_table7_4.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["contact"] == "line" and r["eps_printed"] not in ("0", "inf")]
    for r in rows:
        assert abs(float(r["J_r"]) / float(r["J_a"]) - float(r["Fr_tan_alpha_over_Fa"])) <= 0.5e-4, r


def test_harris_tables_record_bound():
    """No printed J is off by more than 2 units of the last digit: the misses are rounding, not misprints."""
    for table in ("7_4", "7_1"):
        with open(PUBLISHED / f"harris2007_vol1_table{table}.csv", newline="") as f:
            for r in csv.DictReader(f):
                if r["eps_printed"] == "0":
                    continue
                J = sjovall_integrals(_eps(r["eps_printed"]), n=N_OF[r["contact"]])
                for col in ("J_r", "J_a") if table == "7_4" else ("J_r",):
                    assert abs(J[col] - float(r[col])) <= 2e-4, (table, r)


def test_one_element_limit():
    """eps -> 0 rows ("1/Z", ratio 1): only the element at psi = 0 touches (discrete limit, OPEN R5-02)."""
    Z = 8
    psi = 2.0 * math.pi * np.arange(Z) / Z
    Pd = 10e-6
    radial = ring_load_distribution(Pd / 2.0 + 1e-9, psi_rad=psi, K_n=1e10, clearance_m=Pd)   # Table 7.1 eps = 0
    assert np.count_nonzero(radial["Q_N"]) == 1
    assert radial["Fr_N"] / (Z * radial["Q_max_N"]) == pytest.approx(1.0 / Z, rel=1e-12)
    alpha = math.radians(30.0)                                                               # Table 7.4 eps = 0
    dr = 1e-6
    combined = ring_load_distribution(dr, -dr / math.tan(alpha) + 1e-10, psi_rad=psi, K_n=1e10, alpha_rad=alpha)
    assert np.count_nonzero(combined["Q_N"]) == 1
    assert combined["Fr_N"] * math.tan(alpha) / combined["Fa_N"] == pytest.approx(1.0, rel=1e-12)
    assert combined["Fa_N"] / (Z * combined["Q_max_N"] * math.sin(alpha)) == pytest.approx(1.0 / Z, rel=1e-12)
    assert sjovall_integrals(0.0)["J_r"] == 0.0              # the continuous limit is 0, not 1/Z


# ---- C3: Stribeck ------------------------------------------------------------------------------------------

def test_stribeck_constants():
    """Q_max = 4.37 F_r/(Z cos a) for balls (Eq. 7.23) and 4.08 for rollers (Eq. 7.25), zero clearance."""
    assert abs(stribeck_constant(n=N_POINT) - 4.37) <= 0.005
    assert abs(stribeck_constant(n=N_LINE) - 4.08) <= 0.005
    assert abs(stribeck_constant(n=10.0 / 9.0) - 4.08) <= 0.005      # Eq. 7.4's n gives the same 4.08


def test_solve_radial_zero_clearance_is_stribeck():
    """The continuous radial solve at zero clearance loads half the ring (eps = 0.5) with Stribeck's Q_max."""
    Z, Fr = 12, 5000.0
    psi = 2.0 * math.pi * np.arange(Z) / Z
    out = solve_ring_displacement(Fr, psi_rad=psi, K_n=8e9, continuous=True)
    assert out["converged"]
    assert out["epsilon"] == pytest.approx(0.5, abs=1e-12)
    assert out["psi_l_rad"] == pytest.approx(math.pi / 2.0, abs=1e-9)
    assert out["Q_max_N"] * Z / Fr == pytest.approx(stribeck_constant(), rel=1e-10)


# ---- C4: closed forms and limits of the integrals ----------------------------------------------------------

@pytest.mark.parametrize("eps", [0.05, 0.3, 0.5, 0.9, 1.0, 1.7, 40.0])
def test_integrals_linear_spring_closed_form(eps):
    """n = 1 (linear springs) integrates in closed form: the quadrature against it."""
    if eps < 1.0:
        pl = math.acos(1.0 - 2.0 * eps)
        c0 = 1.0 - 1.0 / (2.0 * eps)             # [1 - (1 - cos)/(2 eps)] = c0 + cos/(2 eps)
        J_a = (c0 * pl + math.sin(pl) / (2.0 * eps)) / math.pi
        J_r = (c0 * math.sin(pl) + (pl + math.sin(pl) * math.cos(pl)) / (4.0 * eps)) / math.pi
    else:
        J_a, J_r = 1.0 - 1.0 / (2.0 * eps), 1.0 / (4.0 * eps)
    J = sjovall_integrals(eps, n=1.0)
    assert J["J_a"] == pytest.approx(J_a, rel=1e-11)
    assert J["J_r"] == pytest.approx(J_r, rel=1e-11)


@pytest.mark.parametrize("n", [N_POINT, N_LINE])
def test_integral_limits(n):
    """eps -> inf: J_r ~ n/(4 eps), J_a ~ 1 - n/(2 eps) (first order in 1/eps); eps -> 0: J_r, J_a -> 0."""
    eps = 1e6
    J = sjovall_integrals(eps, n=n)
    assert J["J_r"] * eps == pytest.approx(n / 4.0, rel=1e-5)
    assert (1.0 - J["J_a"]) * eps == pytest.approx(n / 2.0, rel=1e-5)
    assert sjovall_integrals(math.inf, n=n) == {"J_r": 0.0, "J_a": 1.0, "psi_l_rad": math.pi,
                                                "Fr_tan_alpha_over_Fa": 0.0}
    small = [sjovall_integrals(e, n=n) for e in (1e-2, 1e-4, 1e-6)]
    assert small[0]["J_r"] > small[1]["J_r"] > small[2]["J_r"] > 0.0
    assert small[2]["J_r"] < 1e-3 and small[2]["Fr_tan_alpha_over_Fa"] == pytest.approx(1.0, abs=1e-5)


def test_line_vs_point():
    """A softer exponent spreads the load: J_a(line) > J_a(point) everywhere; J_r(line) > J_r(point) for a
    partial zone up to eps = 0.9, and the Stribeck constant of a line is the smaller."""
    for eps in (0.1, 0.3, 0.5, 0.7, 0.9, 1.25, 5.0):
        line, point = sjovall_integrals(eps, n=N_LINE), sjovall_integrals(eps, n=N_POINT)
        assert line["J_a"] > point["J_a"]
        if eps <= 0.9:
            assert line["J_r"] > point["J_r"]
    assert stribeck_constant(n=N_LINE) < stribeck_constant(n=N_POINT)


# ---- C5: load zone geometry (Eqs 7.10-7.13, 7.58-7.65) and the force identities ---------------------------

def test_clearance_load_zone():
    """eps = (1 - P_d/(2 delta_r))/2 (Eq. 7.12), psi_l = arccos(P_d/(2 delta_r)) (Eq. 7.13), no load beyond
    psi_l, and F_r = Z K_n (delta_r - P_d/2)^n J_r(eps) (Eq. 7.22); interference gives eps > 0.5."""
    Z, K_n = 15, 1.2e10
    psi = 2.0 * math.pi * np.arange(Z) / Z
    for Pd, dr in ((15e-6, 40e-6), (-8e-6, 20e-6)):
        out = ring_load_distribution(dr, psi_rad=psi, K_n=K_n, clearance_m=Pd)
        assert out["epsilon"] == pytest.approx(0.5 * (1.0 - Pd / (2.0 * dr)), rel=1e-14)
        assert out["psi_l_rad"] == pytest.approx(math.acos(Pd / (2.0 * dr)), rel=1e-12)
        wrapped = np.abs(np.angle(np.exp(1j * psi)))
        assert np.all(out["Q_N"][wrapped > out["psi_l_rad"] + 1e-12] == 0.0)
        assert out["Fr_continuous_N"] == pytest.approx(Z * K_n * (dr - Pd / 2.0) ** 1.5 * out["J_r"], rel=1e-14)
        assert (out["epsilon"] > 0.5) == (Pd < 0.0)


def test_combined_load_zone():
    """eps = (1 + delta_a tan(a)/delta_r)/2 (Eq. 7.61), psi_l = arccos(-delta_a tan(a)/delta_r) (Eq. 7.65)."""
    alpha, dr, da = math.radians(40.0), 10e-6, -3e-6
    out = ring_load_distribution(dr, da, psi_rad=np.linspace(0.0, 2.0 * math.pi, 7, endpoint=False), K_n=1e10,
                                 alpha_rad=alpha)
    assert out["epsilon"] == pytest.approx(0.5 * (1.0 + da * math.tan(alpha) / dr), rel=1e-13)
    assert out["psi_l_rad"] == pytest.approx(math.acos(-da * math.tan(alpha) / dr), rel=1e-12)


def test_solve_combined_continuous():
    """Zero clearance, combined load: eps from F_r tan(a)/F_a = J_r/J_a (Table 7.4); Q_max from Eq. 7.71 and
    from Eq. 7.66 with cos(a) (Eq. 7.70 prints sin(a), OPEN R5-05)."""
    Z, alpha, Fr, Fa = 16, math.radians(25.0), 3000.0, 4000.0
    out = solve_ring_displacement(Fr, Fa, psi_rad=2.0 * math.pi * np.arange(Z) / Z, K_n=9e9, alpha_rad=alpha,
                                  continuous=True)
    assert out["converged"]
    assert out["J_r"] / out["J_a"] == pytest.approx(Fr * math.tan(alpha) / Fa, rel=1e-10)
    assert out["Q_max_N"] == pytest.approx(Fa / (out["J_a"] * Z * math.sin(alpha)), rel=1e-10)   # Eq. 7.71
    assert out["Q_max_N"] == pytest.approx(Fr / (out["J_r"] * Z * math.cos(alpha)), rel=1e-10)   # Eq. 7.66
    assert out["epsilon"] == pytest.approx(0.5 * (1.0 + out["delta_a_m"] * math.tan(alpha) / out["delta_r_m"]),
                                           rel=1e-12)


@pytest.mark.parametrize("alpha_deg, Fa, Pd, psi0", [(0.0, 0.0, 15e-6, 0.0), (0.0, 0.0, 15e-6, math.pi / 9.0),
                                                     (0.0, 0.0, -5e-6, 0.1), (30.0, 6000.0, 0.0, 0.0),
                                                     (30.0, 6000.0, 4e-6, 0.2)])
def test_solve_discrete_balance(alpha_deg, Fa, Pd, psi0):
    """Element by element: the sums balance the applied loads and the loads are Q = K_n delta^n."""
    Z, K_n, Fr = 9, 7e9, 8900.0
    psi = psi0 + 2.0 * math.pi * np.arange(Z) / Z
    out = solve_ring_displacement(Fr, Fa, psi_rad=psi, K_n=K_n, alpha_rad=math.radians(alpha_deg), clearance_m=Pd)
    assert out["converged"]
    assert abs(out["residual_Fr_N"]) <= 1e-9 * Fr and abs(out["residual_Fa_N"]) <= 1e-9 * max(Fa, Fr)
    assert out["Q_N"] == pytest.approx(K_n * np.clip(out["delta_psi_m"], 0.0, None) ** 1.5, rel=1e-14)


def test_discrete_tends_to_continuous():
    """Many elements: the element sums tend to the Sjovall integrals (Harris's continuous tables)."""
    Z, Fr, Pd = 360, 5e4, 10e-6
    psi = 2.0 * math.pi * np.arange(Z) / Z
    discrete = solve_ring_displacement(Fr, psi_rad=psi, K_n=5e9, clearance_m=Pd)
    continuous = solve_ring_displacement(Fr, psi_rad=psi, K_n=5e9, clearance_m=Pd, continuous=True)
    assert discrete["Q_max_N"] == pytest.approx(continuous["Q_max_N"], rel=1e-4)
    assert discrete["delta_r_m"] == pytest.approx(continuous["delta_r_m"], rel=1e-4)


# ---- C6: the series of two contacts (Eqs 7.5-7.7) -----------------------------------------------------------

def test_hertz_series():
    """delta_i + delta_o = (Q/K_n)^(1/n) (Eqs 7.5-7.7); equal contacts give K/2^n; delta_r = delta_n + P_d/2."""
    K_i, K_o, Q = 1.1e10, 1.4e10, np.array([0.0, 120.0, 4513.0])
    for n in (N_POINT, N_LINE):
        out = ball_deflection(Q, K_i=K_i, K_o=K_o, n=n, clearance_m=15e-6)
        assert out["delta_n_m"] == pytest.approx((Q / out["K_n_N_per_m_pow_n"]) ** (1.0 / n), rel=1e-14)
        assert out["delta_r_m"] == pytest.approx(out["delta_n_m"] + 7.5e-6, rel=1e-14)
        assert ball_deflection(1.0, K_i=K_i, K_o=K_i, n=n)["K_n_N_per_m_pow_n"] == pytest.approx(K_i / 2.0**n)


def test_series_drives_load_zone():
    """The solved Q_max through ball_deflection gives back the solved delta_r (radial, clearance; s004 use).
    K_i, K_o: the order of a 12.7 mm ball on a 65 mm pitch diameter, groove 0.52 d (Hertz, about 3.1e10 and
    3.3e10 N/m^1.5)."""
    Ki, Ko = 3.1e10, 3.3e10
    K_n = ball_deflection(1.0, K_i=Ki, K_o=Ko)["K_n_N_per_m_pow_n"]
    out = solve_ring_displacement(8900.0, psi_rad=2.0 * math.pi * np.arange(9) / 9, K_n=K_n, clearance_m=15e-6)
    back = ball_deflection(out["Q_max_N"], K_i=Ki, K_o=Ko, clearance_m=15e-6)
    assert back["delta_r_m"] == pytest.approx(out["delta_r_m"], rel=1e-12)


# ---- C7: domain errors -------------------------------------------------------------------------------------

def test_bad_inputs():
    psi = np.zeros(1)
    with pytest.raises(ValueError):
        sjovall_integrals(-0.1)
    with pytest.raises(ValueError):
        sjovall_integrals(math.nan)
    with pytest.raises(ValueError):
        sjovall_integrals(0.5, n=0.9)
    with pytest.raises(ValueError):
        ball_deflection(-1.0, K_i=1e10, K_o=1e10)
    with pytest.raises(ValueError):
        ball_deflection(1.0, K_i=0.0, K_o=1e10)
    with pytest.raises(ValueError):
        ring_load_distribution(-1e-6, psi_rad=psi, K_n=1e10)
    with pytest.raises(ValueError):
        ring_load_distribution(1e-6, psi_rad=[], K_n=1e10)
    with pytest.raises(ValueError):
        ring_load_distribution(1e-6, psi_rad=psi, K_n=1e10, alpha_rad=math.pi / 2.0)
    with pytest.raises(ValueError):
        solve_ring_displacement(100.0, 50.0, psi_rad=psi, K_n=1e10)                       # Fa at alpha = 0
    with pytest.raises(ValueError):
        solve_ring_displacement(100.0, 0.0, psi_rad=psi, K_n=1e10, alpha_rad=0.3)         # alpha > 0, no Fa
    with pytest.raises(ValueError):
        solve_ring_displacement(0.0, psi_rad=psi, K_n=1e10)
    alpha = math.radians(30.0)                                   # Fr tan(a)/Fa = 1.2: beyond one element
    with pytest.raises(ValueError, match="single-row limit"):
        solve_ring_displacement(1.2 * 1000.0 / math.tan(alpha), 1000.0,
                                psi_rad=2.0 * math.pi * np.arange(9) / 9, K_n=1e10, alpha_rad=alpha)


@pytest.mark.parametrize("continuous", [False, True])
def test_just_beyond_single_row_limit(continuous):
    """F_r tan(a)/F_a = 1.05, just above the one-element limit 1: no displacement balances it, the solve
    raises instead of returning a roundoff root at a huge delta_r (bearing-expert review, MUST)."""
    alpha, Fa = math.radians(40.0), 4000.0
    with pytest.raises(ValueError, match="single-row limit"):
        solve_ring_displacement(1.05 * Fa / math.tan(alpha), Fa, psi_rad=2.0 * math.pi * np.arange(13) / 13,
                                K_n=1e10, alpha_rad=alpha, continuous=continuous)
