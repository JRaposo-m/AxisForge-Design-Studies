"""r04_involute against the formula columns of Fernandes 2015b Table 1 and Martins 2015 Tables 3-4 (to half a
unit of the last printed digit), ISO 21771 identities, and the printed values that do not follow (listed)."""
import csv
import math
from pathlib import Path

import pytest

from studies.r00_analytical_references.r04_involute import inv, inv_inverse, mesh_forces, pair_geometry, path_point

PUBLISHED = Path(__file__).resolve().parents[1] / "published" / "fernandes2015b_t1_martins2015_t3_t4.csv"
with PUBLISHED.open(encoding="utf-8", newline="") as _f:
    ROWS = list(csv.DictReader(_f))


def _printed(source, gear, symbol, member=None, stage=None):
    """The printed string of one value (``member`` pinion / wheel; ``stage`` K1..K9)."""
    out = [r["value"] for r in ROWS if r["source_key"] == source and r["symbol"] == symbol
           and f"gear={gear}" in r["condition"] and (member is None or member in r["condition"])
           and (stage is None or f"stage={stage};" in r["condition"] + ";")]
    assert len(out) == 1, (source, gear, symbol, member, stage, out)
    return out[0]


def _half_unit(printed: str) -> float:
    """Half a unit of the last printed digit [same unit]."""
    s = printed.lstrip("+")
    return 0.5 * 10.0 ** -(len(s.split(".")[1]) if "." in s else 0)


def _pair(gear, source="Fernandes2015b", **tips):
    member = ("pinion", "gear (wheel)") if source == "Fernandes2015b" else ("pinion", "wheel")
    x = "x_z" if source == "Fernandes2015b" else "x"
    f = lambda sym, m=None: float(_printed(source, gear, sym, m))       # noqa: E731
    return pair_geometry(int(f("z", member[0])), int(f("z", member[1])), f("m"), f("α"), f("β"),
                         f(x, member[0]), f(x, member[1]), f("b"), **tips)


GEARS = ("C40", "H501", "H951", "Transfer 1", "Transfer 2", "m=8", "m=10")


@pytest.mark.parametrize("gear", GEARS)
def test_centre_distance_from_the_profile_shifts_is_the_printed_one(gear):
    """ISO 21771 Eq. (55) (zero backlash) and Eqs (56)-(58): a_w from x1 + x2 against the printed a."""
    printed = _printed("Fernandes2015b", gear, "a")
    assert abs(_pair(gear).a_w - float(printed)) <= _half_unit(printed)


@pytest.mark.parametrize("gear", [g for g in GEARS if g != "m=8"])     # m=8: eps_beta not extracted
def test_overlap_ratio(gear):
    """ISO 21771 Eq. (93): eps_beta = b sin(beta) / (pi m_n)."""
    printed = _printed("Fernandes2015b", gear, "ε_β")
    assert abs(_pair(gear).eps_beta - float(printed)) <= _half_unit(printed)


@pytest.mark.parametrize("gear", ["H951", "Transfer 1", "Transfer 2", "m=10"])
def test_transverse_contact_ratio_from_the_printed_tips(gear):
    """ISO 21771 Eqs (77), (90) with the printed tip diameters (tip alteration included in them)."""
    member = ("pinion", "gear (wheel)")
    d_a = [float(_printed("Fernandes2015b", gear, "d_a", m)) for m in member]
    printed = _printed("Fernandes2015b", gear, "ε_α")
    assert abs(_pair(gear, d_a1=d_a[0], d_a2=d_a[1]).eps_alpha - float(printed)) <= _half_unit(printed)


def test_printed_values_that_do_not_follow_are_recorded():
    """Listed, not judged (README section 6): C40 eps_alpha 1.4625 from its tips (printed 1.44, C3 item 3);
    H501 1.4631 (printed 1.45); Fernandes C40 wheel d_a 115.54 is a misprint of 118.54 (Martins Table 3, and
    the tip from x); the tips of H501, H951, Transfer 1-2, m=10 include a tip alteration (not from x alone)."""
    c40 = _pair("C40", d_a1=float(_printed("Martins2015", "C40", "d_a", "pinion")),
                d_a2=float(_printed("Martins2015", "C40", "d_a", "wheel")))
    assert round(c40.eps_alpha, 4) == 1.4625 and _printed("Fernandes2015b", "C40", "ε_α") == "1.44"
    h501 = _pair("H501", d_a1=float(_printed("Fernandes2015b", "H501", "d_a", "pinion")),
                 d_a2=float(_printed("Fernandes2015b", "H501", "d_a", "gear (wheel)")))
    assert round(h501.eps_alpha, 4) == 1.4631
    assert _printed("Fernandes2015b", "C40", "d_a", "gear (wheel)") == "115.54"


@pytest.mark.parametrize("member,source", [("pinion", "Fernandes2015b"), ("wheel", "Martins2015")])
def test_c40_tips_from_the_profile_shifts(member, source):
    """ISO 21771 Eq. (33) with h_aP = m_n, k = 0: the C40 tips (no tip alteration)."""
    g = _pair("C40")
    printed = _printed(source, "C40", "d_a", member)
    assert abs((g.d_a1 if member == "pinion" else g.d_a2) - float(printed)) <= _half_unit(printed)


STAGES = ("K1", "K5", "K7", "K9")


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("gear,label", [("C40", "Spur C40"), ("H501", "Helical H501/951")])
def test_mesh_forces_of_martins_table4(stage, gear, label):
    """Martins 2015 Table 4: F_r (and F_a) from the wheel torque at the working pitch circle."""
    torque = float(next(r["value"] for r in ROWS if r["symbol"] == "-" and f"stage={stage}" in r["condition"]
                        and r["quantity"] == "wheel torque"))
    f = mesh_forces(_pair(gear, source="Martins2015"), torque, on=2)
    for symbol, value in (("Fr", f.F_r), ("Fa", f.F_a)):
        printed = next(r["value"] for r in ROWS if r["symbol"] == symbol and f"stage={stage};gear={label}" in r["condition"])
        # tolerance: half a unit of the printed force plus the printed torque's half unit carried through
        tol = _half_unit(printed) + value * 0.005 / torque
        if (gear, stage, symbol) == ("H501", "K7", "Fa"):        # recorded: 0.53 N off, 0.005 N past the tolerance
            assert tol < abs(value - float(printed)) < 0.54, (symbol, value, printed)
            continue
        assert abs(value - float(printed)) <= tol, (symbol, value, printed)


def test_involute_inverse_and_identities():
    # gear range; below about 1e-3 rad tan(a) - a cancels in floating point (not a gear angle)
    for alpha in (0.01, 0.1, math.radians(20.0), 1.2):
        assert math.isclose(inv_inverse(inv(alpha)), alpha, rel_tol=1e-10)   # P-12, an iterated inverse
    g = _pair("H501")
    # ISO 21771 Eq. (87): T1T2 = rho_A1 + rho_A2; Eq. (81): rho_C1 = d_b1 tan(alpha_wt) / 2; Eqs (85)-(86)
    assert math.isclose(g.T1T2, g.a_w * math.sin(g.alpha_wt_rad), rel_tol=1e-14)
    assert math.isclose(g.T1C, math.sqrt(g.d_w1**2 - g.d_b1**2) / 2.0, rel_tol=1e-12)
    assert math.isclose(g.T1E - g.T1B, g.p_bt) and math.isclose(g.T1D - g.T1A, g.p_bt)
    # the pitch point: no sliding (ISO 21771 Eqs (114)-(115) with u = z2/z1)
    c = path_point(g, g.T1C)
    assert abs(c.zeta1) < 1e-12 and abs(c.zeta2) < 1e-12 and c.on_path and c.single_pair
    assert math.isclose(c.rho1_t + c.rho2_t, g.T1T2)
    a = path_point(g, g.T1A)
    assert math.isclose(a.zeta1, 1.0 - a.rho2_t / ((g.z2 / g.z1) * a.rho1_t))
    # normal-plane radii: rho_t / cos(beta_b) for both flanks and the reduced radius (R2-11: convention)
    assert math.isclose(c.rho_red_n * math.cos(g.beta_b_rad), c.rho1_t * c.rho2_t / (c.rho1_t + c.rho2_t))
    # the couple of the axial force about each axis: F_a r_w
    f = mesh_forces(g, 100.0)
    assert math.isclose(f.couple_2, f.F_a * g.d_w2 / 2000.0) and math.isclose(f.couple_1 / f.couple_2, g.z1 / g.z2)
    with pytest.raises(ValueError):
        mesh_forces(g, 100.0, on=3)
    # beta_b two ways (ISO 21771 Eqs (5), (6))
    assert math.isclose(math.sin(g.beta_b_rad), math.sin(g.beta_rad) * math.cos(g.alpha_n_rad), rel_tol=1e-13)
