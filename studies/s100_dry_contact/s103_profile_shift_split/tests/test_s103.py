"""Fast tests of s103: the split keeps the sum and a_w, the bounds, equal sliding through v107, Hertz vs v107."""
import pytest

from studies.s100_dry_contact.s103_profile_shift_split import (as_cut, bounds, split_pair, split_point, tip_thickness,
                                                               x_sum)
from studies.s100_dry_contact.s103_profile_shift_split.construction import PAIRS, STAGE, WHEEL_TORQUE_NM
from studies.s100_dry_contact.s103_profile_shift_split.study import sweep

T = WHEEL_TORQUE_NM[STAGE]


@pytest.mark.parametrize("name", list(PAIRS))
def test_a_split_keeps_the_sum_the_centre_distance_and_the_force(name):
    a, b = split_pair(name), split_pair(name, 0.5 * x_sum(name))
    assert b.x1 + b.x2 == pytest.approx(x_sum(name), abs=1e-15)
    assert b.al == pytest.approx(a.al, rel=1e-14) and b.T1C == pytest.approx(a.T1C, rel=1e-14)
    assert split_point(b, T)[0]["F_bn_N"] == pytest.approx(split_point(a, T)[0]["F_bn_N"], rel=1e-14)


def test_bounds_of_c40_ideal_and_as_cut():
    b = bounds("C40")
    # ideal flanks: involute interference below, pointed pinion tip above
    assert (b["lo_by"], b["hi_by"]) == ("T1A = 0", "s_a1 = 0")
    assert split_pair("C40", b["x1_lo"], check=False).T1A == pytest.approx(0.0, abs=1e-9)
    assert tip_thickness(split_pair("C40", b["x1_hi"], check=False).gear1) == pytest.approx(0.0, abs=1e-9)
    # as cut: undercut of the pinion below (x_Emin 0.0641, ISO 21771 Eq. (135)) and of the wheel above
    assert (b["lo_cut_by"], b["hi_cut_by"]) == ("x1 = x_Emin1", "x2 = x_Emin2")
    assert b["x1_lo_cut"] == pytest.approx(0.0641, abs=1e-4) and b["x1_hi_cut"] == pytest.approx(0.7570, abs=1e-4)
    # wheel tip on the pinion fillet, d_Nf1 = d_Ff1 (T1A = T1Ff1, Eq. (128)): 0.961 (physics review hand value)
    (x_f,) = b["roots"]["d_Nf1 = d_Ff1"]
    assert x_f == pytest.approx(0.961, abs=1e-3)
    cut = as_cut(split_pair("C40", x_f, check=False))
    assert cut.d_Nf1 == pytest.approx(cut.gear1.d_Ffs, abs=1e-9)


def test_h501_as_cut_ends_on_the_pinion_fillet():
    b = bounds("H501")
    assert b["hi_cut_by"] == "d_Nf1 = d_Ff1" and b["x1_hi_cut"] < b["x1_hi"]


@pytest.mark.parametrize("name, d_a, eps", [("C40", (82.457, 118.365), 1.4377), ("H501", (80.373, 116.564), 1.4628)])
def test_tip_alteration_gives_feup_printed_tips(name, d_a, eps):
    # FEUP prints C40 82.46 / (115.54, R5-171) eps 1.44; H501 80.37 / 116.57 eps 1.45 (R-306)
    p = split_pair(name, addendum_reduction=True)
    assert (p.gear1.da, p.gear2.da) == pytest.approx(d_a, abs=1e-3)
    assert p.epslon_alpha == pytest.approx(eps, abs=1e-4)


def test_c40_sweep_passes_v107_and_equal_sliding_holds_in_r04_numbers():
    out = sweep("C40")
    assert all(r["verdict"] == "PASS" for r in out["criteria"]), [r for r in out["criteria"] if r["verdict"] != "PASS"]
    assert {r["id"] for r in out["criteria"]} >= {"S0/G", "S0/P", "S0/F", "S1", "S2/A", "S2/C", "S2/B full"}
    ess = next(r for r in out["rows"] if r["tag"] == "equal sliding")
    assert ess["zeta_f1"] == pytest.approx(ess["zeta_f2"], rel=1e-10)
    # zeta_f1 at A rises (less negative) and zeta_f2 at E falls as x1 grows: the split trades the two roots
    rows = sorted((r for r in out["rows"] if r["tag"] == "sweep"), key=lambda r: r["x1"])
    assert all(b["zeta_f1"] > a["zeta_f1"] and b["zeta_f2"] < a["zeta_f2"] for a, b in zip(rows, rows[1:]))
    # the splits outside the as-cut range are kept and tagged
    b = out["bounds"]
    assert all(r["made_valid"] == (b["x1_lo_cut"] <= r["x1"] <= b["x1_hi_cut"]) for r in rows)
