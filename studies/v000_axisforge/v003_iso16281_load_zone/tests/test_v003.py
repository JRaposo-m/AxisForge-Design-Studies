"""Fast tests of v003: the systems, the comparator on single bearings, the literature, the sign check."""
import math

import numpy as np
import pytest

from studies.v000_axisforge.v003_iso16281_load_zone import (angular_contact_7210, check, cylindrical_nu210,
                                                            deep_groove_6210, elements, hertz_series_cp, literature,
                                                            r03_inputs, sign_check_shaft)
from studies.v000_axisforge.v003_iso16281_load_zone.study import literature_series, sign_check, solve_bearing


def test_the_systems_are_returned_unsolved():
    b = deep_groove_6210(0.02)
    assert (b.Z, b.Dw, b.Dpw, b.s) == (9, 12.7, 70.0, 0.02)
    # diametral clearance: alpha_0 = arccos(1 - s/(2A)), A = (0.52 + 0.52 - 1) Dw (ISO/TS 16281 4.2.2)
    assert b.alpha_0 == pytest.approx(math.acos(1.0 - 0.02 / (2.0 * 0.04 * 12.7)), rel=1e-12)
    system = sign_check_shaft(270.0)
    assert [x.label for x in system.bearings] == ["6210_A", "6210_B"] and len(system.radial_loads) == 1


def test_r03_inputs_convert_once():
    b = deep_groove_6210(0.03)
    inp = r03_inputs(b)
    assert inp["K_n"] == pytest.approx(b.cp * 1e3**1.5) and inp["clearance_m"] == pytest.approx(3e-5)
    assert inp["alpha_rad"] == 0.0 and inp["n"] == 1.5
    assert r03_inputs(angular_contact_7210())["clearance_m"] == 0.0           # its s is derived, not a clearance
    assert r03_inputs(cylindrical_nu210())["n"] == pytest.approx(10.0 / 9.0)


@pytest.mark.parametrize("s_mm, Fr", [(0.0, 5000.0), (0.02, 1000.0), (0.06, 10000.0)])
def test_check_passes_on_the_deep_groove_bearing(s_mm, Fr):
    b = deep_groove_6210(s_mm)
    rows = check(b, solve_bearing(b, Fr))
    assert [r["id"] for r in rows] == ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    assert all(r["verdict"] == "PASS" for r in rows), rows


def test_zero_clearance_loads_half_the_ring():
    """s = 0: the discrete Stribeck number of 9 balls, 9 / sum(cos^2.5 psi_j), about 4.38 (eps = 0.5 holds by
    construction at s = 0, C5 restates C4: not asserted)."""
    b = deep_groove_6210(0.0)
    r = solve_bearing(b, 5000.0)
    psi = 2.0 * math.pi * np.arange(9) / 9
    stribeck_9 = 9.0 / np.sum(np.clip(np.cos(psi), 0.0, None) ** 2.5)
    assert max(r.rows[0].Q_j) * 9 / 5000.0 == pytest.approx(stribeck_9, rel=1e-5)


def test_elements_carry_both_judges():
    b = deep_groove_6210(0.02)
    rows = elements(b, solve_bearing(b, 5000.0))
    assert len(rows) == 9 and rows[0]["phi_j_rad"] == 0.0
    for r in rows:
        assert r["Q_r03_solve_N"] == pytest.approx(r["Q_axisforge_N"], rel=1e-5, abs=1e-6)


def test_record_families_are_records():
    for b, Fa in ((angular_contact_7210(), 4000.0), (cylindrical_nu210(), 0.0)):
        rows = check(b, solve_bearing(b, 3000.0, Fa))
        assert {r["verdict"] for r in rows if r["id"] != "C7"} == {"RECORD"}


def test_cp_against_the_hertz_series():
    b = deep_groove_6210(0.0)
    assert hertz_series_cp(b) == pytest.approx(b.cp, rel=5e-3)


def test_literature():
    """Hamrock within 1 % / 2 % (L1, L2); Hernot reproduced by r03 to the printed digits (L4-L7)."""
    assert [c.name for c in literature.cases()] == ["hamrock1983_sec14_2_dgbb", "hernot2000_218_acbb"]
    rows, _ = literature_series()
    assert {r["id"]: r["verdict"] for r in rows} == {"L1": "PASS", "L2": "PASS", "L3": "RECORD", "L4": "PASS",
                                                     "L5": "PASS", "L6": "PASS", "L7": "PASS"}


def test_sign_check_records_r003():
    """RECORD (R-003): the result is recorded, not judged. Today the offset is 180 deg and |r| = 2 F_r (evidence on
    R-003 / R-396); if AxisForge changes its convention this test fails and the README must be updated."""
    rows = sign_check()
    assert len(rows) == 4 and all(r["verdict"] == "RECORD" and r["solver_ok"] for r in rows)
    for r in rows:
        assert abs(abs(r["offset_deg"]) - 180.0) < 1e-6 and r["residual_over_Fr"] == pytest.approx(2.0, rel=1e-9)
