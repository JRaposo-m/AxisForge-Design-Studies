"""Fast tests of s002: the bearings, the 4' limit, the tilt physics, the study criteria."""
import math

import pytest

from studies.s000_axisforge.s001_bearing_clearance_and_contact_angle import load_node, records, solve_bearing
from studies.s000_axisforge.s002_bearing_tilt import PSI_ISO_LIMIT_MRAD, bearings


def _case(bearing, Fr, Fa, psi_mrad):
    a, p, e = solve_bearing(bearing, load_node(bearing, Fr, Fa), psi_rad=1e-3 * psi_mrad)
    return records(bearing, a, postprocessed=p, error=e)[0]


def test_iso_limit_is_four_arc_minutes():
    assert PSI_ISO_LIMIT_MRAD == pytest.approx(1e3 * math.pi / (180 * 15), rel=1e-12)   # 4' = 1/15 deg


def test_bearings_are_s001s_at_20_um():
    b = bearings()
    assert set(b) == {"deep_groove", "cylindrical_roller"}
    assert all(x.s == pytest.approx(0.02) for x in b.values())


def test_radial_ball_floats_axially_by_ri_psi():
    """Fa = 0: the tilt moves delta_a by a fraction of -Ri psi (the loaded balls sit at cos(phi) 0.77..1) and leaves
    Q_max (nearly) unchanged."""
    b = bearings()["deep_groove"]
    c0, c1 = _case(b, 5000.0, 0.0, 0.0), _case(b, 5000.0, 0.0, 1.0)
    assert 0.77 < (c0["delta_a_mm"] - c1["delta_a_mm"]) / (b.Ri * 1e-3) < 1.0
    assert c1["Q_max_N"] == pytest.approx(c0["Q_max_N"], rel=1e-4)
    assert c1["Mz_Nmm"] > 0.0 and math.isnan(c1["epsilon"])          # tilt stiffness > 0; no zone formula when tilted


def test_roller_mz_has_the_opposite_sign_of_the_ball():
    """R5-76: Eq. (46) and Eq. (18) report Mz in opposite senses for the same psi."""
    c = _case(bearings()["cylindrical_roller"], 5000.0, 0.0, 1.0)
    assert c["Mz_Nmm"] < 0.0


def test_study(tmp_path):
    """The whole sweep (about 6 s): T1-T3 PASS, T4 recorded as found, four figures."""
    from studies.s000_axisforge.s002_bearing_tilt import study
    out = study.run(tmp_path)
    v = {c["id"] + c["criterion"][:5]: c["verdict"] for c in out["criteria"]}
    assert [v[k] for k in v if k.startswith(("T1", "T2", "T3"))] == ["PASS"] * 3
    assert len(out["figures"]) == 4 and all(p.is_file() for p in out["figures"])
    assert {c["flag"] for c in out["cases"] if c["psi_in_mrad"] > PSI_ISO_LIMIT_MRAD} == {""}   # balls only: no flag
    assert max(c["psi_in_mrad"] for c in out["cases"] if c["kind"] == "cylindrical_roller") < PSI_ISO_LIMIT_MRAD
