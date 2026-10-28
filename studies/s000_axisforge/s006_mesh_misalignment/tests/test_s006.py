"""Fast tests of s006: the stage, the projection on the plane of action, the criteria on a few points."""
import math

import numpy as np
import pytest

from studies.s000_axisforge.s006_mesh_misalignment import line_of_action, mesh_misalignment, stage
from studies.s000_axisforge.s006_mesh_misalignment.study import point


def test_line_of_action_is_the_transverse_force_on_the_wheel():
    """Its angle to the tangent e_t is the working pressure angle alpha_wt (spur: |l . e_c| = sin alpha_wt)."""
    resolved = stage(span_mm=200.0, position="offset")
    mesh = resolved.meshes[0]
    l = line_of_action(resolved.shafts[1])
    pair_alpha_wt = math.atan(mesh.F_r_N / mesh.F_tw_N)
    assert np.linalg.norm(l) == pytest.approx(1.0) and abs(l[0]) == pytest.approx(math.sin(pair_alpha_wt), rel=1e-12)


def test_misalignment_projection_and_sign():
    """A wheel slope along l of 1e-4 rad over b = 23 mm is +2.3 um; across the plane of action it is f_perp."""
    l = np.array([0.6, 0.8])
    f = mesh_misalignment((0.0, 0.0), (0.6e-4, 0.8e-4), (0.0, 0.0), (0.0, 0.0), l, 23.0)
    assert f["f_sh_bend_um"] == pytest.approx(2.3) and f["f_perp_um"] == pytest.approx(0.0, abs=1e-12)
    f = mesh_misalignment((0.0, 0.0), (-0.8e-4, 0.6e-4), (0.0, 0.0), (0.0, 0.0), l, 23.0)
    assert f["f_sh_bend_um"] == pytest.approx(0.0, abs=1e-12) and f["f_perp_um"] == pytest.approx(2.3)


@pytest.mark.parametrize("position, driver", [("centred", "pinion"), ("offset", "wheel"), ("overhung", "pinion")])
def test_spur_point_meets_s1_s2_s4(position, driver):
    p = point(span_mm=150.0, position=position, driver=driver)
    assert all(r["verdict"] == "PASS" for r in p["v002"] if r["verdict"] != "RECORD")
    assert all(r["verdict"] == "PASS" for r in p["identities"] if r["verdict"] != "RECORD")
    assert p["point"]["S4"] == "PASS"


def test_centred_gear_has_no_misalignment():
    """Symmetric span, same load on both shafts' mirror: slope 0 at mid-span, f_sh_bend = 0 (limit case)."""
    assert point(span_mm=300.0, position="centred")["point"]["f_sh_bend_um"] == pytest.approx(0.0, abs=1e-9)


def test_helical_couple_shifts_the_reactions_by_c_over_span():
    p = point(span_mm=200.0, position="offset", beta_deg=30.0, hand=-1)["point"]
    assert p["S3"] == "PASS" and p["tag"] == "convention-dependent (J-16)"


def test_driver_mirrors_the_plane_of_action_and_keeps_f_sh_bend():
    """Pinion or wheel driving at the same pinion sense: l mirrors about e_c, the loads mirror, f_sh_bend is the same."""
    a, b = (point(span_mm=250.0, position="offset", driver=d)["point"] for d in ("pinion", "wheel"))
    assert b["l_z"] == pytest.approx(-a["l_z"]) and b["f_sh_bend_um"] == pytest.approx(a["f_sh_bend_um"], rel=1e-9)


def test_offset_spur_misalignment_equals_the_simply_supported_slope():
    """Spur, pins at 0 and L, gear at a = 0.3 L (b' = L - a): each shaft carries F_bt along -l / +l, so
    theta(a) = F a b' (b' - a) / (3 L EI) along l on each (beam on two supports, point load at a), and
    f_sh_bend = 2 b F_bt a b' (b' - a) / (3 L EI) > 0 (the separation grows toward mid-span)."""
    L, d = 200.0, 40.0
    p = point(span_mm=L, position="offset")
    mesh = stage(span_mm=L, position="offset").meshes[0]
    F_bt = math.hypot(mesh.F_tw_N, mesh.F_r_N)
    EI = 210000.0 * math.pi * d**4 / 64.0
    a, bb = 0.3 * L, 0.7 * L
    assert p["point"]["f_sh_bend_um"] == pytest.approx(1e3 * 23.0 * 2 * F_bt * a * bb * (bb - a) / (3 * L * EI), rel=1e-10)
