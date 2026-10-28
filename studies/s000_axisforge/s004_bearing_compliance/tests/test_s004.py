"""Fast tests of s004: the rigid-body line, the fixed point, its limits and identities."""
import numpy as np
import pytest

from studies.s000_axisforge.s004_bearing_compliance import bearings, coupled_tilt, rigid_line, shaft
from studies.s000_axisforge.s004_bearing_compliance.study import EB, N_ELEMENTS, point, settlements, solve


def test_rigid_line_passes_through_the_displaced_centres():
    v, rot = rigid_line([0.0, 100.0, 200.0], 0.0, 200.0, (0.01, -0.02), (0.03, 0.02))
    assert v[0] == pytest.approx([0.01, -0.02]) and v[2] == pytest.approx([0.03, 0.02])
    assert rot == pytest.approx([1e-4, 2e-4])


def test_equal_bearings_and_centred_gear_give_no_rotation():
    """Limit: equal reactions on equal bearings move both centres alike, psi_coupled = psi_rigid."""
    ss = shaft(load_factor=0.5, position="centred")
    c = coupled_tilt(ss, solve(ss, EB, N_ELEMENTS), bearings(0.02, "ball"))
    assert c["converged"] and np.max(np.abs(c["rotation_yz"])) < 1e-12


def test_fixed_point_converges_and_the_shaft_moves_along_the_force_on_the_bearing():
    ss = shaft(load_factor=1.0)
    rigid = solve(ss, EB, N_ELEMENTS)
    c = coupled_tilt(ss, rigid, bearings(0.0, "ball"))
    assert c["converged"] and c["passes"] <= 5
    R_A = np.array([rigid.R_xy[0], rigid.R_xz[0]])
    assert float(c["d_yz"][0] @ R_A) < 0.0          # opposite to the support force on the shaft


def test_coupled_tilt_refuses_a_hyperstatic_shaft():
    """Scope guard: the reactions are held fixed, valid on two pins only."""
    ss = shaft(load_factor=0.5)
    with pytest.raises(ValueError, match="two pins only"):
        coupled_tilt(ss, solve(ss, EB, N_ELEMENTS), bearings(0.02, "ball") * 2)


def test_offset_gear_moves_the_tilt_from_the_heavier_pin_to_the_lighter():
    """Physical direction: A (0.7 of the load) yields more, the rigid rotation lowers psi_A and raises psi_B."""
    p = point(floating="ball", s_key="C3-like", load_factor=1.0, position="offset")
    ratio = {r["pin"]: r["ratio"] for r in p["rows"]}
    assert ratio["A"] < 1.0 < ratio["B"]
    assert all(r["ratio_closed_form"] == pytest.approx(r["ratio"], abs=0.01) for r in p["rows"])


def test_point_meets_the_identities():
    """S1 (v002.check of rigid and settled FEM), the settled slope at the pins = the fixed point's psi."""
    p = point(floating="roller", s_key="C3-like", load_factor=0.5, position="offset")
    assert all(r["verdict"] == "PASS" for r in p["v002"] if r["verdict"] != "RECORD")
    assert all(r["settled_theta_err_rad"] < 1e-12 for r in p["rows"])
    assert set(settlements(p["coupled"])) == {"XY", "XZ"}
