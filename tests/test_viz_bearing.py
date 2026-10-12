"""viz.bearing + viz.scene: the bearing zoom of ``validation/bearings/iso16281_ball_contact``, off-screen."""
import math

import numpy as np
import pytest
import pyvista as pv
from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281.contact_solver import (
    ISO16281BallSolver)

from viz import bearing as vb, scene as vs

pv.OFF_SCREEN = True


def _solved(load_case, name, Fa=None):
    """Bearing and result of one load case, solved as ``01_contacts_vs_hertz.py`` does."""
    cd = load_case("iso16281_ball_contact")
    load = cd.LOADS[name]
    bearing = cd.build_bearing(load["s"])
    result = ISO16281BallSolver().solve_contact(bearing, Fr_xz=load["Fr"], Fr_xy=0.0,
                                                Fa=load["Fa"] if Fa is None else Fa,
                                                delta_r_init=0.0, delta_a_init=0.0, psi=0.0, phi_Fr=0.0)
    return bearing, result


@pytest.fixture
def heavy(load_case):
    bearing, result = _solved(load_case, "radial_heavy")
    return bearing, result, vb.bearing_scene(bearing, result, case="radial load, heavy")


def test_blocks_arrays_and_values(heavy):
    bearing, _, scene = heavy
    assert scene.keys() == ["ring_inner", "ring_outer", "balls", "contacts_inner", "contacts_outer",
                            "load_Fr", "load_labels"]                 # Fa = 0: no axial arrow
    assert {"Q", "alpha"} <= set(scene["balls"].array_names)
    balls = scene["balls"]
    Q = np.array([balls["Q"][balls["j"] == j][0] for j in range(bearing.Z)])
    assert np.count_nonzero(np.isnan(Q)) == 4                         # 5 of 9 balls loaded
    p_inner, p_outer = scene["contacts_inner"]["p0"], scene["contacts_outer"]["p0"]
    assert len(p_inner) == len(p_outer) == 5 and np.all(p_outer > 0.0)
    # same load, less conforming contact on the inner raceway (convex in rolling): higher pressure
    assert np.all(p_inner > p_outer)
    # one bar for both raceways, round ticks; radial load: alpha_j = 0 exactly, one-unit range
    assert vs.clim_of(scene, vb.FIELDS) == {"Q": (0.0, 5000.0), "p0": (0.0, 3200.0), "alpha": (0.0, 1.0)}


def test_balls_and_rings(heavy):
    bearing, result, scene = heavy
    balls, phi = scene["balls"], result.phi_j_global(0)
    for j in range(bearing.Z):                                        # pitch circle, phi from +Z toward +Y
        expected = [bearing.position, 0.5 * bearing.Dpw * math.sin(phi[j]), 0.5 * bearing.Dpw * math.cos(phi[j])]
        assert balls.points[balls["j"] == j].mean(axis=0) == pytest.approx(expected, abs=1e-3)
    # ball 0 is at phi_Fr and the arrow points along phi_Fr: by construction, not a check of row 15
    assert scene["load_Fr"].center[2] > 0.0 and int(np.argmax(result.row.Q_j)) == 0
    # groove bottoms clear the ball by a quarter of the diametral clearance s each
    ball_0 = [bearing.position, 0.0, 0.5 * bearing.Dpw]
    for ring in ("ring_inner", "ring_outer"):
        assert scene[ring].n_open_edges == 0                          # closed surface, no seam
        gap = np.linalg.norm(scene[ring].points - ball_0, axis=1).min()
        assert gap == pytest.approx(0.5 * bearing.Dw + 0.25 * bearing.s, abs=1e-3)


@pytest.mark.parametrize("Fa", [4450.0, -4450.0])
def test_contact_points_and_ellipses(load_case, Fa):
    bearing, result = _solved(load_case, "combined", Fa=Fa)
    scene = vb.bearing_scene(bearing, result, ellipse_scale=1.0)
    inner, outer = scene["contacts_inner"], scene["contacts_outer"]
    c_inner, c_outer, phi = inner.cell_centers().points, outer.cell_centers().points, result.phi_j_global(0)
    for k, j in enumerate(inner["j"]):
        e_r = np.array([0.0, math.sin(phi[j]), math.cos(phi[j])])
        e_phi = np.array([0.0, math.cos(phi[j]), -math.sin(phi[j])])
        ball = np.array([bearing.position, 0.0, 0.0]) + 0.5 * bearing.Dpw * e_r
        assert 0.5 * (c_inner[k] + c_outer[k]) == pytest.approx(ball, abs=1e-9)
        # Fa > 0: inner contact on the -x side of the ball; mirrored for Fa < 0
        offset = c_inner[k, 0] - bearing.position
        assert offset == pytest.approx(-0.5 * bearing.Dw * math.sin(result.row.alpha_j[j]), abs=1e-9)
        assert offset * Fa < 0.0
        # the major semi-axis is s_y, across rolling: perpendicular to the circumferential direction
        rim = inner.extract_cells(k).points - c_inner[k]
        major = rim[np.argmax(np.linalg.norm(rim, axis=1))]
        assert np.linalg.norm(major) == pytest.approx(inner["s_y"][k], rel=1e-6)
        assert major @ e_phi == pytest.approx(0.0, abs=1e-6)


def test_side_view_is_the_section_through_ball_0(load_case):
    bearing, result = _solved(load_case, "combined")
    scene = vb.bearing_scene(bearing, result, view="side")
    assert list(np.unique(scene["balls"]["j"])) == [0] and scene["contacts_inner"].n_cells == 1
    assert scene["ring_inner"].n_lines > 0 and np.allclose(scene["ring_inner"].points[:, 1], 0.0)
    assert scene.keys()[5:] == ["radial_lines", "contact_lines", "section_labels", "load_Fr", "load_Fa", "load_labels"]
    assert scene["section_labels"]["label"][3].strip() == f"alpha = {math.degrees(result.row.alpha_j[0]):.4g} deg"
    # Fa > 0 is drawn along +x, Fr along +z, both in the section plane
    assert scene["load_Fa"].bounds[1] - scene["load_Fa"].bounds[0] == pytest.approx(0.2 * bearing.b)
    assert scene["load_Fr"].bounds[5] - scene["load_Fr"].bounds[4] == pytest.approx(0.2 * bearing.b)
    # the contact line passes through both contacts, tilted by alpha_j from the radial (+Z) direction
    a, b = scene["contact_lines"].points
    assert np.linalg.norm(b - a) == pytest.approx((1.0 + 2.0 * vb.CONTACT_LINE_OVERHANG) * bearing.Dw)
    assert math.atan2(b[0] - a[0], b[2] - a[2]) == pytest.approx(result.row.alpha_j[0])
    assert np.cross(b - a, scene["contacts_inner"].cell_centers().points[0] - a) == pytest.approx(0.0, abs=1e-9)


def test_save_and_explore(heavy, tmp_path):
    _, _, scene = heavy
    paths = vs.save(scene, vb.FIELDS, tmp_path / "viz", "radial_heavy")
    assert [p.name for p in paths] == ["radial_heavy.vtm"] + [f"radial_heavy_{f}.png" for f in vb.FIELDS]
    assert all(p.is_file() and p.stat().st_size > 0 for p in paths)
    assert pv.read(paths[0]).keys() == scene.keys()
    vs.explore(scene, vb.FIELDS)                     # off-screen: no window opens
