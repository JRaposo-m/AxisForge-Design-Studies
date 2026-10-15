"""viz.contact and viz.shaft: the contact zoom and the shaft view, off-screen."""
import numpy as np
import pytest
import pyvista as pv

from slippy_bridge.contact import Grid, contact_case, hertz, pair, static_step
from viz import contact as vc, scene as vs, shaft as vsh

pv.OFF_SCREEN = True


def test_contact_scene(load_case, tmp_path):
    pytest.importorskip("pyfftw")                    # SlipPY's FFT backend, as in test_slippy_bridge
    case = contact_case(load_case("iso16281_ball_contact").build_bearing(0.0), "inner", 1000.0)
    ref = hertz(case)
    grid = Grid.around(case, h=ref["s_x_m"] / 4.0)   # coarser than the default s_x / 10: a fast solve
    _, p = static_step.run(case, pair(case, grid), grid, tmp_path / "slippy")
    scene = vc.contact_scene(case, grid, p, case="test contact")
    assert scene.keys() == ["pressure", "contours"]
    surface, factor = scene["pressure"], scene.field_data["height_factor"][0]
    p0, s_x, s_y = ref["p0_Pa"] * 1e-6, ref["s_x_m"] * 1e3, ref["s_y_m"] * 1e3
    # MPa and mm: the BEM maximum is the Hertz p0 within the error of this coarse grid
    assert surface["p"].max() == pytest.approx(p0, rel=0.03)
    # x = rolling (short axis), y across: no pressure well outside the ellipse, near p0 at its centre
    rho2 = (surface.points[:, 0] / s_x) ** 2 + (surface.points[:, 1] / s_y) ** 2
    assert surface["p"][rho2 > 1.6**2].max() == 0.0 and surface["p"][rho2 < 0.25].min() > 0.8 * p0
    # the height is the pressure times the printed factor; along rolling the solved domain is cropped
    assert surface.points[:, 2] == pytest.approx(surface["p"] * factor)
    assert surface.bounds[1] <= 1.5 * s_x < grid.X.max() * 1e3 and surface.bounds[3] <= 1.5 * s_y
    assert f"surface height = p x {factor:g} mm/MPa" in scene.field_data["annotation_p"][0]
    clim = vs.clim_of(scene, vc.FIELDS)
    assert clim["dp"][0] == -clim["dp"][1] < 0.0 == clim["p"][0]       # diverging map: symmetric limits
    top = vc.contact_scene(case, grid, p, view="top")
    assert top.keys() == ["pressure", "hertz_ellipse"]
    # the outline is the Hertz ellipse drawn 1.5 grid spacings outside, clear of the rim nodes
    out = 1.5 * grid.h * 1e3
    assert top["hertz_ellipse"].bounds[:4] == pytest.approx((-s_x - out, s_x + out, -s_y - out, s_y + out))
    # the numbers printed on the dp figure are those of the data
    inside = surface["dp"][rho2 < 0.81]
    assert f"dp within {inside.min():+.0f}..{inside.max():+.0f} MPa" in top.field_data["annotation_dp"][0]
    paths = vs.save(top, vc.FIELDS, tmp_path / "viz", "contact")
    assert [f.name for f in paths] == ["contact.vtm", "contact_p.png", "contact_dp.png"]
    assert all(f.stat().st_size > 0 for f in paths)
    with pytest.raises(ValueError, match="shape"):
        vc.contact_scene(case, grid, p[:-1])


def test_shaft_scene(load_case, tmp_path):
    cd = load_case("fem_stepped_distributed_gear")
    system = cd.build_system()
    results = cd.solve(system, cd.THEORIES["timoshenko/cowper"], n_elements=40)["shaft1"]
    scene = vsh.shaft_scene(system.shafts[0], results, case="test", labels={"g_driver:Ft": "Gear tangential force"})
    assert scene.keys() == ["shaft", "undeformed", "bearings", "loads", "load_labels"]
    shaft, factor = scene["shaft"], scene.field_data["magnification"][0]
    rest = shaft.points - factor * shaft["displacement"]              # where the shaft is drawn from
    assert shaft.n_open_edges == 0                                    # closed surface
    assert shaft["v"].max() == pytest.approx(results.v_max)           # [mm]
    # the drawn shaft is the stepped shaft moved by the stored (0, v_xy, v_xz) times the factor
    assert np.all(shaft["displacement"][:, 0] == 0.0)
    assert shaft["displacement"][:, 2].max() == pytest.approx(results.v_xz.max())     # sign kept
    assert (rest[:, 0].min(), rest[:, 0].max()) == pytest.approx((0.0, cd.SHAFT_LENGTH_MM))
    assert np.hypot(rest[:, 1], rest[:, 2]).max() == pytest.approx(14.0, abs=1e-4)    # 28 mm diameter
    assert scene["undeformed"].bounds[:2] == pytest.approx((16.0, 183.0))   # grey shaft between the bearings
    assert scene["bearings"].bounds[5] == pytest.approx(23.5)         # NU204: D = 47 mm
    # bending stress per section: at the step x = 115 mm (25 -> 28 mm) the same moment gives two
    # values, in the ratio of the section moduli; the stored maximum lies inside the 25 mm seat
    at_step = np.isclose(rest[:, 0], 115.0) & (np.hypot(rest[:, 1], rest[:, 2]) > 12.0)
    assert shaft["sigma_b"][at_step].max() / shaft["sigma_b"][at_step].min() == pytest.approx((28 / 25) ** 3)
    assert shaft["sigma_b"].max() == pytest.approx(results.sigma_b_max)   # [MPa]
    # three radial loads; the gear tangential force on the driving gear (-z since the D-0010 mesh loads, Q05)
    # comes from above, its tip on the drawn shaft
    labels = list(scene["load_labels"]["label"])                      # a number per arrow, then its legend line
    assert labels[4:] == ["  3", "3: Gear tangential force = 5000 N"] and labels[1].startswith("1: parabolic user load")
    tail_z = factor * np.interp(100.0, results.x, results.v_xz) + 12.5 + vsh.ARROW_LENGTH_OVER_LENGTH * 200.0
    assert scene["load_labels"].points[4] == pytest.approx(
        [100.0, factor * np.interp(100.0, results.x, results.v_xy), tail_z])
    paths = vs.save(scene, vsh.FIELDS, tmp_path / "viz", "shaft1")
    assert len(paths) == 3 and all(f.stat().st_size > 0 for f in paths)
