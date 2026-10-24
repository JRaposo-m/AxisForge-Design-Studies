"""Contact zoom: the pressure of one dry contact over the contact plane, against Hertz.

What is drawn (contact plane; lengths in mm, x = rolling direction, y = across rolling):

* the pressure solved by SlipPY (BEM, ``slippy_bridge.contact.static_step.run``) as a surface
  whose height is the pressure times a factor printed on the figure, coloured by the pressure
  ``p`` [MPa] or by its difference to Hertz ``dp = p - p_Hertz`` [MPa];
* in the top view, the outline of the Hertz contact ellipse (semi-axes ``s_x`` along rolling,
  ``s_y`` across).

Two fixed views (``CAMERAS``), both in parallel projection with x and y at the same scale:
``"hill"`` looks at the pressure surface from the side and above, with lines of equal pressure on
it; ``"top"`` looks straight down on the contact plane, where the footprint is compared with the
Hertz ellipse.

Nothing is solved or read from a file here: the caller runs the static step and passes its grid
and its pressure array. ``slippy_bridge`` works in SI (Pa, m); the values are converted to MPa and
mm with ``viz.scene.MPA_PER_PA`` and ``MM_PER_M``.
"""
from __future__ import annotations

import numpy as np
import pyvista as pv

from slippy_bridge.contact import Contact, hertz
from slippy_bridge.contact.measures import rel_err

from .scene import MM_PER_M, MPA_PER_PA, set_view, tick_values

__all__ = ["FIELDS", "CAMERAS", "contact_scene"]

# array name: (display name, unit, colour map). The difference is signed: diverging map, white = 0.
FIELDS = {
    "p": ("Contact pressure p (SlipPY)", "MPa", "plasma"),
    "dp": ("Pressure difference p - p_Hertz", "MPa", "RdBu_r"),
}

_UM_PER_MM = 1e3        # grid spacing printed in micrometres
# Drawing only: height of the Hertz maximum p0 as a fraction of the larger semi-axis.
HILL_HEIGHT_OVER_SEMI_AXIS = 0.5
# Fixed views. name: (direction from the focal point to the camera, direction drawn upward).
# "hill": from the +x, -y side and above; pressure (+z) up.
# "top": straight down on the contact plane; -x up, so y (the long axis of a ball-raceway contact)
#     runs left to right.
CAMERAS = {
    "hill": ((0.62, -0.45, 0.64), (0.0, 0.0, 1.0)),
    "top": ((0.0, 0.0, 1.0), (-1.0, 0.0, 0.0)),
}
_LINE_COLOR = (0.1, 0.1, 0.1)


def contact_scene(contact, grid, p, case: str = "", view: str = "hill", crop: float = 1.5) -> pv.MultiBlock:
    """Scene of the pressure of one point contact, solved by SlipPY, with its Hertz reference.

    Parameters
    ----------
    contact : slippy_bridge.contact.ContactCase or Contact
        The contact (SI): radii [m], moduli [Pa], load ``Q`` [N]. Point contacts only. The same
        object that was given to ``static_step.run``.
    grid : slippy_bridge.contact.Grid
        The grid the pressure was solved on: node coordinates ``X`` (rolling, array axis 0) and
        ``Y`` [m], spacing ``h`` [m], contact centre at the origin.
    p : ndarray, shape ``grid.X.shape``
        Pressure [Pa] on the nodes of ``grid``, the second return value of
        ``slippy_bridge.contact.static_step.run`` (rolling along axis 0).
    case : str, optional
        Which contact this is, in plain words, printed first on the figure.
    view : {"hill", "top"}, optional
        Key of ``CAMERAS``.
    crop : float, optional
        The grid is drawn within ``crop`` Hertz semi-axes of the centre in each direction [-]; the
        solved domain is larger along rolling (4 semi-axes by default) and would shrink the
        picture. What is drawn and what was solved are both printed on the figure.

    Returns
    -------
    pyvista.MultiBlock
        ``pressure``: the surface ``z = p * factor`` over the grid nodes, point data ``p`` [MPa]
        and ``dp`` [MPa]. Hill view: ``contours``, lines of equal ``p`` at the ticks of the colour
        bar. Top view: ``hertz_ellipse``, the outline of the Hertz ellipse, drawn 1.5 grid spacings
        outside it. ``field_data``: ``annotation``, ``annotation_p``, ``annotation_dp``,
        ``camera_position``, ``parallel_scale`` (read by ``viz.scene``) and ``height_factor``
        [mm/MPa]. Coordinates in mm; x and y are drawn with the same scale.

    Raises
    ------
    ValueError
        For a line contact, or a pressure array that does not have the shape of the grid.

    Notes
    -----
    ``p_Hertz`` is the semi-ellipsoid ``p0 sqrt(1 - (x/s_x)^2 - (y/s_y)^2)`` of
    ``slippy_bridge.contact.hertz(case)["pressure"]``, evaluated on the same nodes: smooth
    surfaces, elastic half-spaces, dry, normal load only (Johnson 1985, ch. 4).

    What ``dp`` shows. (1) The discretisation of the contact edge: the Hertz pressure has an
    infinite slope there, and a shift of the edge by one node is worth about
    ``p0 sqrt(2 h / s_y)`` (about 340 MPa for the 6210-like inner contact at 4.5 kN on the default
    grid). The stripes on the long flanks are the staircase of the grid along an edge that is
    nearly parallel to y, not physical waves. (2) The finite, periodic domain of SlipPY's FFT
    kernel: it shifts the pressure at the centre (-0.3 % in that case with the default domain of
    +/-4 s_x; +0.1 % with +/-8 s_x, where the difference inside the ellipse drops to about 10 MPa).

    Hertz and SlipPY share the half-space assumption, so their agreement is a numerical
    verification, not a validation of Hertz for this contact: across rolling the ellipse is not
    small against the groove radius (``s_y / r_groove`` about 0.38, a conformal contact).

    The height of the surface is a drawing scale, not a length: ``factor`` [mm/MPa] is chosen so
    that ``p0`` stands ``HILL_HEIGHT_OVER_SEMI_AXIS`` larger semi-axes high, and rounded to one
    significant digit so it can be printed.
    """
    contact_case = contact.case if isinstance(contact, Contact) else contact
    if contact_case.line:
        raise ValueError("contact_scene draws point contacts only")
    if np.shape(p) != grid.X.shape:
        raise ValueError(f"p has shape {np.shape(p)}, the grid {grid.X.shape}")
    ref = hertz(contact_case)                           # SI
    p0 = ref["p0_Pa"] * MPA_PER_PA
    s_x, s_y = ref["s_x_m"] * MM_PER_M, ref["s_y_m"] * MM_PER_M
    h = grid.h * MM_PER_M

    # nodes within crop semi-axes of the centre; X varies along axis 0 (rolling), Y along axis 1
    x, y = grid.X[:, 0] * MM_PER_M, grid.Y[0, :] * MM_PER_M
    keep_x, keep_y = np.abs(x) <= crop * s_x, np.abs(y) <= crop * s_y
    window = np.ix_(keep_x, keep_y)
    p_mpa = np.asarray(p, dtype=float)[window] * MPA_PER_PA
    dp = p_mpa - ref["pressure"](grid.X[window], grid.Y[window]) * MPA_PER_PA
    image = pv.ImageData(dimensions=(keep_x.sum(), keep_y.sum(), 1), spacing=(h, h, 1.0),
                         origin=(x[keep_x][0], y[keep_y][0], 0.0))
    image["p"] = p_mpa.ravel(order="F")                 # VTK point order: x runs fastest
    image["dp"] = dp.ravel(order="F")

    larger = max(s_x, s_y)
    factor = float(f"{HILL_HEIGHT_OVER_SEMI_AXIS * larger / p0:.0e}")       # one significant digit [mm/MPa]
    direction, view_up = CAMERAS[view]
    # lines are set 2 % of the hill height toward the camera: in a parallel projection that does
    # not move them on the picture, and the surface they lie on cannot cover them
    toward_camera = 0.02 * factor * p0 * np.array(direction) / np.linalg.norm(direction)

    scene = pv.MultiBlock()
    scene["pressure"] = image.warp_by_scalar("p", factor=factor, normal=(0.0, 0.0, 1.0))

    # Numbers of the difference to Hertz, printed on the dp figure: inside the ellipse and at the rim
    rho = np.hypot(grid.X[window] * MM_PER_M / s_x, grid.Y[window] * MM_PER_M / s_y)
    inside = dp[rho < 0.9]
    worst = np.unravel_index(np.argmax(np.abs(dp)), dp.shape)
    edge_node = p0 * np.sqrt(2.0 * h / s_y)             # Hertz pressure one node inside the edge on the long axis
    p_max = float(p_mpa.max())
    # half-lengths of the solved domain and of the drawn window, in Hertz semi-axes
    solved_x, solved_y = min(-x[0], x[-1]) / s_x, min(-y[0], y[-1]) / s_y
    drawn_x, drawn_y = np.abs(x[keep_x]).max() / s_x, np.abs(y[keep_y]).max() / s_y
    scene.field_data["annotation"] = [
        f"{case}: Q = {contact_case.load:.0f} N".lstrip(": "),
        f"Hertz: p0 = {p0:.0f} MPa, s_x = {s_x:.3f} mm (rolling), s_y = {s_y:.3f} mm",
        f"SlipPY: max p = {p_max:.0f} MPa ({rel_err(p_max, p0):+.2%} vs Hertz); "
        f"grid {grid.X.shape[0]} x {grid.X.shape[1]}, h = {h * _UM_PER_MM:.2f} um",
    ]
    lines_p = [f"solved domain +/-{solved_x:.2g} s_x, +/-{solved_y:.2g} s_y; "
               f"drawn +/-{drawn_x:.2g} s_x, +/-{drawn_y:.2g} s_y"]
    lines_dp = [
        f"inside the ellipse (rho < 0.9): dp within {inside.min():+.0f}..{inside.max():+.0f} MPa "
        f"({np.abs(inside).max() / p0:.1%} of p0)",
        f"largest dp = {dp[worst]:+.0f} MPa at rho = {rho[worst]:.2f}; "
        f"one grid node at the edge is worth ~{edge_node:.0f} MPa",
        "stripes along the flanks = staircase of the grid",
    ]
    if view == "hill":
        # the surface is drawn without lighting: lines of equal pressure give it its shape
        levels = [t for t in tick_values(0.0, p_max)[1:] if t < p_max]
        scene["contours"] = scene["pressure"].contour(isosurfaces=levels, scalars="p").translate(toward_camera)
        scene["contours"].clear_data()                  # geometry only: a fixed colour, not a field
        scene["contours"].field_data["color"] = _LINE_COLOR
        # the Hertz ellipse is the foot of the hill, which hides it from the side: not drawn here
        note = [f"surface height = p x {factor:g} mm/MPa; black lines: p = "
                + ", ".join(f"{t:g}" for t in levels) + " MPa",
                "footprint against the Hertz ellipse: see the top view"]
        lines_p, lines_dp = note, lines_dp + note[:1]
        set_view(scene, [0.0, 0.0, 0.5 * factor * p0], direction, 0.8 * crop * larger, view_up)
    else:
        # Hertz ellipse, thin and 1.5 grid spacings outside the rim, so that it does not cover the
        # one or two nodes of the rim where the difference to Hertz is largest
        u = np.linspace(0.0, 2.0 * np.pi, 361)
        scene["hertz_ellipse"] = pv.lines_from_points(np.column_stack(
            [(s_x + 1.5 * h) * np.cos(u), (s_y + 1.5 * h) * np.sin(u), np.zeros_like(u)]) + toward_camera)
        scene["hertz_ellipse"].field_data["color"] = _LINE_COLOR
        scene["hertz_ellipse"].field_data["line_width"] = [2.0]
        lines_p.append("top view, x and y at the same scale; black outline: Hertz ellipse")
        lines_dp[-1] += "; black outline: Hertz ellipse"
        # the frame is 4/3 wider than high and holds the drawn length across rolling with a margin
        set_view(scene, [0.0, 0.0, 0.0], direction, 0.75 * 1.15 * drawn_y * s_y, view_up)
    scene.field_data["annotation_p"], scene.field_data["annotation_dp"] = lines_p, lines_dp
    scene.field_data["height_factor"] = [factor]
    return scene
