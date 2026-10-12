"""Bearing zoom: one deep groove ball bearing with its ISO/TS 16281 load distribution, in 3D.

What is drawn (AxisForge global frame, D-0010: x = shaft axis; lengths in mm):

* the two rings, revolved from their cross-section with the raceway groove;
* the balls at the angles the solver returns, coloured by the ball load ``Q`` [N] or the operating
  contact angle ``alpha`` [deg];
* one Hertz contact ellipse per loaded ball and raceway, coloured by the maximum Hertz pressure
  ``p0`` [MPa], enlarged by a factor that is printed on the figure;
* in the side view, the contact line of the ball (inner contact - ball centre - outer contact), whose tilt
  from the radial direction is the contact angle;
* the radial (and axial) force of the result, as an arrow at the bearing centre.

Two fixed views (``CAMERAS``), both in parallel projection with +Z up: ``"oblique"`` shows the whole
bearing; ``"side"`` is the axial section through the top ball, x horizontal, where the axial shift
of the two contacts under an axial load is read directly.

Angles. AxisForge measures the element angle ``phi`` from +Z toward +Y (``BearingResult.phi_Fr``,
``BearingResult.phi_j_global``), so the radial unit vector at ``phi`` is ``(0, sin(phi), cos(phi))``.
This is the opposite sense to a right-hand rotation about +X.

Whose force the arrows are. Inside the solver ``phi_j = 0`` is the direction in which the radial
force is applied to the inner ring (``LoadDistributionResult`` docstring), and the most loaded ball
is there. On the shaft FEM path ``Fr_xz``, ``Fr_xy`` and ``Fa`` are the reaction of the support on
the shaft (``BearingResult`` docstring, Warnings: an open review item of AxisForge). ``viz`` draws
the forces as stored and the balls as returned by ``phi_j_global`` and does not compensate, so the
radial arrow always points at the loaded zone.

Hertz values come from ``slippy_bridge.contact`` (D-0019), which works in SI; they are converted to
MPa and mm here, with ``viz.scene.MPA_PER_PA`` and ``MM_PER_M``.
"""
from __future__ import annotations

import math

import numpy as np
import pyvista as pv

from slippy_bridge.contact import contacts, hertz

from .scene import MM_PER_M, MPA_PER_PA, revolve, set_view

__all__ = ["FIELDS", "CAMERAS", "bearing_scene"]

# array name: (display name, unit, colour map). ``p0`` is carried by both contact blocks, so the two
# raceways are compared on one colour bar. Balls use viridis (no grey in it: grey = unloaded),
# pressures plasma, so a load is not read as a pressure.
FIELDS = {
    "Q": ("Ball load Q", "N", "viridis"),
    "p0": ("Hertz pressure p0, inner and outer raceway", "MPa", "plasma"),
    "alpha": ("Contact angle alpha", "deg", "viridis"),
}

# Drawing only: AxisForge has no shoulder geometry. Height of the ring shoulders above the groove
# bottom as a fraction of the ball diameter; a usual proportion of deep groove bearings.
SHOULDER_HEIGHT_OVER_DW = 0.22
# Fixed views, +Z drawn up (the radial load of the studies is along +Z, so the loaded zone is at the
# top). name: (direction from the focal point to the camera, height of the focal point on +Z in
# pitch radii, half the frame height in outer diameters).
# "oblique": from the +x side, raised toward +Z and with no y component, so the picture is symmetric
#     about the load line and the ellipses near it are seen face-on; the bearing fills about 70 % of
#     the frame height.
# "side": along +y onto the plane XZ, x to the right, centred on the top ball.
CAMERAS = {
    "oblique": ((0.80, 0.0, 0.60), 0.0, 0.667),
    "side": ((0.0, -1.0, 0.0), 1.0, 0.18),
}
# Drawing only: the contact line is drawn this far beyond each contact [ball diameters], otherwise
# it would be hidden inside the ball.
CONTACT_LINE_OVERHANG = 0.25
_ARROW_COLOR = (0.75, 0.1, 0.1)
_LINE_COLOR = (0.1, 0.1, 0.1)
_REFERENCE_COLOR = (0.6, 0.6, 0.6)


def _ring(bearing, race: str) -> pv.PolyData:
    """Surface of one ring, revolved about the shaft axis from its cross-section with the groove.

    The groove is the circular arc of radius ``ri`` / ``re`` whose centre of curvature lies at the
    radius AxisForge uses: ``bearing.Ri`` for the inner ring (ISO/TS 16281: distance of the inner
    groove centre from the axis) and ``Ri - A cos(alpha_0)`` for the outer ring, the two centres
    being ``A cos(alpha_0)`` apart radially when the rings are concentric (``U_j`` at zero load in
    ``ISO16281BallSolver.elements``). The rings are drawn concentric and axially centred, unloaded.
    Shoulder height: drawing only (``SHOULDER_HEIGHT_OVER_DW``).
    """
    # inner ring: material below the groove (side = +1); outer ring: above it (side = -1)
    side, r_groove, r_back = (1.0, bearing.ri, bearing.d / 2) if race == "inner" else (-1.0, bearing.re, bearing.D / 2)
    r_centre = bearing.Ri if race == "inner" else bearing.Ri - bearing.A * math.cos(bearing.alpha_0)
    height = SHOULDER_HEIGHT_OVER_DW * bearing.Dw
    r_land = r_centre - side * (r_groove - height)                   # shoulder (land) radius
    t = np.linspace(1.0, -1.0, 41) * math.acos(1.0 - height / r_groove)   # groove arc, +x edge to -x edge
    x = np.concatenate([[-bearing.b / 2, bearing.b / 2, bearing.b / 2], r_groove * np.sin(t), [-bearing.b / 2]])
    r = np.concatenate([[r_back, r_back, r_land], r_centre - side * r_groove * np.cos(t), [r_land]])
    # cross-section in the plane y = 0 (phi = 0), closed, turned once about x
    return revolve(bearing.position + x, r, resolution=180)


def bearing_scene(bearing, result, case: str = "", ellipse_scale: float = 4.0,
                  view: str = "oblique") -> pv.MultiBlock:
    """Scene of a single-row ball bearing and its ISO/TS 16281 load distribution.

    Parameters
    ----------
    bearing : axisforge Bearing
        Assembled ball bearing [mm]: ``d``, ``D``, ``b``, ``position``, ``Dw``, ``Dpw``, ``ri``,
        ``re``, ``Ri``, ``A``, ``alpha_0`` [rad], ``s`` (diametral clearance).
    result : axisforge BallBearingResult
        Solved load distribution of ``bearing`` (single row): ``Q_j`` [N], ``alpha_j`` [rad],
        ``phi_j_global`` [rad], ``Fr``, ``Fa`` [N], ``phi_Fr`` [rad].
    case : str, optional
        Load case in plain words, printed on the figure ("radial load, heavy").
    ellipse_scale : float, optional
        Enlargement [-] of the contact ellipses, both semi-axes; printed on the figure. At true
        size the ellipse (about 0.2 mm by 1.8 mm on a 12.7 mm ball) cannot be seen.
    view : {"oblique", "side"}, optional
        Key of ``CAMERAS``. ``"side"`` is the axial section in the plane XZ: the rings are cut by
        that plane (their cross-section is drawn as a line) and only the balls whose centre lies in
        it are kept, with their contacts. It needs ``phi_Fr = 0`` (radial load along +Z), which
        puts ball 0 in the plane.

    Returns
    -------
    pyvista.MultiBlock
        Blocks ``ring_inner``, ``ring_outer`` (no data), ``balls`` (point data ``Q`` [N],
        ``alpha`` [deg], ``j`` [-]), ``contacts_inner``, ``contacts_outer`` (one polygon per loaded
        ball; cell data ``p0`` [MPa], ``s_x``, ``s_y`` [mm] at true size, ``j`` [-]),
        ``load_Fr``, ``load_Fa`` if ``Fa != 0`` (arrows) and ``load_labels``; in the side
        view also ``radial_lines``, ``contact_lines`` (one segment per ball) and
        ``section_labels``. ``field_data``: ``annotation``, ``annotation_<field>``,
        ``camera_position`` and ``parallel_scale``, read by ``viz.scene``. Coordinates in mm,
        AxisForge global frame (x = shaft axis).

    Raises
    ------
    ValueError
        If no loaded ball is in the view (there is no contact to draw).

    Notes
    -----
    Ball centre: ``(position, (Dpw/2) sin(phi_j), (Dpw/2) cos(phi_j))``, on the pitch circle and
    the bearing mid-plane for every load. A simplification: with an axial load the real centre moves
    by about 0.09 mm in the ``combined`` case (take-up of the axial play), so the drawn contact
    points sit about 0.02 mm off the drawn groove. Unloaded balls carry NaN and are drawn grey.

    Contact normal: ``n = sin(alpha_j) e_x + cos(alpha_j) e_r``, the direction from the outer to
    the inner groove centre (``U_j``, ``V_j`` of ``ISO16281BallSolver.elements``).
    ``sign(alpha_j) = sign(A sin(alpha_0) + delta_a) = sign(Fa)`` (ISO/TS 16281 Eq. (17) as coded
    in ``ISO16281BallSolver.solve_contact``).

    Contact points: inner ``centre - (Dw/2) n``, outer ``centre + (Dw/2) n``. The inner contact is
    on the -x side of the ball for ``Fa > 0``, which holds when ``Fa`` is the axial load on the
    inner ring along +x. The contact line joins the two through the ball centre and is drawn
    ``CONTACT_LINE_OVERHANG`` ball diameters longer at each end (drawing only); side view only,
    where its tilt from the vertical is the contact angle.

    Each ellipse is flat, in the tangent plane of its contact, with the Hertz semi-axis ``s_x``
    along rolling (circumferential) and ``s_y`` across rolling (along the groove), from
    ``slippy_bridge.contact.hertz``: smooth, elastic half-spaces, dry, normal load only, raceway
    radii at ``alpha_j``, the same ``Q_j`` on both raceways (no centrifugal force in AxisForge).
    ``s_y / r_groove`` reaches about 0.38: a conformal contact, where half-space Hertz is the usual
    bearing approximation. Enlarged and flat, an ellipse leaves the curved surfaces: it shows the
    position and the proportions of the contact, not its extent.

    Arrows: ``load_Fr`` along ``phi_Fr`` (magnitude ``result.Fr``), ``load_Fa`` along the sign of
    ``Fa`` on x, with a fixed length (not to scale). In the oblique view ``Fr`` starts at the
    bearing centre and ``Fa`` beside it; in the side view both are drawn small, in and below the
    inner ring section. The figure prints the forces as "on the inner ring", which is how the solver
    reads them (module docstring).
    """
    row = result.row                                    # single-row only (raises on a multi-row bearing)
    centre = np.array([bearing.position, 0.0, 0.0])
    e_x = np.array([1.0, 0.0, 0.0])
    phi = np.asarray(result.phi_j_global(0))            # [rad] from +Z toward +Y
    e_r = np.column_stack([np.zeros_like(phi), np.sin(phi), np.cos(phi)])     # radial, per ball
    e_phi = np.column_stack([np.zeros_like(phi), np.cos(phi), -np.sin(phi)])  # circumferential = rolling
    ball_centres = centre + 0.5 * bearing.Dpw * e_r
    loaded = row.Q_j > 0.0
    # the section shows only the balls cut through their centre by the plane XZ
    shown = np.abs(ball_centres[:, 1]) < 1e-6 * bearing.Dpw if view == "side" else np.full(row.Z, True)
    if not np.any(shown & loaded):
        raise ValueError(f"{bearing.label or bearing.designation}: no loaded ball in view {view!r}")

    scene = pv.MultiBlock()
    for race in ("inner", "outer"):
        ring = _ring(bearing, race)
        if view == "side":
            ring = ring.slice(normal="y", origin=centre)                    # cross-section in the plane XZ
            ring.field_data["color"] = _LINE_COLOR
        scene[f"ring_{race}"] = ring

    points = pv.PolyData(ball_centres[shown])
    points["Q"] = np.where(loaded, row.Q_j, np.nan)[shown]                  # [N]
    # [rad] -> [deg]; rounded: a radial load leaves 1e-10 deg of solver noise, which is 0
    points["alpha"] = np.where(loaded, np.round(np.degrees(row.alpha_j), 6), np.nan)[shown]
    points["j"] = np.arange(row.Z)[shown]
    scene["balls"] = points.glyph(geom=pv.Sphere(radius=bearing.Dw / 2, theta_resolution=48, phi_resolution=48),
                                  scale=False, orient=False)

    u = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)[:, None]          # ellipse parameter
    drawn = {race: dict(points=[], p0=[], s_x=[], s_y=[], j=[]) for race in ("inner", "outer")}
    line_ends = []
    for c in contacts(bearing, result):                 # loaded balls, inner and outer raceway
        j, alpha, race = c.location["element_j"], c.location["alpha_rad"], c.location["race"]
        if not shown[j]:
            continue
        ref = hertz(c.case)                             # SI
        s_x, s_y = ref["s_x_m"] * MM_PER_M, ref["s_y_m"] * MM_PER_M
        normal = math.sin(alpha) * e_x + math.cos(alpha) * e_r[j]           # outer -> inner groove centre
        across = math.cos(alpha) * e_x - math.sin(alpha) * e_r[j]           # in the tangent plane, across rolling
        # inner contact on the side of the ball facing the axis (alpha = 0), outer contact opposite
        side = -1.0 if race == "inner" else 1.0
        contact_point = ball_centres[j] + side * 0.5 * bearing.Dw * normal
        line_ends.append(ball_centres[j] + side * (0.5 + CONTACT_LINE_OVERHANG) * bearing.Dw * normal)
        d = drawn[race]
        d["points"].append(contact_point + ellipse_scale * (s_x * np.cos(u) * e_phi[j] + s_y * np.sin(u) * across))
        d["p0"].append(ref["p0_Pa"] * MPA_PER_PA)
        d["s_x"].append(s_x)
        d["s_y"].append(s_y)
        d["j"].append(j)
    for race, d in drawn.items():
        n = len(d["j"])
        ellipses = pv.PolyData.from_regular_faces(np.concatenate(d["points"]),
                                                  np.arange(n * len(u)).reshape(n, len(u)))
        ellipses.cell_data["p0"] = np.array(d["p0"], dtype=float)           # [MPa]
        ellipses.cell_data["s_x"] = np.array(d["s_x"], dtype=float)         # [mm], true size
        ellipses.cell_data["s_y"] = np.array(d["s_y"], dtype=float)         # [mm], true size
        ellipses.cell_data["j"] = np.array(d["j"], dtype=int)
        ellipses.field_data["active_only"] = [1]        # no grey ellipses in the Q and alpha figures
        scene[f"contacts_{race}"] = ellipses

    # Header. The bearing is named by what is in the object (boundary dimensions, balls), not by a
    # catalogue designation: a study may use catalogue dimensions with its own internal geometry.
    header = [f"Deep-groove ball bearing {bearing.d:g} x {bearing.D:g} x {bearing.b:g} mm, Z = {row.Z}: {case}",
              f"Fr = {result.Fr:.0f} N, Fa = {result.Fa:.0f} N on the inner ring; "
              f"diametral clearance s = {bearing.s:g} mm"]
    balls_loaded = f"loaded balls {row.n_loaded} / {row.Z}"
    Q_max, alpha_max = float(np.nanmax(points["Q"])), float(np.nanmax(points["alpha"]))
    p_inner, p_outer = max(drawn["inner"]["p0"]), max(drawn["outer"]["p0"])
    # forces as stored in the result (see the module docstring); arrows of fixed length
    e_load = np.array([0.0, math.sin(result.phi_Fr), math.cos(result.phi_Fr)])      # along phi_Fr
    e_side = np.array([0.0, math.cos(result.phi_Fr), -math.sin(result.phi_Fr)])     # across it
    e_axial = math.copysign(1.0, result.Fa) * e_x                                   # along Fa
    sign = "+" if result.Fa > 0 else "-"
    if view == "side":
        # Contact line and, lighter, the radial line through the ball centre: the angle between
        # them is the contact angle. contacts() gives inner then outer of each ball, so consecutive
        # ends are the two ends of one line. The radial line is set just behind the section plane
        # (the camera is on the -y side), so the contact line covers it when alpha = 0.
        reach = (0.5 + CONTACT_LINE_OVERHANG) * bearing.Dw
        behind = np.array([0.0, 1e-3 * bearing.Dw, 0.0])
        scene["radial_lines"] = pv.line_segments_from_points(np.concatenate(
            [[ball_centres[j] - reach * e_r[j] + behind, ball_centres[j] + reach * e_r[j] + behind]
             for j in drawn["inner"]["j"]]))
        scene["radial_lines"].field_data["color"] = _REFERENCE_COLOR
        scene["contact_lines"] = pv.line_segments_from_points(np.array(line_ends))
        scene["contact_lines"].field_data["color"] = _LINE_COLOR
        # ring names inside the two sections, between the groove bottom and the bore / outer
        # surface; the name and the angle of the contact line in the gap between the rings, beside
        # the ball (two short lines of text: one long line would run over the ring section)
        gap = centre + [0.33 * bearing.b, 0.0, 0.5 * bearing.Dpw]
        labels = pv.PolyData(np.array([
            centre + [-0.45 * bearing.b, 0.0, 0.25 * (bearing.D + bearing.Dpw + bearing.Dw)],
            centre + [0.1 * bearing.b, 0.0, 0.25 * (bearing.d + bearing.Dpw - bearing.Dw)],
            gap + [0.0, 0.0, 0.1 * bearing.Dw], gap - [0.0, 0.0, 0.1 * bearing.Dw]]))
        labels["label"] = ["outer ring", "inner ring", "  black: contact line", f"  alpha = {alpha_max:.4g} deg"]
        scene["section_labels"] = labels
        # small arrows clear of the contacts: Fr in the inner ring section, Fa beside it
        length = 0.2 * bearing.b
        arrows = {"Fr": (centre + [-0.4 * bearing.b, 0.0, 0.5 * bearing.d + 0.04 * bearing.b], e_load),
                  f"Fa ({sign}x)": (centre + [0.55 * bearing.b, 0.0, 0.5 * bearing.d + 0.16 * bearing.b], e_axial)}
        header.append(f"axial section through the most loaded ball, on the load line; {balls_loaded}")
        note = "ring section schematic (shoulder height not to scale)"
        lines = {"Q": (f"at this ball: ball load Q = {Q_max:.0f} N", note),
                 "alpha": (f"at this ball: contact angle alpha = {alpha_max:.4g} deg", note),
                 "p0": (f"at this ball: p0 inner = {p_inner:.0f} MPa, outer = {p_outer:.0f} MPa",
                        f"{note}; contact ellipses drawn x{ellipse_scale:g}")}
    else:
        # Fr from the centre; Fa beside it, so the two are not read as one double arrow: seen along
        # the shaft the axial arrow is foreshortened, and the label says which way it points
        length = 0.36 * bearing.d
        page = "out of" if e_axial[0] * CAMERAS[view][0][0] > 0 else "into"
        arrows = {"Fr (radial)": (centre, e_load),
                  f"Fa ({page} page)": (centre - 0.8 * length * e_side, e_axial)}       # short: no ball under it
        note = "arrows not to scale; solid grey ball = unloaded (Q = 0)"     # true where balls are coloured
        lines = {"Q": (f"max ball load Q = {Q_max:.0f} N; {balls_loaded}", note),
                 "alpha": (f"max contact angle alpha = {alpha_max:.4g} deg; {balls_loaded}", note),
                 "p0": (f"max Hertz pressure p0 = {max(p_inner, p_outer):.0f} MPa "
                        f"({'inner' if p_inner >= p_outer else 'outer'} raceway); {balls_loaded}",
                        f"ellipses drawn x{ellipse_scale:g}; inner contact = ellipse nearer the centre; "
                        "arrows not to scale")}
    if result.Fa == 0.0:
        del arrows[list(arrows)[1]]
    for name, (start, direction) in arrows.items():
        arrow = pv.Arrow(start=start, direction=direction, scale=length, tip_radius=0.12, shaft_radius=0.045)
        arrow.field_data["color"] = _ARROW_COLOR
        scene[f"load_{name[:2]}"] = arrow
    # names beside the arrows, at mid-height and past their extent along x (text runs to the right:
    # an arrow along x would otherwise be written over); the values are in the header
    boxes = [scene[f"load_{name[:2]}"].bounds for name in arrows]
    labels = pv.PolyData(np.array([[b[1], 0.5 * (b[2] + b[3]), 0.5 * (b[4] + b[5])] for b in boxes]))
    labels["label"] = [f"  {name}" for name in arrows]          # drawn to the right of the point
    scene["load_labels"] = labels
    scene.field_data["annotation"] = header
    for field, text in lines.items():                           # printed when that field is coloured
        scene.field_data[f"annotation_{field}"] = list(text)
    direction, focal_height, half_height = CAMERAS[view]
    focal_point = centre + [0.0, 0.0, focal_height * 0.5 * bearing.Dpw]
    set_view(scene, focal_point, direction, half_height * bearing.D)
    return scene
