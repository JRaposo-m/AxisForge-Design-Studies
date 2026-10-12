"""Shaft view: one stepped shaft with its FEM deflection and bending stress, in 3D.

What is drawn (AxisForge global frame, D-0010: x = shaft axis left to right; lengths in mm):

* the shaft, revolved from its stepped profile, moved by the deflection of the beam solution times
  a magnification printed on the figure, coloured by the deflection ``v`` [mm] or the nominal
  bending stress ``sigma_b`` [MPa];
* the same shaft undeformed, in translucent grey, as the reference of the deflection;
* the bearings as plain rings at the undeformed position, and the radial loads as arrows that
  touch the deflected shaft.

Planes and signs are those of ``ShaftResults`` and of the loads, as stored: ``v_xy`` is drawn along
+y and ``v_xz`` along +z; a radial load points along ``(0, Fy, Fz)``. Nothing is turned or
compensated here (the directions of the gear forces are an open review item of AxisForge).

Everything is already in mm, N and MPa (AxisForge units): no unit conversion in this module.
"""
from __future__ import annotations

import numpy as np
import pyvista as pv

from .scene import revolve, set_view

__all__ = ["FIELDS", "shaft_scene"]

# array name: (display name, unit, colour map)
FIELDS = {
    "v": ("Deflection magnitude v", "mm", "viridis"),      # hypot(v_xy, v_xz)
    # no "|M|/W" on the figure: VTK takes a pair of "|" for mathematical markup and garbles the line
    "sigma_b": ("Bending stress magnitude M/W, outer fibre", "MPa", "inferno"),
}

# Drawing only: the largest drawn deflection as a fraction of the shaft length (sets the
# magnification), the length of the load arrows as a fraction of the shaft length.
DRAWN_DEFLECTION_OVER_LENGTH = 0.12
ARROW_LENGTH_OVER_LENGTH = 0.15
# The one fixed view: from the -y side and above, so x runs left to right; +z up.
CAMERA_DIRECTION = (-0.30, -0.80, 0.52)
_ARROW_COLOR = (0.75, 0.1, 0.1)


def shaft_scene(shaft_system, results, case: str = "", labels: dict | None = None) -> pv.MultiBlock:
    """Scene of one shaft and its beam FEM results.

    Parameters
    ----------
    shaft_system : axisforge ShaftSystem
        ``shaft`` (sections with ``diameter``, ``inner_diameter`` [mm]; ``axial_start(i)``,
        ``axial_end(i)``, ``diameter_at(x)``), ``bearings`` (``position``, ``d``, ``D``, ``b`` [mm]),
        ``radial_loads`` (``Fy``, ``Fz`` [N]) and ``distributed_radial_loads`` (``resultant_Fy``,
        ``resultant_Fz`` [N]), each with ``position`` [mm] and ``label``.
    results : axisforge ShaftResults
        Of the same shaft: per node ``x`` [mm], ``v``, ``v_xy``, ``v_xz`` [mm], ``M`` [N mm],
        ``W`` [mm^3], and the maxima ``v_max``, ``x_v_max``, ``sigma_b_max``, ``x_sigma_b_max``.
    case : str, optional
        Case and beam theory in plain words, printed first on the figure.
    labels : dict, optional
        ``{load.label: text}``: plain words for a load in the legend, in place of the AxisForge
        load name. The force [N] is added to the text.

    Returns
    -------
    pyvista.MultiBlock
        ``shaft``: the deflected surface (point data ``v`` [mm], ``sigma_b`` [MPa],
        ``displacement`` [mm], the true one, not magnified). ``undeformed``: the same surface at
        rest, between the bearings. ``bearings``: one ring per bearing, bore ``d``, outer diameter
        ``D``, width ``b``. ``loads``, ``load_labels``: one numbered arrow per radial load and the
        legend of the numbers (absent if there is no load).
        ``field_data``: ``annotation``, ``annotation_<field>``, ``camera_position``,
        ``parallel_scale`` (read by ``viz.scene``) and ``magnification`` [-], the factor on the
        drawn deflection: the one-digit number that draws ``v_max`` as about
        ``DRAWN_DEFLECTION_OVER_LENGTH`` of the shaft length (0 if the shaft does not deflect).
        Coordinates in mm.

    Notes
    -----
    Beam model, drawn as a solid. The FEM gives one deflection ``(v_xy, v_xz)`` per node of the
    shaft axis. Every point of a cross-section is given the values of its ``x`` (linear
    interpolation between nodes) and is moved by ``(0, v_xy, v_xz)``: the sections translate, their
    rotation (the slope of the beam) is not drawn. The supports are rigid, so the shaft does not
    move at the bearings.

    Bending stress. ``sigma_b = |M| / W`` is the nominal stress of the outer fibre, without stress
    concentration: one magnitude per cross-section, although the real stress runs from ``+sigma``
    to ``-sigma`` across it. Here each section is coloured with the resultant moment ``M``
    interpolated along x and the section modulus ``W`` that AxisForge stores for that section, so
    the colour jumps exactly at a change of diameter. AxisForge's own nodal ``sigma_b`` puts on a
    step node the value of the section to its RIGHT (``ShaftResults`` docstring, Warnings); where
    the smaller section is on the left of a step, the drawn field is therefore higher than the
    stored one there, and its maximum can exceed ``results.sigma_b_max``. Both maxima are printed.
    Shoulder and end faces are coloured with the same per-section value down to the axis.

    Loads. One arrow per radial load, along its resultant ``(0, Fy, Fz)``, with the tip on the
    deflected shaft at the load position (for a distributed load: the resultant at its
    ``position``); their length is fixed (not to scale). Each arrow carries a number; the legend
    below the shaft gives its name and force. Torques and axial loads are not drawn.
    """
    shaft = shaft_system.shaft
    x_nodes = np.asarray(results.x, dtype=float)
    # Profile: per section, its two ends and the FEM nodes inside it, at the section radius. A
    # diameter step therefore has two points at the same x (the shoulder face), each with the
    # stress of its own section.
    x_out, r_out, r_in, sigma = [], [], [], []
    for i, section in enumerate(shaft.sections):
        start, end = shaft.axial_start(i), shaft.axial_end(i)
        x_i = np.unique(np.concatenate([[start, end], x_nodes[(x_nodes > start) & (x_nodes < end)]]))
        x_out.append(x_i)
        r_out.append(np.full_like(x_i, 0.5 * section.diameter))
        r_in.append(np.full_like(x_i, 0.5 * section.inner_diameter))        # 0 for a solid section
        # W of this section: AxisForge stores on the node at its left end the section to the right
        W_i = results.W[np.argmin(np.abs(x_nodes - start))]                 # [mm^3]
        sigma.append(np.interp(x_i, x_nodes, results.M) / W_i)              # [N mm / mm^3 = MPa]
    x_out, r_out, r_in, sigma = (np.concatenate(a) for a in (x_out, r_out, r_in, sigma))
    # closed loop: outer surface left to right, then the bore (or the axis) back
    x, r = np.concatenate([x_out, x_out[::-1]]), np.concatenate([r_out, r_in[::-1]])
    displacement = np.column_stack([np.zeros_like(x), np.interp(x, x_nodes, results.v_xy),
                                    np.interp(x, x_nodes, results.v_xz)])   # [mm], as stored
    undeformed = revolve(x, r, v=np.interp(x, x_nodes, results.v),
                         sigma_b=np.concatenate([sigma, sigma[::-1]]), displacement=displacement)

    length = shaft.total_length
    # one significant digit, so it can be printed; no deflection, nothing to magnify
    magnification = float(f"{DRAWN_DEFLECTION_OVER_LENGTH * length / results.v_max:.0e}") if results.v_max > 0 else 0.0
    scene = pv.MultiBlock()
    scene["shaft"] = undeformed.warp_by_vector("displacement", factor=magnification)
    # Reference, geometry only. Kept between the bearings: inside them the shaft hardly moves, and
    # two coincident surfaces (coloured and grey) would be drawn as streaks.
    rest = pv.PolyData(undeformed.points, undeformed.faces)
    if shaft_system.bearings:
        left = min(b.position + 0.5 * b.b for b in shaft_system.bearings)
        right = max(b.position - 0.5 * b.b for b in shaft_system.bearings)
        rest = rest.clip("x", origin=(left, 0.0, 0.0), invert=False).clip("x", origin=(right, 0.0, 0.0))
    scene["undeformed"] = rest
    scene["bearings"] = pv.merge([
        revolve(b.position + 0.5 * b.b * np.array([-1.0, 1.0, 1.0, -1.0]),
                0.5 * np.array([b.d, b.d, b.D, b.D])) for b in shaft_system.bearings])

    # (load, Fy, Fz, default note): a point load has its components, a distributed one its resultants [N]
    loads = ([(ld, ld.Fy, ld.Fz, "") for ld in shaft_system.radial_loads]
             + [(ld, ld.resultant_Fy, ld.resultant_Fz, ", resultant") for ld in shaft_system.distributed_radial_loads])
    # Screen axes of the fixed view (+Z up), to set the legend of the loads below the shaft: a long
    # text beside an arrow would run over the shaft, a bearing or the edge of the frame.
    half_height = 0.42 * length
    focal_point = np.array([shaft.axial_start(0) + 0.5 * length, 0.0, 0.0])
    toward_camera = np.array(CAMERA_DIRECTION) / np.linalg.norm(CAMERA_DIRECTION)
    screen_right = np.cross([0.0, 0.0, 1.0], toward_camera)
    screen_right /= np.linalg.norm(screen_right)
    screen_up = np.cross(toward_camera, screen_right)
    arrows, points, texts = [], [], []
    for load, Fy, Fz, note in loads:
        magnitude = float(np.hypot(Fy, Fz))
        if magnitude == 0.0:
            continue
        direction = np.array([0.0, Fy, Fz]) / magnitude
        # tip on the surface of the drawn (deflected) shaft, on the side the force comes from
        axis_point = np.array([load.position, magnification * np.interp(load.position, x_nodes, results.v_xy),
                               magnification * np.interp(load.position, x_nodes, results.v_xz)])
        tip = axis_point - 0.5 * shaft.diameter_at(load.position) * direction
        tail = tip - ARROW_LENGTH_OVER_LENGTH * length * direction
        arrows.append(pv.Arrow(start=tail, direction=direction, scale=ARROW_LENGTH_OVER_LENGTH * length,
                               tip_radius=0.12, shaft_radius=0.045))
        number = len(arrows)
        # the caller's words, or AxisForge's own load name made readable
        name = (labels or {}).get(load.label, load.label.replace("_", " ").replace(":", " ") + note)
        # a number at the tail of the arrow, and its line in the legend (one line per load, upward
        # from just above the colour bar); both are drawn to the right of their point
        points += [tail, focal_point - 0.92 * half_height * screen_right
                   - (0.54 + 0.062 * (number - 1)) * half_height * screen_up]
        texts += [f"  {number}", f"{number}: {name} = {magnitude:.0f} N"]
    if arrows:
        scene["loads"] = pv.merge(arrows)
        scene["loads"].field_data["color"] = _ARROW_COLOR
        load_labels = pv.PolyData(np.array(points))
        load_labels["label"] = texts
        scene["load_labels"] = load_labels

    drawn = int(np.argmax(sigma))
    if np.isclose(sigma[drawn], results.sigma_b_max, rtol=1e-3):
        stress = f"max M/W = {results.sigma_b_max:.4g} MPa at x = {results.x_sigma_b_max:g} mm (drawn = AxisForge value)"
    else:   # a step where the smaller section governs: AxisForge evaluates the section to the right
        stress = (f"max M/W: drawn {sigma[drawn]:.4g} MPa at x = {x_out[drawn]:g} mm; "
                  f"AxisForge {results.sigma_b_max:.4g} MPa at x = {results.x_sigma_b_max:g} mm")
    scene.field_data["annotation"] = [
        f"Shaft {results.name}: {case}".rstrip(": "),
        f"x from the left end, L = {length:g} mm, bearings at x = "
        + " and ".join(f"{b.position:g}" for b in shaft_system.bearings) + " mm (rigid supports)",
        "loads as stored by AxisForge (gear force directions under review), not to scale",
        f"grey: undeformed shaft and bearings; deflection drawn x{magnification:g}",
    ]
    scene.field_data["annotation_v"] = [
        f"max deflection magnitude v = {results.v_max:.4g} mm at x = {results.x_v_max:g} mm",
        "cross-sections drawn displaced, slope not shown",
    ]
    scene.field_data["annotation_sigma_b"] = [
        stress,
        "one value per section: real stress +sigma to -sigma; no notch effect at fillets",
    ]
    scene.field_data["magnification"] = [magnification]
    set_view(scene, focal_point, CAMERA_DIRECTION, half_height)
    return scene
