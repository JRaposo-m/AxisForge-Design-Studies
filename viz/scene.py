"""Generic PyVista scene: named blocks, one field at a time, the same picture in a window and in a PNG.

A *scene* is a ``pyvista.MultiBlock`` whose blocks are named meshes. Every result is stored on the
mesh it lives on (``point_data`` / ``cell_data``); nothing is kept beside the geometry. The module
that builds a scene (``viz.bearing``, ``viz.contact``, ``viz.shaft``) also gives a ``fields`` dict::

    fields = {array name: (display name, unit, colour map)}

Only one field is coloured at a time, on every block that carries an array of that name. How the
other blocks are drawn is set by the module that builds the scene:

``block.point_data["label"]``
    Strings: the block is a set of labelled points (no surface is drawn).
``block.field_data["color"]``
    RGB in [0, 1]: drawn opaque in this colour whatever the field (arrows, lines). Lines are 4 px
    wide, or ``block.field_data["line_width"]`` px.
``block.field_data["active_only"]``
    The block is left out when it does not carry the active field.
otherwise
    Context: translucent grey with a dark silhouette, so what lies inside the assembly stays visible.

And for the whole scene:

``scene.field_data["camera_position"]``, ``scene.field_data["parallel_scale"]``
    3 x 3 array (camera position, focal point, view-up) and half the height of the frame in scene
    units, written by :func:`set_view`. The one fixed view of the scene; the projection is parallel
    (no perspective).
``scene.field_data["annotation"]``, ``scene.field_data["annotation_<array name>"]``
    Lines of text printed on every figure, then the lines printed only when that field is active
    (its maximum, what is drawn for it). Optional.

Colour maps must be perceptually uniform: sequential (viridis, plasma, ...) for a quantity with a
physical zero, diverging (one of ``DIVERGING``: white = 0) for a signed difference. Equal steps in
the value then look like equal steps in colour, which a rainbow map does not give. Grey is reserved
for "no value" (NaN) and for context. The coloured blocks are drawn without lighting, so a colour is the
colour of the bar, and with their cell edges 3 px wide in the same colours, so a small or edge-on
cell (a contact ellipse seen from the side) does not vanish.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pyvista as pv
from matplotlib.ticker import MaxNLocator

__all__ = ["MPA_PER_PA", "MM_PER_M", "DIVERGING", "revolve", "set_view", "tick_values", "clim_of", "draw",
           "explore", "save"]

# The one place of the unit conversion in viz: slippy_bridge works in SI, the figures in MPa and mm.
MPA_PER_PA = 1e-6
MM_PER_M = 1e3
# Colour maps for signed fields: their limits are made symmetric, so the middle colour is 0.
DIVERGING = ("RdBu_r", "RdBu", "coolwarm", "bwr", "seismic")

WINDOW_SIZE = (1600, 1200)      # [px] readable on a projector with the font size below
GREY = "#8c8c8c"                # NaN ("no value") and context blocks
CONTEXT_OPACITY = 0.08          # [-] blocks that do not carry the active field
FONT = "courier"                # VTK's built-in arial draws "[" and "]" rounded: "[N]" reads as "(N)"
FONT_SIZE = 16                  # PyVista units, 2 px each


def revolve(x, r, resolution: int = 72, **point_data) -> pv.PolyData:
    """Closed surface of revolution about the x axis of the closed profile ``(x, r)``.

    The profile lies in the plane y = 0 (``z = r``); ``point_data`` arrays (one value per profile
    point) are carried to every point of the circle that the profile point sweeps. ``resolution``
    is the number of steps around the axis.
    """
    profile = pv.lines_from_points(np.column_stack([x, np.zeros_like(x), r]), close=True)
    for name, values in point_data.items():
        profile[name] = values
    surface = profile.extrude_rotate(resolution=resolution, rotation_axis=(1.0, 0.0, 0.0), capping=False)
    # triangles instead of strips and the 0/360 deg seam merged (the outline of the body can then
    # be found and no seam line is drawn); zero-area faces on the axis of a solid body dropped
    return surface.triangulate().clean(tolerance=1e-9, polys_to_lines=False, lines_to_points=False)


def set_view(scene, focal_point, direction, half_height, view_up=(0.0, 0.0, 1.0)) -> None:
    """Store the one fixed view of ``scene``: where the camera looks from, and how much it shows.

    Parameters
    ----------
    scene : pyvista.MultiBlock
    focal_point : sequence of 3 float
        Centre of the frame, scene units.
    direction : sequence of 3 float
        From the focal point to the camera (any length).
    half_height : float
        Half the height of the frame, scene units. The projection is parallel, so this sets the
        size of the picture; the distance of the camera does not.
    view_up : sequence of 3 float, optional
        Direction drawn upward.
    """
    focal_point = np.asarray(focal_point, dtype=float)
    direction = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
    # far enough to be outside the scene; in a parallel projection the distance changes nothing else
    scene.field_data["camera_position"] = np.array(
        [focal_point + 20.0 * half_height * direction, focal_point, view_up]).ravel()
    scene.field_data["parallel_scale"] = [half_height]


def tick_values(lo: float, hi: float, cmap: str = "") -> np.ndarray:
    """Round tick values that cover ``[lo, hi]``: at most 5 steps of 1, 2, 2.5, 4, 5 or 8 x 10^k.

    For a diverging colour map (``DIVERGING``) the ticks are symmetric about 0. The first and the
    last tick are the colour limits of :func:`clim_of`; the colour bar is labelled at every tick.
    """
    return MaxNLocator(nbins=5, steps=[1, 2, 2.5, 4, 5, 8, 10], symmetric=cmap in DIVERGING).tick_values(lo, hi)


def clim_of(scenes, fields) -> dict:
    """Colour limits of every field over one scene or over several.

    Several scenes that are to be compared (the load cases of one bearing) must share the limits,
    otherwise the same colour means a different value in each figure. The lower limit is at most 0:
    the fields drawn here (loads, pressures, angles) have a physical zero, and a bar that starts at
    the smallest value would exaggerate the differences. The limits are the first and the last of
    the round :func:`tick_values`. A field declared with a diverging colour map (a signed
    difference) gets limits symmetric about 0, so that the middle colour (white) is 0.

    Parameters
    ----------
    scenes : pyvista.MultiBlock or sequence of pyvista.MultiBlock
    fields : dict
        ``{array name: (display name, unit, colour map)}``.

    Returns
    -------
    dict
        ``{array name: (min, max)}`` in the unit of the array, over all blocks that carry the array;
        NaN ("no value") is ignored. A field that is 0 everywhere or has no value gets (0, 1), so the
        colour bar stays valid.
    """
    scenes = [scenes] if isinstance(scenes, pv.MultiBlock) else list(scenes)
    out = {}
    for field in fields:
        ranges = [s.get_data_range(field, allow_missing=True) for s in scenes]   # (inf, -inf) if no value
        lo, hi = min(0.0, *(r[0] for r in ranges)), max(r[1] for r in ranges)
        ticks = tick_values(lo, hi if hi > lo else lo + 1.0, fields[field][2])
        out[field] = (float(ticks[0]), float(ticks[-1]))
    return out


def draw(plotter, scene, fields, field, clim=None):
    """Draw ``scene`` on ``plotter`` coloured by ``field``; the one drawing routine of window and PNG.

    Calling it again with another field replaces everything it drew; the camera is not touched.

    Parameters
    ----------
    plotter : pyvista.Plotter
    scene : pyvista.MultiBlock
    fields : dict
        ``{array name: (display name, unit, colour map)}``.
    field : str
        Key of ``fields`` to colour by. The colour bar is titled "display name [unit]".
    clim : dict, optional
        ``{array name: (min, max)}``. Default: :func:`clim_of` of this scene.
    """
    name, unit, cmap = fields[field]
    limits = (clim or clim_of(scene, fields))[field]
    for title in list(plotter.scalar_bars.keys()):          # the bar of the previous field
        plotter.remove_scalar_bar(title)
    plotter.clear_actors()
    for key in scene.keys():
        block = scene[key]
        if "label" in block.point_data:
            plotter.add_point_labels(block, "label", font_size=2 * FONT_SIZE, text_color="black",
                                     font_family=FONT, shape=None, show_points=False, always_visible=True,
                                     justification_horizontal="left", justification_vertical="center")
        elif field in block.array_names:
            look = dict(scalars=field, clim=limits, cmap=cmap, nan_color=GREY, reset_camera=False,
                        lighting=False)                     # unlit: the colour is the colour of the bar
            # horizontal bar at the bottom: room for a long "name [unit]" title; round ticks
            plotter.add_mesh(block, **look, scalar_bar_args=dict(
                title=f"{name} [{unit}]", vertical=False, fmt="%g",
                n_labels=len(tick_values(*limits, cmap)),
                title_font_size=2 * FONT_SIZE, label_font_size=2 * FONT_SIZE,
                position_x=0.25, position_y=0.03, width=0.5, height=0.1, color="black", font_family=FONT))
            # VTK draws the labels of the first bar of a window left-aligned on the ticks (later
            # bars centred): centre them, so a value is read under its tick in every figure
            plotter.scalar_bars[f"{name} [{unit}]"].GetLabelTextProperty().SetJustificationToCentered()
            plotter.add_mesh(block, **look, style="wireframe", line_width=3, show_scalar_bar=False)
        elif "color" in block.field_data:
            width = block.field_data["line_width"][0] if "line_width" in block.field_data else 4.0
            plotter.add_mesh(block, color=tuple(block.field_data["color"]), line_width=width, reset_camera=False)
        elif "active_only" not in block.field_data:
            plotter.add_mesh(block, color=GREY, opacity=CONTEXT_OPACITY, reset_camera=False, smooth_shading=True,
                             split_sharp_edges=True, silhouette=dict(color="#606060", line_width=2))
    lines = [str(s) for key in ("annotation", f"annotation_{field}") if key in scene.field_data
             for s in scene.field_data[key]]
    if lines:
        # Pixel position of the lower-left corner of the text block, from the top of the window: a
        # corner annotation would rescale the font with the length of the lines.
        plotter.add_text("\n".join(lines), font_size=FONT_SIZE, color="black",
                         font=FONT, position=(16, plotter.window_size[1] - 16 - 2.3 * FONT_SIZE * len(lines)))
    # the camera is fixed before the first actor exists: fit the near/far planes to what is drawn now
    plotter.renderer.reset_camera_clipping_range()


def _plotter(scene, off_screen):
    """Plotter with the fixed look and the scene's camera, shared by :func:`explore` and :func:`save`."""
    plotter = pv.Plotter(off_screen=off_screen, window_size=WINDOW_SIZE)
    plotter.set_background("white")
    plotter.enable_depth_peeling(number_of_peels=10)        # translucent context needs sorted layers
    # AxisForge global frame (D-0010): x is the shaft axis
    plotter.add_axes(line_width=4, color="black", viewport=(0.0, 0.0, 0.2, 0.2))
    plotter.camera_position = np.asarray(scene.field_data["camera_position"]).reshape(3, 3).tolist()
    plotter.enable_parallel_projection()                    # equal lengths look equal: no perspective
    plotter.camera.parallel_scale = float(scene.field_data["parallel_scale"][0])
    return plotter


def explore(scene, fields, clim=None):
    """Open an interactive window on ``scene``; the slider at the bottom right selects the field.

    On a change the coloured blocks, the colour map, the colour bar and the annotation are redrawn
    by :func:`draw`; the view the user has turned to is kept.

    Parameters
    ----------
    scene : pyvista.MultiBlock
    fields : dict
        ``{array name: (display name, unit, colour map)}``.
    clim : dict, optional
        ``{array name: (min, max)}``, see :func:`draw`.
    """
    plotter = _plotter(scene, off_screen=None)
    # Open the window before drawing: it then has its real size (a screen smaller than WINDOW_SIZE
    # shrinks it), which the position of the annotation needs.
    plotter.show(interactive_update=True, auto_close=False)
    shown = [next(iter(fields))]
    draw(plotter, scene, fields, shown[0], clim)

    def select(field):
        # The widget also calls this once while it is being created; drawing from inside that call
        # crashes VTK at show() (access violation with the silhouettes), so only a change redraws.
        if field != shown[0]:
            shown[0] = field
            draw(plotter, scene, fields, field, clim)
            slider.Off()                                    # draw() cleared the slider's own actor
            slider.On()                                     # with the others: put it back

    # right of the colour bar, clear of the header
    slider = plotter.add_text_slider_widget(select, list(fields), value=0, pointa=(0.80, 0.12),
                                            pointb=(0.97, 0.12), color="black")
    if not plotter.off_screen:                              # no event loop without a display (tests)
        plotter.iren.start()                                # returns when the window is closed
    plotter.close()


def save(scene, fields, folder, name, clim=None) -> list:
    """Write the scene and one figure per field.

    ``<name>.vtm`` holds the blocks with all their arrays (PyVista also writes a folder ``<name>/``
    with one file per block; open the ``.vtm`` in ParaView). ``<name>_<field>.png`` is the view of
    :func:`explore` for that field, drawn off-screen by the same :func:`draw` with the same camera.

    Parameters
    ----------
    scene : pyvista.MultiBlock
    fields : dict
        ``{array name: (display name, unit, colour map)}``; one PNG per key.
    folder : path-like
        Output folder, created if missing. Generated output: a git-ignored ``results/`` (D-0020).
    name : str
        File stem.
    clim : dict, optional
        ``{array name: (min, max)}``, see :func:`draw`. Figures that are compared need common limits.

    Returns
    -------
    list of pathlib.Path
        The ``.vtm`` first, then the PNG files in the order of ``fields``.
    """
    folder = pathlib.Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = [folder / f"{name}.vtm"]
    scene.save(paths[0])
    plotter = _plotter(scene, off_screen=True)
    for field in fields:
        draw(plotter, scene, fields, field, clim)
        paths.append(folder / f"{name}_{field}.png")
        plotter.screenshot(paths[-1])
    plotter.close()
    return paths
