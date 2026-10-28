"""Figures of a bearing load distribution against its judge: element loads around the ring, load-zone sweeps.

Every function takes records (``list[dict]``, units in the keys), draws with the paper style
(``common.style.PAPER``) and writes one PNG to ``results_dir`` named
``<case>_bearing_loads_<function>[_<variant>].png`` (D-0036 amendment item 10); it returns that path.
Semantics of the style sheet: a computed result (AxisForge) is a filled marker, a formula (r03) a line, a
published value an open marker. An optional ``caption`` (inputs, convention tags such as R-003) is written
under the axes.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ...common.figures import MARKERS, figure_path, save
from ...common.style import PAPER

__all__ = ["element_loads", "load_zone_sweep", "lamina_map"]

_COMPUTED = "AxisForge"     # the source drawn as filled markers
_PUBLISHED = "published"    # the source drawn as open markers


def _style(source: str, k: int, refs: int, colour, open_sources=()) -> dict:
    """Marker / line style of one (series, source): computed filled, published (or ``open_sources``) open,
    formulas lines."""
    if source == _COMPUTED:
        return dict(linestyle="none", marker=MARKERS[k % len(MARKERS)], markersize=4, color=colour)
    if source == _PUBLISHED or source in open_sources:
        return dict(linestyle="none", marker=MARKERS[k % len(MARKERS)], markersize=6, markerfacecolor="none",
                    color=colour)
    return dict(linestyle=("-" if k == 0 else "--") if refs == 0 else ":", linewidth=0.8, color=colour)


def element_loads(records: list[dict], results_dir, case: str, *, polar: bool = False, marks_deg=(),
                  variant: str = "", caption: str = "") -> Path:
    """Rolling-element loads around the bearing: computed elements on the judge's continuous load zone.

    Shows: the normal load ``Q`` [N] of each element against its azimuth ``phi`` [deg]; per ``series`` (a
    bearing, a clearance, a load) one colour. Records whose ``source`` is "AxisForge" are filled markers at the
    elements, "published" open markers, any other source (r03 at the same ring displacement, the zero-clearance
    shape ``Q_max cos^n(phi)``) a line: the first solid (dashed from the second series on), further sources of
    the same series dotted. Cartesian (default): ``phi`` in [-180, 180] deg, ``Q = 0`` drawn thin, so the
    load-zone edge ``psi_l`` shows where the line meets 0. Polar: ``Q`` as the radius at the angle ``phi``
    (0 at the top, counter-clockwise: the frame named in ``caption``); ``marks_deg`` (a load direction) are
    drawn as dash-dotted radial lines.

    Parameters
    ----------
    records : list of dict
        Fields: ``series`` (str), ``source`` ("AxisForge", "published" or the judge's name), ``phi_deg``,
        ``Q_N``. The azimuth frame (local, 0 on the load line; or global, D-0010) is the caller's: name it in
        ``caption`` with its convention tag (R-003 for the global frame).
    results_dir : path
        Folder of the figure (the case's git-ignored ``results/``).
    case : str
        Case id, first part of the file name.
    polar : bool, optional
        Polar axes instead of Cartesian.
    marks_deg : sequence of float, optional
        Azimuths [deg] to mark (applied load direction, expected Q_max).
    variant, caption : str, optional
        Last part of the file name; caption line under the axes.

    Returns
    -------
    Path
        ``<case>_bearing_loads_element_loads[_<variant>].png``. Called by v003, s001, s002.
    """
    series = list(dict.fromkeys(r["series"] for r in records))
    with plt.style.context(PAPER):
        if polar:
            fig, ax = plt.subplots(figsize=(4.6, 4.6), subplot_kw={"projection": "polar"})
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(1)
        else:
            fig, ax = plt.subplots(figsize=(5.9, 3.2))
        colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for k, s in enumerate(series):
            colour, refs = colours[k % len(colours)], 0
            for source in dict.fromkeys(r["source"] for r in records if r["series"] == s):
                rows = [r for r in records if r["series"] == s and r["source"] == source]
                phi = np.array([r["phi_deg"] for r in rows], dtype=float)
                if not polar:
                    phi = (phi + 180.0) % 360.0 - 180.0            # [-180, 180): the load line in the middle
                order = np.argsort(phi)
                phi, Q = phi[order], np.array([r["Q_N"] for r in rows], dtype=float)[order]
                style = _style(source, k, refs, colour)
                if source not in (_COMPUTED, _PUBLISHED):
                    refs += 1
                ax.plot(np.radians(phi) if polar else phi, Q, label=f"{s}, {source}", **style)
        for m in marks_deg:
            if polar:
                ax.plot([np.radians(m)] * 2, [0.0, ax.get_ylim()[1]], color="k", linestyle="-.", linewidth=0.7)
            else:
                ax.axvline((m + 180.0) % 360.0 - 180.0, color="k", linestyle="-.", linewidth=0.7)
        if not polar:
            ax.axhline(0.0, color="0.6", linewidth=0.4)
            ax.set_xlim(-180.0, 180.0)
            ax.set_xticks(range(-180, 181, 45))
            ax.set_xlabel(r"azimuth $\varphi$ [deg]")
            ax.set_ylabel(r"element load $Q$ [N]")
        ax.legend(fontsize=6, loc="upper right" if not polar else "lower left")
        return save(fig, figure_path(results_dir, case, "bearing_loads", "element_loads", variant), caption)


def load_zone_sweep(records: list[dict], results_dir, case: str, *, x: str, x_label: str, panels: dict,
                    limits: dict | None = None, log_x: bool = False, x_marks: dict | None = None,
                    open_sources: tuple = (), variant: str = "", caption: str = "") -> Path:
    """Load-zone quantities along a sweep: computed points on the judge's curves, with their limit values.

    Shows: one panel per ``quantity`` (e.g. the load distribution factor ``eps`` and Stribeck's
    ``Z Q_max / F_r``), the value ``y`` against the swept ``x``; per ``series`` one colour; "AxisForge" =
    filled markers, "published" = open markers, any other source (r03) = a line. ``limits`` (a quantity's
    known value at a limit of the sweep: ``eps = 0.5`` and 4.37 at zero clearance) are thin dotted
    horizontal lines with their value written at the right edge.

    Parameters
    ----------
    records : list of dict
        Fields: the key named by ``x``, ``quantity`` (str), ``y``, ``series`` (str), ``source`` (str).
    x, x_label : str
        Record key of the swept parameter and its axis label.
    panels : dict
        ``quantity -> y axis label``, in panel order.
    limits : dict, optional
        ``quantity -> limit value``.
    log_x : bool, optional
        Logarithmic x axis.
    x_marks : dict, optional
        ``label -> x``: dash-dotted vertical lines in every panel (a validity limit of the sweep, e.g. the
        ISO/TS 16281 4' tilt), the label at the top of the first panel.
    open_sources : tuple of str, optional
        Further sources drawn as open markers, like "published" (a point of another case, e.g. s001 at psi = 0).
    results_dir, case, variant, caption
        As :func:`element_loads`.

    Returns
    -------
    Path
        ``<case>_bearing_loads_load_zone_sweep[_<variant>].png``. Called by v003, s001, s002.
    """
    series = list(dict.fromkeys(r["series"] for r in records))
    limits = limits or {}
    with plt.style.context(PAPER):
        fig, axes = plt.subplots(len(panels), 1, sharex=True, squeeze=False, figsize=(5.0, 1.0 + 2.0 * len(panels)))
        colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for ax, (quantity, y_label) in zip(axes[:, 0], panels.items()):
            for k, s in enumerate(series):
                colour, refs = colours[k % len(colours)], 0
                for source in dict.fromkeys(r["source"] for r in records if r["series"] == s):
                    rows = sorted((r for r in records if r["series"] == s and r["source"] == source
                                   and r["quantity"] == quantity), key=lambda r: r[x])
                    if not rows:
                        continue
                    style = _style(source, k, refs, colour, open_sources)
                    if source not in (_COMPUTED, _PUBLISHED, *open_sources):
                        refs += 1
                    ax.plot([r[x] for r in rows], [r["y"] for r in rows], label=f"{s}, {source}", **style)
            for name, xm in (x_marks or {}).items():
                ax.axvline(xm, color="k", linestyle="-.", linewidth=0.6)
                if ax is axes[0, 0]:
                    ax.annotate(name, (xm, 1.0), xycoords=("data", "axes fraction"), fontsize=6, ha="center",
                                va="bottom")
            if quantity in limits:
                ax.axhline(limits[quantity], color="k", linestyle=":", linewidth=0.6)
                ax.annotate(f"{limits[quantity]:g}", (1.0, limits[quantity]), xycoords=("axes fraction", "data"),
                            fontsize=6, ha="left", va="center")
            ax.set_ylabel(y_label)
            if log_x:
                ax.set_xscale("log")
        axes[-1, 0].set_xlabel(x_label)
        axes[0, 0].legend(fontsize=6, ncol=2, loc="best")
        return save(fig, figure_path(results_dir, case, "bearing_loads", "load_zone_sweep", variant), caption)


def lamina_map(records: list[dict], results_dir, case: str, *, panel: str, panel_label: str = "",
               variant: str = "", caption: str = "") -> Path:
    """Lamina loads of a roller bearing as a colour map: rolling elements around the ring by lamina along the roller.

    Shows: one panel per value of ``panel`` (e.g. the tilt psi), the lamina load ``q`` [N] in colour (one scale
    for all panels, white = 0) against the element azimuth ``phi`` [deg] (x, wrapped to [-180, 180)) and the
    lamina centre ``x_k`` [mm] along the roller (y). An untilted roller shows a band symmetric about
    ``x_k = 0`` (the profile relieves the ends); a tilt moves the load to one end on one side of the ring and to
    the other end on the other side. The sign of ``x_k`` is that of ISO/TS 16281 Fig. 3 / Eq. (41) as given by
    AxisForge ``lamina_records`` (it ties the map to the sign of psi). Lamina loads are independent nonlinear
    springs (Eq. (36), Winkler-type, no shear coupling), not edge pressures.

    Parameters
    ----------
    records : list of dict
        Fields: the key named by ``panel``, ``phi_deg`` (the roller's azimuth; local or global, name it in
        ``caption`` with its convention tag), ``x_mm`` (lamina centre), ``q_N``.
    panel : str
        Record key whose values make the panels, in order of appearance.
    panel_label : str, optional
        Text before the panel value in each panel title.
    results_dir, case, variant, caption
        As :func:`element_loads`.

    Returns
    -------
    Path
        ``<case>_bearing_loads_lamina_map[_<variant>].png``. Called by s002.
    """
    values = list(dict.fromkeys(r[panel] for r in records))
    q_max = max(r["q_N"] for r in records) or 1.0
    with plt.style.context(PAPER):
        fig, axes = plt.subplots(1, len(values), sharey=True, squeeze=False, figsize=(1.0 + 1.6 * len(values), 3.0))
        for ax, v in zip(axes[0], values):
            rows = [r for r in records if r[panel] == v]
            phi = np.array([(r["phi_deg"] + 180.0) % 360.0 - 180.0 for r in rows])
            ph, xs = np.unique(phi), np.unique([r["x_mm"] for r in rows])
            Q = np.zeros((len(xs), len(ph)))
            for f, r in zip(phi, rows):
                Q[np.searchsorted(xs, r["x_mm"]), np.searchsorted(ph, f)] = r["q_N"]
            mesh = ax.pcolormesh(ph, xs, np.ma.masked_equal(Q, 0.0), shading="nearest", cmap="viridis", vmin=0.0,
                                 vmax=q_max)
            ax.set_title(f"{panel_label}{v:g}", fontsize=7)
            ax.set_xlim(-180.0, 180.0)
            ax.set_xticks((-180, -90, 0, 90, 180))
            ax.set_xlabel(r"$\varphi$ [deg]")
        axes[0, 0].set_ylabel(r"lamina $x_k$ [mm]")
        fig.colorbar(mesh, ax=axes[0].tolist(), label=r"$q_{jk}$ [N]")
        return save(fig, figure_path(results_dir, case, "bearing_loads", "lamina_map", variant), caption)
