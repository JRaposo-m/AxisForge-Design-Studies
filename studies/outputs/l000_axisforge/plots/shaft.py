"""Figures of a shaft against its reference: fields along the axis, mesh errors, ratio sweeps.

Every function takes records (``list[dict]``, units in the keys), draws with the paper style
(``common.style.PAPER``) and writes one PNG to ``results_dir`` named
``<case>_shaft_<function>[_<variant>].png`` (D-0036 amendment item 10); it returns that path.
Semantics of the style sheet: a numerical result is a filled marker, a formula a line, a published
value an open marker. An optional ``caption`` (inputs, convention tags such as R-003 or J-16) is
written under the axes.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ...common.figures import MARKERS, figure_path, save
from ...common.style import PAPER

__all__ = ["deflection_along_shaft", "error_vs_elements", "ratio_sweep"]

_FLOOR = 1e-17          # [-] errors below this (exact zeros) are drawn here on a log axis


def deflection_along_shaft(records: list[dict], results_dir, case: str, *, supports_mm=(), loads_mm=(),
                           steps_mm=(), variant: str = "", caption: str = "") -> Path:
    """Deflection and cross-section rotation along a shaft, the computed nodes on the reference curves.

    Shows: two panels sharing the axial coordinate x [mm]: the deflection v [mm] (top) and the
    rotation theta [mrad] (bottom) of one bending plane. Per ``series`` (a beam model, a solver) one
    colour: the records whose ``source`` is "FEM" are filled markers at the nodes, every other source
    (r01, a closed form) is a line: the first solid (dashed from the second series on), any further
    source of the same series dotted. Supports are open triangles on v = 0 with a dotted vertical line,
    loads dash-dotted vertical lines, section steps (changes of diameter) thin solid grey vertical
    lines; v = 0 is drawn thin.

    Parameters
    ----------
    records : list of dict
        Fields: ``series`` (str), ``source`` ("FEM" or the reference's name), ``x_mm``, ``v_mm``,
        ``theta_rad``. One plane per call (filter before; the plane goes into ``variant``).
    results_dir : path
        Folder of the figure (the case's git-ignored ``results/``).
    case : str
        Case id, first part of the file name.
    supports_mm, loads_mm, steps_mm : sequence of float
        Support, load and section-step positions [mm] to mark.
    variant, caption : str, optional
        Last part of the file name; caption line under the axes.

    Returns
    -------
    Path
        ``<case>_shaft_deflection_along_shaft[_<variant>].png``. Called by v001, v002, s006, v005, s004,
        s003.
    """
    series = list(dict.fromkeys(r["series"] for r in records))
    with plt.style.context(PAPER):
        fig, (ax_v, ax_t) = plt.subplots(2, 1, sharex=True, figsize=(5.9, 4.2))
        colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for k, s in enumerate(series):
            colour = colours[k % len(colours)]
            refs = 0
            for source in dict.fromkeys(r["source"] for r in records if r["series"] == s):
                rows = sorted((r for r in records if r["series"] == s and r["source"] == source), key=lambda r: r["x_mm"])
                x = [r["x_mm"] for r in rows]
                v = [r["v_mm"] for r in rows]
                t = [1e3 * r["theta_rad"] for r in rows]
                if source == "FEM":
                    style = dict(linestyle="none", marker=MARKERS[k % len(MARKERS)], markersize=3.5 - 0.6 * k,
                                 color=colour)
                else:
                    style = dict(linestyle=("-" if k == 0 else "--") if refs == 0 else ":", linewidth=0.8, color=colour)
                    refs += 1
                ax_v.plot(x, v, label=f"{s}, {source}", **style)
                ax_t.plot(x, t, **style)
        for ax in (ax_v, ax_t):
            ax.axhline(0.0, color="0.6", linewidth=0.4)
            for xs in supports_mm:
                ax.axvline(xs, color="k", linestyle=":", linewidth=0.6)
            for xl in loads_mm:
                ax.axvline(xl, color="0.4", linestyle="-.", linewidth=0.6)
            for xd in steps_mm:
                ax.axvline(xd, color="0.75", linestyle="-", linewidth=0.5)
        ax_v.plot(list(supports_mm), [0.0] * len(supports_mm), linestyle="none", marker="^", markersize=7,
                  markerfacecolor="none", color="k", label="support")
        if loads_mm:
            ax_v.plot([], [], color="0.4", linestyle="-.", linewidth=0.6, label="load")
        if steps_mm:
            ax_v.plot([], [], color="0.75", linestyle="-", linewidth=0.5, label="section step")
        ax_v.set_ylabel(r"$v$ (mm)")
        ax_t.set_ylabel(r"$\theta$ (mrad)")
        ax_t.set_xlabel(r"$x$ (mm)")
        ax_v.legend(fontsize=7, loc="best")
        return save(fig, figure_path(results_dir, case, "shaft", "deflection_along_shaft", variant), caption)


def error_vs_elements(records: list[dict], results_dir, case: str, *, tol: float | None = None,
                      order_guides: dict | None = None, variant: str = "", caption: str = "") -> Path:
    """Relative error of a mesh against its reference versus the number of elements, with the tolerance band.

    Shows: ``|rel_err|`` [-] (log) against ``n_elements`` [-] (log), one colour and marker per ``series``,
    filled markers joined by thin lines; exact zeros are drawn at 1e-17. The tolerance ``tol`` is a
    dashed horizontal line with the band below it shaded (inside = pass). ``order_guides`` draws, for
    a series, the slope ``n^-p`` through its first point (thin dotted), so an O(h^p) error shows as a
    parallel line and a nodally exact element as a flat floor at round-off.

    Parameters
    ----------
    records : list of dict
        Fields: ``n_elements`` [-], ``series`` (str), ``rel_err`` [-].
    results_dir, case, variant, caption
        As :func:`deflection_along_shaft`.
    tol : float, optional
        Tolerance [-] of the judged series.
    order_guides : dict, optional
        ``{series: p}``, the formal order of that series' error.

    Returns
    -------
    Path
        ``<case>_shaft_error_vs_elements[_<variant>].png``. Called by v001, v002.
    """
    series = list(dict.fromkeys(r["series"] for r in records))
    with plt.style.context(PAPER):
        fig, ax = plt.subplots(figsize=(5.0, 3.2))
        colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for k, s in enumerate(series):
            rows = sorted((r for r in records if r["series"] == s), key=lambda r: r["n_elements"])
            n = np.array([r["n_elements"] for r in rows], dtype=float)
            e = np.maximum(np.abs([r["rel_err"] for r in rows]), _FLOOR)
            colour = colours[k % len(colours)]
            ax.plot(n, e, marker=MARKERS[k % len(MARKERS)], color=colour, linestyle="-", linewidth=0.5, label=s)
            if order_guides and s in order_guides:
                p = order_guides[s]
                ax.plot(n, e[0] * (n / n[0]) ** -p, color=colour, linestyle=":", linewidth=0.6,
                        label=rf"$n^{{-{p:g}}}$")
        if tol is not None:
            lo = min(_FLOOR, ax.get_ylim()[0])
            ax.axhline(tol, color="k", linestyle="--", linewidth=0.6, label=f"tolerance {tol:g}")
            ax.axhspan(lo, tol, color="0.9", zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ticks = sorted({r["n_elements"] for r in records})
        ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel(r"elements $n$ (-)")
        ax.set_ylabel(r"$|\varepsilon|$ (-)")
        ax.legend(fontsize=6, loc="center left")
        return save(fig, figure_path(results_dir, case, "shaft", "error_vs_elements", variant), caption)


def ratio_sweep(records: list[dict], results_dir, case: str, *, x: str, x_label: str, y_label: str,
                limit: float | None = None, log_x: bool = True, legend_outside: bool = False, variant: str = "",
                caption: str = "", band: tuple[float, float] | None = None, open_sources=()) -> Path:
    """A dimensionless result along a sweep: computed points, the closed-form curve, printed values.

    Shows: ``y`` [-] against the swept ``x``; per ``series`` one colour; ``source`` "FEM" = filled
    markers, "published" = open markers, any other source (r01, a closed form) = a line. ``limit``,
    the value the sweep must reach at its end (a slender limit, an infinite mesh), is a thin dotted
    horizontal line. When "FEM" records carry ``rel_err`` (FEM against its reference at the same x), a
    lower panel shows it (filled markers, signed, linear axis), so a residual far below the visible
    scale of the upper panel is seen.

    Parameters
    ----------
    records : list of dict
        Fields: the key named by ``x``, ``y`` [-], ``series`` (str), ``source`` (str); optional
        ``rel_err`` [-] on the "FEM" records.
    x, x_label, y_label : str
        Record key of the swept parameter and the two axis labels.
    limit : float, optional
        Limit value of ``y``.
    log_x : bool, optional
        Logarithmic x axis (default).
    legend_outside : bool, optional
        Legend to the right of the axes (many series; s004), instead of the best place inside.
    band : (float, float), optional
        A range of ``y`` shaded grey (s003: the clearance band ``|d| < s/2``, i.e. -1 ... 1 of ``d / (s/2)``).
    open_sources : sequence of str, optional
        Sources drawn as open markers like "published" (s003: a closed-form point), labelled by their name.
    results_dir, case, variant, caption
        As :func:`deflection_along_shaft`.

    Returns
    -------
    Path
        ``<case>_shaft_ratio_sweep[_<variant>].png``. Called by v001 (shear effect vs L/d; Onate's
        r_w vs elements), v002 (share of the helical couple in a reaction vs the helix angle), s006 (mesh misalignment vs span
        and helix angle, pin slopes vs span; y in um / mrad), v005, s004, s003 (reactions and ring displacement
        over the seat offset).
    """
    series = list(dict.fromkeys(r["series"] for r in records))
    residual = any("rel_err" in r for r in records if r["source"] == "FEM")
    with plt.style.context(PAPER):
        if residual:
            fig, (ax, ax_r) = plt.subplots(2, 1, sharex=True, figsize=(5.0, 4.2), height_ratios=(2, 1))
        else:
            fig, ax = plt.subplots(figsize=(5.0, 3.2))
        colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for k, s in enumerate(series):
            colour = colours[k % len(colours)]
            if residual:
                rows = sorted((r for r in records if r["series"] == s and "rel_err" in r), key=lambda r: r[x])
                ax_r.plot([r[x] for r in rows], [r["rel_err"] for r in rows], linestyle="-", linewidth=0.5,
                          marker=MARKERS[k % len(MARKERS)], markersize=4, color=colour)
            for source in dict.fromkeys(r["source"] for r in records if r["series"] == s):
                rows = sorted((r for r in records if r["series"] == s and r["source"] == source), key=lambda r: r[x])
                xs, ys = [r[x] for r in rows], [r["y"] for r in rows]
                if source == "FEM":
                    ax.plot(xs, ys, linestyle="none", marker=MARKERS[k % len(MARKERS)], markersize=4, color=colour,
                            label=f"{s}, FEM")
                elif source == "published" or source in open_sources:
                    ax.plot(xs, ys, linestyle="none", marker="o", markersize=7, markerfacecolor="none", color=colour,
                            label=f"{s}, {source}")
                else:
                    ax.plot(xs, ys, linestyle="-" if k == 0 else "--", linewidth=0.8, color=colour, label=f"{s}, {source}")
        if limit is not None:
            ax.axhline(limit, color="0.5", linestyle=":", linewidth=0.6, label=f"limit {limit:g}")
        if band is not None:
            ax.axhspan(*band, color="0.9", zorder=0, label=f"band {band[0]:g} ... {band[1]:g}")
        if log_x:
            ax.set_xscale("log")
        if residual:
            ax_r.axhline(0.0, color="0.6", linewidth=0.4)
            ax_r.set_ylabel(r"FEM / ref $-$ 1 (-)")
            ax_r.set_xlabel(x_label)
        else:
            ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if legend_outside:
            ax.legend(fontsize=5, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
        else:
            ax.legend(fontsize=7, loc="best")
        return save(fig, figure_path(results_dir, case, "shaft", "ratio_sweep", variant), caption)
