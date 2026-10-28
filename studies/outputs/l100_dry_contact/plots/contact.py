"""Figures of one dry contact against its Hertz reference: error sweeps and pressure cuts.

Every function takes records (``list[dict]``, units in the keys), draws with the paper style
(``common.style.PAPER``) and writes one PNG to ``results_dir`` named
``<case>_contact_<function>[_<variant>].png`` (D-0036 amendment item 10); it returns that path.
Semantics of the style sheet: a numerical result is a filled marker, a formula a solid line.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.lines import Line2D

from ...common.figures import MARKERS, figure_path, save
from ...common.style import PAPER

__all__ = ["relative_errors", "pressure_cut", "load_approach"]

def relative_errors(records: list[dict], results_dir, case: str, *, x: str, x_label: str, variant: str = "",
                    log_x: bool = True, caption: str = "", slope: float | None = None) -> Path:
    """Absolute relative error of BEM quantities against their reference, along a sweep.

    Shows: ``|rel_err|`` [%] (log axis) against the swept parameter ``x``; one colour and marker
    per ``quantity`` (delta, p0, b, load, ...), one panel per ``series`` (kernel, control, ...),
    tagged (a), (b), ... under its x axis; filled markers joined by thin lines in sweep order, x
    ticks at the swept values. The tolerance of a judged record (``tol`` > 0 and ``verdict`` not
    ``RECORD``) is drawn along the sweep as a dashed line of the quantity's colour. ``slope`` draws, in each
    panel, a thin dotted guide ``x^slope`` through the first point of the first quantity (an expected scaling).

    Parameters
    ----------
    records : list of dict
        Fields: ``x`` (the key named by ``x``: a/h, span, alpha, ... [-]), ``quantity`` (str),
        ``series`` (str, panel), ``rel_err`` [-]; optional ``tol`` [-] and ``verdict``.
    results_dir : path
        Folder of the figure (the case's git-ignored ``results/``).
    case : str
        Case id, first part of the file name.
    x, x_label : str
        Record key of the swept parameter and its axis label (style sheet label rules).
    variant : str, optional
        Last part of the file name.
    log_x : bool, optional
        Logarithmic x axis (default).
    caption : str, optional
        Caption line under the axes (inputs, convention tags such as J-16).
    slope : float, optional
        Exponent of the guide line (log-log).

    Returns
    -------
    Path
        ``<case>_contact_relative_errors[_<variant>].png``. Called by v101, v102, v103, v104, v003, v105,
        v002 (shaft errors vs settlement and helix angle; the file name keeps ``contact``), s101, v005, s003.
    """
    series = list(dict.fromkeys(r["series"] for r in records))
    quantities = list(dict.fromkeys(r["quantity"] for r in records))
    with plt.style.context(PAPER):
        ncols = min(2, len(series))
        nrows = -(-len(series) // ncols)
        fig, axes = plt.subplots(nrows, ncols, sharey=True, squeeze=False, figsize=(5.9, 2.4 * nrows + 0.3))
        for ax in axes.flat[len(series):]:
            ax.set_visible(False)
        colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        handles, any_tol = {}, False
        for n, (ax, s) in enumerate(zip(axes.flat, series)):
            for k, q in enumerate(quantities):
                rows = sorted((r for r in records if r["series"] == s and r["quantity"] == q), key=lambda r: r[x])
                if not rows:
                    continue
                colour = colours[k % len(colours)]
                (handles[q],) = ax.plot([r[x] for r in rows], [100.0 * max(abs(r["rel_err"]), 1e-12) for r in rows],
                                        marker=MARKERS[k % len(MARKERS)], color=colour, linestyle="-",
                                        linewidth=0.5, label=q)
                # nan > 0 is False: rows without a tolerance draw none
                tol = [(r[x], 100.0 * r["tol"]) for r in rows
                       if r.get("tol", 0.0) > 0.0 and r.get("verdict", "") != "RECORD"]
                if tol:
                    any_tol = True
                    ax.plot(*zip(*tol), color=colour, linestyle="--", linewidth=0.6)
            first = sorted((r for r in records if r["series"] == s and r["quantity"] == quantities[0]),
                           key=lambda r: r[x])
            if slope is not None and first:
                xs = np.array([first[0][x], first[-1][x]], dtype=float)
                y0 = 100.0 * max(abs(first[0]["rel_err"]), 1e-12)
                ax.plot(xs, y0 * (xs / xs[0]) ** slope, color="0.5", linestyle=":", linewidth=0.7)
                handles[f"slope {slope:g}"] = Line2D([], [], color="0.5", linestyle=":", linewidth=0.7)
            ax.set_yscale("log")
            if log_x:
                ax.set_xscale("log")
            ticks = sorted({round(r[x], 2) for r in records if r["series"] == s})
            ax.set_xticks(ticks, [f"{t:.3g}" for t in ticks])
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_xlabel(f"{x_label}\n({chr(97 + n)}) {s}")
        for row in axes:
            row[0].set_ylabel("|relative error| (%)")      # not eps: v003 plots the load-distribution factor eps too
        if any_tol:
            handles["tolerance"] = Line2D([], [], color="k", linestyle="--", linewidth=0.6)
        fig.legend(handles.values(), handles.keys(), loc="outside right center")
        return save(fig, figure_path(results_dir, case, "contact", "relative_errors", variant), caption)


def pressure_cut(records: list[dict], results_dir, case: str, *, variant: str = "") -> Path:
    """BEM pressure along cuts through the contact centre against the Hertz profile.

    Shows: ``p/p0_H`` (p0 of the Hertz reference) against ``xi``, the coordinate along the cut
    divided by the Hertz semi-axis of that cut [-]; the Hertz profile ``sqrt(1 - xi^2)`` (Johnson
    1985 Eq. (3.39) for a point contact, Eq. (4.45) for a line) as a solid line, the BEM nodes as
    filled markers, one marker per ``axis``.

    Parameters
    ----------
    records : list of dict
        Fields: ``axis`` (str: the cut, e.g. "b (minor)", "a (major)"), ``xi`` (coordinate / Hertz
        semi-axis of the cut) [-], ``p_over_p0_ref`` (p / p0 Hertz) [-].
    results_dir, case, variant
        As :func:`relative_errors`.

    Returns
    -------
    Path
        ``<case>_contact_pressure_cut[_<variant>].png``. Called by v101, v102, v103.
    """
    axes_ = list(dict.fromkeys(r["axis"] for r in records))
    with plt.style.context(PAPER):
        fig, ax = plt.subplots(figsize=(4.2, 2.8))
        xi = np.linspace(-1.0, 1.0, 401)
        ax.plot(xi, np.sqrt(1.0 - xi**2), color="k", linestyle="-", label="Hertz")
        for k, name in enumerate(axes_):
            rows = [r for r in records if r["axis"] == name]
            # the first series larger: both stay visible where they coincide (a circle)
            ax.plot([r["xi"] for r in rows], [r["p_over_p0_ref"] for r in rows], linestyle="none",
                    marker=MARKERS[k % len(MARKERS)], markersize=4.0 if k == 0 else 2.5, label=f"BEM, {name}")
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(0.0, 1.1)
        ax.set_xlabel(r"$\xi$ = coordinate / Hertz semi-axis")
        ax.set_ylabel(r"$p/p_{0,\mathrm{H}}$")
        ax.legend(loc="upper right", fontsize=7)
        return save(fig, figure_path(results_dir, case, "contact", "pressure_cut", variant))


def load_approach(records: list[dict], results_dir, case: str, *, slope: float = 1.5, variant: str = "",
                  caption: str = "") -> Path:
    """Normal load against approach of contacts, log-log: BEM points on their Hertz laws, with a slope guide.

    Shows: the load ``Q`` [N] against the approach ``delta`` [um], both logarithmic, one colour per ``series``
    (a contact); records whose ``source`` is "BEM" are filled markers, any other source (the Hertz law
    ``Q = K delta^1.5``) a line of the same colour. A thin dotted line of slope ``slope`` through the first
    BEM point of the first series lets the exponent be read from the shape: a Hertz contact runs parallel to it,
    a conforming one bends away at large approach.

    Parameters
    ----------
    records : list of dict
        Fields: ``series`` (str), ``source`` ("BEM" or the law's name), ``delta_m`` [m] (drawn in um: x 1e6, the
        one conversion), ``Q_N``.
    results_dir, case, variant, caption
        As :func:`relative_errors` (``caption`` a line under the axes).
    slope : float, optional
        Exponent of the guide line (1.5 for a point contact).

    Returns
    -------
    Path
        ``<case>_contact_load_approach[_<variant>].png``. Called by s101.
    """
    series = list(dict.fromkeys(r["series"] for r in records))
    with plt.style.context(PAPER):
        fig, ax = plt.subplots(figsize=(5.0, 3.6))
        colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for k, s in enumerate(series):
            colour = colours[k % len(colours)]
            for source in dict.fromkeys(r["source"] for r in records if r["series"] == s):
                rows = sorted((r for r in records if r["series"] == s and r["source"] == source),
                              key=lambda r: r["delta_m"])
                d = [1e6 * r["delta_m"] for r in rows]
                Q = [r["Q_N"] for r in rows]
                if source == "BEM":
                    ax.plot(d, Q, linestyle="none", marker=MARKERS[k % len(MARKERS)], markersize=4, color=colour,
                            label=f"{s}, BEM")
                else:
                    ax.plot(d, Q, linestyle="-" if k == 0 else "--", linewidth=0.8, color=colour, label=f"{s}, {source}")
        first = sorted((r for r in records if r["series"] == series[0] and r["source"] == "BEM"),
                       key=lambda r: r["delta_m"])
        if first:
            d0, Q0 = 1e6 * first[0]["delta_m"], first[0]["Q_N"]
            d = np.array([d0, 1e6 * first[-1]["delta_m"]])
            ax.plot(d, Q0 * (d / d0) ** slope, color="0.5", linestyle=":", linewidth=0.7, label=f"slope {slope:g}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"approach $\delta$ [um]")
        ax.set_ylabel(r"normal load $Q$ [N]")
        ax.legend(fontsize=6, loc="upper left")
        return save(fig, figure_path(results_dir, case, "contact", "load_approach", variant), caption)
