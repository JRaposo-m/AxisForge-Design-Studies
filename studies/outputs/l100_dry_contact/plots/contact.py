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

__all__ = ["relative_errors", "pressure_cut"]

def relative_errors(records: list[dict], results_dir, case: str, *, x: str, x_label: str, variant: str = "",
                    log_x: bool = True) -> Path:
    """Absolute relative error of BEM quantities against their reference, along a sweep.

    Shows: ``|rel_err|`` [%] (log axis) against the swept parameter ``x``; one colour and marker
    per ``quantity`` (delta, p0, b, load, ...), one panel per ``series`` (kernel, control, ...),
    tagged (a), (b), ... under its x axis; filled markers joined by thin lines in sweep order, x
    ticks at the swept values. The tolerance of a judged record (``tol`` > 0 and ``verdict`` not
    ``RECORD``) is drawn along the sweep as a dashed line of the quantity's colour.

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

    Returns
    -------
    Path
        ``<case>_contact_relative_errors[_<variant>].png``. Called by v101, v102, v103, v104.
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
            ax.set_yscale("log")
            if log_x:
                ax.set_xscale("log")
            ticks = sorted({round(r[x], 2) for r in records if r["series"] == s})
            ax.set_xticks(ticks, [f"{t:.3g}" for t in ticks])
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_xlabel(f"{x_label}\n({chr(97 + n)}) {s}")
        for row in axes:
            row[0].set_ylabel(r"$|\varepsilon|$ (%)")
        if any_tol:
            handles["tolerance"] = Line2D([], [], color="k", linestyle="--", linewidth=0.6)
        fig.legend(handles.values(), handles.keys(), loc="outside right center")
        return save(fig, figure_path(results_dir, case, "contact", "relative_errors", variant))


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
