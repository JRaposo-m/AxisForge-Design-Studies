"""
validation/_support/figures/shaft_studies.py -- report figures for the
shaft FEM validation cases (field comparison, convergence, shaft layout).

Moved here on 2026-10-06 from the two copies of ``figures.py`` that lived in
each case folder (the stepped-shaft copy, a superset of the uniform one, is
the one kept: it adds ``shaft_layout`` and fixed abscissa ticks in
``convergence``). The style sheet is ``style/axisforge.mplstyle``
(repository root), activated through ``validation._support.style``.

Every function here is pure: data in, ``matplotlib.figure.Figure`` out.
Nothing is computed (differences, errors and orders are computed by the
caller) and nothing is saved -- the caller decides whether to display or
``fig.savefig``. Appearance comes only from ``style/axisforge.mplstyle``.

Conventions
-----------
* Axis labels follow ISO 80000-1: quantity symbol in italics, unit after
  a solidus, e.g. ``$v$ / mm``.
* Series identity is never carried by colour alone: each series has its
  own line style or marker, and a legend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.figure import Figure

from validation._support.style import INK, MUTED, STYLE_PATH, use_style  # noqa: F401  (re-exported)


@dataclass
class Series:
    """One curve to draw.

    Attributes
    ----------
    label : str
        Legend text.
    x, y : array_like
        Abscissa [mm] and ordinate (units of the plot).
    color : str, optional
        Matplotlib colour; default follows the style's categorical order.
    linestyle : str
        ``"-"``, ``"--"``, ``"-."``, ``":"`` or ``"none"`` (markers only).
    marker : str or None
        Marker symbol, e.g. ``"o"``; None for a plain line.
    """

    label: str
    x: Sequence[float]
    y: Sequence[float]
    color: str | None = None
    linestyle: str = "-"
    marker: str | None = None
    kwargs: dict = field(default_factory=dict)


def _draw(ax, s: Series) -> None:
    ax.plot(s.x, s.y, label=s.label, color=s.color, linestyle=s.linestyle,
            marker=s.marker, markerfacecolor="white" if s.marker else None,
            markeredgewidth=1.0, **s.kwargs)


def _supports(ax, positions: Sequence[float], names: Sequence[str]) -> None:
    for x0, name in zip(positions, names):
        ax.axvline(x0, color=MUTED, linestyle=":", linewidth=0.8, zorder=0)
        ax.annotate(name, xy=(x0, 0.98), xycoords=("data", "axes fraction"),
                    xytext=(3, 0), textcoords="offset points",
                    ha="left", va="top", fontsize=7, color=MUTED)


def field_comparison(series: Sequence[Series], deltas: Sequence[Series], *,
                     title: str, ylabel: str, delta_label: str,
                     supports: Sequence[float] = (), support_names: Sequence[str] = (),
                     xlabel: str = r"$x$ / mm") -> Figure:
    """Field along the shaft (top) and its difference to a reference (bottom).

    Parameters
    ----------
    series : sequence of Series
        Curves of the field itself, e.g. v(x) for several models.
    deltas : sequence of Series
        Differences to the reference, already computed by the caller
        (e.g. relative error in %). Same styles as ``series`` for
        consistent reading.
    title : str
        Panel title (left-aligned).
    ylabel, delta_label : str
        ISO 80000 axis labels for the top and bottom panels.
    supports : sequence of float, optional
        Support positions [mm], drawn as dotted verticals.
    support_names : sequence of str, optional
        Labels for the supports (e.g. ``("A", "B")``).
    xlabel : str
        Abscissa label.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, (ax, axd) = plt.subplots(2, 1, sharex=True, figsize=(6.3, 4.4),
                                  gridspec_kw={"height_ratios": [2.2, 1.0]})
    for s in series:
        _draw(ax, s)
    for s in deltas:
        _draw(axd, s)
    for a in (ax, axd):
        _supports(a, supports, support_names if a is ax else [""] * len(supports))
    axd.axhline(0.0, color=INK, linewidth=0.6)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    axd.set_ylabel(delta_label)
    axd.set_xlabel(xlabel)
    ax.legend(loc="best")
    return fig


def convergence(h: Sequence[float], errors: Sequence[Series], *,
                title: str, ylabel: str, reference_orders: dict[int, int] | None = None,
                xlabel: str = r"$h$ / mm") -> Figure:
    """Log-log mesh convergence plot with reference slopes.

    Parameters
    ----------
    h : sequence of float
        Characteristic element size of each mesh [mm] (used to place the
        reference slopes).
    errors : sequence of Series
        Error measure vs. element size, one series per model/quantity.
        ``x`` must hold element sizes [mm].
    title, ylabel : str
        Panel title and ISO 80000 ordinate label.
    reference_orders : dict of int to int, optional
        ``{p: i}`` draws a thin dashed reference line e ~ h^p that follows
        series ``errors[i]``, offset below it by a factor 3 (anchored at
        its finest mesh). Default ``{2: 0}``.
    xlabel : str
        Abscissa label.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots()
    for s in errors:
        _draw(ax, s)
    h = np.asarray(h, dtype=float)
    hh = np.array([h.min(), h.max()])
    for p, i in ({2: 0} if reference_orders is None else reference_orders).items():
        sx, sy = np.asarray(errors[i].x, float), np.asarray(errors[i].y, float)
        x0, y0 = sx[np.argmin(sx)], sy[np.argmin(sx)] / 3.0
        yy = y0 * (hh / x0) ** p
        ax.plot(hh, yy, color=MUTED, linestyle=(0, (4, 3)), linewidth=0.7, zorder=1)
        ax.annotate(rf"$\propto h^{{{p}}}$", xy=(hh[0], yy[0]), xytext=(2, -3),
                    textcoords="offset points", ha="left", va="top",
                    fontsize=8, color=MUTED)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = sorted({float(v) for e in errors for v in np.asarray(e.x, float)})
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.grid(True, which="major")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left")
    return fig


def shaft_layout(edges: Sequence[float], diameters: Sequence[float],
                 loads: Sequence[Series], *, title: str,
                 supports: Sequence[float] = (), support_names: Sequence[str] = (),
                 load_label: str = r"$q$ / (N/mm)",
                 xlabel: str = r"$x$ / mm") -> Figure:
    """Shaft geometry (top) and applied distributed load intensities (bottom).

    Parameters
    ----------
    edges : sequence of float
        Section boundaries [mm], length ``len(diameters) + 1``.
    diameters : sequence of float
        Outer diameter of each section [mm].
    loads : sequence of Series
        Load intensity curves q(x) [N/mm] to draw in the lower panel.
    title : str
        Panel title.
    supports, support_names : sequence
        Support positions [mm] and labels.
    load_label, xlabel : str
        ISO 80000 axis labels.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, (ax, axq) = plt.subplots(2, 1, sharex=True, figsize=(6.3, 3.4),
                                  gridspec_kw={"height_ratios": [1.0, 1.4]})
    for x0, x1, d in zip(edges[:-1], edges[1:], diameters):
        ax.fill_between([x0, x1], [-d / 2, -d / 2], [d / 2, d / 2],
                        color="#d9d9d4", edgecolor=INK, linewidth=0.8)
        ax.annotate(rf"$\varnothing${d:g}", xy=(0.5 * (x0 + x1), d / 2), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7, color=INK)
    ax.axhline(0.0, color=MUTED, linestyle=(0, (6, 3, 1, 3)), linewidth=0.6)
    for x0, name in zip(supports, support_names):
        ax.plot([x0], [-max(diameters) / 2 - 2.0], marker="^", color=INK, markersize=7)
        ax.annotate(name, xy=(x0, -max(diameters) / 2 - 2.0), xytext=(6, -2),
                    textcoords="offset points", fontsize=7, color=INK)
    ax.set_ylim(-max(diameters) / 2 - 5.0, max(diameters) / 2 + 5.0)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_ylabel(r"$r$ / mm")
    ax.grid(False)
    ax.set_title(title)
    for s in loads:
        _draw(axq, s)
        axq.fill_between(s.x, 0.0, s.y, color=s.color, alpha=0.15, linewidth=0)
    axq.axhline(0.0, color=INK, linewidth=0.6)
    axq.set_ylabel(load_label)
    axq.set_xlabel(xlabel)
    axq.legend(loc="upper right")
    return fig
