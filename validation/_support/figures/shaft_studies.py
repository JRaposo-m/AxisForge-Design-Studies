"""
validation/_support/figures/shaft_studies.py -- report figures for the
shaft FEM validation cases (field comparison, convergence, shaft layout).

Moved here on 2026-10-06 from the two copies of ``figures.py`` that lived in
each case folder (the stepped-shaft copy, a superset of the uniform one, is
the one kept: it adds ``shaft_layout`` and fixed abscissa ticks in
``convergence``).

Every function here is pure: data in, ``matplotlib.figure.Figure`` out.
Nothing is computed (differences, errors and orders are computed by the
caller) and nothing is saved -- the caller decides whether to display or
``fig.savefig``. Appearance comes only from ``style/axisforge.mplstyle``,
activated by the caller (``validation._support.style.use_style()``).

Conventions (``style/README.md``)
---------------------------------
* Axis labels: symbol, then the unit in parentheses (``$v$ (mm)``); a
  dimensionless quantity as a bare ratio. No titles: per-figure information
  goes to the ``note`` line under the figure.
* A series says what it is (``Series.kind``): a numerical result is drawn as
  filled markers, a formula as a line, a published value as open markers.
  Identity is never carried by colour alone: line style and marker differ.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

# kind -> default drawing (semantics of style/README.md); a Series may override line style and marker
_KINDS = {
    "numerical": dict(linestyle="none", marker="o"),                         # filled markers
    "theory": dict(linestyle="-", marker=None),                              # own formula: solid line
    "other": dict(linestyle="--", marker=None),                              # other formula / estimate
    "experiment": dict(linestyle="none", marker="o", markerfacecolor="none"),  # published: open markers
}


@dataclass
class Series:
    """One curve to draw.

    Attributes
    ----------
    label : str
        Legend text; an empty label keeps the series out of the legend.
    x, y : array_like
        Abscissa [mm] and ordinate (units of the plot).
    kind : {"numerical", "theory", "other", "experiment"}
        What the series is; it sets the default drawing (see ``_KINDS``).
    linestyle : str, optional
        ``"-"``, ``"--"``, ``"-."``, ``":"`` or ``"none"``; overrides the kind's default, e.g. a
        line through dense FEM nodal values.
    marker : str, optional
        Marker symbol, e.g. ``"s"``; overrides the kind's default (``"none"`` for no marker).
    kwargs : dict
        Further ``Axes.plot`` keywords (not colours: the sheet sets them).
    """

    label: str
    x: Sequence[float]
    y: Sequence[float]
    kind: str = "numerical"
    linestyle: str | None = None
    marker: str | None = None
    kwargs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in _KINDS:
            raise ValueError(f"Series.kind must be one of {sorted(_KINDS)}, got {self.kind!r}")


def _draw(ax, s: Series, **extra):
    style = dict(_KINDS[s.kind])
    if s.linestyle is not None:
        style["linestyle"] = s.linestyle
    if s.marker is not None:
        style["marker"] = s.marker
    return ax.plot(s.x, s.y, label=s.label, **style, **s.kwargs, **extra)[0]


def _supports(ax, positions: Sequence[float], names: Sequence[str]) -> None:
    for x0, name in zip(positions, names):
        ax.axvline(x0, linestyle=":", linewidth=plt.rcParams["axes.linewidth"], zorder=0)
        ax.annotate(name, xy=(x0, 0.98), xycoords=("data", "axes fraction"),
                    xytext=(3, 0), textcoords="offset points", ha="left", va="top")


def _note(fig: Figure, note: str | None) -> None:
    if note:
        fig.text(0.5, 0.0, note, ha="center", va="top", wrap=True)


def field_comparison(series: Sequence[Series], deltas: Sequence[Series], *,
                     ylabel: str, delta_label: str,
                     supports: Sequence[float] = (), support_names: Sequence[str] = (),
                     xlabel: str = r"$x$ (mm)", note: str | None = None) -> Figure:
    """Field along the shaft (a) and its difference to a reference (b).

    Parameters
    ----------
    series : sequence of Series
        Curves of the field itself, e.g. v(x) for several models.
    deltas : sequence of Series
        Differences to the reference, already computed by the caller
        (e.g. relative error in %). A delta with the label of a series takes
        that series' colour, so the two panels read alike.
    ylabel, delta_label : str
        Axis labels of panels (a) and (b), "symbol (unit)".
    supports : sequence of float, optional
        Support positions [mm], drawn as dotted verticals.
    support_names : sequence of str, optional
        Labels for the supports (e.g. ``("A", "B")``).
    xlabel : str
        Abscissa label.
    note : str, optional
        One line of information under the figure (mesh, shaft, ...).

    Returns
    -------
    matplotlib.figure.Figure
    """
    width, height = plt.rcParams["figure.figsize"]
    # two stacked panels: the sheet width, 1.2 times the sheet height
    fig, (ax, axd) = plt.subplots(2, 1, sharex=True, figsize=(width, 1.2 * height),
                                  gridspec_kw={"height_ratios": [2.2, 1.0]})
    colours = {line.get_label(): line.get_color() for line in (_draw(ax, s) for s in series)}
    for s in deltas:
        _draw(axd, s, **({"color": colours[s.label]} if s.label in colours else {}))
    for a in (ax, axd):
        _supports(a, supports, support_names if a is ax else [""] * len(supports))
    axd.axhline(0.0, linewidth=plt.rcParams["axes.linewidth"])
    ax.set_ylabel(ylabel)
    axd.set_ylabel(delta_label)
    axd.set_xlabel(xlabel)
    for a, tag in ((ax, "(a)"), (axd, "(b)")):
        a.text(0.02, 0.95, tag, transform=a.transAxes, va="top")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncols=min(3, len(series)))
    _note(fig, note)
    return fig


def convergence(h: Sequence[float], errors: Sequence[Series], *,
                ylabel: str, reference_orders: dict[int, int] | None = None,
                xlabel: str = r"$h$ (mm)", note: str | None = None) -> Figure:
    """Log-log mesh convergence plot with reference slopes.

    Parameters
    ----------
    h : sequence of float
        Characteristic element size of each mesh [mm] (used to place the
        reference slopes).
    errors : sequence of Series
        Error measure vs. element size, one series per model/quantity.
        ``x`` must hold element sizes [mm].
    ylabel : str
        Ordinate label.
    reference_orders : dict of int to int, optional
        ``{p: i}`` draws a dotted reference line e ~ h^p that follows
        series ``errors[i]``, offset below it by a factor 3 (anchored at
        its finest mesh). Default ``{2: 0}``.
    xlabel : str
        Abscissa label.
    note : str, optional
        One line of information under the figure.

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
        # a reference slope is a formula, not a result: thin dotted line in the sheet's line colour
        ax.plot(hh, yy, linestyle=":", color=plt.rcParams["lines.color"], zorder=1)
        ax.annotate(rf"$\propto h^{{{p}}}$", xy=(hh[0], yy[0]), xytext=(2, -3),
                    textcoords="offset points", ha="left", va="top")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = sorted({float(v) for e in errors for v in np.asarray(e.x, float)})
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.legend(*ax.get_legend_handles_labels(), loc="outside right upper")
    _note(fig, note)
    return fig


def shaft_layout(edges: Sequence[float], diameters: Sequence[float],
                 loads: Sequence[Series], *,
                 supports: Sequence[float] = (), support_names: Sequence[str] = (),
                 load_label: str = r"$q$ (N/mm)",
                 xlabel: str = r"$x$ (mm)", note: str | None = None) -> Figure:
    """Shaft geometry (a) and applied distributed load intensities (b).

    Parameters
    ----------
    edges : sequence of float
        Section boundaries [mm], length ``len(diameters) + 1``.
    diameters : sequence of float
        Outer diameter of each section [mm].
    loads : sequence of Series
        Load intensity curves q(x) [N/mm] to draw in the lower panel.
    supports, support_names : sequence
        Support positions [mm] and labels.
    load_label, xlabel : str
        Axis labels, "symbol (unit)".
    note : str, optional
        One line of information under the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, (ax, axq) = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [1.0, 1.4]})
    for x0, x1, d in zip(edges[:-1], edges[1:], diameters):
        ax.add_patch(Rectangle((x0, -d / 2), x1 - x0, d, fill=False))      # section outline
        ax.annotate(rf"$\varnothing${d:g}", xy=(0.5 * (x0 + x1), d / 2), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom")
    ax.axhline(0.0, linestyle="-.", linewidth=plt.rcParams["axes.linewidth"])   # shaft axis
    y_support = -max(diameters) / 2 - 2.0                  # supports just under the largest section [mm]
    ax.plot(supports, [y_support] * len(supports), linestyle="none", marker="^")
    for x0, name in zip(supports, support_names):
        ax.annotate(name, xy=(x0, y_support), xytext=(6, -2), textcoords="offset points")
    ax.set_ylim(-max(diameters) / 2 - 5.0, max(diameters) / 2 + 5.0)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_ylabel(r"$r$ (mm)")
    for s in loads:
        _draw(axq, s)
    axq.axhline(0.0, linewidth=plt.rcParams["axes.linewidth"])
    axq.set_ylabel(load_label)
    axq.set_xlabel(xlabel)
    for a, tag in ((ax, "(a)"), (axq, "(b)")):
        a.text(0.02, 0.95, tag, transform=a.transAxes, va="top")
    fig.legend(*axq.get_legend_handles_labels(), loc="outside right upper")
    _note(fig, note)
    return fig
