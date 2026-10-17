"""Figures of the ``validation/contact`` cases: convergence of a sweep and pressure cuts.

Data in, ``matplotlib.figure.Figure`` out; appearance only from ``style/axisforge.mplstyle``, which the
caller activates (``validation._support.style.use_style()``) before calling. Semantics of the sheet
(``style/README.md``): SlipPY results are filled markers, the Hertz judge is a solid line.

Symbols
-------
The bridge names are used: ``s_x`` is the semi-axis of the contact along rolling (``x``, array
axis 0), ``s_y`` the semi-axis across it (``y``, array axis 1), and depth is ``z/s_min`` with
``s_min = min(s_x, s_y)``. Johnson (1985) and Harris & Kotzalas write ``a`` for the major and ``b``
for the minor semi-axis. For a line contact and for a circle ``s_min`` is Johnson's ``a`` (the
semi-width, resp. the radius). On a ball-raceway contact the groove makes the ellipse long across
rolling, so ``s_x`` is the minor semi-axis ``b`` and ``s_y`` the major ``a``.
"""
from __future__ import annotations

import math

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.figure import Figure

__all__ = ["convergence", "pressure_cuts"]

_MARKERS = ("o", "s", "^", "D")            # one per quantity, resp. per build
_MIN_POINTS = 3                            # fewer points show no trend: the series is skipped


def convergence(records: list[dict], out_png=None) -> Figure:
    """Relative error of the SlipPY BEM against Hertz along the mesh and the domain sweeps.

    One panel per build for the mesh sweep: ``|error|`` of the maximum pressure, the two semi-axes
    and the approach against the grid spacing ``h`` (log-log), one marker per quantity. One panel
    for the domain sweep: signed error of the maximum pressure against the half-length ``L_x`` of
    the domain along rolling, in units of ``s_x``, one marker per build.

    Parameters
    ----------
    records : list of dict
        From ``validation._support.convergence.sweep``: ``build``, ``sweep`` (``"mesh"`` or
        ``"domain"``), ``span_x`` (``L_x/s_x`` [-]), ``h_m`` (grid spacing [m]) and the errors
        ``err_p0``, ``err_s_x``, ``err_s_y``, ``err_delta`` (fractions, against SlipPY's
        ``hertz_full`` as the bridge records them).
    out_png : path, optional
        If given, the figure is also saved there; its folder must exist.

    Returns
    -------
    matplotlib.figure.Figure
        The caller closes it. Without any series of at least 3 points the figure has no axes,
        only a note.

    Notes
    -----
    A series with fewer than 3 finite points shows no trend and is skipped; a quantity without a
    finite Hertz value (line contact: ``s_y``, approach) gives no points. Skipped series are named in
    the note line under the figure. A zero error cannot be drawn on a log axis and is dropped.
    """
    quantities = (("err_p0", "$p_0$"), ("err_s_x", "$s_x$"), ("err_s_y", "$s_y$"), ("err_delta", r"$\delta$"))
    builds = list(dict.fromkeys(r["build"] for r in records))
    skipped = []

    def points(rows, x_key, y_key, x_scale, absolute):
        """Finite (x, y) pairs of one series in plot units (y in %); None if too short."""
        xy = [(r[x_key] * x_scale, (abs(r[y_key]) if absolute else r[y_key]) * 100.0)   # fraction -> %
              for r in rows if math.isfinite(r[y_key])]
        if absolute:
            xy = [(x, y) for x, y in xy if y > 0.0]      # a zero error has no place on a log axis
        return xy if len(xy) >= _MIN_POINTS else None

    mesh_panels = {}                                       # build -> [(symbol, marker, xy)]
    for build in builds:
        mesh = sorted((r for r in records if r["build"] == build and r["sweep"] == "mesh"), key=lambda r: r["h_m"])
        drawn = []
        for (key, symbol), marker in zip(quantities, _MARKERS):
            xy = points(mesh, "h_m", key, 1e6, absolute=True)      # h [m] -> [um]
            if xy:
                drawn.append((symbol, marker, xy))
            elif any(math.isfinite(r[key]) for r in mesh):
                skipped.append(f"{symbol} {build} (mesh, {len(mesh)} point(s))")
        if drawn:
            mesh_panels[build] = drawn
    domain_series = []                                     # (build, marker, xy)
    for build, marker in zip(builds, _MARKERS):
        domain = sorted((r for r in records if r["build"] == build and r["sweep"] == "domain"),
                        key=lambda r: r["span_x"])
        xy = points(domain, "span_x", "err_p0", 1.0, absolute=False)
        if xy:
            domain_series.append((build, marker, xy))
        elif domain:
            skipped.append(f"$p_0$ {build} (domain, {len(domain)} point(s))")

    n_panels = len(mesh_panels) + bool(domain_series)
    fig = plt.figure()
    axes = fig.subplots(1, n_panels, squeeze=False)[0] if n_panels else []
    tags = iter("abcd")
    for ax, (build, drawn) in zip(axes, mesh_panels.items()):
        for symbol, marker, xy in drawn:
            x, y = zip(*xy)
            ax.loglog(x, y, linestyle="none", marker=marker, label=symbol)
        # ticks at the swept spacings, plain numbers: log minor labels (3x10^-1, 4x10^-1) collide
        ax.set_xticks(sorted({x for _, _, xy in drawn for x, _ in xy}))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_xlabel(r"$h$ ($\mu$m)")
        ax.set_ylabel(r"$|\varepsilon|$ (%)")             # absolute value: the sign is lost on a log axis
        ax.text(0.03, 0.95, f"({next(tags)}) {build.replace('_', ' ')}", transform=ax.transAxes, va="top")
    if mesh_panels:
        # one key for the quantity markers, outside the axes (direct labels would overlap: the
        # errors of the semi-axes often coincide)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="outside right upper")
    if domain_series:
        ax = axes[-1]
        for k, (build, marker, xy) in enumerate(domain_series):
            x, y = zip(*xy)
            ax.plot(x, y, linestyle="none", marker=marker)
            # direct label at the last point, above for the first build and below for the second
            ax.annotate(build.replace("_", " "), xy[-1], xytext=(-4, 6 if k % 2 == 0 else -6),
                        textcoords="offset points", ha="right", va="bottom" if k % 2 == 0 else "top")
        ax.axhline(0.0, linestyle=":", linewidth=plt.rcParams["axes.linewidth"])
        ax.set_xlabel(r"$L_x/s_x$")
        ax.set_ylabel(r"$\varepsilon_{p_0}$ (%)")          # signed
        ax.text(0.03, 0.95, f"({next(tags)})", transform=ax.transAxes, va="top")
    note = (r"$\varepsilon$ = SlipPY / Hertz (SlipPY hertz_full) - 1; $L_x$: domain half-length along rolling"
            if n_panels else "no series with 3 or more points: nothing to draw")
    if skipped:
        note += "; not drawn (< 3 points): " + ", ".join(skipped)
    fig.text(0.5, 0.0, note, ha="center", va="top", wrap=True)
    if out_png is not None:
        fig.savefig(out_png)
    return fig


def pressure_cuts(p: np.ndarray, grid, judge: dict, out_png=None) -> Figure:
    """Contact pressure along and across rolling through its maximum, against the Hertz pressure.

    Parameters
    ----------
    p : ndarray
        Pressure [Pa] on ``grid`` (rolling along axis 0), from ``static_step.run``.
    grid : slippy_bridge.contact.Grid
        The grid of the solution, node coordinates ``X``, ``Y`` [m] centred on the contact.
    judge : dict
        Hertz solution of the contact from ``references.contact.hertz`` (``hertz_point`` or
        ``hertz_line``, computed by the caller): ``p0_Pa`` [Pa], ``s_x_m``, ``s_y_m`` [m]. A
        non-finite ``s_y_m`` marks a line contact (strip unbounded along ``y``).
    out_png : path, optional
        If given, the figure is also saved there; its folder must exist.

    Returns
    -------
    matplotlib.figure.Figure
        ``p/p_0`` against ``x/s_x`` and, for a point contact, against ``y/s_y`` on the same limits;
        a line contact has no ``y`` panel (the strip is unbounded along ``y``). The caller closes it.

    Notes
    -----
    The cuts pass through the node of maximum pressure, not the centre node: a solution shifted
    off the apex would otherwise be cut off its peak. The Hertz curve is evaluated at the same
    offset of the cut, ``p = p0 sqrt(1 - (x/s_x)^2 - (y/s_y)^2)`` (Johnson 1985, ch. 4; for a line
    contact the ``y`` term is absent). SlipPY nodes are drawn as filled markers, every k-th node
    so that the Hertz line stays visible; the maximum node is always among them.
    """
    p0, s_x, s_y = judge["p0_Pa"], judge["s_x_m"], judge["s_y_m"]
    line = not math.isfinite(s_y)                          # hertz_line returns s_y_m = inf
    i0, j0 = np.unravel_index(np.argmax(p), p.shape)      # rolling = axis 0
    x_c, y_c = grid.X[i0, j0] / s_x, (0.0 if line else grid.Y[i0, j0] / s_y)   # cut offsets [-]

    cuts = [("x", grid.X[:, j0] / s_x, p[:, j0] / p0, i0, y_c)]
    if not line:
        cuts.append(("y", grid.Y[i0, :] / s_y, p[i0, :] / p0, j0, x_c))
    fig = plt.figure()
    axes = fig.subplots(1, len(cuts), sharex=True, sharey=True, squeeze=False)[0]
    for ax, (symbol, u, cut, k_max, offset), tag in zip(axes, cuts, "ab"):
        fine = np.linspace(u[0], u[-1], 801)
        # OPEN commit 4: take the references 'pressure' key
        hertz_cut = np.sqrt(np.clip(1.0 - fine**2 - offset**2, 0.0, None))   # dimensionless Hertz
        ax.plot(fine, hertz_cut, linestyle="-")
        k = max(1, len(u) // 25)                           # about 25 markers per cut
        drawn = np.arange(k_max % k, len(u), k)            # every k-th node, through the maximum
        ax.plot(u[drawn], cut[drawn], linestyle="none", marker="o")
        ax.set_xlabel(f"${symbol}/s_{symbol}$")
        if len(cuts) > 1:
            ax.text(0.03, 0.95, f"({tag})", transform=ax.transAxes, va="top")
        if symbol == "x":
            # direct labels: Hertz on its line at x/s_x = 0.6, SlipPY beside the node nearest x/s_x = -0.7
            ax.annotate("Hertz", (0.6, math.sqrt(max(1.0 - 0.6**2 - offset**2, 0.0))), xytext=(6, 6),
                        textcoords="offset points")
            i = drawn[np.argmin(np.abs(u[drawn] + 0.7))]
            ax.annotate("SlipPY", (u[i], cut[i]), xytext=(-8, 0), textcoords="offset points", ha="right",
                        va="center")
    axes[0].set_ylabel(r"$p/p_0$")
    where = (f"cut through the node of maximum pressure, $x/s_x$ = {x_c:.3g}" +
             ("" if line else f", $y/s_y$ = {y_c:.3g}") +
             f"; Hertz (references.contact.hertz): $p_0$ = {p0 * 1e-6:.4g} MPa, $s_x$ = {s_x * 1e6:.4g} $\\mu$m" +
             ("" if line else f", $s_y$ = {s_y * 1e6:.4g} $\\mu$m"))
    fig.text(0.5, 0.0, where, ha="center", va="top", wrap=True)
    if out_png is not None:
        fig.savefig(out_png)
    return fig
