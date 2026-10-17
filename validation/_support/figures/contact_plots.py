"""Figures of the ``validation/contact`` cases: convergence of a sweep, pressure cuts, subsurface stresses
(depth profile on the axis, contours under a line contact) and the EHL scaffolds (centre line, film survey).

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
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.figure import Figure

from references.contact.hertz import hertz_pressure
from validation._support.figures import _note, _stacked, _tag, _words

__all__ = ["convergence", "ehl_centreline", "film_survey", "pressure_cuts", "subsurface_contours", "subsurface_depth"]

_MARKERS = ("o", "s", "^", "D")            # one per quantity (mesh panels)
_BUILD_MARKERS = ("+", "x")                # one per build (domain panel): never a quantity marker
_MIN_POINTS = 3                            # fewer points show no trend: the series is skipped
_SAME_RTOL = 1e-6                          # two builds whose mesh errors agree to this are drawn once
_LOG_PAD = 1.3                             # factor of room below and above the data on a log axis


def convergence(records: list[dict], out_png=None, *, criterion: tuple[str, float] | None = None,
                default_span: float | None = None) -> Figure:
    """Relative error of the SlipPY BEM against Hertz along the mesh and the domain sweeps.

    One panel per build for the mesh sweep: ``|error|`` of the maximum pressure, the two semi-axes
    and the approach against the grid spacing ``h`` (log-log), one marker per quantity, the same
    error scale in every mesh panel. One panel for the domain sweep: signed error of the maximum
    pressure against the half-length ``L_x`` of the domain along rolling, in units of ``s_x``, one
    marker per build (``+``, ``x``: never a quantity marker), labelled directly.

    Parameters
    ----------
    records : list of dict
        From ``validation._support.convergence.sweep``: ``build``, ``sweep`` (``"mesh"`` or
        ``"domain"``), ``span_x`` (``L_x/s_x`` [-]), ``h_m`` (grid spacing [m]) and the errors
        ``err_p0``, ``err_s_x``, ``err_s_y``, ``err_delta`` (fractions, against SlipPY's own Hertz
        solution as the bridge records them).
    out_png : path, optional
        If given, the figure is also saved there; its folder must exist.
    criterion : (str, float), optional
        ``(label, limit [%])`` of the case criterion, e.g. ``("E3", 1.0)``, written in the note
        line only. No line is drawn: the case judges the criterion against the independent Hertz
        reference, while the plotted errors are against SlipPY's own Hertz solution.
    default_span : float, optional
        The case's default domain half-length ``L_x/s_x`` [-], drawn as a dotted vertical in the
        domain panel and labelled "default domain".

    Returns
    -------
    matplotlib.figure.Figure
        The caller closes it. Without any series of at least 3 points the figure has no axes,
        only a note.

    Notes
    -----
    A series with fewer than 3 finite points shows no trend and is skipped; a quantity without a
    finite Hertz value (line contact: ``s_y``, approach) gives no points. Skipped series are named in
    the note line under the figure. A zero error cannot be drawn on a log axis and is dropped. When
    a build gives the same mesh errors as an earlier one (to a relative 1e-6), it is not drawn
    again and the note says so.
    """
    quantities = (("err_p0", "$p_0$"), ("err_s_x", "$s_x$"), ("err_s_y", "$s_y$"), ("err_delta", r"$\delta$"))
    builds = list(dict.fromkeys(r["build"] for r in records))
    skipped, same = [], []

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
        if not drawn:
            continue
        twin = next((b for b, d in mesh_panels.items() if len(d) == len(drawn) and all(
                     s1 == s2 and np.allclose(xy1, xy2, rtol=_SAME_RTOL, atol=0.0)
                     for (s1, _, xy1), (s2, _, xy2) in zip(d, drawn))), None)
        if twin is None:
            mesh_panels[build] = drawn
        else:
            same.append(f"{build} build identical to {twin}, drawn once")
    domain_series = []                                     # (build, marker, xy)
    for build, marker in zip(builds, _BUILD_MARKERS):
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
    if mesh_panels:
        values = [y for drawn in mesh_panels.values() for _, _, xy in drawn for _, y in xy]
        lo, hi = min(values), max(values)
    for ax, (build, drawn) in zip(axes, mesh_panels.items()):
        for symbol, marker, xy in drawn:
            x, y = zip(*xy)
            ax.loglog(x, y, linestyle="none", marker=marker, label=symbol)
        ax.set_ylim(lo / _LOG_PAD, hi * _LOG_PAD)         # one error scale for every mesh panel
        # plain numbers on the log error axis ("0.6", not "6x10^-1"), majors and minors alike
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
        ax.yaxis.set_minor_formatter(mticker.FormatStrFormatter("%g"))
        # ticks at the swept spacings, 3 significant digits: log minor labels (3x10^-1, 4x10^-1) collide
        ax.set_xticks(sorted({x for _, _, xy in drawn for x, _ in xy}))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3g"))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_xlabel(r"$h$ ($\mu$m)")
        ax.set_ylabel(r"$\vert\varepsilon\vert$ (%)")   # absolute value: the sign is lost on a log axis
        _tag(ax, f"({next(tags)}) {build}" if n_panels > 1 else build)   # one panel: no letter
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
            ax.annotate(_words(build), xy[-1], xytext=(-6, 6 if k % 2 == 0 else -6), textcoords="offset points",
                        ha="right", va="bottom" if k % 2 == 0 else "top")
        ax.axhline(0.0, linestyle=":", linewidth=plt.rcParams["axes.linewidth"])
        if default_span is not None:
            # labelled at the bottom inside the frame, left of its line, clear of the panel tag above
            ax.axvline(default_span, linestyle=":", linewidth=plt.rcParams["axes.linewidth"])
            ax.annotate("default domain", (default_span, 0.0), xycoords=("data", "axes fraction"),
                        xytext=(-3, 3), textcoords="offset points", ha="right", va="bottom")
        ax.set_xticks(sorted({x for _, _, xy in domain_series for x, _ in xy}))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3g"))
        ax.set_xlabel(r"$L_x/s_x$")
        ax.set_ylabel(r"$\varepsilon_{p_0}$ (%)")          # signed
        if n_panels > 1:
            _tag(ax, f"({next(tags)})")
    note = (r"$\varepsilon$: relative error of the SlipPY result against SlipPY's own Hertz solution"
            + ("; $L_x$: half-length of the computational domain along rolling" if domain_series else "")
            if n_panels else "no series with 3 or more points: nothing to draw")
    if criterion is not None:
        note += f"; criterion {criterion[0]}: {criterion[1]:g} % against the independent Hertz reference (case table)"
    if same:
        note += "; " + "; ".join(same)
    if skipped:
        note += "; not drawn (< 3 points): " + ", ".join(skipped)
    _note(fig, note)
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
    off the apex would otherwise be cut off its peak. The Hertz curve is the judge's pressure
    (``references.contact.hertz.hertz_pressure``, a closed-form evaluation, no solve) along the
    same cut. SlipPY nodes are drawn as filled markers, every k-th node so that the Hertz line
    stays visible; the maximum node is always among them.
    """
    p0, s_x, s_y = judge["p0_Pa"], judge["s_x_m"], judge["s_y_m"]
    line = not math.isfinite(s_y)                          # hertz_line returns s_y_m = inf
    i0, j0 = np.unravel_index(np.argmax(p), p.shape)      # rolling = axis 0
    x0_m, y0_m = grid.X[i0, j0], grid.Y[i0, j0]            # the node of maximum pressure [m]

    # (symbol, node coordinate [m], SlipPY cut [Pa], index of the maximum, semi-axis [m])
    cuts = [("x", grid.X[:, j0], p[:, j0], i0, s_x)]
    if not line:
        cuts.append(("y", grid.Y[i0, :], p[i0, :], j0, s_y))
    fig = plt.figure()
    axes = fig.subplots(1, len(cuts), sharex=True, sharey=True, squeeze=False)[0]
    for ax, (symbol, coord_m, cut_Pa, k_max, s_m), tag in zip(axes, cuts, "ab"):
        fine_m = np.linspace(coord_m[0], coord_m[-1], 801)
        # the judge along the same cut: the other coordinate held at the node of maximum pressure
        hertz_Pa = hertz_pressure(judge, fine_m, y0_m) if symbol == "x" else hertz_pressure(judge, x0_m, fine_m)
        u, fine, cut = coord_m / s_m, fine_m / s_m, cut_Pa / p0   # dimensionless
        ax.plot(fine, hertz_Pa / p0, linestyle="-")
        # about 12 markers across the contact width in every cut, so (a) and (b) of a circular
        # contact look alike whatever the grid extent along each axis
        k = max(1, int(np.count_nonzero(np.abs(u) < 1.0)) // 12)
        drawn = np.arange(k_max % k, len(u), k)            # every k-th node, through the maximum
        ax.plot(u[drawn], cut[drawn], linestyle="none", marker="o")
        ax.set_xlabel(f"${symbol}/s_{symbol}$")
        if len(cuts) > 1:
            _tag(ax, f"({tag})")
        if symbol == "x":
            # direct labels: Hertz on its line at x/s_x = 0.6, SlipPY beside the node nearest x/s_x = -0.7
            j = np.argmin(np.abs(fine - 0.6))
            ax.annotate("Hertz", (fine[j], hertz_Pa[j] / p0), xytext=(6, 6), textcoords="offset points")
            i = drawn[np.argmin(np.abs(u[drawn] + 0.7))]
            ax.annotate("SlipPY", (u[i], cut[i]), xytext=(-8, 0), textcoords="offset points", ha="right",
                        va="center")
    axes[0].set_ylabel(r"$p/p_0$")
    where = (f"cut through the node of maximum pressure, $x/s_x$ = {x0_m / s_x:.3g}" +
             ("" if line else f", $y/s_y$ = {y0_m / s_y:.3g}") +
             f"; Hertz: $p_0$ = {p0 * 1e-6:.4g} MPa, $s_x$ = {s_x * 1e6:.4g} $\\mu$m" +
             ("" if line else f", $s_y$ = {s_y * 1e6:.4g} $\\mu$m"))
    _note(fig, where)
    if out_png is not None:
        fig.savefig(out_png)
    return fig


# key of a references.contact.hertz stress dict -> (axis label, line style); drawn in this order
_ROUND_LEVELS = (0.15, 0.20, 0.25, 0.28, 0.29)   # tau1/p0 contour values of Johnson Fig. 4.5b [-]

_STRESS = {
    "sigma_x_over_p0": (r"$\sigma_x/p_0$", "-"),
    "sigma_r_over_p0": (r"$\sigma_r/p_0$", "-"),
    "sigma_z_over_p0": (r"$\sigma_z/p_0$", "--"),
    "tau1_over_p0": (r"$\tau_1/p_0$", "-."),
}


def subsurface_depth(z_over_s_min, theory: dict, *, numerical: dict | None = None,
                     published: Sequence[tuple[float, float, str]] = (), note: str | None = None) -> Figure:
    """Stresses on the axis of a Hertz contact against depth (Johnson 1985, Figs 4.3 and 4.5a manner).

    Stress over ``p0`` on the abscissa, depth ``z/s_min`` downward on the ordinate. The closed forms
    are lines labelled directly, the maximum of ``tau1`` is marked with an ``x`` and its value written
    in the note line; published check points are open markers, numerical (BEM) values filled markers.

    Parameters
    ----------
    z_over_s_min : array_like
        Depth below the surface over ``s_min = min(s_x, s_y)`` [-] (Johnson's ``z/a`` for a line
        or a circular contact).
    theory : dict
        A stress dict of ``references.contact.hertz`` evaluated on the axis at ``z_over_s_min``
        (``hertz_line_stress(0, z)`` or ``hertz_point_axis_stress(z, nu)``): keys
        ``sigma_x_over_p0`` or ``sigma_r_over_p0``, ``sigma_z_over_p0``, ``tau1_over_p0`` [-].
        Other keys (``tau_xz_over_p0``, zero on the axis) are not drawn.
    numerical : dict, optional
        ``{key: (z_over_s_min, values)}`` of a numerical solution, keys as in ``theory``.
        # OPEN R-268, R-269: the SlipPY subsurface field is not drawn until its defects are ruled.
    published : sequence of (float, float, str), optional
        ``(stress_over_p0, z_over_s_min, source)`` printed check points, e.g. Johnson's
        ``(0.300, 0.785, "Johnson 1985, Table 4.1")`` for the line contact.
    note : str, optional
        Line under the figure (case, Poisson's ratio).

    Returns
    -------
    matplotlib.figure.Figure
        The caller closes it.
    """
    z = np.asarray(z_over_s_min, dtype=float)
    fig, ax = plt.subplots()
    drawn = [k for k in _STRESS if k in theory]
    curves = {k: np.asarray(theory[k], dtype=float) for k in drawn}
    values = np.concatenate(list(curves.values()))
    lo, hi = values.min(), values.max()
    for i, key in enumerate(drawn):
        label, linestyle = _STRESS[key]
        y = curves[key]
        ax.plot(y, z, linestyle=linestyle, color=plt.rcParams["lines.color"])
        # direct label beside the curve, each at its own depth so that the labels do not stack, on
        # the side with more free room up to the next curve, the zero line or the frame
        j = int(np.argmin(np.abs(z - z.max() * (0.35 + 0.2 * i))))
        others = np.array([c[j] for k2, c in curves.items() if k2 != key] + [0.0])
        room_left = y[j] - max([lo] + [v for v in others if v < y[j]])
        room_right = min([hi] + [v for v in others if v > y[j]]) - y[j]
        left = room_left > room_right
        ax.annotate(label, (y[j], z[j]), xytext=(-6 if left else 6, 0), textcoords="offset points",
                    ha="right" if left else "left", va="center")
    maximum = None
    if "tau1_over_p0" in theory:
        t1 = curves["tau1_over_p0"]
        k = int(np.argmax(t1))
        ax.plot([t1[k]], [z[k]], linestyle="none", marker="x", color=plt.rcParams["lines.color"])
        # its value in the note line: inside the axes the text would cross the curves
        dz = float(np.min(np.diff(z)))
        maximum = (rf"$\times$: $\tau_{{1,\max}} = {t1[k]:.3f}\,p_0$ at $z = {z[k]:.3f}\,s_\mathrm{{min}}$ "
                   rf"on the grid, step $\Delta z = {dz:.2g}\,s_\mathrm{{min}}$")
    # OPEN R-268, R-269: accepted for a numerical field; the SlipPY subsurface field is not passed yet
    for key, (zn, values) in (numerical or {}).items():
        ax.plot(np.asarray(values, dtype=float), np.asarray(zn, dtype=float), linestyle="none", marker="o",
                label=f"numerical {_STRESS[key][0]}")
    for value, depth, source in published:
        ax.plot([value], [depth], linestyle="none", marker="o", markerfacecolor="none",
                color=plt.rcParams["lines.color"], label=source)
    ax.axvline(0.0, linewidth=plt.rcParams["axes.linewidth"])
    ax.set_ylim(z.max(), z.min())                          # depth downward, as in Johnson
    ax.set_xlabel(r"$\sigma/p_0$")
    ax.set_ylabel(r"$z/s_\mathrm{min}$")
    if published or numerical:
        ax.legend(loc="lower left")                        # frameless; deep stresses are small, left is free
    _note(fig, "; ".join(t for t in (maximum, note) if t))
    return fig


def subsurface_contours(x_over_s_x, z_over_s_min, tau1_over_p0, *, levels: Sequence[float] | None = None,
                        note: str | None = None) -> Figure:
    """Contours of the maximum shear stress ``tau1/p0`` under a line contact (Johnson 1985, Fig. 4.5b manner).

    Black labelled line contours in the ``x-z`` plane, depth downward, equal aspect; the maximum
    is marked with an ``x`` and its value written in the note line. Line contact now; a point contact needs the
    full 3-D field (FG-09, deferred).

    Parameters
    ----------
    x_over_s_x : array_like, shape (n_x,)
        Position along rolling over the semi-width ``s_x`` [-].
    z_over_s_min : array_like, shape (n_z,)
        Depth over ``s_min`` (= ``s_x`` for a line contact) [-].
    tau1_over_p0 : array_like, shape (n_z, n_x)
        In-plane maximum shear over ``p0`` (``references.contact.hertz.hertz_line_stress`` on the
        grid), rows = depth.
    levels : sequence of float, optional
        Contour levels [-]; default the round values 0.15, 0.20, 0.25, 0.28, 0.29 that lie
        inside the field (Johnson Fig. 4.5b manner), else 0.5-0.98 of the maximum, rounded.
    note : str, optional
        Line under the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The caller closes it.

    Notes
    -----
    Each contour label sits where its contour crosses a column below the maximum (searched from the
    maximum's column to the left), not near the surface where the low levels crowd. The contact
    strip ``|x/s_x| <= 1`` is drawn as a thick line at the surface. The maximum is read on the grid,
    so the note gives the grid step: the depth is known to that step (0.787 on a 0.0067 step against
    Johnson's 0.785).
    """
    x, z = np.asarray(x_over_s_x, dtype=float), np.asarray(z_over_s_min, dtype=float)
    t1 = np.asarray(tau1_over_p0, dtype=float)
    k = np.unravel_index(int(np.argmax(t1)), t1.shape)
    if levels is None:
        levels = [lv for lv in _ROUND_LEVELS if lv < t1[k]]     # round values inside the field
        if len(levels) < 3:                                      # a field of other magnitude
            levels = np.round(t1[k] * np.array([0.5, 0.67, 0.83, 0.93, 0.98]), 2)
    levels = np.asarray(levels, dtype=float)
    fig, ax = plt.subplots()
    cs = ax.contour(x, z, t1, levels=levels, colors=plt.rcParams["lines.color"])
    # each label where its contour crosses a column below the maximum: start on the maximum's column
    # and move left until the level is crossed deeper than the maximum, so the labels run down and
    # out from the maximum and never crowd the surface
    positions = []
    for level in levels:
        for col in range(k[1], -1, -1):
            crossings = np.flatnonzero(np.diff(np.sign(t1[:, col] - level)))
            # below the maximum and above the bottom tenth: a label at the frame is clipped
            deep = crossings[(z[crossings] > z[k[0]]) & (z[crossings] < 0.9 * z.max())]
            if len(deep):
                positions.append((x[col], z[deep[0]]))
                break
    for label in ax.clabel(cs, fmt="%.3g", inline=True, manual=positions):
        label.set_rotation(0)                              # upright, readable on a downward depth axis
    ax.plot([x[k[1]]], [z[k[0]]], linestyle="none", marker="x", color=plt.rcParams["lines.color"])
    # the contact strip |x/s_x| <= 1 on the surface: a thick line, labelled directly below it
    ax.plot([-1.0, 1.0], [0.0, 0.0], linestyle="-", linewidth=3.0 * plt.rcParams["lines.linewidth"],
            color=plt.rcParams["lines.color"], solid_capstyle="butt", clip_on=False)
    ax.annotate("contact", (0.0, 0.0), xytext=(0, 4), textcoords="offset points", ha="center",
                va="bottom")                               # above the frame, clear of the contours
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(z.max(), z.min())                          # depth downward
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x/s_x$")
    ax.set_ylabel(r"$z/s_\mathrm{min}$")
    dz = float(np.min(np.diff(z)))
    maximum = (rf"$\times$: $\tau_{{1,\max}} = {t1[k]:.3f}\,p_0$ at $z = {z[k[0]]:.3f}\,s_\mathrm{{min}}$ "
               rf"on the grid, step $\Delta z = {dz:.2g}\,s_\mathrm{{min}}$")
    _note(fig, "; ".join(t for t in (maximum, note) if t))
    return fig


def ehl_centreline(x_over_s_x, p_over_p_h, h_m, *, h_c_formula_m: float | None = None,
                   note: str | None = None) -> Figure:
    """Pressure and film thickness along the centre line of an EHL contact (two stacked panels).

    Shows the pressure spike at the outlet and the film constriction under it on one abscissa,
    without a twin axis (Hamrock & Dowson 1977 Fig. 9, Venner & Bos 1994 Figs 3-4 manner).

    Conventions (pitfall): ``M``, ``L``, ``H`` are in Moes form, with the speed parameter ``U``
    built from the mean speed ``u_e = (u1 + u2)/2`` (SlipPY's ``rolling_speed`` is the mean); some
    Moes/Venner papers use the sum ``u1 + u2``, which shifts ``H``. A Hamrock-Dowson ``H`` must be
    converted to the Moes ``H`` before it shares the axes. ``s_x`` is SlipPY's
    ``hertzian_half_width``, ``R_x`` its ``radius_in_rolling_direction``.

    # OPEN R-074: no EHL solution is available yet (``slippy_bridge.lubrication.ehl.run`` raises);
    # this function is exercised on synthetic data only.

    Parameters
    ----------
    x_over_s_x : array_like
        Position along rolling over the Hertz semi-axis ``s_x`` [-] (inlet negative).
    p_over_p_h : array_like
        Pressure over the Hertz maximum pressure ``p_h`` (= ``p0`` of the dry Hertz contact under the
        same load, the EHL name) [-] (numerical: filled markers).
    h_m : array_like
        Film thickness [m]; drawn in um.
    h_c_formula_m : float, optional
        Central film thickness of a published formula [m], drawn as a dotted horizontal line.
    note : str, optional
        Line under the figure (formula, operating point).

    Returns
    -------
    matplotlib.figure.Figure
        Panels (a) ``p/p_h``, (b) ``h`` in um. The caller closes it.
    """
    # OPEN R-074: scaffold, synthetic data only until an EHL solution exists
    x = np.asarray(x_over_s_x, dtype=float)
    p = np.asarray(p_over_p_h, dtype=float)
    h_um = np.asarray(h_m, dtype=float) * 1e6              # [m] -> [um]
    every = max(1, len(x) // 40)                           # a dense solution shows about 40 markers
    fig, (ax_p, ax_h) = _stacked(2)
    ax_p.plot(x, p, linestyle="none", marker="o", markevery=every)
    ax_h.plot(x, h_um, linestyle="none", marker="o", markevery=every)
    if h_c_formula_m is not None:
        ax_h.axhline(h_c_formula_m * 1e6, linestyle=":", linewidth=plt.rcParams["axes.linewidth"])
        # labelled at the left (inlet) end, where the film is thick and clear of the markers
        ax_h.annotate(r"$h_c$ formula", (0.0, h_c_formula_m * 1e6), xycoords=("axes fraction", "data"),
                      xytext=(3, 2), textcoords="offset points", ha="left", va="bottom")
    ax_p.set_ylabel(r"$p/p_h$")
    ax_h.set_ylabel(r"$h$ ($\mu$m)")
    ax_h.set_xlabel(r"$x/s_x$")
    _note(fig, "; ".join(t for t in (r"$p_h$: Hertz maximum pressure $p_0$ of the same contact", note) if t))
    return fig


def film_survey(M, H_numerical, *, formulas: dict, film: str = "central", L: float | None = None,
                note: str | None = None) -> Figure:
    """Dimensionless film thickness against the load parameter, numerical against formulas (log-log).

    Nijenbanning et al. 1994 Fig. 12 manner: numerical points as filled markers, the first formula
    as a solid line, the others dashed and dotted, each labelled directly at its right end; decade
    tick labels on ``M``.

    Conventions (pitfall): ``M``, ``L``, ``H`` are in Moes form, with the speed parameter ``U``
    built from the mean speed ``u_e = (u1 + u2)/2`` (SlipPY's ``rolling_speed`` is the mean); some
    Moes/Venner papers use the sum ``u1 + u2``, which shifts ``H``. A Hamrock-Dowson ``H`` must be
    converted to the Moes ``H`` before it shares the axes. ``s_x`` is SlipPY's
    ``hertzian_half_width``, ``R_x`` its ``radius_in_rolling_direction``.

    # OPEN R-074, GL-13: no numerical EHL survey yet; ``references.contact.ehl`` formulas other
    # than ``e_prime`` raise. Exercised on synthetic data only.

    Parameters
    ----------
    M : array_like
        Moes load parameter [-] of the numerical points.
    H_numerical : array_like
        Dimensionless film thickness of the numerical solutions [-], Moes form.
    formulas : dict
        ``{label: (M, H)}`` of published formulas [-] for the same film, the first drawn solid; the
        label is the formula's name, e.g. ``"Moes 1992"``.
    film : {"central", "minimum"}
        Which film thickness ``H`` is: written in the ordinate label (``H_c`` or ``H_min``).
    L : float, optional
        Moes lubricant parameter [-] of the survey, written in the note line.
    note : str, optional
        Further text for the note line.

    Returns
    -------
    matplotlib.figure.Figure
        The caller closes it.
    """
    # OPEN R-074, GL-13: scaffold, synthetic data only
    symbols = {"central": r"$H_\mathrm{c}$", "minimum": r"$H_\mathrm{min}$"}
    if film not in symbols:
        raise ValueError(f"film must be 'central' or 'minimum', got {film!r}")
    fig, ax = plt.subplots()
    for (label, (m, h)), linestyle in zip(formulas.items(), ("-", "--", ":", "-.")):
        m, h = np.asarray(m, dtype=float), np.asarray(h, dtype=float)
        ax.loglog(m, h, linestyle=linestyle)
        ax.annotate(label, (m[-1], h[-1]), xytext=(4, 0), textcoords="offset points", ha="left", va="center")
    M, H = np.asarray(M, dtype=float), np.asarray(H_numerical, dtype=float)
    ax.loglog(M, H, linestyle="none", marker="o")
    # below the last point: the formula labels sit at the right ends of their lines
    ax.annotate("numerical", (M[-1], H[-1]), xytext=(0, -6), textcoords="offset points", ha="center", va="top")
    # H spans less than a decade: plain numbers on its ticks (not "4x10^0"); M keeps decade labels
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(mticker.FormatStrFormatter("%g"))
    ax.set_xlabel(r"$M$")
    ax.set_ylabel(symbols[film])
    _note(fig, "; ".join(t for t in ((f"$L$ = {L:g}" if L is not None else None), note) if t))
    return fig
