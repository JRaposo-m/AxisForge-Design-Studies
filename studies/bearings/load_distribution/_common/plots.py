"""
Figures of the load distribution studies (one place). Their look comes from the house sheet
``style/axisforge.mplstyle``; no colours, fonts or sizes are set here (``style/README.md``).

Figures of one bearing type (``<question>/<type>/plots``)
--------------------------------------------------------
summary/<metric>.png             one large figure per quantity against the first axis
                                 (one line per value of the second axis, if any)
load_distribution/<point>.png    the bearing seen along its axis, one figure per point of the
                                 sweep: rings, rolling elements and the load line Q(phi) around
                                 them; the SAME radial scale in every figure of the sweep
load_distribution_overview[__<value>].png
                                 the same drawings side by side (same scale), at most 3 per row;
                                 one overview per value of the second axis, if any
roller_lamina_profile[__<value>].png
                                 line contact only: lamina loads along the most loaded roller,
                                 one figure per value of the second axis, if any

Comparison figures (``<question>/compare/plots``)
-------------------------------------------------
<x>[/<value>]/<metric>.png       one figure per quantity, one line per bearing type; one folder
                                 per value of the second axis, if any

Every figure is drawn for the bearing of the first shaft: with spur gears the two shafts give the
same results (see the checks in report.txt); with helical gears the second shaft has Fa of
opposite sign and is kept in the CSV, not in the figures.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# The look comes only from the house sheet: the study entry points (``study.figures``,
# ``compare.compare_types``) call ``plt.style.use(STYLE_PATH)`` once before drawing. A study cannot
# import ``validation._support.style``, so the path is resolved here from the repository layout.
STYLE_PATH = Path(__file__).resolve().parents[4] / "style" / "axisforge.mplstyle"

# Same text as ``validation._support.figures.bearing_plots.A2_NOTE`` (a study cannot import
# validation; one string twice until the A2 ruling, FG-34).
A2_NOTE = r"A2: global direction of $\varphi_j = 0$ under review (R-003); $\varphi_j$ local, 0 on the load line"

# series identity by marker and line style (the sheet's cycle adds the colour)
KIND_STYLE = {
    "ball": dict(marker="o", linestyle="-"),
    "roller": dict(marker="s", linestyle="--"),
    "angular": dict(marker="^", linestyle="-."),
}
KIND_NAME = {"ball": "deep groove ball", "angular": "angular contact ball",
             "roller": "cylindrical roller"}
MARKERS = ("o", "s", "^", "D", "v", "<", ">")     # one per line of a family (with the cycle's line style)

# (column, axis label, logarithmic y, file name) of the summary figures
SUMMARY_METRICS = [
    ("n_loaded", r"loaded elements $n$", False, "n_loaded"),
    ("zone_half_angle_deg", "loaded zone half-angle (deg)", False, "zone_half_angle"),
    ("Q_max_N", r"largest element load $Q_\mathrm{max}$ (N)", False, "Q_max"),
    ("Q_max_over_Fr_per_Z", r"$Q_\mathrm{max}/(F_r/Z)$", False, "Q_max_over_mean"),
    ("delta_r_mm", r"radial approach $\delta_r$ (mm)", False, "delta_r"),
    ("Kr_N_per_mm", r"secant radial stiffness $K_r$ (N/mm)", False, "Kr"),
    ("L10r_Mrev", r"reference life $L_{10r}$ ($10^6$ rev)", True, "L10r"),
]
AXIS_LABEL = {
    "s_mm": "radial internal clearance $s$ (mm)",
    "psi_mrad": r"inner-ring tilt $\psi$ (mrad)",
    "alpha0_deg": r"free contact angle $\alpha_0$ (deg)",
    "power_W": "transmitted power $P$ (W)",
}


def axis_label(name: str) -> str:
    """Axis label of a swept axis or result column (its name if unknown)."""
    return AXIS_LABEL.get(name, name)


def short_axis_name(name: str) -> str:
    """Axis label without its unit "(unit)", for legends."""
    label = axis_label(name)
    return label.rsplit(" (", 1)[0] if label.endswith(")") else label


def reference_shaft(rows: list[dict]) -> str:
    """Shaft drawn in the figures: the first one that appears in the rows."""
    return rows[0]["shaft"] if rows else ""


def point_name(point: dict, axis_names) -> str:
    """File-name friendly label of a point of the sweep, e.g. ``s_mm_0.01``."""
    return "__".join(f"{a}_{point[a]:g}" for a in axis_names)


def point_title(point: dict, axis_names) -> str:
    """Readable label of a point of the sweep, e.g. ``s = 0.01 mm``."""
    parts = []
    for a in axis_names:
        label = axis_label(a)
        unit = label.rsplit(" (", 1)[-1].rstrip(")") if label.endswith(")") else ""
        parts.append(f"{short_axis_name(a)} = {point[a]:g} {unit}".strip())
    return ", ".join(parts)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)                                   # dpi, bbox from the sheet
    plt.close(fig)


def _note(fig, text: str) -> None:
    """The figure's information line (what a title used to carry), under the axes."""
    fig.text(0.5, 0.0, text, ha="center", va="top", wrap=True)


def _series(rows, x_key, key):
    rows = sorted(rows, key=lambda r: r[x_key])
    x = np.array([r[x_key] for r in rows], dtype=float)
    y = np.array([r[key] if r["ok"] else math.nan for r in rows], dtype=float)
    return x, y


def _groups(rows, group_axis):
    if group_axis is None:
        return [(None, rows)]
    values = sorted({r[group_axis] for r in rows})
    return [(v, [r for r in rows if r[group_axis] == v]) for v in values]


# =============================================================================
# Summary figures
# =============================================================================


def plot_summary(rows: list[dict], out_dir: Path, *, x_axis: str, group_axis: str | None = None,
                 note: str = "") -> None:
    """One figure per quantity against the first axis.

    Parameters
    ----------
    rows: list of dict
        Rows of one bearing type.
    out_dir: Path
        Folder of the figures (``plots/summary``).
    x_axis: str
        Axis on the x axis.
    group_axis: str, optional (None)
        Axis whose values are drawn as separate lines (with a legend).
    note: str
        Prefix of the information line under the figure.
    """
    shaft = reference_shaft(rows)
    rows = [r for r in rows if r["shaft"] == shaft]
    groups = _groups(rows, group_axis)
    for key, label, log, file_name in SUMMARY_METRICS:
        fig, ax = plt.subplots()
        drawn = False
        for marker, (value, group) in zip(MARKERS, groups):
            x, y = _series(group, x_axis, key)
            if np.all(np.isnan(y)):
                continue
            drawn = True
            name = None if value is None else f"{short_axis_name(group_axis)} = {value:g}"
            ax.plot(x, y, marker=marker, label=name)       # line style and colour from the cycle
        if not drawn:
            plt.close(fig)
            continue
        if log:
            ax.set_yscale("log")
        ax.set_xlabel(axis_label(x_axis))
        ax.set_ylabel(label)
        if group_axis is not None:
            fig.legend(*ax.get_legend_handles_labels(), loc="outside right upper")
        _note(fig, f"{note} ({shaft})" if note else shaft)
        _save(fig, out_dir / f"{file_name}.png")


# =============================================================================
# Load distribution in polar coordinates
# =============================================================================

R_INNER, R_PITCH, R_OUTER = 0.50, 0.72, 0.94      # inner ring, element centres, outer ring
R_BAR = 1.00                                       # zero-load circle of the load line
BAR_SPAN = 0.85                                    # radial length of the reference load


def _draw_bearing(ax, phi_deg, Q, q_ref, kind, Z, compact=False):
    """Draw one bearing seen along its axis, with the element loads as a load line around it.

    phi_j = 0 (the load line) points down: the radial load Fr on the inner ring acts downwards;
    phi_j grows anticlockwise. Loaded elements are filled, unloaded ones open. A schematic: the
    radii are fixed fractions of the drawing, not the bearing geometry.
    """
    theta = np.radians(np.asarray(phi_deg, dtype=float))
    Q = np.asarray(Q, dtype=float)
    ink = plt.rcParams["lines.color"]                   # the sheet's line colour, for every mark
    thin = plt.rcParams["axes.linewidth"]
    ax.set_theta_zero_location("S")
    ax.set_theta_direction(1)
    circle = np.linspace(0.0, 2.0 * np.pi, 361)

    # rings: two thin circles each
    for r in (R_INNER - 0.06, R_INNER, R_OUTER, R_OUTER + 0.04):
        ax.plot(circle, np.full_like(circle, r), color=ink, linestyle="-", linewidth=thin)

    # rolling elements: filled when loaded, open when not; the marker shrinks with Z so that
    # neighbours do not overlap (9 elements = the reference size)
    loaded = Q > 0.0
    marker = "s" if kind == "roller" else "o"
    size = ((1.6 if compact else 2.5) * plt.rcParams["lines.markersize"] * 9.0 / max(Z, 9)) ** 2
    ax.scatter(theta[loaded], np.full(loaded.sum(), R_PITCH), s=size, marker=marker, c=ink, zorder=3)
    ax.scatter(theta[~loaded], np.full((~loaded).sum(), R_PITCH), s=size, marker=marker,
               facecolors="none", edgecolors=ink, zorder=3)

    # element loads as a closed line around the bearing (same scale for every point of the
    # sweep): the radius above the dashed zero circle is Q_j / q_ref
    r_load = R_BAR + BAR_SPAN * Q / q_ref
    ax.plot(circle, np.full_like(circle, R_BAR), color=ink, linestyle="--", linewidth=thin, zorder=1)
    order = np.argsort(theta)                                     # closed line, straight segments
    ax.plot(np.append(theta[order], theta[order][0] + 2.0 * np.pi),
            np.append(r_load[order], r_load[order][0]), color=ink, linestyle="-", marker=".", zorder=3)
    for th, r in zip(theta[loaded], r_load[loaded]):              # thin stem to its element
        ax.plot([th, th], [R_OUTER + 0.04, r], color=ink, linestyle=":", linewidth=thin, zorder=2)
    if not compact:                                               # value of every loaded element
        for th, r, q in zip(theta[loaded], r_load[loaded], Q[loaded]):
            ax.text(th, r + 0.11, f"{q:.0f}", ha="center", va="center", zorder=5)

    # radial load on the inner ring
    ax.annotate("", xy=(0.0, R_INNER - 0.12), xytext=(0.0, 0.0), arrowprops=dict(arrowstyle="-|>", color=ink))
    ax.text(np.radians(25.0), 0.22, r"$F_r$")

    ax.set_ylim(0.0, R_BAR + BAR_SPAN + 0.15)
    ax.set_yticks([])
    angles = np.array([0, 90, 270]) if compact else np.arange(0, 360, 30)
    ax.set_thetagrids(angles, [f"{(a + 180) % 360 - 180:d}°" for a in angles])
    ax.grid(False)                                      # set_thetagrids switches the polar grid back on
    ax.spines["polar"].set_visible(False)


def _info(row) -> str:
    def f(key, spec):
        value = row.get(key, math.nan)
        return "-" if isinstance(value, float) and math.isnan(value) else format(value, spec)
    return (f"loaded {f('n_loaded', '.0f')} of {f('Z', '.0f')}, zone ±{f('zone_half_angle_deg', '.0f')}°, "
            f"$F_r$ = {f('Fr_N', '.0f')} N, $Q_\\mathrm{{max}}$ = {f('Q_max_N', '.0f')} N, "
            f"$Q_\\mathrm{{max}}/(F_r/Z)$ = {f('Q_max_over_Fr_per_Z', '.2f')}, "
            f"$\\delta_r$ = {1e3 * row.get('delta_r_mm', math.nan):.1f} $\\mu$m")


def _azimuth_note(q_ref: float) -> str:
    """Scale of the load line and the azimuth convention of the polar drawings (A2 stamp)."""
    return (f"load line: $Q_j$ above the dashed zero circle, scale {q_ref:.0f} N = largest element load of "
            "the sweep; $\\varphi_j$ = 0 at the bottom (load line, direction of $F_r$), positive anticlockwise. "
            + A2_NOTE)


def plot_polar_distributions(rows: list[dict], q_rows: list[dict], plots_dir: Path, *,
                             axis_names, note: str = "") -> None:
    """The load distribution of every point of the sweep, drawn as the bearing itself.

    Writes one figure per point (``load_distribution/<point>.png``) and an overview with all
    of them (``load_distribution_overview.png``). Every drawing uses the same radial scale
    (the largest element load of the sweep), so the load lines can be compared between figures.

    Parameters
    ----------
    rows: list of dict
        Rows of one bearing type.
    q_rows: list of dict
        Element loads of one bearing type.
    plots_dir: Path
        ``<question>/<type>/plots``.
    axis_names: sequence of str
        Names of the swept axes.
    note: str
        Prefix of the information line under the figure.
    """
    shaft = reference_shaft(rows)
    points = [r for r in rows if r["shaft"] == shaft]
    by_point: dict[tuple, list[dict]] = {}
    for q in q_rows:
        if q["shaft"] == shaft:
            by_point.setdefault(tuple(q[a] for a in axis_names), []).append(q)
    q_ref = max((q["Q_N"] for qs in by_point.values() for q in qs), default=0.0)
    if q_ref <= 0.0 or not points:
        return
    kind = points[0]["kind"]
    Z = int(points[0]["Z"])

    drawable = []
    for row in points:
        elements = by_point.get(tuple(row[a] for a in axis_names), [])
        if not elements:
            continue
        drawable.append((row, elements))
        fig = plt.figure()
        ax = fig.add_subplot(projection="polar")
        _draw_bearing(ax, [e["phi_deg"] for e in elements], [e["Q_N"] for e in elements],
                      q_ref, kind, Z)
        _note(fig, f"{note}, {point_title(row, axis_names)} ({shaft}): {_info(row)}\n{_azimuth_note(q_ref)}")
        _save(fig, plots_dir / "load_distribution" / f"{point_name(row, axis_names)}.png")

    # one overview per value of the axes after the first (one overview if there is one axis)
    first, others = axis_names[0], list(axis_names[1:])
    groups: dict[tuple, list] = {}
    for row, elements in drawable:
        groups.setdefault(tuple(row[a] for a in others), []).append((row, elements))
    for key, members in groups.items():
        group = dict(zip(others, key))
        name = ("load_distribution_overview.png" if not others else
                f"load_distribution_overview__{point_name(group, others)}.png")
        heading = note if not others else f"{note}, {point_title(group, others)}"
        _overview(members, plots_dir / name, [first], q_ref, kind, Z,
                  f"{heading} ({shaft})\n{_azimuth_note(q_ref)}")


def _overview(members, path, label_axes, q_ref, kind, Z, note):
    """Polar drawings side by side (at most 3 per row), same radial scale, panels (a), (b), ..."""
    n = len(members)
    ncol = min(3, n)
    nrow = math.ceil(n / ncol)
    width = plt.rcParams["figure.figsize"][0]
    # the sheet width; each row as high as one panel is wide (a polar drawing is square)
    fig = plt.figure(figsize=(width, nrow * width / ncol))
    for i, (row, elements) in enumerate(members):
        ax = fig.add_subplot(nrow, ncol, i + 1, projection="polar")
        _draw_bearing(ax, [e["phi_deg"] for e in elements], [e["Q_N"] for e in elements],
                      q_ref, kind, Z, compact=True)
        ax.text(0.0, 1.0, f"({chr(ord('a') + i)})", transform=ax.transAxes, ha="left", va="top")
    infos = "; ".join(f"({chr(ord('a') + i)}) {point_title(row, label_axes)}: {_info(row)}"
                      for i, (row, _) in enumerate(members))
    _note(fig, f"{note}\n{infos}")
    _save(fig, path)


# =============================================================================
# Roller laminae
# =============================================================================


def plot_lamina_profile(lam_rows: list[dict], plots_dir: Path, *, x_axis: str,
                        group_axis: str | None = None, note: str = "") -> None:
    """Lamina loads q_k along the most loaded roller, one line per value of the first axis.

    One figure per value of the second axis (``roller_lamina_profile__<value>.png``), or a single
    ``roller_lamina_profile.png`` when there is one axis.

    Parameters
    ----------
    lam_rows: list of dict
        Lamina loads of a line-contact bearing.
    plots_dir: Path
        ``<question>/<type>/plots``.
    x_axis: str
        First swept axis (one line per value).
    group_axis: str, optional (None)
        Second swept axis (one figure per value).
    note: str
        Prefix of the information line under the figure.
    """
    if not lam_rows:
        return
    shaft = reference_shaft(lam_rows)
    lam_rows = [r for r in lam_rows if r["shaft"] == shaft]
    for value, rows in _groups(lam_rows, group_axis):
        if value is None:
            path, suffix = plots_dir / "roller_lamina_profile.png", ""
        else:
            path = plots_dir / f"roller_lamina_profile__{group_axis}_{value:g}.png"
            suffix = f", {point_title({group_axis: value}, [group_axis])}"
        values = sorted({r[x_axis] for r in rows})
        fig, ax = plt.subplots()
        for i, v in enumerate(values):
            pts = sorted((r for r in rows if r[x_axis] == v), key=lambda r: r["x_mm"])
            # marker list (7) against the cycle's line styles (5): distinct pairs up to 35 lines
            ax.plot([r["x_mm"] for r in pts], [r["q_N"] for r in pts], marker=MARKERS[i % len(MARKERS)],
                    label=f"{v:g}")
        ax.set_xlabel(r"position along the roller $x_k$ (mm)")
        ax.set_ylabel(r"lamina load $q_k$ (N)")
        fig.legend(*ax.get_legend_handles_labels(), title=axis_label(x_axis), loc="outside right upper")
        _note(fig, f"{note}: most loaded roller ({shaft}{suffix})")
        _save(fig, path)


# =============================================================================
# Comparison between types
# =============================================================================


def plot_compare(datasets: dict[str, list[dict]], out_dir: Path, *, x_axis: str,
                 group_axis: str | None = None, group_value: float | None = None,
                 note: str = "") -> None:
    """One figure per quantity, one line per bearing type.

    Parameters
    ----------
    datasets: dict
        {type folder name: rows}.
    out_dir: Path
        Folder of the figures.
    x_axis: str
        Axis or result column on the x axis.
    group_axis: str, optional (None)
        Second swept axis; only ``group_value`` of it is drawn.
    group_value: float, optional (None)
        Value of ``group_axis`` to draw; None takes its last value.
    note: str
        Information line under the figures.
    """
    selected = {}
    for name, rows in datasets.items():
        shaft = reference_shaft(rows)
        rows = [r for r in rows if r["shaft"] == shaft]
        if group_axis is not None and rows:
            value = max(r[group_axis] for r in rows) if group_value is None else group_value
            rows = [r for r in rows if r[group_axis] == value]
        selected[name] = rows
    for key, label, log, file_name in SUMMARY_METRICS:
        fig, ax = plt.subplots()
        drawn = False
        for name, rows in selected.items():
            x, y = _series(rows, x_axis, key)
            if np.all(np.isnan(y)):
                continue
            drawn = True
            kind = rows[0]["kind"] if rows else ""
            style = KIND_STYLE.get(kind, dict(marker="D", linestyle=":"))
            ax.plot(x, y, label=KIND_NAME.get(kind, name), **style)
        if not drawn:
            plt.close(fig)
            continue
        if log:
            ax.set_yscale("log")
        ax.set_xlabel(axis_label(x_axis))
        ax.set_ylabel(label)
        fig.legend(*ax.get_legend_handles_labels(), loc="outside right upper")
        _note(fig, note)
        _save(fig, out_dir / f"{file_name}.png")
