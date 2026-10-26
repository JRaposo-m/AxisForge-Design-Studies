"""
Figure style and figures of the load distribution studies (one place).

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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# --- style ---------------------------------------------------------------------------------
SURFACE, INK, INK_2, INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, RING = "#e1e0d9", "#c3c2b7", "#d9d7cf"
RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
        "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
LOAD_CMAP = LinearSegmentedColormap.from_list("element_load", RAMP)
# categorical, fixed order (validated: lightness, chroma, CVD and contrast on SURFACE); marker
# and line style are the secondary encoding
KIND_STYLE = {
    "ball": dict(color="#2a78d6", marker="o", linestyle="-"),
    "roller": dict(color="#eb6834", marker="s", linestyle="--"),
    "angular": dict(color="#1f9d74", marker="^", linestyle="-."),
}
KIND_NAME = {"ball": "deep groove ball", "angular": "angular contact ball",
             "roller": "cylindrical roller"}
FIGSIZE = (9.0, 5.6)
DPI = 160

# (column, axis label, logarithmic y, file name) of the summary figures
SUMMARY_METRICS = [
    ("n_loaded", "loaded elements [-]", False, "n_loaded"),
    ("zone_half_angle_deg", "loaded zone half-angle [deg]", False, "zone_half_angle"),
    ("Q_max_N", r"largest element load $Q_{max}$ [N]", False, "Q_max"),
    ("Q_max_over_Fr_per_Z", r"$Q_{max} / (F_r / Z)$ [-]", False, "Q_max_over_mean"),
    ("delta_r_mm", r"radial approach $\delta_r$ [mm]", False, "delta_r"),
    ("Kr_N_per_mm", r"secant radial stiffness $K_r$ [N/mm]", False, "Kr"),
    ("L10r_Mrev", r"reference life $L_{10r}$ [$10^6$ rev]", True, "L10r"),
]
AXIS_LABEL = {
    "s_mm": "radial internal clearance s [mm]",
    "psi_mrad": r"inner-ring tilt $\psi$ [mrad]",
    "alpha0_deg": r"free contact angle $\alpha_0$ [deg]",
    "power_W": "transmitted power P [W]",
}

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                     "legend.fontsize": 10})


def ramp(n: int) -> list[str]:
    """``n`` colours of the blue ramp, light to dark.

    Parameters
    ----------
    n: int
        Number of colours.

    Returns
    -------
    colours: list of str
        Hexadecimal colours.
    """
    if n <= 1:
        return [RAMP[len(RAMP) // 2]]
    return [RAMP[i] for i in np.linspace(0, len(RAMP) - 1, n).round().astype(int)]


def style_axes(ax) -> None:
    """Apply the report style to one Cartesian axes."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.title.set_color(INK)


def style_legend(legend) -> None:
    """Apply the report style to one legend."""
    if legend is None:
        return
    for text in legend.get_texts():
        text.set_color(INK_2)
    if legend.get_title() is not None:
        legend.get_title().set_color(INK_2)


def axis_label(name: str) -> str:
    """Axis label of a swept axis or result column (its name if unknown)."""
    return AXIS_LABEL.get(name, name)


def short_axis_name(name: str) -> str:
    """Axis label without its unit, for legends."""
    return axis_label(name).split(" [")[0]


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
        unit = label.split("[")[-1].rstrip("]") if "[" in label else ""
        parts.append(f"{short_axis_name(a)} = {point[a]:g} {unit}".strip())
    return ", ".join(parts)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


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
                 title: str = "") -> None:
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
    title: str
        Title prefix.
    """
    shaft = reference_shaft(rows)
    rows = [r for r in rows if r["shaft"] == shaft]
    groups = _groups(rows, group_axis)
    colours = ramp(len(groups)) if group_axis else [KIND_STYLE.get(rows[0]["kind"], {}).get(
        "color", RAMP[4])] if rows else [RAMP[4]]
    for key, label, log, file_name in SUMMARY_METRICS:
        fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=SURFACE)
        drawn = False
        for colour, (value, group) in zip(colours, groups):
            x, y = _series(group, x_axis, key)
            if np.all(np.isnan(y)):
                continue
            drawn = True
            name = None if value is None else f"{short_axis_name(group_axis)} = {value:g}"
            ax.plot(x, y, color=colour, marker="o", markersize=6, linewidth=2.0, label=name)
        if not drawn:
            plt.close(fig)
            continue
        if log:
            ax.set_yscale("log")
        ax.set_xlabel(axis_label(x_axis))
        ax.set_ylabel(label)
        ax.set_title(f"{title} ({shaft})" if title else shaft)
        style_axes(ax)
        if group_axis is not None:
            style_legend(ax.legend(frameon=False))
        _save(fig, out_dir / f"{file_name}.png")


# =============================================================================
# Load distribution in polar coordinates
# =============================================================================

R_INNER, R_PITCH, R_OUTER = 0.50, 0.72, 0.94      # inner ring, element centres, outer ring
R_BAR = 1.00                                       # zero-load circle of the load line
BAR_SPAN = 0.85                                    # radial length of the reference load


def _draw_bearing(ax, phi_deg, Q, q_ref, kind, Z, compact=False):
    """Draw one bearing seen along its axis, with the element loads as a load line around it.

    phi_j = 0 (the load line) points down: the radial load Fr on the inner ring acts downwards.
    """
    theta = np.radians(np.asarray(phi_deg, dtype=float))
    Q = np.asarray(Q, dtype=float)
    ax.set_theta_zero_location("S")
    ax.set_theta_direction(1)
    ax.set_facecolor(SURFACE)
    circle = np.linspace(0.0, 2.0 * np.pi, 361)

    # rings
    ax.fill_between(circle, R_INNER - 0.06, R_INNER, color=RING, linewidth=0)
    ax.fill_between(circle, R_OUTER, R_OUTER + 0.04, color=RING, linewidth=0)

    # rolling elements: filled with the load colour when loaded, hollow when not
    loaded = Q > 0.0
    colours = [LOAD_CMAP(q / q_ref) if q > 0.0 else SURFACE for q in Q]
    marker = "s" if kind == "roller" else "o"
    size = (60 if compact else 140) * (9.0 / max(Z, 9)) ** 2 * 2.2
    ax.scatter(theta, np.full_like(theta, R_PITCH), s=size, marker=marker, c=colours,
               edgecolors=INK_2, linewidths=0.8, zorder=3)

    # element loads as a closed line around the bearing (same scale for every point of the
    # sweep): the radius above the dashed zero circle is Q_j / q_ref
    r_load = R_BAR + BAR_SPAN * Q / q_ref
    ax.plot(circle, np.full_like(circle, R_BAR), color=AXIS, linewidth=0.8, linestyle="--",
            zorder=1)
    order = np.argsort(theta)                                     # closed line, straight segments
    ax.plot(np.append(theta[order], theta[order][0] + 2.0 * np.pi),
            np.append(r_load[order], r_load[order][0]), color=RAMP[6], linewidth=2.0, zorder=3)
    for th, r in zip(theta[loaded], r_load[loaded]):              # thin stem to its element
        ax.plot([th, th], [R_OUTER + 0.04, r], color=RAMP[3], linewidth=0.8, zorder=2)
    ax.scatter(theta, r_load, s=22 if compact else 36, color=RAMP[6], edgecolors=SURFACE,
               linewidths=0.8, zorder=4)
    for th, r, q in zip(theta[loaded], r_load[loaded], Q[loaded]):  # value of every loaded element
        ax.text(th, r + (0.13 if compact else 0.11), f"{q:.0f}", ha="center", va="center",
                fontsize=7.5 if compact else 9.5, color=INK, zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none",
                          alpha=0.85))

    # radial load on the inner ring
    ax.annotate("", xy=(0.0, R_INNER - 0.12), xytext=(0.0, 0.0),
                arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=1.6))
    ax.text(np.radians(25.0), 0.22, r"$F_r$", color=INK, fontsize=9 if compact else 11)

    ax.set_ylim(0.0, R_BAR + BAR_SPAN + 0.15)
    ax.set_yticks([])
    angles = np.array([0, 90, 270]) if compact else np.arange(0, 360, 30)
    ax.set_thetagrids(angles, [f"{(a + 180) % 360 - 180:d}°" for a in angles],
                      color=INK_MUTED, fontsize=8 if compact else 9)
    ax.grid(color=GRID, linewidth=0.6)
    ax.spines["polar"].set_visible(False)


def _info(row) -> str:
    def f(key, spec):
        value = row.get(key, math.nan)
        return "-" if isinstance(value, float) and math.isnan(value) else format(value, spec)
    return (f"loaded {f('n_loaded', '.0f')} of {f('Z', '.0f')} | zone ±{f('zone_half_angle_deg', '.0f')}° | "
            f"Fr = {f('Fr_N', '.0f')} N\nQ_max = {f('Q_max_N', '.0f')} N | "
            f"Q_max/(Fr/Z) = {f('Q_max_over_Fr_per_Z', '.2f')} | "
            f"delta_r = {1e3 * row.get('delta_r_mm', math.nan):.1f} um")


def plot_polar_distributions(rows: list[dict], q_rows: list[dict], plots_dir: Path, *,
                             axis_names, title: str = "") -> None:
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
    title: str
        Title prefix.
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
    scale_note = (f"load line: Q_j [N] above the dashed zero circle, scale {q_ref:.0f} N = "
                  "largest element load of the sweep")

    drawable = []
    for row in points:
        elements = by_point.get(tuple(row[a] for a in axis_names), [])
        if not elements:
            continue
        drawable.append((row, elements))
        fig = plt.figure(figsize=(7.6, 8.4), facecolor=SURFACE)
        ax = fig.add_subplot(projection="polar")
        _draw_bearing(ax, [e["phi_deg"] for e in elements], [e["Q_N"] for e in elements],
                      q_ref, kind, Z)
        ax.set_title(f"{title}\n{point_title(row, axis_names)}  ({shaft})", color=INK, pad=18)
        fig.text(0.5, 0.05, _info(row), ha="center", color=INK_2, fontsize=10)
        fig.text(0.5, 0.015, scale_note + "; phi_j = 0 on the load line", ha="center",
                 color=INK_MUTED, fontsize=8)
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
        heading = title if not others else f"{title}, {point_title(group, others)}"
        _overview(members, plots_dir / name, [first], q_ref, kind, Z,
                  f"{heading} ({shaft})\n{scale_note}")


def _overview(members, path, label_axes, q_ref, kind, Z, suptitle):
    """Polar drawings side by side (at most 3 per row), same radial scale."""
    n = len(members)
    ncol = min(3, n)
    nrow = math.ceil(n / ncol)
    height = 5.9 * nrow + 1.4
    fig = plt.figure(figsize=(5.6 * ncol, height), facecolor=SURFACE)
    for i, (row, elements) in enumerate(members):
        ax = fig.add_subplot(nrow, ncol, i + 1, projection="polar")
        _draw_bearing(ax, [e["phi_deg"] for e in elements], [e["Q_N"] for e in elements],
                      q_ref, kind, Z, compact=True)
        ax.text(0.5, 1.13, point_title(row, label_axes), transform=ax.transAxes, ha="center",
                va="bottom", color=INK, fontsize=12)
        ax.text(0.5, 1.01, _info(row), transform=ax.transAxes, ha="center", va="bottom",
                color=INK_2, fontsize=8.5, linespacing=1.4)
    fig.suptitle(suptitle, color=INK, fontsize=13)
    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.02, top=1.0 - 2.0 / height,
                        hspace=0.38, wspace=0.18)
    _save(fig, path)


# =============================================================================
# Roller laminae
# =============================================================================


def plot_lamina_profile(lam_rows: list[dict], plots_dir: Path, *, x_axis: str,
                        group_axis: str | None = None, title: str = "") -> None:
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
    title: str
        Title prefix.
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
        fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=SURFACE)
        for colour, v in zip(ramp(len(values)), values):
            pts = sorted((r for r in rows if r[x_axis] == v), key=lambda r: r["x_mm"])
            ax.plot([r["x_mm"] for r in pts], [r["q_N"] for r in pts], color=colour,
                    linewidth=2.0, label=f"{v:g}")
        ax.set_xlabel(r"position along the roller $x_k$ [mm]")
        ax.set_ylabel(r"lamina load $q_k$ [N]")
        ax.set_title(f"{title}: most loaded roller ({shaft}{suffix})")
        style_axes(ax)
        style_legend(ax.legend(title=axis_label(x_axis), frameon=False))
        _save(fig, path)


# =============================================================================
# Comparison between types
# =============================================================================


def plot_compare(datasets: dict[str, list[dict]], out_dir: Path, *, x_axis: str,
                 group_axis: str | None = None, group_value: float | None = None,
                 title: str = "") -> None:
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
    title: str
        Figure title.
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
        fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=SURFACE)
        drawn = False
        for name, rows in selected.items():
            x, y = _series(rows, x_axis, key)
            if np.all(np.isnan(y)):
                continue
            drawn = True
            kind = rows[0]["kind"] if rows else ""
            style = KIND_STYLE.get(kind, dict(color=INK_2, marker="o", linestyle="-"))
            ax.plot(x, y, markersize=6, linewidth=2.0, label=KIND_NAME.get(kind, name), **style)
        if not drawn:
            plt.close(fig)
            continue
        if log:
            ax.set_yscale("log")
        ax.set_xlabel(axis_label(x_axis))
        ax.set_ylabel(label)
        ax.set_title(title)
        style_axes(ax)
        style_legend(ax.legend(frameon=False))
        _save(fig, out_dir / f"{file_name}.png")
