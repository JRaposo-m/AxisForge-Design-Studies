# studies\bearings\clearance_load_distribution\outputs.py
"""
Everything the clearance study writes or draws: checks, console summary, CSV files and
figures. study.py only computes the rows and calls report().

To change a plot, a table or a CSV, edit this file; study.py does not need to change.
Inputs are the rows built by study._row (one per s and bearing), the long Q_j rows, and
the lamina rows of the most loaded roller (line-contact bearings only).

Bearings are told apart by `kind` ("ball" / "roller"); ref_labels = {kind: label} says
which bearing of each kind is drawn.
"""
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --- checks ------------------------------------------------------------------
EQUILIBRIUM_RTOL = 1e-6        # |Fr - sum(Fr_row)| / Fr
SYMMETRY_RTOL = 1e-6           # same Q_max expected in identical bearings of one kind

# --- file names --------------------------------------------------------------
CSV_ROWS = "clearance_sweep.csv"
CSV_QJ = "clearance_sweep_Qj.csv"
CSV_LAMINA = "clearance_sweep_lamina.csv"
FIG_LOAD = "load_distribution.png"
FIG_SUMMARY = "clearance_summary.png"
FIG_LAMINA = "roller_lamina_profile.png"

KIND_NAME = {"ball": "deep groove ball", "roller": "cylindrical roller"}

# =============================================================================
# Checks and console summary
# =============================================================================


def check_rows(rows):
    bad = [r for r in rows if (not r["ok"]) or (r["equil_err_rel"] > EQUILIBRIUM_RTOL)
           or (r["error"] and not r["postprocessed"])]
    print(f"\nconvergence / equilibrium / errors: {len(rows) - len(bad)}/{len(rows)} clean")
    for r in bad:
        print(f"  CHECK s={r['s_mm']:.3f} {r['label']}: ok={r['ok']} "
              f"equil_err={r['equil_err_rel']:.2e} {r['error']}")


def check_symmetry(rows, s_values):
    """The shafts and loads are identical, so the bearings of one kind (one per shaft)
    must give the same Q_max (sign of psi does not matter: mirror symmetry).
    Different kinds are not comparable and are checked separately."""
    for kind in sorted({r["kind"] for r in rows}):
        worst = 0.0
        for s in s_values:
            q = [r["Q_max_N"] for r in rows if r["s_mm"] == s and r["kind"] == kind and r["ok"]]
            if len(q) > 1 and max(q) > 0.0:
                worst = max(worst, (max(q) - min(q)) / max(q))
        flag = "OK" if worst <= SYMMETRY_RTOL else "CHECK"
        print(f"worst Q_max spread across the {kind} bearings: {worst:.2e}  [{flag}]")


def _fmt(x, spec):
    return "-".rjust(len(format(0.0, spec))) if (x is None or math.isnan(x)) else format(x, spec)


def print_summary(rows, ref_labels):
    for kind, label in ref_labels.items():
        print(f"\n{label}   ({KIND_NAME.get(kind, kind)}, psi = shaft slope from the FEM)")
        print(f"{'s [um]':>7} {'alpha0':>7} {'n_load':>6} {'zone':>6} {'Q_max':>8} "
              f"{'Qmax/(Fr/Z)':>12} {'d_r [mm]':>9} {'psi[mrad]':>9} {'L10h [h]':>10} "
              f"{'lam_frac':>8} {'pk/mean':>8} {'ok':>3}")
        for r in rows:
            if r["label"] == label:
                print(f"{1e3 * r['s_mm']:7.1f} {r['alpha0_deg']:7.2f} {r['n_loaded']:6.0f} "
                      f"{r['zone_half_angle_deg']:6.1f} {r['Q_max_N']:8.1f} "
                      f"{r['Q_max_over_Fr_per_Z']:12.3f} {r['delta_r_mm']:9.4f} "
                      f"{r['psi_mrad']:9.3f} {r['L10h_h']:10.0f} "
                      f"{_fmt(r['lamina_loaded_frac'], '8.2f')} "
                      f"{_fmt(r['lamina_peak_over_mean'], '8.2f')} {str(r['ok']):>3}")

# =============================================================================
# Files
# =============================================================================


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# =============================================================================
# Figures
# =============================================================================
# Colours: s is an ordered magnitude -> one-hue sequential ramp (blue, light -> dark).
# The two bearing kinds are categorical -> palette slots 1 (blue) and 2 (orange), also
# told apart by marker and line style so colour is never the only cue.

_SURFACE, _INK, _INK_2, _INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
_GRID, _AXIS = "#e1e0d9", "#c3c2b7"
_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
         "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
_KIND_STYLE = {"ball": dict(color="#2a78d6", marker="o", linestyle="-"),
               "roller": dict(color="#eb6834", marker="s", linestyle="--")}


def _ramp(n):
    idx = np.linspace(0, len(_RAMP) - 1, n).round().astype(int) if n > 1 else [len(_RAMP) // 2]
    return [_RAMP[i] for i in idx]


def _style_axes(ax):
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS)
    ax.grid(True, color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=_INK_MUTED, labelsize=9)
    ax.xaxis.label.set_color(_INK_2)
    ax.yaxis.label.set_color(_INK_2)
    ax.title.set_color(_INK)


def _style_legend(leg):
    for text in leg.get_texts():
        text.set_color(_INK_2)
    if leg.get_title() is not None:
        leg.get_title().set_color(_INK_2)


def plot_load_distribution(q_rows, rows, path, s_values, ref_labels):
    """Q_j / Fr (external radial load of the bearing) vs element angle, one panel per
    bearing kind, one line per s."""
    Fr = {(r["label"], r["s_mm"]): r["Fr_N"] for r in rows}
    n = len(ref_labels)
    fig, axes = plt.subplots(1, n, figsize=(5.8 * n, 4.6), facecolor=_SURFACE, squeeze=False)
    colours = _ramp(len(s_values))
    for ax, (kind, label) in zip(axes[0], ref_labels.items()):
        _style_axes(ax)
        for colour, s in zip(colours, s_values):
            f = Fr.get((label, s), math.nan)
            pts = sorted((q["phi_deg"], q["Q_N"] / f) for q in q_rows
                         if q["label"] == label and q["s_mm"] == s)
            if pts and f > 0.0:
                x, y = zip(*pts)
                ax.plot(x, y, color=colour, marker="o", markersize=5, linewidth=1.6,
                        label=f"{1e3 * s:g}")
        ax.set_title(f"{KIND_NAME.get(kind, kind)}  -  {label}", fontsize=10, loc="left")
        ax.set_xlabel("element angle from the load line  [deg]")
        ax.set_ylabel("element load  Q_j / Fr  [-]")
        ax.set_xticks(range(-180, 181, 60))
    _style_legend(axes[0][-1].legend(title="s  [µm]", fontsize=9, title_fontsize=9,
                                     frameon=False, loc="upper right"))
    fig.suptitle("Load distribution", x=0.01, ha="left", fontsize=12, color=_INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=_SURFACE)
    return fig


def plot_summary(rows, path, s_values, ref_labels):
    """Panels vs s: Q_max normalised by the mean load per element Fr/Z (same for every
    bearing, so ball and roller are directly comparable), loaded elements, delta_r, and
    L10r in one panel per kind (the lives differ by orders of magnitude, so a shared axis
    would flatten the smaller one). The sixth cell holds the legend."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.2), facecolor=_SURFACE)
    # (key, ylabel, kinds drawn): kinds = None -> all
    panels = [("Q_max_over_Fr_per_Z", "Q_max / (Fr / Z)  [-]", None),
              ("n_loaded", "loaded elements  [-]", None),
              ("delta_r_mm", "radial approach  delta_r  [mm]", None),
              ("L10r_Mrev", "L10r  [10^6 rev]", ["ball"]),
              ("L10r_Mrev", "L10r  [10^6 rev]", ["roller"])]
    for ax, (key, ylabel, kinds) in zip(axes.ravel(), panels):
        _style_axes(ax)
        for kind, label in ref_labels.items():
            if kinds is not None and kind not in kinds:
                continue
            pts = [(1e3 * r["s_mm"], r[key]) for r in rows
                   if r["label"] == label and not math.isnan(r[key])]
            if pts:
                x, y = zip(*pts)
                ax.plot(x, y, markersize=6, linewidth=1.6, label=KIND_NAME.get(kind, kind),
                        **_KIND_STYLE[kind])
        if kinds is not None:
            ax.set_title(KIND_NAME.get(kinds[0], kinds[0]), fontsize=10, loc="left")
        ax.set_xlabel("radial internal clearance  s  [µm]")
        ax.set_ylabel(ylabel)
        if key == "n_loaded":
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    # legend in the last cell (taken from the delta_r panel)
    legend_ax = axes[1][2]
    legend_ax.axis("off")
    handles, labels = axes[0][2].get_legend_handles_labels()
    _style_legend(legend_ax.legend(handles, labels, loc="center left", frameon=False, fontsize=10))
    fig.suptitle("Clearance sweep  (psi = shaft slope from the FEM)", x=0.01, y=0.985,
                 ha="left", fontsize=12, color=_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor=_SURFACE)
    return fig


def plot_lamina_profile(lam_rows, rows, path, s_values, ref_labels):
    """Lamina load q_k of the most loaded roller along its length, divided by the
    external radial load of the bearing Fr (dimensionless), one line per s.
    Line-contact reference bearings only."""
    line = {k: lab for k, lab in ref_labels.items()
            if any(r["label"] == lab for r in lam_rows)}
    if not line:
        return None
    Fr = {(r["label"], r["s_mm"]): r["Fr_N"] for r in rows}
    fig, axes = plt.subplots(1, len(line), figsize=(5.8 * len(line), 4.4),
                             facecolor=_SURFACE, squeeze=False)
    colours = _ramp(len(s_values))
    for ax, (kind, label) in zip(axes[0], line.items()):
        _style_axes(ax)
        for colour, s in zip(colours, s_values):
            pts = sorted((r["x_mm"], r["q_N"]) for r in lam_rows
                         if r["label"] == label and r["s_mm"] == s)
            f = Fr.get((label, s), math.nan)
            if pts and f > 0.0:
                x, q = zip(*pts)
                ax.plot(x, np.array(q) / f, color=colour, linewidth=1.6, label=f"{1e3 * s:g}")
        ax.set_title(f"{label}  -  most loaded roller", fontsize=10, loc="left")
        ax.set_xlabel("position along the roller  x_k  [mm]")
        ax.set_ylabel("q_k / Fr  [-]")
    _style_legend(axes[0][-1].legend(title="s  [µm]", fontsize=9, title_fontsize=9,
                                     frameon=False, loc="upper left",
                                     bbox_to_anchor=(1.02, 1.0)))       # outside: curves fill the axes
    fig.suptitle("Lamina load profile (ISO/TS 16281 sec. 5.2)", x=0.01, ha="left",
                 fontsize=12, color=_INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=_SURFACE)
    return fig

# =============================================================================
# Entry point (called by study.py)
# =============================================================================


def report(rows, q_rows, lam_rows, *, s_values, ref_labels, out_dir, row_fields, q_fields,
           lamina_fields, show=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(out_dir / CSV_ROWS, row_fields, rows)
    write_csv(out_dir / CSV_QJ, q_fields, q_rows)
    write_csv(out_dir / CSV_LAMINA, lamina_fields, lam_rows)

    print_summary(rows, ref_labels)
    check_rows(rows)
    check_symmetry(rows, s_values)

    plot_load_distribution(q_rows, rows, out_dir / FIG_LOAD, s_values, ref_labels)
    plot_summary(rows, out_dir / FIG_SUMMARY, s_values, ref_labels)
    plot_lamina_profile(lam_rows, rows, out_dir / FIG_LAMINA, s_values, ref_labels)
    print(f"\noutputs written to {out_dir}")
    if show:
        plt.show()