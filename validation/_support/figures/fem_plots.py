"""
validation/_support/figures/fem_plots.py

Single-solve figures for one shaft's ShaftResults -- the plotting sibling of
outputs/reports/solver_results/fem_report.py (same result shape, same "content vs
writer" split). Ported from the old fixtures/studies/shafts/fem_studies/
outputs/plots.py; the figure design is unchanged.

Discipline: every *_figure() function is PURE (ShaftResults in, matplotlib
Figure out) and never calls savefig/show. Only write_shaft_figures() and
write_fem_plots() touch the filesystem.

Figures per shaft (9 files, same names as before):

    bending_moment_components.png   M_xz (solid) / M_xy (dashed), one axis
    bending_moment_resultant.png    M(x) on top, phase underneath
    shear_components.png            V_xz / V_xy
    shear_resultant.png             V(x) + phase
    deflection_components.png       v_xz / v_xy
    deflection_resultant.png        v(x) + phase
    torsion.png                     T(x)            [N.m]
    bending_stress.png              sigma_b(x) + phase of the M vector
    shear_stress.png                tau(x)

Design conventions (kept from the old module):
  - Monochrome: xz solid, xy dashed, both black -- colour carried no
    information (they are two orthogonal planes, not two quantities).
  - Phase = degrees(atan2(xz, xy)): 0 deg = XY plane, +-90 deg = XZ plane,
    the same convention as loads.py's theta_deg (Fy = F cos(theta),
    Fz = F sin(theta), theta from +Y toward +Z).
  - The phase axis is FIXED at -180..180 deg (ticks every 45 deg), never
    autoscaled, so the same angle always sits at the same height across
    figures and shafts.
  - sigma_b is derived from the RESULTANT moment, so its phase is the phase
    of (M_xz, M_xy) -- computed from those, not from sigma_b (a scalar).
  - tau and T have no xz/xy pair, so they are single-line figures.
  - d(x), W(x), Wt(x) and bearing-reaction bar charts are deliberately not
    plotted (text report only).

CHANGES relative to the old module:
  1. Phase is MASKED (NaN) where the vector magnitude is negligible
     (< 1e-6 of the curve's own maximum). Before, atan2 of two signed zeros
     returned +-180 deg -- e.g. atan2(0.0, -0.0) = 180, atan2(-0.0, -0.0) =
     -180 -- so every node where M, V or v is exactly zero (shaft ends,
     supports, free ends) drew a spurious +-180 deg spike. Those nodes are
     common: the reports print "-0.0" at x=0 and at the supports.
  2. Figures are built with matplotlib.figure.Figure directly instead of
     pyplot. No GUI backend is needed (safe headless / in batch runs), and
     there is no global figure registry to leak, so no plt.close() dance.
  3. Inputs are now (system, results: dict[name -> ShaftResults]), the same
     shape as write_fem_report(); the results library no longer exists.
  4. Docstring reduced to what the code does today (the old one was a
     revision log).

Not plotted yet, but available on ShaftResults now: phi (twist angle, rad)
and u (axial displacement, mm). The old docstring said neither existed.
Each would be one function plus one _FIGURE_REGISTRY entry.

Folder layout: <base_dir>/plots[/<subdir>]/<shaft_name>/<file>.png
(one subfolder per shaft; `subdir` keeps several result sets side by side).

Dependency (results only, read-only): axisforge.results.fem_results.
shaft_results.ShaftResults (TYPE_CHECKING only).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:  # pragma: no cover
    from axisforge.core.mechanical_system.parallel_axis.spur_helical.gear_system import (
        SpurHelicalGearSystem,
    )
    from axisforge.results.fem_results.shaft_results import ShaftResults

__all__ = [
    "bending_moment_components_figure", "bending_moment_resultant_figure",
    "shear_components_figure", "shear_resultant_figure",
    "deflection_components_figure", "deflection_resultant_figure",
    "torsion_figure",
    "bending_stress_figure", "shear_stress_figure",
    "write_shaft_figures",
    "write_fem_plots",
]

_PHASE_TICKS = [-180, -135, -90, -45, 0, 45, 90, 135, 180]


# ---------------------------------------------------------------------------
# Shared figure builders
# ---------------------------------------------------------------------------

def _phase_deg(y_xz, y_xy) -> np.ndarray:
    """degrees(atan2(xz, xy)) at every node -- raw, as in the old plots.py.

    No masking: the phase is plotted wherever the results have data, so
    the figure shows exactly what the solver returned (including the
    ill-defined phase where the resultant is ~0)."""
    return np.degrees(np.arctan2(np.asarray(y_xz, dtype=float),
                                 np.asarray(y_xy, dtype=float)))


def _line_figure(
    x, y, *,
    title: str, ylabel: str, xlabel: str = "x [mm]", color: str = "black",
    mark: tuple[float, float, str] | None = None,
) -> Figure:
    """One line, one axis. `mark` = (x, y, label) draws a red marker plus a
    dotted crosshair and an annotation (unused by the figures below; kept
    for a future single-component figure)."""
    fig = Figure(figsize=(9, 5))
    ax = fig.subplots()
    ax.plot(x, y, "-", color=color, linewidth=1.6)

    if mark is not None:
        x_m, y_m, label = mark
        ax.plot([x_m], [y_m], "o", color="tab:red", markersize=7, zorder=5)
        ax.axhline(y_m, color="tab:red", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.axvline(x_m, color="tab:red", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.annotate(
            f"{label} = {y_m:.4g}\n@ x = {x_m:.2f} mm",
            xy=(x_m, y_m), xytext=(10, 10), textcoords="offset points",
            fontsize=8, color="tab:red",
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _components_figure(
    x, y_xz, y_xy, *, title: str, ylabel: str, xlabel: str = "x [mm]",
) -> Figure:
    """xz (solid) and xy (dashed) together on one axis, both black."""
    fig = Figure(figsize=(9, 5))
    ax = fig.subplots()
    ax.plot(x, y_xz, "-", color="black", linewidth=1.6, label="xz")
    ax.plot(x, y_xy, "--", color="black", linewidth=1.6, label="xy")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def _resultant_and_phase_figure(
    x, y_xz, y_xy, y_resultant, *,
    title: str, ylabel_resultant: str, xlabel: str = "x [mm]",
) -> Figure:
    """Two subplots sharing x: resultant magnitude on top, phase angle
    underneath (fixed -180..180 deg axis, gridline every 45 deg; NaN --
    i.e. a gap in the line -- where the vector is ~0)."""
    fig = Figure(figsize=(9, 8))
    ax_top, ax_bot = fig.subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": [2, 1.3]},
    )

    ax_top.plot(x, y_resultant, "-", color="black", linewidth=1.6)
    ax_top.set_title(title)
    ax_top.set_ylabel(ylabel_resultant)
    ax_top.grid(True, alpha=0.3)

    ax_bot.plot(x, _phase_deg(y_xz, y_xy), "-", color="black", linewidth=1.4, zorder=3)
    for t in _PHASE_TICKS:
        ax_bot.axhline(
            t, color="gray", linewidth=1.0 if t == 0 else 0.6,
            linestyle="-" if t == 0 else "--", alpha=0.7 if t == 0 else 0.4,
            zorder=1,
        )
    ax_bot.set_ylim(-180.0, 180.0)
    ax_bot.set_yticks(_PHASE_TICKS)
    ax_bot.set_ylabel("phase [deg]  (0 deg = xy plane)")
    ax_bot.set_xlabel(xlabel)
    ax_bot.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    return fig


def _shaft_title(shaft_name: str | None, quantity: str) -> str:
    return f"{shaft_name} -- {quantity}" if shaft_name else quantity


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def bending_moment_components_figure(result: "ShaftResults", shaft_name: str | None = None) -> Figure:
    return _components_figure(
        result.x_nodes, result.M_xz, result.M_xy,
        title=_shaft_title(shaft_name, "Bending moment -- M_xz / M_xy(x)"),
        ylabel="M [N.mm]",
    )


def bending_moment_resultant_figure(result: "ShaftResults", shaft_name: str | None = None) -> Figure:
    return _resultant_and_phase_figure(
        result.x_nodes, result.M_xz, result.M_xy, result.M,
        title=_shaft_title(shaft_name, "Resultant bending moment M(x)"),
        ylabel_resultant="M [N.mm]",
    )


def shear_components_figure(result: "ShaftResults", shaft_name: str | None = None) -> Figure:
    return _components_figure(
        result.x_nodes, result.V_xz, result.V_xy,
        title=_shaft_title(shaft_name, "Shear force -- V_xz / V_xy(x)"),
        ylabel="V [N]",
    )


def shear_resultant_figure(result: "ShaftResults", shaft_name: str | None = None) -> Figure:
    return _resultant_and_phase_figure(
        result.x_nodes, result.V_xz, result.V_xy, result.V,
        title=_shaft_title(shaft_name, "Resultant shear force V(x)"),
        ylabel_resultant="V [N]",
    )


def deflection_components_figure(result: "ShaftResults", shaft_name: str | None = None) -> Figure:
    return _components_figure(
        result.x_nodes, result.v_xz, result.v_xy,
        title=_shaft_title(shaft_name, "Deflection -- v_xz / v_xy(x)"),
        ylabel="v [mm]",
    )


def deflection_resultant_figure(result: "ShaftResults", shaft_name: str | None = None) -> Figure:
    return _resultant_and_phase_figure(
        result.x_nodes, result.v_xz, result.v_xy, result.v,
        title=_shaft_title(shaft_name, "Resultant deflection v(x)"),
        ylabel_resultant="v [mm]",
    )


def torsion_figure(result: "ShaftResults", shaft_name: str | None = None) -> Figure:
    # T is in N.m (not N.mm) -- same unit discontinuity as ShaftResults.
    return _line_figure(
        result.x_nodes, result.T,
        title=_shaft_title(shaft_name, "Torque T(x)"),
        ylabel="T [N.m]",
    )


def bending_stress_figure(result: "ShaftResults", shaft_name: str | None = None) -> Figure:
    return _resultant_and_phase_figure(
        result.x_nodes, result.M_xz, result.M_xy, result.sigma_b,
        title=_shaft_title(shaft_name, "Bending stress sigma_b(x)"),
        ylabel_resultant="sigma_b [MPa]",
    )


def shear_stress_figure(result: "ShaftResults", shaft_name: str | None = None) -> Figure:
    return _line_figure(
        result.x_nodes, result.tau,
        title=_shaft_title(shaft_name, "Torsional shear stress tau(x)"),
        ylabel="tau [MPa]",
    )


# ---------------------------------------------------------------------------
# Registry + writers
# ---------------------------------------------------------------------------

_FIGURE_REGISTRY: dict[str, Callable[["ShaftResults", "str | None"], Figure]] = {
    "bending_moment_components.png": bending_moment_components_figure,
    "bending_moment_resultant.png": bending_moment_resultant_figure,
    "shear_components.png": shear_components_figure,
    "shear_resultant.png": shear_resultant_figure,
    "deflection_components.png": deflection_components_figure,
    "deflection_resultant.png": deflection_resultant_figure,
    "torsion.png": torsion_figure,
    "bending_stress.png": bending_stress_figure,
    "shear_stress.png": shear_stress_figure,
}


def write_shaft_figures(
    result: "ShaftResults",
    shaft_name: str,
    base_dir: "str | Path",
    dpi: int = 150,
    subdir: str | None = None,
) -> list[Path]:
    """
    Build and save every figure in _FIGURE_REGISTRY for ONE shaft under
    <base_dir>/plots[/<subdir>]/<shaft_name>/. Returns the written paths.
    """
    out_dir = Path(base_dir) / "plots" / (subdir or "") / shaft_name
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for filename, build_fn in _FIGURE_REGISTRY.items():
        fig = build_fn(result, shaft_name)
        out_path = out_dir / filename
        fig.savefig(out_path, dpi=dpi)
        written.append(out_path)
    return written


def write_fem_plots(
    system: "SpurHelicalGearSystem",
    results: "dict[str, ShaftResults]",
    base_dir: "str | Path",
    dpi: int = 150,
    subdir: str | None = None,
) -> dict[str, list[Path]]:
    """
    Walk system.shafts (the system's own order) and write the figures for
    each shaft that has an entry in `results`. Shafts without a result are
    skipped (absent from the returned dict, not present with an empty
    list). Returns {shaft_name: [written PNG paths]}.
    """
    out: dict[str, list[Path]] = {}
    for ss in system.shafts:
        result = results.get(ss.name)
        if result is None:
            continue
        out[ss.name] = write_shaft_figures(result, ss.name, base_dir, dpi=dpi, subdir=subdir)
    return out