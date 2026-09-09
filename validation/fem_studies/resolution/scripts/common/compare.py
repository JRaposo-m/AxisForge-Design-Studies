"""
validation/abaqus_comparison/scripts/common/compare.py

Shared AxisForge-vs-Abaqus comparison engine. Reads one AxisForge
resolution CSV (produced by
fixtures.studies.shafts.fem_studies.outputs.resolution_csv.resolution_csv,
already in every results/.../csv/<shaft>_resolution.csv) and one
Abaqus export CSV (hand-cleaned by you into a plain header + comma
CSV -- see abaqus_results/README.md for the exact convention), and
produces:

  <shaft>_comparison.csv          -- full merged table, one row per
                                      AxisForge x-node: x_mm, the
                                      AxisForge value, the Abaqus value
                                      (interpolated onto that x), abs
                                      diff, pct diff -- for every
                                      compared column.
  <shaft>_comparison_report.txt   -- human-readable summary: max/mean/
                                      RMS abs+pct diff per column, and
                                      how many points were excluded
                                      from pct diff (near-zero Abaqus
                                      reference -- flagged, not
                                      reported as a meaningless huge %).
  <shaft>_comparison_<col>.png    -- AxisForge vs Abaqus overlay plot,
                                      one per compared column.

Alignment: the two meshes essentially never share x-node positions
(different mesh generators). Abaqus's own column is linearly
interpolated (np.interp) onto AxisForge's x_nodes -- not the other way
round -- because AxisForge's grid is coarser and always spans the full
shaft length by construction (mesh_1D always places nodes at every
load/bearing/section-change position), so it is the safer one to
resample onto; extrapolation past Abaqus's own x-range raises instead
of silently clamping/guessing.

Column mapping: which AxisForge column corresponds to which Abaqus
column is NOT guessed -- it's an explicit dict, `DEFAULT_COLUMN_MAP`
below, overridable per call. The default was determined empirically
(see this suite's own validation conversation): Abaqus U2 <-> AxisForge
v_xy_mm (vertical plane), Abaqus U3 <-> AxisForge v_xz_mm (horizontal
plane) -- NOT U1/U2 in the naive order. If a future case also wants to
compare bending moment or section force, extend the map (and make sure
the Abaqus report actually exported that field -- SF1/SF2/SM1/SM2 for
a beam-element Abaqus model, if that's what you're building it as).

Units: AxisForge works in mm throughout (see resolution_csv's own
column names). Abaqus models are commonly built in SI (m, N, Pa)
instead, so a raw Abaqus displacement column is typically 1000x an
AxisForge one -- not a rounding difference, a unit mismatch that would
silently produce ~1000x "error" numbers if not corrected. Column names
that self-document their unit (e.g. "U2_m") are expected and
encouraged; `DEFAULT_COLUMN_SCALE` gives the multiplier applied to each
mapped Abaqus column BEFORE diffing (1000.0 for a "_m" -> mm column,
1.0 for anything already in mm). Extend both dicts together when you
add a column.

No case_*.py-specific knowledge here -- this module only knows how to
diff two already-produced CSVs. Per-case orchestration (which shafts,
which shear_theory folders, where the Abaqus files are expected) lives
in each case's own compare_case_*.py, using scripts/common/paths.py the
same way every case_*.py already does.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# axisforge_column -> abaqus_column. Extend per-case if you export more
# fields from Abaqus (e.g. add "M_xy_Nmm": "SM1" once you confirm your
# Abaqus model actually reports section moments at the same stations).
# Column NAMES keep the "_m" suffix because that's literally the header
# string in this suite's raw Abaqus export (see
# load_abaqus_raw_paired_csv's default value_names) -- it does NOT mean
# the values are in meters, see DEFAULT_COLUMN_SCALE below.
DEFAULT_COLUMN_MAP: dict[str, str] = {
    "v_xy_mm": "U2_m",
    "v_xz_mm": "U3_m",
}

# axisforge_column -> multiplier applied to the mapped Abaqus column
# before diffing, to bring it into the same unit as the AxisForge
# column (mm). Keep in step with DEFAULT_COLUMN_MAP -- same keys, one
# multiplier per mapped column.
#
# 1.0, NOT 1000 -- confirmed empirically against this suite's own
# case_radial_single_position data: AxisForge v_xy_mm/v_xz_mm and the
# raw Abaqus U2_m/U3_m columns already agree to within ~1-9% with NO
# scaling applied (e.g. at x=100mm: AxisForge v_xy_mm=0.0005639,
# Abaqus U2_m=0.0006051 -- ratio 1.073). This means the Abaqus model is
# built in an mm-based unit system (mm-N-MPa, the common Abaqus
# convention for mechanical parts), NOT SI meters, despite the "_m"
# column-name suffix -- that suffix is Abaqus's own XY Data curve/field
# name, not a claim about units. An earlier version of this suite
# assumed SI (scale=1000) from eyeballing raw magnitudes alone, without
# this point-by-point check -- that gave ~99.9% "error" across the
# board, which was purely the 1000x unit mismatch, not a real solver
# discrepancy. If you ever rebuild the Abaqus model in true SI units,
# change this back to 1000.0 and re-verify the same way (a handful of
# points, ratio close to 1 with no scaling = right unit already).
DEFAULT_COLUMN_SCALE: dict[str, float] = {
    "v_xy_mm": 1.0,
    "v_xz_mm": 1.0,
}

# Fraction of a column's own max |Abaqus value| below which pct_diff is
# reported as NaN (flagged) rather than a huge/meaningless percentage
# near a zero-crossing. RELATIVE, not an absolute mm/MPa/whatever value
# -- this suite's deflections range from ~1e-4 mm to several mm
# depending on the case, so a fixed absolute threshold would either
# flag everything (small-deflection cases) or nothing (large-deflection
# cases). 1% of the column's own peak is a reasonable default; override
# per call (e.g. tighten for a case where even small values are
# meaningful, loosen if the curve has many near-zero crossings).
NEAR_ZERO_FRACTION_DEFAULT = 0.01

# Decimal places both sides are rounded to BEFORE diffing. A boundary
# or symmetry point that is truly zero rarely comes back as an exact
# 0.0 from a solver -- Abaqus in particular tends to print something
# like 5.9925e-34 for a support/free-end point where the real answer
# is zero. Compared raw against AxisForge's clean 0.0, that produces a
# meaningless "-100.000 %" (0 vs 6e-34 is technically "off by 100%",
# but it's floating-point noise cancelling against floating-point
# noise, not a solver discrepancy). Rounding both columns to this many
# decimal places first collapses that noise to a clean 0.0 on both
# sides -- 10 decimals is far beyond the precision either CSV actually
# carries (both are ~6 significant digits), so it never rounds away a
# real, physically meaningful value.
ABS_ZERO_DECIMALS = 10


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_axisforge_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "x_mm" not in df.columns:
        raise ValueError(
            f"{path}: expected an 'x_mm' column (AxisForge resolution_csv "
            f"format), found {list(df.columns)}"
        )
    return df


def load_abaqus_csv(path: Path, x_col: str = "x_mm") -> pd.DataFrame:
    # skipinitialspace handles "x_mm, U2_m, U3_m"-style headers (space
    # after the comma); the explicit strip is a second safety net for
    # any stray whitespace pandas' own option doesn't catch (e.g. in a
    # data row, or if the file was hand-edited after export).
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    if x_col not in df.columns:
        raise ValueError(
            f"{path}: expected an '{x_col}' column, found {list(df.columns)} "
            f"-- pass x_col= to override, or re-export with that header "
            f"(see abaqus_results/README.md)."
        )
    return df


def load_abaqus_raw_paired_csv(
    path: Path,
    x_indices: tuple[int, int] = (0, 2),
    value_indices: tuple[int, int] = (1, 3),
    value_names: tuple[str, str] = ("U2_m", "U3_m"),
    x_col: str = "x_mm",
    sep: str = ";",
    decimal: str = ",",
    x_tol: float = 1e-3,
) -> pd.DataFrame:
    """
    Reads the RAW shape Abaqus gives you straight out of an XY Data
    report export with two curves side by side -- no header, ';'
    separator, ',' decimal, 4 columns: x, curve_1, x (repeated), curve_2
    (e.g. "60;0,000441783;60;-0,00044919"). No manual cleanup needed --
    save the export exactly as Abaqus writes it and point this at it.

    Returns the SAME shape load_abaqus_csv() would from a clean file:
    a DataFrame with `x_col` plus one column per `value_names` entry --
    so it's a drop-in for compare_shaft(..., abaqus_loader=this).

    Validates the two x columns (`x_indices`) agree within `x_tol` mm --
    Abaqus repeating x per curve is only safe to collapse into one
    column if they're actually the same points; a mismatch here usually
    means the export mixed two different result sets and raises rather
    than silently picking one.

    If your export has a different column count/order (e.g. 3 curves,
    or curve/x swapped), pass `x_indices`/`value_indices`/`value_names`
    to match -- this function never guesses the layout, it only assumes
    that layout is consistent from run to run once you've told it once.
    """
    raw = pd.read_csv(path, sep=sep, decimal=decimal, header=None)

    x_a = raw.iloc[:, x_indices[0]].to_numpy()
    x_b = raw.iloc[:, x_indices[1]].to_numpy()
    if np.max(np.abs(x_a - x_b)) > x_tol:
        raise ValueError(
            f"{path}: x columns {x_indices} disagree by more than "
            f"{x_tol} -- expected both to be the same station grid "
            f"(Abaqus repeats x once per exported curve). Got x[{x_indices[0]}]="
            f"{x_a.tolist()} vs x[{x_indices[1]}]={x_b.tolist()}."
        )

    out = {x_col: x_a}
    for idx, name in zip(value_indices, value_names):
        out[name] = raw.iloc[:, idx].to_numpy()
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Alignment + diff
# ---------------------------------------------------------------------------

def align_and_diff(
    af: pd.DataFrame,
    abq: pd.DataFrame,
    column_map: dict[str, str] = DEFAULT_COLUMN_MAP,
    column_scale: dict[str, float] = DEFAULT_COLUMN_SCALE,
    x_col_af: str = "x_mm",
    x_col_abq: str = "x_mm",
    near_zero_fraction: float = NEAR_ZERO_FRACTION_DEFAULT,
) -> pd.DataFrame:
    """
    Interpolates each mapped Abaqus column onto AxisForge's own x_nodes
    (linear, np.interp) and returns one row per AxisForge x-node with,
    for every `af_col` in `column_map`:
        <af_col>              -- AxisForge value
        <af_col>_abaqus       -- Abaqus value, interpolated onto x_af,
                                  AFTER applying column_scale[af_col]
                                  (unit conversion -- see this module's
                                  own top docstring, e.g. 1000.0 for a
                                  "_m" Abaqus column -> mm)
        <af_col>_abs_diff     -- af - abaqus (both already in the same
                                  unit at this point)
        <af_col>_pct_diff     -- 100 * abs_diff / abaqus, NaN where
                                  |abaqus value| < near_zero_fraction *
                                  max(|abaqus column|) -- a RELATIVE
                                  floor per column, not a fixed
                                  absolute value (see this module's own
                                  top docstring / NEAR_ZERO_FRACTION_DEFAULT)
                                  -- OR where the value rounds to 0.0
                                  at ABS_ZERO_DECIMALS places (a "true
                                  zero" boundary point on both sides,
                                  see ABS_ZERO_DECIMALS's own docstring).

    Both `af` and the interpolated Abaqus values are rounded to
    ABS_ZERO_DECIMALS decimal places before diffing -- see
    ABS_ZERO_DECIMALS's own docstring for why.

    Raises ValueError up front (not a silent NaN column) if a mapped
    column is missing from either input, if `column_scale` is missing
    an entry present in `column_map`, or if any AxisForge x falls
    outside the Abaqus x-range (extrapolation would be a guess, not a
    comparison).
    """
    x_af = af[x_col_af].to_numpy()
    x_abq = abq[x_col_abq].to_numpy()

    if x_af.min() < x_abq.min() - 1e-6 or x_af.max() > x_abq.max() + 1e-6:
        raise ValueError(
            f"AxisForge x range [{x_af.min():.3f}, {x_af.max():.3f}] mm "
            f"is not fully covered by the Abaqus x range "
            f"[{x_abq.min():.3f}, {x_abq.max():.3f}] mm -- refusing to "
            f"extrapolate. Check the Abaqus export covers the full shaft."
        )

    out: dict[str, np.ndarray] = {"x_mm": x_af}

    for af_col, abq_col in column_map.items():
        if af_col not in af.columns:
            raise ValueError(f"AxisForge CSV missing column '{af_col}'")
        if abq_col not in abq.columns:
            raise ValueError(
                f"Abaqus CSV missing column '{abq_col}' (mapped from "
                f"AxisForge column '{af_col}') -- found {list(abq.columns)}"
            )
        if af_col not in column_scale:
            raise ValueError(
                f"column_scale has no entry for '{af_col}' -- every key in "
                f"column_map needs a matching multiplier in column_scale "
                f"(use 1.0 if the Abaqus column is already in the same "
                f"unit as AxisForge's)."
            )

        scale = column_scale[af_col]
        af_vals = af[af_col].to_numpy()
        abq_scaled = abq[abq_col].to_numpy() * scale
        abq_interp = np.interp(x_af, x_abq, abq_scaled)

        # Collapse floating-point "almost zero" noise (e.g. Abaqus's
        # 5.9925e-34 at a true-zero support point) to a clean 0.0 on
        # both sides before diffing -- see ABS_ZERO_DECIMALS docstring.
        af_vals = np.round(af_vals, ABS_ZERO_DECIMALS)
        abq_interp = np.round(abq_interp, ABS_ZERO_DECIMALS)

        abs_diff = af_vals - abq_interp
        peak = np.max(np.abs(abq_scaled))
        near_zero = near_zero_fraction * peak if peak > 0 else 0.0
        # Absolute floor (10^-ABS_ZERO_DECIMALS) always applies too, on
        # top of the relative one -- so a point that rounded to exactly
        # 0.0 above is always flagged as near-zero, even in the (rare)
        # edge case where the relative floor alone would not have
        # caught it (e.g. a column whose own peak is also tiny).
        abs_floor = 10 ** (-ABS_ZERO_DECIMALS)
        near_zero = max(near_zero, abs_floor)
        with np.errstate(divide="ignore", invalid="ignore"):
            pct_diff = np.where(
                np.abs(abq_interp) >= near_zero,
                100.0 * abs_diff / abq_interp,
                np.nan,
            )

        out[af_col] = af_vals
        out[f"{af_col}_abaqus"] = abq_interp
        out[f"{af_col}_abs_diff"] = abs_diff
        out[f"{af_col}_pct_diff"] = pct_diff

    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Report + plots
# ---------------------------------------------------------------------------

def write_comparison_report(
    diff_df: pd.DataFrame,
    column_map: dict[str, str],
    out_path: Path,
    title: str = "",
) -> None:
    """
    Writes a per-point table FIRST -- x_mm, AxisForge value, Abaqus
    value, absolute deviation, relative deviation (%) -- one table per
    compared column, so every point can be checked by eye against the
    two source values rather than only against a summary statistic.
    The max/mean/RMS summary block still follows each table (useful for
    a one-line verdict), but the table is the primary content now.
    """
    lines = [title, "=" * len(title)] if title else []
    lines.append("")

    for af_col in column_map:
        abaqus_col_label = column_map[af_col]
        val_col = af_col
        abq_col = f"{af_col}_abaqus"
        abs_col = f"{af_col}_abs_diff"
        pct_col = f"{af_col}_pct_diff"

        x_vals = diff_df["x_mm"].to_numpy()
        af_vals = diff_df[val_col].to_numpy()
        abq_vals = diff_df[abq_col].to_numpy()
        abs_vals = diff_df[abs_col].to_numpy()
        pct_vals = diff_df[pct_col].to_numpy()
        n_total = len(pct_vals)
        n_valid = int(np.sum(~np.isnan(pct_vals)))
        n_flagged = n_total - n_valid

        lines.append(f"{af_col}  (vs Abaqus '{abaqus_col_label}')")
        lines.append("=" * 78)
        header = (
            f"{'x_mm':>10} | {'AxisForge':>14} | {'Abaqus':>14} | "
            f"{'abs diff':>14} | {'rel diff [%]':>13}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for x, af_v, abq_v, abs_v, pct_v in zip(
            x_vals, af_vals, abq_vals, abs_vals, pct_vals
        ):
            pct_str = f"{pct_v:>13.3f}" if not np.isnan(pct_v) else f"{'n/a (~0)':>13}"
            lines.append(
                f"{x:>10.3f} | {af_v:>14.6g} | {abq_v:>14.6g} | "
                f"{abs_v:>14.6g} | {pct_str}"
            )
        lines.append("")

        lines.append("resumo:")
        lines.append(f"  max |abs diff|  : {np.max(np.abs(abs_vals)):.6g}")
        lines.append(f"  mean |abs diff| : {np.mean(np.abs(abs_vals)):.6g}")
        lines.append(f"  RMS abs diff    : {np.sqrt(np.mean(abs_vals**2)):.6g}")
        if n_valid:
            lines.append(f"  max |pct diff|  : {np.nanmax(np.abs(pct_vals)):.3f} %")
            lines.append(f"  mean |pct diff| : {np.nanmean(np.abs(pct_vals)):.3f} %")
        else:
            lines.append("  pct diff        : n/a (every point near zero)")
        lines.append(
            f"  points          : {n_valid}/{n_total} usados no pct diff "
            f"({n_flagged} marcados near-zero-Abaqus, excluídos)"
        )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_comparison_plots(
    diff_df: pd.DataFrame,
    column_map: dict[str, str],
    out_dir: Path,
    shaft_name: str,
) -> None:
    x = diff_df["x_mm"].to_numpy()
    for af_col, abq_col in column_map.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x, diff_df[af_col], "o-", label=f"AxisForge {af_col}", ms=3)
        ax.plot(x, diff_df[f"{af_col}_abaqus"], "s--", label=f"Abaqus {abq_col}", ms=3)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel(af_col)
        ax.set_title(f"{shaft_name} -- {af_col} vs Abaqus {abq_col}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"{shaft_name}_comparison_{af_col}.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def compare_shaft(
    axisforge_csv: Path,
    abaqus_csv: Path,
    out_dir: Path,
    shaft_name: str,
    column_map: dict[str, str] = DEFAULT_COLUMN_MAP,
    column_scale: dict[str, float] = DEFAULT_COLUMN_SCALE,
    title: str = "",
    near_zero_fraction: float = NEAR_ZERO_FRACTION_DEFAULT,
    abaqus_loader: Callable[[Path], pd.DataFrame] = load_abaqus_csv,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    af = load_axisforge_csv(axisforge_csv)
    abq = abaqus_loader(abaqus_csv)
    diff_df = align_and_diff(
        af, abq, column_map=column_map, column_scale=column_scale,
        near_zero_fraction=near_zero_fraction,
    )

    csv_path = out_dir / f"{shaft_name}_comparison.csv"
    diff_df.to_csv(csv_path, index=False)

    write_comparison_report(
        diff_df, column_map, out_dir / f"{shaft_name}_comparison_report.txt",
        title=title or f"{shaft_name} -- AxisForge vs Abaqus",
    )
    write_comparison_plots(diff_df, column_map, plots_dir, shaft_name)

    return csv_path