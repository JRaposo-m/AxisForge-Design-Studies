"""
validation/fem_studies/resolution/uniform_shafts/point_load/case_radial_single_position/analytical/comparison/compare_analytical_case_radial_single_position.py

Comparison runner for the closed-form analytical solution
(analytical/analytical_case_radial_single_position.py) -- diffs each
shaft's closed-form CSV against that SAME theory's own AxisForge FEM
resolution CSV (euler_bernoulli/results/csv/ or
timoshenko/results/<shear_theory>/csv/, one level up from analytical/
at the case root). This is a DIFFERENT reference than the Abaqus
comparisons in euler_bernoulli/comparison/ and timoshenko/comparison/
-- those diff AxisForge vs Abaqus; this one diffs AxisForge vs
closed-form, independent of Abaqus entirely. Nothing here compares
Abaqus against the analytical solution directly (nothing stops you
from doing that by hand from their own CSVs if a 3-way discrepancy
ever needs disentangling).

Lives inside case_radial_single_position/analytical/comparison/ --
common/paths.py's case_dir() pops the one "comparison" level, landing
on case_radial_single_position/analytical/ (NOT the true case root --
same mechanism euler_bernoulli/comparison/ and timoshenko/comparison/
already rely on, see case_dir()'s own docstring). results_dir()/
comparison_dir() from here therefore resolve against analytical/'s own
results/ and comparison/ with the theory as an explicit extra segment
(analytical/ has ONE results/ folder shared by both theories, split
internally -- see paths.py's own docstring). The FEM's own resolution
CSVs are NOT reachable that way (they live under euler_bernoulli/ and
timoshenko/, siblings of analytical/, not under it) -- built directly
off case_dir(__file__).parent (the true case root) instead.
"""
from __future__ import annotations

import sys
from pathlib import Path

_p = Path(__file__).resolve()
_root = None
for _ancestor in _p.parents:
    if (_ancestor / "common").is_dir():
        _root = _ancestor
        break
if _root is None:
    raise RuntimeError(
        f"{__file__}: no ancestor directory containing 'common/' "
        f"found -- this file must live somewhere inside the suite tree."
    )
sys.path.insert(0, str(_root))
from common.paths import case_dir, results_dir, comparison_dir  # noqa: E402
from common.compare import compare_shaft, load_abaqus_csv  # noqa: E402


SHAFT_NAMES = ("shaft1", "shaft2")
SHEAR_THEORIES = ("cowper", "hutchinson")

# Analytical CSV columns already match AxisForge's own naming (v_xy_mm,
# M_xy_Nmm, V_xy_N, etc. -- see analytical_case_radial_single_position.py's
# own _CSV_COLUMNS) and units, so this is an IDENTITY map/scale, no
# "_m"-suffix translation the Abaqus side needs (see common/compare.py's
# own docstring on that). load_abaqus_csv is reused purely as a generic
# "CSV with an x_mm column" loader here -- nothing Abaqus-specific
# about it despite the name; common/compare.py's own docstring already
# notes this module only knows how to diff two already-produced CSVs.
#
# Bending + shear + deflection, both per-plane AND the resultant --
# matches the analytical CSV's own column set, which in turn mirrors
# resolution_csv.py's BENDING & SHEAR / DEFLECTION columns (see that
# script's own comment on why u_mm/theta_*/T_Nm are NOT included: this
# closed-form solver only models bending + transverse shear). Every
# compare.py column gets its own report table AND its own overlay plot
# automatically (write_comparison_plots() iterates column_map) -- no
# separate plotting code needed here for M/V on top of v.
_COLUMN_MAP = {
    "v_xy_mm": "v_xy_mm", "v_xz_mm": "v_xz_mm", "v_mm": "v_mm",
    "M_xy_Nmm": "M_xy_Nmm", "M_xz_Nmm": "M_xz_Nmm", "M_Nmm": "M_Nmm",
    "V_xy_N": "V_xy_N", "V_xz_N": "V_xz_N", "V_N": "V_N",
}
_COLUMN_SCALE = {col: 1.0 for col in _COLUMN_MAP}

# True case root -- ONE level up from analytical/, where
# euler_bernoulli/ and timoshenko/ (the FEM's own output) actually
# live, as siblings of analytical/ rather than children of it.
_CASE_ROOT = case_dir(__file__).parent


def _run(theory_label: str, fem_csv_dir: Path,
         analytical_extra: tuple, comparison_extra: tuple) -> None:
    analytical_csv_dir = results_dir(__file__, *analytical_extra) / "csv"
    out_base = comparison_dir(__file__, *comparison_extra)

    for shaft in SHAFT_NAMES:
        fem_csv = fem_csv_dir / f"{shaft}_resolution.csv"
        analytical_csv = analytical_csv_dir / f"{shaft}_analytical.csv"
        out_dir = out_base / shaft

        if not fem_csv.exists():
            print(f"  [SKIP] {shaft} ({theory_label}): {fem_csv} not found -- "
                  f"run case_radial_single_position.py first.")
            continue
        if not analytical_csv.exists():
            print(f"  [SKIP] {shaft} ({theory_label}): {analytical_csv} not found -- "
                  f"run analytical_case_radial_single_position.py first.")
            continue

        compare_shaft(
            fem_csv, analytical_csv, out_dir, shaft_name=shaft,
            column_map=_COLUMN_MAP, column_scale=_COLUMN_SCALE,
            abaqus_loader=load_abaqus_csv,
            reference_label="Analytical",
            title=(f"case_radial_single_position -- {shaft} "
                   f"({theory_label}) -- AxisForge FEM vs closed-form analytical"),
        )
        print(f"  [OK] {shaft} ({theory_label}): "
              f"{out_dir / f'{shaft}_comparison_report.txt'}")


def main() -> None:
    print("[euler_bernoulli]")
    _run(
        "euler_bernoulli",
        fem_csv_dir=_CASE_ROOT / "euler_bernoulli" / "results" / "csv",
        analytical_extra=("euler_bernoulli",),
        comparison_extra=("euler_bernoulli",),
    )

    for shear_theory in SHEAR_THEORIES:
        print(f"[timoshenko/{shear_theory}]")
        _run(
            f"timoshenko/{shear_theory}",
            fem_csv_dir=_CASE_ROOT / "timoshenko" / "results" / shear_theory / "csv",
            analytical_extra=("timoshenko", shear_theory),
            comparison_extra=("timoshenko", shear_theory),
        )


if __name__ == "__main__":
    main()
