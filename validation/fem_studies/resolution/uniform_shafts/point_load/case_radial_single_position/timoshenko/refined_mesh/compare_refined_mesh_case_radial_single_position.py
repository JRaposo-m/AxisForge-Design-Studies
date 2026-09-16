"""
compare_refined_mesh_case_radial_single_position.py

Diffs the refined-mesh Timoshenko CSV (produced by
refine_mesh_case_radial_single_position.py, same folder) against the
closed-form analytical CSV -- same compare_shaft()/common/compare.py
machinery the existing compare_analytical_case_radial_single_position.py
already uses, just pointed at timoshenko/refined_mesh/results/ instead
of timoshenko/results/.

Builds paths directly off CASE_ROOT rather than common/paths.py's
case_dir()/results_dir() helpers -- those assume comparison/ sits ONE
level below euler_bernoulli/timoshenko/analytical (case root -> theory
-> comparison), but this script sits TWO levels down
(case root -> timoshenko -> refined_mesh -> comparison), a nesting
those helpers were not written for.
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
    raise RuntimeError(f"{__file__}: no ancestor directory containing 'common/' found.")
sys.path.insert(0, str(_root))
from common.compare import compare_shaft, load_abaqus_csv  # noqa: E402

SHAFT_NAMES = ("shaft1", "shaft2")
SHEAR_THEORY = "cowper"

_COLUMN_MAP = {
    "v_xy_mm": "v_xy_mm", "v_xz_mm": "v_xz_mm", "v_mm": "v_mm",
    "M_xy_Nmm": "M_xy_Nmm", "M_xz_Nmm": "M_xz_Nmm", "M_Nmm": "M_Nmm",
    "V_xy_N": "V_xy_N", "V_xz_N": "V_xz_N", "V_N": "V_N",
}
_COLUMN_SCALE = {col: 1.0 for col in _COLUMN_MAP}

CASE_ROOT = Path(__file__).resolve().parents[2]  # .../case_radial_single_position/


def main() -> None:
    fem_csv_dir = CASE_ROOT / "timoshenko" / "refined_mesh" / "results" / SHEAR_THEORY / "csv"
    analytical_csv_dir = CASE_ROOT / "analytical" / "results" / "timoshenko" / SHEAR_THEORY / "csv"
    out_base = CASE_ROOT / "timoshenko" / "refined_mesh" / "comparison" / SHEAR_THEORY

    for shaft in SHAFT_NAMES:
        fem_csv = fem_csv_dir / f"{shaft}_resolution.csv"
        analytical_csv = analytical_csv_dir / f"{shaft}_analytical.csv"
        out_dir = out_base / shaft

        if not fem_csv.exists():
            print(f"  [SKIP] {shaft}: {fem_csv} not found -- run "
                  f"refine_mesh_case_radial_single_position.py first.")
            continue
        if not analytical_csv.exists():
            print(f"  [SKIP] {shaft}: {analytical_csv} not found -- run "
                  f"analytical_case_radial_single_position.py first.")
            continue

        compare_shaft(
            fem_csv, analytical_csv, out_dir, shaft_name=shaft,
            column_map=_COLUMN_MAP, column_scale=_COLUMN_SCALE,
            abaqus_loader=load_abaqus_csv,
            reference_label="Analytical",
            title=(f"case_radial_single_position -- {shaft} "
                   f"(timoshenko/{SHEAR_THEORY}, REFINED MESH) -- AxisForge FEM vs closed-form analytical"),
        )
        print(f"  [OK] {shaft}: {out_dir / f'{shaft}_comparison_report.txt'}")


if __name__ == "__main__":
    main()
