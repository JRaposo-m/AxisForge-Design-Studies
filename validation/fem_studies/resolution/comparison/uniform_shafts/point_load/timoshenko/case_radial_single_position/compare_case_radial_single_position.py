"""
validation/abaqus_comparison/comparison/uniform_shafts/point_load/timoshenko/case_radial_single_position/compare_case_radial_single_position.py

Comparison runner for case_radial_single_position.py -- diffs every
shaft's AxisForge resolution CSV (both shear theories) against the
matching hand-exported Abaqus CSV. Uses scripts/common/compare.py for
the actual diff/report/plot work; this script only knows WHICH files
belong to THIS case, via scripts/common/paths.py.

Deliberately placed INSIDE comparison/.../case_radial_single_position/
-- next to the output it produces -- rather than under scripts/ beside
case_radial_single_position.py. That's a valid choice (your own
organisation preference): scripts/common/paths.py's root discovery
finds the suite root (the ancestor with both scripts/ and results/ as
subdirectories) from wherever this file sits, so the sys.path bootstrap
below works the same regardless. What does NOT work the same:
case_subtree() can no longer infer "which case does this belong to"
from this file's OWN position (it isn't under scripts/ at all), so
`CASE_SUBTREE` below says it explicitly instead -- see
`case_subtree_override` in scripts/common/paths.py's own docstring.

Expected inputs (you provide the abaqus_results/ ones by hand -- see
abaqus_results/README.md for the exact convention):

  results/uniform_shafts/point_load/timoshenko/case_radial_single_position/
      cowper/csv/shaft1_resolution.csv, shaft2_resolution.csv
      hutchinson/csv/shaft1_resolution.csv, shaft2_resolution.csv
      (already produced by case_radial_single_position.py, which lives
      under scripts/uniform_shafts/point_load/timoshenko/)

  abaqus_results/uniform_shafts/point_load/timoshenko/case_radial_single_position/
      shaft1_abaqus.csv, shaft2_abaqus.csv
      (ONE Abaqus export per shaft -- not per shear_theory: the Abaqus
      model's own transverse-shear formulation is a single fixed
      setting in that model, it isn't a cowper/hutchinson pair, so both
      AxisForge shear-theory runs are compared against the SAME Abaqus
      file for a given shaft. Saved EXACTLY as the Abaqus XY Data
      report gives it to you -- no header, ';' separator, ',' decimal,
      x repeated per curve -- see abaqus_results/README.md, "formato
      raw". This runner uses load_abaqus_raw_paired_csv accordingly;
      switch to compare.load_abaqus_csv below if you ever hand-clean a
      file into the plain-header convention instead.)

Outputs, written right alongside this script, mirroring results/'s own
per-shear_theory split:

  cowper/shaft1_comparison.csv, _report.txt, _comparison_v_xy_mm.png, ...
  cowper/shaft2_comparison...
  hutchinson/shaft1_comparison...
  hutchinson/shaft2_comparison...
"""
from __future__ import annotations

import sys
from pathlib import Path

_p = Path(__file__).resolve()
_root = None
for _ancestor in _p.parents:
    if (_ancestor / "scripts").is_dir() and (_ancestor / "results").is_dir():
        _root = _ancestor
        break
if _root is None:
    raise RuntimeError(
        f"{__file__}: no ancestor directory containing both 'scripts/' "
        f"and 'results/' found -- this file must live somewhere inside "
        f"the abaqus_comparison suite tree."
    )
sys.path.insert(0, str(_root / "scripts"))
from common.paths import results_dir, abaqus_results_dir, comparison_dir  # noqa: E402
from common.compare import (  # noqa: E402
    compare_shaft, DEFAULT_COLUMN_MAP, load_abaqus_raw_paired_csv,
)


SHAFT_NAMES = ("shaft1", "shaft2")
SHEAR_THEORIES = ("cowper", "hutchinson")

# This script's own position (under comparison/.../case_radial_single_position/)
# does NOT say which case it belongs to -- it isn't under scripts/, so
# there's no "position relative to scripts/" to infer from. Say it
# directly instead; must match the case_*.py's own location under
# scripts/ (uniform_shafts/point_load/timoshenko/case_radial_single_position.py).
CASE_SUBTREE = "uniform_shafts/point_load/timoshenko/case_radial_single_position"


def main() -> None:
    abq_dir = abaqus_results_dir(__file__, case_subtree_override=CASE_SUBTREE)

    for shear_theory in SHEAR_THEORIES:
        csv_dir = results_dir(__file__, case_subtree_override=CASE_SUBTREE) / shear_theory / "csv"
        out_dir = comparison_dir(__file__, shear_theory, case_subtree_override=CASE_SUBTREE)

        print(f"[{shear_theory}]")
        for shaft in SHAFT_NAMES:
            axisforge_csv = csv_dir / f"{shaft}_resolution.csv"
            abaqus_csv = abq_dir / f"{shaft}_abaqus.csv"

            # Skip THIS shaft only -- not the whole run -- if either side
            # is missing. A case can have its Abaqus model built one
            # shaft at a time; a missing shaft2 export shouldn't block
            # comparing the shaft1 you already have.
            if not axisforge_csv.exists():
                print(f"  [SKIP] {shaft}: {axisforge_csv} not found -- "
                      f"run case_radial_single_position.py first.")
                continue
            if not abaqus_csv.exists():
                print(f"  [SKIP] {shaft}: {abaqus_csv} not found -- "
                      f"see abaqus_results/README.md for the expected "
                      f"filename/format, then re-run.")
                continue

            compare_shaft(
                axisforge_csv, abaqus_csv, out_dir, shaft_name=shaft,
                column_map=DEFAULT_COLUMN_MAP,
                abaqus_loader=load_abaqus_raw_paired_csv,
                title=(f"case_radial_single_position -- {shaft} "
                       f"({shear_theory}) -- AxisForge vs Abaqus"),
            )
            print(f"  [OK] {shaft}: {out_dir / f'{shaft}_comparison_report.txt'}")


if __name__ == "__main__":
    main()