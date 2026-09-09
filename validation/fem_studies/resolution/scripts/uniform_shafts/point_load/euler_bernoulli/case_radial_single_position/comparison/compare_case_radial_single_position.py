"""
validation/abaqus_comparison/uniform_shafts/point_load/euler_bernoulli/case_radial_single_position/comparison/compare_case_radial_single_position.py

Comparison runner for the Euler-Bernoulli variant of
case_radial_single_position.py -- diffs each shaft's AxisForge
resolution CSV against the matching hand-exported Abaqus CSV.

NO shear-theory loop here, unlike the timoshenko sibling script --
Euler-Bernoulli has no shear correction, so there is exactly ONE
AxisForge solve per shaft, not a cowper/hutchinson pair. results/ for
this case therefore has no <shear_theory>/ subfolder either -- csv/ and
plots/ sit directly under the case root.

Lives inside case_radial_single_position/comparison/ -- see
scripts/common/paths.py's case_dir() docstring for how a script found
directly inside a "comparison" folder resolves its case root one level
further up.
"""
from __future__ import annotations

import sys
from pathlib import Path

_p = Path(__file__).resolve()
_root = None
for _ancestor in _p.parents:
    if (_ancestor / "scripts" / "common").is_dir():
        _root = _ancestor
        break
if _root is None:
    raise RuntimeError(
        f"{__file__}: no ancestor directory containing 'scripts/common/' "
        f"found -- this file must live somewhere inside the suite tree."
    )
sys.path.insert(0, str(_root / "scripts"))
from common.paths import results_dir, abaqus_results_dir, comparison_dir  # noqa: E402
from common.compare import (  # noqa: E402
    compare_shaft, DEFAULT_COLUMN_MAP, load_abaqus_raw_paired_csv,
)


SHAFT_NAMES = ("shaft1", "shaft2")


def main() -> None:
    abq_dir = abaqus_results_dir(__file__)
    csv_dir = results_dir(__file__) / "csv"
    out_base = comparison_dir(__file__)

    for shaft in SHAFT_NAMES:
        axisforge_csv = csv_dir / f"{shaft}_resolution.csv"
        abaqus_csv = abq_dir / f"{shaft}_abaqus.csv"
        out_dir = out_base / shaft

        if not axisforge_csv.exists():
            print(f"[SKIP] {shaft}: {axisforge_csv} not found -- "
                  f"run the euler_bernoulli case_radial_single_position.py first.")
            continue
        if not abaqus_csv.exists():
            print(f"[SKIP] {shaft}: {abaqus_csv} not found -- "
                  f"see abaqus_results/README.md, then re-run.")
            continue

        compare_shaft(
            axisforge_csv, abaqus_csv, out_dir, shaft_name=shaft,
            column_map=DEFAULT_COLUMN_MAP,
            abaqus_loader=load_abaqus_raw_paired_csv,
            title=(f"case_radial_single_position -- {shaft} "
                   f"(euler_bernoulli) -- AxisForge vs Abaqus"),
        )
        print(f"[OK] {shaft}: {out_dir / f'{shaft}_comparison_report.txt'}")


if __name__ == "__main__":
    main()
