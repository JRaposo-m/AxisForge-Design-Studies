"""
validation/abaqus_comparison/uniform_shafts/point_load/timoshenko/case_axial_and_moment/comparison/compare_case_axial_and_moment.py

Comparison runner for case_axial_and_moment.py -- diffs every shaft's
AxisForge resolution CSV (both shear theories, PLUS the abaqus_kGA
override pass -- see below) against the matching hand-exported Abaqus
CSV. Same engine (scripts/common/compare.py) and same
per-case-orchestration split as
case_radial_single_position/comparison/compare_case_radial_single_position.py
-- this script only knows WHICH files belong to THIS case.

Lives inside case_axial_and_moment/comparison/, one level below
case_axial_and_moment.py itself -- see scripts/common/paths.py's
case_dir() docstring for how a script found directly inside a
"comparison" folder resolves its case root one level further up, so
results_dir()/abaqus_results_dir()/comparison_dir() all resolve
correctly from here with no extra arguments.

This case is the reason EXTENDED_COLUMN_MAP earns its keep: shaft1
carries an AxialLoad alone (Fa=2000 N @ x=190mm) and shaft2 combines
AxialLoad + ExternalMoment @ x=100mm -- so u(x) (axial displacement)
has real, non-trivial variation to validate here, unlike
case_radial_single_position where u stays near-zero throughout.
DEFAULT_COLUMN_MAP (v_xy_mm/v_xz_mm only) would validate the two
transverse deflections but say nothing about whether the axial DOF
itself is being solved correctly -- exactly the quantity this case was
built to exercise.

PASSES -- "cowper", "hutchinson", "abaqus_kGA":
    case_axial_and_moment.py now writes a THIRD results/ folder,
    results/abaqus_kGA/csv/, alongside cowper/ and hutchinson/ --
    produced with TimoshenkoBeam's kGA_override set to Abaqus's own
    reported K*G(23)*A = K*G(13)*A = 2.22606E+07 (its *Preprint,
    model=YES section-properties printout), bypassing shear_theory
    entirely for that pass (see that script's own top docstring).
    This is arguably the MOST important pass to diff against Abaqus
    here: if AxisForge's Timoshenko element reproduces Abaqus's
    u(x)/v(x)/theta(x) once fed the exact same transverse shear
    stiffness Abaqus used internally, that isolates whether any
    remaining cowper/hutchinson-vs-Abaqus gap is coming from the
    shear-correction-factor THEORY choice, or from something else in
    the element formulation. "abaqus_kGA" is compared against the SAME
    abaqus_results/*_abaqus.csv files as the other two passes -- there
    is still only ONE Abaqus model/export per shaft, this only changes
    which AxisForge run is being diffed against it.

Expected inputs (you provide the abaqus_results/ ones by hand -- see
abaqus_results/README.md for the exact convention):

  results/
      cowper/csv/shaft1_resolution.csv, shaft2_resolution.csv
      hutchinson/csv/shaft1_resolution.csv, shaft2_resolution.csv
      abaqus_kGA/csv/shaft1_resolution.csv, shaft2_resolution.csv
      (all three already produced by case_axial_and_moment.py, one
      level up)

  abaqus_results/
      shaft1_abaqus.csv, shaft2_abaqus.csv
      (ONE Abaqus export per shaft -- not per pass, same reasoning as
      case_radial_single_position's own comparison script: the Abaqus
      model's transverse-shear formulation is a single fixed setting,
      not a cowper/hutchinson/abaqus_kGA triple -- all three AxisForge
      passes are diffed against the very same two files. Each file
      MUST carry the full 5-curve export -- U1, U2, U3, UR2, UR3,
      selected and exported together from Abaqus's XY Data list, in
      that exact order, raw format: no header, ';' separator, ','
      decimal, x repeated per curve. A 4-column (U2/U3-only) export
      will raise an IndexError here -- see this suite's own validation
      conversation for that exact failure mode; if you only have a
      4-column export for this case, use DEFAULT_COLUMN_MAP and
      load_abaqus_raw_paired_csv with no value_names= override instead,
      the same way case_radial_single_position.py's own comparison
      script was reverted to do.)

Outputs, written right alongside this script, mirroring results/'s own
per-pass split:

  cowper/shaft1_comparison.csv, _report.txt, plots/shaft1_comparison_*.png, ...
  cowper/shaft2_comparison...
  hutchinson/shaft1_comparison...
  hutchinson/shaft2_comparison...
  abaqus_kGA/shaft1_comparison...
  abaqus_kGA/shaft2_comparison...
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
    compare_shaft, EXTENDED_COLUMN_MAP, EXTENDED_COLUMN_SCALE,
    load_abaqus_raw_paired_csv,
)


SHAFT_NAMES = ("shaft1", "shaft2")

# The three AxisForge passes case_axial_and_moment.py now writes under
# results/<pass>/csv/ -- "abaqus_kGA" is not a shear_theory, it is the
# kGA_override pass (see this module's own top docstring), but it is
# just another results/<label>/csv/ folder to diff here, so it slots
# into the same loop with no special-casing needed below.
PASSES = ("cowper", "hutchinson", "abaqus_kGA")

# Order these were selected/exported in Abaqus's XY Data list -- must
# match the column order in every abaqus_results/*_abaqus.csv file, see
# this module's own top docstring.
ABAQUS_CURVE_NAMES = ("U1", "U2", "U3", "UR2", "UR3")


def _abaqus_loader(path: Path):
    return load_abaqus_raw_paired_csv(path, value_names=ABAQUS_CURVE_NAMES)


def main() -> None:
    abq_dir = abaqus_results_dir(__file__)

    for pass_label in PASSES:
        csv_dir = results_dir(__file__) / pass_label / "csv"
        pass_dir = comparison_dir(__file__, pass_label)

        print(f"[{pass_label}]")
        for shaft in SHAFT_NAMES:
            axisforge_csv = csv_dir / f"{shaft}_resolution.csv"
            abaqus_csv = abq_dir / f"{shaft}_abaqus.csv"
            out_dir = pass_dir / shaft   # <-- per-shaft subfolder

            if not axisforge_csv.exists():
                print(f"  [SKIP] {shaft}: {axisforge_csv} not found -- "
                      f"run case_axial_and_moment.py first.")
                continue
            if not abaqus_csv.exists():
                print(f"  [SKIP] {shaft}: {abaqus_csv} not found -- "
                      f"see abaqus_results/README.md for the expected "
                      f"filename/format, then re-run.")
                continue

            compare_shaft(
                axisforge_csv, abaqus_csv, out_dir, shaft_name=shaft,
                column_map=EXTENDED_COLUMN_MAP,
                column_scale=EXTENDED_COLUMN_SCALE,
                abaqus_loader=_abaqus_loader,
                title=(f"case_axial_and_moment -- {shaft} "
                       f"({pass_label}) -- AxisForge vs Abaqus"),
            )
            print(f"  [OK] {shaft}: {out_dir / f'{shaft}_comparison_report.txt'}")

if __name__ == "__main__":
    main()