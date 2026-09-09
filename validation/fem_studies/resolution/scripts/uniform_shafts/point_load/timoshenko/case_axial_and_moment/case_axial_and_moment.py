"""
validation/abaqus_comparison/uniform_shafts/point_load/timoshenko/case_axial_and_moment/case_axial_and_moment.py

Case group: uniform shaft, Timoshenko beam theory, AxialLoad and
ExternalMoment (both always point loads -- core.loads.AxialLoad /
ExternalMoment have no distributed variant, unlike RadialLoad). Same
1-stage spur-gear vehicle and fixed BC as case_radial_single_position.py
-- see that case's own header and the suite's top-level README.md for
the shared discipline.

Two shafts, two different load TYPES/combinations (not just positions,
since that's already covered by case_radial_single_position.py):

    shaft1  AxialLoad only        : Fa = +2000 N (tension) @ x=190.0 mm
                                     (applied at the non-locating/roller
                                     end on purpose -- with only ONE
                                     locating bearing reacting axial load
                                     regardless of where Fa is applied,
                                     this isolates whether AxialLoad's
                                     node position matters to u(x) at all
                                     for a straight uniform shaft, a
                                     useful sanity check against Abaqus).
    shaft2  AxialLoad + ExternalMoment combined @ x=100.0 mm :
                                     Fa = +2000 N, M = 15000 N*mm,
                                     theta=90 deg -- a genuine combination
                                     case (two load types superposed at
                                     the same station), not just two
                                     loads at different positions.

BC: fixed for this whole suite -- ball (locating) @10mm, roller
(non-locating) @190mm -- see scripts/common/bearings.py.
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
from common.bearings import build_case_bearings  # noqa: E402
from common.paths import results_dir  # noqa: E402

from axisforge.core.loads import AxialLoad, ExternalMoment  # noqa: E402
from axisforge.fixtures.construction.construction_capabilities import ConstructionCapabilities  # noqa: E402
from axisforge.fixtures.construction.outputs.text_report import write_construction_report  # noqa: E402
from axisforge.fixtures.studies.study_capabilities import StudyCapabilities  # noqa: E402
from axisforge.fixtures.studies.outputs.text_report import write_studies_report  # noqa: E402
from axisforge.fixtures.studies.shafts.fem_studies.outputs.plots import write_resolution_plots  # noqa: E402
from axisforge.fixtures.studies.shafts.fem_studies.outputs.resolution_csv import resolution_csv  # noqa: E402


RESULTS_DIR = results_dir(__file__)

SHAFT_LENGTH_MM = 200.0
SHAFT_DIAMETER_MM = 40.0

AXIAL_N = 2000.0
AXIAL_X_SHAFT1_MM = 190.0
COMBINED_X_SHAFT2_MM = 100.0
MOMENT_NMM = 15_000.0
MOMENT_THETA_DEG = 90.0


def build_system() -> tuple["ConstructionCapabilities", "SpurHelicalGearSystem"]:
    construction = ConstructionCapabilities(
        shaft=("shafts.generic",),
        bearings=("bearings.deep_groove_ball", "bearings.cylindrical_roller"),
        gears=("gears.spur",),
        system=("systems.parallel_axis_linear",),
    )
    objs = construction.resolve()

    make_shaft = objs["factory"]
    SectionSpec = objs["SectionSpec"]
    make_deep_groove_ball_bearing = objs["make_deep_groove_ball_bearing"]
    make_cylindrical_roller_bearing = objs["make_cylindrical_roller_bearing"]
    make_spur_gear = objs["make_spur_gear"]
    ShaftSpec = objs["ShaftSpec"]
    StageSpec = objs["StageSpec"]
    build_linear_system = objs["build_linear_system"]

    def make_shaft_geometry(name: str):
        return make_shaft(
            sections=[SectionSpec(length=SHAFT_LENGTH_MM, diameter=SHAFT_DIAMETER_MM, label=name)],
            transitions={},
            name=name,
        ).shaft

    shaft1 = make_shaft_geometry("shaft1")
    shaft2 = make_shaft_geometry("shaft2")
    bearings1 = build_case_bearings(make_deep_groove_ball_bearing, make_cylindrical_roller_bearing, "shaft1")
    bearings2 = build_case_bearings(make_deep_groove_ball_bearing, make_cylindrical_roller_bearing, "shaft2")

    g_driver = make_spur_gear(mn=2.0, z=20, b=15.0, position=100.0, label="g_driver")
    g_driven = make_spur_gear(mn=2.0, z=40, b=15.0, position=100.0, label="g_driven")

    axial_only = AxialLoad(AXIAL_X_SHAFT1_MM, AXIAL_N, label="axial_test_load")
    axial_combined = AxialLoad(COMBINED_X_SHAFT2_MM, AXIAL_N, label="axial_test_load")
    moment_combined = ExternalMoment(
        COMBINED_X_SHAFT2_MM, MOMENT_NMM, theta_deg=MOMENT_THETA_DEG, label="moment_test_load",
    )

    shaft_specs = [
        ShaftSpec(shaft=shaft1, bearings=bearings1, speed_rpm=1450.0,
                  loads=(axial_only,), name="shaft1"),
        ShaftSpec(shaft=shaft2, bearings=bearings2, speed_rpm=725.0,
                  loads=(axial_combined, moment_combined), name="shaft2"),
    ]
    stage_specs = [
        StageSpec(gear_driver=g_driver, gear_driven=g_driven, phi_deg=0.0, label="stage1"),
    ]

    system = build_linear_system(
        shaft_specs, stage_specs, P=1000.0, rotation_dir_source=1,
        label="uniform_point_load_axial_and_moment",
    )
    return construction, system


def main() -> None:
    construction, system = build_system()

    # Construction report FIRST, once per case -- see
    # case_radial_single_position.py's own main() for why this is
    # unconditional and written outside the shear_theory loop.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_construction_report(
        system, RESULTS_DIR / "construction_report.txt",
        title="uniform_shafts/point_load/timoshenko/case_axial_and_moment -- Construction",
    )
    print(f"[OK] construction_report.txt written under {RESULTS_DIR} "
          f"-- use this to build the equivalent Abaqus model")

    study = StudyCapabilities(construction=construction, shaft_fem=("shaft_fem.timoshenko_rigid",))
    solve_system = study.resolve()["solve_system"]

    print("uniform_shafts/point_load/timoshenko/case_axial_and_moment")
    print(f"  shaft1: AxialLoad {AXIAL_N:.0f} N @ x={AXIAL_X_SHAFT1_MM:.1f} mm (alone)")
    print(f"  shaft2: AxialLoad {AXIAL_N:.0f} N + ExternalMoment {MOMENT_NMM:.0f} N*mm "
          f"@ x={COMBINED_X_SHAFT2_MM:.1f} mm (combined)")

    for shear_theory in ("cowper", "hutchinson"):
        library = solve_system(system, construction, shear_theory=shear_theory)

        for ss in system.shafts:
            r = library.get_or_none(ss.name)
            if r is None:
                continue
            print(f"    [{shear_theory}] {ss.name:8s} "
                  f"v_max={r.v_max:7.4f} mm @ x={r.x_v_max:6.1f} mm  "
                  f"sigma_b_max={r.sigma_b_max:8.2f} MPa @ x={r.x_sigma_b_max:6.1f} mm")

        out_dir = RESULTS_DIR / shear_theory
        out_dir.mkdir(parents=True, exist_ok=True)
        write_studies_report(
            system, out_dir / f"report_resolution_{shear_theory}.txt",
            title=f"uniform_shafts/point_load/timoshenko -- axial + moment ({shear_theory})",
            shaft_fem_library=library,
        )
        write_resolution_plots(library, system, out_dir)

        csv_dir = out_dir / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        for ss in system.shafts:
            r = library.get_or_none(ss.name)
            if r is None:
                continue
            (csv_dir / f"{ss.name}_resolution.csv").write_text(
                resolution_csv(r, decimals=10), encoding="utf-8",
            )
        print(f"[OK] [{shear_theory}] report + plots + csv written under {out_dir}")


if __name__ == "__main__":
    main()
