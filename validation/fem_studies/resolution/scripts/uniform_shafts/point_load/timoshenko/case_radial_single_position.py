"""
validation/abaqus_comparison/scripts/uniform_shafts/point_load/timoshenko/case_radial_single_position.py

Case group: uniform shaft, Timoshenko beam theory, ONE extra point
RadialLoad, position varied. See the suite's top-level README.md for the
scripts/ vs results/ vs abaqus_results/ vs comparison/ split, and for the
"vehicle, not mechanism" discipline shared by every case script.

A single 1-stage gear chain (2 shafts) is built purely to get two
independent, individually-resolved ShaftSystem objects out of one script
run; the gear mesh itself is a SPUR pair (zero axial thrust, so it never
pollutes the axial DOF that the axial/moment case group cares about)
with small, FIXED, point Fr/Ft -- present identically in every case in
this file, not what is being compared.

What actually varies between the two shafts is the POSITION of one
explicit RadialLoad (core.loads.RadialLoad, source="user"):

    shaft1  extra RadialLoad @ x= 60.0 mm  (near the locating/ball end)
    shaft2  extra RadialLoad @ x=140.0 mm  (near the non-locating/roller end)

Same magnitude/direction on both (500 N, theta=270 deg) -- only the axial
position changes, so a diff against the two equivalent Abaqus models
isolates the effect of load position on v(x)/M(x)/bearing reactions,
with everything else (BC, section, gear-mesh background load) held fixed.

BC: fixed for this whole suite -- ball (locating) @10mm, roller
(non-locating) @190mm -- see scripts/common/bearings.py.

Outputs: results/uniform_shafts/point_load/timoshenko/case_radial_single_position/<shear_theory>/
(report .txt, plots/*.png, csv/*.csv) -- see scripts/common/paths.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

_p = Path(__file__).resolve()
while _p.name != "scripts":
    _p = _p.parent
sys.path.insert(0, str(_p))
from common.bearings import build_case_bearings  # noqa: E402
from common.paths import results_dir  # noqa: E402

from axisforge.core.loads import RadialLoad  # noqa: E402
from axisforge.fixtures.construction.construction_capabilities import ConstructionCapabilities  # noqa: E402
from axisforge.fixtures.construction.outputs.text_report import write_construction_report  # noqa: E402
from axisforge.fixtures.studies.study_capabilities import StudyCapabilities  # noqa: E402
from axisforge.fixtures.studies.outputs.text_report import write_studies_report  # noqa: E402
from axisforge.fixtures.studies.shafts.fem_studies.outputs.plots import write_resolution_plots  # noqa: E402
from axisforge.fixtures.studies.shafts.fem_studies.outputs.resolution_csv import resolution_csv  # noqa: E402


RESULTS_DIR = results_dir(__file__)

SHAFT_LENGTH_MM = 200.0
SHAFT_DIAMETER_MM = 40.0

LOAD_N = 500.0
LOAD_THETA_DEG = 270.0
LOAD_X_SHAFT1_MM = 60.0
LOAD_X_SHAFT2_MM = 140.0


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

    load1 = RadialLoad(LOAD_X_SHAFT1_MM, LOAD_N, theta_deg=LOAD_THETA_DEG, label="radial_test_load")
    load2 = RadialLoad(LOAD_X_SHAFT2_MM, LOAD_N, theta_deg=LOAD_THETA_DEG, label="radial_test_load")

    shaft_specs = [
        ShaftSpec(shaft=shaft1, bearings=bearings1, speed_rpm=1450.0,
                  loads=(load1,), name="shaft1"),
        ShaftSpec(shaft=shaft2, bearings=bearings2, speed_rpm=725.0,
                  loads=(load2,), name="shaft2"),
    ]
    stage_specs = [
        StageSpec(gear_driver=g_driver, gear_driven=g_driven, phi_deg=0.0, label="stage1"),
    ]

    system = build_linear_system(
        shaft_specs, stage_specs, P=1000.0, rotation_dir_source=1,
        label="uniform_point_load_radial_single_position",
    )
    return construction, system


def main() -> None:
    construction, system = build_system()

    # Construction report FIRST, once per case (geometry/bearings/gears/
    # loads don't depend on shear theory) -- this is the file to build
    # the equivalent Abaqus model FROM: exact shaft sections, bearing
    # positions/types/arrangement, gear positions, and every load
    # (tagged "user" vs "gear_mesh" via loads_block()) with its own
    # position/magnitude/direction. Written at the case root, not inside
    # a <shear_theory>/ subfolder.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_construction_report(
        system, RESULTS_DIR / "construction_report.txt",
        title="uniform_shafts/point_load/timoshenko/case_radial_single_position -- Construction",
    )
    print(f"[OK] construction_report.txt written under {RESULTS_DIR} "
          f"-- use this to build the equivalent Abaqus model")

    study = StudyCapabilities(construction=construction, shaft_fem=("shaft_fem.timoshenko_rigid",))
    solve_system = study.resolve()["solve_system"]

    print("uniform_shafts/point_load/timoshenko/case_radial_single_position")
    print(f"  shaft1: RadialLoad {LOAD_N:.0f} N @ x={LOAD_X_SHAFT1_MM:.1f} mm")
    print(f"  shaft2: RadialLoad {LOAD_N:.0f} N @ x={LOAD_X_SHAFT2_MM:.1f} mm")

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
            title=f"uniform_shafts/point_load/timoshenko -- radial load, position sweep ({shear_theory})",
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