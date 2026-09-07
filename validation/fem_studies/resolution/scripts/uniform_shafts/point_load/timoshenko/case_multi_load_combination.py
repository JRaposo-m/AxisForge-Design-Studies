"""
validation/abaqus_comparison/scripts/uniform_shafts/point_load/timoshenko/case_multi_load_combination.py

Case group: uniform shaft, Timoshenko beam theory, FULL combination --
two RadialLoads at different positions/angles, one AxialLoad, one
ExternalMoment, all on the SAME shaft. This is the stress-test case for
this folder: if the solver's superposition of point loads (RadialLoad
decomposed per-plane, AxialLoad on the u DOF, ExternalMoment decomposed
per-plane, plus the gear-mesh's own point Fr/Ft/T) matches Abaqus here,
the simpler single-load-type cases in this folder
(case_radial_single_position.py, case_axial_and_moment.py) are
lower-risk by construction.

Two shafts, two different combinations (not the same combination twice):

    shaft1  RadialLoad A @ x=50.0 mm  (500 N, theta=0 deg)
            RadialLoad B @ x=150.0 mm (300 N, theta=180 deg)
            AxialLoad    @ x=190.0 mm (1500 N, tension)
            (no external moment -- radial+radial+axial combination)

    shaft2  RadialLoad   @ x=100.0 mm (500 N, theta=270 deg)
            AxialLoad    @ x=100.0 mm (1500 N, tension)
            ExternalMoment @ x=100.0 mm (15000 N*mm, theta=90 deg)
            (all three at the SAME station -- worst-case superposition
            at one node, radial+axial+moment combination)

BC: fixed for this whole suite -- ball (locating) @10mm, roller
(non-locating) @190mm -- see scripts/common/bearings.py.
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

from axisforge.core.loads import AxialLoad, ExternalMoment, RadialLoad  # noqa: E402
from axisforge.fixtures.construction.construction_capabilities import ConstructionCapabilities  # noqa: E402
from axisforge.fixtures.construction.outputs.text_report import write_construction_report  # noqa: E402
from axisforge.fixtures.studies.study_capabilities import StudyCapabilities  # noqa: E402
from axisforge.fixtures.studies.outputs.text_report import write_studies_report  # noqa: E402
from axisforge.fixtures.studies.shafts.fem_studies.outputs.plots import write_resolution_plots  # noqa: E402
from axisforge.fixtures.studies.shafts.fem_studies.outputs.resolution_csv import resolution_csv  # noqa: E402


RESULTS_DIR = results_dir(__file__)

SHAFT_LENGTH_MM = 200.0
SHAFT_DIAMETER_MM = 40.0


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

    # shaft1: radial + radial + axial, three different stations
    shaft1_loads = (
        RadialLoad(50.0, 500.0, theta_deg=0.0, label="radial_A"),
        RadialLoad(150.0, 300.0, theta_deg=180.0, label="radial_B"),
        AxialLoad(190.0, 1500.0, label="axial_A"),
    )
    # shaft2: radial + axial + moment, ALL at the same station (x=100)
    shaft2_loads = (
        RadialLoad(100.0, 500.0, theta_deg=270.0, label="radial_C"),
        AxialLoad(100.0, 1500.0, label="axial_B"),
        ExternalMoment(100.0, 15_000.0, theta_deg=90.0, label="moment_A"),
    )

    shaft_specs = [
        ShaftSpec(shaft=shaft1, bearings=bearings1, speed_rpm=1450.0,
                  loads=shaft1_loads, name="shaft1"),
        ShaftSpec(shaft=shaft2, bearings=bearings2, speed_rpm=725.0,
                  loads=shaft2_loads, name="shaft2"),
    ]
    stage_specs = [
        StageSpec(gear_driver=g_driver, gear_driven=g_driven, phi_deg=0.0, label="stage1"),
    ]

    system = build_linear_system(
        shaft_specs, stage_specs, P=1000.0, rotation_dir_source=1,
        label="uniform_point_load_multi_combination",
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
        title="uniform_shafts/point_load/timoshenko/case_multi_load_combination -- Construction",
    )
    print(f"[OK] construction_report.txt written under {RESULTS_DIR} "
          f"-- use this to build the equivalent Abaqus model")

    study = StudyCapabilities(construction=construction, shaft_fem=("shaft_fem.timoshenko_rigid",))
    solve_system = study.resolve()["solve_system"]

    print("uniform_shafts/point_load/timoshenko/case_multi_load_combination")
    print("  shaft1: RadialLoad@50 + RadialLoad@150 + AxialLoad@190")
    print("  shaft2: RadialLoad + AxialLoad + ExternalMoment, all @ x=100")

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
            title=f"uniform_shafts/point_load/timoshenko -- full combination ({shear_theory})",
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