"""
validation/abaqus_comparison/uniform_shafts/point_load/euler_bernoulli/case_radial_single_position/case_radial_single_position.py

Euler-Bernoulli sibling of the Timoshenko case_radial_single_position.py
-- same "vehicle, not mechanism" 1-stage gear chain, same two shafts,
same varied-position RadialLoad. See that file's own docstring for the
full rationale; not repeated here.

NO shear-theory split -- Euler-Bernoulli has no shear correction factor
(cubic Hermite shape functions, no independent shear-strain field), so
there is exactly ONE solve per shaft, not a cowper/hutchinson pair.
Outputs therefore sit directly under the case root, no <shear_theory>/
subfolder.

Outputs: results/ (report_resolution.txt, plots/*.png, csv/*.csv),
relative to this case's own folder -- see scripts/common/paths.py.
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

from axisforge.core.loads import RadialLoad, TorqueLoad  # noqa: E402
from axisforge.fixtures.construction.construction_capabilities import ConstructionCapabilities  # noqa: E402
from axisforge.fixtures.construction.outputs.text_report import write_construction_report  # noqa: E402
from axisforge.fixtures.studies.study_capabilities import StudyCapabilities  # noqa: E402
from axisforge.fixtures.studies.outputs.text_report import write_studies_report  # noqa: E402
from axisforge.fixtures.studies.shafts.fem_studies.outputs.plots import write_resolution_plots  # noqa: E402
from axisforge.fixtures.studies.shafts.fem_studies.outputs.resolution_csv import resolution_csv  # noqa: E402


RESULTS_DIR = results_dir(__file__)

SHAFT_LENGTH_MM = 200.0
SHAFT_DIAMETER_MM = 20.0

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
        shaft_specs, stage_specs, P=10000.0, rotation_dir_source=1,
        label="uniform_point_load_radial_single_position",
    )

    # Same torsional-equilibrium fix as the timoshenko sibling -- torsion
    # is solved as pure statics (_solve_torsion), independent of which
    # beam element handles bending, so this is needed here too.
    shaft2_sys = system.shafts[-1]
    net_before = sum(ld.magnitude for ld in shaft2_sys.torque_loads)
    shaft2_sys.add_load(TorqueLoad(
        position=SHAFT_LENGTH_MM,
        magnitude=-net_before,
        label="output_coupling",
        source="user",
    ))

    return construction, system


def main() -> None:
    construction, system = build_system()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_construction_report(
        system, RESULTS_DIR / "construction_report.txt",
        title="uniform_shafts/point_load/euler_bernoulli/case_radial_single_position -- Construction",
    )
    print(f"[OK] construction_report.txt written under {RESULTS_DIR} "
          f"-- use this to build the equivalent Abaqus model")

    study = StudyCapabilities(construction=construction, shaft_fem=("shaft_fem.euler_bernoulli_rigid",))
    solve_system = study.resolve()["solve_system"]

    print("uniform_shafts/point_load/euler_bernoulli/case_radial_single_position")
    print(f"  shaft1: RadialLoad {LOAD_N:.0f} N @ x={LOAD_X_SHAFT1_MM:.1f} mm")
    print(f"  shaft2: RadialLoad {LOAD_N:.0f} N @ x={LOAD_X_SHAFT2_MM:.1f} mm")

    # Single solve -- no shear_theory loop -- theory="euler" is already
    # pinned by the "shaft_fem.euler_bernoulli_rigid" capability.
    library = solve_system(system, construction)

    for ss in system.shafts:
        r = library.get_or_none(ss.name)
        if r is None:
            continue
        print(f"    {ss.name:8s} "
              f"v_max={r.v_max:7.4f} mm @ x={r.x_v_max:6.1f} mm  "
              f"sigma_b_max={r.sigma_b_max:8.2f} MPa @ x={r.x_sigma_b_max:6.1f} mm")

    write_studies_report(
        system, RESULTS_DIR / "report_resolution.txt",
        title="uniform_shafts/point_load/euler_bernoulli -- radial load, position sweep",
        shaft_fem_library=library,
    )
    write_resolution_plots(library, system, RESULTS_DIR)

    csv_dir = RESULTS_DIR / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    for ss in system.shafts:
        r = library.get_or_none(ss.name)
        if r is None:
            continue
        (csv_dir / f"{ss.name}_resolution.csv").write_text(
            resolution_csv(r, decimals=10), encoding="utf-8",
        )
    print(f"[OK] report + plots + csv written under {RESULTS_DIR}")


if __name__ == "__main__":
    main()
