"""
check_resolution_fem_convergence_gears.py

Exploratory script (not pytest). Same 2-stage linear chain fixture as
check_resolution_fem.py (build_system() duplicated here, same
convention already used by check_resolution_fem_euler.py/
check_resolution_fem_compare.py -- each check script owns its own copy
rather than importing a shared one).

Exercises "shaft_fem.convergence.gears.timoshenko": runs
convergence_study.run_convergence() restricted to gear face-width
intervals only (regions={"gears"}, pinned by the capability -- see
study_capabilities.py's own _convergence_require()), Timoshenko theory,
for every shaft in the system. This fixture's gears all have b=15.0mm
(face width defined, above MIN_FACE_WIDTH_FOR_CONVERGENCE_MM), so all
four gear intervals across the three shafts are expected to produce a
real convergence attempt -- none skipped.

This fixture has no DistributedRadialLoad anywhere, so this check does
NOT exercise "external_distributed" -- that is deliberately left to
check_resolution_fem_convergence_external_distributed.py, which adds
one to shaft3 specifically to have something to converge on. Running
"external_distributed"/"total" against THIS unmodified fixture would
report a valid but empty (0 intervals) convergence -- not wrong, just
not a useful demonstration of that capability.

Console output stays terse (fail-loud/succeed-quiet, same convention
as check_resolution_fem.py): one summary line per shaft (how many
intervals converged out of how many attempted), not a full per-level
dump -- that detail lives in the written .txt report.

Also writes the Studies text report via
fixtures.studies.text_report.write_studies_report(), passing only
convergence_library= (no shaft_fem_library=/comparison=, since this
script only ran the convergence study) -- the SHAFT MESH CONVERGENCE
section is the only one printed, per that function's own
presence-driven design.
"""
from __future__ import annotations

from pathlib import Path

from axisforge.core.loads import RadialLoad
from axisforge.fixtures.construction.construction_capabilities import ConstructionCapabilities
from axisforge.fixtures.studies.study_capabilities import StudyCapabilities
from axisforge.fixtures.studies.outputs.text_report import write_studies_report

HERE = Path(__file__).resolve().parent


def build_system() -> tuple["ConstructionCapabilities", "SpurHelicalGearSystem"]:
    construction = ConstructionCapabilities(
        shaft=("shafts.stepped",),
        bearings=("bearings.deep_groove_ball",),
        gears=("gears.spur",),
        system=("systems.parallel_axis_linear",),
    )
    objs = construction.resolve()

    make_stepped_shaft = objs["factory"]
    SectionSpec = objs["SectionSpec"]
    make_deep_groove_ball_bearing = objs["make_deep_groove_ball_bearing"]
    make_spur_gear = objs["make_spur_gear"]
    ShaftSpec = objs["ShaftSpec"]
    StageSpec = objs["StageSpec"]
    build_linear_system = objs["build_linear_system"]

    def make_shaft_geometry(name: str):
        return make_stepped_shaft(
            sections=[
                SectionSpec(length=30.0, diameter=30.0, label=f"{name}_seat_A"),
                SectionSpec(length=140.0, diameter=50.0, label=f"{name}_body"),
                SectionSpec(length=30.0, diameter=30.0, label=f"{name}_seat_B"),
            ],
            fillet_radii=[2.0, 2.0],
            name=name,
        ).shaft

    def make_shaft_bearings(name: str):
        return (
            make_deep_groove_ball_bearing(
                d=20.0, D=42.0, Dw=7.0, Dpw=31.0, Z=9, E=25.0, s=0.02,
                position=10.0, label=f"{name}_brg_A",
            ),
            make_deep_groove_ball_bearing(
                d=20.0, D=42.0, Dw=7.0, Dpw=31.0, Z=9, E=25.0, s=0.02,
                position=190.0, label=f"{name}_brg_B",
            ),
        )

    shaft1 = make_shaft_geometry("shaft1")
    shaft2 = make_shaft_geometry("shaft2")
    shaft3 = make_shaft_geometry("shaft3")
    bearings1 = make_shaft_bearings("shaft1")
    bearings2 = make_shaft_bearings("shaft2")
    bearings3 = make_shaft_bearings("shaft3")

    g1_driver = make_spur_gear(mn=2.0, z=20, b=15.0, position=100.0, label="g1_driver")
    g2_driven = make_spur_gear(mn=2.0, z=40, b=15.0, position=100.0, label="g2_driven")
    g3_driver = make_spur_gear(mn=2.0, z=20, b=15.0, position=150.0, label="g3_driver")
    g4_driven = make_spur_gear(mn=2.0, z=40, b=15.0, position=150.0, label="g4_driven")

    sprocket_load = RadialLoad(190.0, 500.0, theta_deg=270.0, label="sprocket_pull")

    shaft_specs = [
        ShaftSpec(shaft=shaft1, bearings=bearings1, speed_rpm=1450.0, name="shaft1"),
        ShaftSpec(shaft=shaft2, bearings=bearings2, speed_rpm=725.0, name="shaft2"),
        ShaftSpec(shaft=shaft3, bearings=bearings3, speed_rpm=362.5,
                  loads=(sprocket_load,), name="shaft3"),
    ]
    stage_specs = [
        StageSpec(gear_driver=g1_driver, gear_driven=g2_driven, phi_deg=0.0, label="stage1"),
        StageSpec(gear_driver=g3_driver, gear_driven=g4_driven, phi_deg=90.0, label="stage2"),
    ]

    system = build_linear_system(
        shaft_specs, stage_specs, P=5000.0, rotation_dir_source=1, label="2stage_chain",
    )
    return construction, system


def main() -> None:
    construction, system = build_system()

    study = StudyCapabilities(
        construction=construction,
        shaft_fem=("shaft_fem.convergence.gears.timoshenko",),
    )
    objs = study.resolve()
    run_convergence = objs["run_convergence"]

    library = run_convergence(system, construction)

    expected_names = {ss.name for ss in system.shafts}
    got_names = set(library.names())
    missing = expected_names - got_names

    ok = not missing
    print(f"[{'OK' if ok else 'FAIL'}] convergence studied {len(library)}/{len(system.shafts)} "
          f"shafts -- {library!r}")
    if missing:
        print(f"    missing from library: {missing}")

    for ss in system.shafts:
        r = library.get_or_none(ss.name)
        if r is None:
            continue
        n_total = len(r.per_load)
        n_conv = sum(1 for rec in r.per_load.values() if rec.converged)
        labels = ", ".join(r.per_load.keys()) or "(none)"
        print(f"    {ss.name:8s} intervals=[{labels}]  converged={n_conv}/{n_total}")

    out_path = HERE / "report_2stage_chain_convergence_gears.txt"
    write_studies_report(
        system, out_path,
        title="2-stage linear chain -- Mesh convergence (gears, Timoshenko)",
        convergence_library=library,
    )
    print(f"[OK] {out_path.name} written ({len(system.shafts)} shafts, "
          f"from the real run_convergence() library above -- no fabricated data)")


if __name__ == "__main__":
    main()