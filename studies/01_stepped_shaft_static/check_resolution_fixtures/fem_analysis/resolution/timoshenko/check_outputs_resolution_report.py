"""
check_resolution_fem.py

Exploratory script (not pytest). Builds the same 2-stage linear chain
as check_linear_system_construction.py / check_outputs_text_report.py
(already-resolved via build_linear_system(), gear-mesh loads + shaft
positions present), then runs it through the Studies stage:
StudyCapabilities -> solve_system() -> RigidBearingFEMResultsLibrary.

build_system() now returns (construction, system) -- solve_system()
takes both: `construction` (the ConstructionCapabilities that built
`system`) is checked via construction.has_capability(
"systems.parallel_axis_linear") to confirm Resolution's prerequisite --
Construction must have been asked to produce an already-resolved
system -- before the FEM solve runs at all. Also exercises the guard
rail: a ConstructionCapabilities() that never requested
"systems.parallel_axis_linear" makes solve_system() raise ValueError,
even when handed the same, perfectly-resolved `system`.

Console output stays terse (fail-loud/succeed-quiet, same convention
as check_outputs_text_report.py): one summary line per shaft, the
handful of numbers worth eyeballing (max bending stress, max
deflection, bearing reactions), not a full dump.

Also writes the Studies text report via
fixtures.studies.text_report.write_studies_report() -- the Studies-wide
aggregator (mirrors write_construction_report()'s own role), passed
`shaft_fem_library=library` and nothing for `comparison=` since this
script only ran ONE theory. From the REAL `library` this script's own
solve_system() call produces -- no stand-in/fabricated ShaftResults
anywhere in this script. An earlier version of this check
(check_outputs_resolution_report.py, now removed) tested the report's
table/block formatting against hand-built stub ShaftResults objects;
that produced numbers that looked like real engineering results but
were not (arbitrary ramps, made-up bearing loads), which read as
confusing/misleading next to the genuinely solver-derived numbers
already used elsewhere in this check -- dropped, on that explicit
feedback, in favour of running the actual solve here and reporting on
ITS output only. write_studies_report() replaced the old
write_resolution_report() (fixtures.studies.text_report used to hold
both the shaft_fem content blocks and the writer; it is now the
Studies-wide aggregator only -- see fem_studies/outputs/
resolution_report.py's own docstring for that split's full history)
without changing any of this reasoning.

Also writes the per-shaft PNG figure set via
fem_studies.outputs.plots.write_resolution_plots() -- one PNG per
physical quantity per shaft (bending moment, shear, deflection,
torsion, section geometry/stress, bearing reactions -- see that
module's own top docstring for the full list and why it is one chart
per figure, not multi-panel composites), written under
HERE/plots/<shaft_name>/. Same "from the real library, no fabricated
data" discipline as the text report above -- write_resolution_plots()
reads the SAME `library` write_studies_report() just wrote from, not a
second solve or a stand-in.
"""
from __future__ import annotations

from pathlib import Path

from axisforge.core.loads import RadialLoad
from axisforge.fixtures.construction.construction_capabilities import ConstructionCapabilities
from axisforge.fixtures.studies.study_capabilities import StudyCapabilities
from axisforge.fixtures.studies.outputs.text_report import write_studies_report
from axisforge.fixtures.studies.shafts.fem_studies.outputs.plots import write_resolution_plots

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
        shaft_fem=("shaft_fem.timoshenko_rigid",),
    )
    objs = study.resolve()
    solve_system = objs["solve_system"]

    print(f"  construction.has_capability('systems.parallel_axis_linear'): "
          f"{construction.has_capability('systems.parallel_axis_linear')} (expect True)")

    # Guard rail -- solve_system() refuses a construction that never
    # requested 'systems.parallel_axis_linear'.
    bad_construction = ConstructionCapabilities()
    try:
        solve_system(system, bad_construction)
        print("  UNEXPECTED: solve_system() accepted a construction without "
              "'systems.parallel_axis_linear'")
    except ValueError as e:
        print(f"  OK -- solve_system() refused: {e}")

    library = solve_system(system, construction)

    expected_names = {ss.name for ss in system.shafts}
    got_names = set(library.names())
    missing = expected_names - got_names

    ok = not missing
    print(f"[{'OK' if ok else 'FAIL'}] solved {len(library)}/{len(system.shafts)} shafts "
          f"-- {library!r}")
    if missing:
        print(f"    missing from library: {missing}")

    for ss in system.shafts:
        r = library.get_or_none(ss.name)
        if r is None:
            continue
        n = len(r.x_nodes)
        brg_str = ", ".join(
            f"{b.label}: Fr={b.Fr:.1f}N Fa={b.Fa:.1f}N" for b in r.bearing_nodes
        )
        print(f"    {ss.name:8s} nodes={n:3d}  "
              f"sigma_b_max={r.sigma_b_max:8.2f} MPa @ x={r.x_sigma_b_max:6.1f} mm  "
              f"v_max={r.v_max:7.4f} mm @ x={r.x_v_max:6.1f} mm  [{brg_str}]")

    out_path = HERE / "report_2stage_chain_resolution.txt"
    write_studies_report(
        system, out_path,
        title="2-stage linear chain -- Studies report (Timoshenko)",
        shaft_fem_library=library,
    )
    print(f"[OK] {out_path.name} written ({len(system.shafts)} shafts, "
          f"from the real solve_system() library above -- no fabricated data)")

    written = write_resolution_plots(library, system, HERE)
    n_files = sum(len(paths) for paths in written.values())
    print(f"[OK] {n_files} plot(s) written under {(HERE / 'plots').name}/ "
          f"for {len(written)}/{len(system.shafts)} shaft(s) "
          f"-- from the same library above, no second solve")


if __name__ == "__main__":
    main()