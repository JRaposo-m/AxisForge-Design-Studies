"""
check_resolution_fem_euler.py

Exploratory script (not pytest). Mirrors check_resolution_fem.py exactly
-- same 2-stage linear chain (build_linear_system(), already resolved,
gear-mesh loads + shaft positions present) -- but drives the Studies
stage through "shaft_fem.euler_bernoulli_rigid" instead of
"shaft_fem.timoshenko_rigid": StudyCapabilities -> solve_system() ->
RigidBearingFEMResultsLibrary, backed by RigidBearingFEMSolver's own
theory="euler" dispatch (StiffnessMatrixBuilder -> EulerBernoulliBeam:
cubic Hermite shape functions, closed-form 6x6 element stiffness, no
independent shear-strain field) rather than theory="timoshenko".

build_system() is byte-for-byte the same construction as
check_resolution_fem.py's own -- duplicated here rather than imported,
same convention that script itself follows relative to
check_linear_system_construction.py -- so the two Resolution checks
(Timoshenko vs Euler-Bernoulli) can be read and run independently, and
so check_resolution_fem_compare.py can solve the SAME system through
both physics in one process rather than trusting two separate scripts'
output to line up.

Same guard rail as check_resolution_fem.py: a ConstructionCapabilities()
that never requested "systems.parallel_axis_linear" makes solve_system()
raise ValueError even when handed the same, perfectly-resolved `system`
-- the check is physics-agnostic, so it is re-run here unchanged rather
than assumed already covered by the Timoshenko script.

Console output stays terse (fail-loud/succeed-quiet, same convention as
check_resolution_fem.py): one summary line per shaft, the handful of
numbers worth eyeballing (max bending stress, max deflection, bearing
reactions), not a full dump. Also writes the Studies text report via
fixtures.studies.text_report.write_studies_report() -- the Studies-wide
aggregator, passed `shaft_fem_library=library` and nothing for
`comparison=` since this script only ran ONE theory -- from the REAL
`library` this script's own solve_system() call produces -- no
stand-in/fabricated ShaftResults, same reasoning as
check_resolution_fem.py's own docstring.

Also writes the per-shaft PNG figure set via
fem_studies.outputs.plots.write_resolution_plots() -- identical call to
check_resolution_fem.py's own, same module, same
HERE/plots/<shaft_name>/ layout, same "from the real library, no
fabricated data" discipline; the only difference is this script's
`library` was solved with theory="euler" rather than "timoshenko".
This script already lives in its own directory, separate from
check_resolution_fem.py's, so there is no HERE/plots/ collision between
the two Resolution checks to design around -- unlike the two text
reports, which DO sit in the same directory and so keep their own
distinct filenames (report_..._euler.txt vs report_....txt).
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
        shaft_fem=("shaft_fem.euler_bernoulli_rigid",),
    )
    objs = study.resolve()
    solve_system = objs["solve_system"]

    print(f"  construction.has_capability('systems.parallel_axis_linear'): "
          f"{construction.has_capability('systems.parallel_axis_linear')} (expect True)")

    # Guard rail -- solve_system() refuses a construction that never
    # requested 'systems.parallel_axis_linear'. Physics-agnostic check,
    # re-run here (not assumed covered by check_resolution_fem.py).
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
          f"(Euler-Bernoulli) -- {library!r}")
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

    out_path = HERE / "report_2stage_chain_resolution_euler.txt"
    write_studies_report(
        system, out_path,
        title="2-stage linear chain -- Studies report (Euler-Bernoulli)",
        shaft_fem_library=library,
    )
    print(f"[OK] {out_path.name} written ({len(system.shafts)} shafts, "
          f"from the real solve_system() library above -- no fabricated data)")

    written = write_resolution_plots(library, system, HERE)
    n_files = sum(len(paths) for paths in written.values())
    print(f"[OK] {n_files} plot(s) written under {(HERE / 'plots').name}/ "
          f"for {len(written)}/{len(system.shafts)} shaft(s) (Euler-Bernoulli) "
          f"-- from the same library above, no second solve")


if __name__ == "__main__":
    main()