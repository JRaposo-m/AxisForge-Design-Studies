"""
check_resolution_fem_compare.py

Exploratory script (not pytest). Builds the SAME 2-stage linear chain as
check_resolution_fem.py / check_resolution_fem_euler.py exactly once
(build_linear_system(), already resolved -- gear-mesh loads + shaft
positions present, one `system` object), then requests the
"shaft_fem.comparison" capability -- comparison_study.run_comparison()
solves THAT SAME `system` twice ("timoshenko" vs "euler"), and
print_comparison()/write_studies_report() report the difference.

This script no longer builds two StudyCapabilities or calls
solve_system() itself -- that duplicated comparison_study.py's own
run_comparison(), and comparison_report.py's own shaft_comparison_block()
table logic was duplicated a second time as an inline print loop here.
Both duplications are gone: this script only resolves ONE capability
("shaft_fem.comparison") and calls the three functions it returns.
Solving the same `system` object twice (rather than trusting two
separate scripts' printed numbers to correspond to "the same shaft")
is still the whole point structurally -- it just now lives inside
run_comparison() instead of in this script.

Compared per shaft: sigma_b_max [MPa], v_max [mm], and each bearing's
Fr/Fa [N] -- unchanged content, now produced by
comparison_report.comparison_table() instead of an inline loop. Bearing
reactions are expected to match closely between the two physics (they
come from static equilibrium on the same loads/positions, not from beam
theory); sigma_b_max and v_max are where Euler-Bernoulli's "no shear
flexibility" assumption is expected to show up as a difference, larger
the shorter/stubbier the span.

Console output stays terse (fail-loud/succeed-quiet, same convention as
the other two scripts): one block per shaft, absolute delta and delta%
for the two physics-dependent quantities, no full dump. Also writes the
Studies text report via write_studies_report(comparison=...) -- passing
BOTH libraries and their labels so the SHAFT_FEM COMPARISON section
appears in the same combined .txt shape the single-theory scripts
produce, rather than a separate comparison-only file.
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
        shaft_fem=("shaft_fem.comparison",),
    )
    objs = study.resolve()
    run_comparison = objs["run_comparison"]
    print_comparison = objs["print_comparison"]

    label_a, label_b = "timoshenko", "euler"
    library_a, library_b = run_comparison(system, construction, theory_a=label_a, theory_b=label_b)

    expected_names = {ss.name for ss in system.shafts}
    missing_a = expected_names - set(library_a.names())
    missing_b = expected_names - set(library_b.names())
    ok = not missing_a and not missing_b

    print(f"[{'OK' if ok else 'FAIL'}] {label_a} solved {len(library_a)}/"
          f"{len(system.shafts)} shafts, {label_b} solved "
          f"{len(library_b)}/{len(system.shafts)} shafts")
    if missing_a:
        print(f"    missing from {label_a} library: {missing_a}")
    if missing_b:
        print(f"    missing from {label_b} library: {missing_b}")
    print()

    print_comparison(library_a, library_b, system, label_a, label_b)

    print("Note: bearing reactions (Fr/Fa) come from static equilibrium on the "
          "same loads/positions in both physics and are expected to match "
          "closely -- sigma_b_max/v_max differences are where Euler-Bernoulli's "
          "no-shear-flexibility assumption is expected to show up, growing as "
          "spans get shorter/stubbier relative to their diameter.")

    out_path = HERE / "report_2stage_chain_comparison.txt"
    write_studies_report(
        system, out_path,
        title=f"2-stage linear chain -- Studies report ({label_a} vs {label_b})",
        comparison=(library_a, library_b, label_a, label_b),
    )
    print(f"\n[OK] {out_path.name} written ({len(system.shafts)} shafts, "
          f"from the real run_comparison() libraries above -- no fabricated data)")


if __name__ == "__main__":
    main()