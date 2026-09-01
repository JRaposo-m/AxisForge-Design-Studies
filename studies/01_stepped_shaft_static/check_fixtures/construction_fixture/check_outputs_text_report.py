"""
check_outputs_text_report.py

Exploratory script (not pytest). Builds the same 2-stage linear chain
as check_linear_system_construction.py, writes its report via
fixtures.outputs.construction.text_report.write_construction_report(),
and does a quick sanity check on the result.

ONE combined .txt file for the whole Construction domain -- topology,
then per shaft its geometry/bearings/gears/ShaftSystem summary/loads,
then meshes -- via write_construction_report(). Two other things
carried over from earlier rounds of feedback:
  - the report is written NEXT TO THIS SCRIPT (Path(__file__).resolve()
    .parent), never into whatever the process's current working
    directory happens to be.
  - console output is deliberately short: one summary line plus the
    handful of numbers worth eyeballing (T_out per shaft, validate()
    error count), not a full dump of report text or a line-by-line
    token checklist. Full detail lives in the .txt file only.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from axisforge.core.loads import RadialLoad
from axisforge.fixtures.capabilities import ConstructionCapabilities
from axisforge.fixtures.outputs.construction.text_report import (
    write_construction_report,
)

HERE = Path(__file__).resolve().parent


def build_system():
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
        StageSpec(gear_driver=g3_driver, gear_driven=g4_driven, phi_deg=0.0, label="stage2"),
    ]

    return build_linear_system(
        shaft_specs, stage_specs, P=5000.0, rotation_dir_source=1, label="2stage_chain",
    )


def main() -> None:
    system = build_system()

    out_path = HERE / "report_2stage_chain.txt"
    text = write_construction_report(
        system, out_path, title="2-stage linear chain -- Construction report",
    )

    expected_tokens = [
        "shaft1", "shaft2", "shaft3",
        "shaft1_brg_A", "shaft1_brg_B", "shaft2_brg_A", "shaft3_brg_B",
        "g1_driver", "g2_driven", "g3_driver", "g4_driven",
        "stage1", "stage2", "sprocket_pull", "[user]", "[gear_mesh]",
    ]
    missing = [tok for tok in expected_tokens if tok not in text]

    # Determinism check writes its throwaway copy to a temp dir -- never
    # into the working folder, which should only ever hold the real report.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / "report_check.txt"
        write_construction_report(
            system, tmp_path, title="2-stage linear chain -- Construction report",
        )
        deterministic = tmp_path.read_text(encoding="utf-8") == text

    validate_errors = system.validate()
    t_out_lines = [ln.strip() for ln in text.splitlines() if "T_out=" in ln]

    ok = not missing and deterministic and not validate_errors
    print(f"[{'OK' if ok else 'FAIL'}] {out_path.name}  "
          f"({len(text)} chars, {len(system.shafts)} shafts, {len(system.links)} links)")
    for ln in t_out_lines:
        print(f"    {ln}")
    if validate_errors:
        print(f"    validate(): {len(validate_errors)} error(s) -- {validate_errors}")
    if missing:
        print(f"    missing tokens: {missing}")
    if not deterministic:
        print("    NOT deterministic across repeated writes")


if __name__ == "__main__":
    main()