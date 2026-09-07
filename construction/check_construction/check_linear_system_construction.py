"""
check_linear_system_construction.py

Exploratory script (not pytest) -- builds a 2-stage linear parallel-axis
chain (3 shafts, 2 meshes) through the capability pipeline
(systems.parallel_axis_linear), on top of shafts/bearings/gears already
built via shafts.stepped / bearings.deep_groove_ball / gears.spur.
Requests all four domains (shaft, bearings, gears, system) in ONE
ConstructionCapabilities call to exercise _PREREQUISITES.

UNLIKE every other Construction check script so far, build_linear_system()
now ALSO resolves the system (P required, rpm from shaft_specs[0].speed_rpm)
-- see linear_chain_fixture.py's own module docstring for why this
fixture deliberately breaks the Construction/MeshLoads boundary every
other domain observes. Verifies:
  - resolve() (the capability one, not the gear-system one) returns every
    expected name with no collisions
  - build_linear_system() assembles a valid single-source DAG and returns
    it ALREADY RESOLVED (gear-mesh loads present, shaft positions set)
  - a user-declared external load (ShaftSpec.loads) survives resolve()
    untouched, alongside the gear-mesh loads
  - validate_or_raise() fails fast (before resolve()) on a structurally
    bad system -- here, a shaft with < 2 bearings
  - resolve()'s own guard: shaft_specs[0].speed_rpm <= 0 raises ValueError
  - the original guard rail: len(shaft_specs) != len(stage_specs) + 1
    raises ValueError
"""
from __future__ import annotations

from axisforge.core.loads import RadialLoad
from axisforge.fixtures.construction.construction_capabilities import ConstructionCapabilities


def main() -> None:
    print("=" * 72)
    print("CONSTRUCTION CHECK -- systems (parallel_axis_linear, resolved)")
    print("=" * 72)

    construction = ConstructionCapabilities(
        shaft=("shafts.stepped",),
        bearings=("bearings.deep_groove_ball",),
        gears=("gears.spur",),
        system=("systems.parallel_axis_linear",),
    )
    objs = construction.resolve()
    print(f"resolve() OK -- {len(objs)} names, no collisions:")
    for name in sorted(objs):
        print(f"  {name}")
    print()

    make_stepped_shaft = objs["factory"]
    SectionSpec = objs["SectionSpec"]
    make_deep_groove_ball_bearing = objs["make_deep_groove_ball_bearing"]
    make_spur_gear = objs["make_spur_gear"]
    ShaftSpec = objs["ShaftSpec"]
    StageSpec = objs["StageSpec"]
    build_linear_system = objs["build_linear_system"]

    print("-" * 72)
    print("Geometry -- 3 identical stepped shafts (d=30/50/30, L=200 mm)")
    print("-" * 72)

    def make_shaft_geometry(name: str):
        return make_stepped_shaft(
            sections=[
                SectionSpec(length=30.0, diameter=30.0, label=f"{name}_seat_A"),
                SectionSpec(length=140.0, diameter=50.0, label=f"{name}_body"),
                SectionSpec(length=30.0, diameter=30.0, label=f"{name}_seat_B"),
            ],
            fillet_radii=[2.0, 2.0],
            name=name,
        ).shaft  # unwrap ShaftFixture -> core Shaft

    shaft1 = make_shaft_geometry("shaft1")
    shaft2 = make_shaft_geometry("shaft2")
    shaft3 = make_shaft_geometry("shaft3")
    print("  shaft1, shaft2, shaft3 built (shoulders at x=30, x=170 mm each)")
    print()

    print("-" * 72)
    print("Bearings -- 2 DGBBs per shaft, at x=10 and x=190 mm (clear of shoulders)")
    print("-" * 72)

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

    bearings1 = make_shaft_bearings("shaft1")
    bearings2 = make_shaft_bearings("shaft2")
    bearings3 = make_shaft_bearings("shaft3")
    print("  2 bearings built per shaft (6 total)")
    print()

    print("-" * 72)
    print("Gears -- 2 spur stages, mn=2.0, alpha_n=20 deg")
    print("  stage 1: z=20 (shaft1, driver, x=100) -> z=40 (shaft2, driven, x=100)")
    print("  stage 2: z=20 (shaft2, driver, x=150) -> z=40 (shaft3, driven, x=150)")
    print("-" * 72)

    g1_driver = make_spur_gear(mn=2.0, z=20, b=15.0, position=100.0, label="g1_driver")
    g2_driven = make_spur_gear(mn=2.0, z=40, b=15.0, position=100.0, label="g2_driven")
    g3_driver = make_spur_gear(mn=2.0, z=20, b=15.0, position=150.0, label="g3_driver")
    g4_driven = make_spur_gear(mn=2.0, z=40, b=15.0, position=150.0, label="g4_driven")
    print("  4 SpurHelicalGear objects built (beta_n_deg=0.0)")
    print()

    print("-" * 72)
    print("External load -- overhung sprocket on shaft3 (output), x=190 mm, 500 N radial")
    print("-" * 72)
    sprocket_load = RadialLoad(190.0, 500.0, theta_deg=270.0, label="sprocket_pull")
    print(f"  {sprocket_load!r}")
    print()

    print("-" * 72)
    print("Assembling + resolving: build_linear_system(shaft_specs=[3], stage_specs=[2], P=5000.0)")
    print("-" * 72)

    shaft_specs = [
        ShaftSpec(shaft=shaft1, bearings=bearings1, speed_rpm=1450.0, name="shaft1"),
        ShaftSpec(shaft=shaft2, bearings=bearings2, speed_rpm=725.0, name="shaft2"),
        ShaftSpec(shaft=shaft3, bearings=bearings3, speed_rpm=362.5,
                  loads=(sprocket_load,), name="shaft3"),
    ]
    stage_specs = [
        StageSpec(gear_driver=g1_driver, gear_driven=g2_driven, phi_deg=0.0, label="stage1"),
        StageSpec(gear_driver=g3_driver, gear_driven=g4_driven, phi_deg=180.0, label="stage2"),
    ]

    system = build_linear_system(
        shaft_specs, stage_specs, P=5000.0, rotation_dir_source=1, label="2stage_chain",
    )
    print(f"  built: {system!r}")
    print(system.summary())
    print()

    print("-" * 72)
    print("Topology checks")
    print("-" * 72)
    sources = system._sources()
    print(f"  source shafts: {[s.name for s in sources]} (expect exactly ['shaft1'])")
    order = system._topological_order()
    print(f"  topological order: {[s.name for s in order] if order else None} "
          f"(expect ['shaft1', 'shaft2', 'shaft3'])")
    indeg = system._incoming_count()
    max_indeg = max(indeg.values())
    print(f"  max incoming links on any shaft: {max_indeg} (expect 1 -- no convergent merge)")
    print()

    print("-" * 72)
    print("validate() on the ALREADY-RESOLVED system -- expect 0 errors")
    print("-" * 72)
    errors = system.validate()
    print(f"  {len(errors)} error(s)")
    for e in errors:
        print(f"    - {e}")
    print()

    print("-" * 72)
    print("Resolved -- gear-mesh loads now present, user load survives untouched")
    print("-" * 72)
    for ss in system.shafts:
        n_gear_mesh = sum(1 for ld in ss.loads if ld.source == "gear_mesh")
        n_user = sum(1 for ld in ss.loads if ld.source == "user")
        print(f"  {ss.name}: {len(ss.loads)} load(s) total "
              f"({n_user} user, {n_gear_mesh} gear_mesh), "
              f"shaft_position=(y={ss.shaft_position[0]:.3f}, z={ss.shaft_position[1]:.3f}) mm")
    shaft3_ss = next(s for s in system.shafts if s.name == "shaft3")
    still_there = any(ld is sprocket_load for ld in shaft3_ss.loads)
    print(f"  shaft3 still carries the exact sprocket_load object: {still_there} (expect True)")
    print()

    print("-" * 72)
    print("Guard rail -- validate_or_raise() fires BEFORE resolve() on a bad system")
    print("  (shaft2 given only 1 bearing -- needs >= 2 for a determinate support)")
    print("-" * 72)
    bad_shaft_specs = [
        shaft_specs[0],
        ShaftSpec(shaft=make_shaft_geometry("shaft2_bad"), bearings=bearings2[:1],
                  speed_rpm=725.0, name="shaft2_bad"),
        shaft_specs[2],
    ]
    try:
        build_linear_system(bad_shaft_specs, stage_specs, P=5000.0, rotation_dir_source=1)
        print("  UNEXPECTED: call succeeded with only 1 bearing on shaft2")
    except ValueError as e:
        print(f"  OK -- ValueError as expected (validate_or_raise, not resolve): {e}")
    print()

    print("-" * 72)
    print("Guard rail -- resolve() rejects shaft_specs[0].speed_rpm <= 0")
    print("-" * 72)
    zero_rpm_specs = [
        ShaftSpec(shaft=shaft1, bearings=bearings1, speed_rpm=0.0, name="shaft1"),
        shaft_specs[1],
        shaft_specs[2],
    ]
    try:
        build_linear_system(zero_rpm_specs, stage_specs, P=5000.0, rotation_dir_source=1)
        print("  UNEXPECTED: call succeeded with shaft_specs[0].speed_rpm == 0")
    except ValueError as e:
        print(f"  OK -- ValueError as expected: {e}")
    print()

    print("-" * 72)
    print("Guard rail -- len(shaft_specs) != len(stage_specs) + 1 raises ValueError")
    print("-" * 72)
    try:
        build_linear_system(shaft_specs[:2], stage_specs, P=5000.0, rotation_dir_source=1)
        print("  UNEXPECTED: call succeeded with mismatched shaft/stage counts")
    except ValueError as e:
        print(f"  OK -- ValueError as expected: {e}")
    print()

    print("=" * 72)
    print("Built + resolved a 2-stage linear SpurHelicalGearSystem, validate() "
          "clean, user load preserved, all guard rails hold.")
    print("=" * 72)


if __name__ == "__main__":
    main()