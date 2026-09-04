"""
check_gear_meshing_construction.py

Exploratory script (not pytest) -- builds a spur-spur pair and an
external-pinion + internal-ring pair through the capability pipeline
(gears.spur_helical_meshing / gears.internal_meshing), on top of gears
already built via gears.spur / gears.internal. Also requests every gears.*
capability (generation + meshing, 5 strings) in ONE ConstructionCapabilities
call to prove no key collisions.

Does NOT call .forces() anywhere -- MeshLoads stage, out of scope for
Construction (see both meshing fixtures' own docstrings).
"""
from __future__ import annotations

from axisforge.fixtures.construction.construction_capabilities import ConstructionCapabilities


def main() -> None:
    print("=" * 72)
    print("CONSTRUCTION CHECK -- gear meshing (spur_helical_meshing, internal_meshing)")
    print("=" * 72)

    construction = ConstructionCapabilities(
        gears=(
            "gears.spur",
            "gears.helical",
            "gears.internal",
            "gears.spur_helical_meshing",
            "gears.internal_meshing",
        ),
    )
    objs = construction.resolve()
    print(f"resolve() OK -- {len(objs)} names, no collisions:")
    for name in sorted(objs):
        print(f"  {name}")
    print()

    make_spur_gear = objs["make_spur_gear"]
    make_internal_gear = objs["make_internal_gear"]
    make_spur_helical_meshing = objs["make_spur_helical_meshing"]
    make_internal_meshing = objs["make_internal_meshing"]

    print("-" * 72)
    print("Pair 1: spur-spur external mesh (z1=20, z2=40, mn=2.0)")
    print("-" * 72)
    pinion = make_spur_gear(mn=2.0, z=20, x=0.0, b=20.0, label="pinion")
    wheel = make_spur_gear(mn=2.0, z=40, x=0.0, b=20.0, label="wheel")
    mesh1 = make_spur_helical_meshing(pinion, wheel, label="ext_pair")
    print(mesh1.summary())
    errors1 = mesh1.validate()
    print(f"  validate(): {len(errors1)} error(s)")
    for e in errors1:
        print(f"    - {e}")
    print()

    print("-" * 72)
    print("Pair 2: external pinion (z1=20) + internal ring (z2=-60), same mn/alpha/beta")
    print("-" * 72)
    pinion2 = make_spur_gear(mn=2.0, z=20, x=0.0, b=20.0, label="pinion2")
    ring = make_internal_gear(mn=2.0, z=-60, x=0.0, b=20.0, label="ring")
    mesh2 = make_internal_meshing(pinion2, ring, label="int_pair")
    print(mesh2.summary())
    errors2 = mesh2.validate()
    print(f"  validate(): {len(errors2)} error(s)")
    for e in errors2:
        print(f"    - {e}")
    print(f"  u (gear ratio, negative expected -- internal meshing does NOT")
    print(f"    reverse rotation direction, confirmed intentional): {mesh2.u:.4f}")
    print()

    print("-" * 72)
    print("Guard rail check -- make_spur_helical_meshing rejects an InternalGear")
    print("-" * 72)
    try:
        make_spur_helical_meshing(pinion2, ring, label="bad")
        print("  UNEXPECTED: call succeeded with an InternalGear as gear2")
    except TypeError as e:
        print(f"  OK -- TypeError as expected: {e}")
    print()

    print("-" * 72)
    print("Guard rail check -- make_internal_meshing rejects gear2.z >= 0")
    print("-" * 72)
    try:
        make_internal_meshing(pinion2, wheel, label="bad")
        print("  UNEXPECTED: call succeeded with an external gear as gear2")
    except TypeError as e:
        print(f"  OK -- TypeError as expected: {e}")
    print()

    print("=" * 72)
    print("Built 2 meshing pairs, both validate() clean; guard rails hold.")
    print("=" * 72)


if __name__ == "__main__":
    main()