"""
check_internal_gear_construction.py

Exploratory script (not pytest -- structured console output, run directly
with `python check_internal_gear_construction.py`). Builds several internal
(ring) gears through the capability pipeline (gears.internal), and prints
everything back out so a human can eyeball whether it's sane.

Out-of-standard characteristics exercised here:
  - profile shift x != 0 (both positive and negative)
  - a range of tooth counts and modules, all passed as NEGATIVE z (the
    convention this factory enforces -- see internal_gear_fixture.py's own
    docstring for why: only z<0 makes da/df come out on the geometrically
    correct side of d for this core's formulas)
  - a mesh-pair interference screen against a plausible mating external
    pinion, via the core's own validate_mesh() -- for VISIBILITY only, not
    a meshing-stage computation (no center distance, no load)

Also exercises the factory's own guard rail: z must be < 0 -- calling
with z >= 0 must raise ValueError.

KNOWN GAPS, NOT fixed here (core, out of scope for this Construction-only
session):
  - with a negative self.z, InternalGear.trimming_interference() /
    validate_mesh() compute `self.z - z1` directly (e.g. -60 - 20 = -80)
    instead of the tooth-count DIFFERENCE the ISO/KHK check actually means
    (|z2| - z1 = 40). That makes trimming_interference() report a false
    positive for every mesh pair once z is negative -- this script's own
    output below shows it happening on every single row.
  - the "fewer_teeth_finer_module" row below is a SECOND deliberately
    invalid gear (mn=1.5, z=-40 puts the tip circle at/below the base
    circle -- validate() correctly flags it), included specifically to
    exercise that check too, not a copy/paste mistake. Because that
    geometry is already invalid, running validate_mesh() against it
    afterward hits `arccos(db/da)` with a ratio > 1 and prints a
    RuntimeWarning (NaN) -- involute_interference() doesn't gate on
    validate() first. Also flagged, also not fixed here.
  Both are validate_mesh()/interference-helper concerns, i.e. meshing-
  stage, out of scope here.

Usage:
    python check_internal_gear_construction.py
"""

from __future__ import annotations

from axisforge.fixtures.capabilities import ConstructionCapabilities


def main() -> None:
    construction = ConstructionCapabilities(gears=("gears.internal",))
    objs = construction.resolve()
    make_internal_gear = objs["make_internal_gear"]

    print("=" * 72)
    print("CONSTRUCTION CHECK -- gears.internal, out-of-standard gears")
    print("=" * 72)

    gears = [
        make_internal_gear(mn=2.0, z=-60, x=0.0,  b=20.0,
                            material_id="42CrMo4", Ra=0.8, label="baseline"),
        make_internal_gear(mn=2.0, z=-60, x=0.3,  b=20.0,
                            material_id="16MnCr5", Ra=1.6, label="positive_shift"),
        make_internal_gear(mn=2.0, z=-80, x=-0.2, b=25.0,
                            material_id="S355", Ra=0.4, label="more_teeth_neg_shift"),
        make_internal_gear(mn=1.5, z=-40, x=0.0,  b=15.0,
                            material_id="AISI_1045", Ra=0.8, label="fewer_teeth_finer_module"),
    ]

    print("-" * 72)
    print("Geometry")
    print("-" * 72)
    header = (f"{'label':<26} {'z':>5} {'mn':>6} {'x':>6} {'b':>6} "
              f"{'d':>9} {'da':>9} {'df':>9} {'db':>9}")
    print(header)
    for g in gears:
        print(f"{g.label:<26} {g.z:>5} {g.mn:>6.2f} {g.x:>6.2f} {g.b:>6.1f} "
              f"{g.d:>9.4f} {g.da:>9.4f} {g.df:>9.4f} {g.db:>9.4f}")
    print()

    print("-" * 72)
    print("validate() per gear -- da must come out BELOW d for every row above")
    print("-" * 72)
    for g in gears:
        errors = g.validate()
        print(f"{g.label}: {len(errors)} error(s)  (da<d: {g.da < g.d})")
        for e in errors:
            print(f"  - {e}")
    print()

    print("-" * 72)
    print("Mesh-pair interference vs. a plausible mating pinion (z1=20, x1=0.0)")
    print("(visibility only -- meshing is out of scope for Construction; see")
    print(" this script's own module docstring for a known core gap this")
    print(" exposes)")
    print("-" * 72)
    for g in gears:
        errs = g.validate_mesh(z1=20, x1=0.0)
        print(f"{g.label}: {len(errs)} interference flag(s)")
        for e in errs:
            print(f"  - {e}")
    print()

    print("-" * 72)
    print("Guard rail check -- z must be < 0")
    print("-" * 72)
    try:
        make_internal_gear(mn=2.0, z=60)
        print("  UNEXPECTED: call succeeded with z=60 (positive)")
    except ValueError as e:
        print(f"  OK -- ValueError as expected: {e}")
    try:
        make_internal_gear(mn=2.0, z=0)
        print("  UNEXPECTED: call succeeded with z=0")
    except ValueError as e:
        print(f"  OK -- ValueError as expected: {e}")
    print()

    print("=" * 72)
    print("KNOWN GAP (core, not fixed here): trimming_interference()/")
    print("validate_mesh() compute self.z - z1 directly, which is wrong once")
    print("self.z is negative (the required convention here) -- see this")
    print("script's own module docstring for the full explanation.")
    print("=" * 72)


if __name__ == "__main__":
    main()