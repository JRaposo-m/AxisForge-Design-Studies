"""
check_helical_gear_construction.py

Exploratory script (not pytest -- structured console output, run directly
with `python check_helical_gear_construction.py`). Builds several helical
gears through the capability pipeline (gears.helical), varying beta_n_deg,
profile shift, and other out-of-standard characteristics, and prints
everything back out so a human can eyeball whether it's sane.

Out-of-standard characteristics exercised here:
  - several distinct helix angles (small, typical, large)
  - profile shift x != 0 (both positive and negative)
  - a deliberately LOW z at a moderate helix angle to trigger the core's
    own undercutting check (z_min depends on beta too -- see
    SpurHelicalGear.undercutting())
  - a different material_id / Ra / face width b per gear

Also exercises the factory's own guard rail: beta_n_deg is REQUIRED and
must be > 0 -- calling with beta_n_deg=0.0, or omitting it entirely, must
raise (ValueError / TypeError respectively), not silently build what is
really a spur gear.

Usage:
    python check_helical_gear_construction.py
"""

from __future__ import annotations

from axisforge.fixtures.construction.construction_capabilities import ConstructionCapabilities


def main() -> None:
    construction = ConstructionCapabilities(gears=("gears.helical",))
    objs = construction.resolve()
    make_helical_gear = objs["make_helical_gear"]

    print("=" * 72)
    print("CONSTRUCTION CHECK -- gears.helical, out-of-standard gears")
    print("=" * 72)

    gears = [
        make_helical_gear(mn=2.0, z=20, beta_n_deg=8.0,  x=0.0, b=20.0,
                           material_id="42CrMo4", Ra=0.8, label="small_helix"),
        make_helical_gear(mn=2.0, z=20, beta_n_deg=15.0, x=0.2, b=20.0,
                           material_id="16MnCr5", Ra=1.6, label="typical_helix_pos_shift"),
        make_helical_gear(mn=2.0, z=20, beta_n_deg=25.0, x=-0.2, b=25.0,
                           material_id="S355", Ra=0.4, label="large_helix_neg_shift"),
        make_helical_gear(mn=2.0, z=8, beta_n_deg=20.0, x=0.0, b=15.0,
                           material_id="AISI_1045", Ra=0.8, label="deliberately_undercut"),
    ]

    print("-" * 72)
    print("Geometry")
    print("-" * 72)
    header = (f"{'label':<28} {'z':>4} {'mn':>6} {'beta':>7} {'x':>6} {'b':>6} "
              f"{'d':>9} {'da':>9} {'df':>9} {'db':>9}")
    print(header)
    for g in gears:
        print(f"{g.label:<28} {g.z:>4} {g.mn:>6.2f} {g.beta_n_deg:>7.2f} {g.x:>6.2f} {g.b:>6.1f} "
              f"{g.d:>9.4f} {g.da:>9.4f} {g.df:>9.4f} {g.db:>9.4f}")
    print()

    print("-" * 72)
    print("validate() per gear")
    print("-" * 72)
    for g in gears:
        errors = g.validate()
        print(f"{g.label}: {len(errors)} error(s)")
        for e in errors:
            print(f"  - {e}")
    print()

    print("-" * 72)
    print("Guard rail check -- beta_n_deg required and must be > 0")
    print("-" * 72)
    try:
        make_helical_gear(mn=2.0, z=20, beta_n_deg=0.0)
        print("  UNEXPECTED: call succeeded with beta_n_deg=0.0")
    except ValueError as e:
        print(f"  OK -- ValueError as expected: {e}")
    try:
        make_helical_gear(mn=2.0, z=20)
        print("  UNEXPECTED: call succeeded without beta_n_deg")
    except TypeError as e:
        print(f"  OK -- TypeError as expected (missing required arg): {e}")
    print()

    print("=" * 72)
    print("NOT covered by this script (fixture gap, not this gear's fault):")
    print("  - Ca/Cf (addendum/dedendum modification) exercised at 0.0 only")
    print("=" * 72)


if __name__ == "__main__":
    main()