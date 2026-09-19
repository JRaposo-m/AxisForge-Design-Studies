"""
check_spur_gear_construction.py

Exploratory script (not pytest -- structured console output, run directly
with `python check_spur_gear_construction.py`). Builds several spur gears
through the capability pipeline (gears.spur), deliberately varying every
out-of-standard characteristic the current Construction-stage gear fixture
supports, and prints everything back out so a human can eyeball whether
it's sane.

Out-of-standard characteristics exercised here:
  - profile shift x != 0 (both positive and negative)
  - non-default alpha_n_deg (14.5deg, an older standard pressure angle)
  - a deliberately LOW z with x=0 to trigger the core's own undercutting
    check (validate() should flag it, not silently build a bad gear)
  - a different material_id / Ra / face width b per gear

Also exercises the factory's own guard rail: beta_n_deg is not a
parameter of make_spur_gear() at all -- passing it raises a plain
TypeError from Python's own argument binding, not a custom check. This
script confirms that stays true.

Usage:
    python check_spur_gear_construction.py
"""

from __future__ import annotations

from axisforge.fixtures.construction.construction_capabilities import ConstructionCapabilities


def main() -> None:
    construction = ConstructionCapabilities(gears=("gears.spur",))
    objs = construction.resolve()
    make_spur_gear = objs["make_spur_gear"]

    print("=" * 72)
    print("CONSTRUCTION CHECK -- gears.spur, out-of-standard gears")
    print("=" * 72)

    gears = [
        make_spur_gear(mn=2.0, z=20, x=0.3, b=18.0, alpha_n_deg=20.0,
                        material_id="42CrMo4", Ra=0.8, label="positive_shift"),
        make_spur_gear(mn=2.0, z=20, x=-0.3, b=18.0, alpha_n_deg=20.0,
                        material_id="16MnCr5", Ra=1.6, label="negative_shift"),
        make_spur_gear(mn=1.5, z=16, x=0.0, b=12.0, alpha_n_deg=14.5,
                        material_id="AISI_1045", Ra=0.4, label="old_std_pressure_angle"),
        make_spur_gear(mn=2.0, z=8, x=0.0, b=15.0, alpha_n_deg=20.0,
                        material_id="S355", Ra=0.8, label="deliberately_undercut"),
    ]

    print("-" * 72)
    print("Geometry")
    print("-" * 72)
    header = (f"{'label':<24} {'z':>4} {'mn':>6} {'x':>6} {'b':>6} "
              f"{'d':>9} {'da':>9} {'df':>9} {'db':>9}")
    print(header)
    for g in gears:
        print(f"{g.label:<24} {g.z:>4} {g.mn:>6.2f} {g.x:>6.2f} {g.b:>6.1f} "
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
    print("Guard rail check -- beta_n_deg is not a parameter of make_spur_gear()")
    print("-" * 72)
    try:
        make_spur_gear(mn=2.0, z=20, beta_n_deg=10.0)
        print("  UNEXPECTED: call succeeded, beta_n_deg was silently accepted")
    except TypeError as e:
        print(f"  OK -- TypeError as expected: {e}")
    print()

    print("=" * 72)
    print("NOT covered by this script (fixture gap, not this gear's fault):")
    print("  - Ca/Cf (addendum/dedendum modification) exercised at 0.0 only")
    print("=" * 72)


if __name__ == "__main__":
    main()