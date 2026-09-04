"""
check_shaft_construction.py

Exploratory script (not pytest -- structured console output, run
directly with `python check_shaft_construction.py`). Builds ONE shaft
that deliberately exercises every out-of-standard characteristic the
current Construction-stage shaft fixture supports, through the
capability pipeline, and prints everything back out so a human can
eyeball whether it's sane.

Out-of-standard characteristics exercised here:
  - N = 6 sections (nothing in the fixture assumes 2 or 3)
  - non-monotonic diameter profile: up, down, up, down, up
    (proves shoulder derivation handles a step DOWN as well as UP)
  - a different fillet radius at every one of the 5 internal transitions
  - a different material_id on every section (not all AISI_1045)
  - a different surface_ra on every section (not all the 0.8 default)

NOT exercised here -- current gap, not a limitation of this script:
  SectionSpec has no field for inner_diameter (hollow sections) or
  keyways, even though ShaftSection (core) supports both. Building
  either of those means bypassing the fixture and constructing
  ShaftSection directly. Flagged at the end of this script's output,
  not silently skipped.

Usage:
    python check_shaft_construction.py
"""

from __future__ import annotations

from axisforge.fixtures.construction.construction_capabilities import ConstructionCapabilities


def main() -> None:
    construction = ConstructionCapabilities(shaft=("shafts.stepped",))
    objs = construction.resolve()
    SectionSpec        = objs["SectionSpec"]
    make_stepped_shaft = objs["factory"]

    # Non-monotonic profile: 30 -up-> 55 -down-> 35 -up-> 60 -down-> 45 -up-> 50
    sections = [
        SectionSpec(length=20.0, diameter=30.0, material_id="AISI_1045",
                    surface_ra=0.4, label="seat_A"),
        SectionSpec(length=45.0, diameter=55.0, material_id="AISI_4140",
                    surface_ra=0.8, label="body_1"),
        SectionSpec(length=15.0, diameter=35.0, material_id="S355",
                    surface_ra=1.6, label="collar"),
        SectionSpec(length=70.0, diameter=60.0, material_id="AISI_4140",
                    surface_ra=0.8, label="body_2"),
        SectionSpec(length=10.0, diameter=45.0, material_id="S355",
                    surface_ra=0.4, label="groove"),
        SectionSpec(length=20.0, diameter=50.0, material_id="AISI_1045",
                    surface_ra=0.8, label="seat_B"),
    ]
    fillet_radii = [3.0, 2.0, 4.0, 2.5, 1.5]   # one per internal transition, all different

    fixture = make_stepped_shaft(sections=sections, fillet_radii=fillet_radii,
                                  name="OFFSTD_CHECK")

    print("=" * 72)
    print("CONSTRUCTION CHECK -- shafts.stepped, out-of-standard shaft")
    print("=" * 72)

    errors = fixture.validate()
    print(f"validate() errors : {len(errors)}")
    for e in errors:
        print(f"  - {e}")
    print()

    print(fixture.summary())
    print()

    print("-" * 72)
    print("Per-section engineering properties (from the core ShaftSection)")
    print("-" * 72)
    header = f"{'i':>2} {'label':<10} {'d[mm]':>8} {'L[mm]':>8} {'A[mm2]':>10} {'I[mm4]':>12} {'J[mm4]':>12} {'W[mm3]':>10} {'Wt[mm3]':>10}"
    print(header)
    for i, sec in enumerate(fixture.shaft.sections):
        print(
            f"{i:>2} {sec.label:<10} {sec.diameter:>8.2f} {sec.length:>8.2f} "
            f"{sec.area:>10.2f} {sec.second_moment_of_area:>12.2f} "
            f"{sec.polar_moment:>12.2f} {sec.section_modulus:>10.2f} "
            f"{sec.polar_section_modulus:>10.2f}"
        )
    print()

    print("-" * 72)
    print("Shoulders (z_position, r/d, D/d) -- Peterson interpolation inputs")
    print("-" * 72)
    for z, sh in fixture.shaft.shoulders():
        print(
            f"  z={z:>7.2f} mm  r={sh.fillet_radius:>5.2f} mm  "
            f"d_small={sh.diameter_small:>6.2f}  d_large={sh.diameter_large:>6.2f}  "
            f"r/d={sh.r_over_d:>6.4f}  D/d={sh.D_over_d:>6.4f}"
        )
    print()

    print("-" * 72)
    print("Query at an arbitrary axial position (midpoint of total_length)")
    print("-" * 72)
    z_mid = fixture.total_length / 2.0
    sec, idx = fixture.shaft.section_at(z_mid)
    print(f"  z_mid = {z_mid:.2f} mm -> section[{idx}] ({sec.label}), "
          f"d={fixture.shaft.diameter_at(z_mid):.2f} mm, "
          f"I={fixture.shaft.I_at(z_mid):.2f} mm4, "
          f"Wt={fixture.shaft.Wt_at(z_mid):.2f} mm3")
    print()

    print("=" * 72)
    print("NOT covered by this script (fixture gap, not this shaft's fault):")
    print("  - inner_diameter (hollow sections) -- no SectionSpec field yet")
    print("  - keyways                          -- no SectionSpec field yet")
    print("=" * 72)


if __name__ == "__main__":
    main()