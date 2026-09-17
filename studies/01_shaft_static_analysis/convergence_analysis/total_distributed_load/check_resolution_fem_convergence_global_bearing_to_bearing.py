"""
check_resolution_fem_convergence_global_bearing_to_bearing.py

Exploratory script (not pytest). Same fixture/geometry as
check_resolution_fem_convergence_total.py (build_system() duplicated
here rather than imported -- same convention every check_*.py script in
this family already follows: each is self-contained, sharing a fixture
DESCRIPTION with its siblings' docstrings, not a shared import).

## NEW (this pass, confirmed with erg 2026-09-17): the GLOBAL half of
## what used to be a single displacement-vs-moment comparison script
## (check_resolution_fem_convergence_total.py). Per erg's own split:
## "quero entao aqui o local para o deslocamento e depois num outro
## ficheiro o global de bearing a bearing" -- LOCAL/per-interval stayed
## in that file (displacement only now); THIS file is the GLOBAL study,
## scoped to domain="bearing_to_bearing" (see
## global_convergence_study.py's own top docstring for the full
## reasoning): refines and tracks convergence only within
## [min(bearing position), max(bearing position)] on each shaft -- the
## main supported span -- rather than the shaft's full extent
## (including any overhang beyond the outermost bearing, e.g. shaft3's
## sprocket_load sitting right at brg_B here). The FEM model solved at
## every grade is still the WHOLE shaft (all loads/BCs included, so
## reactions and the M/v field everywhere are correct) -- only what
## gets refined and sampled for the GCI criterion is scoped to the
## bearing span.

## ASSUMPTION FLAGGED, not confirmed with erg: "bearing a bearing" was
## read as "restrict the refinement/tracking DOMAIN to the span between
## the two bearings", not "evaluate the tracked criterion exactly AT
## each bearing position". If the latter was actually meant, that's a
## `GlobalMetricSpec(..., criterion="node", node_x=<bearing.position>)`
## per bearing instead of (or in addition to) the domain restriction
## below -- straightforward to add once confirmed; left out for now to
## avoid guessing a second design decision on top of the domain one.

Tracks BOTH v and M (components=("res",) -- resultant only, per the
same "resultant unless told otherwise" convention the per-interval
moment study already uses), each with its own default
GlobalMetricSpec thresholds (v: 1%/Fs=1.25; M: 2%/Fs=3.0 -- looser,
same reasoning as everywhere else in this platform: M is a derived,
one-order-lower-accuracy quantity). criterion="max" (the library
default) for both -- the peak response over the bearing span is what
this script reports; see global_convergence_study.py's
GlobalMetricSpec docstring for the other criteria ("mean", "rms",
"max_minus_mean", "node") if a different question is wanted later.

Console output and report-writing follow the same shape as the other
convergence checks -- see check_resolution_fem_convergence_gears.py
for the shared reasoning (not repeated here).
"""
from __future__ import annotations

from pathlib import Path

from axisforge.core.loads import RadialLoad, DistributedRadialLoad
from axisforge.fixtures.construction.construction_capabilities import ConstructionCapabilities
from axisforge.fixtures.studies.study_capabilities import StudyCapabilities
from axisforge.fixtures.studies.outputs.text_report import write_studies_report
from axisforge.fixtures.studies.shafts.convergence_studies.global_convergence_study import (
    default_global_metrics,
)

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

    process_load = DistributedRadialLoad(
        x_lo=165.0, x_hi=185.0, magnitude=300.0, theta_deg=270.0,
        label="bushing_process_load", source="user",
    )

    shaft_specs = [
        ShaftSpec(shaft=shaft1, bearings=bearings1, speed_rpm=1450.0, name="shaft1"),
        ShaftSpec(shaft=shaft2, bearings=bearings2, speed_rpm=725.0, name="shaft2"),
        ShaftSpec(shaft=shaft3, bearings=bearings3, speed_rpm=362.5,
                  loads=(sprocket_load, process_load), name="shaft3"),
    ]
    stage_specs = [
        StageSpec(gear_driver=g1_driver, gear_driven=g2_driven, phi_deg=0.0, label="stage1"),
        StageSpec(gear_driver=g3_driver, gear_driven=g4_driven, phi_deg=90.0, label="stage2"),
    ]

    system = build_linear_system(
        shaft_specs, stage_specs, P=5000.0, rotation_dir_source=1, label="2stage_chain",
    )
    return construction, system


def _summarize(library, system) -> None:
    """
    Same shape as check_resolution_fem_convergence_total.py's own
    _summarize() -- reused unmodified. A "global" study only ever has
    one entry in .per_load (label="global"), so n_total is always 1
    per shaft here; kept generic rather than hardcoded to 1, in case a
    future caller merges a global and a per-interval library together.
    """
    expected_names = {ss.name for ss in system.shafts}
    got_names = set(library.names())
    missing = expected_names - got_names

    ok = not missing
    print(f"    [{'OK' if ok else 'FAIL'}] {len(library)}/{len(system.shafts)} shafts")
    if missing:
        print(f"        missing from library: {missing}")

    for ss in system.shafts:
        r = library.get_or_none(ss.name)
        if r is None:
            continue
        n_total = len(r.per_load)
        n_conv = sum(1 for rec in r.per_load.values() if rec.converged)
        for label, rec in r.per_load.items():
            print(f"        {ss.name:8s} [{label}] span=[{rec.x_lo:.2f}, "
                  f"{rec.x_hi:.2f}] mm  converged={rec.converged}  "
                  f"levels={len(rec.levels)}")
        print(f"        {ss.name:8s} -- {n_conv}/{n_total} converged")


def main() -> None:
    construction, system = build_system()

    study = StudyCapabilities(
        construction=construction,
        shaft_fem=("shaft_fem.convergence.total.timoshenko",),
    )
    objs = study.resolve()
    run_convergence = objs["run_convergence"]
    # ## NOTE: same assumption as check_resolution_fem_convergence_total.py
    # -- beam_theory is expected to already be PINNED to "timoshenko" by
    # study_capabilities.py's partial application for this capability
    # string; `regions` (also pinned by that capability) has NO EFFECT
    # for study_kind="global" -- see convergence_study.run_convergence()'s
    # own docstring -- there are no per-interval regions in a global
    # study, the domain is set by `domain=` below instead.

    print("[RUN] GLOBAL convergence, bearing-to-bearing "
          "(study_kind='global', domain='bearing_to_bearing', v_res + M_res)")
    library_global = run_convergence(
        system, construction,
        shear_theory="cowper", integration_method="exact",
        study_kind="global",
        global_metrics=default_global_metrics(components=("res",), criterion="max"),
        domain="bearing_to_bearing",
    )
    _summarize(library_global, system)

    out_global = HERE / "report_2stage_chain_convergence_global_bearing_to_bearing.txt"
    write_studies_report(
        system, out_global,
        title="2-stage linear chain -- Mesh convergence, GLOBAL bearing-to-bearing "
              "(v_res + M_res, Timoshenko)",
        convergence_library=library_global,
    )
    print(f"\n[OK] {out_global.name} written")


if __name__ == "__main__":
    main()