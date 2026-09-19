"""
check_resolution_fem_convergence_total.py

Exploratory script (not pytest). Same fixture as
check_resolution_fem_convergence_external_distributed.py (shaft3's
extra "bushing_process_load" DistributedRadialLoad included -- see that
script's own docstring for exactly why/where it was placed).

Exercises "shaft_fem.convergence.total.timoshenko": runs
convergence_study.run_convergence() with regions={"gears",
"external_distributed"} (pinned by the capability), Timoshenko theory.

## CHANGED (this pass, confirmed with erg 2026-09-17): this script is
## now LOCAL/DISPLACEMENT ONLY -- study_kind="displacement"
## (per-interval submodels, v_xz/v_xy/v_res, one fixed point per
## interval). The moment run (study_kind="moment") and the v-vs-M
## comparison block this script used to also print are REMOVED --
## moved to a dedicated file per erg's own split: "quero entao aqui o
## local para o deslocamento e depois num outro ficheiro o global de
## bearing a bearing". This file keeps exactly the LOCAL/displacement
## half of what used to be here; there is no longer any
## MomentConvergenceStudy dependency, no default_moment_metrics import,
## and no _compare() helper. The GLOBAL, bearing-to-bearing study
## (both v and M, over the main span between bearings, not per feature)
## now lives in check_resolution_fem_convergence_global_bearing_to_bearing.py.
##
## ## ASSUMPTIONS FLAGGED, not verified against real code:
##   1. `objs["run_convergence"]` (resolved through StudyCapabilities)
##      is assumed to accept `study_kind=`/`metrics=` kwarg passthrough
##      unchanged from a direct convergence_study.run_convergence() call
##      -- study_capabilities.py's `_convergence_require()` only pins
##      `beam_theory`/`regions` via functools.partial, so this should
##      hold, but hasn't been re-run against the real capability
##      resolution since this pass.
##   2. ShaftResultsReader().read(global_solver)'s exact signature --
##      see convergence_study.py's own flagged assumption, inherited
##      here since this script now depends on it transitively.

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
from axisforge.solvers.mesh.metric_spec import default_displacement_metrics

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
        labels = ", ".join(r.per_load.keys()) or "(none)"
        print(f"        {ss.name:8s} intervals=[{labels}]  converged={n_conv}/{n_total}")


def main() -> None:
    construction, system = build_system()

    study = StudyCapabilities(
        construction=construction,
        shaft_fem=("shaft_fem.convergence.total.timoshenko",),
    )
    objs = study.resolve()
    run_convergence = objs["run_convergence"]
    # ## NOTE: beam_theory is expected to already be PINNED to "timoshenko"
    # by study_capabilities.py's partial application for this capability
    # string (same convention solve_system() uses) -- not passed here.
    # shear_theory/integration_method are passed at call time, same as
    # case_radial_single_position.py's solve_system() usage.

    print("[RUN] displacement convergence, LOCAL per-interval "
          "(study_kind='displacement', v_xz/v_xy/v_res)")
    library_v = run_convergence(
        system, construction,
        shear_theory="cowper", integration_method="exact",
        study_kind="displacement",
        metrics=default_displacement_metrics(),
    )
    _summarize(library_v, system)

    out_v = HERE / "report_2stage_chain_convergence_total_v.txt"
    write_studies_report(
        system, out_v,
        title="2-stage linear chain -- Mesh convergence, LOCAL DISPLACEMENT "
              "(gears + external distributed, Timoshenko)",
        convergence_library=library_v,
    )
    print(f"\n[OK] {out_v.name} written")


if __name__ == "__main__":
    main()