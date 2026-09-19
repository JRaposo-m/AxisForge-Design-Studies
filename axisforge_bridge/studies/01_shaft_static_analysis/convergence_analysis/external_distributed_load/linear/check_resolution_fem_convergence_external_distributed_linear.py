"""
check_resolution_fem_convergence_external_distributed_linear.py

Exploratory script (not pytest). Controlled variant of
check_resolution_fem_convergence_external_distributed.py -- SAME span
[165.0, 185.0] mm, SAME theta_deg=270.0, SAME total resultant force
(300 N), on the SAME shaft3. The ONLY thing that changes is the
intensity profile: that script's "bushing_process_load" is UNIFORM
(magnitude=300.0 -> q(x) = 300/20 = 15 N/mm constant, per
DistributedRadialLoad's own default -- see loads.py's own docstring,
confirmed uniform is the default, NOT linear); this script's
"bushing_process_load_linear" passes a callable instead, ramping
linearly from q1=5 N/mm at x_lo to q2=25 N/mm at x_hi -- average
(q1+q2)/2 = 15 N/mm over the same 20 mm span integrates to the exact
same 300 N resultant, so any difference in the convergence behaviour
below (observed order p, GCI, number of levels to converge) is
attributable to the SHAPE of the load, not its magnitude.

Purpose: check whether MeshConvergenceStudy's observed order of
convergence p changes between a uniform and a linear intensity profile
over the same span. A uniform DistributedRadialLoad takes the
closed-form path in bending_moment_contribution() (self._uniform=True);
a callable magnitude takes the numerical-quadrature path instead
(self._uniform=False, see loads.py's own __init__) -- p is measured
from how the FEM's own point metric changes with mesh refinement, not
from which of those two paths computed the load's contribution, so in
principle p should still land close to the same value (the FEM element
order does not change), but this is exactly the kind of assumption
worth checking against a real run rather than asserting from the
formula alone -- see check_resolution_fem_convergence_external_distributed.py's
own run, where 'bushing_process_load' (uniform) already showed p in
[2.40, 2.44], noticeably higher than gears' p~=2.00 -- so this
comparison also tells us whether that elevated p was about the span/
evaluation point, or about the uniform profile itself.

Console output and report-writing follow the same shape as the other
convergence checks -- see check_resolution_fem_convergence_gears.py for
the shared reasoning (not repeated here). Compare this run's printed
p / f_h0 values against check_resolution_fem_convergence_external_distributed.py's
own console output (or the two written .txt reports side by side) to
read the actual answer.
"""
from __future__ import annotations

from pathlib import Path

from axisforge.core.loads import RadialLoad, DistributedRadialLoad
from axisforge.fixtures.construction.construction_capabilities import ConstructionCapabilities
from axisforge.fixtures.studies.study_capabilities import StudyCapabilities
from axisforge.fixtures.studies.outputs.text_report import write_studies_report

HERE = Path(__file__).resolve().parent

# Ramp endpoints for the linear intensity profile -- see this module's
# own top docstring for why (q1 + q2) / 2 * (x_hi - x_lo) must equal
# the uniform case's 300 N for the comparison to isolate shape alone.
_LINEAR_X_LO = 165.0
_LINEAR_X_HI = 185.0
_LINEAR_Q1 = 5.0    # N/mm at x_lo
_LINEAR_Q2 = 25.0   # N/mm at x_hi -- average (5+25)/2 = 15 N/mm * 20 mm = 300 N


def _linear_q(x: float,
              x_lo: float = _LINEAR_X_LO, x_hi: float = _LINEAR_X_HI,
              q1: float = _LINEAR_Q1, q2: float = _LINEAR_Q2) -> float:
    """q(x) [N/mm], linear ramp from q1 at x_lo to q2 at x_hi."""
    t = (x - x_lo) / (x_hi - x_lo)
    return q1 + t * (q2 - q1)


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

    # Same span/theta/total-force as check_resolution_fem_convergence_
    # external_distributed.py's "bushing_process_load" -- linear q(x)
    # instead of uniform. See this module's own top docstring.
    process_load_linear = DistributedRadialLoad(
        x_lo=_LINEAR_X_LO, x_hi=_LINEAR_X_HI, magnitude=_linear_q, theta_deg=270.0,
        label="bushing_process_load_linear", source="user",
    )

    shaft_specs = [
        ShaftSpec(shaft=shaft1, bearings=bearings1, speed_rpm=1450.0, name="shaft1"),
        ShaftSpec(shaft=shaft2, bearings=bearings2, speed_rpm=725.0, name="shaft2"),
        ShaftSpec(shaft=shaft3, bearings=bearings3, speed_rpm=362.5,
                  loads=(sprocket_load, process_load_linear), name="shaft3"),
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

    # Sanity check on the controlled-comparison setup itself -- if this
    # ever fails, the p/GCI comparison against the uniform run is no
    # longer apples-to-apples.
    expected_total_N = (_LINEAR_Q1 + _LINEAR_Q2) / 2.0 * (_LINEAR_X_HI - _LINEAR_X_LO)
    assert abs(expected_total_N - 300.0) < 1e-9, (
        f"linear ramp endpoints no longer match the uniform case's 300 N "
        f"(got {expected_total_N:.3f} N) -- fix _LINEAR_Q1/_LINEAR_Q2."
    )

    study = StudyCapabilities(
        construction=construction,
        shaft_fem=("shaft_fem.convergence.external_distributed.timoshenko",),
    )
    objs = study.resolve()
    run_convergence = objs["run_convergence"]

    library = run_convergence(system, construction)

    expected_names = {ss.name for ss in system.shafts}
    got_names = set(library.names())
    missing = expected_names - got_names

    ok = not missing
    print(f"[{'OK' if ok else 'FAIL'}] convergence studied {len(library)}/{len(system.shafts)} "
          f"shafts -- {library!r}")
    if missing:
        print(f"    missing from library: {missing}")

    for ss in system.shafts:
        r = library.get_or_none(ss.name)
        if r is None:
            continue
        n_total = len(r.per_load)
        n_conv = sum(1 for rec in r.per_load.values() if rec.converged)
        labels = ", ".join(r.per_load.keys()) or "(none)"
        print(f"    {ss.name:8s} intervals=[{labels}]  converged={n_conv}/{n_total}")
        for label, rec in r.per_load.items():
            if rec.gci_history:
                p_res = getattr(rec.gci_history[-1]["res"], "p", float("nan"))
                print(f"        '{label}': p_res={p_res:.4f} (levels tried: {len(rec.levels)})")

    out_path = HERE / "report_2stage_chain_convergence_external_distributed_linear.txt"
    write_studies_report(
        system, out_path,
        title="2-stage linear chain -- Mesh convergence (external distributed, LINEAR profile, Timoshenko)",
        convergence_library=library,
    )
    print(f"[OK] {out_path.name} written ({len(system.shafts)} shafts, "
          f"from the real run_convergence() library above -- no fabricated data)")


if __name__ == "__main__":
    main()