"""
check_resolution_fem_convergence_external_distributed_cubic.py

Exploratory script (not pytest). Third rung of the same controlled
comparison started by check_resolution_fem_convergence_external_
distributed.py (uniform, 0th-degree q(x)) and continued by
check_resolution_fem_convergence_external_distributed_linear.py
(1st-degree ramp q(x)) -- SAME span [165.0, 185.0] mm, SAME
theta_deg=270.0, SAME 300 N total resultant, SAME shaft3, SAME
Timoshenko theory ("shaft_fem.convergence.external_distributed.
timoshenko" -- theory does NOT change here, only the load's intensity
profile does). This script's "bushing_process_load_cubic" passes a
THIRD-DEGREE polynomial callable instead of a constant or a linear
ramp.

Profile definition: with t = (x - x_lo) / (x_hi - x_lo) in [0, 1],

    q(t) = q_avg + K * (t - 0.5)**3

q_avg = 15 N/mm (same mean as both earlier cases). The cubic term
(t - 0.5)**3 is ANTISYMMETRIC about the span's midpoint and integrates
to exactly zero over t in [0, 1] -- so the total resultant stays
EXACTLY 300 N (q_avg * 20 mm) regardless of K, by construction, not by
numerical coincidence; see main()'s own assert, which checks this both
analytically and via numerical quadrature as a belt-and-braces sanity
check on the profile itself.

K = 80.0 N/mm is chosen so the endpoint values match the earlier
linear ramp's own endpoints as closely as this shape allows: at t=0,
q = 15 - 80*0.125 = 5 N/mm; at t=1, q = 15 + 80*0.125 = 25 N/mm --
IDENTICAL endpoint values to the linear ramp's q1=5/q2=25. What
differs is everything in between: the linear ramp has constant slope
across the whole span, while this cubic is flat (zero slope) at the
midpoint t=0.5 and steepens only near the ends (an S-shaped profile,
inflection at the midpoint). So this comparison isolates CURVATURE of
the load profile specifically -- not just "is it uniform or does it
vary" (that was the uniform-vs-linear comparison) -- while still
matching the linear case's endpoints exactly, so any difference in the
observed order p is attributable to the profile's curvature alone.

Also the ONLY profile in this family with a real, non-placeholder
Euler-Bernoulli sibling -- see
check_resolution_fem_convergence_external_distributed_cubic_euler_bernoulli.py.
Every lower-degree profile tested so far (uniform, linear, and the
separate quadratic investigation) turned out to need no real Euler-
Bernoulli convergence study at all (either textbook closed-form-exact,
or noise-dominated residuals too small to trust as a real order) --
3rd degree is where that stops holding, based on this project's own
empirical read of the quadratic case, so this is the first profile
where an actual Euler-Bernoulli run (not a text-only placeholder) is
expected to show a real, non-trivial p.

Compare this run's printed p_res / f_h0 values, and the written .txt
report, against BOTH earlier reports side by side:
  - check_resolution_fem_convergence_external_distributed.py (uniform)
  - check_resolution_fem_convergence_external_distributed_linear.py (linear ramp)
  - check_resolution_fem_convergence_external_distributed_cubic_euler_bernoulli.py (same profile, Euler-Bernoulli)
All four share span, theta, and total force -- the only things that
ever change across them are q(x)'s shape and, for the last one, theory.

Console output and report-writing follow the same shape as the other
convergence checks -- see check_resolution_fem_convergence_gears.py for
the shared reasoning (not repeated here).
"""
from __future__ import annotations

from pathlib import Path

from axisforge.core.loads import RadialLoad, DistributedRadialLoad
from axisforge.fixtures.construction.construction_capabilities import ConstructionCapabilities
from axisforge.fixtures.studies.study_capabilities import StudyCapabilities
from axisforge.fixtures.studies.outputs.text_report import write_studies_report

HERE = Path(__file__).resolve().parent

# Cubic profile parameters -- see this module's own top docstring for
# why q_avg * (x_hi - x_lo) = 300 N EXACTLY regardless of K (the cubic
# term is antisymmetric about the midpoint and integrates to zero), and
# why K=80.0 was chosen to match the linear ramp's own endpoint values.
_CUBIC_X_LO = 165.0
_CUBIC_X_HI = 185.0
_CUBIC_Q_AVG = 15.0   # N/mm, mean over the span (same as uniform/linear cases)
_CUBIC_K = 80.0        # N/mm, cubic-term coefficient -- endpoints land at 5/25 N/mm


def _cubic_q(x: float,
             x_lo: float = _CUBIC_X_LO, x_hi: float = _CUBIC_X_HI,
             q_avg: float = _CUBIC_Q_AVG, k: float = _CUBIC_K) -> float:
    """q(x) [N/mm], 3rd-degree profile: q_avg + k * (t - 0.5)**3, t in [0, 1]."""
    t = (x - x_lo) / (x_hi - x_lo)
    return q_avg + k * (t - 0.5) ** 3


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

    # Same span/theta/total-force as the uniform and linear cases --
    # cubic q(x) instead. See this module's own top docstring.
    process_load_cubic = DistributedRadialLoad(
        x_lo=_CUBIC_X_LO, x_hi=_CUBIC_X_HI, magnitude=_cubic_q, theta_deg=270.0,
        label="bushing_process_load_cubic", source="user",
    )

    shaft_specs = [
        ShaftSpec(shaft=shaft1, bearings=bearings1, speed_rpm=1450.0, name="shaft1"),
        ShaftSpec(shaft=shaft2, bearings=bearings2, speed_rpm=725.0, name="shaft2"),
        ShaftSpec(shaft=shaft3, bearings=bearings3, speed_rpm=362.5,
                  loads=(sprocket_load, process_load_cubic), name="shaft3"),
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

    # Sanity check on the controlled-comparison setup itself. Analytic
    # check first: the cubic term (t-0.5)**3 integrates to exactly zero
    # over t in [0, 1], so the resultant must equal q_avg * span EXACTLY,
    # independent of K -- if this ever fails, _CUBIC_Q_AVG/_CUBIC_X_LO/
    # _CUBIC_X_HI drifted apart. Numerical quadrature check second, as a
    # belt-and-braces cross-check on _cubic_q() itself (catches a typo
    # in the callable that the analytic argument alone wouldn't).
    expected_total_N = _CUBIC_Q_AVG * (_CUBIC_X_HI - _CUBIC_X_LO)
    assert abs(expected_total_N - 300.0) < 1e-9, (
        f"cubic profile's mean no longer matches the uniform/linear cases' 300 N "
        f"(got {expected_total_N:.3f} N) -- fix _CUBIC_Q_AVG/_CUBIC_X_LO/_CUBIC_X_HI."
    )
    n_quad = 2001
    xs = [_CUBIC_X_LO + i * (_CUBIC_X_HI - _CUBIC_X_LO) / (n_quad - 1) for i in range(n_quad)]
    ys = [_cubic_q(x) for x in xs]
    # Composite trapezoidal rule -- plain Python, no extra dependency.
    h = (_CUBIC_X_HI - _CUBIC_X_LO) / (n_quad - 1)
    quad_total_N = h * (sum(ys) - 0.5 * ys[0] - 0.5 * ys[-1])
    assert abs(quad_total_N - 300.0) < 1e-3, (
        f"_cubic_q() itself does not integrate to 300 N (quadrature gave "
        f"{quad_total_N:.3f} N) -- check the callable, not just the constants."
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

    out_path = HERE / "report_2stage_chain_convergence_external_distributed_cubic.txt"
    write_studies_report(
        system, out_path,
        title="2-stage linear chain -- Mesh convergence (external distributed, CUBIC profile, Timoshenko)",
        convergence_library=library,
    )
    print(f"[OK] {out_path.name} written ({len(system.shafts)} shafts, "
          f"from the real run_convergence() library above -- no fabricated data)")


if __name__ == "__main__":
    main()