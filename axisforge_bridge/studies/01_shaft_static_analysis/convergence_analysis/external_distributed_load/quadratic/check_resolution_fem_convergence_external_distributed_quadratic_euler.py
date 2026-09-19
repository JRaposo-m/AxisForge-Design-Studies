"""
check_resolution_fem_convergence_external_distributed.py


Profile: SAME span [165.0, 185.0] mm, SAME theta_deg=270.0, SAME 300 N
total resultant, SAME shaft3 as every other script in this family --
only the theory (Timoshenko, via
"shaft_fem.convergence.external_distributed.timoshenko") and the
load shape (quadratic instead of uniform/linear/cubic) differ.

With t = (x - x_lo) / (x_hi - x_lo) in [0, 1]:

    q(t) = q_avg + K * ((t - 0.5)**2 - 1/12)

q_avg = 15 N/mm (same mean as every other profile in this family). The
term (t-0.5)**2 has mean 1/12 over t in [0, 1] -- subtracting 1/12
makes the added term itself mean-zero over the span, so the total
resultant is EXACTLY q_avg * 20 mm = 300 N regardless of K, by
construction (same reasoning as the cubic script's antisymmetric term,
just for an EVEN function instead of an odd one -- see main()'s own
assert, checked both analytically and by quadrature).

K = 60.0 N/mm makes q(0) = q(1) = q_avg + K*(0.25 - 1/12) = 15 + 10 =
25 N/mm (matching every earlier script's own upper endpoint) and
q(0.5) = q_avg - K/12 = 15 - 5 = 10 N/mm -- a symmetric "valley" shape,
higher at both ends of the span and lower in the middle. This is
qualitatively different from the linear ramp (odd, monotonic) and the
cubic profile (odd, S-shaped, flat slope at the centre): a quadratic
term is EVEN about the midpoint, the simplest shape a linear (Timoshenko)
OR a cubic (Timoshenko) element cannot represent from two end
values alone -- it is the natural next rung after linear specifically
because a straight line has no way to reproduce it, unlike an odd cubic
term which a cubic Hermite element's own basis can absorb structurally.

Compare this run's p_res / f_h0 against
check_resolution_fem_convergence_external_distributed_linear_timoshenko.py
(same theory, degree-1 load) to read the answer directly: if THIS
script also comes back EXACT/degenerate, the closed-form threshold is
above 2nd degree; if it shows a real p and GCI here (unlike the linear
Timoshenko case), the threshold sits between degree 1 and degree 2.

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

# Quadratic profile parameters -- see this module's own top docstring
# for why q_avg * (x_hi - x_lo) = 300 N EXACTLY regardless of K (the
# (t-0.5)**2 - 1/12 term is mean-zero over the span by construction),
# and why K=60.0 was chosen to match the earlier scripts' own upper
# endpoint value of 25 N/mm.
_QUAD_X_LO = 165.0
_QUAD_X_HI = 185.0
_QUAD_Q_AVG = 15.0   # N/mm, mean over the span (same as every other case)
_QUAD_K = 60.0        # N/mm, quadratic-term coefficient -- endpoints land at 25 N/mm, centre at 10 N/mm


def _quadratic_q(x: float,
                  x_lo: float = _QUAD_X_LO, x_hi: float = _QUAD_X_HI,
                  q_avg: float = _QUAD_Q_AVG, k: float = _QUAD_K) -> float:
    """q(x) [N/mm], 2nd-degree profile: q_avg + k * ((t-0.5)**2 - 1/12), t in [0, 1]."""
    t = (x - x_lo) / (x_hi - x_lo)
    return q_avg + k * ((t - 0.5) ** 2 - 1.0 / 12.0)


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

    # Same span/theta/total-force as every other case in this family --
    # quadratic q(x) instead. See this module's own top docstring.
    process_load_quadratic = DistributedRadialLoad(
        x_lo=_QUAD_X_LO, x_hi=_QUAD_X_HI, magnitude=_quadratic_q, theta_deg=270.0,
        label="bushing_process_load_quadratic", source="user",
    )

    shaft_specs = [
        ShaftSpec(shaft=shaft1, bearings=bearings1, speed_rpm=1450.0, name="shaft1"),
        ShaftSpec(shaft=shaft2, bearings=bearings2, speed_rpm=725.0, name="shaft2"),
        ShaftSpec(shaft=shaft3, bearings=bearings3, speed_rpm=362.5,
                  loads=(sprocket_load, process_load_quadratic), name="shaft3"),
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
    # check first: (t-0.5)**2 - 1/12 is mean-zero over t in [0, 1], so
    # the resultant must equal q_avg * span EXACTLY, independent of K --
    # if this ever fails, _QUAD_Q_AVG/_QUAD_X_LO/_QUAD_X_HI drifted
    # apart. Numerical quadrature check second, as a belt-and-braces
    # cross-check on _quadratic_q() itself.
    expected_total_N = _QUAD_Q_AVG * (_QUAD_X_HI - _QUAD_X_LO)
    assert abs(expected_total_N - 300.0) < 1e-9, (
        f"quadratic profile's mean no longer matches the rest of the family's 300 N "
        f"(got {expected_total_N:.3f} N) -- fix _QUAD_Q_AVG/_QUAD_X_LO/_QUAD_X_HI."
    )
    n_quad = 2001
    xs = [_QUAD_X_LO + i * (_QUAD_X_HI - _QUAD_X_LO) / (n_quad - 1) for i in range(n_quad)]
    ys = [_quadratic_q(x) for x in xs]
    h = (_QUAD_X_HI - _QUAD_X_LO) / (n_quad - 1)
    quad_total_N = h * (sum(ys) - 0.5 * ys[0] - 0.5 * ys[-1])
    assert abs(quad_total_N - 300.0) < 1e-3, (
        f"_quadratic_q() itself does not integrate to 300 N (quadrature gave "
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

    out_path = HERE / "report_2stage_chain_convergence_external_distributed_quadratic_timoshenko.txt"
    write_studies_report(
        system, out_path,
        title="2-stage linear chain -- Mesh convergence (external distributed, QUADRATIC profile, Timoshenko)",
        convergence_library=library,
    )
    print(f"[OK] {out_path.name} written ({len(system.shafts)} shafts, "
          f"from the real run_convergence() library above -- no fabricated data)")


if __name__ == "__main__":
    main()