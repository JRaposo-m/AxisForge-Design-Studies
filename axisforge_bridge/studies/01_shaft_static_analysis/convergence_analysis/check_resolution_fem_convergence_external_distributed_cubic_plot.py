"""
check_resolution_fem_convergence_external_distributed_cubic_plot.py

Exploratory script (not pytest, not a convergence-study check --
matplotlib plot instead). Same cubic-profile shaft3 fixture as
check_resolution_fem_convergence_external_distributed_cubic.py (and its
Euler-Bernoulli sibling) -- SAME span [165.0, 185.0] mm, SAME
theta_deg=270.0, SAME 300 N total resultant, SAME _cubic_q() profile.

Purpose: step away from the abstract p/GCI numbers (which kept coming
back noise-dominated and hard to read for Euler-Bernoulli, degree after
degree) and just look at the two actual deflection shapes side by side:

  - Timoshenko, solved on the "grade_2" mesh -- the finest of the three
    levels the real convergence run
    (check_resolution_fem_convergence_external_distributed_cubic.py)
    actually used for this interval before declaring CONVERGED. Its own
    printed "Extra mandatory nodes" line for this exact interval is
    reused here verbatim as extra_mandatory, so this is not a
    re-guessed mesh -- it is literally the same 13-node grade_2 mesh
    that run already validated as converged.
  - Euler-Bernoulli, solved with NO refinement at all (plain
    global_solver.solve(ss), no extra_mandatory) -- deliberately the
    coarsest/baseline mesh, since every Euler-Bernoulli convergence
    check tried so far (uniform through quartic) came back
    noise-dominated/near-exact regardless of refinement, so there is no
    reason yet to believe refining it would change anything -- this
    plot is itself one more way of checking that assumption, visually.

ASSUMPTION flagged for the user to correct if wrong (not verified
against RigidBearingFEMSolver's own source in this session): solve()
accepts extra_mandatory=<list of floats> as a per-shaft refinement
kwarg (mirrors convergence_study.py's own documented plan for how
MeshRefinementResult.all_extra_nodes would be consumed downstream), and
the theory string for Euler-Bernoulli at the solver level is "euler"
(per study_capabilities.py's own _convergence_require() branches, e.g.
".gears.euler_bernoulli" pins theory="euler") -- adjust _EULER_THEORY
below if the actual constructor expects "euler_bernoulli" instead.
Post-solve() result access confirmed against RigidBearingFEMSolver's own
source (rigid_bearing.py): x_nodes, d_total_xz, d_total_xy are plain
attributes, but each is a flat, INTERLEAVED [u, v, theta] vector per
node (3 * len(x_nodes) long) -- transverse deflection v sits at every
3rd entry starting at offset 1 (v = arr[1::3]), same convention
_boundary_dofs()/return_values() use internally (3*i+1 for v). Two
earlier guesses (a plain d_xz/d_xy attribute, then a per-node-triplet
reshape) were both wrong before this one -- confirmed only once the
user pasted the solver's actual source. theory="euler" for
Euler-Bernoulli is confirmed working (the user's own run solved without
error), even though the constructor itself does no validation --
RigidBearingFEMSolver.__init__ just forwards theory verbatim to
StiffnessMatrixBuilder(theory=theory).

Output: a single PNG with two stacked subplots (d_xz, d_xy vs x),
Timoshenko-grade_2 vs Euler-Bernoulli-baseline overlaid on each, the
load span [165, 185] mm shaded for reference. No .txt report -- this
script's whole point is the picture, not another convergence table.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from axisforge.core.loads import RadialLoad, DistributedRadialLoad
from axisforge.fixtures.construction.construction_capabilities import ConstructionCapabilities
from axisforge.solvers.machine_elements.shaft.fem_solvers.rigid_bearing import (
    RigidBearingFEMSolver,
)

HERE = Path(__file__).resolve().parent

# Same cubic profile as check_resolution_fem_convergence_external_
# distributed_cubic.py -- see that script's own top docstring for the
# antisymmetric-mean-zero-term reasoning.
_CUBIC_X_LO = 165.0
_CUBIC_X_HI = 185.0
_CUBIC_Q_AVG = 15.0
_CUBIC_K = 80.0

# Verbatim from check_resolution_fem_convergence_external_distributed_
# cubic.py's own real run -- the "Extra mandatory nodes (union, for
# Mesh1D)" line printed for shaft3's 'bushing_process_load_cubic'
# interval, i.e. the actual grade_2 (finest, converged) mesh that run
# used -- NOT re-derived here, reused as-is so this plot reflects the
# same mesh the real convergence result already validated.
_GRADE_2_NODES_SHAFT3 = [
    165.000, 166.250, 167.500, 168.750, 170.000, 171.583, 173.167,
    174.750, 176.333, 178.500, 180.667, 182.833, 185.000,
]

# See this module's own top docstring -- flagged assumption, adjust if
# RigidBearingFEMSolver's real constructor expects "euler_bernoulli".
_EULER_THEORY = "euler"


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
    shaft3 = next(ss for ss in system.shafts if ss.name == "shaft3")

    # Timoshenko, solved on the exact grade_2 mesh the real convergence
    # run already validated for this interval.
    solver_timo = RigidBearingFEMSolver(theory="timoshenko")
    solver_timo.solve(shaft3, extra_mandatory=_GRADE_2_NODES_SHAFT3)

    # Euler-Bernoulli, no refinement at all -- baseline mesh only.
    solver_euler = RigidBearingFEMSolver(theory=_EULER_THEORY)
    solver_euler.solve(shaft3)

    print(f"[OK] Timoshenko grade_2 solved -- {len(solver_timo.x_nodes)} node(s)")
    print(f"[OK] Euler-Bernoulli baseline solved -- {len(solver_euler.x_nodes)} node(s)")

    fig, (ax_xz, ax_xy) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # RigidBearingFEMSolver's DOF layout is [u, v, theta] per node,
    # INTERLEAVED (confirmed from the solver's own source: _boundary_dofs
    # uses 3*i / 3*i+1 / 3*i+2 for u/v/theta) -- transverse deflection v
    # sits at every 3rd entry starting at offset 1, not the whole array.
    for ax, attr, plane_label in ((ax_xz, "d_total_xz", "xz"), (ax_xy, "d_total_xy", "xy")):
        ax.axvspan(_CUBIC_X_LO, _CUBIC_X_HI, color="orange", alpha=0.15,
                   label="bushing_process_load_cubic span" if attr == "d_total_xz" else None)
        v_timo = getattr(solver_timo, attr)[1::3]
        v_euler = getattr(solver_euler, attr)[1::3]
        ax.plot(solver_timo.x_nodes, v_timo,
                "o-", color="tab:blue", markersize=3,
                label="Timoshenko (grade_2, 13 nodes)")
        ax.plot(solver_euler.x_nodes, v_euler,
                "s--", color="tab:red", markersize=3,
                label="Euler-Bernoulli (baseline, no refinement)")
        ax.set_ylabel(f"v_{plane_label} [mm]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    ax_xy.set_xlabel("x [mm]")
    fig.suptitle("shaft3 deflection -- cubic distributed load\n"
                 "Timoshenko (grade_2) vs Euler-Bernoulli (no refinement)")
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()