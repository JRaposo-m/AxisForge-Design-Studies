"""
validation/fem_studies/resolution/uniform_shafts/point_load/case_radial_single_position/timoshenko/refined_mesh/refine_mesh_case_radial_single_position.py

Mesh-convergence study for every shaft in this case, GLOBAL / bearing-to-
bearing (the whole supported span [x_A, x_B] refined at once), plus an
optional manual node-forcing step for known AxisForge-vs-Abaqus
divergence spots, then a production re-solve of the Timoshenko path with
the union of both node sets.

REWRITE (this pass) -- no more fixtures / StudyCapabilities / run_convergence():
the fixtures layer is gone and the core no longer has a dispatcher, so the
grade loop that run_convergence() used to hide is written out here, on
top of the core pieces it was built from:

    RigidSupportFEMSolver(settings).solve(ss)      -> base ShaftResults
    Grader(x_lo, x_hi, base x_nodes)               -> grade_N node sets
        (used internally by SubmodelSolver.solve(..., grade))
    SubmodelSolver(x_lo, x_hi).solve(base, ss, settings, grade)
                                                   -> SubmodelResult per grade
    MeshConvergenceStudy(requests, ...).add_level()-> Richardson GCI
    RigidSupportFEMSolver(settings).solve(ss, extra_mandatory=nodes)
                                                   -> refined ShaftResults

What changed in the SCIENCE, not just the plumbing (flag, do not hide):
  1. SubmodelResult has NO v_res / M_res fields any more -- only the
     per-plane arrays (v_xz, v_xy, M_xz, M_xy, ...). The old
     default_global_metrics(components=("res",)) therefore cannot be
     reproduced literally. Requests below are ("v_xz","max"),
     ("v_xy","max"), ("M_xz","max"), ("M_xy","max"): max|.| over the span
     for each plane (same aggregation the old global study used for the
     resultant). Adding v_res/M_res as properties on SubmodelResult is a
     one-liner in core -- not done here, needs your authorization.
  2. MeshConvergenceStudy has ONE (gci_threshold, safety_factor) for all
     its requests (MetricSpec per-metric overrides are gone). The old
     script used (0.01, 1.25) for v and (0.02, 3.0) for M, so two
     studies are run side by side, fed with the SAME SubmodelResult each
     grade -- one for v, one for M -- to keep those thresholds.
  3. Element-constant M recovery (Timoshenko 2-node element): M at a
     node is the element-midpoint value, so max|M| is not the true peak
     and may converge non-monotonically -> RichardsonGCI falls back to
     _DummyGCI (converged=False, reported as such). If M never converges
     that is a property of the recovery, not of this script.
  4. report_mesh_convergence_global.txt is written by the bridge's
     outputs/solver_results/convergence_report.write_convergence_report()
     (replaces write_studies_report).

Written to timoshenko/refined_mesh/results/<shear_theory>/ (report,
plots, csv -- same writers as the base case) and
timoshenko/refined_mesh/report_mesh_convergence_global.txt.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# Importing the case script also sets sys.path for `common` and `axisforge_bridge`.
from case_radial_single_position import (  # noqa: E402
    build_system, solve_system, _write_theory_outputs,
)

from axisforge.mesh.shaft.beam_model_settings import BeamModelSettings  # noqa: E402
from axisforge.results.convergence_results.convergence_results import (  # noqa: E402
    ConvergenceRecord, MeshRefinementResult,
)
from axisforge.solvers.machine_elements.shaft.fem_solvers.global_solver.rigid_support import (  # noqa: E402
    RigidSupportFEMSolver,
)
from axisforge.solvers.machine_elements.shaft.fem_solvers.submodel_solver.lagrange_multipliers import (  # noqa: E402
    SubmodelSolver,
)
from axisforge.solvers.mesh.convergence_solver import (  # noqa: E402
    MeshConvergenceStudy, build_mesh_refinement_result,
)
from axisforge_bridge.outputs.solver_results.convergence_report import (  # noqa: E402
    write_convergence_report,
)


SHEAR_THEORY = "cowper"
INTEGRATION_METHOD = "single_point"

V_GCI_THRESHOLD = 0.01
V_SAFETY_FACTOR = 1.25
M_GCI_THRESHOLD = 0.02
M_SAFETY_FACTOR = 3.0
MAX_LEVELS = 8            # grade_0 .. grade_{MAX_LEVELS-1}

# (field, eval form) -- see convergence_solver.py's eval-form registry.
# "max" = max|values| over the whole bearing-to-bearing span.
V_REQUESTS = [("v_xz", "max"), ("v_xy", "max")]
M_REQUESTS = [("M_xz", "max"), ("M_xy", "max")]

# ---------------------------------------------------------------------
# Manual node forcing at known AxisForge-vs-Abaqus divergence spots
# (per erg: "gera 5 nodes em cada lado dos que estao a divergir, mas de
# forma manual" -- NOT GCI-driven, just prescribed extra density; unioned
# with the converged nodes afterwards). x-locations are a best guess from
# the pasted M_Nmm comparison plot for shaft1 (divergence around
# x~180-190 mm, in the bare gear->bearing gap). Plain numbers, adjust.
DIVERGENCE_POINTS_MM: dict[str, list[float]] = {
    "shaft1": [185.0],
}
N_MANUAL_NODES_PER_SIDE = 5
MANUAL_NODE_SPACING_MM = 2.0

CASE_ROOT = Path(__file__).resolve().parents[2]  # .../case_radial_single_position/


# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------

def _bearing_span(ss) -> tuple[float, float]:
    """domain='bearing_to_bearing': [min, max] of the bearing positions."""
    pos = sorted(b.position for b in ss.bearings)
    return pos[0], pos[-1]


def _manual_divergence_nodes(ss) -> list[float]:
    """N_MANUAL_NODES_PER_SIDE extra nodes on each side of every point in
    DIVERGENCE_POINTS_MM[ss.name], MANUAL_NODE_SPACING_MM apart, clipped
    to the shaft's own span (read off ss.shaft, like Mesh1D does --
    ShaftSystem has no `.length`)."""
    points = DIVERGENCE_POINTS_MM.get(ss.name, [])
    if not points:
        return []
    x_min = ss.shaft.axial_start(0)
    x_max = ss.shaft.axial_end(ss.shaft.n_sections - 1)
    nodes: set[float] = set()
    for x0 in points:
        for i in range(1, N_MANUAL_NODES_PER_SIDE + 1):
            for x in (x0 - i * MANUAL_NODE_SPACING_MM, x0 + i * MANUAL_NODE_SPACING_MM):
                if x_min <= x <= x_max:
                    nodes.add(round(x, 6))
    return sorted(nodes)


# ---------------------------------------------------------------------
# The grade loop (what run_convergence(study_kind="global") used to do)
# ---------------------------------------------------------------------

def run_global_convergence(
    solve_level: Callable[[str], object],
    x_lo: float,
    x_hi: float,
    max_levels: int = MAX_LEVELS,
) -> tuple[ConvergenceRecord, ConvergenceRecord]:
    """
    Refine [x_lo, x_hi] by successive bisection (grade_0, grade_1, ...),
    feeding each grade's SubmodelResult to two MeshConvergenceStudy
    instances (v thresholds / M thresholds). Stops when BOTH have
    converged (needs >= 3 grades) or max_levels is exhausted.

    solve_level(grade) -> SubmodelResult  (injected so the loop can be
    tested without a FEM solve).

    Returns (v_record, m_record). A record's x_final is the node set of
    the grade at which IT converged (or the last grade tried, see
    finalize_unconverged) -- bisection is nested, so the finer of the two
    contains the other.
    """
    v_study = MeshConvergenceStudy(V_REQUESTS, gci_threshold=V_GCI_THRESHOLD,
                                   safety_factor=V_SAFETY_FACTOR)
    m_study = MeshConvergenceStudy(M_REQUESTS, gci_threshold=M_GCI_THRESHOLD,
                                   safety_factor=M_SAFETY_FACTOR)
    v_rec = v_study.new_record("global_v", x_lo, x_hi)
    m_rec = m_study.new_record("global_M", x_lo, x_hi)

    last = None
    for lvl in range(max_levels):
        grade = f"grade_{lvl}"
        last = solve_level(grade)
        if not v_rec.converged:
            v_study.add_level(v_rec, grade, last)
        if not m_rec.converged:
            m_study.add_level(m_rec, grade, last)
        if v_rec.converged and m_rec.converged:
            break

    for rec, study in ((v_rec, v_study), (m_rec, m_study)):
        if not rec.converged and last is not None:
            study.finalize_unconverged(rec, last)
    return v_rec, m_rec


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main() -> None:
    system = build_system()
    settings = BeamModelSettings(
        beam_theory="timoshenko", shear_theory=SHEAR_THEORY,
        integration_method=INTEGRATION_METHOD,
    )

    print("[RUN] GLOBAL mesh convergence, bearing-to-bearing (v_xz/v_xy and M_xz/M_xy, max|.|)")
    base_results = solve_system(system, settings)   # grade reference mesh

    refinements: dict[str, MeshRefinementResult] = {}
    extra_mandatory: dict[str, list[float]] = {}

    for ss in system.shafts:
        x_lo, x_hi = _bearing_span(ss)
        base = base_results[ss.name]
        sub = SubmodelSolver(x_lo, x_hi)

        v_rec, m_rec = run_global_convergence(
            lambda grade, _sub=sub, _base=base, _ss=ss: _sub.solve(_base, _ss, settings, grade),
            x_lo, x_hi,
        )

        for rec in (v_rec, m_rec):
            print(f"    {ss.name:8s} [{rec.label}] span=[{rec.x_lo:.2f}, {rec.x_hi:.2f}] mm  "
                  f"converged={rec.converged}  levels={len(rec.levels)}")

        # (a) nodes the global study converged to (both metrics), (b) manual nodes.
        refinement = build_mesh_refinement_result(
            ss.name, {v_rec.label: v_rec, m_rec.label: m_rec},
        )
        refinements[ss.name] = refinement
        converged_nodes = set(refinement.all_extra_nodes)
        manual_nodes = _manual_divergence_nodes(ss)
        nodes = sorted(converged_nodes | set(manual_nodes))
        extra_mandatory[ss.name] = nodes
        print(f"  [{ss.name}] total extra node(s) (global-converged union manual): "
              f"{len(nodes)} (manual={len(manual_nodes)})")

    # Production solve on the refined mesh.
    refined: dict = {}
    for ss in system.shafts:
        refined[ss.name] = RigidSupportFEMSolver(settings).solve(
            ss, extra_mandatory=extra_mandatory[ss.name],
        )

    out_dir = CASE_ROOT / "timoshenko" / "refined_mesh" / "results" / SHEAR_THEORY
    _write_theory_outputs(
        system, refined, out_dir,
        title=f"case_radial_single_position -- refined mesh (timoshenko/{SHEAR_THEORY})",
    )

    report_path = CASE_ROOT / "timoshenko" / "refined_mesh" / "report_mesh_convergence_global.txt"
    write_convergence_report(
        refinements, report_path,
        title=f"case_radial_single_position -- refined mesh (timoshenko/{SHEAR_THEORY}) "
              f"-- GLOBAL bearing-to-bearing convergence",
        header_lines=[
            "domain = bearing_to_bearing; metric = max|.| over the span, per bending plane",
            f"v (v_xz, v_xy): GCI threshold {V_GCI_THRESHOLD}, safety factor {V_SAFETY_FACTOR}",
            f"M (M_xz, M_xy): GCI threshold {M_GCI_THRESHOLD}, safety factor {M_SAFETY_FACTOR}",
            f"max levels = {MAX_LEVELS}",
        ],
    )
    print(f"\n[OK] {report_path.name} written ({len(refinements)} shaft(s), domain='bearing_to_bearing')")


if __name__ == "__main__":
    main()