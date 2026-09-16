"""
refine_mesh_case_radial_single_position.py

Runs the ACTUAL mesh-convergence study (MeshConvergenceStudy, displacement
metric -- unmodified from convergence_solver.py) for each shaft in this
case, collects MeshRefinementResult.all_extra_nodes per shaft, then
re-solves the PRODUCTION Timoshenko path (case_radial_single_position.py's
own ts_solve_system, via StudyCapabilities/shaft_fem.timoshenko_rigid)
with those nodes forced into the mesh via extra_mandatory -- so the CSV
this writes comes out of the SAME resolution_csv.py pipeline every other
result in this suite does, not a hand-rolled M reconstruction (that
approach in plot_bearing_span_M_field.py had a sign-convention bug and
is being abandoned in favor of this).

Written to timoshenko/refined_mesh/results/<shear_theory>/csv/ -- one
level below timoshenko/refined_mesh/ itself, mirroring the
timoshenko/results/<shear_theory>/ convention the un-refined case uses.

TODO -- one guess I could not confirm without fem_simple.py's own
solve_system() signature: extra_mandatory is passed as a dict keyed by
shaft name ({shaft_name: [x1, x2, ...]}), per convergence_study.py's own
docstring ("Per-shaft extra_mandatory ... consumed downstream by
fem_simple.solve_system()'s own extra_mandatory dict"). If solve_system()
rejects this shape, paste its signature and I'll fix the call.

Bypasses convergence_study.py's own run_convergence() fixture entirely --
that file still imports the wrong class name (RigidBearingFEMSolver,
same bug rigid_support.py had) and constructs it with theory=theory
(also wrong -- needs a BeamModelSettings, not a bare string), so it
cannot run as-is. Calling MeshConvergenceStudy directly here sidesteps
that without touching the fixture file; worth fixing convergence_study.py
itself separately once this path is validated. Deliberately NOT fixed
here -- run_convergence()/check_resolution_fem_convergence_total.py are
left outdated/bypassed on purpose, same reasoning as above.

Also writes a SHAFT MESH CONVERGENCE report, same
write_studies_report(..., convergence_library=...) call
check_resolution_fem_convergence_total.py uses -- but built from a
ConvergenceResultsLibrary assembled BY HAND here (one MeshRefinementResult
per shaft, its per_load merged from BOTH the REGIONS study `result` and
the gap study `gap_result` computed below), not from run_convergence()
itself (bypassed, see above). MeshRefinementResult.per_load is a plain
mutable dict, so this merge is a straight dict.update() -- both `result`
and `gap_result` already share the same shaft_name (built from the same
`ss`), so nothing needs re-keying.
"""
from __future__ import annotations

import sys
from pathlib import Path

# case_radial_single_position.py lives two levels up from here
# (timoshenko/refined_mesh/ -> timoshenko/ -> case_radial_single_position/)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from case_radial_single_position import build_system, _write_theory_outputs  # noqa: E402

_p = Path(__file__).resolve()
_root = None
for _ancestor in _p.parents:
    if (_ancestor / "common").is_dir():
        _root = _ancestor
        break
if _root is None:
    raise RuntimeError(f"{__file__}: no ancestor directory containing 'common/' found.")
sys.path.insert(0, str(_root))

from axisforge.mesh.shaft.beam_model_settings import BeamModelSettings  # noqa: E402
from axisforge.solvers.machine_elements.shaft.fem_solvers.rigid_support import (  # noqa: E402
    RigidSupportFEMSolver,
)
from axisforge.solvers.mesh.convergence_solver import MeshConvergenceStudy  # noqa: E402
from axisforge.fixtures.studies.study_capabilities import StudyCapabilities  # noqa: E402
from axisforge.fixtures.studies.shafts.convergence_studies.convergence_library import (  # noqa: E402
    ConvergenceResultsLibrary,
)
from axisforge.fixtures.studies.outputs.text_report import write_studies_report  # noqa: E402


SHEAR_THEORY = "cowper"
INTEGRATION_METHOD = "single_point"
GCI_THRESHOLD = 0.01
SAFETY_FACTOR = 1.25
MAX_LEVELS = 8
REGIONS = {"gears", "external_distributed"}  # matches run_convergence()'s own default

# intervals_from_shaft_system()'s VALID_REGIONS is {"gears",
# "external_distributed", "bearings"} -- there is no region for a plain
# point RadialLoad, so LOAD_X_SHAFT1_MM=60.0 / LOAD_X_SHAFT2_MM=140.0
# never get refined by REGIONS alone. Their position IS always an exact
# FEM node (Mesh1D nodes exactly at point-load positions -- confirmed in
# axisforge_validation_and_roadmap.md), so displacement there is already
# nodally exact regardless of mesh. M is not: this Timoshenko element's
# bending_strain_matrix() has a B_b independent of zeta, so M is
# reconstructed as a piecewise-CONSTANT field per element, and its
# accuracy anywhere along the shaft still depends on how small the
# elements are THERE, not just near a load.
#
# First attempt only added a custom interval around each point load's own
# position -- wrong scope: shaft1's only point load sits at x=60, to the
# LEFT of the gear, so the whole span to the RIGHT of the gear
# (x=107.5..190, nothing but the right bearing in it) never got a single
# custom interval and stayed at whatever coarse baseline mesh
# intervals_from_shaft_system() produced -- confirmed by the pasted
# comparison plot: AxisForge M tracks the analytical curve closely up to
# the gear, then diverges hard from ~107.5 to 190. So this now refines
# EVERY remaining gap along the shaft's own bearing-to-bearing span (see
# _shaft_gaps() below), not just the ones that happen to contain a point
# load -- matching "avaliar sempre o meio de cada elemento de cada
# seccao", not just around explicit loads.
#
# Every such gap has at least one bearing on one of its two sides (bearings
# are the domain's own ends) -- so one interval boundary always has to
# rest on a bearing. This makes the GCI signal on that interval
# structurally degenerate: displacement AT a rigid bearing is an exact
# v=0 Dirichlet BC at every mesh grade, so f_coarse=f_medium=f_fine=0
# always, GCI is a 0/0 that can never report "converged", and
# _converge_one_load() then blindly bisects all the way to max_levels
# every time (confirmed: 8 levels => 2**8 subdivisions => 265 nodes just
# for shaft1's left gap alone, far more than an M convergence check
# actually needs). So these gap intervals get their OWN
# MeshConvergenceStudy instance with a deliberately small max_levels
# (GAP_MAX_LEVELS below) instead of reusing MAX_LEVELS -- a fixed, modest
# amount of uniform refinement per gap, not a GCI-driven search that
# can't ever succeed here.
GAP_MAX_LEVELS = 4  # 2**4 = 16x subdivision of each gap -- enough to
                     # meaningfully shrink element size for the
                     # piecewise-constant M field without exploding node
                     # count the way MAX_LEVELS=8 did

CASE_ROOT = Path(__file__).resolve().parents[2]  # .../case_radial_single_position/


def _anchor_interval(ss, x0: float) -> tuple[float, float] | None:
    """
    Bounds for a custom convergence interval straddling x0 (a point load
    position, or a bare gap's own midpoint -- this doesn't care which).

    A bearing anchor uses the FAR edge of its own extent (the edge away
    from x0) so the whole bearing -- extent AND position -- ends up fully
    INSIDE the interval: this is required both to satisfy
    _eval_points_for_interval()'s "does not fully contain bearing '...'"
    check, and to guarantee bearing.position itself becomes a recognized
    eval point (the check only ever adds bearing.position, never an
    extent edge, to the point list).

    A gear/distributed-load anchor instead uses the NEAR edge (the edge
    closest to x0), which keeps that gear/load entirely OUTSIDE the
    interval -- no containment check applies to a feature that isn't
    inside at all, and this avoids re-refining the region REGIONS already
    covers via "gears"/"external_distributed".

    Returns None if there's no anchor on one side (x0 sits at a shaft
    end past every bearing/gear/distributed load).
    """
    left_candidates: list[float] = []
    right_candidates: list[float] = []

    for b in ss.bearings:
        lo_b, hi_b = ss.bearing_extent(b)
        if b.position < x0 - 1e-6:
            left_candidates.append(lo_b)     # far edge -- bearing ends up inside
        elif b.position > x0 + 1e-6:
            right_candidates.append(hi_b)    # far edge -- bearing ends up inside

    for ge in ss.gears:
        lo_g, hi_g = ss.gear_extent(ge)
        if ge.position < x0 - 1e-6:
            left_candidates.append(hi_g)     # near edge -- gear stays outside
        elif ge.position > x0 + 1e-6:
            right_candidates.append(lo_g)    # near edge -- gear stays outside

    for dl in ss.distributed_radial_loads:
        centre = (dl.x_lo + dl.x_hi) / 2.0
        if centre < x0 - 1e-6:
            left_candidates.append(dl.x_hi)  # near edge -- load stays outside
        elif centre > x0 + 1e-6:
            right_candidates.append(dl.x_lo)  # near edge -- load stays outside

    if not left_candidates or not right_candidates:
        return None
    return max(left_candidates), min(right_candidates)


def _shaft_gaps(ss) -> list[float]:
    """
    Midpoint of every gap along the shaft's own bearing-to-bearing span
    that ISN'T already a gear/distributed-load extent covered by REGIONS
    (that gap gets refined separately, with a real non-degenerate GCI
    signal, by the main `study` in main()). Everything else -- including
    a span with NO feature in it at all, like shaft1's bare
    x=107.5..190 gear-to-bearing span (its only point load sits at x=60,
    entirely on the OTHER side of the gear, so that whole span was never
    touched by anything before) -- is returned as a gap to refine.
    """
    bearings_sorted = sorted(ss.bearings, key=lambda b: b.position)
    x_A, x_B = bearings_sorted[0].position, bearings_sorted[-1].position

    anchors: set[float] = {round(x_A, 4), round(x_B, 4)}
    region_spans: list[tuple[float, float]] = []
    for ge in ss.gears:
        lo_g, hi_g = ss.gear_extent(ge)
        anchors.add(round(lo_g, 4))
        anchors.add(round(hi_g, 4))
        if "gears" in REGIONS:
            region_spans.append((lo_g, hi_g))
    for dl in ss.distributed_radial_loads:
        anchors.add(round(dl.x_lo, 4))
        anchors.add(round(dl.x_hi, 4))
        if "external_distributed" in REGIONS:
            region_spans.append((dl.x_lo, dl.x_hi))

    anchors_sorted = sorted(anchors)
    gap_midpoints: list[float] = []
    for lo, hi in zip(anchors_sorted[:-1], anchors_sorted[1:]):
        if hi - lo < 1e-6:
            continue
        if any(lo >= r_lo - 1e-6 and hi <= r_hi + 1e-6 for r_lo, r_hi in region_spans):
            continue  # REGIONS already refines this span with a real GCI signal
        gap_midpoints.append((lo + hi) / 2.0)
    return gap_midpoints


def main() -> None:
    construction, system = build_system()

    settings = BeamModelSettings(beam_theory="timoshenko",
                                  shear_theory=SHEAR_THEORY,
                                  integration_method=INTEGRATION_METHOD)

    extra_mandatory: dict[str, list[float]] = {}
    convergence_library = ConvergenceResultsLibrary()

    for ss in system.shafts:
        global_solver = RigidSupportFEMSolver(settings)
        global_solver.solve(ss)

        study = MeshConvergenceStudy(
            global_solver,
            gci_threshold=GCI_THRESHOLD,
            safety_factor=SAFETY_FACTOR,
            max_levels=MAX_LEVELS,
        )
        intervals, skipped = MeshConvergenceStudy.intervals_from_shaft_system(ss, regions=REGIONS)
        for reason in skipped:
            print(f"  [SKIP] {ss.name}: {reason}")

        result = study.run(ss, intervals)
        nodes = set(result.all_extra_nodes)
        print(f"  [{ss.name}] convergence study found {len(nodes)} extra node(s) "
              f"from REGIONS={sorted(REGIONS)}")
        for label, rec in result.per_load.items():
            print(f"      interval '{label}': converged={rec.converged}  levels={rec.levels}")

        # custom intervals for every remaining gap along the shaft --
        # see GAP_MAX_LEVELS / _shaft_gaps() / _anchor_interval() above.
        # NOT limited to spans that happen to contain a point load: a
        # span with no feature in it at all still has its own M value
        # governed by a single large, piecewise-constant element, and
        # needs the same treatment. Run through a SEPARATE
        # MeshConvergenceStudy instance with a small, deliberate
        # max_levels instead of MAX_LEVELS, since the GCI here is
        # structurally degenerate (see the comment above) and would
        # otherwise always blindly bisect all the way to MAX_LEVELS.
        gap_intervals: list[tuple[float, float, str]] = []
        seen_bounds: set[tuple[float, float]] = set()
        for x0 in _shaft_gaps(ss):
            bounds = _anchor_interval(ss, x0)
            if bounds is None:
                print(f"  [{ss.name}] gap around x={x0:.1f} has no bearing/gear "
                      f"anchor on both sides -- skipped")
                continue
            lo, hi = bounds
            key = (round(lo, 4), round(hi, 4))
            if key in seen_bounds:
                continue
            seen_bounds.add(key)
            label = f"gap@[{lo:.1f},{hi:.1f}]"
            gap_intervals.append((lo, hi, label))
            print(f"  [{ss.name}] added custom gap interval '{label}': "
                  f"[{lo:.2f}, {hi:.2f}]")

        if gap_intervals:
            gap_study = MeshConvergenceStudy(
                global_solver,
                gci_threshold=GCI_THRESHOLD,
                safety_factor=SAFETY_FACTOR,
                max_levels=GAP_MAX_LEVELS,
            )
            gap_result = gap_study.run(ss, gap_intervals)
            gap_nodes = set(gap_result.all_extra_nodes)
            nodes |= gap_nodes
            print(f"  [{ss.name}] gap refinement added {len(gap_nodes)} "
                  f"more node(s) (max_levels={GAP_MAX_LEVELS})")
            for label, rec in gap_result.per_load.items():
                print(f"      interval '{label}': converged={rec.converged}  levels={rec.levels}")

            # Merge the gap study's intervals into the SAME
            # MeshRefinementResult the REGIONS study already produced --
            # both were built off this same `ss`, so result.shaft_name ==
            # gap_result.shaft_name already; per_load is a plain dict, so
            # this is just a dict.update() (no key collisions expected,
            # REGIONS labels are gear/dist-load labels, gap labels are
            # "gap@[...]").
            result.per_load.update(gap_result.per_load)

        convergence_library.store(result)

        nodes = sorted(nodes)
        extra_mandatory[ss.name] = nodes
        print(f"  [{ss.name}] total extra node(s): {len(nodes)}")

    ts_study = StudyCapabilities(construction=construction, shaft_fem=("shaft_fem.timoshenko_rigid",))
    ts_solve_system = ts_study.resolve()["solve_system"]

    ts_library = ts_solve_system(
        system, construction,
        shear_theory=SHEAR_THEORY,
        integration_method=INTEGRATION_METHOD,
        extra_mandatory=extra_mandatory,  # TODO: confirm shape against fem_simple.py
    )

    out_dir = CASE_ROOT / "timoshenko" / "refined_mesh" / "results" / SHEAR_THEORY
    _write_theory_outputs(
        system, out_dir, ts_library,
        title=f"case_radial_single_position -- refined mesh (timoshenko/{SHEAR_THEORY})",
    )

    # SHAFT MESH CONVERGENCE report -- same write_studies_report(...,
    # convergence_library=...) call check_resolution_fem_convergence_total.py
    # uses, but fed from `convergence_library` assembled by hand above
    # (REGIONS + gap studies merged per shaft) instead of run_convergence()
    # itself, which stays bypassed/outdated for this case (see module
    # docstring). Written next to this case's own refined-mesh results,
    # not under results/<shear_theory>/ -- this report is one artifact
    # covering ALL shafts/intervals studied for the refinement, not a
    # per-shear-theory resolution output.
    convergence_report_path = CASE_ROOT / "timoshenko" / "refined_mesh" / "report_mesh_convergence.txt"
    write_studies_report(
        system, convergence_report_path,
        title=f"case_radial_single_position -- refined mesh (timoshenko/{SHEAR_THEORY}) -- mesh convergence",
        convergence_library=convergence_library,
    )
    print(f"[OK] {convergence_report_path.name} written "
          f"({len(convergence_library)} shaft(s) -- REGIONS={sorted(REGIONS)} "
          f"intervals + per-gap intervals merged)")


if __name__ == "__main__":
    main()
