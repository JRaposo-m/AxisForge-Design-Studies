"""
refine_mesh_case_radial_single_position.py

Mesh-convergence study for every shaft in this case, GLOBAL/bearing-to-
bearing (v_res + M_res together, over the whole supported span -- see
axisforge/fixtures/studies/shafts/convergence_studies/global_convergence_study.py's
own docstring), plus an optional manual node-forcing step for known
AxisForge-vs-Abaqus divergence spots, then re-solves the production
Timoshenko path (StudyCapabilities -> shaft_fem.timoshenko_rigid) with
the union of both.

## CHANGED (this pass, confirmed with erg 2026-09-17): "refaz o codigo
## no mesmo sentido do global bearing to bearing" -- the whole
## per-interval REGIONS study (displacement AND moment, run
## separately), the load-free-gap detection (_shaft_gaps()/
## _anchor_interval()), and the "gaps get no GCI, here's a warning
## instead" workaround are REMOVED. That machinery existed specifically
## because MeshConvergenceStudy/MomentConvergenceStudy only ever
## studied REGIONS intervals (gear face widths, distributed-load
## spans) and had no way to say anything meaningful about the plain
## shaft material between them -- which is exactly the domain
## restriction that motivated bypassing convergence_study.run_convergence()
## in the first place ("custom gap intervals aren't expressible
## through intervals_from_shaft_system() yet", see this module's
## previous top docstring).
##
## study_kind="global", domain="bearing_to_bearing" removes that
## limitation structurally rather than by adding more special-casing:
## it refines and tracks GCI over the WHOLE bearing-to-bearing span at
## once (v_res and M_res, each aggregated as max(|.|) over that span --
## see GlobalMetricSpec's own docstring in global_convergence_study.py),
## so a load-free stretch is no longer a blind spot that needs its own
## detection/warning/manual-override machinery -- it's just part of
## the one domain being refined. This script can now go straight
## through convergence_study.run_convergence() instead of hand-rolling
## its own RigidSupportFEMSolver/ShaftResultsReader/
## MeshConvergenceStudy loop.
##
## What's KEPT: DIVERGENCE_POINTS_MM / _manual_divergence_nodes() --
## that mechanism addresses a DIFFERENT problem (a known
## AxisForge-vs-Abaqus mismatch at a specific spot, found by external
## comparison, not something a GCI mesh-convergence criterion measures
## by definition) and stays as an independent, still-manual override on
## top of whatever the global study already converges to.
##
## The v-vs-M "does one converge without the other" comparison
## (_compare()) is also removed -- the single GLOBAL study now reports
## v_res and M_res side by side per grade level in one table (see
## global_convergence_study.py's report shape via convergence_report.py),
## so there is no longer a separate library per metric to diff;
## write_studies_report()'s own table already shows that comparison
## directly.

Written to timoshenko/refined_mesh/results/<shear_theory>/csv/.
"""
from __future__ import annotations

import sys
from pathlib import Path

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

from axisforge.fixtures.studies.study_capabilities import StudyCapabilities  # noqa: E402
from axisforge.fixtures.studies.outputs.text_report import write_studies_report  # noqa: E402
from axisforge.fixtures.studies.shafts.convergence_studies.convergence_study import (  # noqa: E402
    run_convergence,
)
from axisforge.fixtures.studies.shafts.convergence_studies.global_convergence_study import (  # noqa: E402
    default_global_metrics,
)


SHEAR_THEORY = "cowper"
INTEGRATION_METHOD = "single_point"

V_GCI_THRESHOLD = 0.01
V_SAFETY_FACTOR = 1.25
M_GCI_THRESHOLD = 0.02
M_SAFETY_FACTOR = 3.0
MAX_LEVELS = 8

# ---------------------------------------------------------------------
# Manual node forcing at known AxisForge-vs-analytical divergence spots
# (per erg: "gera 5 nodes em cada lado dos que estao a divergir, mas de
# forma manual" -- NOT GCI-driven, just prescribed extra density. This
# addresses a DIFFERENT question than the global convergence study
# below -- an external AxisForge-vs-Abaqus mismatch, not mesh GCI --
# so it stays independent and manual, unioned in afterwards.
#
# x-locations below are a best guess from the pasted M_Nmm comparison
# plot for shaft1 (visible divergence around x~180-190mm, in the bare
# gear->bearing gap with no load in it). Adjust freely -- these are
# plain numbers, not derived from anything.
DIVERGENCE_POINTS_MM: dict[str, list[float]] = {
    "shaft1": [185.0],
}
N_MANUAL_NODES_PER_SIDE = 5
MANUAL_NODE_SPACING_MM = 2.0

CASE_ROOT = Path(__file__).resolve().parents[2]  # .../case_radial_single_position/


def _manual_divergence_nodes(ss) -> list[float]:
    """N_MANUAL_NODES_PER_SIDE extra nodes on each side of every point in
    DIVERGENCE_POINTS_MM[ss.name], spaced MANUAL_NODE_SPACING_MM apart,
    clipped to the shaft's own span.

    ## FIXED (this pass): was `ss.length` -- ShaftSystem has no such
    ## attribute (confirmed by the AttributeError this raised at
    ## runtime). Mesh1D._mandatory_positions() computes the shaft's own
    ## extent the same way every other module in this codebase does --
    ## shaft.axial_start(0) (always 0.0) to
    ## shaft.axial_end(shaft.n_sections - 1) -- so that's what this now
    ## reads too, off ss.shaft (the actual geometry object), not off
    ## ShaftSystem itself.
    """
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


def _summarize(library, system) -> None:
    """
    Same generic shape as check_resolution_fem_convergence_global_bearing_to_bearing.py's
    own _summarize() -- a "global" study has exactly one entry
    (label="global") in .per_load per shaft.
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
        for label, rec in r.per_load.items():
            print(f"        {ss.name:8s} [{label}] span=[{rec.x_lo:.2f}, {rec.x_hi:.2f}] mm  "
                  f"converged={rec.converged}  levels={len(rec.levels)}")


def main() -> None:
    construction, system = build_system()

    print("[RUN] GLOBAL mesh convergence, bearing-to-bearing "
          "(study_kind='global', domain='bearing_to_bearing', v_res + M_res)")
    library_global = run_convergence(
        system, construction,
        beam_theory="timoshenko",
        shear_theory=SHEAR_THEORY,
        integration_method=INTEGRATION_METHOD,
        study_kind="global",
        domain="bearing_to_bearing",
        global_metrics=default_global_metrics(
            components=("res",),
            v_gci_threshold=V_GCI_THRESHOLD, v_safety_factor=V_SAFETY_FACTOR,
            M_gci_threshold=M_GCI_THRESHOLD, M_safety_factor=M_SAFETY_FACTOR,
        ),
        max_levels=MAX_LEVELS,
    )
    _summarize(library_global, system)

    # ------------------------------------------------------------------
    # extra_mandatory for the production solve: union of (a) whatever
    # node set the global study converged to for this shaft, and (b)
    # the manual AxisForge-vs-Abaqus divergence nodes above. (a) already
    # covers BOTH v and M together -- no separate union needed the way
    # the old per-interval version had to union two libraries.
    # ------------------------------------------------------------------
    extra_mandatory: dict[str, list[float]] = {}
    for ss in system.shafts:
        result = library_global.get_or_none(ss.name)
        converged_nodes = set(result.all_extra_nodes) if result is not None else set()
        manual_nodes = _manual_divergence_nodes(ss)
        nodes = sorted(converged_nodes | set(manual_nodes))
        extra_mandatory[ss.name] = nodes
        print(f"  [{ss.name}] total extra node(s) (global-converged union manual): "
              f"{len(nodes)} (manual={len(manual_nodes)})")

    ts_study = StudyCapabilities(construction=construction, shaft_fem=("shaft_fem.timoshenko_rigid",))
    ts_solve_system = ts_study.resolve()["solve_system"]

    ts_library = ts_solve_system(
        system, construction,
        shear_theory=SHEAR_THEORY,
        integration_method=INTEGRATION_METHOD,
        extra_mandatory=extra_mandatory,
    )

    out_dir = CASE_ROOT / "timoshenko" / "refined_mesh" / "results" / SHEAR_THEORY
    _write_theory_outputs(
        system, out_dir, ts_library,
        title=f"case_radial_single_position -- refined mesh (timoshenko/{SHEAR_THEORY})",
    )

    report_path = CASE_ROOT / "timoshenko" / "refined_mesh" / "report_mesh_convergence_global.txt"
    write_studies_report(
        system, report_path,
        title=f"case_radial_single_position -- refined mesh (timoshenko/{SHEAR_THEORY}) "
              f"-- GLOBAL bearing-to-bearing convergence (v_res + M_res)",
        convergence_library=library_global,
    )
    print(f"\n[OK] {report_path.name} written ({len(library_global)} shaft(s), "
          f"domain='bearing_to_bearing')")


if __name__ == "__main__":
    main()