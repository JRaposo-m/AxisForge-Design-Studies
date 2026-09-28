"""
axisforge_bridge/construction/outputs/fem_report.py

Report block for shaft FEM results -- the output of
RigidSupportFEMSolver.solve() (axisforge.solvers.machine_elements.shaft.
fem_solvers.global_solver.rigid_support), i.e. a ShaftResults instance.

fem_results_block(result) is a summary view, not a dump of ShaftResults:
it reports the envelope (max bending moment/deflection/bending stress/
shear stress/twist, each with its own location) plus per-bearing
reactions (ShaftResults.bearing_nodes) -- exactly the quantities an
engineer checks first. Every field it reads is already fully
post-processed by build_shaft_result(); this module does no computation
of its own.

CONTENT ONLY -- this module builds a text block, it does not write any
file. text_report.py is the single place that assembles every domain's
blocks into the one combined construction report .txt and writes it.

Dependency (core only, read-only access):
  axisforge.results.fem_results.shaft_results
      ShaftResults, BearingNodeData
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from axisforge.results.fem_results.shaft_results import ShaftResults


def fem_results_block(result: "ShaftResults") -> str:
    """
    ASCII block: node count, bending/deflection/stress/twist envelope
    (value + location), then one line per bearing node with its
    resolved reactions (Fr/Fa/moments).
    """
    lines = [
        f"  FEM results ({len(result.x_nodes)} nodes, beam theory via "
        f"solver settings):",
        f"    M_max         : {result.M_max:.2f} N*mm   @ x={result.x_M_max:.2f} mm",
        f"    v_max (defl.) : {result.v_max:.4f} mm     @ x={result.x_v_max:.2f} mm",
        f"    sigma_b_max   : {result.sigma_b_max:.2f} MPa  @ x={result.x_sigma_b_max:.2f} mm",
        f"    tau_max       : {result.tau_max:.2f} MPa  @ x={result.x_tau_max:.2f} mm",
        f"    phi_max (twist): {result.phi_max:.6f} rad  @ x={result.x_phi_max:.2f} mm",
    ]

    if result.bearing_nodes:
        lines.append(f"    bearing reactions ({len(result.bearing_nodes)}):")
        for bn in result.bearing_nodes:
            lines.append(
                f"      [{bn.label}] x={bn.position:.2f} mm  "
                f"Fr={bn.Fr:.2f} N  Fa={bn.Fa:.2f} N  "
                f"M_xz={bn.M_xz:.2f} N*mm  M_xy={bn.M_xy:.2f} N*mm"
            )
    else:
        lines.append("    bearing reactions: (none)")

    return "\n".join(lines)
