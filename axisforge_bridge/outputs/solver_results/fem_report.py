"""
axisforge_bridge/outputs/solver_results/fem_report.py

Report blocks + writer for shaft FEM results -- the output of
RigidSupportFEMSolver.solve() (axisforge.solvers.machine_elements.shaft.
fem_solvers.global_solver.rigid_support), i.e. a ShaftResults instance.

Two levels, both CONTENT built from an already-solved ShaftResults (this
module does no computation of its own -- every field it reads is already
post-processed by build_shaft_result()):

  fem_results_block(result)   -- SHORT summary (envelope + bearing
                                 reactions). Kept unchanged: this is what
                                 construction/text_report.write_construction_report
                                 embeds when given fem_results=.
  shaft_result_block(result)  -- FULL per-shaft block: nodes, governing
                                 values, and the per-node / per-bearing
                                 tables (see below). Ported from the old
                                 fixtures studies text_report, extended
                                 with u / theta_xz / theta_xy columns.
  write_fem_report(...)       -- the one function here that writes a file:
                                 a "SHAFT_FEM" report with one full block
                                 per shaft, in the system's own shaft order.

Full block layout (per shaft):
  summary            M_max, v_max, sigma_b_max, tau_max (+ x of each)
  BENDING & SHEAR    x, M_xz, M_xy, M [N.mm]; V_xz, V_xy, V [N]
  DEFLECTION & TORSION  x, u, v_xz, v_xy, v [mm]; theta_xz, theta_xy [rad]; T [N.m]
  SECTION & STRESS   x, d [mm]; W, Wt [mm^3]; sigma_b, tau [MPa]
  BEARING REACTIONS  from the reaction-vector arrays (bearing_positions,
                     R_xz, R_xy, R, R_axial), labelled by index-aligned
                     lookup into bearing_nodes
  BEARING NODE DATA  two tables: displacements & rotations, and loads
                     (from the TOTAL force vector -- a different array
                     from the reaction table above, so kept separate)

Deliberate formatting choices (unchanged from the old writer):
  - per-node arrays are split into three tables, not one wide one;
  - units follow ShaftResults exactly, including the discontinuity:
    M is N.mm, T is N.m -- the headers say so, nothing is normalised;
  - the raw solver arrays (K, d_total_*, f_*_*, dof lists) and
    torsion_contributions are NOT printed.
phi_max / phi (twist) is available on ShaftResults but is not part of the
reference layout, so it is not printed here (fem_results_block does).

Writes with encoding="utf-8" explicitly. Owns its own RULE constant and its
own write() call (no shared IO helper, on purpose).

Dependency (results only, read-only):
  axisforge.results.fem_results.shaft_results
      ShaftResults, BearingNodeData (TYPE_CHECKING only)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

RULE = "=" * 72
SUB = "-" * 72

if TYPE_CHECKING:  # pragma: no cover
    from axisforge.core.mechanical_system.parallel_axis.spur_helical.gear_system import (
        SpurHelicalGearSystem,
    )
    from axisforge.results.fem_results.shaft_results import ShaftResults


# ---------------------------------------------------------------------------
# Short summary (embedded by write_construction_report)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Table formatting -- small internal helper, no third-party dependency.
# ---------------------------------------------------------------------------

def _table(headers: list[str], rows: list[list[str]]) -> str:
    """
    Fixed-width ASCII table: `headers` and every row are already-formatted
    strings (caller controls precision/units per column); this only pads
    each column to its widest cell, joins with 2-space gutters and adds a
    rule line under the header. Returns "  (no data)" if `rows` is empty.
    """
    if not rows:
        return "  (no data)"
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows))
        for i in range(len(headers))
    ]

    def fmt_row(cells: list[str]) -> str:
        return "  " + "  ".join(c.rjust(w) for c, w in zip(cells, widths))

    lines = [fmt_row(headers), "  " + "  ".join("-" * w for w in widths)]
    lines += [fmt_row(r) for r in rows]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Full content blocks
# ---------------------------------------------------------------------------

def summary_block(result: "ShaftResults") -> str:
    """Governing values + their axial location: M_max, v_max,
    sigma_b_max, tau_max."""
    return "\n".join([
        f"  M_max          : {result.M_max:12.2f} N.mm  @ x={result.x_M_max:8.2f} mm",
        f"  v_max          : {result.v_max:12.4f} mm    @ x={result.x_v_max:8.2f} mm",
        f"  sigma_b_max    : {result.sigma_b_max:12.2f} MPa   @ x={result.x_sigma_b_max:8.2f} mm",
        f"  tau_max        : {result.tau_max:12.2f} MPa   @ x={result.x_tau_max:8.2f} mm",
    ])


def bending_shear_table(result: "ShaftResults") -> str:
    """Per-node table: x, M_xz, M_xy, M [N.mm]; V_xz, V_xy, V [N]."""
    headers = ["x[mm]", "M_xz[N.mm]", "M_xy[N.mm]", "M[N.mm]",
               "V_xz[N]", "V_xy[N]", "V[N]"]
    rows = [
        [f"{x:.1f}", f"{mxz:.1f}", f"{mxy:.1f}", f"{m:.1f}",
         f"{vxz:.1f}", f"{vxy:.1f}", f"{v:.1f}"]
        for x, mxz, mxy, m, vxz, vxy, v in zip(
            result.x_nodes, result.M_xz, result.M_xy, result.M,
            result.V_xz, result.V_xy, result.V,
        )
    ]
    return _table(headers, rows)


def deflection_torsion_table(result: "ShaftResults") -> str:
    """Per-node table: u, v_xz, v_xy, v [mm]; theta_xz, theta_xy [rad];
    T [N.m] -- T's unit differs from the bending-moment table's N.mm,
    per ShaftResults' own docstring."""
    headers = ["x[mm]", "u[mm]", "v_xz[mm]", "v_xy[mm]", "v[mm]",
               "theta_xz[rad]", "theta_xy[rad]", "T[N.m]"]
    rows = [
        [f"{x:.1f}", f"{u:.4f}", f"{vxz:.4f}", f"{vxy:.4f}", f"{v:.4f}",
         f"{txz:.6f}", f"{txy:.6f}", f"{t:.3f}"]
        for x, u, vxz, vxy, v, txz, txy, t in zip(
            result.x_nodes, result.u, result.v_xz, result.v_xy, result.v,
            result.theta_xz, result.theta_xy, result.T,
        )
    ]
    return _table(headers, rows)


def section_stress_table(result: "ShaftResults") -> str:
    """Per-node table: d [mm]; W, Wt [mm^3]; sigma_b, tau [MPa]."""
    headers = ["x[mm]", "d[mm]", "W[mm^3]", "Wt[mm^3]",
               "sigma_b[MPa]", "tau[MPa]"]
    rows = [
        [f"{x:.1f}", f"{d:.2f}", f"{w:.1f}", f"{wt:.1f}",
         f"{sb:.2f}", f"{t:.2f}"]
        for x, d, w, wt, sb, t in zip(
            result.x_nodes, result.d, result.W, result.Wt,
            result.sigma_b, result.tau,
        )
    ]
    return _table(headers, rows)


def bearing_reactions_table(result: "ShaftResults") -> str:
    """
    Per-bearing table from the REACTION-vector-derived summary arrays
    (bearing_positions, R_xz, R_xy, R, R_axial), labelled by reading
    bearing_nodes[i].label at the same index (both are built from the
    same bearing loop, so they are index-aligned).
    """
    labels = [n.label for n in result.bearing_nodes]
    headers = ["label", "position[mm]", "R_xz[N]", "R_xy[N]", "R[N]", "R_axial[N]"]
    rows = []
    for i, pos in enumerate(result.bearing_positions):
        label = labels[i] if i < len(labels) else f"(bearing {i})"
        rows.append([
            label, f"{pos:.2f}",
            f"{result.R_xz[i]:.2f}", f"{result.R_xy[i]:.2f}",
            f"{result.R[i]:.2f}", f"{result.R_axial[i]:.2f}",
        ])
    return _table(headers, rows)


def bearing_node_kinematics_table(result: "ShaftResults") -> str:
    """Per-bearing table: position, displacements (u, v_xz, v_xy), shaft
    slope at the node (theta_xz, theta_xy) and seat misalignment angle
    (psi_xz, psi_xy)."""
    headers = ["label", "position[mm]", "u[mm]", "v_xz[mm]", "v_xy[mm]",
               "theta_xz[rad]", "theta_xy[rad]", "psi_xz[rad]", "psi_xy[rad]"]
    rows = [
        [n.label, f"{n.position:.2f}", f"{n.u:.4f}", f"{n.v_xz:.4f}", f"{n.v_xy:.4f}",
         f"{n.theta_xz:.6f}", f"{n.theta_xy:.6f}", f"{n.psi_xz:.6f}", f"{n.psi_xy:.6f}"]
        for n in result.bearing_nodes
    ]
    return _table(headers, rows)


def bearing_node_loads_table(result: "ShaftResults") -> str:
    """Per-bearing table: radial/axial force and bending moment at the
    bearing node, from the TOTAL force vector (not the reaction vector
    used by bearing_reactions_table -- numerically close, but a different
    computed quantity)."""
    headers = ["label", "position[mm]", "Fr_xz[N]", "Fr_xy[N]", "Fr[N]",
               "Fa[N]", "M_xz[N.mm]", "M_xy[N.mm]"]
    rows = [
        [n.label, f"{n.position:.2f}", f"{n.Fr_xz:.2f}", f"{n.Fr_xy:.2f}",
         f"{n.Fr:.2f}", f"{n.Fa:.2f}", f"{n.M_xz:.2f}", f"{n.M_xy:.2f}"]
        for n in result.bearing_nodes
    ]
    return _table(headers, rows)


def shaft_result_block(result: "ShaftResults") -> str:
    """Every block above, in order, for one shaft: summary, bending &
    shear, deflection & torsion, section & stress, bearing reactions,
    bearing node data."""
    sections = [
        f"  nodes          : {len(result.x_nodes)}",
        "",
        summary_block(result),
        "",
        "BENDING & SHEAR",
        SUB,
        bending_shear_table(result),
        "",
        "DEFLECTION & TORSION",
        SUB,
        deflection_torsion_table(result),
        "",
        "SECTION & STRESS",
        SUB,
        section_stress_table(result),
        "",
        "BEARING REACTIONS",
        SUB,
        bearing_reactions_table(result),
        "",
    ]
    if result.bearing_nodes:
        sections.append(f"BEARING NODE DATA ({len(result.bearing_nodes)}) -- DISPLACEMENTS & ROTATIONS")
        sections.append(SUB)
        sections.append(bearing_node_kinematics_table(result))
        sections.append("")
        sections.append(f"BEARING NODE DATA ({len(result.bearing_nodes)}) -- LOADS (total force vector)")
        sections.append(SUB)
        sections.append(bearing_node_loads_table(result))
    else:
        sections.append("BEARING NODE DATA: (none)")
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def write_fem_report(
    system: "SpurHelicalGearSystem",
    results: "dict[str, ShaftResults]",
    path: "str | Path",
    title: str = "",
) -> str:
    """
    Write ONE .txt: a SHAFT_FEM section with one "SHAFT: <name>" block per
    shaft in system.shafts (the system's own order), each holding
    shaft_result_block() for results[name], or "(no result)" when that
    shaft has no entry. Returns the written text.

    Never solves anything: `results` is the {ShaftSystem.name: ShaftResults}
    mapping the caller already produced with RigidSupportFEMSolver.

    A relative `path` resolves against the process's working directory --
    build an absolute one at the call site
    (Path(__file__).resolve().parent / "report.txt") to land next to the
    calling script.
    """
    header_title = title or system.label or "FEM report"
    sections = [RULE, header_title.center(72), RULE, "",
                RULE, "SHAFT_FEM", RULE]

    for ss in system.shafts:
        sections.append(RULE)
        sections.append(f"SHAFT: {ss.name}")
        sections.append(RULE)

        result = results.get(ss.name)
        if result is None:
            sections.append("  (no result -- not solved)")
        else:
            sections.append(shaft_result_block(result))
        sections.append("")

    sections.append(RULE)
    text = "\n".join(sections) + "\n"
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text