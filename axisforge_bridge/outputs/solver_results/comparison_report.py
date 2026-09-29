"""
axisforge_bridge/outputs/solver_results/comparison_report.py

Report writer for a TWO-SOLVE comparison of the same SpurHelicalGearSystem
-- e.g. Timoshenko vs Euler-Bernoulli, cowper vs hutchinson, base mesh vs
refined mesh. Label-agnostic: it never assumes which two configurations
produced the two sets of results, only that they belong to the same
`system` (same shaft names).

"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

RULE = "=" * 88
SUB = "-" * 88

_X_TOL = 1e-6   # mm -- two nodes are "the same node" if closer than this

if TYPE_CHECKING:  # pragma: no cover
    from axisforge.core.mechanical_system.parallel_axis.spur_helical.gear_system import (
        SpurHelicalGearSystem,
    )
    from axisforge.results.fem_results.shaft_results import ShaftResults


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _table(headers: list[str], rows: list[list[str]]) -> str:
    """Fixed-width ASCII table: right-aligned, 2-space gutters, rule line
    under the header. "(no data)" if `rows` is empty."""
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


def _delta_pct(a: float, b: float, atol: float = 1e-6) -> float:
    """
    % change of b relative to a, guarded against division by near-zero:
    M/V/Fa at a free end or an undriven bearing DOF is physically zero but
    comes back as FEM noise (~1e-13). If BOTH |a| and |b| are within `atol`
    they are the same zero and this returns 0.0 without a ratio. Past that
    guard, a == 0.0 with |b| > atol is a real zero-to-nonzero jump and
    returns inf (unmeasurable in relative terms -- not faked as 0 %).
    """
    if abs(a) <= atol and abs(b) <= atol:
        return 0.0
    if a == 0.0:
        return 0.0 if b == 0.0 else float("inf")
    return (b - a) / abs(a) * 100.0


def _pair_nodes(
    result_a: "ShaftResults",
    result_b: "ShaftResults",
) -> list[tuple[float, int, int]] | None:
    """
    (x, i_a, i_b) for every node of A that has a node of B at the same x
    (|dx| <= _X_TOL). None if there is no common node at all. Handles equal
    meshes (all nodes paired, same result as the old index zip), and
    unequal ones (only the shared nodes) -- see module docstring.
    """
    xb = list(result_b.x_nodes)
    pairs: list[tuple[float, int, int]] = []
    used: set[int] = set()
    for ia, x in enumerate(result_a.x_nodes):
        best = None
        for ib, x2 in enumerate(xb):
            if ib in used:
                continue
            if abs(x - x2) <= _X_TOL and (best is None or abs(x - x2) < abs(x - xb[best])):
                best = ib
        if best is not None:
            used.add(best)
            pairs.append((x, ia, best))
    return pairs or None


def _no_common_nodes(label_a: str, label_b: str, result_a, result_b) -> str:
    return (f"  (no common nodes -- {len(result_a.x_nodes)} nodes "
            f"({label_a}) vs {len(result_b.x_nodes)} nodes ({label_b}), skipped)")


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------

def comparison_table(result_a, result_b, label_a: str, label_b: str) -> str:
    """Governing scalars: sigma_b_max, v_max, then each bearing's Fr/Fa
    (bearing_nodes zipped index-for-index -- same system, same bearing
    order). Columns: quantity, value A, value B, delta (b - a), delta %."""
    headers = ["quantity", label_a, label_b, "delta", "delta %"]
    rows = [
        ["sigma_b_max [MPa]", f"{result_a.sigma_b_max:.4f}",
         f"{result_b.sigma_b_max:.4f}",
         f"{result_b.sigma_b_max - result_a.sigma_b_max:.4f}",
         f"{_delta_pct(result_a.sigma_b_max, result_b.sigma_b_max):.2f}%"],
        ["v_max [mm]", f"{result_a.v_max:.4f}", f"{result_b.v_max:.4f}",
         f"{result_b.v_max - result_a.v_max:.4f}",
         f"{_delta_pct(result_a.v_max, result_b.v_max):.2f}%"],
        ["M_max [N.mm]", f"{result_a.M_max:.2f}", f"{result_b.M_max:.2f}",
         f"{result_b.M_max - result_a.M_max:.2f}",
         f"{_delta_pct(result_a.M_max, result_b.M_max):.2f}%"],
    ]
    for na, nb in zip(result_a.bearing_nodes, result_b.bearing_nodes):
        rows.append([
            f"{na.label} Fr [N]", f"{na.Fr:.4f}", f"{nb.Fr:.4f}",
            f"{nb.Fr - na.Fr:.4f}", f"{_delta_pct(na.Fr, nb.Fr):.2f}%",
        ])
        rows.append([
            f"{na.label} Fa [N]", f"{na.Fa:.4f}", f"{nb.Fa:.4f}",
            f"{nb.Fa - na.Fa:.4f}", f"{_delta_pct(na.Fa, nb.Fa):.2f}%",
        ])
    return _table(headers, rows)


def bending_shear_comparison_table(result_a, result_b, label_a: str, label_b: str) -> str:
    """Per-node x, M[N.mm], V[N] (resultants) for both sides + delta %."""
    pairs = _pair_nodes(result_a, result_b)
    if pairs is None:
        return _no_common_nodes(label_a, label_b, result_a, result_b)
    headers = ["x[mm]", f"M[N.mm] ({label_a})", f"M[N.mm] ({label_b})", "M delta %",
               f"V[N] ({label_a})", f"V[N] ({label_b})", "V delta %"]
    rows = [
        [f"{x:.1f}", f"{result_a.M[ia]:.1f}", f"{result_b.M[ib]:.1f}",
         f"{_delta_pct(result_a.M[ia], result_b.M[ib]):.2f}%",
         f"{result_a.V[ia]:.1f}", f"{result_b.V[ib]:.1f}",
         f"{_delta_pct(result_a.V[ia], result_b.V[ib]):.2f}%"]
        for x, ia, ib in pairs
    ]
    return _table(headers, rows)


def deflection_comparison_table(result_a, result_b, label_a: str, label_b: str) -> str:
    """Per-node x, v[mm] (resultant) for both sides, delta, delta %."""
    pairs = _pair_nodes(result_a, result_b)
    if pairs is None:
        return _no_common_nodes(label_a, label_b, result_a, result_b)
    headers = ["x[mm]", f"v[mm] ({label_a})", f"v[mm] ({label_b})", "delta[mm]", "delta %"]
    rows = [
        [f"{x:.1f}", f"{result_a.v[ia]:.4f}", f"{result_b.v[ib]:.4f}",
         f"{result_b.v[ib] - result_a.v[ia]:.4f}",
         f"{_delta_pct(result_a.v[ia], result_b.v[ib]):.2f}%"]
        for x, ia, ib in pairs
    ]
    return _table(headers, rows)


def stress_comparison_table(result_a, result_b, label_a: str, label_b: str) -> str:
    """Per-node x, sigma_b[MPa] for both sides, delta, delta %. Any delta
    here should already be explained by the M delta above (same W(x))."""
    pairs = _pair_nodes(result_a, result_b)
    if pairs is None:
        return _no_common_nodes(label_a, label_b, result_a, result_b)
    headers = ["x[mm]", f"sigma_b[MPa] ({label_a})", f"sigma_b[MPa] ({label_b})",
               "delta[MPa]", "delta %"]
    rows = [
        [f"{x:.1f}", f"{result_a.sigma_b[ia]:.2f}", f"{result_b.sigma_b[ib]:.2f}",
         f"{result_b.sigma_b[ib] - result_a.sigma_b[ia]:.2f}",
         f"{_delta_pct(result_a.sigma_b[ia], result_b.sigma_b[ib]):.2f}%"]
        for x, ia, ib in pairs
    ]
    return _table(headers, rows)


def shaft_comparison_block(name: str, result_a, result_b, label_a: str, label_b: str) -> str:
    """One shaft's full comparison section: node counts and how many were
    paired (shown even when equal, so a mesh difference is visible right
    above the tables it affects), governing scalars, then the three
    per-node tables."""
    pairs = _pair_nodes(result_a, result_b)
    n_pairs = len(pairs) if pairs else 0
    return "\n".join([
        f"  nodes ({label_a}) : {len(result_a.x_nodes)}",
        f"  nodes ({label_b}) : {len(result_b.x_nodes)}",
        f"  nodes in common  : {n_pairs}  (per-node tables below use these only)",
        "",
        comparison_table(result_a, result_b, label_a, label_b),
        "",
        "BENDING & SHEAR (cross-validation -- expected to match closely between "
        "beam theories on the same mesh)",
        SUB,
        bending_shear_comparison_table(result_a, result_b, label_a, label_b),
        "",
        "DEFLECTION",
        SUB,
        deflection_comparison_table(result_a, result_b, label_a, label_b),
        "",
        "SECTION & STRESS",
        SUB,
        stress_comparison_table(result_a, result_b, label_a, label_b),
    ])


def write_comparison_report(
    results_a: "dict[str, ShaftResults]",
    results_b: "dict[str, ShaftResults]",
    system: "SpurHelicalGearSystem",
    path: "str | Path",
    label_a: str = "a",
    label_b: str = "b",
    title: str = "",
) -> str:
    """
    Write ONE .txt comparing two already-solved result sets for the SAME
    `system` -- one "SHAFT: <name>" section per shaft in system.shafts
    (system order), each holding shaft_comparison_block(), or "(missing
    from: ...)" naming whichever side has no result for that name.
    Returns the written text.

    results_a, results_b : {ShaftSystem.name: ShaftResults}, e.g. two
        solve_system() calls on the same `system`.
    path : .txt to write (parent created if missing).
    label_a, label_b : column headers (e.g. "timoshenko"/"euler_bernoulli").
    title : header title; defaults to "<system.label> -- comparison".
    """
    header_title = title or (
        f"{system.label} -- comparison" if system.label
        else "Resolution comparison report"
    )
    sections = [RULE, header_title.center(88), RULE, ""]

    for ss in system.shafts:
        sections.append(RULE)
        sections.append(f"SHAFT: {ss.name}")
        sections.append(RULE)

        result_a = results_a.get(ss.name)
        result_b = results_b.get(ss.name)
        if result_a is None or result_b is None:
            missing = []
            if result_a is None:
                missing.append(label_a)
            if result_b is None:
                missing.append(label_b)
            sections.append(f"  (missing from: {', '.join(missing)})")
        else:
            sections.append(shaft_comparison_block(ss.name, result_a, result_b, label_a, label_b))
        sections.append("")

    sections.append(RULE)
    text = "\n".join(sections) + "\n"
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text