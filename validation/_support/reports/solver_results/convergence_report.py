"""
validation/_support/reports/solver_results/convergence_report.py

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from axisforge.results.convergence_results.convergence_results import (
        ConvergenceRecord,
        MeshRefinementResult,
    )

__all__ = [
    "levels_history_table",
    "gci_detail_table",
    "interval_block",
    "shaft_convergence_block",
    "write_convergence_report",
]


# ---------------------------------------------------------------------------
# Table helper -- duplicated by convention, see this module's own docstring.
# ---------------------------------------------------------------------------

def _table(headers: list[str], rows: list[list[str]]) -> str:
    """Fixed-width ASCII table -- one column per header, left-aligned."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def _fmt_row(cells: list[str]) -> str:
        return "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells))

    lines = [_fmt_row(headers), _fmt_row(["-" * w for w in widths])]
    lines.extend(_fmt_row(row) for row in rows)
    return "\n".join(lines)


def _fmt_gci_pct(value: float) -> str:
    """value is a fraction (0.01 = 1%%), formatted as a percentage string,
    SIGNED (e_m_c/e_f_m carry a meaningful sign -- direction of change
    between levels -- and it is not stripped here). NaN (a _DummyGCI's
    GCI_f_m -- non-uniform refinement ratio, degenerate order, or a
    metric landing exactly on 0.0 -- or a field _DummyGCI does not have
    at all) prints as 'n/a', never as the literal string 'nan'.

    6 decimal places (not 3) -- widened specifically so a percentage
    like 0.000012%% doesn't round down to the same 0.000%% as a genuine
    zero; see gci_detail_table()'s own docstring for why telling those
    two apart matters when p looks suspicious."""
    if value != value:  # NaN check without importing math for one use
        return "n/a"
    return f"{value * 100:.6f}%"


def _fmt_num(value: float, fmt: str = ".4f") -> str:
    """Plain numeric formatter (r, p, f_h0 -- NOT percentages) with the
    same NaN-safety as _fmt_gci_pct, for fields _DummyGCI does not
    carry at all (getattr(..., float('nan')) upstream)."""
    if value != value:
        return "n/a"
    return format(value, fmt)


def _metric_names(rec: "ConvergenceRecord") -> list[str]:
    """
    Ordered list of tracked metric names for this record -- e.g.
    ["v_xz", "v_xy", "v_res"] for a displacement run, ["M_xz", "M_xy",
    "M_res"] for a moment run, or whatever future MetricSpec list a
    caller passes. MeshConvergenceStudy runs the exact same self._metrics
    list at every grade level, so every entry in point_metrics_history
    shares the same key set -- reading it off the FIRST level is
    enough, dict insertion order (Python 3.7+) keeps it stable and
    matches the order the caller's `metrics` list was built in.

    Falls back to gci_history's keys if point_metrics_history is
    somehow empty (interval never solved a single level -- e.g. it
    raised before the first grade completed). Returns [] only if
    neither has anything, which the callers below already handle by
    producing an empty table/no lines, same as any other "(none)" case
    in this module.
    """
    if rec.point_metrics_history:
        return list(rec.point_metrics_history[0].keys())
    if rec.gci_history:
        return list(rec.gci_history[0].keys())
    return []


def _unit(name: str) -> str:
    """Unit of a metric from its name ("v_xz:max" -> "mm", "M_xy:max" ->
    "N.mm", "V_xz:max" -> "N", "theta_xy:max" -> "rad"). Unknown -> "-"."""
    field = name.split(":", 1)[0]
    if field.startswith("theta"):
        return "rad"
    head = field[:1]
    return {"v": "mm", "u": "mm", "M": "N.mm", "V": "N"}.get(head, "-")


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------

def levels_history_table(rec: "ConvergenceRecord") -> str:
    """
    Per-level metric history for one interval: every TRACKED metric's
    value at every grade tried (one value column per name from
    _metric_names()), plus that metric's GCI_f_m (fine-vs-medium, see
    this module's own top docstring for why not GCI_m_c) once 3+
    levels exist.

    rec.gci_history[i] is the GCI dict computed from the triple of
    levels ending at rec.levels[i + 2] (Richardson needs 3 consecutive
    levels, and MeshConvergenceStudy._converge_one_load() only appends
    to gci_history once len(history) >= 3) -- so a level's GCI columns
    stay "--" for its first two entries, never a KeyError/IndexError.
    """
    names = _metric_names(rec)
    headers = (
        ["level"]
        + [f"{name} [{_unit(name)}]" for name in names]
        + [f"GCI_{name}" for name in names]
    )
    rows: list[list[str]] = []
    n_gci = len(rec.gci_history)

    for i, (level, metrics) in enumerate(zip(rec.levels, rec.point_metrics_history)):
        # 10 decimal places -- a diagnostic widening: two levels that
        # print as bit-for-bit identical at 5 decimals can still differ
        # starting at the 8th/9th digit, and that residual difference
        # is exactly what feeds RichardsonGCI.p -- rounding it away
        # here would hide the very thing gci_detail_table()'s raw
        # delta_m_c/delta_f_m columns are meant to let you inspect.
        value_cells = [f"{metrics[name]:.10f}" for name in names]

        gci_index = i - 2
        if 0 <= gci_index < n_gci:
            gci = rec.gci_history[gci_index]
            gci_cells = [_fmt_gci_pct(gci[name].GCI_f_m) for name in names]
        else:
            gci_cells = ["--"] * len(names)

        rows.append([level] + value_cells + gci_cells)

    return _table(headers, rows)


def gci_detail_table(rec: "ConvergenceRecord") -> str:
    """
    Full RichardsonGCI detail, one row per (level-transition, tracked
    metric) pair, in the same chronological order levels_history_table()
    walks. Surfaces everything RichardsonGCI computes that
    levels_history_table() has no room for -- r (refinement ratio), p
    (OBSERVED ORDER OF CONVERGENCE), delta_m_c/delta_f_m (RAW, signed
    differences in mm -- read straight off rec.point_metrics_history,
    not derived from the relative e_m_c/e_f_m), e_m_c/e_f_m (signed
    RELATIVE error, medium-vs-coarse and fine-vs-medium), GCI_m_c (the
    one levels_history_table() omits), and f_h0 (the
    Richardson-extrapolated estimate of the converged value itself) --
    rather than duplicating GCI_f_m%, which levels_history_table()
    already tables per level.

    The "metric" column (was "plane" before this module became
    metric-name-generic) holds whatever _metric_names() returns for
    this record -- "v_xz"/"v_xy"/"v_res", "M_xz"/"M_xy"/"M_res", or any
    future MetricSpec.name.

    delta_m_c/delta_f_m are printed in scientific notation specifically
    to make a diagnostic question easy to answer at a glance: is the
    actual change between two mesh levels (this column) meaningfully
    larger than floating-point/solver noise, or is p being computed
    from two numbers that are both already indistinguishable from zero
    at that precision? p is a ratio of logs of these two raw deltas
    (see RichardsonGCI.p's own formula) -- when both deltas are tiny
    and of comparable magnitude to solver round-off, p becomes
    numerically unstable and can land anywhere without that meaning
    genuine super-convergence. A p noticeably far from the FEM
    element's theoretical order is worth treating as suspect until
    delta_m_c/delta_f_m here confirm the underlying signal was actually
    large enough to trust. This is also the column to check first for a
    moment interval that keeps reporting NOT CONVERGED near a zero
    crossing: a small delta with a large e_m_c/e_f_m (relative error
    computed against a near-zero f_coarse/f_medium) is exactly the
    signature of that limitation, not a real meshing problem.

    A _DummyGCI entry (non-uniform refinement ratio, a degenerate
    observed order -- e.g. p ~= 0 or a non-monotonic level-to-level
    change -- or a metric landing exactly on 0.0; see
    convergence_solver.py's own RichardsonGCI/_DummyGCI docstrings)
    only ever carries GCI_m_c/GCI_f_m/converged; r/p/e_m_c/e_f_m/f_h0
    print as 'n/a' via getattr(..., float("nan")) rather than raising
    AttributeError. delta_m_c/delta_f_m are still computed either way
    -- they come from point_metrics_history, not from the GCI object.
    """
    names = _metric_names(rec)
    headers = ["transition", "metric", "unit", "r", "p",
               "delta_m_c", "delta_f_m",
               "e_m_c", "e_f_m", "GCI_m_c", "GCI_f_m", "f_h0", "converged"]
    rows: list[list[str]] = []

    for i, gci_dict in enumerate(rec.gci_history):
        lvl_c, lvl_m, lvl_f = rec.levels[i], rec.levels[i + 1], rec.levels[i + 2]
        transition = f"{lvl_c}->{lvl_m}->{lvl_f}"
        m_c = rec.point_metrics_history[i]
        m_m = rec.point_metrics_history[i + 1]
        m_f = rec.point_metrics_history[i + 2]
        for name in names:
            delta_m_c = m_m[name] - m_c[name]
            delta_f_m = m_f[name] - m_m[name]

            gci = gci_dict[name]
            r = getattr(gci, "r", float("nan"))
            p = getattr(gci, "p", float("nan"))
            e_m_c = getattr(gci, "e_m_c", float("nan"))
            e_f_m = getattr(gci, "e_f_m", float("nan"))
            f_h0 = getattr(gci, "f_h0", float("nan"))
            rows.append([
                transition, name, _unit(name),
                _fmt_num(r, ".6f"), _fmt_num(p, ".6f"),
                f"{delta_m_c:.3e}", f"{delta_f_m:.3e}",
                _fmt_gci_pct(e_m_c), _fmt_gci_pct(e_f_m),
                _fmt_gci_pct(gci.GCI_m_c), _fmt_gci_pct(gci.GCI_f_m),
                _fmt_num(f_h0, ".10f"), str(gci.converged),
            ])

    return _table(headers, rows)


def interval_block(rec: "ConvergenceRecord") -> str:
    """
    One interval's full block: header line, final-mesh line, the
    per-level metric/GCI_f_m history table, the full Richardson-detail
    table (r, p, e_m_c, e_f_m, GCI_m_c, f_h0 per transition per tracked
    metric), and -- once at least one GCI has been computed -- one
    Richardson-extrapolated final-estimate line PER tracked metric
    (f_h0, from the LAST/finest transition available, with its own
    observed order p). Previously this always assumed a "_res" plane
    and printed exactly one such line; it now prints one per name in
    _metric_names(), so a moment run reports M_xz/M_xy/M_res estimates
    instead of silently reporting nothing (or crashing on a "res" key
    that a differently-named metrics list never had).

    ## CHANGED (this pass, confirmed with erg 2026-09-17): the
    ## `if not rec.applicable: return ...` short-circuit that used to
    ## sit at the top of this function is REMOVED -- ConvergenceRecord
    ## no longer has an `applicable`/`skip_reason` field (see
    ## convergence_results.py's own CHANGED note). Every record reaching
    ## this function went through a real solve attempt, so it always
    ## renders the full CONVERGED/NOT CONVERGED block below.
    """
    if rec.converged:
        status = "CONVERGED"
    else:
        status = f"NOT CONVERGED (stopped after {len(rec.levels)} level(s))"

    lines = [
        f"  Interval '{rec.label}'  [{rec.x_lo:.3f}, {rec.x_hi:.3f}] mm  -- {status}",
    ]
    if rec.x_final:
        lines.append(f"    final mesh: {len(rec.x_final)} node(s)")
    else:
        lines.append("    final mesh: (none -- no level was solved)")

    lines.append("")
    lines.append(levels_history_table(rec))

    if rec.gci_history:
        lines.append("")
        lines.append("  Richardson GCI detail (per transition, per metric):")
        lines.append(gci_detail_table(rec))

        last = rec.gci_history[-1]
        lines.append("")
        for name in _metric_names(rec):
            gci = last[name]
            p = getattr(gci, "p", float("nan"))
            f_h0 = getattr(gci, "f_h0", float("nan"))
            lines.append(
                f"  Richardson-extrapolated {name} (p={_fmt_num(p, '.6f')}): "
                f"{_fmt_num(f_h0, '.10f')} {_unit(name)}"
            )

    return "\n".join(lines)


def shaft_convergence_block(result: "MeshRefinementResult") -> str:
    """
    Full convergence block for one shaft: a one-line summary followed
    by one interval_block() per entry in result.per_load (insertion
    order), then the union of all_extra_nodes ready for
    Mesh1D(..., extra_mandatory=...).

    An empty result.per_load (e.g. run_convergence() was called with
    regions={} on a shaft, or the shaft had no gear/distributed-load
    interval to begin with) reports "0/0 interval(s)" rather than
    printing nothing -- fail-loud/visible, same convention as every
    other block module's "(none)"/"(missing from: ...)" lines.

    ## CHANGED (this pass, confirmed with erg 2026-09-17): the
    ## skipped-interval count (n_skipped/skipped_suffix, keyed off
    ## rec.applicable) is REMOVED -- there is no more "skipped, no load"
    ## outcome (see convergence_results.py's/convergence_solver.py's own
    ## CHANGED notes). The summary line is back to a plain
    ## "N/M interval(s) converged".
    """
    n_total = len(result.per_load)
    n_converged = sum(1 for rec in result.per_load.values() if rec.converged)

    lines = [
        f"Shaft '{result.shaft_name}' -- mesh convergence: "
        f"{n_converged}/{n_total} interval(s) converged",
        "",
    ]

    for rec in result.per_load.values():
        lines.append(interval_block(rec))
        lines.append("")

    extra_nodes = result.all_extra_nodes
    if extra_nodes:
        nodes_str = ", ".join(f"{x:.3f}" for x in extra_nodes)
        lines.append(f"  Extra mandatory nodes (union, for Mesh1D): [{nodes_str}]")
    else:
        lines.append("  Extra mandatory nodes (union, for Mesh1D): (none)")

    return "\n".join(lines).rstrip("\n") + "\n"


def write_convergence_report(
    results: "dict[str, MeshRefinementResult]",
    path: "str | Path",
    title: str = "",
    header_lines: "tuple[str, ...] | list[str]" = (),
) -> str:
    """
    Write ONE .txt with one shaft_convergence_block() per shaft, in the
    dict's order. `header_lines` are printed under the title (e.g. the GCI
    thresholds / safety factors / domain used, which the result shapes do
    not carry). Returns the written text.
    """
    rule = "=" * 88
    lines = [rule, (title or "Mesh convergence report").center(88), rule]
    lines += list(header_lines)
    lines.append("")
    for shaft_name, res in results.items():
        lines.append(shaft_convergence_block(res))
    text = "\n".join(lines).rstrip("\n") + "\n"
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return text