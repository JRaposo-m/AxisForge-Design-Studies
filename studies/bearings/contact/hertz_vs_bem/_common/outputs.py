"""Measurements and result files of one ``hertz_vs_bem`` run (one build of one type).

SlipPy itself returns the solution (``model.solve()``: ``loads_z``, ``interference``,
``converged``, ``total_normal_load``); this module does not store those fields again. It turns
them into the numbers the question compares with Hertz and writes them to ``<type>/<build>/data/``::

    results.csv   one row per grid resolution (source of truth, columns = ``FIELDS``)
    meta.json     the case (``ContactCase.to_dict``), its hash, the build and the columns
    report.txt    the same rows as a readable table

``compare`` reads these files only; it refuses to compare builds whose ``case_hash`` differs.
No SlipPy import here: the module works on plain arrays and dicts.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

__all__ = ["FIELDS", "pressure_from_loads", "measure", "make_row", "report_lines", "write_results",
           "read_results"]

# "_ref" is the analytic Hertz value (hertz_full), "_num" is measured on the SlipPy solution.
# a is the larger and b the smaller contact semi-axis, so the comparison does not depend on how
# x and y are oriented in the SlipPy arrays. Units in brackets. Edit here and every build follows.
FIELDS = [
    "case_hash", "label", "build",
    "n", "h_m", "extent_m", "nodes_in_2b", "apex_node_offset",
    "a_ref_m", "b_ref_m", "p0_ref_Pa", "delta_ref_m",
    "a_num_m", "b_num_m", "p0_num_Pa", "delta_num_m", "load_num_N",
    "err_a", "err_b", "err_p0", "err_delta", "err_load",
    "converged", "solve_time_s",
]


def pressure_from_loads(loads_z, h: float, total_load: float, rtol: float = 1e-3) -> np.ndarray:
    """Return the pressure field [Pa] from SlipPy's ``loads_z``, whichever unit it is in.

    The unit of ``loads_z`` is not assumed: if its sum equals ``total_load`` [N] it is a force per
    node and the pressure is ``loads_z / h**2``; if ``sum * h**2`` equals ``total_load`` it already
    is a pressure.

    Parameters
    ----------
    loads_z : numpy.ndarray
        ``loads_z`` from the solution.
    h : float
        Grid spacing [m] (square cells).
    total_load : float
        Load the solution carries [N] (``total_normal_load``).
    rtol : float, optional
        Relative tolerance of the identification.

    Raises
    ------
    ValueError
        If neither interpretation reproduces ``total_load``.
    """
    loads = np.asarray(loads_z, dtype=float)
    if math.isclose(loads.sum(), total_load, rel_tol=rtol):
        return loads / h**2
    if math.isclose(loads.sum() * h**2, total_load, rel_tol=rtol):
        return loads
    raise ValueError(f"loads_z identified neither as force (sum {loads.sum():.6g}) nor as "
                     f"pressure (sum*h^2 {loads.sum() * h**2:.6g}); total load {total_load:.6g} N")


def measure(p: np.ndarray, h: float, line: bool = False) -> dict:
    """Measure the contact on a pressure field.

    Parameters
    ----------
    p : numpy.ndarray
        Pressure [Pa] on the square grid.
    h : float
        Grid spacing [m].

    Returns
    -------
    dict
        ``a`` and ``b`` [m] (larger and smaller half-extent of the loaded nodes, one node of
        resolution), ``p0`` [Pa] (maximum pressure), ``load`` [N] and ``apex_offset`` (distance
        in nodes between the pressure maximum and the array centre).
    """
    mask = p > 1e-6 * p.max()
    rows, cols = np.where(mask)
    half_r = (rows.max() - rows.min() + 1) * h / 2.0
    half_c = (cols.max() - cols.min() + 1) * h / 2.0
    if line:
        # a strip: the loaded nodes span the whole array along the line axis; a is the other half-extent
        span_r = len(np.unique(rows)) == p.shape[0]
        span_c = len(np.unique(cols)) == p.shape[1]
        if span_r == span_c:
            raise ValueError("line contact: the loaded strip does not span exactly one array axis")
        half = [half_c if span_r else half_r, float("inf")]
    else:
        half = sorted([half_r, half_c], reverse=True)
    i, j = np.unravel_index(np.argmax(p), p.shape)
    centre = (p.shape[0] / 2.0, p.shape[1] / 2.0)
    return {"a": half[0], "b": half[1], "p0": float(p.max()), "load": float(p.sum() * h**2),
            "apex_offset": float(abs(i - centre[0]) if line and span_c else
                                 abs(j - centre[1]) if line else math.hypot(i - centre[0], j - centre[1]))}


def _err(num: float, ref: float) -> float:
    return float("nan") if ref is None or not math.isfinite(ref) or ref == 0 else (num - ref) / ref


def make_row(case, build: str, n: int, h: float, extent: float, ref: dict, solution: dict,
             solve_time_s: float, load_total: float | None = None) -> dict:
    """Build one result row from the Hertz reference and the SlipPy solution.

    Parameters
    ----------
    case : ContactCase
        The contact.
    build : str
        Name of the build.
    n : int
        Nodes per side.
    h, extent : float
        Grid spacing and side of the square domain [m].
    ref : dict
        ``hertz_full(**case.hertz_args())``; ``contact_radii``, ``max_pressure`` and (if present)
        ``total_deflection`` are read.
    solution : dict
        ``model.solve()``: ``loads_z``, ``interference``, ``converged``, ``total_normal_load``.
    solve_time_s : float
        Wall time of ``solve()`` [s].
    load_total : float, optional
        Total load the solution must carry [N]; defaults to ``case.load``. For a line contact give
        the load per unit length times the length of the periodic domain.

    Returns
    -------
    dict
        One row with the keys of ``FIELDS``.
    """
    radii = sorted((float(r) for r in ref["contact_radii"]), reverse=True)
    finite = [r for r in radii if math.isfinite(r)]
    a_ref = finite[0]                               # line contact: the half-width
    b_ref = finite[1] if len(finite) > 1 else float("inf")
    minor = b_ref if math.isfinite(b_ref) else a_ref
    p0_ref = float(ref["max_pressure"])
    delta_ref = float(ref.get("total_deflection", float("nan")))
    p = pressure_from_loads(solution["loads_z"], h, float(solution["total_normal_load"]))
    m = measure(p, h, line=case.line)
    delta = float(solution["interference"])
    return {
        "case_hash": case.case_hash(), "label": case.label, "build": build,
        "n": n, "h_m": h, "extent_m": extent,
        "nodes_in_2b": 2.0 * minor / h,
        "apex_node_offset": m["apex_offset"],
        "a_ref_m": a_ref, "b_ref_m": b_ref, "p0_ref_Pa": p0_ref, "delta_ref_m": delta_ref,
        "a_num_m": m["a"], "b_num_m": m["b"], "p0_num_Pa": m["p0"], "delta_num_m": delta,
        "load_num_N": m["load"],
        "err_a": _err(m["a"], a_ref), "err_b": _err(m["b"], b_ref),
        "err_p0": _err(m["p0"], p0_ref), "err_delta": _err(delta, delta_ref),
        "err_load": _err(m["load"], case.load if load_total is None else load_total),
        "converged": bool(solution["converged"]), "solve_time_s": solve_time_s,
    }


def report_lines(case, build: str, rows: list[dict]) -> list[str]:
    """Return the lines of ``report.txt``: Hertz reference, numerical results, relative errors.

    The Hertz values do not depend on the grid, so they are printed once. Lengths are shown in
    mm (contact semi-axes, domain) or um (grid spacing, approach), pressures in MPa.
    """
    r0 = rows[0]
    out = [f"{case.label} / {build}   case_hash {case.case_hash()}", ""]
    out += ["HERTZ (analytic, hertz_full)",
            f"  a = {r0['a_ref_m'] * 1e3:.5f} mm   b = {r0['b_ref_m'] * 1e3:.5f} mm   "
            f"p0 = {r0['p0_ref_Pa'] * 1e-6:.2f} MPa   delta = {r0['delta_ref_m'] * 1e6:.4f} um   "
            f"load = {case.load:g} {'N/m' if case.line else 'N'}", ""]

    def table(title, header, cells):
        out.append(title)
        out.append("  " + "  ".join(f"{h:>11}" for h in header))
        for r in rows:
            out.append("  " + "  ".join(f"{c:>11}" for c in cells(r)))
        out.append("")

    table("GRID", ["n", "h [um]", "extent [mm]", "nodes in 2b", "apex off."],
          lambda r: [f"{int(r['n'])}", f"{r['h_m'] * 1e6:.3f}", f"{r['extent_m'] * 1e3:.4f}",
                     f"{r['nodes_in_2b']:.1f}", f"{r['apex_node_offset']:.1f}"])
    table("NUMERICAL (SlipPy)", ["n", "a [mm]", "b [mm]", "p0 [MPa]", "delta [um]", "load [N]",
                                 "converged", "time [s]"],
          lambda r: [f"{int(r['n'])}", f"{r['a_num_m'] * 1e3:.5f}", f"{r['b_num_m'] * 1e3:.5f}",
                     f"{r['p0_num_Pa'] * 1e-6:.2f}", f"{r['delta_num_m'] * 1e6:.4f}",
                     f"{r['load_num_N']:.4f}", str(bool(r['converged'])), f"{r['solve_time_s']:.2f}"])
    table("RELATIVE ERROR vs HERTZ [%]  (numerical - Hertz) / Hertz",
          ["n", "a", "b", "p0", "delta", "load"],
          lambda r: [f"{int(r['n'])}"] + [f"{r[k] * 100:+.3f}" for k in
                                           ("err_a", "err_b", "err_p0", "err_delta", "err_load")])
    out.append("a, b: half-extent of the loaded nodes (resolution = one node); a >= b.")
    return out


def write_results(out_dir, case, build: str, rows: list[dict]) -> None:
    """Write ``results.csv``, ``meta.json`` and ``report.txt`` to ``out_dir`` (created if missing).

    Raises
    ------
    ValueError
        If a row misses a column of ``FIELDS`` or carries another ``case_hash``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for r in rows:
        missing = [f for f in FIELDS if f not in r]
        if missing:
            raise ValueError(f"row misses columns: {missing}")
        if r["case_hash"] != case.case_hash():
            raise ValueError("row case_hash differs from the case being stored")
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    meta = {"case": case.to_dict(), "case_hash": case.case_hash(), "build": build, "fields": FIELDS}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    lines = report_lines(case, build, rows)
    (out / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_results(out_dir) -> tuple[dict, list[dict]]:
    """Read what :func:`write_results` wrote: ``(meta, rows)`` with numeric columns as float."""
    out = Path(out_dir)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    with open(out / "results.csv", newline="", encoding="utf-8") as fh:
        rows = [{k: _num(v) for k, v in r.items()} for r in csv.DictReader(fh)]
    return meta, rows


def _fmt(x) -> str:
    return f"{x:.4e}" if isinstance(x, float) else str(x)


def _num(v: str):
    try:
        return float(v)
    except ValueError:
        return v
