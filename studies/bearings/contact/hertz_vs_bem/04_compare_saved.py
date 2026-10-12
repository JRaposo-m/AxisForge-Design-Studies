"""04 (temporary): the new results against the ``data/results.csv`` saved before the bridge.

Reads the five old ``<type>/<build>/data/results.csv`` and the three new ``results.csv`` in
``results/``; solves nothing. It exists only so that the old ``data/`` folders can be deleted
knowingly, and goes with them.

The grids differ (old: square, non-dyadic spacing, half-width 6 or 3 semi-axes, contact edge
measured to the node; new: the bridge grid, dyadic spacing, edge measured below one node), so only
two things are compared, per type and build:

* the Hertz reference (``p0``, ``a``, ``b``, ``delta``): it must be the same contact, so any
  difference is a change of SlipPY's ``hertz_full`` between the two runs;
* the finest mesh of each run: the measured values and their errors against Hertz. The approach
  of the line contact is left out of the comparison: it has no reference value (plane strain) and
  the value SlipPY returns grows with the period of the domain.

Old columns: ``a`` is the larger semi-axis and ``b`` the smaller (``inf`` for the line contact);
the new records have ``s_x`` (rolling) and ``s_y`` and are reordered here. Errors are fractions in
both. The old ``ball_flat/reference`` is the composite build.

Run after 01, 02 and 03: ``python 04_compare_saved.py``.
"""
import math
from pathlib import Path

from axisforge.outputs._format import format_table, read_records_csv

from slippy_bridge.contact.outputs import rel_err

from _common.case import RESULTS

HERE = Path(__file__).resolve().parent
OLD = (("ball_flat", "reference"), ("ball_raceway", "composite"), ("ball_raceway", "two_profiles"),
       ("roller_raceway", "composite"), ("roller_raceway", "two_profiles"))
TEXT = ("case_hash", "label", "build", "sweep", "element", "subtype", "race")
BOOL = ("converged", "line", "load_ok", "edge_clear", "margin_ok", "resolution_ok", "within_reference")
QUANTITIES = (("p0 [Pa]", "p0_ref_Pa", "p0_num_Pa", "err_p0"), ("a [m]", "a_ref_m", "a_num_m", "err_a"),
              ("b [m]", "b_ref_m", "b_num_m", "err_b"), ("delta [m]", "delta_ref_m", "delta_num_m", "err_delta"))


def in_old_columns(r: dict) -> dict:
    """One new record with the column names of the old files (``a`` >= ``b``, ``b`` = inf for a line)."""
    y_longer = math.isfinite(r["s_y_ref_m"]) and r["s_y_ref_m"] > r["s_x_ref_m"]
    a, b = ("s_y", "s_x") if y_longer else ("s_x", "s_y")
    return {"p0_ref_Pa": r["p0_ref_Pa"], "p0_num_Pa": r["p0_Pa"], "err_p0": r["err_p0"],
            "a_ref_m": r[f"{a}_ref_m"], "a_num_m": r[f"{a}_m"], "err_a": r[f"err_{a}"],
            "b_ref_m": r[f"{b}_ref_m"], "b_num_m": r[f"{b}_m"], "err_b": r[f"err_{b}"],
            "delta_ref_m": r["delta_ref_m"], "delta_num_m": r["delta_m"], "err_delta": r["err_delta"]}


def main() -> None:
    values, grids = [], []
    for kind, old_build in OLD:
        build = "composite" if old_build == "reference" else old_build
        old = read_records_csv(HERE / kind / old_build / "data" / "results.csv",
                               text_fields=TEXT, bool_fields=BOOL)
        new = read_records_csv(RESULTS / kind / "results.csv", text_fields=TEXT, bool_fields=BOOL)
        # the mesh sweep runs on the study constants; the AxisForge run has its own hash
        new = [r for r in new if r["build"] == build and r["sweep"] == "mesh"]
        hashes = {r["case_hash"] for r in old} | {r["case_hash"] for r in new}
        if len(hashes) != 1:
            raise ValueError(f"{kind}/{build}: not the same contact, case_hash {sorted(hashes)}")

        o = max(old, key=lambda r: r["n"])                           # finest mesh of each run
        n = min(new, key=lambda r: r["h_m"])
        grids.append({"type": kind, "build": build, "h_old_m": o["h_m"], "n_old": o["n"],
                      "span_old": o["extent_m"] / (2.0 * o["a_ref_m"]), "h_new_m": n["h_m"],
                      "n_x": n["n_x"], "n_y": n["n_y"], "span_x": n["span_x"]})
        n = in_old_columns(n)
        for name, ref, num, err in QUANTITIES:
            values.append({"type": kind, "build": build, "quantity": name,
                           "hertz_old": o[ref], "hertz_new": n[ref], "d_hertz": rel_err(n[ref], o[ref]),
                           "bem_old": o[num], "bem_new": n[num],
                           # nan reference: the approach of a line contact, see the module text
                           "d_bem": "not defined" if math.isnan(o[ref]) else rel_err(n[num], o[num]),
                           "err_old": o[err], "err_new": n[err]})

    head = (("type", "type", "s"), ("build", "build", "s"))
    text = "\n".join([
        "Finest grid of each run (span = half-length of the domain in semi-axes: old in a, new in s_x)",
        format_table(head + (("h_old_m", "h old [um]", ".3f", 1e6), ("n_old", "n old", ".0f"),
                             ("span_old", "span old", ".1f"), ("h_new_m", "h new [um]", ".3f", 1e6),
                             ("n_x", "n_x new", ".0f"), ("n_y", "n_y new", ".0f"),
                             ("span_x", "span_x new", ".1f")), grids),
        "",
        "Hertz reference and finest-mesh values; d = (new - old) / old; err = (BEM - Hertz) / Hertz",
        format_table(head + (("quantity", "quantity", "s"), ("hertz_old", "Hertz old", ".9e"),
                             ("hertz_new", "Hertz new", ".9e"), ("d_hertz", "d Hertz", "+.1e"),
                             ("bem_old", "BEM old", ".6e"), ("bem_new", "BEM new", ".6e"),
                             ("d_bem", "d BEM [%]", "+.3f", 100.0), ("err_old", "err old [%]", "+.3f", 100.0),
                             ("err_new", "err new [%]", "+.3f", 100.0)), values)])
    (RESULTS / "compare_saved.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
