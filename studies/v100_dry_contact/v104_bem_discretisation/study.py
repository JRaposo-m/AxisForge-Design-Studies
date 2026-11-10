"""Run of v104: h, domain and kernel sweeps on two smooth contacts, and the grid rule they give.

``python -m studies.v100_dry_contact.v104_bem_discretisation.study`` writes into ``results/``
(git-ignored): ``v104_check.csv`` (one discretisation record per solve), ``v104_judged.csv``
(D1, L1-L3), ``v104_grid_rule.csv`` and the figures of ``studies/outputs``.
Load control throughout (the nominal 500 N): the errors of approach and p0 are read at a fixed load.
"""
from __future__ import annotations

from pathlib import Path

from axisforge.outputs._format import write_records_csv
from slippy_bridge.contact import Grid, hertz, solve

from studies.outputs.l100_dry_contact.plots import contact as plots

from . import literature
from .check import check, judge
from .construction import B_OVER_H, B_OVER_H_SPAN_SWEEP, KERNELS, SPANS, systems

RESULTS = Path(__file__).resolve().parent / "results"
RULE_TOLS = (5e-3, 1e-3)     # stated errors of the grid rule, |err_delta| and |err_p0| [-]


def _solve(contact, hz, b_over_h, span, freq, results):
    grid = Grid.around(contact, hz, h=hz.b / b_over_h, span_x=span, span_y=span)
    return solve(contact, grid, results / "slippy", hertz_result=hz, use_frequency_domain=freq)


def run(results=RESULTS, b_over_h=B_OVER_H, spans=SPANS) -> dict:
    """Every sweep on both systems and both kernels; write and return the records."""
    records = []
    for contact in systems():
        hz = hertz(contact)                                    # bridge Hertz: sizes the grids only
        for kernel, (freq, span_h) in KERNELS.items():
            for boh in b_over_h:
                records.append({"sweep": "h", **check(_solve(contact, hz, boh, span_h, freq, results))})
            for span in spans:
                records.append({"sweep": "span",
                                **check(_solve(contact, hz, B_OVER_H_SPAN_SWEEP, span, freq, results))})
            if contact.label == "ball_on_flat":
                records.append({"sweep": "muser", **check(_solve(contact, hz, literature.MUSER_B_OVER_H,
                                                                 literature.MUSER_SPAN[kernel], freq, results))})
    judged = judge(records)
    rule = _grid_rule(records)
    write_records_csv(results / "v104_check.csv", records)
    write_records_csv(results / "v104_judged.csv", judged)
    write_records_csv(results / "v104_grid_rule.csv", rule)
    figures = []
    for sweep, x, label in (("h", "b_over_h", r"$b/h$"), ("span", "span_x", r"span / semi-axis")):
        rows = [{"series": f"{r['kernel']}, {r['label']}", x: r[x], "quantity": q.removeprefix("err_"),
                 "rel_err": r[q]} for r in records if r["sweep"] == sweep for q in ("err_delta", "err_p0", "err_b")]
        figures.append(plots.relative_errors(rows, results, "v104", x=x, x_label=label, variant=sweep))
    return {"check": records, "judged": judged, "grid_rule": rule, "figures": figures}


def _grid_rule(records):
    """The coarsest b/h of the h sweep from which |err_delta| and |err_p0| stay within each stated error."""
    out = []
    for label in dict.fromkeys(r["label"] for r in records):
        for kernel in KERNELS:
            sweep = sorted((r for r in records if r["sweep"] == "h" and r["label"] == label and r["kernel"] == kernel),
                           key=lambda r: r["b_over_h"])
            for tol in RULE_TOLS:
                for q in ("err_delta", "err_p0"):
                    ok = [r for i, r in enumerate(sweep) if all(abs(s[q]) <= tol for s in sweep[i:])]
                    out.append({"label": label, "kernel": kernel, "quantity": q.removeprefix("err_"), "tol": tol,
                                "span_x": sweep[0]["span_x"],
                                "b_over_h_min": ok[0]["b_over_h"] if ok else float("nan"),
                                "reached": bool(ok)})
    return out


if __name__ == "__main__":
    from axisforge.outputs._format import format_table
    out = run()
    print(format_table((("label", "contact", "s"), ("kernel", "kernel", "s"), ("quantity", "q", "s"),
                        ("tol", "tol [%]", ".2f", 100.0), ("b_over_h_min", "b/h min", ".2f"),
                        ("reached", "reached", "s")), out["grid_rule"]))
