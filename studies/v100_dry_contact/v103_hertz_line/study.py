"""Run of v103: the two rollers on a periodic strip, h and domain sweeps (frequency kernel, load control).

``python -m studies.v100_dry_contact.v103_hertz_line.study`` writes into ``results/`` (git-ignored):
``v103_bem.csv`` (bridge records), ``v103_check.csv`` and the figures of ``studies/outputs``. Load control
only: the approach of a line is not defined locally (an interference would be a number of the grid).
"""
from __future__ import annotations

from pathlib import Path

from axisforge.outputs._format import write_records_csv
from slippy_bridge.contact import Grid, hertz, solve

from studies.outputs.l100_dry_contact.plots import contact as plots
from studies.v100_dry_contact.v101_hertz_point import pressure_cuts

from . import literature
from .check import check

RESULTS = Path(__file__).resolve().parent / "results"
B_OVER_H = (8.0, 16.0, 32.0, 64.0)      # h sweep, half-width over spacing asked [-]
SPANS = (2.0, 4.0, 8.0, 16.0)           # domain sweep, half-length in half-widths [-]
SPAN_H, B_OVER_H_SPAN = 4.0, 16.0       # the fixed value of the other sweep
N_Y = 16                                # nodes along the periodic axis (y)


def run(results=RESULTS) -> dict:
    """Both sweeps; write and return the records."""
    (case,) = literature.cases()
    contact = case.build()
    hz = hertz(contact)                                         # bridge Hertz: sizes the grid only
    bem, checks, cuts = [], [], []
    for sweep, boh, span in [("h", b, SPAN_H) for b in B_OVER_H] + [("span", B_OVER_H_SPAN, s) for s in SPANS]:
        grid = Grid.around(contact, hz, h=hz.b / boh, span_x=span, n_y_line=N_Y)
        r = solve(contact, grid, results / "slippy", hertz_result=hz, use_frequency_domain=True)
        bem.append({"sweep": sweep, **r.record()})
        checks += [{"sweep": sweep, **row} for row in check(r)]
        if sweep == "h" and boh == B_OVER_H_SPAN:
            cuts = pressure_cuts(r)
    write_records_csv(results / "v103_bem.csv", bem)
    write_records_csv(results / "v103_check.csv", checks)
    figures = [plots.relative_errors([{**r, "series": f"{s} sweep"} for r in checks
                                      if r["id"] in ("C3", "C4") and r["sweep"] == s], results, "v103", x=x,
                                     x_label=label, variant=s)
               for s, x, label in (("h", "b_over_h", r"$b/h$"), ("span", "span_x", "span / half-width"))]
    figures.append(plots.pressure_cut(cuts, results, "v103", variant="bh16_asked"))
    return {"bem": bem, "check": checks, "figures": figures}


if __name__ == "__main__":
    from axisforge.outputs._format import format_table
    out = run()
    print(format_table((("sweep", "sweep", "s"), ("b_over_h", "b/h", ".1f"), ("span_x", "span", ".2f"),
                        ("id", "id", "s"), ("quantity", "q", "s"), ("measured", "BEM", ".6g"),
                        ("reference", "ref", ".6g"), ("rel_err", "err [%]", "+.4f", 100.0),
                        ("tol", "tol [%]", ".3f", 100.0), ("verdict", "verdict", "s")), out["check"]))
