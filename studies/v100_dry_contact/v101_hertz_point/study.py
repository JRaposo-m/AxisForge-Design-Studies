"""Run of v101: the steel ball on a flat through the BEM, three grids, two kernels, two load controls.

``python -m studies.v100_dry_contact.v101_hertz_point.study`` writes into ``results/`` (git-ignored):
``v101_bem.csv`` (one bridge record per solve), ``v101_check.csv`` (check records, Johnson's
constants included) and the figures of ``studies/outputs``.
"""
from __future__ import annotations

from pathlib import Path

from axisforge.outputs._format import write_records_csv
from slippy_bridge.contact import Grid, hertz, solve

from studies.outputs.l100_dry_contact.plots import contact as plots

from . import literature
from .check import check, pressure_cuts, reference
from .construction import A_OVER_H

RESULTS = Path(__file__).resolve().parent / "results"
# kernel -> (use_frequency_domain, domain half-length in contact radii on both axes): the spatial kernel has
# no images (2 is enough, v104_bem_discretisation); the frequency kernel at the bridge's SPAN_X_0 = 4
KERNELS = {"spatial": (False, 2.0), "frequency": (True, 4.0)}


def run(results=RESULTS, a_over_h=A_OVER_H) -> dict:
    """Solve every grid x kernel x control on Johnson's system; write and return the records.

    Interference control is the primary run (J-7): the approach given is r02's at the nominal
    load; load control is the cross-check.
    """
    (case,) = literature.cases()
    contact = case.build()
    hz = hertz(contact)                                  # bridge Hertz: sizes the grid only
    delta0 = reference(contact, contact.load).delta_m    # r02 approach at the nominal load [m]
    bem, checks, solved = [], [], {}
    for label, (freq, span) in KERNELS.items():
        for aoh in a_over_h:
            grid = Grid.around(contact, hz, h=hz.a / aoh, span_x=span, span_y=span)
            for control, kwargs in (("interference", {"interference": delta0}), ("normal_load", {})):
                r = solve(contact, grid, results / "slippy", hertz_result=hz, use_frequency_domain=freq,
                          **kwargs)
                bem.append({"kernel": label, "a_over_h_asked": aoh, **r.record()})
                checks += check(r)
                solved[(label, aoh, control)] = r
    write_records_csv(results / "v101_bem.csv", bem)
    write_records_csv(results / "v101_check.csv", checks)
    judged = [aoh for aoh in a_over_h if aoh >= 16.0][0]          # the coarsest judged grid
    figures = [plots.relative_errors([{**r, "series": f"{r['kernel']}, {r['load_control']}"} for r in checks
                                      if r["id"] in ("C1", "C2", "C3") or r["quantity"] == "b"],
                                     results, "v101", x="a_over_h", x_label=r"$a/h$"),
               plots.pressure_cut(pressure_cuts(solved[("spatial", judged, "interference")]), results,
                                  "v101", variant=f"spatial_ah{judged:g}_asked")]
    return {"bem": bem, "check": checks, "figures": figures}


if __name__ == "__main__":
    from axisforge.outputs._format import format_table
    out = run()
    print(format_table((("kernel", "kernel", "s"), ("load_control", "control", "s"), ("a_over_h", "a/h", ".2f"),
                        ("id", "id", "s"), ("quantity", "quantity", "s"), ("rel_err", "rel err [%]", "+.4f", 100.0),
                        ("tol", "tol [%]", ".3f", 100.0), ("verdict", "verdict", "s")), out["check"]))
