"""Run of v102: the crossed-cylinder alpha sweep (Table 1) and the three Table 5 contacts.

``python -m studies.v100_dry_contact.v102_hertz_ellipses.study`` writes into ``results/`` (git-ignored):
``v102_bem.csv`` (bridge records), ``v102_check.csv`` and the figures of ``studies/outputs``. Spatial
kernel, interference control (J-7; the approach given is r02's at the nominal load), grids of v104's rule.
"""
from __future__ import annotations

from pathlib import Path

from axisforge.outputs._format import write_records_csv
from slippy_bridge.contact import Grid, hertz, solve

from studies.outputs.l100_dry_contact.plots import contact as plots
from studies.v100_dry_contact.v101_hertz_point import pressure_cuts, reference

from . import literature
from .check import check

RESULTS = Path(__file__).resolve().parent / "results"
B_OVER_H = (16.0, 32.0)  # minor semi-axis over spacing asked [-]: v104's rule (b/h >= 16) and one finer level
SPAN = 2.0               # domain half-length in semi-axes, both axes (spatial kernel: v104 D1)
CUT_CASE = "hamrock_brewe_t5_ball_outer_ring_contact"     # the pressure cut drawn (a ball in a concave groove)


def run(results=RESULTS, b_over_h=B_OVER_H) -> dict:
    """Solve every literature system on a principal grid (the ellipse on the grid lines) at each b/h."""
    bem, checks, cuts = [], [], []
    for case in literature.cases():
        contact = case.build()
        hz = hertz(contact)                                    # bridge Hertz: sizes the grid only
        delta0 = reference(contact, contact.load).delta_m
        for boh in b_over_h:
            grid = Grid.around(contact, hz, h=hz.b / boh, span_x=SPAN, span_y=SPAN, angle="principal")
            r = solve(contact, grid, results / "slippy", hertz_result=hz, use_frequency_domain=False,
                      interference=delta0)
            alpha = contact.kappa_max / contact.kappa_min
            bem.append({"literature_case": case.name, "alpha": alpha, **r.record()})
            checks += [{"literature_case": case.name, "alpha": alpha, "b_over_h_asked": boh, **row}
                       for row in check(r, published=case.published)]
            if case.name == CUT_CASE and boh == b_over_h[0]:
                cuts = pressure_cuts(r)
    write_records_csv(results / "v102_bem.csv", bem)
    write_records_csv(results / "v102_check.csv", checks)
    # relative errors only: the orientation (E2, an angle, exact to 1e-15 rad) is in the tables; k of the
    # circle (alpha 1) is 1 by symmetry
    sweep = [{**r, "series": f"crossed cylinders, b/h {r['b_over_h_asked']:g} asked"} for r in checks
             if "_t1_" in r["literature_case"] and (r["id"] in ("C1", "C3") or r["quantity"] == "b"
                                                     or (r["id"] == "E1" and r["alpha"] > 1.0 + 1e-9))]
    figures = [plots.relative_errors(sweep, results, "v102", x="alpha", x_label=r"$\alpha = R_y/R_x$"),
               plots.pressure_cut(cuts, results, "v102", variant="ball_outer_ring")]
    return {"bem": bem, "check": checks, "figures": figures}


if __name__ == "__main__":
    from axisforge.outputs._format import format_table
    out = run()
    print(format_table((("literature_case", "case", "s"), ("b_over_h", "b/h", ".1f"), ("id", "id", "s"),
                        ("quantity", "quantity", "s"), ("measured", "BEM", ".6g"), ("reference", "ref", ".6g"),
                        ("rel_err", "err [%]", "+.4f", 100.0), ("tol", "tol [%]", ".3f", 100.0),
                        ("verdict", "verdict", "s")), out["check"]))
