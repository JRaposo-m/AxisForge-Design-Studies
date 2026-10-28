"""Run of v105: the bridge's ball-raceway contacts of solved bearings against r02 Hertz and Hamrock 1983.

``python -m studies.v100_dry_contact.v105_ball_contacts.study`` writes into ``results/`` (git-ignored):
``v105_check.csv`` (C1-C5 per contact, S1-S2 per element), ``v105_literature.csv`` (Hamrock) and the figures of
``studies/outputs``. No BEM solve (OPEN R5-81: one loaded ball-groove contact through the bridge's static solve
is s101's question).
"""
from __future__ import annotations

import math
from pathlib import Path

from axisforge.outputs._format import write_records_csv
from slippy_bridge.contact.core.machine_elements.bearings import DeepGrooveBallContact
from slippy_bridge.contact.solvers.hertz import hertz as bridge_hertz

from studies.outputs.l000_axisforge.plots.bearing_loads import load_zone_sweep
from studies.outputs.l100_dry_contact.plots.contact import relative_errors
# OPEN R5-80: the one-bearing AxisForge solve of v003
from studies.v000_axisforge.v003_iso16281_load_zone.study import solve_bearing

# OPEN R5-80 (as R2-02): the printed-digit tolerance of v102, imported (a third copy in the repo otherwise)
from studies.v100_dry_contact.v102_hertz_ellipses.check import half_unit
from studies.v000_axisforge.v003_iso16281_load_zone.check import MM

from . import literature
from .check import check, series_records
from .construction import load_cases

RESULTS = Path(__file__).resolve().parent / "results"
FIGURE_CASE = "6210 s 0 Fr 1000 N"          # the load case of the around-the-bearing figure


def contacts_of(case: dict):
    """Build the bearing, solve it with AxisForge (``psi = 0``), and let the bridge build every loaded contact."""
    bearing = case["build"]()
    result = solve_bearing(bearing, case["Fr_N"], case["Fa_N"])
    return bearing, result, case["contact"].from_bearing(bearing, result, elements="loaded")


def contact_series() -> list[dict]:
    """C0-C5 and S1-S2 of every load case."""
    rows = []
    for case in load_cases():
        bearing, result, contacts = contacts_of(case)
        recs = (check(contacts, bearing, result, label=case["name"])
                + series_records(contacts, bearing, label=case["name"]))
        rows += [{**r, "Fr_N": case["Fr_N"], "Fa_N": case["Fa_N"], "role": case["role"]} for r in recs]
    return rows


def literature_series() -> list[dict]:
    """Hamrock sec. 14.2: the bridge's relative radii (L1-L4, judged to half a unit), curvature sums and radius
    ratios (L5-L8, RECORD), the exact k of the bridge's Hertz at alpha = 0 against Hamrock's ``alpha^(2/pi)`` fit
    (L9, L10, RECORD) and the bridge's compressions at F_max against Hamrock's fitted ones (L11, L12, RECORD)."""
    (case,) = literature.cases()
    bearing = case.build()
    p = literature.published()
    F_max = float(p["F_max"]["value"])
    contacts = DeepGrooveBallContact.from_bearing(bearing, load=F_max, alpha=0.0)   # Fa = 0: every alpha_j = 0
    out = []

    def row(lid, quantity, unit, measured, symbol, judged):
        printed = p[symbol]["value"]
        ref, tol = float(printed), half_unit(printed)          # relative half unit of the last printed digit
        err = measured / ref - 1.0
        out.append({"case": "v105", "source": case.name, "id": lid, "quantity": quantity, "unit": unit,
                    "measured": measured, "reference": ref, "rel_err": err, "tol": tol,
                    "verdict": ("PASS" if abs(err) <= tol else "MISS") if judged else "RECORD",
                    "note": p[symbol]["note"]})

    for c in contacts:
        s = "i" if c.race == "inner" else "o"
        assert c.psi == 0.0, "the minor axis must lie along rolling to read R_x = 1/kappa_max"
        R_x, R_y = 1.0 / c.kappa_max, 1.0 / c.kappa_min          # the minor axis is along rolling (psi = 0)
        h = bridge_hertz(c)                                       # exact Hertz at Hamrock's geometry (alpha = 0)
        n = 1 if s == "i" else 2
        row(f"L{n}", f"R_x {c.race} (bridge)", "m", R_x, f"R_x_{s}", True)
        row(f"L{n + 2}", f"R_y {c.race} (bridge)", "m", R_y, f"R_y_{s}", True)
        row(f"L{n + 4}", f"1/R_x + 1/R_y {c.race}", "1/m", 1.0 / R_x + 1.0 / R_y, f"inv_R_{s}", False)
        row(f"L{n + 6}", f"R_y/R_x {c.race}", "-", R_y / R_x, f"alpha_r_{s}", False)
        row(f"L{n + 8}", f"k = a/b {c.race} (bridge Hertz, exact) vs Hamrock's alpha^(2/pi) fit", "-", h.a / h.b,
            f"k_{s}", False)
        row(f"L{n + 10}", f"delta {c.race} at F_max (bridge Hertz) vs Hamrock's fits", "m", h.delta, f"delta_{s}",
            False)
    return sorted(out, key=lambda r: int(r["id"][1:]))


def figures(rows) -> list[Path]:
    """Per-ball a, p0 around the bearing (bridge open, r02 lines); C1-C4 error vs r02's size ratio; the two ratios."""
    out = []
    pts = []
    for r in rows:
        if r["label"] != FIGURE_CASE or r["id"] not in ("C1", "C3"):
            continue
        scale, quantity = (1.0 / MM, "a") if r["id"] == "C1" else (1e-6, "p0")
        for source, value in (("bridge", r["measured"]), ("r02", r["reference"])):
            pts.append({"phi_deg": r["phi_deg"], "quantity": quantity, "y": value * scale, "series": r["race"],
                        "source": source})
    out.append(load_zone_sweep(pts, RESULTS, "v105", x="phi_deg", x_label=r"local azimuth $\varphi$ [deg]",
                               panels={"a": r"$a$ [mm]", "p0": r"$p_0$ [MPa]"}, open_sources=("bridge",),
                               variant="around_6210",
                               caption=f"{FIGURE_CASE}: bridge contacts (open) on r02 (lines); local azimuth (0 on the "
                                       "load line), no convention tag"))
    # the most loaded element of every load case of the sweep: the error against r02's size ratio (flat: the
    # transfer does not depend on the size), tolerance dashed
    err = [{"size_ratio_r02": r["size_ratio_r02"], "quantity": r["quantity"], "series": r["race"],
            "rel_err": r["rel_err"],
            "tol": r["tol"], "verdict": r["verdict"]}
           for r in rows if r["role"] == "size sweep" and r["j"] == 0 and r["id"] in ("C1", "C2", "C3", "C4")]
    out.append(relative_errors(err, RESULTS, "v105", x="size_ratio_r02", x_label=r"r02 size ratio $a / r_1$ [-]",
                               variant="conformity"))
    # validity against the load: r02's per-body ratio (a / r1, the ball) and the bridge's size ratio on the
    # relative radii (SB-07), most loaded ball; the 0.1 policy line (OPEN R5-88)
    val = []
    for r in rows:
        if r["role"] == "size sweep" and r["j"] == 0 and r["id"] == "C5":
            val += [{"Fr_N": r["Fr_N"], "quantity": "size", "y": r["size_ratio_r02"], "series": r["race"],
                     "source": "r02 a/r1 (per body)"},
                    {"Fr_N": r["Fr_N"], "quantity": "size", "y": r["bridge_size_ratio"], "series": r["race"],
                     "source": "bridge size ratio (relative radii, SB-07)"}]
    out.append(load_zone_sweep(val, RESULTS, "v105", x="Fr_N", x_label=r"radial load $F_r$ [N]",
                               panels={"size": "size ratio [-]"}, limits={"size": 0.1}, log_x=True,
                               open_sources=("bridge size ratio (relative radii, SB-07)",), variant="validity",
                               caption="6210, zero clearance, most loaded ball; 0.1 = P-02 policy (OPEN R5-88), "
                                       "evidence on SB-07"))
    return out


def run() -> dict:
    """Every series, records written to ``results/``, figures drawn; returns the records and figure paths."""
    rows = contact_series()
    lit = literature_series()
    RESULTS.mkdir(exist_ok=True)
    write_records_csv(RESULTS / "v105_check.csv", rows)
    write_records_csv(RESULTS / "v105_literature.csv", lit)
    return {"check": rows, "literature": lit, "figures": figures(rows)}


if __name__ == "__main__":
    out = run()
    judged = [r for r in out["check"] + out["literature"] if r["verdict"] != "RECORD"]
    worst = max((abs(r["rel_err"]) for r in judged), default=math.nan)
    print(f"v105: {sum(r['verdict'] == 'PASS' for r in judged)} PASS, {sum(r['verdict'] == 'MISS' for r in judged)} "
          f"MISS of {len(judged)} judged; worst |rel_err| {worst:.2e}; "
          f"{sum(r['verdict'] == 'RECORD' for r in out['check'])} RECORD rows")
