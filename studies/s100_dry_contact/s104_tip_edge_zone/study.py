"""Run of s104: the C40 contacts near A and E at K9 on three grids, integral measures against Hertz, v107 / v103 checks.

``python -m studies.s100_dry_contact.s104_tip_edge_zone.study`` writes into ``results/`` (git-ignored):
``s104_points.csv`` (one row per point: d_end/b, measures on the finest grid, Hertz values, departure, Richardson
per measure, K1), ``s104_levels.csv`` (every grid level), ``s104_v103.csv`` (v103.check of the points far from the
edge), ``s104_v107.csv`` (v107.check of the pair at K9), ``s104_onset.csv`` (d_end/b at 1 % and 5 % departure)
and the figures of ``studies/outputs``.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from axisforge.outputs._format import write_records_csv
from slippy_bridge.contact import hertz

from studies.outputs.l000_axisforge.plots.bearing_loads import load_zone_sweep
from studies.outputs.l100_dry_contact.plots.contact import pressure_cut
from studies.v100_dry_contact.v103_hertz_line import check as v103_check
from studies.v100_dry_contact.v107_fzg_contact_path import check as v107_check

from .construction import TARGET_D_OVER_B, contact_at, contacts_near_end, edge_of, fzg_c40
from .model import METRICS, grid_series, hertz_measures

RESULTS = Path(__file__).resolve().parent / "results"
K1_TOL = 0.01         # [-] change of each dimensionless measure from h/2 to h/4 (proposals s104: < 1 %)
V103_FROM = 3.0       # [-] d_end/b from which a point is a plain line contact for v103 (its own conditions)
ONSET = (0.01, 0.05)  # departures from Hertz whose d_end/b marks the onset of the edge effect
EDGE_NAME = {"A": "A (tip of gear 2)", "E": "E (tip of gear 1)"}
CAPTION = ("FZG C40 K9 (wheel 323.27 N m), pinion driving, L0 share; BEM frequency kernel, periodic strip, span "
           "max(10 b, 2 x R2-08 rule at the actual M); "
           "sharp tip (SAG_PAST_END, SB-65)")


def departure(m: dict, ref: dict, *, centroid: bool = True, x_c_far: float = 0.0) -> float:
    """Largest departure of the measures from Hertz: ``|b_eq/b - 1|``, ``|edge share - Hertz|`` and, with
    ``centroid``, ``|x_c/b - x_c_far|``.

    * ``departure`` (README 6, fixed before the run): ``x_c_far = 0``;
    * ``departure_edge`` (added after the run, R5-116): without the centroid;
    * ``departure_baseline`` (added after the run, physics review): the centroid minus its far-field value at the
      same end (``d_end/b`` = 10), which removes the curvature-gradient offset and keeps the most sensitive measure.

    Far from the edge ``x_c/b`` stays at 0.008-0.011 near A and 0.002 near E: the contact lies 4.7-6.7 b of ARC
    LENGTH from gear 1's base circle, where rho1 = 4.3 mm varies by about 11 % across the contact
    (d rho1/ds = r_b1/rho1); that shifts the load away from gear 2's tip, the same sign as the edge shift.
    """
    d = max(abs(m["b_eq_over_b"] - ref["b_eq_over_b"]), abs(m["edge_share"] - ref["edge_share"]))
    return max(d, abs(m["x_c_over_b"] - ref["x_c_over_b"] - x_c_far)) if centroid else d


def point_series(system: dict):
    """Every point near A and E, and C: grid series, measures, Hertz values, v103 where it applies."""
    slippy = RESULTS / "slippy"
    points, levels, v103, cuts = [], [], [], []
    todo = [(end, t, g, c) for end in ("A", "E") for t, g, c in contacts_near_end(system, end)]
    todo.append(("C", math.inf, float(system["path"]["C"]), contact_at(system, float(system["path"]["C"]), "C")))
    for end, target, g_y, c in todo:
        hertz_ref = hertz(c)
        edge = edge_of(c, end)                # at C: the nearest end (23 b, outside the domain), its own side
        rows, summary, fine = grid_series(c, edge["x_edge_m"], slippy, label=c.label)
        levels += [{"end": end, "target_d_end_over_b": target, "point": c.label, **r} for r in rows]
        ref = hertz_measures(edge["x_edge_m"], hertz_ref.b)
        m = rows[-1]
        k1 = all(abs(summary[k]["change_fine_medium"]) <= K1_TOL for k in METRICS)
        rec = {"end": end, "point": c.label, "target_d_end_over_b": target, "g_y_mm": g_y, "edge": edge["edge"],
               "d_end_m": edge["d_end_m"], "d_end_over_b": edge["d_end_m"] / hertz_ref.b, "share": c.share,
               "w_N_per_m": c.load, "b_H_m": hertz_ref.b, "p0_H_Pa": hertz_ref.p0, "hertz_valid": hertz_ref.valid,
               **{k: m[k] for k in ("x_c_over_b", "b_eq_over_b", "p_m_over_p_m_H", "edge_share", "p0_over_p0_H",
                                    "singular", "d_end_nodes", "b_over_h")},
               **{f"hertz_{k}": ref[k] for k in ("x_c_over_b", "b_eq_over_b", "edge_share")},
               "departure": departure(m, ref), "departure_edge": departure(m, ref, centroid=False),
               "hertz_reason": hertz_ref.reason,
               "d_end_root_1_over_b": c.d_end_root_1_m / hertz_ref.b}
        for k in METRICS:
            s = summary[k]
            rec.update({f"{k}_status": s["status"], f"{k}_p": s["p"], f"{k}_extrapolated": s["extrapolated"],
                        f"{k}_gci": s["gci_fine"], f"{k}_change": s["change_fine_medium"]})
        rec.update({"K1_tol": K1_TOL, "K1": "PASS" if k1 else "MISS"})
        points.append(rec)
        if edge["d_end_m"] / hertz_ref.b >= V103_FROM:
            v103 += [{"end": end, "target_d_end_over_b": target, **r} for r in v103_check(fine, case="s104")]
        if end == "A" and target in (0.0, 0.5, 1.0, 3.0) or end == "C":
            x = np.asarray(fine.grid.X)[:, 0]
            p = np.asarray(fine.p).mean(axis=1)
            axis = "C (no edge)" if end == "C" else f"A, d_end/b = {target:g}"
            cuts += [{"axis": axis, "xi": float(xi), "p_over_p0_ref": float(pi)}
                     for xi, pi in zip(x / hertz_ref.b, p / hertz_ref.p0) if abs(xi) <= 1.6]
    for end in ("A", "E"):                 # departure_baseline: the centroid against its far-field value
        far = next(p for p in points if p["end"] == end and p["target_d_end_over_b"] == max(TARGET_D_OVER_B))
        for p in points:
            if p["end"] == end:
                m = {k: p[k] for k in ("b_eq_over_b", "edge_share", "x_c_over_b")}
                ref = {k: p[f"hertz_{k}"] for k in ("b_eq_over_b", "edge_share", "x_c_over_b")}
                p["departure_baseline"] = departure(m, ref, x_c_far=far["x_c_over_b"])
    return points, levels, v103, cuts


def onset(points: list[dict]) -> list[dict]:
    """d_end/b where the departure from Hertz first reaches 1 % and 5 %, coming from far (linear interpolation),
    for the three definitions of :func:`departure` (the second and third added after the run)."""
    out = []
    for end in ("A", "E"):
        rows = sorted((p for p in points if p["end"] == end), key=lambda p: -p["d_end_over_b"])
        for key in ("departure", "departure_edge", "departure_baseline"):
            for level in ONSET:
                hit = math.nan
                for far, near in zip(rows, rows[1:]):
                    if far[key] < level <= near[key]:
                        t = (level - far[key]) / (near[key] - far[key])
                        hit = far["d_end_over_b"] + t * (near["d_end_over_b"] - far["d_end_over_b"])
                        break
                b_mm = (next(p for p in rows if p["d_end_over_b"] <= hit)["b_H_m"] * 1e3 if math.isfinite(hit)
                        else math.nan)
                out.append({"end": end, "edge": rows[0]["edge"], "definition": key, "departure": level,
                            "d_end_over_b": hit, "d_end_mm": hit * b_mm})
    return out


def figures(points, levels, cuts) -> list[Path]:
    """Pressure next to the tip vs Hertz; measures vs d_end/b with the Hertz limit; the grid levels."""
    out = [pressure_cut(cuts, RESULTS, "s104", variant="near_A")]
    sweep = []
    for p in points:
        if p["end"] == "C":
            continue
        series = EDGE_NAME[p["end"]]
        for k in ("b_eq_over_b", "x_c_over_b", "edge_share"):
            sweep += [{"d_end_over_b": p["d_end_over_b"], "quantity": k, "y": p[k], "series": series, "source": "BEM"},
                      {"d_end_over_b": p["d_end_over_b"], "quantity": k, "y": p[f"hertz_{k}"], "series": series,
                       "source": "Hertz (no edge)"}]
    out.append(load_zone_sweep(sweep, RESULTS, "s104", x="d_end_over_b", x_label=r"distance to the tip $d_{end}/b$ (arc on the tip flank) [-]",
                               panels={"b_eq_over_b": r"$b_{eq}/b$ [-]", "x_c_over_b": r"$x_c/b$ [-]",
                                       "edge_share": "load within b of the edge [-]"},
                               limits={"b_eq_over_b": 1.0, "x_c_over_b": 0.0}, x_marks={"d_end = b": 1.0},
                               filled_sources=("BEM",), variant="measures",
                               caption=CAPTION + "; finest grid b/h = 64"))
    grid = []
    for lv in levels:
        if not (lv["end"] == "C" or (lv["end"] == "A" and lv["target_d_end_over_b"] in (0.0, 0.5, 3.0))):
            continue
        series = "C" if lv["end"] == "C" else f"A {lv['target_d_end_over_b']:g}"
        grid += [{"b_over_h": lv["b_over_h"], "quantity": k, "y": lv[k], "series": series, "source": "BEM"}
                 for k in (*METRICS, "p0_over_p0_H")]
    for p in points:
        if not (p["end"] == "C" or (p["end"] == "A" and p["target_d_end_over_b"] in (0.0, 0.5, 3.0))):
            continue
        series = "C" if p["end"] == "C" else f"A {p['target_d_end_over_b']:g}"
        span = [lv["b_over_h"] for lv in levels if lv["point"] == p["point"]] if "point" in levels[0] else []
        xs = (min(span), max(span)) if span else (16.0, 64.0)       # the series' own b/h (h is rounded dyadic)
        for k in METRICS:                                        # nan where Richardson found no order (oscillatory)
            grid += [{"b_over_h": x, "quantity": k, "y": p[f"{k}_extrapolated"], "series": series,
                      "source": "Richardson extrapolation"} for x in xs]
    out.append(load_zone_sweep(grid, RESULTS, "s104", x="b_over_h", x_label=r"Hertz half-width over spacing $b/h$ [-]",
                               panels={"b_eq_over_b": r"$b_{eq}/b$", "x_c_over_b": r"$x_c/b$",
                                       "edge_share": "edge share", "p0_over_p0_H": r"$p_0/p_{0,H}$ (singular: diverges)"},
                               log_x=True, filled_sources=("BEM",), variant="grid",
                               caption=CAPTION + "; h, h/2, h/4; line = Richardson extrapolation (max_p 2)"))
    return out


def run() -> dict:
    """Everything; records and figures written to ``results/`` (copied to the vault by hand, not by the case)."""
    system = fzg_c40()
    v107 = v107_check(system["pair"], system["wheel_torque_Nm"], data=system["data"], case="s104")
    points, levels, v103, cuts = point_series(system)
    on = onset(points)
    RESULTS.mkdir(exist_ok=True)
    files = {"s104_points.csv": points, "s104_levels.csv": levels, "s104_v103.csv": v103, "s104_v107.csv": v107,
             "s104_onset.csv": on}
    for name, rows in files.items():
        write_records_csv(RESULTS / name, rows)
    figs = figures(points, levels, cuts)
    return {"points": points, "levels": levels, "v103": v103, "v107": v107, "onset": on, "figures": figs}


if __name__ == "__main__":
    out = run()
    v = [r for r in out["v103"] + out["v107"] if r["verdict"] != "RECORD"]
    print(f"s104: K1 {sum(p['K1'] == 'PASS' for p in out['points'])}/{len(out['points'])} PASS; v103 + v107 judged "
          f"{sum(r['verdict'] == 'PASS' for r in v)} PASS, {sum(r['verdict'] == 'MISS' for r in v)} MISS")
    for o in out["onset"]:
        print(f"  {o['end']} {o['definition']}: {o['departure']:.0%} at d_end/b = {o['d_end_over_b']:.3g} "
              f"({o['d_end_mm']:.3g} mm)")
