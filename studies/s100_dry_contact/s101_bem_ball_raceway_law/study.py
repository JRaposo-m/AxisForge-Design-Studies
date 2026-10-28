"""Run of s101: the BEM normal law of four ball-raceway contacts against Hertz, with its grid convergence.

``python -m studies.s100_dry_contact.s101_bem_ball_raceway_law.study`` writes into ``results/`` (git-ignored):
``s101_points.csv`` (one record per contact and approach), ``s101_laws.csv`` (fitted K, n per contact),
``s101_v105.csv`` (v105.check of every contact built), ``s101_richardson.csv``, ``s101_criteria.csv`` and the
figures of ``studies/outputs/l100_dry_contact``.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from axisforge.outputs._format import write_records_csv
from slippy_bridge.contact import hertz

from numerical_solvers import RichardsonOptions, richardson
from studies.outputs.l100_dry_contact.plots import contact as plots
from studies.v100_dry_contact.v101_hertz_point import pressure_cuts
from studies.v000_axisforge.v003_iso16281_load_zone import r02_bodies
from studies.v100_dry_contact.v105_ball_contacts import check as v105_check
from studies.v100_dry_contact.v105_ball_contacts import series_records

from .construction import B_OVER_H, B_OVER_H_LEVELS, DELTA_FACTORS, FR_N, SPAN, contacts
from .model import bem_law, fit_power_law, solve_at_approach

RESULTS = Path(__file__).resolve().parent / "results"
MM = 1e-3                        # [m/mm]: the one conversion (AxisForge's c_p is in N/mm^1.5)
# [-] B1: Q_BEM / Q_H at equal approach; v101's 0.5 % on the approach (spatial kernel, b/h >= 16) read as a load
# error, Q ~ delta^1.5: 1.5 x 0.5 % (v101 C2)
TOL_Q = 0.0075
CONFORMITY_JUDGED = 0.1          # a / r_g up to which Hertz holds (MG-08, v105's CONFORMITY_MAX)
CAPTION = (f"most loaded ball of 6210 (alpha 0) / 7210 (alpha_0 40) at Fr {FR_N:g} N, 12.7 mm, grooves 0.52, steel; "
           f"spatial kernel, b/h >= {B_OVER_H:g}, half-length {SPAN:g}; interference control")


def point_records(name, bearing, contact, solves) -> list[dict]:
    """One record per approach: BEM against Hertz at the same approach, the conformity, the grid."""
    r_g = r02_bodies(bearing, contact.race)["r_g_m"]
    d_ref = hertz(contact).delta
    out = []
    for d, r, hz, secs in solves:
        accepted = bool(r.converged and r.edge_clear)
        a_rg = hz.a / r_g
        judged = accepted and a_rg <= CONFORMITY_JUDGED and hz.b / r.grid.h >= B_OVER_H
        Q_H = r.contact.load                         # the Hertz load at this approach (the grid was sized on it)
        ratio = r.load_found / Q_H
        out.append({"contact": name, "race": contact.race, "alpha_deg": math.degrees(contact.alpha_rad),
                    "delta_m": d, "delta_over_delta_H_ref": d / d_ref, "Q_bem_N": r.load_found, "Q_hertz_N": Q_H,
                    "Q_ratio": ratio, "a_m": r.a, "a_hertz_m": hz.a, "b_m": r.b, "b_hertz_m": hz.b, "p0_Pa": r.p0,
                    "p0_hertz_Pa": hz.p0, "a_over_r_g": a_rg, "r_g_m": r_g, "h_m": r.grid.h, "b_over_h": hz.b / r.grid.h,
                    "n_x": r.grid.X.shape[0], "n_y": r.grid.X.shape[1], "converged": r.converged,
                    "edge_clear": r.edge_clear, "accepted": accepted, "seconds": secs,
                    "rel_err_Q": ratio - 1.0, "tol": TOL_Q,
                    "verdict": ("PASS" if abs(ratio - 1.0) <= TOL_Q else "MISS") if judged else "RECORD"})
    return out


def convergence(name, contact, results) -> tuple[list[dict], dict]:
    """Richardson on the load found at delta_H(Q_ref) over three grids h, 2h, 4h (powers of two)."""
    hz = hertz(contact)
    hs = [2.0 ** math.floor(math.log2(hz.b / boh)) for boh in B_OVER_H_LEVELS]        # coarse to fine [m]

    def evaluate(h):
        r, _, _ = solve_at_approach(contact, hz.delta, results, h_m=h, span=SPAN)
        return {"Q_N": r.load_found if (r.converged and r.edge_clear) else math.nan}
    # flat_atol = 0: never call a load series flat on an absolute scale (R3-10); max_p = 2, the formal order the
    # brief gives for the BEM load at fixed approach (R3-17)
    res = richardson(evaluate, hs, options=RichardsonOptions(kind="size", flat_atol=0.0, max_p=2.0,
                                                             stop_when_converged=False), name="s101_grid", label=name)
    recs = [{**r, "contact": name, "b_over_h": hz.b / r["level"] if r.get("level") else math.nan} for r in res.records()]
    return recs, {"contact": name, "status": res.final["Q_N"].status, "p": res.final["Q_N"].p,
                  "gci_fine_medium": res.final["Q_N"].gci_fine_medium, "Q_extrapolated_N": res.final["Q_N"].f_extrapolated,
                  "Q_hertz_N": contact.load}


def _conformity_at(points, level) -> float:
    """a / r_g where |Q_BEM / Q_H - 1| first reaches ``level`` (linear interpolation in the sweep order); nan if never."""
    rows = sorted((p for p in points if p["accepted"]), key=lambda p: p["a_over_r_g"])
    e = [abs(p["rel_err_Q"]) for p in rows]
    for k in range(1, len(rows)):
        if e[k - 1] < level <= e[k]:
            x0, x1 = rows[k - 1]["a_over_r_g"], rows[k]["a_over_r_g"]
            return x0 + (level - e[k - 1]) / (e[k] - e[k - 1]) * (x1 - x0)
    return math.nan


def _slope(points) -> float:
    """Slope of log |Q_BEM / Q_H - 1| against log a/r_g over the accepted points (2 expected: the quartic gap)."""
    rows = [p for p in points if p["accepted"] and p["rel_err_Q"] != 0.0]
    return float(np.polyfit(np.log([p["a_over_r_g"] for p in rows]), np.log([abs(p["rel_err_Q"]) for p in rows]), 1)[0])


def criteria(points, checks, conv) -> list[dict]:
    """The criteria of README section 6 (B1-B4, V), one record each."""
    out = []

    def add(cid, what, ok, detail):
        out.append({"id": cid, "criterion": what, "verdict": "RECORD" if ok is None else ("PASS" if ok else "MISS"),
                    "detail": detail})

    judged = [p for p in points if p["verdict"] != "RECORD"]
    add("B1", f"Q_BEM / Q_H at equal approach within {TOL_Q:.2%} where a/r_g <= {CONFORMITY_JUDGED}",
        bool(judged) and all(p["verdict"] == "PASS" for p in judged),
        f"{sum(p['verdict'] == 'PASS' for p in judged)}/{len(judged)}; largest |err| "
        f"{max((abs(p['rel_err_Q']) for p in judged), default=math.nan):.3%}")
    for name in dict.fromkeys(p["contact"] for p in points):
        rows = sorted((p for p in points if p["contact"] == name and p["accepted"]), key=lambda p: p["a_over_r_g"])
        e = np.array([p["rel_err_Q"] for p in rows])
        mono = bool(np.all(np.diff(e) <= 0.0) or np.all(np.diff(e) >= 0.0))
        stable = bool(np.all(e[e != 0] < 0) or np.all(e[e != 0] > 0))
        add("B2", f"{name}: departure monotonic and sign-stable in a/r_g", mono and stable,
            f"Q_BEM/Q_H - 1 from {e[0]:+.3%} (a/r_g {rows[0]['a_over_r_g']:.3f}) to {e[-1]:+.3%} "
            f"(a/r_g {rows[-1]['a_over_r_g']:.3f}); a/r_g at 1 %: {_conformity_at(rows, 0.01):.3f}, "
            f"at 3 %: {_conformity_at(rows, 0.03):.3f}")
    add("B3", "every solve converged with the pressure clear of the domain edge",
        all(p["accepted"] for p in points), f"{sum(p['accepted'] for p in points)}/{len(points)}")
    for c in conv:
        add("B4", f"{c['contact']}: grid convergence of Q at delta_H(Q_ref): GCI_fm <= {TOL_Q:.2%}",
            c["status"] in ("monotonic", "flat") and c["gci_fine_medium"] <= TOL_Q,
            f"status {c['status']}, p {c['p']:.2f}, GCI_fm {c['gci_fine_medium']:.2e}, "
            f"Q_extrapolated/Q_H - 1 {c['Q_extrapolated_N'] / c['Q_hertz_N'] - 1:+.3%}")
    v = [k for k in checks if k["verdict"] in ("PASS", "MISS")]
    add("V", "v105.check (bridge Hertz vs r02) PASS on every judged record of the contacts built",
        bool(v) and all(k["verdict"] == "PASS" for k in v), f"{sum(k['verdict'] == 'PASS' for k in v)}/{len(v)}")
    return out


def run(results=RESULTS) -> dict:
    """Solve the four laws, the grid convergence and v105.check; write the tables and figures."""
    results = Path(results)
    results.mkdir(parents=True, exist_ok=True)
    points, laws, checks, conv_recs, conv, keep = [], [], [], [], [], {}
    built = contacts()
    for name, (bearing, ld, contact) in built.items():
        d_ref = hertz(contact).delta
        law, solves = bem_law(name, contact, [f * d_ref for f in DELTA_FACTORS], results, b_over_h=B_OVER_H, span=SPAN)
        pts = point_records(name, bearing, contact, solves)
        points += pts
        # the contact as built from the AxisForge result (the BEM sweep changes only its load)
        checks += [{**k, "contact": name} for k in v105_check([contact], bearing, ld, case="s101", label=name)]
        K_n, n = fit_power_law(law.delta_m, law.Q_N)            # a descriptor of the table, not the law
        laws.append({"contact": name, "K_hertz_N_per_m1_5": law.K_hertz_N_per_m1_5, "c_per_m": law.c_per_m,
                     "c_times_delta_H_ref": law.c_per_m * d_ref, "fit_rms": law.fit_rms,
                     "n_points": len(law.delta_m), "Q_ratio_at_Q_ref": float(law.Q(d_ref)) / contact.load,
                     "descriptor_K_N_per_m_n": K_n, "descriptor_n": n,
                     "slope_err_vs_a_over_r_g": _slope(pts)})
        keep[name] = (law, solves, contact, d_ref)
        if name.startswith("6210"):
            recs, final = convergence(name, contact, results)
            conv_recs += recs
            conv.append(final)
    # the 6210 ball: the Hertz series of its two bridge contacts through v105 (S1 vs r02, S2 vs c_p), and the series
    # of its two BEM laws at Q_ref (RECORD)
    dg = built["6210 inner"][0]
    pair = [built["6210 inner"][2], built["6210 outer"][2]]
    checks += [{**k, "contact": "6210 ball"} for k in series_records(pair, dg, case="s101", label="6210 ball")]
    Q_ref = pair[0].load
    K_bem = Q_ref / float(keep["6210 inner"][0].delta(Q_ref) + keep["6210 outer"][0].delta(Q_ref)) ** 1.5   # [N/m^1.5]
    K_h = Q_ref / (keep["6210 inner"][3] + keep["6210 outer"][3]) ** 1.5
    laws.append({"contact": "6210 ball series at Q_ref", "K_hertz_N_per_m1_5": K_h, "Q_ratio_at_Q_ref": K_bem / K_h,
                 "K_bem_N_per_m1_5": K_bem, "cp_axisforge_N_per_mm1_5": dg.cp,
                 "K_bem_over_cp": K_bem * MM ** 1.5 / dg.cp})
    crit = criteria(points, checks, conv)
    for name, recs in (("points", points), ("laws", laws), ("v105", checks), ("richardson", conv_recs),
                       ("criteria", crit)):
        write_records_csv(results / f"s101_{name}.csv", recs)
    return {"points": points, "laws": laws, "v105": checks, "richardson": conv_recs, "conv": conv, "criteria": crit,
            "figures": figures(points, keep, conv_recs, conv, results), "keep": keep}


def figures(points, keep, conv_recs, conv, results) -> list:
    """Figures of README section 7 (studies/outputs/l100_dry_contact/plots/contact.py)."""
    figs = []
    law_recs = [{"series": p["contact"], "source": "BEM", "delta_m": p["delta_m"], "Q_N": p["Q_bem_N"]}
                for p in points if p["accepted"]]
    for name, (_, _, contact, d_ref) in keep.items():
        for d in np.geomspace(0.04, 1.6, 30) * d_ref:
            law_recs.append({"series": name, "source": "Hertz", "delta_m": float(d),
                             "Q_N": contact.load * (d / d_ref) ** 1.5})
    figs.append(plots.load_approach(law_recs, results, "s101", caption=CAPTION))
    err = []
    for p in points:
        for q, m, h in (("Q", "Q_bem_N", "Q_hertz_N"), ("a", "a_m", "a_hertz_m"), ("b", "b_m", "b_hertz_m"),
                        ("p0", "p0_Pa", "p0_hertz_Pa")):
            err.append({"a_over_r_g": p["a_over_r_g"], "quantity": q, "series": p["contact"],
                        "rel_err": p[m] / p[h] - 1.0, "tol": TOL_Q if q == "Q" else math.nan,
                        "verdict": p["verdict"] if q == "Q" else "RECORD"})
    figs.append(plots.relative_errors(err, results, "s101", x="a_over_r_g", x_label=r"$a_H / r_g$ (Hertz, equal approach)",
                                      variant="vs_conformity", slope=2.0,
                                      caption=f"{CAPTION}; tolerance drawn where judged (a/r_g <= {CONFORMITY_JUDGED}); "
                                              "dotted: slope 2 (the quartic gap)"))
    # pressure along the two axes of the 6210 inner contact at delta_H(Q_ref)
    law, solves, contact, d_ref = keep["6210 inner"]
    near = min(solves, key=lambda s: abs(s[0] - d_ref))
    figs.append(plots.pressure_cut(pressure_cuts(near[1]), results, "s101", variant="6210_inner_Qref"))
    conv_err = [{"b_over_h": r["b_over_h"], "quantity": "Q", "series": r["contact"],
                 "rel_err": r["value"] / next(c["Q_extrapolated_N"] for c in conv if c["contact"] == r["contact"]) - 1,
                 "tol": TOL_Q, "verdict": "PASS"} for r in conv_recs if r.get("metric") == "Q_N"]
    figs.append(plots.relative_errors(conv_err, results, "s101", x="b_over_h", x_label=r"$b_H/h$ (actual)",
                                      variant="richardson", caption=f"{CAPTION}; error against the Richardson "
                                                                    "extrapolation at delta_H(Q_ref)"))
    return figs


if __name__ == "__main__":
    from axisforge.outputs._format import format_table
    out = run()
    print(format_table((("id", "id", "s"), ("criterion", "criterion", "s"), ("verdict", "verdict", "s"),
                        ("detail", "detail", "s")), out["criteria"]))
