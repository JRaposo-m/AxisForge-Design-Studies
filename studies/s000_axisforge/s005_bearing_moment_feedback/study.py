"""Run of s005: the bearing tilting moment fed back to s004's two-bearing shaft, over span, axial load and clearance.

``python -m studies.s000_axisforge.s005_bearing_moment_feedback.study`` writes into ``results/`` (git-ignored):
``s005_points.csv`` (one row per run and pin: psi, m, Mz, R, Q_max with and without feedback, the tilt stiffness,
the Newton summary, the checks), ``s005_v002.csv``, ``s005_v003.csv``, ``s005_newton.csv`` (every Newton iteration),
``s005_moment_curve.csv`` (moment on the shaft vs tilt of one bearing), ``s005_criteria.csv`` and the figures of
``studies/outputs``.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from axisforge.outputs._format import format_table, write_records_csv

from studies.outputs.common.plots.convergence import newton_history
from studies.outputs.l000_axisforge.plots.bearing_loads import load_zone_sweep
from studies.outputs.l100_dry_contact.plots.contact import relative_errors
from studies.s000_axisforge.s004_bearing_compliance import coupled_tilt, ring_displacement
from studies.v000_axisforge.v002_shaft_stepped_gear import check as v002_check
from studies.v000_axisforge.v002_shaft_stepped_gear.study import solve      # OPEN R5-54 (s004): v002's solve
from studies.v000_axisforge.v003_iso16281_load_zone import check as v003_check

from .construction import FA_OVER_FR, FLOATING, S_KEYS, SPANS_MM, axial_loads, build
from .model import BEARING_TOL, EB, N_ELEMENTS, NEWTON, moment_balance, moment_on_shaft, solve_feedback

RESULTS = Path(__file__).resolve().parent / "results"
TOL_BALANCE = 1e-8          # [-] F4: moment equilibrium including Mz, relative to the largest term (lead, proposals.md)
TOL_PSI_RAD = 1e-10         # [rad] F5 superposition and L1 limit (s004's stop rule)
TOL_R_REL = 1e-8            # [-] F5 reactions: the settled FEM cancels k d terms (s004 S4: round-off 6e-10)
TOL_V002 = 1e-9             # [-] v002.check field tolerance (drawn in the F1 figure)
H_PSI_RAD = 1e-6            # [rad] central-difference step of the tilt stiffness record (psi 0.03-2 mrad)
PSI_CURVE_MRAD = tuple(np.linspace(-2.0, 2.0, 9))      # tilt sweep of the moment-tilt curve [mrad]
CAPTION = ("s004 shaft (d 40 mm, pins 0 and L, spur H501 pinion at 0.3 L, 330 N.m); 6210-like at A; Fa at A only, "
           "signed in the solver's axial sense (its global X sense open, R5-190); Euler-Bernoulli 40 el.")
MOMENT = {"ball": "B 6210-like: moment on the shaft -Mz, Eq. (18)",
          "roller": "B NU210-like: moment on the shaft +Mz, Eq. (46) with (40) (virtual work); as coded -Mz = RECORD"}


def _tilt_stiffness(bearing, R_yz, psi_rad: float, e, Fa_N: float) -> float:
    """-d(G . e)/d psi [N mm/rad] at the solution (central difference): positive = the bearing resists its tilt."""
    g = [float(moment_on_shaft(bearing, ring_displacement(bearing, R_yz, (psi_rad + s * H_PSI_RAD) * e, Fa_N=Fa_N,
                                                          tol=BEARING_TOL), R_yz) @ e) for s in (1.0, -1.0)]
    return -(g[0] - g[1]) / (2.0 * H_PSI_RAD)


def _row(o: dict, k: int, ident: dict, base: dict | None, pair, Fa) -> dict:
    """One record of pin ``k`` of a solve ``o`` (:func:`solve_feedback`); ``base`` = the feedback-0 solve."""
    s, nr = o["bearing"][k], o["newton"]
    R = float(np.linalg.norm(o["R_yz"][k]))
    rec = {**ident, "pin": "AB"[k], "x_mm": o["x_mm"][k], "psi_mrad": 1e3 * o["psi_rad"][k],
           "psi_rigid_pins_mrad": 1e3 * o["theta0_e_rad"][k], "m_Nmm": o["m_Nmm"][k], "Mz_Nmm": s["Mz_Nmm"],
           "G_e_Nmm": float(s["G_yz"] @ o["e_yz"]), "R_N": R, "Fa_N": Fa[k], "Fa_over_Fr_actual": Fa[k] / R,
           "Q_max_N": s["Q_max_N"], "delta_r_mm": s["delta_r_mm"], "n_loaded": s["n_loaded"],
           "psi_in_mrad": 1e3 * s["psi_in_rad"], "theta_n_rad": o["theta_n_rad"][k],
           "k_tilt_Nmm_per_rad": _tilt_stiffness(pair[k], o["R_yz"][k], o["psi_rad"][k], o["e_yz"], Fa[k]),
           "ok": s["ok"], "balanced": s["balanced"], "residual_N": s["residual_N"],
           "converged": nr.converged, "reason": nr.reason, "n_iter": nr.n_iter,
           "n_residual_evaluations": nr.n_residual_evaluations, "residual_norm": nr.residual_norm,
           "line_search_cuts": sum(1 for h in nr.history if math.isfinite(h.step_length) and h.step_length < 1.0),
           "superposition_theta_err_rad": o["superposition_theta_err_rad"],
           "superposition_R_err": o["superposition_R_err"]}
    if base is not None:
        b = base["bearing"][k]
        Rb = float(np.linalg.norm(base["R_yz"][k]))
        rec.update(psi_nofb_mrad=1e3 * base["psi_rad"][k], dpsi_mrad=1e3 * (o["psi_rad"][k] - base["psi_rad"][k]),
                   R_nofb_N=Rb, dR_over_R=R / Rb - 1.0, Q_max_nofb_N=b["Q_max_N"],
                   dQ_max_over_Q_max=s["Q_max_N"] / b["Q_max_N"] - 1.0, Mz_nofb_Nmm=b["Mz_Nmm"],
                   # the ratio only where the rigid tilt is not near zero (review: 0.2 mrad, bearing-expert S2)
                   psi_ratio=(o["psi_rad"][k] / base["psi_rad"][k] if abs(base["psi_rad"][k]) > 2e-4 else math.nan))
    return rec


def point(*, span_mm: float, floating: str, s_key: str, Fa_over_Fr: float) -> dict:
    """One shaft + bearing pair: feedback 0 (limit), 1 (the result) and the RECORD readings."""
    c = build(span_mm=span_mm, floating=floating, s_key=s_key, Fa_over_Fr=Fa_over_Fr)
    ss, pair = c["system"], c["bearings"]
    Fa = axial_loads(solve(ss, EB, N_ELEMENTS), Fa_over_Fr)
    base_ident = {"floating": floating, "s_key": s_key, "s_mm": c["s_mm"], "span_mm": span_mm,
                  "Fa_over_Fr": Fa_over_Fr, "Fa_A_N": Fa[0]}
    tag = f"{floating}, {s_key}, L {span_mm:g}, Fa/Fr {Fa_over_Fr:+g}"
    arm = [2.0 * b.Ri / b.Dpw if hasattr(b, "Ri") else math.nan for b in pair]      # SP-16 arm ratio (balls)
    runs = [("limit", 0.0), ("main", 1.0)]
    runs.append(("conjugate_arm", tuple(arm)) if floating == "ball" else ("roller_as_coded", (1.0, -1.0)))
    rows, newton_recs, v002, v003, out = [], [], [], [], {}
    for run, fb in runs:
        o = solve_feedback(ss, pair, Fa_N=Fa, feedback=fb, label=f"{tag}, {run}")
        out[run] = o
        judged = floating == "ball" and run in ("limit", "main")
        ident = {**base_ident, "run": run, "feedback_A": float(o["feedback"][0]), "feedback_B": float(o["feedback"][1]),
                 "judged": judged, "arm_ratio_2Ri_over_Dpw": arm[0]}
        bal = moment_balance(o["final_system"], o["final"], o["x_mm"], o["G_fed_yz"])
        base = None if run == "limit" else out["limit"]
        for k in range(2):
            rows.append({**_row(o, k, ident, base, pair, Fa), "balance_rel": bal["rel"],
                         "balance_scale_Nmm": bal["scale_Nmm"]})
        newton_recs += [{**r, **ident} for r in o["newton"].records()]
        v002 += [{**r, **ident} for r in v002_check(o["final_system"], o["final"], EB,
                                                     settlements_mm=o["settlements_mm"], case="s005", label=tag)]
    # L1: no feedback = s004's fixed point (Fa = 0 only: s004 gives no axial load)
    lim = out["limit"]
    if Fa_over_Fr == 0.0:
        ct = coupled_tilt(ss, lim["rigid"], pair)
        diff = max(abs(float(p @ lim["e_yz"]) - q) for p, q in zip(ct["psi_yz"], lim["psi_rad"]))
    else:
        diff = math.nan
    for r in rows:
        r["L1_diff_rad"] = diff if r["run"] == "limit" else math.nan
    # v003 at psi = 0 at the final reactions of the result (its own rule judges deep groove at Fa = 0); the moment at
    # psi = 0 and its closed-form bound Mz0 / (Fa Dpw/2) in (0, 1] (added at review, bearing-expert S3): by Eqs
    # (17)-(18) it is the mean of cos(phi_j) weighted by the axial ball forces
    main = out["main"]
    for k, b in enumerate(pair):
        z = ring_displacement(b, main["R_yz"][k], (0.0, 0.0), Fa_N=Fa[k], tol=BEARING_TOL)
        v003 += [{**r, **base_ident, "pin": "AB"[k]} for r in v003_check(b, z["analysis"].load_distribution,
                                                                         case="s005", label=b.label)]
        for r in rows:
            if r["pin"] == "AB"[k] and r["run"] != "limit":
                r["Mz_psi0_Nmm"] = z["Mz_Nmm"]
                r["Mz_tilt_part_Nmm"] = r["Mz_Nmm"] - z["Mz_Nmm"]
                r["Mz0_over_Fa_Dpw2"] = (z["Mz_Nmm"] / (Fa[k] * b.Dpw / 2.0) if Fa[k] != 0.0 and hasattr(b, "A")
                                         else math.nan)
    return {"rows": rows, "newton": newton_recs, "v002": v002, "v003": v003}


def moment_curve(span_mm: float = 200.0, s_key: str = "C3-like") -> list[dict]:
    """Moment of each bearing on the shaft along e against its tilt, at the rigid-pin reactions (sign of the tilt
    stiffness: a restoring bearing gives a falling line). Ball A at the three axial loads; roller B with the
    virtual-work sign and as coded."""
    recs = []
    for floating, k, fas, readings in (("ball", 0, FA_OVER_FR, ((1.0, "-Mz18"),)),
                                       ("roller", 1, (0.0,), ((1.0, "+Mz46"), (-1.0, "-Mz46 as coded (RECORD)")))):
        for fa in fas:
            c = build(span_mm=span_mm, floating=floating, s_key=s_key, Fa_over_Fr=fa)
            rigid = solve(c["system"], EB, N_ELEMENTS)
            Fa = axial_loads(rigid, fa)
            R = np.array([rigid.R_xy[k], rigid.R_xz[k]])
            e = np.array([rigid.R_xy[0], rigid.R_xz[0]]) / np.hypot(rigid.R_xy[0], rigid.R_xz[0])
            b = c["bearings"][k]
            for psi in PSI_CURVE_MRAD:
                ring = ring_displacement(b, R, 1e-3 * psi * e, Fa_N=Fa[k], tol=BEARING_TOL)
                G = float(moment_on_shaft(b, ring, R) @ e)
                for f, reading in readings:
                    recs.append({"psi_mrad": float(psi), "quantity": floating, "y": f * G,
                                 "series": f"{'A' if k == 0 else 'B'}, Fa/Fr {fa:+g}, {reading}",
                                 "source": "virtual work" if f == 1.0 else "as coded"})
    return recs


def criteria(rows, v002, v003) -> list[dict]:
    """F1-F5, L1, S5 of README 6 (judged: ball at B, feedback 0 and 1); the same for the roller as RECORD; R0."""
    out = []

    def crit(cid, text, ok, values, verdict=None):
        n_pass = sum(bool(v) for v in ok)
        return {"id": cid, "criterion": text, "n": len(ok), "n_pass": n_pass,
                "largest": max(values) if values else math.nan,
                "verdict": verdict or ("PASS" if n_pass == len(ok) else "MISS")}

    for floating in FLOATING:
        J = [r for r in rows if r["floating"] == floating and r["run"] in ("limit", "main")]
        j2 = [r for r in v002 if r["floating"] == floating and r["run"] in ("limit", "main") and r["verdict"] != "RECORD"]
        j3 = [r for r in v003 if r["verdict"] != "RECORD" and r["floating"] == floating]
        lim = [r for r in J if r["run"] == "limit" and math.isfinite(r["L1_diff_rad"])]
        rec = None if floating == "ball" else "RECORD"
        sfx = "" if floating == "ball" else " (roller at B, RECORD)"
        out += [crit("F1" + sfx, "v002.check of the final FEM (judged records)", [r["verdict"] == "PASS" for r in j2],
                     [abs(r["rel_err"]) for r in j2 if math.isfinite(r["rel_err"])], rec),
                crit("F2" + sfx, "v003.check at psi = 0 (judged records)", [r["verdict"] == "PASS" for r in j3],
                     [abs(r["rel_err"]) for r in j3], rec),
                crit("F3" + sfx, f"Newton converged (residual_abs {NEWTON['residual_abs']:g})",
                     [r["converged"] for r in J], [r["residual_norm"] for r in J], rec),
                crit("F4" + sfx, f"moment equilibrium incl. Mz, rel {TOL_BALANCE:g}",
                     [r["balance_rel"] <= TOL_BALANCE for r in J], [r["balance_rel"] for r in J], rec),
                crit("F5a" + sfx, f"final FEM slope = Newton psi, {TOL_PSI_RAD:g} rad",
                     [r["superposition_theta_err_rad"] <= TOL_PSI_RAD for r in J],
                     [r["superposition_theta_err_rad"] for r in J], rec),
                crit("F5b" + sfx, f"final FEM reactions = flexibility reactions, rel {TOL_R_REL:g}",
                     [r["superposition_R_err"] <= TOL_R_REL for r in J], [r["superposition_R_err"] for r in J], rec),
                crit("L1" + sfx, f"feedback 0 = s004 coupled_tilt (Fa = 0), {TOL_PSI_RAD:g} rad",
                     [r["L1_diff_rad"] <= TOL_PSI_RAD for r in lim], [r["L1_diff_rad"] for r in lim], rec),
                crit("S5" + sfx, "every bearing solve ok and balanced",
                     [bool(r["ok"]) and bool(r["balanced"]) for r in J], [r["residual_N"] for r in J], rec)]
    r0 = [r for r in rows if r["run"] == "main" and math.isfinite(r.get("Mz0_over_Fa_Dpw2", math.nan))]
    out.append(crit("R0 (added at review)", "Mz0 / (Fa Dpw/2) in (0, 1] (Eqs (17)-(18) closed form)",
                    [0.0 < r["Mz0_over_Fa_Dpw2"] <= 1.0 for r in r0], [r["Mz0_over_Fa_Dpw2"] for r in r0], "RECORD"))
    return out


def run(results=RESULTS) -> dict:
    """The sweep, tables and figures."""
    results = Path(results)
    results.mkdir(parents=True, exist_ok=True)
    rows, newton_recs, v002, v003 = [], [], [], []
    for floating in FLOATING:
        for Fa_over_Fr in FA_OVER_FR:
            for s_key in S_KEYS:
                for span in SPANS_MM:
                    p = point(span_mm=span, floating=floating, s_key=s_key, Fa_over_Fr=Fa_over_Fr)
                    rows += p["rows"]
                    newton_recs += p["newton"]
                    v002 += p["v002"]
                    v003 += p["v003"]
    crit = criteria(rows, v002, v003)
    curve = moment_curve()
    for name, recs in (("s005_points.csv", rows), ("s005_v002.csv", v002), ("s005_v003.csv", v003),
                       ("s005_newton.csv", newton_recs), ("s005_moment_curve.csv", curve),
                       ("s005_criteria.csv", crit)):
        write_records_csv(results / name, recs)
    return {"points": rows, "v002": v002, "v003": v003, "newton": newton_recs, "criteria": crit,
            "figures": _figures(rows, newton_recs, v002, curve, results)}


def _figures(rows, newton_recs, v002, curve, results) -> list[Path]:
    """Feedback vs span per axial load; the changes; F1 errors with the tolerance; the moment-tilt curve; Newton."""
    figs = []
    source = {"limit": "no feedback (s004 limit at Fa = 0)", "main": "Mz fed back",
              "roller_as_coded": "Eq. (46) as coded (RECORD)"}
    for floating in FLOATING:
        for fa_set, fa_tag in (((0.0,), "Fa0"), ((0.5, -0.5), "Fa05")):
            sweep, change = [], []
            for r in rows:
                if (r["floating"] != floating or r["s_key"] != "C3-like" or r["Fa_over_Fr"] not in fa_set
                        or r["run"] not in source):
                    continue
                series = f"{r['pin']}, Fa/Fr {r['Fa_over_Fr']:+g}"
                for q in ("psi_mrad", "m_Nmm", "R_N", "Q_max_N"):
                    sweep.append({"span_mm": r["span_mm"], "quantity": q, "y": r[q], "series": series,
                                  "source": source[r["run"]]})
                if r["run"] != "limit":
                    for q in ("dpsi_mrad", "dR_over_R", "dQ_max_over_Q_max"):
                        change.append({"span_mm": r["span_mm"], "quantity": q, "y": r[q], "series": series,
                                       "source": source[r["run"]]})
            cap = (f"{MOMENT[floating]}; s 20 um (C3-like); psi = slope along the load plane e; m = moment fed back on the "
                   f"shaft along e; lines: no feedback, filled: fed back\n{CAPTION}")
            figs.append(load_zone_sweep(sweep, results, "s005", x="span_mm", x_label="span L [mm]",
                                        panels={"psi_mrad": r"$\psi$ [mrad]", "m_Nmm": r"$m$ (moment on shaft) [N mm]",
                                                "R_N": r"$|R|$ [N]", "Q_max_N": r"$Q_{max}$ [N]"},
                                        filled_sources=("Mz fed back",), open_sources=(source["roller_as_coded"],),
                                        variant=f"feedback_{floating}_{fa_tag}", caption=cap))
            figs.append(load_zone_sweep(change, results, "s005", x="span_mm", x_label="span L [mm]",
                                        panels={"dpsi_mrad": r"$\Delta\psi$ [mrad]", "dR_over_R": r"$\Delta R/R$ [-]",
                                                "dQ_max_over_Q_max": r"$\Delta Q_{max}/Q_{max}$ [-]"},
                                        limits={"dpsi_mrad": 0.0, "dR_over_R": 0.0, "dQ_max_over_Q_max": 0.0},
                                        filled_sources=("Mz fed back",), open_sources=(source["roller_as_coded"],),
                                        variant=f"change_{floating}_{fa_tag}",
                                        caption=f"change with feedback against no feedback (dotted: 0); {cap}"))
    errs = {}
    for r in v002:
        if r["run"] != "main" or r["verdict"] == "RECORD" or not math.isfinite(r["rel_err"]):
            continue
        key = (r["floating"], r["span_mm"], f"{r['quantity']} {r['plane']}")
        errs[key] = max(errs.get(key, 0.0), abs(r["rel_err"]))
    figs.append(relative_errors([{"span_mm": L, "quantity": q, "series": f"B {fl}", "rel_err": v, "tol": TOL_V002,
                                  "verdict": "PASS" if v <= TOL_V002 else "MISS"} for (fl, L, q), v in errs.items()],
                                results, "s005", x="span_mm", x_label="span L [mm]", log_x=False, variant="v002_F1",
                                caption="F1: largest v002.check error of the final FEM (Mz fed back) per span, every "
                                        "clearance and Fa; tolerance 1e-9 dashed; L 150: 0.25 mm element (R5-177)"))
    figs.append(load_zone_sweep(curve, results, "s005", x="psi_mrad", x_label=r"tilt $\psi$ along e [mrad]",
                                panels={"ball": r"ball A: $G\cdot e$ [N mm]", "roller": r"roller B: $G\cdot e$ [N mm]"},
                                limits={"ball": 0.0, "roller": 0.0}, open_sources=("as coded",),
                                filled_sources=("virtual work",), variant="moment_curve",
                                caption="moment of the bearing on the shaft vs its tilt at the rigid-pin reactions "
                                        f"(L 200, C3-like): falling = restoring; the Fa != 0 offset is the moment at psi 0\n"
                                        f"{CAPTION}"))
    for floating in FLOATING:
        sel = [{**h, "label": h["label"].replace(f"{floating}, ", "")} for h in newton_recs
               if h["floating"] == floating and h["run"] == "main" and h["s_key"] == "C3-like"]
        figs.append(newton_history(sel, results, "s005", tol=NEWTON["residual_abs"], variant=f"main_{floating}",
                                   caption=f"Mz fed back, C3-like, every span and Fa/Fr; {MOMENT[floating]}\n{CAPTION}"))
    return figs


if __name__ == "__main__":
    out = run()
    print(format_table((("id", "id", "s"), ("criterion", "criterion", "s"), ("n", "n", "d"), ("n_pass", "pass", "d"),
                        ("largest", "largest", ".2e"), ("verdict", "verdict", "s")), out["criteria"]))
    print(format_table((("floating", "B", "s"), ("span_mm", "L", "g"), ("Fa_over_Fr", "Fa/Fr", "+g"),
                        ("pin", "pin", "s"), ("psi_nofb_mrad", "psi_0", ".4f"), ("dpsi_mrad", "dpsi", ".4f"),
                        ("dR_over_R", "dR/R", ".4f"), ("dQ_max_over_Q_max", "dQ/Q", ".4f"), ("m_Nmm", "m", ".1f"),
                        ("k_tilt_Nmm_per_rad", "k_tilt", ".3g"), ("n_iter", "it", "d")),
                       [r for r in out["points"] if r["run"] == "main" and r["s_key"] == "C3-like"]))
