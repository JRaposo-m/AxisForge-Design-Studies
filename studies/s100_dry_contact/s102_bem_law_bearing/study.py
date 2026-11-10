"""Run of s102: the 6210 load distribution with three ball laws (c_p, Hertz series, BEM series) against AxisForge.

``python -m studies.s100_dry_contact.s102_bem_law_bearing.study`` writes into ``results/`` (git-ignored):
``s102_cases.csv`` (one record per case and law), ``s102_elements.csv``, ``s102_effects.csv`` (BEM effect and
ISO-fit effect apart), ``s102_newton.csv`` (residual history of every solve), ``s102_laws.csv`` (the BEM series
laws), ``s102_v003.csv``, ``s102_criteria.csv`` and the figures of ``studies/outputs``.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from axisforge.outputs._format import write_records_csv
from slippy_bridge.contact import hertz
from slippy_bridge.contact.core.machine_elements.bearings import DeepGrooveBallContact

from studies.outputs.common.plots import convergence
from studies.outputs.l000_axisforge.plots import bearing_loads as plots
from studies.s000_axisforge.s001_bearing_clearance_and_contact_angle import (load_node, loaded_zone, records,
                                                                             solve_bearing)
from studies.s100_dry_contact.s101_bem_ball_raceway_law import bem_law
from studies.v000_axisforge.v003_iso16281_load_zone import check as v003_check

from .construction import LAW_DELTA_FACTORS, cases
from .model import law_mm, power_law, series_law, solve_distribution

RESULTS = Path(__file__).resolve().parent / "results"
TOL_IDENTITY = 1e-6      # [-] c_p law vs AxisForge's own solve: AxisForge's root tolerance (SOLVER_TOLERANCE)
TOL_INVARIANCE = 1e-8    # [-] Stribeck invariance, s = 0, Fa = 0: Newton residual 1e-10 of the load, 100x margin
TAG = "local azimuth phi_j (0 on the radial-load line); global placement convention-dependent (R-003)"


def _bem_series(bearing, Q_max, alpha, key, results, law_rows):
    """The ball's BEM series law at the contact angle ``alpha`` [rad], sized on ``Q_max`` [N] (s101's bem_law)."""
    pair = {}
    for race in ("inner", "outer"):
        (c,) = DeepGrooveBallContact.from_bearing(bearing, load=Q_max, alpha=alpha, races=(race,))
        law, solves = bem_law(f"{key} {race}", c, [f * hertz(c).delta for f in LAW_DELTA_FACTORS], results)
        pair[race] = law
        law_rows += [{"law": key, "race": race, "alpha_deg": math.degrees(alpha), "delta_m": d, "Q_bem_N": r.load_found,
                      "Q_hertz_N": r.contact.load, "Q_ratio": r.load_found / r.contact.load,
                      "Q_law_N": float(law.Q(d)), "converged": r.converged, "edge_clear": r.edge_clear,
                      "b_over_h": hz.b / r.grid.h, "c_per_m": law.c_per_m, "fit_rms": law.fit_rms}
                     for d, r, hz, _ in solves]
    ball = series_law(key, pair["inner"], pair["outer"])
    law_rows.append({"law": key, "race": "ball series", "alpha_deg": math.degrees(alpha), "c_per_m": ball.c_per_m,
                     "fit_rms": ball.fit_rms, "K_hertz_N_per_m1_5": ball.K_hertz_N_per_m1_5})
    return ball


def run(results=RESULTS) -> dict:
    results = Path(results)
    results.mkdir(parents=True, exist_ok=True)
    rows, elements, newton_rows, checks, law_rows, laws = [], [], [], [], [], {}
    for case in cases():
        b, Fr, Fa = case["bearing"], case["Fr_N"], case["Fa_N"]
        ident = {k: case[k] for k in ("name", "s_mm", "Fr_N", "Fa_N", "Fa_over_Fr")}
        # AxisForge's own ISO/TS 16281 solve (s001's model), the reference and v003's input
        analysis, _, error = solve_bearing(b, load_node(b, Fr, Fa), postprocess=False)
        ref, ref_el = records(b, analysis, error=error, kind="deep_groove")
        checks += [{**ident, **k} for k in v003_check(b, analysis.load_distribution, case="s102")]
        j_max = int(np.argmax([e["Q_N"] for e in ref_el]))
        alpha = math.radians(ref_el[j_max]["alpha_deg"])
        key = f"alpha {round(math.degrees(alpha), 3) + 0.0:.3f} deg"       # + 0.0: -0.000 is the same law as 0.000
        # OPEN R5-100: one BEM law per operating angle, that of the most loaded ball (Fa = 0: alpha 0 for every case)
        if key not in laws:
            laws[key] = _bem_series(b, ref["Q_max_N"], alpha, key, results, law_rows)
        bem = laws[key]
        rows.append({**ident, "law": "AxisForge ISO/TS 16281", **{k: ref[k] for k in (
            "delta_r_mm", "delta_a_mm", "Q_max_N", "n_loaded", "epsilon", "alpha_max_deg")},
            "Kr_N_per_mm": Fr / ref["delta_r_mm"], "converged": ref["ok"], "reason": "scipy root (AxisForge)"})
        elements += [{**ident, "law": "AxisForge ISO/TS 16281", **{k: e[k] for k in ("j", "phi_local_deg", "Q_N")}}
                     for e in ref_el]
        # the law's constant in its own units: c_p is AxisForge's [N/mm^1.5], the series laws are SI [N/m^1.5]
        for law_name, law, K in (("c_p (ISO fit)", power_law(b.cp), {"cp_N_per_mm1_5": b.cp}),
                                 ("Hertz series", law_mm(bem, hertz=True),
                                  {"K_hertz_N_per_m1_5": bem.K_hertz_N_per_m1_5}),
                                 ("BEM series", law_mm(bem), {"K_hertz_N_per_m1_5": bem.K_hertz_N_per_m1_5,
                                                              "c_per_m": bem.c_per_m})):
            label = f"{case['name']}: {law_name}"
            summ, res, el = solve_distribution(b, law, Fr, Fa, name="s102", label=label)
            _, eps = loaded_zone(b, summ["delta_r_mm"], summ["delta_a_mm"])
            # ident last but law: the Newton summary carries its own "name" ("s102") and "label"
            rows.append({**summ, **ident, "law": law_name, "epsilon": eps, "law_key": key, **K})
            elements += [{**ident, "law": law_name, **e} for e in el]
            newton_rows += [{**r, "case": case["name"], "law": law_name} for r in res.records()]
    effects = _effects(rows)
    crit = criteria(rows, elements, checks)
    for name, recs in (("cases", rows), ("elements", elements), ("effects", effects), ("newton", newton_rows),
                       ("laws", law_rows), ("v003", checks), ("criteria", crit)):
        write_records_csv(results / f"s102_{name}.csv", recs)
    return {"cases": rows, "elements": elements, "effects": effects, "newton": newton_rows, "laws": law_rows,
            "v003": checks, "criteria": crit, "figures": figures(rows, elements, effects, newton_rows, results)}


def _effects(rows) -> list[dict]:
    """BEM effect = BEM series / Hertz series - 1; ISO-fit effect = Hertz series / c_p - 1; per case and quantity.

    delta_r_el (clearance taken out, Fa = 0 only) and delta_max show the elastic effect undiluted by the closed
    clearance s/2;
    Kt is the tangent stiffness (the secant Fr / delta_r adds nothing to delta_r: R5-101)."""
    out = []
    by = {(r["name"], r["law"]): r for r in rows}
    for name in dict.fromkeys(r["name"] for r in rows):
        cp, hz, bem = by[(name, "c_p (ISO fit)")], by[(name, "Hertz series")], by[(name, "BEM series")]
        for q in ("delta_r_mm", "delta_r_el_mm", "delta_max_mm", "Q_max_N", "epsilon", "Kt_N_per_mm"):
            if q == "delta_r_el_mm" and cp["Fa_N"] != 0.0:
                continue        # with Fa the rings shift axially and delta_r - s/2 is no elastic measure (~0)
            out.append({**{k: cp[k] for k in ("name", "s_mm", "Fr_N", "Fa_N", "Fa_over_Fr")}, "quantity": q,
                        "c_p": cp[q], "hertz_series": hz[q], "bem_series": bem[q],
                        "bem_effect": bem[q] / hz[q] - 1.0, "iso_fit_effect": hz[q] / cp[q] - 1.0,
                        "total": bem[q] / cp[q] - 1.0})
    return out


def criteria(rows, elements, checks) -> list[dict]:
    out = []

    def add(cid, what, ok, detail):
        out.append({"id": cid, "criterion": what, "verdict": "RECORD" if ok is None else ("PASS" if ok else "MISS"),
                    "detail": detail})

    worst_Q = worst_dr = 0.0
    for name in dict.fromkeys(r["name"] for r in rows):
        af = {e["j"]: e["Q_N"] for e in elements if e["name"] == name and e["law"] == "AxisForge ISO/TS 16281"}
        mine = {e["j"]: e["Q_N"] for e in elements if e["name"] == name and e["law"] == "c_p (ISO fit)"}
        q_max = max(af.values())
        worst_Q = max(worst_Q, max(abs(mine[j] - af[j]) for j in af) / q_max)
        r_af = next(r for r in rows if r["name"] == name and r["law"].startswith("AxisForge"))
        r_me = next(r for r in rows if r["name"] == name and r["law"] == "c_p (ISO fit)")
        worst_dr = max(worst_dr, abs(r_me["delta_r_mm"] / r_af["delta_r_mm"] - 1.0))
    add("N1", f"limit case: the c_p law in this solver reproduces AxisForge's solve (Q_j, delta_r) within {TOL_IDENTITY:g}",
        worst_Q <= TOL_IDENTITY and worst_dr <= TOL_IDENTITY,
        f"max |dQ_j| / Q_max {worst_Q:.1e}, max |d delta_r| / delta_r {worst_dr:.1e}")
    judged = [k for k in checks if k["verdict"] in ("PASS", "MISS")]
    add("N2", "v003.check of AxisForge's solve: every judged record PASS", bool(judged)
        and all(k["verdict"] == "PASS" for k in judged), f"{sum(k['verdict'] == 'PASS' for k in judged)}/{len(judged)}")
    mine = [r for r in rows if not r["law"].startswith("AxisForge")]
    add("N3", "every Newton solve converged with ||F / f_scale|| <= 1e-10 (numerical_solvers.newton)",
        all(r["converged"] and r["residual_norm"] <= 1e-10 for r in mine),
        f"{sum(r['converged'] for r in mine)}/{len(mine)}; max residual {max(r['residual_norm'] for r in mine):.1e}; "
        f"iterations {min(r['n_iter'] for r in mine)}-"
        f"{max(r['n_iter'] for r in mine)}; fallback used {sum(bool(r['fallback']) for r in mine)}; reasons "
        f"{sorted({r['reason'] for r in mine})}")
    # Stribeck invariance: at s = 0, Fa = 0 every alpha_j = 0 and delta_j = delta_r cos(phi_j), so with Q = K delta^1.5
    # Q_j / Fr = cos^1.5(phi_j) / sum_k cos^2.5(phi_k) does not depend on K (Stribeck; Harris2007_vol1 Table 7.1, PDF p. 158): it checks the
    # kinematics and the law swap apart from AxisForge's root tolerance (N1 shares AxisForge's elements function)
    worst_q = 0.0
    for name in dict.fromkeys(r["name"] for r in rows if r["s_mm"] == 0.0 and r["Fa_N"] == 0.0):
        q = {law: np.array([e["Q_N"] for e in elements if e["name"] == name and e["law"] == law])
             for law in ("c_p (ISO fit)", "Hertz series")}
        Fr = next(r["Fr_N"] for r in rows if r["name"] == name)
        worst_q = max(worst_q, float(np.max(np.abs(q["Hertz series"] - q["c_p (ISO fit)"]))) / Fr)
    add("N5", f"limit case s = 0, Fa = 0 (Stribeck invariance): Q_j / Fr of the c_p and Hertz-series laws equal "
              f"within {TOL_INVARIANCE:g}", worst_q <= TOL_INVARIANCE, f"max |d(Q_j / Fr)| {worst_q:.1e}")
    add("N4", "BEM effect and ISO-fit effect reported apart (s102_effects.csv)", None, "RECORD")
    return out


def figures(rows, elements, effects, newton_rows, results) -> list:
    figs = []
    for name, variant in (("6210 s 0 um, Fr 5000 N, Fa/Fr 0", "s0_Fa0"), ("6210 s 20 um, Fr 5000 N, Fa/Fr 0.5", "s20_Fa05")):
        recs = [{"series": e["law"], "source": "AxisForge" if e["law"].startswith("AxisForge") else e["law"],
                 "phi_deg": e["phi_local_deg"], "Q_N": e["Q_N"]}
                for e in elements if e["name"] == name and e["law"] != "c_p (ISO fit)"]
        figs.append(plots.element_loads(recs, results, "s102", variant=variant,
                                        caption=f"{name}; AxisForge markers, this solver's Hertz-series and BEM-series "
                                                f"laws as lines through the balls; {TAG}"))
    sweep = []
    for e in effects:
        for kind in ("bem_effect", "iso_fit_effect"):
            sweep.append({"Fr_N": e["Fr_N"], "quantity": e["quantity"], "y": 100.0 * e[kind], "source": "AxisForge",
                          "series": f"{kind.replace('_', ' ')}, s {1e3 * e['s_mm']:g} um, Fa/Fr {e['Fa_over_Fr']:g}"})
    figs.append(plots.load_zone_sweep(sweep, results, "s102", x="Fr_N", x_label=r"$F_r$ [N]",
                                      panels={"Q_max_N": r"$\Delta Q_\max$ [%]",
                                              "delta_max_mm": r"$\Delta\delta_\max$ [%]",
                                              "delta_r_el_mm": r"$\Delta(\delta_r - s/2)$ [%]",
                                              "epsilon": r"$\Delta\varepsilon$ [%]",
                                              "Kt_N_per_mm": r"$\Delta K_t$ [%]"},
                                      variant="effects", caption="BEM effect = BEM series / Hertz series - 1; ISO-fit "
                                                                 "effect = Hertz series / c_p - 1; 6210, psi 0"))
    figs.append(convergence.newton_history([r for r in newton_rows if r["case"].startswith("6210 s 20 um, Fr 5000")],
                                           results, "s102", tol=1e-10, caption="6210 s 20 um, Fr 5000 N, both Fa/Fr, "
                                                                               "three laws; numerical_solvers.newton"))
    return figs


if __name__ == "__main__":
    from axisforge.outputs._format import format_table
    out = run()
    print(format_table((("id", "id", "s"), ("criterion", "criterion", "s"), ("verdict", "verdict", "s"),
                        ("detail", "detail", "s")), out["criteria"]))
