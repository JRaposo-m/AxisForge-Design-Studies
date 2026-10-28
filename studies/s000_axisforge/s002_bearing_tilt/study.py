"""Run of s002: an imposed tilt psi on s001's deep groove and roller bearings, two load levels.

``python -m studies.s000_axisforge.s002_bearing_tilt.study`` writes into ``results/`` (git-ignored):
``s002_cases.csv`` (one record per bearing, load and tilt), ``s002_elements.csv`` (per rolling element),
``s002_laminae.csv`` (roller lamina loads q_jk), ``s002_v003.csv`` (v003.check at psi = 0),
``s002_criteria.csv`` and the figures of ``studies/outputs/l000_axisforge``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from axisforge.config import SOLVER_TOLERANCE
from axisforge.outputs._format import write_records_csv
from axisforge.outputs.solvers.bearings.load_distribution import load_distribution_results as af_ld

from studies.outputs.l000_axisforge.plots import bearing_loads as plots
from studies.s000_axisforge.s001_bearing_clearance_and_contact_angle import (BALANCE_RTOL, load_node, records,
                                                                             solve_bearing)
from studies.v000_axisforge.v003_iso16281_load_zone import check as v003_check

from .construction import FA_OVER_FR, FR_N, PSI_BALL_MRAD, PSI_ISO_LIMIT_MRAD, PSI_ROLLER_MRAD, bearings

RESULTS = Path(__file__).resolve().parent / "results"
TAG = "local azimuth phi_j (0 on the radial-load line); global placement convention-dependent (R-003)"
# [-] T1 (psi = 0 through psi_override equals the slope path of s001) and T4 (Mz odd in psi): the same solver and
# start point, so agreement is to the root tolerance of the ISO/TS 16281 solver (axisforge.config)
TOL_SAME = SOLVER_TOLERANCE
FLAG_4MIN = "psi > 4' (ISO 5.3.3, Eq. (60))"
PSI_CHECK_MRAD = 1.0                       # T4: Mz(-psi) = -Mz(psi) at this tilt


def _solve(bearing, Fr, Fa, psi_mrad, ident):
    """One solve through s001's model at an imposed tilt; returns (case, elements, laminae, analysis)."""
    analysis, post, error = solve_bearing(bearing, load_node(bearing, Fr, Fa), psi_rad=1e-3 * psi_mrad)
    case, elements = records(bearing, analysis, postprocessed=post, error=error, **ident)
    # the 4' limit is that of Eq. (60) in the roller L10r only (ISO 5.3.3); ball rows are not flagged
    roller = analysis is not None and analysis.load_distribution.is_line_contact
    case["flag"] = FLAG_4MIN if roller and abs(psi_mrad) > PSI_ISO_LIMIT_MRAD else ""
    laminae = []
    if analysis is not None and analysis.load_distribution.is_line_contact:
        phi = np.degrees(bearing.phi_j)
        laminae = [{**ident, "j": r["j"], "k": r["k"], "phi_deg": float(phi[r["j"]]), "x_mm": r["x_mm"],
                    "delta_mm": r["delta_mm"], "q_N": r["q_N"]}
                   for r in af_ld.lamina_records(analysis.load_distribution)]
    return case, elements, laminae, analysis


def run(results=RESULTS) -> dict:
    """Solve the tilt sweep, check psi = 0 through v003 and against s001's slope path, write tables and figures."""
    results = Path(results)
    results.mkdir(parents=True, exist_ok=True)
    cases, elements, laminae, checks, same = [], [], [], [], []
    for kind, bearing in bearings().items():
        ball = kind == "deep_groove"
        for Fr in FR_N:
            for ratio in (FA_OVER_FR if ball else (0.0,)):
                psis = PSI_BALL_MRAD if ball else PSI_ROLLER_MRAD
                for psi in (*psis, -PSI_CHECK_MRAD):          # the negative tilt only for T4
                    ident = {"kind": kind, "Fr_level_N": Fr, "Fa_over_Fr": ratio, "psi_in_mrad": psi}
                    case, el, lam, analysis = _solve(bearing, Fr, ratio * Fr, psi, ident)
                    cases.append(case)
                    elements += el
                    laminae += lam
                    if psi == 0.0 and analysis is not None:
                        checks += [{**ident, **r} for r in v003_check(bearing, analysis.load_distribution,
                                                                      case="s002")]
                        # T1: s001's path without an override (psi from the node slopes, zero here)
                        ref, _, _ = solve_bearing(bearing, load_node(bearing, Fr, ratio * Fr), psi_rad=None)
                        q, q0 = analysis.load_distribution.rows[0], ref.load_distribution.rows[0]
                        Q, Q0 = np.asarray(q.Q_j), np.asarray(q0.Q_j)
                        same.append({**ident, "max_dQ_over_Qmax": float(np.max(np.abs(Q - Q0)) / Q0.max()),
                                     "d_delta_r_rel": analysis.load_distribution.delta_r / ref.load_distribution.delta_r
                                     - 1.0})
    crit = criteria(cases, checks, same)
    for name, recs in (("cases", cases), ("elements", elements), ("laminae", laminae), ("v003", checks),
                       ("psi0_vs_s001", same), ("criteria", crit)):
        write_records_csv(results / f"s002_{name}.csv", recs)
    return {"cases": cases, "elements": elements, "laminae": laminae, "v003": checks, "same": same, "criteria": crit,
            "figures": figures(cases, elements, laminae, results)}


def criteria(cases: list[dict], checks: list[dict], same: list[dict]) -> list[dict]:
    """The criteria of README section 6, one record each."""
    out = []

    def add(cid, what, ok, detail):
        out.append({"id": cid, "criterion": what, "verdict": ("PASS" if ok else "MISS") if ok is not None else "RECORD",
                    "detail": detail})

    worst = max(max(abs(s["max_dQ_over_Qmax"]), abs(s["d_delta_r_rel"])) for s in same)
    add("T1", "psi = 0 imposed (psi_override) equals s001's slope path (Q_j, delta_r)", worst <= TOL_SAME,
        f"largest {worst:.1e} (tol {TOL_SAME:g}), {len(same)} cases")
    judged = [c for c in checks if c["verdict"] in ("PASS", "MISS")]
    add("T2", "v003.check at psi = 0: every judged record PASS (deep groove, Fa = 0)",
        bool(judged) and all(c["verdict"] == "PASS" for c in judged),
        f"{sum(c['verdict'] == 'PASS' for c in judged)}/{len(judged)}")
    bad = [c for c in cases if not (c["ok"] and c["balanced"])]
    add("T3", f"every case ok and balanced (residual <= {BALANCE_RTOL:g} max(Fr, Fa))", not bad,
        f"{len(cases) - len(bad)}/{len(cases)}; max residual {max(c['residual_N'] for c in cases):.1e} N")
    for kind, judged_ in (("deep_groove", True), ("cylindrical_roller", None)):
        rows = [c for c in cases if c["kind"] == kind and c["Fa_over_Fr"] == 0.0]
        pos = [c for c in rows if c["psi_in_mrad"] > 0.0]
        # Mz of Eq. (18) / (46) is the external moment that holds the tilt: Mz psi > 0 is a positive tilt stiffness
        stiff = all(c["Mz_Nmm"] * c["psi_in_mrad"] > 0.0 for c in pos)
        odd = []
        for Fr in FR_N:
            m = {c["psi_in_mrad"]: c["Mz_Nmm"] for c in rows if c["Fr_level_N"] == Fr}
            odd.append(abs(m[-PSI_CHECK_MRAD] + m[PSI_CHECK_MRAD]) / abs(m[PSI_CHECK_MRAD]))
        ok = stiff and max(odd) <= TOL_SAME
        add("T4", f"{kind}, Fa = 0: positive tilt stiffness Mz psi > 0 (Mz = applied moment, the solver's own sign) "
                  "and Mz odd in psi",
            ok if judged_ else None,
            f"Mz psi > 0 {stiff}; |Mz(-1)+Mz(+1)|/|Mz(+1)| {max(odd):.1e}; Mz at +1 mrad "
            + ", ".join(f"{c['Mz_Nmm']:.0f}" for c in pos if c["psi_in_mrad"] == PSI_CHECK_MRAD) + " N.mm")
    return out


def figures(cases, elements, laminae, results) -> list:
    """Figures of README section 7 through studies/outputs/l000_axisforge/plots/bearing_loads."""
    caption = f"s 20 um, psi imposed; {TAG}"
    figs = []
    # both tilt signs at Fa > 0: the sign of psi against Fa matters (R5-75, R-003)
    for kind, ratio, psis, name in (("deep_groove", 0.5, (-1.0, 0.0, 1.0, 2.0), "DGBB, Fa/Fr 0.5"),
                                    ("cylindrical_roller", 0.0, (0.0, 0.5, 1.0), "CRB")):
        recs = [{"series": f"{name}, psi {e['psi_in_mrad']:g} mrad", "source": "AxisForge",
                 "phi_deg": e["phi_local_deg"], "Q_N": e["Q_N"]}
                for e in elements if e["kind"] == kind and e["Fa_over_Fr"] == ratio
                and e["Fr_level_N"] == FR_N[-1] and e["psi_in_mrad"] in psis]
        figs.append(plots.element_loads(recs, results, "s002", variant=kind,
                                        caption=f"Fr {FR_N[-1]:g} N; {caption}"))
    sweep = []
    for c in cases:
        series = (f"{'DGBB' if c['kind'] == 'deep_groove' else 'CRB'}, Fr {c['Fr_level_N']:g} N"
                  + (f", Fa/Fr {c['Fa_over_Fr']:g}" if c["kind"] == "deep_groove" else ""))
        for q in ("Q_max_N", "Mz_Nmm", "delta_a_mm", "delta_r_mm"):
            sweep.append({"psi_mrad": c["psi_in_mrad"], "quantity": q, "y": c[q], "series": series,
                          "source": "psi 0 = s001 (T1)" if c["psi_in_mrad"] == 0.0 else "AxisForge"})
    figs.append(plots.load_zone_sweep(sweep, results, "s002", x="psi_mrad", x_label=r"imposed tilt $\psi$ [mrad]",
                                      panels={"Q_max_N": r"$Q_\max$ [N]", "Mz_Nmm": r"$M_z$ [N mm]",
                                              "delta_r_mm": r"$\delta_r$ [mm]", "delta_a_mm": r"$\delta_a$ [mm]"},
                                      x_marks={"ISO 4' (roller Eq. (60) only)": PSI_ISO_LIMIT_MRAD},
                                      open_sources=("psi 0 = s001 (T1)",), variant="tilt",
                                      caption=f"{caption}; Mz = applied moment in each solver's own sign (ball "
                                              "Eq. (18), roller Eq. (46): opposite senses, R5-76); sign of psi against "
                                              "Fa convention-dependent (R-003)"))
    lam = [{**r, "psi": r["psi_in_mrad"]} for r in laminae
           if r["Fr_level_N"] == FR_N[-1] and r["psi_in_mrad"] in (0.0, 0.25, 0.5, 1.0)]
    figs.append(plots.lamina_map(lam, results, "s002", panel="psi", panel_label="psi [mrad] = ", variant="roller",
                                 caption=f"CRB, Fr {FR_N[-1]:g} N; lamina loads (independent springs, not edge "
                                         f"pressures); {caption}"))
    return figs


if __name__ == "__main__":
    from axisforge.outputs._format import format_table
    out = run()
    print(format_table((("id", "id", "s"), ("criterion", "criterion", "s"), ("verdict", "verdict", "s"),
                        ("detail", "detail", "s")), out["criteria"]))
