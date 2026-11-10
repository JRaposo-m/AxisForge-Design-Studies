"""Run of s004: a gear-loaded shaft on two ISO/TS 16281 bearings against the rigid-pin FEM, over clearance and load.

``python -m studies.s000_axisforge.s004_bearing_compliance.study`` writes into ``results/`` (git-ignored):
``s004_points.csv`` (one row per bearing and point: psi rigid / coupled, d, delta_r, Q_max, passes, the closed-form
ratio), ``s004_v002.csv`` (v002.check of the rigid and the settled FEM), ``s004_v003.csv`` (v003.check of every
bearing at psi = 0), ``s004_history.csv`` (fixed-point passes), ``s004_criteria.csv`` and the figures of
``studies/outputs``.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from axisforge.outputs._format import format_table, write_records_csv

from studies.outputs.common.plots.convergence import newton_history
from studies.outputs.l000_axisforge.plots import shaft as plots
from studies.s000_axisforge.s006_mesh_misalignment import bearing_tilts, gear_station, line_of_action
from studies.v000_axisforge.v001_shaft_uniform.check import node_value, reference, section
from studies.v000_axisforge.v002_shaft_stepped_gear import THEORIES, along_shaft, check
from studies.v000_axisforge.v002_shaft_stepped_gear.study import solve      # OPEN R5-54: v002's settled solve
from studies.v000_axisforge.v003_iso16281_load_zone import check as v003_check

from .construction import FLOATING, LOAD_FACTORS, POSITIONS, S_MM, SPAN_MM, TORQUE_REF_NM, bearings, shaft
from .model import DISPLACEMENT_SIGN, TOL_PSI_RAD, coupled_tilt, rigid_line, ring_displacement

RESULTS = Path(__file__).resolve().parent / "results"
EB = THEORIES["euler_bernoulli"]
N_ELEMENTS = 40                 # uniform 5 mm elements; pins (0, L) and the gear (0.3 L, 0.5 L) on nodes
TOL_IDENTITY = 1e-10            # [-] reactions of the settled FEM vs the rigid FEM (proposals.md s004)
DECISION = 0.10                 # [-] |psi_coupled / psi_rigid - 1| above this -> s002 / s006 use coupled psi
# 4' = 1.1636 mrad, ISO/TS 16281 5.3.3 (rollers only, R5-77); the same value as s002 PSI_ISO_LIMIT_MRAD, not
# imported (it would add a second study->study edge)
LIMIT_4_MIN_RAD = math.radians(4.0 / 60.0)
M_TO_MM = 1e3                   # v003 C4 records delta_r in m
CAPTION = (f"pinion shaft of s006 (d 40 mm, pins at 0 and {SPAN_MM:g} mm, spur H501 pinion), T = factor x "
           f"{TORQUE_REF_NM:g} N.m; bearings s001 6210-like (A, B) or NU210-like (B); Euler-Bernoulli; d = delta_r "
           "along the force on the bearing (statics); no published value for this layout (R-103)")


def settlements(c: dict) -> dict:
    """The coupled bearing displacements as v002 settlements: ``{"XY": (d_A_y, d_B_y), "XZ": (d_A_z, d_B_z)}`` [mm]."""
    (dA, dB) = c["d_yz"]
    return {"XY": (float(dA[0]), float(dB[0])), "XZ": (float(dA[1]), float(dB[1]))}


def point(*, floating: str, s_key: str, load_factor: float, position: str, sign: float = DISPLACEMENT_SIGN) -> dict:
    """One shaft + bearing pair: rigid FEM, fixed point, settled FEM, v003 at psi = 0; the records of every criterion.

    Tilts are the signed components along the support-force direction R_hat of each pin (AxisForge's projection,
    s006 ``bearing_tilts`` ``psi_load_rad``), so a sign reversal of the tilt would show in the ratio; the moduli
    are recorded too. ``ratio_closed_form`` is the estimate 1 + (d_B - d_A) . R_hat / (L theta_r01 . R_hat) with
    the r01 slopes (v001 ``reference``) and the r03 delta_r at psi = 0 (v003 C4 reference): no fixed point, no
    AxisForge number; it judges the size and direction of the ratio.
    """
    ss = shaft(load_factor=load_factor, position=position)
    rigid = solve(ss, EB, N_ELEMENTS)
    pair = bearings(S_MM[s_key], floating)
    c = coupled_tilt(ss, rigid, pair, sign=sign)
    settle = settlements(c)
    settled = solve(ss, EB, N_ELEMENTS, settlements_mm=settle)
    ident = {"floating": floating, "s_key": s_key, "s_mm": S_MM[s_key], "load_factor": load_factor,
             "torque_Nm": load_factor * TORQUE_REF_NM, "position": position, "sign": sign}
    e = line_of_action(ss)          # unit (Y, Z) of the mesh load on this (pinion) shaft: the figures' plane
    x_g = gear_station(SPAN_MM, position)
    tilts = {t["x_mm"]: t for t in bearing_tilts(ss, rigid)}
    r01 = reference(ss, EB)
    L = c["x_mm"][1] - c["x_mm"][0]
    # psi = 0 solve of each bearing at its rigid reaction: v003's judge (r03 has no tilt) and the closed-form d
    v003, zero, d_cf = [], [], []
    for b, R in zip(pair, c["R_yz"]):
        z = ring_displacement(b, R, (0.0, 0.0))
        recs = [] if z["analysis"] is None else v003_check(b, z["analysis"].load_distribution, case="s004",
                                                           label=b.label)
        v003 += [{**r, **ident} for r in recs]
        zero.append(z)
        dr03 = next((M_TO_MM * float(r["reference"]) for r in recs if r["id"] == "C4"), math.nan)
        d_cf.append(DISPLACEMENT_SIGN * dr03 * R / np.linalg.norm(R))          # along the force on the bearing
    rot_cf = (d_cf[1] - d_cf[0]) / L
    rows = []
    for k, (b, xk) in enumerate(zip(pair, c["x_mm"])):
        pc, br, R = c["psi_yz"][k], c["bearing"][k], c["R_yz"][k]
        r_hat = R / np.linalg.norm(R)
        tr = tilts[xk]
        th01 = np.array([float(r01[pl].theta(np.array([xk]))[0]) for pl in ("XY", "XZ")])
        # the settled FEM's own slope at the pin must be the fixed point's psi (rigid-body superposition)
        th_set = np.array([node_value(settled, xk, "theta_xy"), node_value(settled, xk, "theta_xz")])
        psi_c_abs = float(np.hypot(br["psi_in_rad"], br["psi_n_rad"]))
        rows.append({**ident, "pin": "AB"[k], "bearing": b.label, "x_mm": xk, "R_N": tr["R_N"],
                     "psi_rigid_mrad": 1e3 * tr["psi_load_rad"], "psi_coupled_mrad": 1e3 * br["psi_in_rad"],
                     "ratio": br["psi_in_rad"] / tr["psi_load_rad"],
                     "ratio_closed_form": float((th01 + rot_cf) @ r_hat / (th01 @ r_hat)),
                     "psi_rigid_abs_mrad": 1e3 * tr["psi_rad"], "psi_coupled_abs_mrad": 1e3 * psi_c_abs,
                     "psi_n_mrad": 1e3 * br["psi_n_rad"],
                     "delta_r_mm": br["delta_r_mm"], "delta_r_psi0_mm": zero[k]["delta_r_mm"],
                     "d_y_mm": float(c["d_yz"][k][0]), "d_z_mm": float(c["d_yz"][k][1]),
                     "Q_max_N": br["Q_max_N"], "n_loaded": br["n_loaded"], "ok": br["ok"], "balanced": br["balanced"],
                     "residual_N": br["residual_N"], "passes": c["passes"], "converged": c["converged"],
                     "last_change_rad": c["history"][-1]["residual_norm"],
                     "settled_theta_err_rad": float(np.max(np.abs(th_set - pc))),
                     "over_4min": bool(floating == "roller" and k == 1 and psi_c_abs > LIMIT_4_MIN_RAD),
                     "v_gear_rigid_mm": float(e @ [node_value(rigid, x_g, "v_xy"), node_value(rigid, x_g, "v_xz")]),
                     "v_gear_coupled_mm": float(e @ [node_value(settled, x_g, "v_xy"), node_value(settled, x_g, "v_xz")]),
                     "theta_gear_rigid_mrad": 1e3 * float(e @ [node_value(rigid, x_g, "theta_xy"),
                                                               node_value(rigid, x_g, "theta_xz")]),
                     "theta_gear_coupled_mrad": 1e3 * float(e @ [node_value(settled, x_g, "theta_xy"),
                                                                 node_value(settled, x_g, "theta_xz")])})
    R_err = max(float(np.max(np.abs(np.subtract(getattr(settled, a), getattr(rigid, a))))) for a in ("R_xy", "R_xz"))
    R_scale = max(float(np.max(np.abs(getattr(rigid, a)))) for a in ("R_xy", "R_xz"))
    # size of the terms that cancel in K d: the stiffest entry of an end element, 12 EI / h^3, times the largest |d|
    EI, _ = section(ss, EB)
    h = float(np.min(np.diff(rigid.x_nodes)))
    kd = 12.0 * EI / h**3 * max(float(np.max(np.abs(d))) for d in c["d_yz"])
    for r in rows:
        r["reaction_identity_err"] = R_err / R_scale
        r["reaction_identity_err_kd"] = R_err / kd if kd > 0.0 else 0.0
    v002 = ([{**r, **ident, "fem": "rigid"} for r in check(ss, rigid, EB, case="s004")]
            + [{**r, **ident, "fem": "settled"} for r in check(ss, settled, EB, settlements_mm=settle, case="s004")])
    hist = [{**h, "label": f"{floating}, {s_key}, x{load_factor:g}, {position}"} for h in c["history"]]
    return {"rows": rows, "v002": v002, "v003": v003, "history": hist, "system": ss, "rigid": rigid,
            "settled": settled, "settle": settle, "coupled": c}


def criteria(rows, v002, v003) -> list[dict]:
    """S1-S5 and the decision D of README 6."""
    base = [r for r in rows if r["sign"] == DISPLACEMENT_SIGN]
    j2 = [r for r in v002 if r["verdict"] != "RECORD"]
    j3 = [r for r in v003 if r["verdict"] != "RECORD"]
    out = [{"id": "S1", "criterion": "v002.check of the rigid and the settled FEM (judged records)", "n": len(j2),
            "n_pass": sum(r["verdict"] == "PASS" for r in j2), "largest": max(r["rel_err"] for r in j2)},
           {"id": "S2", "criterion": "v003.check of every bearing at psi = 0 (judged: deep groove)", "n": len(j3),
            "n_pass": sum(r["verdict"] == "PASS" for r in j3), "largest": max(abs(r["rel_err"]) for r in j3)},
           {"id": "S3", "criterion": f"fixed point converged, max |delta psi| < {TOL_PSI_RAD:g} rad", "n": len(base),
            "n_pass": sum(r["converged"] for r in base), "largest": max(r["last_change_rad"] for r in base)},
           {"id": "S4", "criterion": f"reactions of the settled FEM = rigid FEM (rel {TOL_IDENTITY:g})", "n": len(base),
            "n_pass": sum(r["reaction_identity_err"] <= TOL_IDENTITY for r in base),
            "largest": max(r["reaction_identity_err"] for r in base)},
           {"id": "S4 (RECORD)", "criterion": "the same error relative to max |k d| (k = 12 EI/h^3): round-off scale",
            "n": len(base), "n_pass": len(base), "largest": max(r["reaction_identity_err_kd"] for r in base),
            "verdict": "RECORD"},
           {"id": "S5", "criterion": "every bearing solve ok and balanced", "n": len(base),
            "n_pass": sum(bool(r["ok"]) and bool(r["balanced"]) for r in base), "largest": max(r["residual_N"] for r in base)}]
    for fl in FLOATING:
        sub = [r for r in base if r["floating"] == fl and r["position"] == "offset"]
        worst = max(sub, key=lambda r: abs(r["ratio"] - 1.0))
        out.append({"id": "D", "criterion": f"{fl} at B, offset gear: max |psi_c/psi_r - 1| > {DECISION:g} -> coupled psi",
                    "n": len(sub), "n_pass": sum(abs(r["ratio"] - 1.0) > DECISION for r in sub),
                    "largest": abs(worst["ratio"] - 1.0),
                    "verdict": "COUPLED NEEDED" if abs(worst["ratio"] - 1.0) > DECISION else "rigid enough"})
    return out


def run(results=RESULTS) -> dict:
    """The sweep (floating x clearance x load x position), the unphysical-sign RECORD, tables and figures."""
    results = Path(results)
    results.mkdir(parents=True, exist_ok=True)
    rows, v002, hist, v003, keep = [], [], [], [], {}
    for fl in FLOATING:
        for s_key in S_MM:
            for pos in POSITIONS:
                for f in LOAD_FACTORS:
                    p = point(floating=fl, s_key=s_key, load_factor=f, position=pos)
                    rows += p["rows"]
                    v002 += p["v002"]
                    v003 += p["v003"]
                    hist += p["history"]
                    if (fl, s_key, f, pos) == ("roller", "C3-like", 1.0, "offset"):
                        keep["field"] = p
            # the opposite displacement sign, unphysical (the gear then deflects less than on rigid pins): a
            # sensitivity RECORD only, offset gear, every load
            for f in LOAD_FACTORS:
                rows += point(floating=fl, s_key=s_key, load_factor=f, position="offset", sign=-DISPLACEMENT_SIGN)["rows"]
    crit = criteria(rows, v002, v003)
    write_records_csv(results / "s004_points.csv", rows)
    write_records_csv(results / "s004_v002.csv", v002)
    write_records_csv(results / "s004_v003.csv", v003)
    write_records_csv(results / "s004_history.csv", hist)
    write_records_csv(results / "s004_criteria.csv", crit)
    return {"points": rows, "v002": v002, "v003": v003, "history": hist, "criteria": crit,
            "figures": _figures(rows, hist, keep["field"], results)}


def _project(records: list[dict], e) -> list[dict]:
    """XY and XZ records of ``along_shaft`` merged into the component along the load direction ``e``."""
    by = {}
    for r in records:
        key = (r["series"], r["source"], round(r["x_mm"], 9))
        by.setdefault(key, {})[r["plane"]] = r
    out = []
    for (series, source, x), planes in by.items():
        if {"XY", "XZ"} <= planes.keys():
            out.append({"series": series, "source": source, "x_mm": x,
                        "v_mm": e[0] * planes["XY"]["v_mm"] + e[1] * planes["XZ"]["v_mm"],
                        "theta_rad": e[0] * planes["XY"]["theta_rad"] + e[1] * planes["XZ"]["theta_rad"]})
    return out


def _figures(rows, hist, field, results) -> list[Path]:
    figs = []
    ss = field["system"]
    e = line_of_action(ss)
    recs = (along_shaft(ss, field["rigid"], EB, series="rigid pins")
            + along_shaft(ss, field["settled"], EB, settlements_mm=field["settle"], series="ISO/TS 16281 bearings"))
    proj = _project(recs, e)
    (dA, dB), x = field["coupled"]["d_yz"], field["coupled"]["x_mm"]
    line_x = np.linspace(0.0, ss.shaft.total_length, 21)
    v_rb, rot = rigid_line(line_x, x[0], x[1], dA, dB)
    proj += [{"series": "rigid-body line", "source": "d_A, d_B", "x_mm": float(xi), "v_mm": float(e @ vi),
              "theta_rad": float(e @ rot)} for xi, vi in zip(line_x, v_rb)]
    figs.append(plots.deflection_along_shaft(
        proj, results, "s004", supports_mm=tuple(x), loads_mm=(gear_station(SPAN_MM, "offset"),),
        variant="roller_C3_x1",
        caption="component along the mesh load; 6210-like at A, NU210-like at B, s 20 um (C3-like), factor 1, offset "
                f"gear\n{CAPTION}"))
    b_label = {"ball": "6210-like",
               "roller": "NU210-like (the r03 roller delta_r is RECORD in v003: its line is no judge)"}
    shown = ("0", "C4-like")               # the clearance extremes; CN- and C3-like lie between (s004_points.csv)
    for fl in FLOATING:
        recs = []
        for r in rows:
            # the unphysical sign +1 stays in s004_points.csv only (sensitivity RECORD, not a reading to compare)
            if r["floating"] != fl or r["sign"] != DISPLACEMENT_SIGN:
                continue
            series = f"{r['s_key']}, pin {r['pin']}"
            if r["position"] == "offset" and r["s_key"] in shown:
                recs += [{"load_factor": r["load_factor"], "y": r["ratio"], "series": series, "source": "FEM"},
                         {"load_factor": r["load_factor"], "y": r["ratio_closed_form"], "series": series,
                          "source": "r01 + r03 at psi 0"}]
            elif r["position"] == "centred" and r["s_key"] == shown[-1]:
                recs.append({"load_factor": r["load_factor"], "y": r["ratio"],
                             "series": f"centred gear (limit), {r['s_key']}, pin {r['pin']}", "source": "FEM"})
        for band in (1.0 + DECISION, 1.0 - DECISION):
            recs += [{"load_factor": f, "y": band, "series": f"decision band {band:g}", "source": "criterion"}
                     for f in (LOAD_FACTORS[0], LOAD_FACTORS[-1])]
        figs.append(plots.ratio_sweep(
            recs, results, "s004", x="load_factor", x_label=f"load factor (x {TORQUE_REF_NM:g} N.m) (-)",
            y_label=r"$\psi_\mathrm{coupled}/\psi_\mathrm{rigid}$ (-)", limit=1.0, log_x=False, legend_outside=True,
            variant=f"psi_ratio_{fl}",
            caption=f"B: {b_label[fl]}; offset gear unless marked; psi = signed component along the support force (no "
                    "sign change in the sweep); outside the band: coupled psi needed; lines: closed form without the "
                    f"fixed point\n{CAPTION}"))
    sel = [h for h in hist if (", x1, " in h["label"] or ", x0.1, " in h["label"]) and h["label"].endswith("offset")]
    figs.append(newton_history(sel, results, "s004", tol=TOL_PSI_RAD, variant="fixed_point",
                               x_label=r"fixed-point pass $k$ [-]", y_label=r"$\max|\Delta\psi|$ of the pass [rad]",
                               caption="FIXED-POINT passes (not Newton); offset gear, factors 0.1 and 1, every "
                                       f"clearance and floating bearing\n{CAPTION}"))
    return figs


if __name__ == "__main__":
    out = run()
    print(format_table((("id", "id", "s"), ("criterion", "criterion", "s"), ("n", "n", "d"), ("n_pass", "pass", "d"),
                        ("largest", "largest", ".2e")), out["criteria"]))
    print(format_table((("floating", "B", "s"), ("s_key", "s", "s"), ("load_factor", "f", "g"), ("pin", "pin", "s"),
                        ("psi_rigid_mrad", "psi_r", ".4f"), ("psi_coupled_mrad", "psi_c", ".4f"), ("ratio", "ratio", ".4f"),
                        ("ratio_closed_form", "cf", ".4f"), ("passes", "passes", "d")),
                       [r for r in out["points"] if r["sign"] == DISPLACEMENT_SIGN and r["position"] == "offset"
                        and r["load_factor"] in (0.1, 1.0)]))
