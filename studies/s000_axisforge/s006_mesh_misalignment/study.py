"""Run of s006: shaft-bending part of f_sh over span, gear position, helix angle, hand and driver.

``python -m studies.s000_axisforge.s006_mesh_misalignment.study`` writes into ``results/`` (git-ignored):
``s006_points.csv`` (one row per stage: f_sh_bend FEM and r01 judge, couple identity, twist),
``s006_bearings.csv`` (slope at every pin), ``s006_v002.csv`` (v002.check of every solved shaft),
``s006_identities.csv`` (v002.gear_identities of every stage), ``s006_criteria.csv`` and the figures of
``studies/outputs/l000_axisforge``.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from axisforge.outputs._format import write_records_csv
from axisforge.solvers.machine_elements.shaft.fem_solvers.global_solver.rigid_support import RigidSupportFEMSolver

from studies.outputs.l000_axisforge.plots import shaft as plots
from studies.v000_axisforge.v001_shaft_uniform.check import TOL_NODAL as TOL
from studies.v000_axisforge.v001_shaft_uniform.check import node_value
from studies.v000_axisforge.v002_shaft_stepped_gear import (THEORIES, along_shaft, check, gear_identities, mesh_judge,
                                                            physical_couples, reference)

from .construction import (ALPHA_N_DEG, FACE_WIDTH_MM, MODULE_MM, PHI_DEG, POSITIONS, SPANS_MM, TORQUE_NM, X, Z,
                           gear_station, stage)
from .model import MM_TO_UM, bearing_tilts, line_of_action, mesh_misalignment, pinion_twist

RESULTS = Path(__file__).resolve().parent / "results"
J16 = "convention-dependent (J-16)"
EB, TIM = THEORIES["euler_bernoulli"], THEORIES["timoshenko/cowper/single_point"]
N_UNIFORM = 40                  # about this many elements, piecewise uniform between stations (EB nodally exact)
N_TIMOSHENKO = 400              # finer for the one-point Timoshenko RECORD (O(h^2) error, about 1e-5 here)
BETA_SWEEP_DEG = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0)   # finer than proposals' 0/15/30 for the line figure
SPAN_BETA_MM = 200.0            # span of the helix sweep
CAPTION = (f"pair mn {MODULE_MM:g} mm, z {Z[0]}/{Z[1]}, x {X[0]:g}/{X[1]:g}, b {FACE_WIDTH_MM:g} mm (H501); shafts d 40 mm "
           f"S355 on rigid pins at 0 and L; T {TORQUE_NM:g} N.m on the pinion; Euler-Bernoulli")


def solve(system, settings, n_uniform: int, x_g: float):
    """``RigidSupportFEMSolver`` on a piecewise-uniform mesh: every station a node, about ``n_uniform`` elements.

    Stations: shaft ends, pins, the gear x_g and the face edges x_g +- b/2. Each interval between
    stations is divided into ``max(1, round(length / h))`` equal elements, h = L / n_uniform, so no
    element is a sliver (a first run overlaid a uniform grid on the stations: elements of 0.1-0.25 mm
    beside 4-8 mm ones lost digits in the Euler-Bernoulli solve, README 9).
    """
    L = system.shaft.total_length
    h = L / n_uniform
    stations = sorted({0.0, L, x_g, x_g - 0.5 * FACE_WIDTH_MM, min(L, x_g + 0.5 * FACE_WIDTH_MM),
                       *(b.position for b in system.bearings)})
    nodes = [float(x) for lo, hi in zip(stations[:-1], stations[1:])
             for x in np.linspace(lo, hi, max(1, round((hi - lo) / h)) + 1)[:-1]] + [L]
    return RigidSupportFEMSolver(settings).solve(system, extra_mandatory=nodes)


def _at(sol_or_result, x: float, fem: bool) -> tuple[tuple[float, float], tuple[float, float]]:
    """((theta_xy, theta_xz), (v_xy, v_xz)) at x: from a ShaftResults node (fem) or from r01 planes (dict)."""
    if fem:
        return ((node_value(sol_or_result, x, "theta_xy"), node_value(sol_or_result, x, "theta_xz")),
                (node_value(sol_or_result, x, "v_xy"), node_value(sol_or_result, x, "v_xz")))
    return ((float(sol_or_result["XY"].theta(x)), float(sol_or_result["XZ"].theta(x))),
            (float(sol_or_result["XY"].v(x)), float(sol_or_result["XZ"].v(x))))


def point(*, span_mm: float, position: str, beta_deg: float = 0.0, hand: int = 1, driver: str = "pinion",
          timoshenko: bool = False) -> dict:
    """One stage solved on both shafts: the point record, bearing records, v002 records, identity records, results.

    Criteria inside (README 6): S1 v002.check of both shafts; S2 v002.gear_identities; S3 reaction shift
    of the couple = C/span per plane and shaft; S4 f_sh_bend from the FEM nodes equals the one from
    the r01 slopes of the same system (v002.reference), within TOL of b max|theta|.
    """
    resolved = stage(span_mm=span_mm, position=position, beta_deg=beta_deg, hand=hand, driver=driver)
    mesh = resolved.meshes[0]
    # the signed mesh forces from the inputs (v002 mesh_judge): pinion = gear 1, sense +1, driver from the path
    judge = mesh_judge(torque_Nm=TORQUE_NM, beta_deg=beta_deg, mn_mm=MODULE_MM, z1=Z[0], z2=Z[1], alpha_n_deg=ALPHA_N_DEG,
                       b_mm=FACE_WIDTH_MM, x1=X[0], x2=X[1], hand1=hand if beta_deg else 0, rot1=1,
                       gear1_drives=driver == "pinion")
    x_g = gear_station(span_mm, position)
    ident = {"span_mm": span_mm, "position": position, "x_g_mm": x_g, "beta_deg": beta_deg,
             "hand": hand if beta_deg else 0, "driver": driver, "tag": J16 if beta_deg else ""}
    l_yz = line_of_action(resolved.shafts[1])          # transverse mesh force on the wheel, normalised
    fem, r01, v002_rows, bearings, couple_err = {}, {}, [], [], 0.0
    couple_scale = judge["F_a_N"] * max(abs(r) for r in judge["rho_mm"]) / span_mm
    for side, ss in enumerate(resolved.shafts, start=1):
        res = solve(ss, EB, N_UNIFORM, x_g)
        v002_rows += [{**r, **ident} for r in check(ss, res, EB, case="s006", label=ss.name)]
        bearings += [{**b, "shaft": ss.name, **ident} for b in bearing_tilts(ss, res)]
        fem[ss.name], r01[ss.name] = res, reference(ss, EB)
        # S3: share of the couple in the reaction at pin A, per plane: FEM - r01 without couple = +C/span
        if beta_deg:
            no_c = reference(ss, EB, couples=False)
            for k, plane in enumerate(("XY", "XZ")):
                d_R = float(getattr(res, f"R_{plane.lower()}")[0]) - no_c[plane].R_A
                F_x = (judge["on_1"] if side == 1 else judge["on_2"])[2]
                C = physical_couples(F_x, judge["rho_mm"][side - 1], PHI_DEG)[k]
                couple_err = max(couple_err, abs(d_R - C / span_mm) / couple_scale)
    names = [ss.name for ss in resolved.shafts]       # (pinion_shaft, wheel_shaft)
    (th_p, v_p), (th_w, v_w) = (_at(fem[n], x_g, True) for n in names)
    f = mesh_misalignment(th_p, th_w, v_p, v_w, l_yz, FACE_WIDTH_MM)
    (rth_p, rv_p), (rth_w, rv_w) = (_at(r01[n], x_g, False) for n in names)
    f_ref = mesh_misalignment(rth_p, rth_w, rv_p, rv_w, l_yz, FACE_WIDTH_MM)
    # largest nodal slope of either shaft over b: never 0 (at a centred gear the slope itself is 0 by symmetry)
    scale = MM_TO_UM * FACE_WIDTH_MM * max(float(np.max(np.hypot(fem[n].theta_xy, fem[n].theta_xz))) for n in names)
    rel = abs(f["f_sh_bend_um"] - f_ref["f_sh_bend_um"]) / scale
    r_b = 0.5 * resolved.shafts[0].gears[0].gear.db                  # pinion base radius [mm] (AxisForge geometry)
    twist = pinion_twist(fem[names[0]], x_g, FACE_WIDTH_MM, r_b)
    rec = {"case": "s006", **ident, "l_y": float(l_yz[0]), "l_z": float(l_yz[1]), "F_tw_N": mesh.F_tw_N,
           "F_a_N": mesh.F_a_N, **f,
           # f per unit transverse force: separates the geometry (beta, a_w) from the force at fixed torque
           "f_sh_bend_per_F_bt_um_per_kN": 1e3 * f["f_sh_bend_um"] / float(np.hypot(mesh.F_tw_N, mesh.F_r_N)), "f_sh_bend_r01_um": f_ref["f_sh_bend_um"], "rel_err_f_sh_bend": rel,
           "S4": "PASS" if rel <= TOL else "MISS",
           "couple_shift_err": couple_err if beta_deg else math.nan,
           "S3": ("PASS" if couple_err <= TOL else "MISS") if beta_deg else "n/a",
           "theta_p_xy_rad": th_p[0], "theta_p_xz_rad": th_p[1], "theta_w_xy_rad": th_w[0], "theta_w_xz_rad": th_w[1],
           "twist_pinion_um": twist["twist_um"], "twist_pinion_rad": twist["twist_rad"]}
    if timoshenko:
        th = {n: _at(solve(ss, TIM, N_TIMOSHENKO, x_g), x_g, True) for n, ss in zip(names, resolved.shafts)}
        thr = {n: _at(reference(ss, TIM), x_g, False) for n, ss in zip(names, resolved.shafts)}
        rec["f_sh_bend_timoshenko_um"] = mesh_misalignment(th[names[0]][0], th[names[1]][0], th[names[0]][1],
                                                      th[names[1]][1], l_yz, FACE_WIDTH_MM)["f_sh_bend_um"]
        rec["f_sh_bend_timoshenko_r01_um"] = mesh_misalignment(thr[names[0]][0], thr[names[1]][0], thr[names[0]][1],
                                                          thr[names[1]][1], l_yz, FACE_WIDTH_MM)["f_sh_bend_um"]
    ids = [{**r, **ident} for r in gear_identities(resolved, judge, theta_c_deg=PHI_DEG, case="s006",
                                                   label=resolved.label)]
    return {"point": rec, "bearings": bearings, "v002": v002_rows, "identities": ids, "results": fem,
            "resolved": resolved}


def r01_misalignment(span_mm: float, position: str, *, beta_deg: float = 0.0, hand: int = 1,
                     driver: str = "pinion") -> float:
    """f_sh_bend [um] from the r01 slopes alone (v002.reference; no FEM): the dense judge curves of the figures."""
    resolved = stage(span_mm=span_mm, position=position, beta_deg=beta_deg, hand=hand, driver=driver)
    x_g = gear_station(span_mm, position)
    (tp, vp), (tw, vw) = (_at(reference(ss, EB), x_g, False) for ss in resolved.shafts)
    return mesh_misalignment(tp, tw, vp, vw, line_of_action(resolved.shafts[1]), FACE_WIDTH_MM)["f_sh_bend_um"]


def criteria(points, v002_rows, ids) -> list[dict]:
    """S1-S4 of README 6 over the whole run."""
    judged = [r for r in v002_rows if r["verdict"] != "RECORD"]
    idj = [r for r in ids if r["verdict"] != "RECORD"]
    s3 = [p for p in points if p["S3"] != "n/a"]
    return [{"id": "S1", "criterion": "v002.check of every solved shaft", "n": len(judged),
             "n_pass": sum(r["verdict"] == "PASS" for r in judged), "largest": max(r["rel_err"] for r in judged)},
            {"id": "S2", "criterion": "v002.gear_identities (I1-I7) of every stage", "n": len(idj),
             "n_pass": sum(r["verdict"] == "PASS" for r in idj),
             "largest": max((r["rel_err"] for r in idj if r["rel_err"] == r["rel_err"]), default=math.nan)},
            {"id": "S3", "criterion": "couple share of R_A = C/span, both planes, both shafts (J-16)", "n": len(s3),
             "n_pass": sum(p["S3"] == "PASS" for p in s3), "largest": max(p["couple_shift_err"] for p in s3)},
            {"id": "S4", "criterion": "f_sh_bend FEM = f_sh_bend from r01 slopes", "n": len(points),
             "n_pass": sum(p["S4"] == "PASS" for p in points), "largest": max(p["rel_err_f_sh_bend"] for p in points)}]


def run(results=RESULTS) -> dict:
    """The span x position x driver grid (spur, judged), the helix sweep (J-16), the tables and the figures."""
    results = Path(results)
    results.mkdir(parents=True, exist_ok=True)
    out = {"points": [], "bearings": [], "v002": [], "identities": []}
    keep = {}
    for driver in ("pinion", "wheel"):
        for position in POSITIONS:
            for span in SPANS_MM:
                p = point(span_mm=span, position=position, driver=driver, timoshenko=True)
                out["points"].append(p["point"])
                for k in ("bearings", "v002", "identities"):
                    out[k] += p[k]
                if span == SPAN_BETA_MM and position == "offset" and driver == "pinion":
                    keep["spur"] = p
            for beta in BETA_SWEEP_DEG[1:]:
                for hand in (1, -1):
                    p = point(span_mm=SPAN_BETA_MM, position=position, beta_deg=beta, hand=hand, driver=driver)
                    out["points"].append(p["point"])
                    for k in ("bearings", "v002", "identities"):
                        out[k] += p[k]
    crit = criteria(out["points"], out["v002"], out["identities"])
    write_records_csv(results / "s006_points.csv", out["points"])
    write_records_csv(results / "s006_bearings.csv", out["bearings"])
    write_records_csv(results / "s006_v002.csv", out["v002"])
    write_records_csv(results / "s006_identities.csv", out["identities"])
    write_records_csv(results / "s006_criteria.csv", crit)
    figures = _figures(out, keep["spur"], results)
    return {**out, "criteria": crit, "figures": figures}


def _figures(out: dict, spur: dict, results: Path) -> list[Path]:
    """Deflection of both shafts (one point), f_sh_bend vs span x position (spur, judged), f_sh_bend vs beta (J-16), psi vs span."""
    figs = []
    resolved = spur["resolved"]
    x_g = gear_station(SPAN_BETA_MM, "offset")
    fields = []
    for ss in resolved.shafts:
        fields += along_shaft(ss, spur["results"][ss.name], EB, series=ss.name.replace("_", " "))
    for plane in ("XY", "XZ"):
        figs.append(plots.deflection_along_shaft(
            [r for r in fields if r["plane"] == plane], results, "s006", supports_mm=(0.0, SPAN_BETA_MM),
            loads_mm=(x_g,), steps_mm=(x_g - 0.5 * FACE_WIDTH_MM, x_g + 0.5 * FACE_WIDTH_MM),   # face edges, drawn as steps
            variant=f"{plane}_offset_L{SPAN_BETA_MM:g}",
            caption=f"plane {plane}; spur, gear at 0.3 L = {x_g:g} mm (grey lines labelled 'section step': the face "
                    f"edges; the shaft is uniform), pinion drives, "
                    f"line of centres +Y\n{CAPTION}"))
    spur_pts = [p for p in out["points"] if p["beta_deg"] == 0.0]
    recs = [{"span_mm": p["span_mm"], "y": p["f_sh_bend_um"], "series": f"{p['position']}, {p['driver']} drives",
             "source": "FEM", "rel_err": p["f_sh_bend_um"] / p["f_sh_bend_r01_um"] - 1.0 if abs(p["f_sh_bend_r01_um"]) > 1e-9 else 0.0}
            for p in spur_pts]
    for driver in ("pinion", "wheel"):
        for position in POSITIONS:
            recs += [{"span_mm": float(s), "y": r01_misalignment(float(s), position, driver=driver),
                      "series": f"{position}, {driver} drives", "source": "r01 slopes"}
                     for s in np.linspace(SPANS_MM[0], SPANS_MM[-1], 13)]
    figs.append(plots.ratio_sweep(recs, results, "s006", x="span_mm", x_label=r"span $L$ (mm)",
                                  y_label=r"$f_{sh,bend}$ over $b$ ($\mu$m)", log_x=False, variant="misalignment_vs_span",
                                  caption="spur (judged: FEM nodes vs r01 slopes, residual panel); f_sh_bend > 0: flanks open "
                                          f"at +X, load at the -X end; centred gear: 0 by symmetry\n{CAPTION}"))
    hel = [p for p in out["points"] if p["position"] == "offset" and (p["beta_deg"] > 0 or p["span_mm"] == SPAN_BETA_MM)]
    recs = []
    for p in hel:
        hands = (p["hand"],) if p["beta_deg"] else (1, -1)          # the spur point belongs to both hands
        recs += [{"beta_deg": p["beta_deg"], "y": p["f_sh_bend_um"], "series": f"hand {h:+d}, {p['driver']} drives",
                  "source": "FEM", "rel_err": p["f_sh_bend_um"] / p["f_sh_bend_r01_um"] - 1.0} for h in hands]
    for driver in ("pinion", "wheel"):
        for hand in (1, -1):
            recs += [{"beta_deg": float(b), "y": r01_misalignment(SPAN_BETA_MM, "offset", beta_deg=float(b), hand=hand,
                                                                   driver=driver),
                      "series": f"hand {hand:+d}, {driver} drives", "source": "r01 slopes"}
                     for b in np.linspace(0.0, BETA_SWEEP_DEG[-1], 13)]
    figs.append(plots.ratio_sweep(recs, results, "s006", x="beta_deg", x_label=r"helix angle $\beta$ (deg)",
                                  y_label=r"$f_{sh,bend}$ over $b$ ($\mu$m)", log_x=False, variant="misalignment_vs_beta",
                                  caption=f"gear at 0.3 L, L {SPAN_BETA_MM:g} mm; beta > 0: {J16} (couple F_a r_w as "
                                          f"ExternalMoment consumed by the FEM); beta 0 = spur\n{CAPTION}"))
    recs = []
    for b in out["bearings"]:
        if b["beta_deg"] == 0.0 and b["driver"] == "pinion":
            pin = "A" if b["x_mm"] == 0.0 else "B"
            recs.append({"span_mm": b["span_mm"], "y": 1e3 * b["psi_rad"], "series": f"{b['position']}, {b['shaft']}, pin {pin}",
                         "source": "FEM"})
    figs.append(plots.ratio_sweep(recs, results, "s006", x="span_mm", x_label=r"span $L$ (mm)",
                                  y_label=r"$\psi$ at the pin (mrad)", log_x=False, variant="psi_vs_span",
                                  caption=f"shaft slope at each rigid pin, spur, pinion drives (the range for s002)\n{CAPTION}"))
    return figs


if __name__ == "__main__":
    from axisforge.outputs._format import format_table
    res = run()
    print(format_table((("id", "id", "s"), ("criterion", "criterion", "s"), ("n", "n", "d"), ("n_pass", "pass", "d"),
                        ("largest", "largest", ".2e")), res["criteria"]))
    print(format_table((("position", "position", "s"), ("span_mm", "L", ".0f"), ("driver", "driver", "s"),
                        ("beta_deg", "beta", ".0f"), ("hand", "hand", "d"), ("f_sh_bend_um", "f_sh_bend um", ".3f"),
                        ("f_sh_bend_timoshenko_um", "f_sh_bend T um", ".3f"), ("twist_pinion_um", "twist um", ".3f")),
                       [p for p in res["points"] if p["beta_deg"] in (0.0, 30.0)]))
