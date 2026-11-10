"""Run of v003: ISO/TS 16281 bearings solved one at a time by AxisForge, against r03, r02 and the literature.

``python -m studies.v000_axisforge.v003_iso16281_load_zone.study`` writes into ``results/`` (git-ignored):
``v003_check.csv`` (check records of every bearing and load), ``v003_elements.csv`` (Q_j per element),
``v003_literature.csv`` (Hamrock 1983, Hernot 2000), ``v003_sign_check.csv`` (R-003) and the figures of
``studies/outputs``.
"""
from __future__ import annotations

import importlib
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from axisforge.mesh import BeamModelSettings
from axisforge.outputs._format import write_records_csv
from axisforge.outputs.solvers.bearings.load_distribution.load_distribution_results import bearing_result_record
from axisforge.results.fem_results.shaft_results import BearingNodeData
from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281.dispatch import resolve_solver_cls
from axisforge.solvers.machine_elements.shaft.fem_solvers.global_solver.rigid_support import RigidSupportFEMSolver

from studies.outputs.l000_axisforge.plots import bearing_loads as plots
from studies.outputs.l100_dry_contact.plots.contact import relative_errors
from studies.r00_analytical_references.r03_load_zone import (ring_load_distribution, solve_ring_displacement,
                                                             stribeck_constant)

from . import literature
from .check import MM, TOL_LOAD_ZONE, check, elements, r03_inputs
from .construction import SIGN_CHECK, angular_contact_7210, cylindrical_nu210, deep_groove_6210, sign_check_shaft

# importing contact_solver registers the ball and roller solvers with the dispatcher (a side effect only)
importlib.import_module("axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281.contact_solver")

RESULTS = Path(__file__).resolve().parent / "results"
S_UM = (0.0, 10.0, 20.0, 30.0, 40.0, 60.0)              # diametral clearance sweep [um]; 0 = Stribeck's limit
FR_N = (1000.0, 2000.0, 5000.0, 10000.0)                # radial loads of the sweep [N]
FR_SHOWN = (1000.0, 10000.0)                            # the two loads of the error figure
ACBB_FR_N = 3000.0                                      # angular contact sweep: F_r fixed, F_a varied
ACBB_FA_N = (4000.0, 6000.0, 10000.0, 20000.0, 40000.0)
CRB_FR_N = 8000.0
# Hamrock (L1, L2): his F_max comes from Stribeck's factor iterated with an approximate fit (Eq. (30)) and from
# Hamrock-Brewe contact fits whose printed values are not self-consistent (OPEN R5-09; r03 with exact and fitted
# Hertz moved delta_o by 1.7 %): Q_max 1 %, delta 2 %, fixed before the AxisForge run
TOL_HAMROCK_Q, TOL_HAMROCK_DELTA = 1e-2, 2e-2
CAPTION = "6210: 9 balls 12.7 mm, Dpw 70 mm, f 0.52, steel; local azimuth (0 on the load line), no convention tag"


def _distribute(bearing, node):
    """ISO/TS 16281 load distribution of one bearing at the loads of one node (``BearingNodeData``), ``psi = 0``.

    The AxisForge solver reads its loads from ``shaft_results.bearing_nodes``; only that list is given (the same
    form as s001 ``model.solve_bearing``; no documented one-bearing entry point: OPEN R5-11). No post-processing.
    """
    solver = resolve_solver_cls(bearing, label=bearing.label)(psi_input=True, postprocess=False)
    out = solver.solve(None, {bearing.label: bearing}, SimpleNamespace(bearing_nodes=[node]),
                       psi_override={bearing.label: 0.0})
    return out[bearing.label].load_distribution


def solve_bearing(bearing, Fr_N: float, Fa_N: float = 0.0):
    """AxisForge load distribution of one bearing under ``F_r`` along +Z (local azimuth = global) and ``F_a``.

    ``F_a > 0`` seats a 7210 row (its ``alpha_j`` grows above ``alpha_0``): the load ON the bearing, in the local
    frame. A FEM node carries the support force on the shaft instead, the opposite sign (R-003).
    """
    return _distribute(bearing, BearingNodeData(label=bearing.label, Fr_xz=Fr_N, Fr_xy=0.0, Fr=Fr_N, Fa=Fa_N))


def _solver_record(label: str, result) -> dict:
    """AxisForge's own solver report (SciPy root hybr/lm; not Newton): ok, residual, function evaluations."""
    r = bearing_result_record(result)
    return {"label": label, "ok": bool(r["ok"]), "residual_N": float(r["residual"]), "n_iter": int(r["n_iter"])}


def _psi(Z: int) -> np.ndarray:
    """Azimuths of ``Z`` equally spaced elements from the load line [rad] (AxisForge's local ``phi_j``)."""
    return 2.0 * math.pi * np.arange(Z) / Z


def dgbb_sweep():
    """6210 over S_UM x FR_N: check records, element records, sweep-figure records, solver records."""
    rows, elems, sweep, solver = [], [], [], []
    for s in S_UM:
        b = deep_groove_6210(s * 1e-3)
        for Fr in FR_N:
            label = f"6210 s {s:g} um Fr {Fr:g} N"
            r = solve_bearing(b, Fr)
            recs = check(b, r, label=label)
            rows += [{**x, "s_um": s, "Fr_N": Fr} for x in recs]
            elems += [{**e, "s_um": s, "Fr_N": Fr} for e in elements(b, r, label=label)]
            solver.append({**_solver_record(label, r), "s_um": s, "Fr_N": Fr})
            Q_max = float(np.max(r.rows[0].Q_j))
            eps = next(x for x in recs if x["id"] == "C5")["measured"]
            sweep += [{"s_um": s, "series": f"F_r {Fr:g} N", "source": "AxisForge", "quantity": "eps", "y": eps},
                      {"s_um": s, "series": f"F_r {Fr:g} N", "source": "AxisForge", "quantity": "stribeck",
                       "y": Q_max * b.Z / Fr}]
    # r03 curves on a fine clearance grid; the bearing is built per s because c_p (K_n) depends on s through
    # alpha_0 in the contact radii
    for s in np.linspace(0.0, S_UM[-1], 31):
        b = deep_groove_6210(s * 1e-3)
        inp = r03_inputs(b)
        psi = _psi(b.Z)
        for Fr in FR_N:
            own = solve_ring_displacement(Fr, psi_rad=psi, K_n=inp["K_n"], clearance_m=inp["clearance_m"])
            sweep += [{"s_um": float(s), "series": f"F_r {Fr:g} N", "source": "r03", "quantity": "eps",
                       "y": own["epsilon"]},
                      {"s_um": float(s), "series": f"F_r {Fr:g} N", "source": "r03", "quantity": "stribeck",
                       "y": float(np.max(own["Q_N"])) * b.Z / Fr}]
    return rows, elems, sweep, solver


def acbb_sweep():
    """7210 (alpha_0 40 deg), F_r fixed, F_a varied: RECORD records (fixed-angle r03 vs load-dependent angle)."""
    b = angular_contact_7210()
    # F_a > 0 seats the row (local frame of the bearing load); a FEM node would carry the opposite sign (R-003)
    rows, elems, solver = [], [], []
    for Fa in ACBB_FA_N:
        label = f"7210 Fr {ACBB_FR_N:g} N Fa {Fa:g} N"
        r = solve_bearing(b, ACBB_FR_N, Fa)
        ratio = ACBB_FR_N * math.tan(b.alpha_0) / Fa
        rows += [{**x, "Fr_tan_a_over_Fa": ratio, "Fa_N": Fa} for x in check(b, r, label=label)]
        elems += [{**e, "Fa_N": Fa} for e in elements(b, r, label=label)]
        solver.append({**_solver_record(label, r), "Fa_N": Fa,
                       "alpha_j_min_deg": math.degrees(min(r.rows[0].alpha_j)),
                       "alpha_j_max_deg": math.degrees(max(r.rows[0].alpha_j))})
    return rows, elems, solver


def crb_case():
    """NU210 (ISO/TS 16281 profile in laminae) under F_r: RECORD records against r03 with n = 10/9, K_n = c_L."""
    b = cylindrical_nu210()
    r = solve_bearing(b, CRB_FR_N)
    label = f"NU210 s {b.s * 1e3:g} um Fr {CRB_FR_N:g} N"
    return check(b, r, label=label), elements(b, r, label=label), [_solver_record(label, r)]


def literature_series() -> list[dict]:
    """L1-L2 Hamrock (AxisForge vs printed), L3 Hamrock Z (RECORD), L4-L7 Hernot (r03 vs printed)."""
    hamrock, hernot = literature.cases()
    out = []
    b = hamrock.build()
    p = literature.published("hamrock1983_sec14_2")
    r = solve_bearing(b, p["F_r"]["value"])
    Q_max = float(np.max(r.rows[0].Q_j))
    delta = float(np.max(r.rows[0].delta_j)) * MM
    for lid, quantity, unit, meas, ref, tol in (
            ("L1", "F_max", "N", Q_max, p["F_max"]["value"], TOL_HAMROCK_Q),
            ("L2", "delta of the most loaded ball", "m", delta, p["delta"]["value"], TOL_HAMROCK_DELTA),
            ("L3", "Z = F_max n / F_r", "-", Q_max * b.Z / p["F_r"]["value"], p["Z"]["value"], float("nan"))):
        err = meas / ref - 1.0
        verdict = "RECORD" if math.isnan(tol) else ("PASS" if abs(err) <= tol else "MISS")
        out.append({"case": "v003", "source": hamrock.name, "id": lid, "quantity": quantity, "unit": unit,
                    "measured": meas, "reference": ref, "rel_err": err, "tol": tol, "verdict": verdict,
                    "judge": "AxisForge vs printed"})
    # Hernot: r03 in the continuous (integral) form Hernot uses (his Eq. (7)), half a unit of each printed digit
    h = hernot.build()
    psi = _psi(h["Z"])
    own = solve_ring_displacement(h["Fr_N"], h["Fa_N"], psi_rad=psi, K_n=h["K_n"], n=h["n"], alpha_rad=h["alpha_rad"],
                                  continuous=True)
    q = literature.published("hernot2000_table1")
    for lid, quantity, unit, meas, sym, half in (
            ("L4", "Q_max", "N", own["Q_max_N"], "Q_max", 0.5),
            ("L5", "delta_a", "mm", own["delta_a_m"] / MM, "delta_a", 0.5e-5),
            ("L6", "delta_r", "mm", own["delta_r_m"] / MM, "delta_r", 0.5e-5),
            ("L7", "eps", "-", own["epsilon"], "eps", 0.5e-5)):
        ref = q[sym]["value"]
        out.append({"case": "v003", "source": hernot.name, "id": lid, "quantity": quantity, "unit": unit,
                    "measured": meas, "reference": ref, "rel_err": meas / ref - 1.0, "tol": half / abs(ref),
                    "verdict": "PASS" if abs(meas - ref) <= half else "MISS", "judge": "r03 vs printed"})
    return out, (b, r)


def sign_check() -> list[dict]:
    """R-003: where Q_max lands in the global frame against the direction of the load on the shaft. RECORD.

    A shaft on two 6210 (``sign_check_shaft``) with one radial load, ``RigidSupportFEMSolver``, then each bearing
    node through the ISO/TS 16281 solver (``psi = 0``, ``Fa = 0``). Physics: the shaft pushes the inner ring
    along the load, so the most loaded ball sits in the load direction ``phi_exp = atan2(F_y, F_z)`` (D-0010:
    from +Z toward +Y). With the node force ``F_r`` = support force ON THE SHAFT (what the FEM stores), the
    inner ring balances when ``r = sum Q_j cos(alpha_j) (cos phi_j, sin phi_j) + (F_r,xz, F_r,xy) = 0``.
    Recorded: ``phi_global(Q_max)``, ``phi_exp``, their difference and ``|r| / |F_r|`` (2.000 when Q_max sits
    opposite the load on the shaft). ``Q_max`` at ``phi_Fr`` holds by construction (the element set turns with
    ``phi_Fr``, R-396): the evidence is the offset and the residual. Radial only (``F_a = 0``): the axial half of
    R-003 is not exercised (OPEN R5-15). Nothing is changed in AxisForge: convention-dependent (R-003).
    """
    beam = BeamModelSettings("timoshenko", "cowper", "single_point")
    rows = []
    for theta in SIGN_CHECK["load_theta_deg"]:
        system = sign_check_shaft(theta)
        load = system.radial_loads[0]
        phi_exp = math.atan2(load.Fy, load.Fz) % (2.0 * math.pi)
        shaft_results = RigidSupportFEMSolver(beam).solve(system)
        for node in shaft_results.bearing_nodes:
            bearing = next(b for b in system.bearings if b.label == node.label)
            res = _distribute(bearing, node)                  # the FEM's own BearingNodeData
            row = res.rows[0]
            Q, alpha = np.asarray(row.Q_j), np.asarray(row.alpha_j)
            phi_g = np.asarray(res.phi_j_global(0))
            radial = Q * np.cos(alpha)
            r_xz = float(np.sum(radial * np.cos(phi_g))) + node.Fr_xz
            r_xy = float(np.sum(radial * np.sin(phi_g))) + node.Fr_xy
            Fr = math.hypot(node.Fr_xz, node.Fr_xy)
            j = int(np.argmax(Q))
            off = (phi_g[j] - phi_exp + math.pi) % (2.0 * math.pi) - math.pi
            rows.append({"theta_deg": theta, "label": node.label, "solver_ok": bool(res.ok), "Fr_xz_N": node.Fr_xz,
                         "Fr_xy_N": node.Fr_xy, "phi_Fr_deg": math.degrees(res.phi_Fr),
                         "phi_load_deg": math.degrees(phi_exp), "phi_Qmax_global_deg": math.degrees(phi_g[j]),
                         "offset_deg": math.degrees(off), "residual_over_Fr": math.hypot(r_xz, r_xy) / Fr,
                         "Q_max_N": float(Q[j]), "tag": "convention-dependent (R-003)", "verdict": "RECORD",
                         "phi_global_deg": [math.degrees(p) for p in phi_g], "Q_N": [float(x) for x in Q]})
    return rows


def _zone_curve(bearing, result, series: str) -> list[dict]:
    """r03 at AxisForge's own displacement on a 1-deg azimuth grid, and the zero-clearance shape Q_max cos^n."""
    inp = r03_inputs(bearing)
    phi = np.radians(np.arange(-180.0, 180.5, 1.0))
    da = result.delta_a * MM if inp["kind"] == "angular_contact" else 0.0
    z = ring_load_distribution(result.delta_r * MM, da, psi_rad=phi, K_n=inp["K_n"], n=inp["n"],
                               alpha_rad=inp["alpha_rad"], clearance_m=inp["clearance_m"])
    shape = z["Q_max_N"] * np.clip(np.cos(phi), 0.0, None) ** inp["n"]
    return ([{"series": series, "source": "r03 at AxisForge delta_r", "phi_deg": float(np.degrees(p)), "Q_N": float(q)}
             for p, q in zip(phi, z["Q_N"])]
            + [{"series": series, "source": "Q_max cos^n (zero clearance)", "phi_deg": float(np.degrees(p)),
                "Q_N": float(q)} for p, q in zip(phi, shape)])


def _af_points(result, series: str, *, global_frame: bool = False) -> list[dict]:
    row = result.rows[0]
    phi = result.phi_j_global(0) if global_frame else row.phi_j
    return [{"series": series, "source": "AxisForge", "phi_deg": math.degrees(p), "Q_N": float(q)}
            for p, q in zip(phi, row.Q_j)]


def figures(rows, sweep, lit, hamrock, acbb_rows, sign) -> list[Path]:
    """The figures of README section 9, through studies/outputs."""
    out = []
    # F1: element loads of the 6210, zero clearance and 40 um, F_r 5000 N
    recs = []
    for s in (0.0, 40.0):
        b = deep_groove_6210(s * 1e-3)
        r = solve_bearing(b, 5000.0)
        recs += _af_points(r, f"s {s:g} um") + _zone_curve(b, r, f"s {s:g} um")
    out.append(plots.element_loads(recs, RESULTS, "v003", variant="6210",
                                   caption=CAPTION + "; F_r 5000 N along phi = 0"))
    # F2: errors of C1-C6 against the clearance, two loads, tolerance dashed
    err = [{"s_um": r["s_um"], "quantity": f"{r['id']} {r['quantity']}", "series": f"F_r {r['Fr_N']:g} N",
            "rel_err": r["rel_err"], "tol": r["tol"], "verdict": r["verdict"]}
           for r in rows if r["Fr_N"] in FR_SHOWN and r["id"] != "C7"]
    out.append(relative_errors(err, RESULTS, "v003", x="s_um", x_label=r"clearance $s$ [um]", variant="6210",
                               log_x=False))
    # F3: eps and Stribeck's Z Q_max / F_r against the clearance, AxisForge on r03, limits at s = 0
    out.append(plots.load_zone_sweep(sweep, RESULTS, "v003", x="s_um", x_label=r"diametral clearance $s$ [um]",
                                     panels={"eps": r"$\varepsilon$ [-]", "stribeck": r"$Z Q_{max} / F_r$ [-]"},
                                     limits={"eps": 0.5, "stribeck": round(stribeck_constant(), 2)}, variant="6210",
                                     caption=CAPTION + "; 4.37 = continuous limit, Z = 9 discrete gives 4.38"))
    # F4: Hamrock's bearing with the printed F_max (open marker at phi = 0)
    b, r = hamrock
    recs = _af_points(r, "Hamrock 14.2") + _zone_curve(b, r, "Hamrock 14.2")
    recs.append({"series": "Hamrock 14.2", "source": "published", "phi_deg": 0.0,
                 "Q_N": next(x["reference"] for x in lit if x["id"] == "L1")})
    out.append(plots.element_loads(recs, RESULTS, "v003", variant="hamrock1983",
                                   caption="Hamrock 1983 sec. 14.2: Z 9, d 12.7 mm, d_e 65 mm, P_d 15 um, F_r 8900 N; "
                                           "local azimuth"))
    # F5: angular contact RECORD: element loads at the lowest thrust, errors against F_r tan(a)/F_a
    ba = angular_contact_7210()
    ra = solve_bearing(ba, ACBB_FR_N, ACBB_FA_N[0])
    recs = _af_points(ra, f"7210 F_a {ACBB_FA_N[0]:g} N") + _zone_curve(ba, ra, f"7210 F_a {ACBB_FA_N[0]:g} N")
    out.append(plots.element_loads([x for x in recs if "cos^n" not in x["source"]], RESULTS, "v003", variant="7210",
                                   caption=f"7210 RECORD: alpha_0 40 deg, Z 13, F_r {ACBB_FR_N:g} N; AxisForge angle "
                                           "follows the load, r03 fixed; local azimuth"))
    err = [{"ratio": r["Fr_tan_a_over_Fa"], "quantity": f"{r['id']} {r['quantity']}", "series": "7210 (RECORD)",
            "rel_err": r["rel_err"], "tol": r["tol"], "verdict": r["verdict"]} for r in acbb_rows if r["id"] != "C7"]
    out.append(relative_errors(err, RESULTS, "v003", x="ratio", x_label=r"$F_r \tan\alpha_0 / F_a$ [-]",
                               variant="7210", log_x=False))
    # F6: sign check, global frame, polar, load direction marked (R-003)
    for rec in sign:
        if rec["label"] != "6210_A":
            continue
        pts = [{"series": f"load at {rec['phi_load_deg']:.0f} deg", "source": "AxisForge", "phi_deg": p, "Q_N": q}
               for p, q in zip(rec["phi_global_deg"], rec["Q_N"])]
        out.append(plots.element_loads(pts, RESULTS, "v003", polar=True,
                                       marks_deg=(rec["phi_load_deg"], rec["phi_Fr_deg"]),
                                       variant=f"sign_check_{rec['theta_deg']:g}",
                                       caption=f"radius Q [N]; global azimuth D-0010 (0 on +Z, toward +Y); dash-dot: load on "
                                               f"the shaft {rec['phi_load_deg']:.0f} deg and phi_Fr "
                                               f"{rec['phi_Fr_deg'] % 360:.0f} deg; offset "
                                               f"{abs(rec['offset_deg']):.0f} deg, |r|/F_r = "
                                               f"{rec['residual_over_Fr']:.2f} (inner-ring balance); "
                                               "convention-dependent (R-003)"))
    return out


def run() -> dict:
    """Every series, records written to ``results/``, figures drawn; returns the records and figure paths."""
    rows, elems, sweep, solver = dgbb_sweep()
    acbb_rows, acbb_elems, acbb_solver = acbb_sweep()
    crb_rows, crb_elems, crb_solver = crb_case()
    lit, hamrock = literature_series()
    sign = sign_check()
    RESULTS.mkdir(exist_ok=True)
    write_records_csv(RESULTS / "v003_check.csv", rows + acbb_rows + crb_rows)
    write_records_csv(RESULTS / "v003_elements.csv", elems + acbb_elems + crb_elems)
    write_records_csv(RESULTS / "v003_literature.csv", lit)
    write_records_csv(RESULTS / "v003_solver.csv", solver + acbb_solver + crb_solver)
    write_records_csv(RESULTS / "v003_sign_check.csv",
                      [{k: v for k, v in r.items() if k not in ("phi_global_deg", "Q_N")} for r in sign])
    figs = figures(rows, sweep, lit, hamrock, acbb_rows, sign)
    return {"check": rows, "acbb": acbb_rows, "crb": crb_rows, "elements": elems, "literature": lit, "sign": sign,
            "solver": solver + acbb_solver + crb_solver, "figures": figs}


if __name__ == "__main__":
    out = run()
    judged = [r for r in out["check"] + out["literature"] if r["verdict"] != "RECORD"]
    print(f"v003: {sum(r['verdict'] == 'PASS' for r in judged)} PASS, {sum(r['verdict'] == 'MISS' for r in judged)} "
          f"MISS of {len(judged)} judged; worst |rel_err| {max(abs(r['rel_err']) for r in judged):.2e} "
          f"(tol {TOL_LOAD_ZONE:g} for C1-C6)")
