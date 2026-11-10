"""Run of s003: a shaft on three ISO/TS 16281 ball bearings against rigid supports, over seat offset e and clearance s.

``python -m studies.s000_axisforge.s003_hyperstatic_shaft.study`` writes into ``results/`` (git-ignored):
``s003_law.csv`` (S1: the ball sums at the ``solve_contact`` solution), ``s003_points.csv`` (one row per bearing
and (s, e): rigid and coupled reaction, d, Q_max, slope, zone azimuths, Newton summary; gear rows ``bearing = 0``),
``s003_v005.csv`` / ``s003_v003.csv`` (v first at s003's own conditions), ``s003_limits.csv`` (linear and stiff
limits), ``s003_criteria.csv``, ``s003_newton_history.csv``, ``s003_elements.csv`` and the figures of
``studies/outputs``.

Every number of a coupled solve comes from :func:`model.solve_shaft` (``numerical_solvers.newton``); the rigid
reference is AxisForge's ``RigidSupportFEMSolver`` with the seat offset as a settlement (v005 ``study.solve``).
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from axisforge.outputs._format import write_records_csv
from scipy.optimize import brentq
from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281.contact_solver import ISO16281BallSolver

from numerical_solvers import NewtonOptions, newton
from studies.outputs.common.plots.convergence import newton_history
from studies.outputs.l000_axisforge.plots import shaft as plots
from studies.outputs.l000_axisforge.plots.bearing_loads import element_loads
from studies.outputs.l100_dry_contact.plots.contact import relative_errors
from studies.s000_axisforge.s001_bearing_clearance_and_contact_angle import loaded_zone
from studies.s000_axisforge.s004_bearing_compliance import rigid_line, ring_displacement
from studies.v000_axisforge.v001_shaft_uniform.check import node_value
from studies.v000_axisforge.v001_shaft_uniform.construction import E_MPA
from studies.v000_axisforge.v003_iso16281_load_zone import check as v003_check
from studies.v000_axisforge.v005_shaft_three_supports import along_shaft, check as v005_check
# OPEN R5-162: v005's condensed beam (hertz_problem), settled solve (solve) and Hertz solve used by an s (R5-54)
from studies.v000_axisforge.v005_shaft_three_supports.construction import DIAMETER_MM
from studies.v000_axisforge.v005_shaft_three_supports.study import EB, hertz_problem, solve, solve_hertz

from .construction import E_UM, N_ELEMENTS, P_N, S_MM, SUPPORTS_MM, X_GEAR_MM, X_MIDDLE_MM, bearings, shaft
from .model import NEWTON, ball_loads, solve_shaft

RESULTS = Path(__file__).resolve().parent / "results"
R003 = "convention-dependent (R-003)"
UM = 1e3                                  # [um/mm], record keys *_um only
# criteria fixed before the first run, from proposals.md s003 (README 6)
TOL_LAW = 1e-9                            # S1: ball sums at the solve_contact solution (tol 1e-12) vs Fr, rel
TOL_LINEAR = 1e-8                         # S3: linear-law limit vs v005 (check and closed form), rel
TOL_EQUILIBRIUM = 1e-8                    # S4: global force and moment balance, rel to P and P l
CONTACT_TOL = 1e-12                       # solve_contact root tolerance of S1 (proposals.md)
LAW_LOADS_N = (100.0, 1000.0, 5000.0)     # S1 loads per clearance
K_LINEAR_N_PER_MM = 3.0e4                 # S3: v005's middle spring (about the shaft's own stiffness at 110 mm)
LIMIT_E_UM = (-50.0, 0.0, 50.0)
STIFF_SCALES = (1.0, 1e2, 1e4, 1e6)       # bearing force x scale, s = 0: the rigid-support limit
# Judges ADDED AT REVIEW, after the first run (physics review MUST 1 and 2; as R5-104, so not pre-run criteria).
# S6 / S9: at s = 0 the ISO law is exactly v005's Hertz law F = K |d|^1.5 with K = scale c_p sum_{cos phi_j > 0}
# cos^2.5 phi_j (delta_j = |d| cos phi_j), solved on the same condensed beam with an analytic Jacobian
# (``hertz_judge``) and, at e = 0, by v005 ``solve_hertz`` itself: forces 1e-8 of P, d 1e-8 of max |d|. The first
# run's "1e-4 of P at scale 1e6" (read from a prototype, R5-167) is dropped; the distance to the rigid FEM and its
# -2/3 slope stay RECORD (S6r). Scale 1e9 stalls at the round-off floor of sqrt(U^2 + V^2) - A (delta / A ~ 1e-8,
# R5-169), so the series stops at 1e6
TOL_HERTZ = 1e-8
# S8: bearing 3 inside its clearance at e = 0 leaves the shaft on bearings 1, 2 with the gear at mid span:
# F = (P/2, P/2, 0) and d_3 - d_1 = P l1^2 l_ov / (16 EI) (end slope P l1^2 / (16 EI) of the simply supported
# span, carried along the unloaded overhang l_ov = x_3 - x_2; d_1 = d_2: equal loads on equal bearings)
TOL_LIFTOFF = 1e-8
EI_NMM2 = E_MPA * math.pi * DIAMETER_MM ** 4 / 64.0       # [N.mm^2] the FEM's uniform EI (Euler-Bernoulli)
CAPTION = (f"v005 shaft (d 30 mm, S355, bearings at {'/'.join(f'{x:g}' for x in SUPPORTS_MM)} mm), P {P_N:g} N "
           f"along -Y at {X_GEAR_MM:g} mm; three s001 6210-like DGBB, middle seat offset e along +Y; EB, "
           f"n {N_ELEMENTS}")


def _rec(cid: str, quantity: str, value: float, tol: float | None, **extra) -> dict:
    """One criterion record: ``rel_err`` = ``value`` (already relative), verdict against ``tol`` (None: RECORD)."""
    verdict = "RECORD" if tol is None else ("PASS" if abs(value) <= tol else "MISS")
    return {"case": "s003", "id": cid, "quantity": quantity, "rel_err": float(value),
            "tol": math.nan if tol is None else tol, "verdict": verdict, **extra}


def hertz_K(bearing, scale: float = 1.0) -> float:
    """[N/mm^1.5] K of the s = 0 law F = K |d|^1.5: ``scale c_p sum cos^2.5 phi_j`` over the balls with cos phi_j > 0."""
    c = np.cos(np.asarray(bearing.phi_j, dtype=float))
    return scale * bearing.cp * float(np.sum(np.clip(c, 0.0, None) ** 2.5))


def hertz_judge(problem: dict, seat_mm: dict, K: float) -> tuple[np.ndarray, np.ndarray]:
    """Judge of the s = 0 solves: the condensed beam on the power law -K sign(d) |d|^1.5, analytic Jacobian.

    v005 ``hertz_problem``'s residual with the seat offset: ``K_c (d + e) - f_c + K sign(d) |d|^1.5`` [N],
    Jacobian ``K_c + diag(1.5 K |d|^0.5)``; independent of ``ball_loads`` and of its finite differences.

    Returns
    -------
    (ndarray, ndarray)
        F [N] on the shaft along +Y and d [mm], in position order.
    """
    pos = list(problem["positions"])
    e = np.array([seat_mm.get(x, 0.0) for x in pos])
    K_c, f_c = problem["K_c"], problem["f_c"]
    P = float(problem["f_scale"][0])
    d0 = (P / len(pos) / K) ** (2.0 / 3.0)                  # [mm] one bearing at a third of the load
    x0 = np.linalg.solve(K_c + (P / len(pos) / d0) * np.eye(len(pos)), f_c - K_c @ e)
    res = newton(lambda d: K_c @ (d + e) - f_c + K * np.sign(d) * np.abs(d) ** 1.5, x0,
                 jacobian=lambda d: K_c + np.diag(1.5 * K * np.abs(d) ** 0.5),
                 options=NewtonOptions(x_scale=np.full(len(pos), d0), f_scale=np.full(len(pos), P),
                                       residual_abs=1e-12, step_rel=1e-14), name="s003_hertz_judge")
    if not res.converged:
        raise RuntimeError(f"Hertz judge did not converge: {res.reason}")
    d = np.asarray(res.x)[np.argsort(pos)]
    return -K * np.sign(d) * np.abs(d) ** 1.5, d


def _vs_hertz(cid: str, c: dict, F_j, d_j, **ident) -> list[dict]:
    """Two judged records of a coupled solve against the power-law judge: forces over P, d over max |d|."""
    return [_rec(cid, "F vs Hertz judge", float(np.max(np.abs(c["F_N"] - F_j))) / P_N, TOL_HERTZ, **ident),
            _rec(cid, "d vs Hertz judge", float(np.max(np.abs(c["d_mm"] - d_j))) / float(np.max(np.abs(d_j))),
                 TOL_HERTZ, **ident)]


def law_check() -> list[dict]:
    """S1: :func:`model.ball_loads` at the ``solve_contact`` solution gives the applied Fr (and delta_a = -A sin a0).

    ``ISO16281BallSolver(tol=1e-12).solve_contact`` (force-controlled, both balances, AxisForge) at Fr, Fa = 0;
    the model sums Eq. (16) at its delta_r must give Fr to ``TOL_LAW``. Its delta_a against the model's fixed
    ``-A sin(alpha_0)`` (Eq. (17) by symmetry) is RECORD [mm].
    """
    out = []
    for s in S_MM:
        b = bearings(s)[SUPPORTS_MM[0]]
        for Fr in LAW_LOADS_N:
            sol = ISO16281BallSolver(tol=CONTACT_TOL).solve_contact(b, Fr, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            F_model = ball_loads(b, sol.delta_r)[0]
            out.append(_rec("S1", "Fr of the ball sums at solve_contact's delta_r vs Fr", F_model / Fr - 1.0, TOL_LAW,
                            s_mm=s, Fr_N=Fr, delta_r_mm=sol.delta_r, ok=sol.ok, residual_N=sol.residual,
                            delta_a_minus_model_mm=sol.delta_a + b.A * math.sin(b.alpha_0)))
    return out


def point(problem: dict, s_mm: float, e_um: float) -> dict:
    """One (s, e): the coupled solve, the rigid FEM, v005.check, the judges, the bearings through s004 / v003."""
    system = shaft()
    B = bearings(s_mm)
    seat = {X_MIDDLE_MM: e_um / UM}
    t0 = time.perf_counter()
    c = solve_shaft(problem, B, seat_mm=seat, label=f"s {s_mm * UM:g} um, e {e_um:+g} um")
    wall = time.perf_counter() - t0
    res, pos = c["newton"], c["positions"]
    rigid = solve(system, EB, N_ELEMENTS, settlements_mm={"XY": seat})
    order = np.argsort([b.position for b in system.bearings])
    R_rigid = np.asarray(rigid.R_xy)[order]
    R_fem = np.asarray(c["result"].R_xy)[order]               # AxisForge post-processing of the coupled field
    ident = {"s_mm": s_mm, "e_um": e_um}
    v005 = [{**r, **ident} for r in v005_check(system, c["result"], EB, springs=c["secant"],
                                               settlements_mm=c["settlements_mm"], case="s003",
                                               label="secant springs", tag=R003)]
    # S4: statics of the coupled state (F on the shaft along +Y balances P along -Y), and R of AxisForge = F
    F, x = c["F_N"], np.asarray(pos)
    crit = [_rec("S4", "sum F - P", (F.sum() - P_N) / P_N, TOL_EQUILIBRIUM, **ident),
            _rec("S4", "sum F (x - x_1) - P (x_P - x_1)",
                 (F @ (x - x[0]) - P_N * (X_GEAR_MM - x[0])) / (P_N * (x[-1] - x[0])), TOL_EQUILIBRIUM, **ident),
            _rec("S4", "AxisForge R_xy of the coupled field - F", float(np.max(np.abs(R_fem - F))) / P_N,
                 TOL_EQUILIBRIUM, **ident)]
    F_hertz = None
    if s_mm == 0.0:                                          # S9: the exact power-law judge (added at review)
        F_hertz, d_hertz = hertz_judge(problem, seat, hertz_K(B[pos[0]]))
        crit += _vs_hertz("S9", c, F_hertz, d_hertz, **ident)
    if e_um == 0.0 and s_mm >= 0.010:                        # S8: bearing 3 inside its clearance (added at review)
        D = P_N * (x[1] - x[0]) ** 2 * (x[2] - x[1]) / (16.0 * EI_NMM2)
        crit += [_rec("S8", "F vs (P/2, P/2, 0)", float(np.max(np.abs(F - np.array([P_N / 2, P_N / 2, 0.0])))) / P_N,
                      TOL_LIFTOFF, **ident),
                 _rec("S8", "d_3 - d_1 vs P l1^2 l_ov / (16 EI)", (c["d_mm"][2] - c["d_mm"][0]) / D - 1.0,
                      TOL_LIFTOFF, closed_form_um=D * UM, d1_um=c["d_mm"][0] * UM, d3_um=c["d_mm"][2] * UM,
                      half_clearance_um=s_mm / 2 * UM, **ident)]
    # the rigid-body line through the outer bearing centres, at the middle bearing (s004 rigid_line, plane XY)
    line, _ = rigid_line(np.array([x[1]]), x[0], x[2], (c["u_mm"][0], 0.0), (c["u_mm"][2], 0.0))
    rows, v003, elements = [], [], []
    for k, xk in enumerate(pos):
        b = B[xk]
        theta = node_value(c["result"], xk, "theta_xy")
        Q = c["Q_j_N"][k]
        loaded = F[k] != 0.0
        # a cage fixed in space, half a pitch off the load line (odd Z, reversed load), at the same |d|
        shifted = SimpleNamespace(A=b.A, alpha_0=b.alpha_0, cp=b.cp, phi_j=np.asarray(b.phi_j) + math.pi / b.Z)
        row = {**ident, "x_mm": xk, "bearing": k + 1, "R_rigid_N": R_rigid[k], "R_N": F[k],
               "R_over_P": F[k] / P_N, "R_rigid_over_P": R_rigid[k] / P_N, "d_um": c["d_mm"][k] * UM,
               "d_over_half_clearance": c["d_mm"][k] / (s_mm / 2.0) if s_mm > 0.0 else math.nan,
               "u_um": c["u_mm"][k] * UM, "Q_max_N": float(Q.max()), "n_loaded": int(np.sum(Q > 0.0)),
               "unloaded": not loaded, "theta_mrad": 1e3 * theta,
               "theta_rigid_mrad": 1e3 * node_value(rigid, xk, "theta_xy"),
               # AxisForge's tilt projection psi . R_hat, R_hat the support force (contact_solver.py:197-199); seen
               # from the physical zone (180 deg off) the tilt is -psi_in (s004 review S1)
               "psi_in_axisforge_mrad": 1e3 * theta * math.copysign(1.0, F[k]) if loaded else math.nan,
               # zone centre, D-0010 theta from +Y toward +Z: physically on the side the shaft centre moves toward
               "zone_theta_deg": (0.0 if c["d_mm"][k] > 0.0 else 180.0) if loaded else math.nan,
               "cage_half_pitch_Fr_change": (ball_loads(shifted, abs(c["d_mm"][k]))[0] / abs(F[k]) - 1.0
                                             if loaded else math.nan),
               "F_hertz_N": F_hertz[k] if F_hertz is not None else math.nan,
               "newton_converged": res.converged, "newton_reason": res.reason, "newton_iter": res.n_iter,
               "newton_residual_norm": res.residual_norm, "newton_n_residual": res.n_residual_evaluations,
               "newton_fallback": res.fallback, "wall_s": wall, "tag": R003}
        if loaded:
            # the same bearing solved force-controlled by AxisForge at the coupled force (s004, psi = 0)
            ring = ring_displacement(b, (F[k], 0.0), (0.0, 0.0))
            row.update(ring_d_um=float(ring["d_yz"][0]) * UM, ring_ok=ring["ok"], ring_balanced=ring["balanced"])
            if ring["analysis"] is not None:                # AxisForge's zone: phi_Fr of the node force (+R), R-003
                phi = math.degrees(ring["analysis"].load_distribution.phi_Fr)          # from +Z toward +Y
                row.update(phi_Fr_axisforge_deg=phi, zone_theta_axisforge_deg=(90.0 - phi) % 360.0)
                v003 += [{**r, **ident, "bearing": k + 1}
                         for r in v003_check(b, ring["analysis"].load_distribution, case="s003", label=b.label)]
            # sign consistency with s004 (the same -1 sign, R5-58); delta_r is invariant under R -> -R, so this
            # cannot detect R-003 (bearing-expert S3)
            crit.append(_rec("S5", "sign of s004 ring_displacement d vs model d (consistency, not R-003)", 0.0 if
                             math.copysign(1.0, ring["d_yz"][0]) == math.copysign(1.0, c["d_mm"][k]) else 1.0, 0.0,
                             **ident, bearing=k + 1,
                             note=f"|d| ring {abs(ring['d_yz'][0]) * UM:.6f} um vs model {abs(c['d_mm'][k]) * UM:.6f} um"))
        if xk == X_MIDDLE_MM:
            row["u_minus_line_um"] = (c["u_mm"][k] - float(line[0, 0])) * UM
            row["e_minus_rigid_line_um"] = e_um                  # rigid supports: the outer centres stay at 0
        rows.append(row)
        elements += [{**ident, "bearing": k + 1, "j": j, "phi_local_deg": math.degrees(p), "Q_N": float(q)}
                     for j, (p, q) in enumerate(zip(b.phi_j, Q))]
    gear = {**ident, "x_mm": X_GEAR_MM, "bearing": 0, "v_um": node_value(c["result"], X_GEAR_MM, "v_xy") * UM,
            "v_rigid_um": node_value(rigid, X_GEAR_MM, "v_xy") * UM,
            "theta_mrad": 1e3 * node_value(c["result"], X_GEAR_MM, "theta_xy"),
            "theta_rigid_mrad": 1e3 * node_value(rigid, X_GEAR_MM, "theta_xy")}
    return {"rows": rows, "gear": gear, "v005": v005, "v003": v003, "criteria": crit, "elements": elements,
            "newton": res, "coupled": c, "rigid": rigid}


def linear_limit(problem: dict) -> list[dict]:
    """S3: the Newton machinery with a linear law ``F = -k d`` vs v005 (its check against r01, and its solve).

    k = ``K_LINEAR_N_PER_MM`` at every bearing, middle seat at e: v005.check of the field (C1-C3 rel errors) and
    the forces against v005 ``study.solve`` with the same springs and settlement, both within ``TOL_LINEAR``.
    """
    system = shaft()
    springs = {x: (K_LINEAR_N_PER_MM, 0.0) for x in SUPPORTS_MM}
    order = np.argsort([b.position for b in system.bearings])
    out = []
    for e_um in LIMIT_E_UM:
        settle = {"XY": {X_MIDDLE_MM: e_um / UM}}
        c = solve_shaft(problem, bearings(0.0), seat_mm=settle["XY"], k_linear_N_per_mm=K_LINEAR_N_PER_MM,
                        label=f"linear, e {e_um:+g} um")
        worst = max(abs(r["rel_err"]) for r in v005_check(system, c["result"], EB, springs=springs,
                                                          settlements_mm=settle, case="s003")
                    if r["verdict"] != "RECORD")
        out.append(_rec("S3", "linear law: v005.check worst judged rel_err", worst, TOL_LINEAR, e_um=e_um,
                        newton_reason=c["newton"].reason, newton_iter=c["newton"].n_iter))
        R = np.asarray(solve(system, EB, N_ELEMENTS, springs=springs, settlements_mm=settle).R_xy)[order]
        out.append(_rec("S3", "linear law: forces vs v005 study.solve", float(np.max(np.abs(c["F_N"] - R))) / P_N,
                        TOL_LINEAR, e_um=e_um))
    return out


def stiff_limit(problem: dict) -> list[dict]:
    """S6: bearing forces x scale, s = 0, against the power-law judge (judged) and the rigid FEM (S6r, RECORD).

    S6 (added at review): every scale against :func:`hertz_judge` at the same scale, ``TOL_HERTZ``; at e = 0 also
    against v005 ``solve_hertz`` (S6v). S6r: the distance to the rigid three-support FEM at the same e; the bearing
    displacements fall as scale^(-2/3) (Q ~ delta^1.5), so the force difference of a linear beam does too;
    ``slope_vs_previous`` = log(err_k / err_{k-1}) / log(scale_k / scale_{k-1}) (-2/3 expected). s = 0 only: with
    s > 0 the limit is rigid supports with play s/2, not plain rigid supports.
    """
    system = shaft()
    order = np.argsort([b.position for b in system.bearings])
    B = bearings(0.0)
    out = []
    for e_um in LIMIT_E_UM:
        seat = {X_MIDDLE_MM: e_um / UM}
        R = np.asarray(solve(system, EB, N_ELEMENTS, settlements_mm={"XY": seat}).R_xy)[order]
        prev = None
        for sc in STIFF_SCALES:
            c = solve_shaft(problem, B, seat_mm=seat, scale=sc, label=f"x {sc:g}, e {e_um:+g} um")
            K = hertz_K(B[SUPPORTS_MM[0]], sc)
            out += _vs_hertz("S6", c, *hertz_judge(problem, seat, K), e_um=e_um, scale=sc,
                             newton_reason=c["newton"].reason, newton_iter=c["newton"].n_iter)
            if e_um == 0.0:                                  # v005's own Hertz solve (it has no seat offset)
                _, res_v, prob_v = solve_hertz(system, EB, N_ELEMENTS, K=K, n=1.5, label=f"v005 x {sc:g}")
                F_v = prob_v["force"](res_v.x)[np.argsort(prob_v["positions"])]
                out.append(_rec("S6v", "F vs v005 solve_hertz", float(np.max(np.abs(c["F_N"] - F_v))) / P_N,
                                TOL_HERTZ, e_um=e_um, scale=sc, v005_newton_reason=res_v.reason))
            err = float(np.max(np.abs(c["F_N"] - R))) / P_N
            slope = math.log(err / prev[1]) / math.log(sc / prev[0]) if prev else math.nan
            out.append(_rec("S6r", "bearing forces x scale vs rigid FEM", err, None, e_um=e_um, scale=sc,
                            slope_vs_previous=slope, newton_reason=c["newton"].reason, newton_iter=c["newton"].n_iter,
                            newton_converged=c["newton"].converged))
            prev = (sc, err)
    return out


def _zero_crossings(rows: list[dict]) -> list[dict]:
    """RECORD: the seat offset e [um] at which the middle reaction changes sign, rigid and coupled, per s.

    Linear interpolation between the two swept e that bracket R_2 = 0 (the rigid R_2 is linear in e exactly).
    With clearance the middle bearing does not cross zero: it is unloaded over a range of e (inside its
    clearance), listed in ``unloaded_e_um``.
    """
    out = []
    for s in S_MM:
        mid = sorted((r for r in rows if r["s_mm"] == s and r["x_mm"] == X_MIDDLE_MM), key=lambda r: r["e_um"])
        for key, name in (("R_rigid_N", "rigid supports"), ("R_N", "ISO bearings")):
            e0 = math.nan
            for a, b in zip(mid, mid[1:]):
                if a[key] * b[key] < 0.0:
                    e0 = a["e_um"] - a[key] * (b["e_um"] - a["e_um"]) / (b[key] - a[key])
            zero = " ".join(f"{r['e_um']:g}" for r in mid if r[key] == 0.0)
            out.append(_rec("Z", f"e at R_2 = 0, {name}", e0, None, s_mm=s, unit="um", unloaded_e_um=zero))
    return out


def lift_off_clearance() -> dict:
    """RECORD: the clearance s_c above which bearing 3 sits inside its clearance at e = 0 (physics review MUST 2).

    At e = 0 with bearing 3 free, d_1 = -delta(s) (the ring displacement of one bearing at P/2) and d_3 = d_1 + D,
    D = P l1^2 l_ov / (16 EI); bearing 3 is unloaded while d_3 <= s/2, i.e. delta(s) + s/2 >= D. s_c solves the
    equality (brentq on s, delta(s) by brentq on :func:`model.ball_loads`). EB beam, rigid housing: the margin is
    model-sensitive.
    """
    x = SUPPORTS_MM
    D = P_N * (x[1] - x[0]) ** 2 * (x[2] - x[1]) / (16.0 * EI_NMM2)

    def delta(s_mm):
        b = bearings(s_mm)[x[0]]
        return brentq(lambda dr: ball_loads(b, dr)[0] - P_N / 2.0, s_mm / 2.0, s_mm / 2.0 + 0.1, xtol=1e-15)

    s_c = brentq(lambda s_mm: delta(s_mm) + s_mm / 2.0 - D, 1e-6, S_MM[-1], xtol=1e-12)
    return _rec("Sc", "clearance s_c above which bearing 3 is unloaded at e = 0", s_c * UM, None, unit="um",
                closed_form_D_um=D * UM)


def run(results=RESULTS) -> dict:
    """Every part, the CSV tables and the figures; returns the records, the verdict counts and the figure paths."""
    results = Path(results)
    results.mkdir(parents=True, exist_ok=True)
    problem = hertz_problem(shaft(), EB, N_ELEMENTS)          # the condensed AxisForge beam, built once
    law = law_check()
    pts = {(s, e): point(problem, s, e) for s in S_MM for e in E_UM}
    limits = linear_limit(problem) + stiff_limit(problem)
    rows = [r for p in pts.values() for r in p["rows"]]
    gear = [p["gear"] for p in pts.values()]
    v005 = [r for p in pts.values() for r in p["v005"]]
    v003 = [r for p in pts.values() for r in p["v003"]]
    crit = law + [r for p in pts.values() for r in p["criteria"]] + limits + _zero_crossings(rows)
    crit.append(lift_off_clearance())
    n_failed = sum(not p["newton"].converged for p in pts.values())
    crit.append(_rec("S0", "every coupled solve converged (Newton): failed solves", n_failed, 0.0, n_solves=len(pts)))
    # S2 / S7: aggregates of the v's own verdicts (value = number of MISS, judged at 0); worst rel_err alongside
    for cid, recs, who in (("S2", v005, "v005.check (secant springs)"), ("S7", v003, "v003.check")):
        judged = [r for r in recs if r["verdict"] != "RECORD"]
        crit.append(_rec(cid, f"{who}: MISS among the judged records", sum(r["verdict"] == "MISS" for r in judged),
                         0.0, n_judged=len(judged), worst_rel_err=max(abs(r["rel_err"]) for r in judged)))
    hist = [{**h, "s_mm": s, "e_um": e} for (s, e), p in pts.items() for h in p["newton"].records()]
    write_records_csv(results / "s003_law.csv", law)
    write_records_csv(results / "s003_points.csv", rows + gear)
    write_records_csv(results / "s003_v005.csv", v005)
    write_records_csv(results / "s003_v003.csv", v003)
    write_records_csv(results / "s003_limits.csv", limits)
    write_records_csv(results / "s003_criteria.csv", crit)
    write_records_csv(results / "s003_newton_history.csv", hist)
    write_records_csv(results / "s003_elements.csv", [r for p in pts.values() for r in p["elements"]])
    figures = _figures(results, pts, rows, v005, crit, limits, hist)
    every = crit + v005 + v003
    return {"law": law, "rows": rows, "gear": gear, "v005": v005, "v003": v003, "criteria": crit, "limits": limits,
            "history": hist, "figures": figures,
            "verdicts": {v: sum(r.get("verdict") == v for r in every) for v in ("PASS", "MISS", "RECORD")}}


def _figures(results, pts, rows, v005, crit, limits, hist) -> list[Path]:
    """The diagnostic figures (README 7): reactions and d vs e, Newton, middle-bearing zone, shaft, stiff limit."""
    figs = []
    for s in (S_MM[0], S_MM[-1]):
        # residual panel: s = 0 against the exact power-law judge (S9); s > 0 has none, v005.check C3 there
        c3 = {r["e_um"]: r["rel_err"] for r in v005 if r["s_mm"] == s and r["id"] == "C3"}
        recs = []
        for r in (r for r in rows if r["s_mm"] == s):
            name = f"R_{r['bearing']}"
            err = (r["R_N"] - r["F_hertz_N"]) / P_N if s == 0.0 else c3[r["e_um"]]
            recs += [{"e_um": r["e_um"], "y": r["R_over_P"], "series": name, "source": "FEM", "rel_err": err},
                     {"e_um": r["e_um"], "y": r["R_rigid_over_P"], "series": name, "source": "rigid supports"}]
        inside = ", ".join(f"R_{r['bearing']} at e {r['e_um']:+g}" for r in rows if r["s_mm"] == s and r["unloaded"])
        resid = ("(F - F_Hertz judge) / P (S9, tol 1e-8)" if s == 0.0
                 else "v005.check C3 at the secant springs (beam consistency, no exact judge for s > 0)")
        figs.append(plots.ratio_sweep(recs, results, "s003", x="e_um", x_label=r"middle seat offset $e$ ($\mu$m)",
                                      y_label=r"$R_i / P$ (-)", log_x=False, variant=f"reactions_s{s * UM:g}um",
                                      caption=f"s = {s * UM:g} um; markers ('FEM'): ISO/TS 16281 bearings (Newton), "
                                              f"lines: rigid supports; residual: {resid}; inside the clearance: "
                                              f"{inside or 'none'}; R on the shaft, {R003}; {CAPTION}"))
    # ring displacement over the half clearance: |d / (s/2)| < 1 = no ball loaded; closed-form d_3 at e = 0 (S8)
    recs = []
    for s in S_MM[1:]:
        for r in (r for r in rows if r["s_mm"] == s and r["bearing"] in (2, 3)):   # bearing 1 never enters it
            recs.append({"e_um": r["e_um"], "y": r["d_over_half_clearance"],
                         "series": f"bearing {r['bearing']}, s {s * UM:g} um", "source": "FEM"})
        s8 = next(c for c in crit if c["id"] == "S8" and c["s_mm"] == s and "d_3" in c["quantity"])
        recs.append({"e_um": 0.0, "y": (s8["d1_um"] + s8["closed_form_um"]) / s8["half_clearance_um"],
                     "series": f"bearing 3, s {s * UM:g} um", "source": "closed form d_1 + P l1^2 l_ov/(16 EI)"})
    figs.append(plots.ratio_sweep(recs, results, "s003", x="e_um", x_label=r"middle seat offset $e$ ($\mu$m)",
                                  y_label=r"$d_k / (s/2)$ (-)", log_x=False, legend_outside=True, band=(-1.0, 1.0),
                                  open_sources=("closed form d_1 + P l1^2 l_ov/(16 EI)",), variant="clearance",
                                  caption="bearings 2 and 3 (bearing 1 stays below -1.6); grey band: inside the "
                                          "clearance (no ball loaded); open circles: S8 closed form at e = 0 (bearing "
                                          f"3 unloaded); d along +Y; {CAPTION}"))
    picked = {(s, e) for s in (S_MM[0], S_MM[-1]) for e in LIMIT_E_UM}
    figs.append(newton_history([{**h, "label": f"s {h['s_mm'] * UM:g} um, e {h['e_um']:+g} um"} for h in hist
                                if (h["s_mm"], h["e_um"]) in picked], results, "s003", tol=NEWTON["residual_abs"],
                               caption="numerical_solvers.newton, central FD Jacobian, line search; residual scaled "
                                       f"by P; unknowns: ring displacements d [mm]; {CAPTION}"))
    recs = []
    for s in S_MM:
        p = pts[(s, 0.0)]
        k = p["coupled"]["positions"].index(X_MIDDLE_MM)
        b = bearings(s)[X_MIDDLE_MM]
        d_r = abs(p["coupled"]["d_mm"][k])
        Q = p["coupled"]["Q_j_N"][k]
        name = f"s {s * UM:g} um"
        phi = np.degrees(np.asarray(b.phi_j, dtype=float))
        phi = np.where(phi > 180.0, phi - 360.0, phi)
        recs += [{"series": name, "source": "AxisForge", "phi_deg": float(f), "Q_N": float(q)} for f, q in zip(phi, Q)]
        # zone shape Q_max [1 - (1 - cos phi) / (2 eps)]^1.5 (Harris2007_vol1 Eqs 7.10-7.15, s001 loaded_zone, at
        # the model's delta_a = -A sin(alpha_0)); s = 0: eps = 1/2, Q_max cos^1.5(phi)
        _, eps = loaded_zone(b, d_r, -b.A * math.sin(b.alpha_0))
        if not eps > 0.0:                                      # unloaded: no zone to draw
            continue
        for f in np.linspace(-180.0, 180.0, 361):
            arg = 1.0 - (1.0 - math.cos(math.radians(f))) / (2.0 * eps)
            recs.append({"series": name, "source": "Q_max [1 - (1 - cos phi)/(2 eps)]^1.5", "phi_deg": float(f),
                         "Q_N": float(Q.max()) * max(arg, 0.0) ** 1.5})
    figs.append(element_loads(recs, results, "s003", variant="middle_e0",
                              caption="middle bearing, e = 0, LOCAL azimuth (0 on the load line). Physically the zone "
                                      "is centred at theta 180 deg (-Y, d < 0); AxisForge places it at +R, 180 deg off "
                                      f"({R003}). Lines: Harris zone shape scaled to the computed Q_max, eps from the "
                                      f"same kinematics: only the side balls test the shape; {CAPTION}"))
    p = pts[(S_MM[2], LIMIT_E_UM[-1])]
    c = p["coupled"]
    system = shaft()
    recs = along_shaft(system, c["result"], EB, springs=c["secant"], settlements_mm=c["settlements_mm"],
                       series="ISO bearings (coupled)")
    recs += along_shaft(system, p["rigid"], EB, settlements_mm=c["settlements_mm"], series="rigid supports")
    figs.append(plots.deflection_along_shaft(recs, results, "s003", supports_mm=SUPPORTS_MM, loads_mm=(X_GEAR_MM,),
                                             variant=f"s{S_MM[2] * UM:g}um_e{LIMIT_E_UM[-1]:+g}um",
                                             caption=f"s {S_MM[2] * UM:g} um; middle seat at {LIMIT_E_UM[-1]:+g} um "
                                                     "(support triangles drawn at v = 0, the rigid shaft passes "
                                                     "through the raised seat); lines r01: the coupled shaft at its "
                                                     f"secant springs, the rigid one with the settled seat; {CAPTION}"))
    for sid, variant, slope, what in (("S6r", "stiff_limit", -2.0 / 3.0,
                                       "vs the rigid FEM (S6r, RECORD); dotted: scale^(-2/3), delta ~ (F/scale)^(2/3)"),
                                      ("S6", "stiff_limit_judge", None,
                                       "vs the exact power-law judge (S6, added at review), dashed: tolerance 1e-8")):
        stiff = [{"scale": r["scale"], "quantity": f"e {r['e_um']:+g} um", "series": "bearing forces, s = 0",
                  "rel_err": r["rel_err"], "tol": r["tol"], "verdict": r["verdict"]}
                 for r in limits if r["id"] == sid and r["quantity"].startswith(("F", "bearing"))]
        figs.append(relative_errors(stiff, results, "s003", x="scale", x_label=r"bearing force scale (-)",
                                    variant=variant, slope=slope, caption=f"max |F - F_ref| / P {what}; {CAPTION}"))
    return figs


if __name__ == "__main__":
    out = run()
    print("s003:", out["verdicts"])
    worst = {}
    for r in out["criteria"]:
        key = (r["id"], r["verdict"])
        if not math.isnan(r["rel_err"]):
            worst[key] = max(worst.get(key, 0.0), abs(r["rel_err"]))
    for (cid, v), e in sorted(worst.items()):
        print(f"{cid:>4} {v:>6} worst {e:.3e}")
    n_it = [r["newton_iter"] for r in out["rows"]]
    print("newton iterations:", min(n_it), "...", max(n_it))
