"""Comparator of v003: one bearing solved by AxisForge ISO/TS 16281 against r03 (and r02 for c_p) (README 6).

:func:`check` takes one AxisForge bearing and its solved ``load_distribution`` result (from this case, its
literature or a study) and returns one comparison record per quantity. The judge is r03 fed the
bearing's own constants (:func:`r03_inputs`, the one place where AxisForge units become SI):

* Deep groove ball bearing under a radial load (``Fa = 0``): **judged**. With ``Fa = 0`` the curvature-centre
  model of AxisForge (ISO/TS 16281 Eqs (12)-(15)) drives ``delta_a`` to ``-A sin(alpha_0)`` and every
  ``alpha_j`` to 0, which is Harris Eq. 7.10, r03 at ``alpha = 0`` with the diametral clearance ``s`` as
  ``P_d`` (bearing-expert review of r03). ``delta_a`` is not compared: AxisForge measures it from the
  touching state at ``alpha_0``, r03 has none.
* Angular contact ball bearing (contact angle follows the load in AxisForge, fixed in r03; gap of order
  ``delta/A``) and cylindrical roller bearing (ISO/TS 16281 reference profile in laminae, absent in r03):
  the same records, **RECORD** only (OPEN R5-12).

The element-by-element comparison uses the LOCAL azimuth ``phi_j`` of AxisForge (0 on the radial-load line,
``families/family.py``), which is r03's ``psi`` (0 on ``delta_r``): no convention enters it. Only the global
azimuth of the sign check is convention-dependent (R-003, ``study.sign_check``).

Units: AxisForge mm, N, MPa, N/mm^n; r03 and r02 SI. Records carry the unit of each quantity.
"""
from __future__ import annotations

import math

import numpy as np

from axisforge.outputs.solvers.bearings.load_distribution.load_distribution_results import element_records

from studies.r00_analytical_references.r02_hertz_contact import hertz
from studies.r00_analytical_references.r03_load_zone import (ball_deflection, ring_load_distribution,
                                                             solve_ring_displacement)

__all__ = ["check", "elements", "r03_inputs", "hertz_series_cp", "r02_bodies", "TOL_LOAD_ZONE", "TOL_CP", "MM", "MPA",
           "N_ROLLER"]

MM = 1e-3                    # [m/mm]: the one conversion of lengths of this case
MPA = 1e6                    # [Pa/MPa]
N_ROLLER = 10.0 / 9.0        # AxisForge's line-contact exponent (ISO/TS 16281), not r03's N_LINE = 1.1
# C1-C6: AxisForge solves its equilibrium with SciPy root (hybr) at SOLVER_TOLERANCE = 1e-6, a relative step
# tolerance on (delta_r, delta_a) (iso_16281/numerics.py); Q ~ delta^1.5 carries 1.5x that into the loads. 1e-5
# is an order above it: anything larger is a model difference, not the root
TOL_LOAD_ZONE = 1e-5
# C7: ISO/TS 16281 Eq. (11) replaces the elliptic integrals of Hertz by a fitted expression of the curvature
# sum and difference; 0.5 % is the accuracy the bearing-expert review proposed for that fit (fixed before the run)
TOL_CP = 5e-3


def r03_inputs(bearing) -> dict:
    """r03 inputs of one AxisForge bearing, converted to SI here and only here.

    Returns
    -------
    dict
        ``kind`` ("deep_groove", "angular_contact", "roller"); ``n`` [-] (1.5 balls, 10/9 rollers, as
        AxisForge); ``K_n`` [N/m^n] (AxisForge ``cp`` [N/mm^1.5] times ``1e3^1.5``, or ``cL`` [N/mm^(10/9)]
        times ``1e3^(10/9)``: the whole element, both contacts); ``alpha_rad`` (0 for deep groove and roller,
        ``alpha_0`` for angular contact); ``clearance_m`` (diametral ``s`` [mm] times 1e-3; 0 for angular
        contact, whose ``s`` is derived from ``alpha_0`` and is not a clearance).
    """
    family = bearing.family.name
    if family == "deep_groove_ball":
        return {"kind": "deep_groove", "n": 1.5, "K_n": bearing.cp * (1.0 / MM) ** 1.5, "alpha_rad": 0.0,
                "clearance_m": bearing.s * MM}
    if family == "angular_contact":
        return {"kind": "angular_contact", "n": 1.5, "K_n": bearing.cp * (1.0 / MM) ** 1.5,
                "alpha_rad": float(bearing.alpha_0), "clearance_m": 0.0}
    if family == "cylindrical_roller":
        return {"kind": "roller", "n": N_ROLLER, "K_n": bearing.cL * (1.0 / MM) ** N_ROLLER, "alpha_rad": 0.0,
                "clearance_m": bearing.s * MM}
    raise ValueError(f"v003 has no r03 mapping for the family {family!r}")


def r02_bodies(bearing, race: str) -> dict:
    """r02 inputs of one ball-groove contact from AxisForge's own attributes (r02 README section 4), SI.

    Returns ``r1_m`` (ball radius), ``r2_m`` (the groove's own principal radii ``(r_x, r_y)``: inner
    ``(r_bx_inner, -ri)``, outer ``(-r_bx_outer, -re)``; AxisForge stores the outer rolling radius as a positive
    magnitude, its concave sign is put here; radii at ``alpha_0``), ``E_Pa``, ``nu`` (ball, ring) and ``r_g_m``
    (groove radius). The one place where a ball contact of AxisForge becomes SI for r02 (v003 C7, v105).
    """
    if race == "inner":
        r2, r_g = (bearing.r_inner[0] * MM, bearing.r_inner[1] * MM), bearing.ri * MM
    elif race == "outer":
        r2, r_g = (-bearing.r_outer[0] * MM, bearing.r_outer[1] * MM), bearing.re * MM
    else:
        raise ValueError(f"race must be 'inner' or 'outer', got {race!r}")
    return {"r1_m": bearing.r_rolling_el[0] * MM, "r2_m": r2, "E_Pa": (bearing.e1 * MPA, bearing.e2 * MPA),
            "nu": (bearing.nu1, bearing.nu2), "r_g_m": r_g}


# radii at alpha_0, as AxisForge's c_p; under F_a = 0 the deep-groove contacts run at alpha = 0: a c_p change
# <= 0.03 % for s <= 60 um (6210, physics review of v003)
def hertz_series_cp(bearing) -> float:
    """``c_p`` [N/mm^1.5] of a ball as the Hertz series of its two contacts, r02 (``hertz_full``) and r03 Eq. 7.6.

    The ball against the inner and the outer groove as :func:`r02_bodies` builds them (radii at ``alpha_0``);
    ``K = F / delta^1.5`` of each contact (exact for Hertz, any ``F``), then
    ``K_n = [(1/K_i)^(2/3) + (1/K_o)^(2/3)]^(-3/2)`` (r03 ``ball_deflection``).
    """
    F_N = 1000.0                                               # any load: delta ~ F^(2/3) exactly
    K = []
    for race in ("inner", "outer"):
        b = r02_bodies(bearing, race)
        K.append(F_N / hertz(b["r1_m"], b["r2_m"], b["E_Pa"], b["nu"], F_N).delta_m ** 1.5)
    K_n = ball_deflection(1.0, K_i=K[0], K_o=K[1])["K_n_N_per_m_pow_n"]
    return K_n * MM ** 1.5                                     # [N/m^1.5] -> [N/mm^1.5]


def _judge(bearing, result) -> dict:
    """r03 at AxisForge's own ring displacement, and r03's own solve at AxisForge's loads."""
    inp = r03_inputs(bearing)
    row = result.rows[0]
    psi = np.asarray(row.phi_j, dtype=float)
    common = dict(psi_rad=psi, K_n=inp["K_n"], n=inp["n"], alpha_rad=inp["alpha_rad"], clearance_m=inp["clearance_m"])
    # AxisForge's delta_a is from its alpha_0 touching state: zero it for the radial families (Fa = 0, alpha = 0)
    da_m = result.delta_a * MM if inp["kind"] == "angular_contact" else 0.0
    same = ring_load_distribution(result.delta_r * MM, da_m, **common)
    own = solve_ring_displacement(result.Fr, result.Fa if inp["kind"] == "angular_contact" else 0.0, **common)
    return {"inp": inp, "row": row, "same": same, "own": own}


def elements(bearing, result, *, case: str = "v003", label: str = "") -> list[dict]:
    """Per-element records: AxisForge ``Q_j``, ``delta_j`` and r03 at the same displacement and at the same load.

    Fields: ``case``, ``label``, ``j``, ``phi_j_rad`` (local, 0 on the load line), ``alpha_j_rad``,
    ``Q_axisforge_N``, ``Q_r03_same_dr_N``, ``Q_r03_solve_N``, ``delta_axisforge_m``, ``delta_r03_solve_m``.
    """
    J = _judge(bearing, result)
    # AxisForge's own element records (phi wrapped to [-pi, pi), delta [mm]), in element order j, plus the r03 columns
    return [{"case": case, "label": label or bearing.label, "j": e["j"], "phi_j_rad": e["phi_rad"],
             "alpha_j_rad": e["alpha_rad"], "Q_axisforge_N": e["Q_N"],
             "Q_r03_same_dr_N": float(J["same"]["Q_N"][e["j"]]), "Q_r03_solve_N": float(J["own"]["Q_N"][e["j"]]),
             "delta_axisforge_m": e["delta_mm"] * MM,
             "delta_r03_solve_m": max(float(J["own"]["delta_psi_m"][e["j"]]), 0.0)} for e in element_records(result)]


def _record(case, label, cid, quantity, unit, measured, reference, rel_err, tol, verdict, note="") -> dict:
    return {"case": case, "label": label, "id": cid, "quantity": quantity, "unit": unit, "measured": float(measured),
            "reference": float(reference), "rel_err": float(rel_err), "tol": float(tol), "verdict": verdict,
            "note": note}


def check(bearing, result, *, case: str = "v003", label: str = "") -> list[dict]:
    """Comparison records of one solved bearing against r03 (C1-C6) and r02 + r03 (C7, balls).

    Parameters
    ----------
    bearing : axisforge Bearing
        Deep groove, angular contact or cylindrical roller, assembled for ISO/TS 16281.
    result : BallBearingResult or RollerBearingResult
        Its ``load_distribution`` with ``psi = 0`` (no tilt: r03 has none). Single row.
    case, label : str
        Written into the records (an s passes its own).

    Returns
    -------
    list of dict
        ``case``, ``label``, ``id``, ``quantity``, ``unit``, ``measured`` (AxisForge), ``reference`` (r03 / r02),
        ``rel_err`` (signed for scalars; for the element sets the largest ``|dQ_j| / Q_max``), ``tol``,
        ``verdict`` (PASS / MISS / RECORD), ``note``. C1 ``Q_j`` vs r03 at AxisForge's ``delta_r`` (the
        distribution law), C2 ``Q_j`` vs r03's own solve at the same ``F_r`` (law and equilibrium), C3 ``Q_max``,
        C4 ``delta_r``, C5 ``eps`` from AxisForge's ``delta_r`` and ``s`` (Harris Eq. 7.12; restates C4), C6
        equilibrium ``sum Q_j cos(alpha_j) cos(phi_j) = F_r`` (AxisForge against its own input: self-consistency, not
        a judge), C7 ``c_p`` vs the Hertz series of r02 (balls).
    """
    label = label or bearing.label
    J = _judge(bearing, result)
    inp, row, same, own = J["inp"], J["row"], J["same"], J["own"]
    judged = inp["kind"] == "deep_groove" and result.Fa == 0.0
    note = "" if judged else f"RECORD: {inp['kind']} (OPEN R5-12)"

    def verdict(err, tol):
        return ("PASS" if abs(err) <= tol else "MISS") if judged else "RECORD"

    Q = np.asarray(row.Q_j, dtype=float)
    Q_max = float(Q.max())
    rows = []
    for cid, quantity, ref in (("C1", "Q_j at AxisForge delta_r", same["Q_N"]), ("C2", "Q_j, r03 solve at F_r", own["Q_N"])):
        err = float(np.max(np.abs(Q - ref))) / Q_max
        rows.append(_record(case, label, cid, quantity, "N", Q_max, float(np.max(ref)), err, TOL_LOAD_ZONE,
                            verdict(err, TOL_LOAD_ZONE), note))
    ref_Qmax = float(np.max(own["Q_N"]))
    rows.append(_record(case, label, "C3", "Q_max", "N", Q_max, ref_Qmax, Q_max / ref_Qmax - 1.0, TOL_LOAD_ZONE,
                        verdict(Q_max / ref_Qmax - 1.0, TOL_LOAD_ZONE), note))
    dr = result.delta_r * MM
    rows.append(_record(case, label, "C4", "delta_r", "m", dr, own["delta_r_m"], dr / own["delta_r_m"] - 1.0,
                        TOL_LOAD_ZONE, verdict(dr / own["delta_r_m"] - 1.0, TOL_LOAD_ZONE), note))
    eps_af = same["epsilon"]          # Harris Eq. 7.12 at alpha = 0, 7.61 with alpha
    # C5 restates C4: eps of AxisForge is r03's definition at AxisForge's delta_r (AxisForge has no eps field), and
    # it is identically 0.5 at s = 0 for any delta_r
    rows.append(_record(case, label, "C5", "eps = (1 - s/(2 delta_r))/2 from AxisForge delta_r (restates C4)", "-",
                        eps_af, own["epsilon"], eps_af / own["epsilon"] - 1.0,
                        TOL_LOAD_ZONE, verdict(eps_af / own["epsilon"] - 1.0, TOL_LOAD_ZONE), note))
    radial = float(np.sum(Q * np.cos(np.asarray(row.alpha_j)) * np.cos(np.asarray(row.phi_j))))
    rows.append(_record(case, label, "C6", "sum Q_j cos(alpha_j) cos(phi_j) (self-consistency, not a judge)", "N",
                        radial, result.Fr,
                        radial / result.Fr - 1.0, TOL_LOAD_ZONE, verdict(radial / result.Fr - 1.0, TOL_LOAD_ZONE),
                        note))
    if inp["kind"] != "roller":
        cp_ref = hertz_series_cp(bearing)
        err = bearing.cp / cp_ref - 1.0
        # c_p is a property of the geometry: judged for every ball bearing, also the RECORD families
        rows.append(_record(case, label, "C7", "c_p vs Hertz series (r02, r03 Eq. 7.6)", "N/mm^1.5", bearing.cp,
                            cp_ref, err, TOL_CP, "PASS" if abs(err) <= TOL_CP else "MISS",
                            f"radii at alpha_0 = {math.degrees(bearing.alpha_0):.3f} deg"))
    return rows
