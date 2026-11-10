"""Comparator of v105: the bridge's ball-raceway contacts of a solved bearing against r02 Hertz (README 6).

:func:`check` takes the contacts ``slippy_bridge`` builds from an AxisForge bearing and its solved ISO/TS 16281
result (``DeepGrooveBallContact.from_bearing``, ``AngularContactBallContact.from_bearing``) and returns one
record per contact and quantity. It checks the TRANSFER AxisForge -> bridge (``from_bearing``), not Hertz: both
sides run SlipPY ``hertz_full`` (judged in v101/v102); what is cross-checked is the bridge's raceway rolling
radius recomputed from ``Dw, Dpw, ri, re, alpha_0`` against AxisForge's stored radii, the concave outer sign, the
mm/MPa -> SI conversion, the equivalent-body reduction and the element's load, azimuth and angle. The independent
judges of the contact geometry are Hamrock's printed radii (``study.literature_series``, L1-L4). Per contact:

* C0 the element's load ``Q``, local azimuth ``phi`` and angle ``alpha`` of the bridge contact against the
  AxisForge result's ``Q_j``, ``phi_j``, ``alpha_j`` (AxisForge's ``element_records``): exact.
* C1-C4 ``a``, ``b``, ``p0``, ``delta`` of the bridge's Hertz (``slippy_bridge.contact.solvers.hertz``, on the
  bridge's own bodies) against r02 ``hertz`` of the ball and the groove from AxisForge's own attributes (v003
  ``r02_bodies``, radii at ``alpha_0``) at AxisForge's ``Q_j``. r02 runs at ``root_error`` 1e-12, the bridge at
  SlipPY's 1e-6, which leaves up to 6.5e-8 (OPEN RC-01): tolerance 1e-7. Judged whenever the element runs at
  ``alpha_0`` (same contact; the bridge builds the raceway at ``alpha_j``, AxisForge's radii are at ``alpha_0``:
  SB-48, a real difference, RECORD). The error does not depend on the size of the contact, so the conformity
  does not gate it (SlipPY-expert review).
* C5 the size of the contact, RECORD (OPEN R5-88): r02's per-body ratio (``validity.ratio``, governed by the ball
  ``a/r1``, 3.8 % above ``a/r_g``) and the bridge's ratio on the relative radii (SB-07).

:func:`series_records` (S1): the Hertz series of the two bridge contacts of one element, ``K_n = Q /
(delta_i + delta_o)^1.5``, against the same series from r02 (v003 ``hertz_series_cp``, reused, OPEN R5-80),
and AxisForge's ``c_p`` (RECORD here: judged as v003 C7).

Units: SI (bridge and r02); AxisForge's mm and MPa are converted once, in v003 ``r02_bodies``.
"""
from __future__ import annotations

import math

from axisforge.outputs.solvers.bearings.load_distribution.load_distribution_results import element_records
from slippy_bridge.contact.solvers.hertz import hertz as bridge_hertz

from studies.r00_analytical_references.r02_hertz_contact import hertz
# OPEN R5-80 (R2-02 pattern): v003's r02 inputs and Hertz series of a bearing's ball, imported, not duplicated
from studies.v000_axisforge.v003_iso16281_load_zone import hertz_series_cp, r02_bodies
from studies.v000_axisforge.v003_iso16281_load_zone.check import MM

__all__ = ["check", "series_records", "TOL_IDENTITY", "SIZE_RATIO_MAX"]

TOL_IDENTITY = 1e-7          # both are hertz_full; SlipPY's root_error 1e-6 in the bridge leaves <= 6.5e-8 (RC-01)
SIZE_RATIO_MAX = 0.1         # r02's per-body size ratio of a contact inside Hertz's range (P-02 policy; OPEN R5-88)
_ANGLE_SAME = 1e-12          # [rad] the element is at alpha_0 (both radii sets are the same contact)


def check(contacts, bearing, result, *, case: str = "v105", label: str = "") -> list[dict]:
    """Comparison records of the bridge's ball contacts of one bearing against AxisForge and r02 (C0-C5).

    Parameters
    ----------
    contacts : list of DeepGrooveBallContact or AngularContactBallContact
        From ``from_bearing(bearing, result, ...)``: each carries its race, element ``j``, ``phi_rad`` (local),
        ``alpha_rad`` and load ``Q`` [N].
    bearing : axisforge Bearing
        The bearing they were built from.
    result : BallBearingResult
        The AxisForge load distribution the contacts were built from (its ``Q_j`` is r02's load).
    case, label : str
        Written into the records.

    Returns
    -------
    list of dict
        ``case``, ``label``, ``contact``, ``race``, ``j``, ``phi_deg`` (local, 0 on the load line), ``alpha_deg``,
        ``Q_N`` (AxisForge), ``a_over_rg``, ``size_ratio_r02`` (per body), ``size_governing``,
        ``bridge_size_ratio`` (relative radii, SB-07), ``bridge_valid``, ``id``, ``quantity``, ``unit``,
        ``measured`` (bridge), ``reference`` (AxisForge / r02), ``rel_err``, ``tol``, ``verdict``, ``note``.
    """
    af = {e["j"]: e for e in element_records(result)}
    rows = []
    for c in contacts:
        e = af[c.j]
        body = r02_bodies(bearing, c.race)
        ref = hertz(body["r1_m"], body["r2_m"], body["E_Pa"], body["nu"], e["Q_N"])
        got = bridge_hertz(c)
        same = abs(c.alpha_rad - bearing.alpha_0) <= _ANGLE_SAME
        note = "" if same else (f"element at alpha_j {math.degrees(c.alpha_rad):.3f} deg, AxisForge radii at alpha_0 "
                                f"{math.degrees(bearing.alpha_0):.3f} deg (SB-48)")
        common = {"case": case, "label": label or bearing.label, "contact": c.label, "race": c.race, "j": c.j,
                  "phi_deg": math.degrees(c.phi_rad), "alpha_deg": math.degrees(c.alpha_rad), "Q_N": e["Q_N"],
                  "a_over_rg": ref.a_m / body["r_g_m"], "size_ratio_r02": ref.validity.ratio,
                  "size_governing": ref.validity.governing, "bridge_size_ratio": got.size_ratio,
                  "bridge_valid": bool(got.valid)}
        # C0: the transfer of the element itself (exact: the bridge copies AxisForge's values)
        for quantity, unit, m, r in (("Q", "N", c.load, e["Q_N"]), ("phi", "rad", c.phi_rad, e["phi_rad"]),
                                     ("alpha", "rad", c.alpha_rad, e["alpha_rad"])):
            diff = m - r
            rows.append({**common, "id": "C0", "quantity": f"{quantity} of the element", "unit": unit, "measured": m,
                         "reference": r, "rel_err": diff / r if r else diff, "tol": 0.0,
                         "verdict": "PASS" if diff == 0.0 else "MISS", "note": "bridge vs AxisForge element record"})
        for cid, quantity, unit, m, r in (("C1", "a", "m", got.a, ref.a_m), ("C2", "b", "m", got.b, ref.b_m),
                                          ("C3", "p0", "Pa", got.p0, ref.p0_Pa),
                                          ("C4", "delta", "m", got.delta, ref.delta_m)):
            err = m / r - 1.0
            rows.append({**common, "id": cid, "quantity": quantity, "unit": unit, "measured": m, "reference": r,
                         "rel_err": err, "tol": TOL_IDENTITY,
                         "verdict": ("PASS" if abs(err) <= TOL_IDENTITY else "MISS") if same else "RECORD",
                         "note": note})
        rows.append({**common, "id": "C5", "quantity": f"size ratio r02 per body ({ref.validity.governing}); bridge on "
                     f"the relative radii {got.size_ratio:.3f} (SB-07), valid {got.valid}", "unit": "-",
                     "measured": ref.validity.ratio, "reference": SIZE_RATIO_MAX, "rel_err": math.nan,
                     "tol": math.nan, "verdict": "RECORD",
                     "note": "inside" if ref.validity.ratio <= SIZE_RATIO_MAX else "outside the P-02 policy 0.1"})
    return rows


def series_records(contacts, bearing, *, case: str = "v105", label: str = "") -> list[dict]:
    """S1: Hertz series of the two bridge contacts of each element against r02's series and AxisForge's ``c_p``.

    For each element with both races: ``K_n = Q / (delta_i + delta_o)^1.5`` [N/mm^1.5] from the bridge's Hertz
    approaches (Harris Eq. 7.5-7.7), against v003 ``hertz_series_cp`` (r02, radii at ``alpha_0``; judged at
    ``TOL_IDENTITY`` when the element is at ``alpha_0``, RECORD otherwise) and against AxisForge's ``c_p``
    (ISO/TS 16281 Eq. (11) fit: RECORD here, judged in v003 C7).
    """
    by_j = {}
    for c in contacts:
        by_j.setdefault(c.j, {})[c.race] = c
    cp_r02 = hertz_series_cp(bearing)
    rows = []
    for j, pair in by_j.items():
        if set(pair) != {"inner", "outer"}:
            continue
        Q = pair["inner"].load
        delta_n = bridge_hertz(pair["inner"]).delta + bridge_hertz(pair["outer"]).delta
        K_n = Q / delta_n ** 1.5 * MM ** 1.5                     # [N/m^1.5] -> [N/mm^1.5]
        same = abs(pair["inner"].alpha_rad - bearing.alpha_0) <= _ANGLE_SAME
        common = {"case": case, "label": label or bearing.label, "contact": f"element {j}", "race": "both", "j": j,
                  "phi_deg": math.degrees(pair["inner"].phi_rad), "alpha_deg": math.degrees(pair["inner"].alpha_rad),
                  "Q_N": Q, "unit": "N/mm^1.5", "measured": K_n, "tol": TOL_IDENTITY}
        err = K_n / cp_r02 - 1.0
        rows.append({**common, "id": "S1", "quantity": "K_n bridge series vs r02 series (v003 hertz_series_cp)",
                     "reference": cp_r02, "rel_err": err,
                     "verdict": ("PASS" if abs(err) <= TOL_IDENTITY else "MISS") if same else "RECORD",
                     "note": "" if same else "element not at alpha_0 (SB-48)"})
        rows.append({**common, "id": "S2", "quantity": "K_n bridge series vs AxisForge c_p (ISO Eq. (11) fit)",
                     "reference": bearing.cp, "rel_err": K_n / bearing.cp - 1.0, "verdict": "RECORD",
                     "tol": math.nan, "note": "judged in v003 C7 (0.5 %)"})
    return rows
