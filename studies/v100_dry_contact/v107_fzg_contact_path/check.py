"""Comparator of v107: an AxisForge gear pair along its path against r04 + r02 and Martins 2015 (README 6).

:func:`check` takes any assembled AxisForge cylindrical pair (from this case's literature or from a study)
and a wheel torque, and compares what AxisForge computes (``pair`` attributes, ``pair_values_at`` at the
points A..E of ``path_points``, ``mesh_forces``) with r04 built from the same basic data, and the Hertz
pressure at C with r02 fed by the AxisForge radii and force (the r02 README section 4 input table, SI).
"""
from __future__ import annotations

import math

import numpy as np
from axisforge.solvers.machine_elements.gears.contact import load_sharing

from studies.r00_analytical_references.r02_hertz_contact import hertz
from studies.r00_analytical_references.r04_involute import mesh_forces, pair_geometry, path_point

from .construction import DRIVER, E_MPA, FAMILY, NU

__all__ = ["check", "TOL_CLOSED_FORM"]

TOL_CLOSED_FORM = 1e-10      # P-12: closed form vs closed form with an iterated inverse involute


def check(pair, wheel_torque_Nm: float, *, data: dict, published: dict | None = None,
          case: str = "v107") -> list[dict]:
    """Comparison records of one FZG pair at one wheel torque.

    * G: working centre distance and T1A..T1E against r04 (``TOL_CLOSED_FORM``, relative);
    * P: at A..E (pinion driving), ``rho1_t``, ``rho2_t`` against r04 (relative) and ``zeta1``, ``zeta2``
      (absolute), ``TOL_CLOSED_FORM``;
    * F: ``F_tw``, ``F_r``, ``F_a``, ``F_bn`` of ``mesh_forces`` (gear-1 torque ``T2 z1/z2``, R-124) against r04
      at the wheel torque;
    * H (spur only): Hertz line pressure from r02 with the AxisForge transverse radii (mm -> m) and the load
      per unit length ``share F_bn / b`` at C, A (``load_sharing``) and B, E (full load, set); RECORD (the FEUP reading of p_H is
      open, Q-09 / SQ-06);
    * L (``published``: Martins 2015 Table 4 printed strings of this pair and stage, ``Fr``, ``Fa``, ``p_H``):
      AxisForge ``F_r``, ``F_a`` within half a printed unit plus the printed torque's half unit carried
      (judged); both p_H readings (Hertz at C, Hertz at A with its share) against the printed p_H, RECORD.

    ``data`` is the basic data the pair was built from (``construction.PAIRS[name]``): r04 never reads the
    AxisForge object it judges.
    """
    z1, z2 = data["z"]
    g = pair_geometry(z1, z2, data["mn"], data["alpha_n_deg"], data["beta_n_deg"], *data["x"], data["b"])
    out = []

    def row(cid, quantity, unit, axisforge, r04, point="", tol=TOL_CLOSED_FORM, rel=True, verdict=None, **extra):
        err = math.nan if verdict == "RECORD" and not math.isfinite(axisforge) else (
            (axisforge / r04 - 1.0) if rel else axisforge - r04)
        v = verdict or ("PASS" if abs(err) <= tol else "MISS")
        out.append({"case": case, "pair": pair.label, "wheel_torque_Nm": wheel_torque_Nm, "id": cid, "point": point,
                    "quantity": quantity, "unit": unit, "axisforge": axisforge, "reference": r04, "err": err,
                    "err_kind": "rel" if rel else "abs", "tol": tol, "verdict": v, **extra})

    row("G", "a_w", "mm", pair.al, g.a_w)
    for k in "ABCDE":
        row("G", f"T1{k}", "mm", getattr(pair, f"T1{k}"), getattr(g, f"T1{k}"))
    rb1 = pair.gear1.rb
    path = FAMILY.path_points(pair, DRIVER)
    dy1 = {k: 2.0 * math.hypot(pair.T1C + gy, rb1) for k, gy in path.items()}
    values = FAMILY.pair_values_at(pair, np.array(list(dy1.values())))
    for i, k in enumerate(path):
        r = path_point(g, g.T1C + float(values["g_y_mm"][i]))
        row("P", "rho1_t", "mm", float(values["rho1_t_mm"][i]), r.rho1_t, k)
        row("P", "rho2_t", "mm", float(values["rho2_t_mm"][i]), r.rho2_t, k)
        row("P", "zeta1", "-", float(values["zeta1"][i]), r.zeta1, k, rel=False)
        row("P", "zeta2", "-", float(values["zeta2"][i]), r.zeta2, k, rel=False)
    # gear-1 torque from the wheel torque (R-124); mesh_forces uses |T1| and the case compares magnitudes only:
    # the sign and the driver are not checked here (the wheel as driver is OPEN R2-13)
    T1 = -wheel_torque_Nm * z1 / z2
    m = FAMILY.mesh_forces(pair, T1, -1 if T1 < 0 else 1)
    f = mesh_forces(g, wheel_torque_Nm, on=2)
    for q, a_, r_ in (("F_tw", m.F_tw_N, f.F_t), ("F_r", m.F_r_N, f.F_r), ("F_bn", m.F_bn_N, f.F_bn)):
        row("F", q, "N", a_, r_)
    row("F", "F_a", "N", m.F_a_N, f.F_a, rel=f.F_a > 0.0)
    if data["beta_n_deg"] == 0.0:                     # spur: one line of length b per pair
        # C and A with the share of load_sharing (1 at C, 1/2 at A); B and E with the full load set here:
        # load_sharing gives 1/2 at the limits B, D of the single-pair zone, where ISO 6336-2 (Z_B) puts the full load
        for k, forced in (("C", None), ("A", None), ("B", 1.0), ("E", 1.0)):
            i = list(path).index(k)
            share = forced if forced is not None else float(load_sharing(pair, np.array([values["g_y_mm"][i]]))["share"][0])
            w = share * m.F_bn_N / (data["b"] * 1e-3)                                  # [N/m]
            h = hertz((float(values["rho1_t_mm"][i]) * 1e-3, math.inf), (float(values["rho2_t_mm"][i]) * 1e-3, math.inf),
                      E_MPA * 1e6, NU, w, line=True)
            row("H", f"p_H Hertz (share {share:g})", "MPa", math.nan, h.p0_Pa * 1e-6, k, tol=math.nan,
                verdict="RECORD", b_H_mm=h.b_m * 1e3, w_N_per_mm=w * 1e-3)
    if published:
        for sym, value in (("Fr", m.F_r_N), ("Fa", m.F_a_N)):
            printed = published[sym]
            tol = 0.5 + value * 0.005 / wheel_torque_Nm
            row("L", f"{sym} Martins Table 4", "N", value, float(printed), tol=tol, rel=False)
        p_H = float(published["p_H"])
        for r in [r for r in out if r["id"] == "H"]:
            out.append({**r, "id": "L", "quantity": f"printed p_H vs {r['quantity']} at {r['point']}",
                        "axisforge": r["reference"], "reference": p_H, "err": r["reference"] / p_H - 1.0,
                        "err_kind": "rel"})
    return out

