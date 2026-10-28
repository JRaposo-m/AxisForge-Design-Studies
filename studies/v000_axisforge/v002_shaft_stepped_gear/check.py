"""Comparator of v002: the shaft side is v001's comparator, the gear side the mesh loads against the gear inputs.

The shaft comparator (stepped sections, couples, settlements, axial statics) is
``studies.v000_axisforge.v001_shaft_uniform.check`` (one comparator for v001, v002 and the s cases;
merged at the v002 minimal review): v002 re-exports ``check``, ``reference``, ``segments``,
``rigid_body``, ``along_shaft``. This module adds the gear side: :func:`mesh_judge` builds the
forces of the mesh with their signs from the gear inputs alone, :func:`gear_identities` compares
what ``Transmission.resolve`` produced (``MeshForces`` and the shaft loads of
``mesh_forces_to_shaft_loads``) with it, and :func:`physical_couples` turns the axial force at the
pitch point into r01 couples (CW-01).

Couples. The FEM consumes ``ExternalMoment.component(plane)`` as an r01 couple of that plane (v001
``plane_couples``). Physically a moment vector acts in XY as +M_Z and in XZ as -M_Y. I5 checks the
two agree; T5 records how far the reading of the names ``My`` (about +Y), ``Mz`` (about +Z) would be
(CW-01). D-0010 rows 21-22, U1, J-16 are open: every number that depends on the couple is tagged
``convention-dependent (J-16)``.

Units: mm, N, N.mm on the shaft side; torque N.m and the diameters in mm on the gear side.
"""
from __future__ import annotations

import math

import numpy as np

from studies.r00_analytical_references.r04_involute import mesh_forces as r04_mesh_forces
from studies.r00_analytical_references.r04_involute import pair_geometry
from studies.v000_axisforge.v001_shaft_uniform.check import (J16, TOL_NODAL, along_shaft, check, plane_couples,
                                                             reference, rigid_body, segments)

__all__ = ["check", "reference", "segments", "rigid_body", "along_shaft", "mesh_judge", "physical_couples",
           "gear_identities", "TOL_NODAL", "TOL_IDENTITY", "J16"]

# [-] identities of floating-point statics and trigonometry (a few operations each): 1e-12 relative (brief)
TOL_IDENTITY = 1e-12


def mesh_judge(*, torque_Nm: float, beta_deg: float, mn_mm: float, z1: int, z2: int, alpha_n_deg: float, b_mm: float,
               x1: float = 0.0, x2: float = 0.0, hand1: int = 0, rot1: int = 1, gear1_drives: bool = True) -> dict:
    """Forces of an external cylindrical mesh with their signs, from the gear inputs alone.

    Magnitudes from r04 (the v calls its r, D-0036 amendment item 6): ``pair_geometry`` (working
    pressure angle from x1 + x2 by ISO_21771_2007 Eq. (55), "for the particular case of zero backlash";
    d_w = d_b / cos(alpha_wt), Eq. (56), doc p. 29 / PDF p. 33; beta_w from the lead, which does not
    depend on the diameter, 4.3.2 Eq. (4), doc p. 18 / PDF p. 22) and ``mesh_forces`` (F_tw = 2000
    abs(T_1) / d_w1, F_r = F_tw tan(alpha_wt), F_a = F_tw tan(beta_w)).

    Signs, in the mesh frame (e_c gear 1 -> gear 2, e_x the shaft axis, e_t = e_x x e_c), force of the
    mesh ON gear 1, with s = +1 when gear 1 drives and -1 when it is driven:

        on_1 = (-F_r, -s rot1 F_tw, +h1 s rot1 F_a),     on_2 = -on_1,     rho = (+d_w1/2, -d_w2/2)

    The flanks only push (an external gear is pushed toward its own axis: -F_r); the tangential force on
    the driver opposes its rotation (on the driven gear it follows it); the axial force is normal to the
    tooth trace at the pitch point: a right-hand trace (h = +1) runs along cos(beta) e_x + h sin(beta) e_t
    there, so F_x = -h tan(beta) F_t. Geometric derivation (v002 gear-expert review; ISO 21771 4.1.4,
    the hand, not transcribed: no library source, issue R5-36).

    Parameters
    ----------
    torque_Nm : float
        Magnitude of the torque on gear 1 [N.m] (the input of the case, not an AxisForge output).
    beta_deg, mn_mm, alpha_n_deg : float
        Helix angle at the reference cylinder [deg], normal module [mm], normal pressure angle [deg].
    z1, z2 : int
        Tooth numbers (external pair, both > 0).
    b_mm : float
        Face width [mm] (r04 needs it for the overlap ratio; the forces do not depend on it).
    x1, x2 : float
        Profile-shift coefficients [-].
    hand1 : int
        Hand of gear 1, +1 right, -1 left, 0 spur.
    rot1 : int
        Sense of rotation of gear 1 about +X, +1 or -1.
    gear1_drives : bool
        True when gear 1 is the driving gear.

    Returns
    -------
    dict
        ``d_w1_mm``, ``d_w2_mm``, ``alpha_wt_rad``, ``F_tw_N``, ``F_r_N``, ``F_a_N``, ``on_1``, ``on_2``
        (F_c, F_t, F_x) [N], ``rho_mm`` (rho_1, rho_2) [mm].
    """
    g = pair_geometry(z1, z2, mn_mm, alpha_n_deg, beta_deg, x1, x2, b_mm)
    f = r04_mesh_forces(g, torque_Nm, on=1)
    s = 1.0 if gear1_drives else -1.0
    on_1 = (-f.F_r, -s * rot1 * f.F_t, hand1 * s * rot1 * f.F_a)
    return {"d_w1_mm": g.d_w1, "d_w2_mm": g.d_w2, "alpha_wt_rad": g.alpha_wt_rad, "F_tw_N": f.F_t, "F_r_N": f.F_r,
            "F_a_N": f.F_a, "on_1": on_1, "on_2": tuple(-v for v in on_1), "rho_mm": (0.5 * g.d_w1, -0.5 * g.d_w2)}


def physical_couples(F_x_N: float, rho_mm: float, theta_c_deg: float) -> tuple[float, float]:
    """Couple of an axial mesh force on one gear as r01 couples ``(C_XY, C_XZ)`` [N.mm], from r x F alone.

    The mesh pushes at the pitch point ``r = rho e_c`` (e_c at ``theta_c_deg`` from +Y toward +Z); the
    moment of its axial part ``F_x e_x`` about the shaft axis is ``rho F_x (e_c x e_x) = rho F_x (0, e_cz,
    -e_cy)``. In the plane XY the r01 couple is +M_Z, in XZ it is -M_Y (CW-01). On a beam on two pins
    such a couple moves +C/s to the first pin and -C/s to the second (s = span), wherever it sits.

    Parameters
    ----------
    F_x_N : float
        Axial force on the gear along +X [N] (:func:`mesh_judge` ``on_i[2]``).
    rho_mm : float
        Signed distance of the pitch point from the gear axis along e_c [mm].
    theta_c_deg : float
        Direction of e_c (gear 1 -> gear 2) in the global frame [deg].
    """
    th = math.radians(theta_c_deg)
    M_Y, M_Z = rho_mm * F_x_N * math.sin(th), -rho_mm * F_x_N * math.cos(th)
    return M_Z, -M_Y


def gear_identities(resolved, judge: dict, *, theta_c_deg: float, case: str = "v002", label: str = "") -> list[dict]:
    """The mesh loads ``Transmission.resolve`` produced, against :func:`mesh_judge` and the pitch-point statics.

    One external pair, gear 1 on ``resolved.shafts[0]``, gear 2 on ``resolved.shafts[1]`` (either may
    be the source). Records:

    * I1 ``F_tw``, I2 ``F_r``, I3 ``F_a`` [N] vs the judge (I1 from the input torque, so the P / omega
      round trip of ``Transmission.resolve`` is judged too);
    * I3s the signed axial force on gear 1, ``MeshForces.on_1[2]`` vs ``+h1 s rot1 F_a`` [N];
    * I4 per shaft, the transverse ``RadialLoad`` (and distributed) sum vs ``F_c e_c + F_t e_t`` of the
      judge [N];
    * I5 per shaft, the couple the FEM consumes (v001 ``plane_couples``) vs :func:`physical_couples` of
      the judge's F_x and rho [N.mm]: J-16;
    * I6 per shaft, the couple's modulus vs ``F_a r_w`` of that gear [N.mm]: J-16;
    * I8 per shaft, the sum of its ``AxialLoad`` vs the judge's F_x of its gear [N];
    * I7 every closure check of ``Transmission`` (action = reaction, torque = moment of the mesh force,
      power of the mesh, separating radial force, torque sum per shaft, train power): its own pass flag
      (rel 1e-9 inside AxisForge), the residual recorded;
    * T5 RECORD: the couple read as the ``ExternalMoment`` names say (``My`` about +Y, ``Mz`` about +Z,
      so C_XY = Mz, C_XZ = -My) minus the physical one: the CW-01 trap, J-16.

    Tolerance of I1-I6, I8: ``TOL_IDENTITY`` relative to F_tw (forces) or to that shaft's ``F_a r_w``
    (couples; absolute 1e-9 N.mm for a spur pair, whose couple is absent).

    Parameters
    ----------
    resolved : axisforge ResolvedTransmission
        From ``construction.gear_system`` (or any one-pair, two-shaft transmission of that layout).
    judge : dict
        :func:`mesh_judge` of the same inputs.
    theta_c_deg : float
        Direction of e_c (gear 1 -> gear 2) in the global frame [deg] (= ``phi_deg`` when gear 1 is on
        the input shaft, ``phi_deg + 180`` when gear 2 is).

    Returns
    -------
    list of dict
        Fields ``case``, ``label``, ``shaft``, ``id``, ``quantity``, ``unit``, ``value``, ``judge``,
        ``abs_diff``, ``ref_scale``, ``rel_err``, ``tol``, ``verdict``, ``tag``.
    """
    mesh = resolved.meshes[0]
    F_tw = judge["F_tw_N"]
    common = {"case": case, "label": label or resolved.label}
    rows = []

    def rec(cid, shaft, quantity, unit, value, ref, scale, tol=TOL_IDENTITY, tag=""):
        diff = float(value - ref) if np.ndim(value) == 0 else float(np.max(np.abs(np.subtract(value, ref))))
        err = abs(diff) / scale if scale > 0.0 else abs(diff)
        verdict = "RECORD" if tol is None else ("PASS" if err <= tol else "MISS")
        rows.append({**common, "shaft": shaft, "id": cid, "quantity": quantity, "unit": unit,
                     "value": float(np.linalg.norm(value)) if np.ndim(value) else float(value),
                     "judge": float(np.linalg.norm(ref)) if np.ndim(ref) else float(ref), "abs_diff": diff,
                     "ref_scale": scale, "rel_err": err, "tol": math.nan if tol is None else tol, "verdict": verdict,
                     "tag": tag})

    rec("I1", "mesh", "F_tw", "N", mesh.F_tw_N, F_tw, F_tw)
    rec("I2", "mesh", "F_r", "N", mesh.F_r_N, judge["F_r_N"], F_tw)
    rec("I3", "mesh", "F_a", "N", mesh.F_a_N, judge["F_a_N"], F_tw)
    rec("I3s", "mesh", "F_1,x signed", "N", mesh.on_1[2], judge["on_1"][2], F_tw)

    th = math.radians(theta_c_deg)
    e_c = np.array([math.cos(th), math.sin(th)])        # (Y, Z) components of e_c
    e_t = np.array([-math.sin(th), math.cos(th)])       # e_x x e_c: e_c turned +90 deg about +X
    for side, ss in ((1, resolved.shafts[0]), (2, resolved.shafts[1])):
        F_c, F_t, F_x = judge["on_1"] if side == 1 else judge["on_2"]
        rho = judge["rho_mm"][side - 1]
        loads_yz = np.array([sum(ld.Fy for ld in ss.radial_loads) + sum(ld.resultant_Fy for ld in ss.distributed_radial_loads),
                             sum(ld.Fz for ld in ss.radial_loads) + sum(ld.resultant_Fz for ld in ss.distributed_radial_loads)])
        rec("I4", ss.name, "transverse force (Y, Z)", "N", loads_yz, F_c * e_c + F_t * e_t, F_tw)
        rec("I8", ss.name, "AxialLoad sum vs F_x", "N", sum(ld.magnitude for ld in ss.axial_loads), F_x, F_tw)
        physical = np.array(physical_couples(F_x, rho, theta_c_deg))
        consumed = np.array([sum(c for _, c in plane_couples(ss, p)) for p in ("XY", "XZ")])
        named = np.array([sum(m.Mz for m in ss.external_moments), -sum(m.My for m in ss.external_moments)])
        couple = judge["F_a_N"] * abs(rho)               # F_a r_w of this gear [N.mm]
        scale, tol5 = (couple, TOL_IDENTITY) if couple > 0.0 else (1.0, 1e-9)
        rec("I5", ss.name, "couple (C_XY, C_XZ) as consumed vs r x F", "N.mm", consumed, physical, scale, tol5, J16)
        rec("I6", ss.name, "couple modulus vs F_a r_w", "N.mm", float(np.hypot(*consumed)), couple, scale, tol5, J16)
        rec("T5", ss.name, "couple (C_XY, C_XZ) read by the names My, Mz vs r x F (CW-01)", "N.mm", named, physical,
            scale, None, J16)
    for name, residual, ok in resolved.checks:
        rows.append({**common, "shaft": "transmission", "id": "I7", "quantity": name, "unit": "", "value": residual,
                     "judge": 0.0, "abs_diff": residual, "ref_scale": math.nan, "rel_err": math.nan, "tol": 1e-9,
                     "verdict": "PASS" if ok else "MISS", "tag": ""})
    return rows
