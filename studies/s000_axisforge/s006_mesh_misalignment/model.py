"""The model of s006: the shaft-bending part of f_sh, bearing tilt and pinion twist from the solved shafts of one stage.

No AxisForge or bridge function projects shaft slopes on the plane of action (proposals.md s006;
``axisforge/solvers/machine_elements/gears/iso6336.py`` ``face_load_factors`` is a placeholder), so the
projection is written here, from the mesh loads on the wheel and the nodal rotations of ``ShaftResults``.

Geometry. Both shafts lie along +X; a gear body is rigid and fixed to its shaft cross-section at the
station x_g, so its axis direction is ``e_x + theta_y e_y + theta_z e_z`` with ``theta_y = theta_xy =
dv_y/dx`` and ``theta_z = theta_xz = dv_z/dx`` (r01 / AxisForge per-plane slopes, D-0010: theta_xz is NOT a
right-hand rotation about Y, which would be -dv_z/dx), small slopes. The transverse part of the normal
tooth force lies on the line of action of the loaded flank, so its unit vector ``l`` (Y, Z) is the
direction of the transverse force of the pinion on the wheel: the sum of the ``RadialLoad`` that
AxisForge put on the wheel shaft (``mesh_forces_to_shaft_loads``), normalised. It follows the driver and
the sense of rotation by itself (D-0010 sign table), with no convention of this study. The plane of
action contains ``l`` and X.

**f_sh_bend, the shaft-bending part of f_sh** (ISO_6336-1_2019 Table 2, doc p. 6 / PDF p. 14: f_sh is the
"component of equivalent misalignment due to deformations of pinion and wheel shafts"; NOT f_ma, which is
the misalignment from manufacturing deviations, doc p. 5 / PDF p. 13; footnote b: the components in the
plane of action are determinant). Bending only, rigid gear bodies; the torsion of the pinion is recorded
apart (:func:`pinion_twist`); f_ma, bearings (f_be) and case (f_ca) are not modelled. The relative
displacement of the wheel with respect to the pinion along ``l`` varies along the face as
``(theta_wheel - theta_pinion) . l`` per unit length, so over the face width b

    f_sh_bend = b (theta_wheel - theta_pinion) . l           [mm; reported in um]

with theta at mid-face: the TANGENT at x_g times b. ISO compares the face ends (F_betax, 7.5.3.4, doc p. 54
/ PDF p. 62); the chord differs from the tangent by (V_L + V_R)(b/2)^3/(6 EI) per shaft, at most 1.6 % of
the value here (offset gear, L 100 mm; 0.1 % at L 400) (s006 physics review).

**Sign: f_sh_bend > 0 when the wheel moves away from the pinion along l toward +X: the flanks open at the
+X end of the face and the load concentrates at the -X end.** The component along ``l x e_x``
(perpendicular to the plane of action, in the transverse plane) changes the centre distance along
the face, not the flank contact to first order: RECORD (``f_perp_um``).

Units: mm, rad inside; misalignments in um (x 1000, written once in :func:`mesh_misalignment`).
"""
from __future__ import annotations

import numpy as np

from studies.v000_axisforge.v001_shaft_uniform.check import node_value

__all__ = ["line_of_action", "mesh_misalignment", "bearing_tilts", "pinion_twist", "MM_TO_UM"]

MM_TO_UM = 1e3                  # [um/mm]


def line_of_action(wheel_shaft) -> np.ndarray:
    """Unit vector (Y, Z) of the transverse line of action of the loaded flank, pinion -> wheel.

    Parameters
    ----------
    wheel_shaft : axisforge ShaftSystem
        The loaded wheel shaft of a ``ResolvedTransmission``: its ``RadialLoad`` (and distributed) are the
        transverse force of the mesh on the wheel, F_c e_c + F_t e_t (``mesh_forces_to_shaft_loads``;
        v002 I4 judges them against the inputs).

    Returns
    -------
    ndarray, shape (2,)
        ``(sum F_y, sum F_z) / |.|``; its modulus before normalising is F_bt (the transverse force on
        the line of action).
    """
    f = np.array([sum(ld.Fy for ld in wheel_shaft.radial_loads) + sum(ld.resultant_Fy for ld in wheel_shaft.distributed_radial_loads),
                  sum(ld.Fz for ld in wheel_shaft.radial_loads) + sum(ld.resultant_Fz for ld in wheel_shaft.distributed_radial_loads)])
    return f / np.linalg.norm(f)


def mesh_misalignment(theta_pinion, theta_wheel, v_pinion, v_wheel, l_yz, b_mm: float) -> dict:
    """Relative misalignment of the two gear axes over the face width, in and across the plane of action.

    Parameters
    ----------
    theta_pinion, theta_wheel : (float, float)
        (theta_xy, theta_xz) of each shaft at the gear station [rad] (FEM nodes or a closed form).
    v_pinion, v_wheel : (float, float)
        (v_xy, v_xz) of each shaft at the gear station [mm].
    l_yz : array (2,)
        Unit line of action (Y, Z), :func:`line_of_action`.
    b_mm : float
        Face width [mm].

    Returns
    -------
    dict
        ``f_sh_bend_um`` (sign: module docstring), ``f_perp_um`` (across the plane of action, RECORD),
        ``dtheta_rad`` (modulus of the relative slope), ``u_rel_um`` (relative displacement of the
        wheel along l at the station: the change of the mesh approach, RECORD).
    """
    d_theta = np.subtract(theta_wheel, theta_pinion)
    n = np.array([-l_yz[1], l_yz[0]])                  # in the transverse plane, perpendicular to l
    return {"f_sh_bend_um": MM_TO_UM * b_mm * float(d_theta @ l_yz), "f_perp_um": MM_TO_UM * b_mm * float(d_theta @ n),
            "dtheta_rad": float(np.hypot(*d_theta)),
            "u_rel_um": MM_TO_UM * float(np.subtract(v_wheel, v_pinion) @ l_yz)}


def bearing_tilts(system, result) -> list[dict]:
    """Slope of the shaft at each pin: the misalignment a bearing there would see (rigid pins: s004).

    ``psi_rad`` = hypot(theta_xy, theta_xz); ``psi_load_rad`` = slope component along the direction of
    the support force on the shaft, (R_xy, R_xz)/R, i.e. the tilt in the plane of X and the bearing
    load; ``psi_n_rad`` = component perpendicular to it (``# OPEN R5-61``: proposals.md "psi_n" read as
    the tilt normal to the load plane). Signs: positive slope toward +v of each plane.

    Returns
    -------
    list of dict
        Per bearing: ``bearing``, ``x_mm``, ``R_N``, ``theta_xy_rad``, ``theta_xz_rad``, ``psi_rad``,
        ``psi_load_rad``, ``psi_n_rad``.
    """
    out = []
    for k, brg in enumerate(system.bearings):
        th = np.array([node_value(result, brg.position, "theta_xy"), node_value(result, brg.position, "theta_xz")])
        R = np.array([result.R_xy[k], result.R_xz[k]], dtype=float)
        r = R / np.linalg.norm(R)
        out.append({"bearing": brg.label or f"pin_{brg.position:g}", "x_mm": brg.position,
                    "R_N": float(np.linalg.norm(R)), "theta_xy_rad": float(th[0]), "theta_xz_rad": float(th[1]),
                    "psi_rad": float(np.hypot(*th)), "psi_load_rad": float(th @ r),
                    "psi_n_rad": float(th @ np.array([-r[1], r[0]]))})
    return out


def pinion_twist(result, x_g_mm: float, b_mm: float, r_b_mm: float) -> dict:
    """Torsional twist of the pinion shaft across the face width (RECORD).

    ``phi(x_g + b/2) - phi(x_g - b/2)`` [rad] times the base radius r_b is the twist seen along the line
    of action [um], signed as AxisForge's phi (positive about +X from -X to +X). A point torque at x_g
    twists the half face on the loaded side by T (b/2)/GJ; a torque spread uniformly over b gives the same
    end-to-end twist T b/(2 GJ) (only the shape along b differs; s006 physics / gear-expert review). The
    section is the d 40 shaft running through the face (the pinion body, root diameter about 62.5 mm, is
    about 6 times stiffer in torsion): an upper bound. RECORD; whether it adds to or offsets f_sh_bend
    depends on the side the torque enters (ISO 6336-1 combines them with sign; R5-63).
    """
    dphi = node_value(result, x_g_mm + 0.5 * b_mm, "phi") - node_value(result, x_g_mm - 0.5 * b_mm, "phi")
    return {"twist_rad": dphi, "twist_um": MM_TO_UM * r_b_mm * dphi}

