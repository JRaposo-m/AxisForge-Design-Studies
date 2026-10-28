"""The model of s004: the rigid-body field of a two-bearing shaft whose bearings yield, and its tilt fixed point.

s004 owns this code; s003 (hyperstatic shaft on ISO/TS 16281 bearings) imports it through
``s004.__init__`` (D-0036 amendment item 11). The bearing model is s001's (``load_node``, ``solve_bearing``,
``records``: AxisForge's ISO/TS 16281 solvers); the shaft is any AxisForge ``ShaftResults`` on two pins.

Physics. On two pins the shaft is statically determinate: the reactions R_A, R_B follow from the loads
alone, so bearings that yield change no force, only the kinematics. Each bearing, loaded by the force the
shaft puts on it (-R_k, R_k the support force on the shaft), lets its inner ring, i.e. the shaft centre,
move by delta_r,k along that force. The shaft then moves as a rigid body on top of the rigid-pin field
(exact superposition for the linear beam):

    v(x) = v_FEM(x) + d_A + (d_B - d_A) (x - x_A) / (x_B - x_A),     theta(x) = theta_FEM(x) + (d_B - d_A) / (x_B - x_A)

(v001 ``rigid_body`` per plane). The bearing tilt is the shaft slope at the bearing, the outer ring held by
the housing: psi_k = theta_FEM(x_k) + (d_B - d_A)/L. But delta_r depends (weakly) on psi through the
bearing's own tilt kinematics, so psi and d are found by a plain fixed point (proposals.md s004):

    psi_k <- theta_FEM(x_k) + (d_B(psi_B) - d_A(psi_A)) / L,        until max |delta psi| < 1e-10 rad

started from the rigid-pin slope. No Newton: the map is a contraction because d_psi(delta_r) is tiny.

Conventions. Vectors are (Y, Z) components: a displacement (v_xy, v_xz), a slope (theta_xy, theta_xz) =
(dv_y/dx, dv_z/dx). The AxisForge bearing solver receives the support force on the shaft as its node force (v003
sign check: Q_max lands 180 deg from the load on the shaft) and projects the node slope on that direction,
``psi = psi_xz cos(phi_Fr) + psi_xy sin(phi_Fr)`` (``contact_solver.py:194-201``); this model feeds psi the same way
(``psi_in = psi . R_hat``) and records the normal part ``psi_n`` that the solver drops (``contact_solver.py:201``).
That feed is the one ``convention-dependent (R-003)`` part: AxisForge places its loaded zone (and the tilt relative
to it) 180 deg off. Here it costs nothing: for Fa = 0 and a symmetric bearing delta_r and Q_max are unchanged under
R -> -R and psi -> -psi (bearing-expert review), so no s004 number depends on it; it matters once Fa != 0 or Mz is
fed back (s003, s005). The displacement of the shaft centre is NOT a convention: by statics the shaft pushes the
inner ring with -R_k, so the centre moves ALONG THE FORCE ON THE BEARING, ``d_k = delta_r,k (-R_hat_k)``
(``DISPLACEMENT_SIGN = -1``; rigid housing, outer ring fixed). AxisForge never maps delta_r back to the shaft. The
study runs the opposite sign once as an unphysical sensitivity RECORD (the gear then deflects less than on rigid
pins).

Scope. ``coupled_tilt`` holds the reactions at their rigid-pin values: valid ONLY for a statically determinate shaft
on two pins (it raises otherwise). On a hyperstatic shaft (s003) the reactions change with d; s003 may reuse
``ring_displacement`` and ``rigid_line``, not ``coupled_tilt``.

Units: mm, N, rad.
"""
from __future__ import annotations

import math

import numpy as np

from studies.s000_axisforge.s001_bearing_clearance_and_contact_angle import load_node, records, solve_bearing
from studies.v000_axisforge.v001_shaft_uniform.check import node_value, rigid_body

__all__ = ["support_force", "ring_displacement", "rigid_line", "coupled_tilt", "TOL_PSI_RAD", "MAX_PASSES",
           "DISPLACEMENT_SIGN"]

TOL_PSI_RAD = 1e-10            # [rad] fixed point stops when the largest tilt change of a pass is below this
MAX_PASSES = 50
# d_k = delta_r (-R_hat): the shaft centre moves along the force on the bearing (statics, not a convention)
DISPLACEMENT_SIGN = -1.0


def support_force(result, k: int) -> np.ndarray:
    """Support force on the shaft at bearing ``k`` of the FEM (``ShaftResults.R_xy``, ``R_xz``), (Y, Z) [N]."""
    return np.array([float(result.R_xy[k]), float(result.R_xz[k])])


def ring_displacement(bearing, R_yz, psi_yz, *, sign: float = DISPLACEMENT_SIGN) -> dict:
    """Shaft-centre displacement at one bearing from its ISO/TS 16281 solve at the FEM reaction and a tilt.

    Parameters
    ----------
    bearing : axisforge Bearing
        s001 build (ISO/TS 16281).
    R_yz : array (2,)
        Support force on the shaft at the bearing, (Y, Z) [N]; the bearing carries |R| radially.
    psi_yz : array (2,)
        Shaft slope at the bearing, (theta_xy, theta_xz) [rad].
    sign : float
        -1: the shaft centre moves along -R_hat (statics, default); +1 is unphysical, a sensitivity check only.

    Returns
    -------
    dict
        ``d_yz`` [mm] (array), ``delta_r_mm``, ``psi_in_rad`` (fed to the solver: psi . R_hat, AxisForge's
        projection), ``psi_n_rad`` (normal part, dropped by the solver), ``Q_max_N``, ``n_loaded``,
        ``ok``, ``balanced``, ``residual_N``, ``analysis`` (the AxisForge result), ``error``.

    Warnings
    --------
    ``psi_in`` follows AxisForge's projection on the SUPPORT force direction (R-003); the tilt seen in the plane of
    the force on the bearing is ``-psi_in``. delta_r is even in psi only for Fa = 0 and a symmetric bearing (s004);
    with Fa != 0 (helical, s003) the sign of psi against Fa matters and must be ruled first.
    """
    R = np.asarray(R_yz, dtype=float)
    Fr = float(np.linalg.norm(R))
    r_hat = R / Fr
    psi = np.asarray(psi_yz, dtype=float)
    psi_in = float(psi @ r_hat)
    psi_n = float(psi @ np.array([-r_hat[1], r_hat[0]]))
    # bearing-solver azimuth phi_Fr, measured from +Z toward +Y: Fr_xz = Fr cos(phi), Fr_xy = Fr sin(phi) (s001
    # load_node, contact_solver.py:194); the AxisForge README (D-0010) measures load angles from +Y toward +Z
    node = load_node(bearing, Fr, phi_Fr_rad=math.atan2(R[0], R[1]))
    analysis, post, error = solve_bearing(bearing, node, psi_rad=psi_in, postprocess=False)
    case, _ = records(bearing, analysis, postprocessed=post, error=error)
    delta_r = case.get("delta_r_mm", math.nan)
    return {"d_yz": sign * delta_r * r_hat, "delta_r_mm": delta_r, "psi_in_rad": psi_in, "psi_n_rad": psi_n,
            "Q_max_N": case.get("Q_max_N", math.nan), "n_loaded": case.get("n_loaded", 0), "ok": case["ok"],
            "balanced": case.get("balanced", False), "residual_N": case.get("residual_N", math.nan),
            "analysis": analysis, "error": error}


def rigid_line(x, x_A: float, x_B: float, d_A_yz, d_B_yz) -> tuple[np.ndarray, np.ndarray]:
    """Rigid-body field through the displaced bearing centres, both planes (v001 ``rigid_body`` per plane).

    Valid for any shaft: it is the kinematics of a rigid body through two displaced points (on a hyperstatic
    shaft it is only the rigid part of the field, the elastic part changes with the reactions).

    Parameters
    ----------
    x : array_like
        Axial stations [mm].
    x_A, x_B : float
        Positions of the two bearing centres [mm].
    d_A_yz, d_B_yz : array (2,)
        Displacement (Y, Z) of each centre [mm] (``ring_displacement`` ``d_yz``).

    Returns
    -------
    (ndarray (n, 2), ndarray (2,))
        Displacement (v_xy, v_xz) at ``x`` [mm] and the constant rotation (theta_xy, theta_xz) [rad].
    """
    cols = [rigid_body(x, x_A, x_B, float(d_A_yz[i]), float(d_B_yz[i])) for i in range(2)]
    return np.column_stack([c[0] for c in cols]), np.array([c[1] for c in cols])


def coupled_tilt(system, result, bearings, *, sign: float = DISPLACEMENT_SIGN, tol: float = TOL_PSI_RAD,
                 max_passes: int = MAX_PASSES) -> dict:
    """Fixed point of the bearing tilts of a shaft on two yielding bearings (module docstring).

    Two pins, statically determinate ONLY: the reactions are held at the rigid-pin values and never updated, which
    is exact only when statics alone fixes them. A hyperstatic shaft (s003) must not use this function; it may reuse
    :func:`ring_displacement` and :func:`rigid_line`.

    Parameters
    ----------
    system : axisforge ShaftSystem
        The shaft on exactly two pins.
    result : axisforge ShaftResults
        Its rigid-pin solve; the pins must be nodes.
    bearings : (Bearing, Bearing)
        The ISO/TS 16281 bearings at pin A and pin B (position order).
    sign : float
        Displacement sign (``DISPLACEMENT_SIGN``; +1 only for the unphysical sensitivity RECORD).
    tol, max_passes : float, int
        Stopping rule: max over both bearings of |psi_new - psi| < ``tol`` [rad].

    Returns
    -------
    dict
        ``converged``, ``passes``, ``history`` (one record per pass: ``k``, ``residual_norm`` = max |delta
        psi| [rad], for the convergence figure), ``psi_rigid_yz`` and ``psi_yz`` per bearing [rad],
        ``d_yz`` per bearing [mm], ``rotation_yz`` [rad], ``bearing`` (the last ``ring_displacement``
        per bearing), ``x_mm`` (pin positions), ``R_yz`` (support force on the shaft per pin [N]); all
        lists in position order A, B.

    Raises
    ------
    ValueError
        Unless the shaft has exactly two bearings and two bearing models are given.
    """
    if len(system.bearings) != 2 or len(bearings) != 2:
        raise ValueError(f"coupled_tilt holds the reactions fixed: two pins only (statically determinate), got "
                         f"{len(system.bearings)} supports and {len(bearings)} bearings")
    pins = sorted(system.bearings, key=lambda b: b.position)
    x = [p.position for p in pins]
    R = [support_force(result, list(system.bearings).index(p)) for p in pins]
    theta = [np.array([node_value(result, xk, "theta_xy"), node_value(result, xk, "theta_xz")]) for xk in x]
    psi = [t.copy() for t in theta]                     # pass 0: the rigid-pin slope
    history, converged = [], False
    for k in range(1, max_passes + 1):
        sol = [ring_displacement(b, Rk, pk, sign=sign) for b, Rk, pk in zip(bearings, R, psi)]
        _, rot = rigid_line([x[0]], x[0], x[1], sol[0]["d_yz"], sol[1]["d_yz"])     # (d_B - d_A) / L per plane
        new = [t + rot for t in theta]
        change = max(float(np.max(np.abs(n - p))) for n, p in zip(new, psi))
        history.append({"k": k, "residual_norm": change})
        psi = new
        if change < tol:
            converged = True
            break
    return {"converged": converged, "passes": len(history), "history": history, "psi_rigid_yz": theta, "psi_yz": psi,
            "d_yz": [s["d_yz"] for s in sol], "rotation_yz": rot, "bearing": sol, "x_mm": x, "R_yz": R}
