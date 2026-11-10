"""The model of s003: a shaft on three ISO/TS 16281 ball bearings, solved for the ring displacements by Newton.

Physics. On three supports the shaft is statically INDETERMINATE: the reactions depend on how much each
bearing yields, so the bearing laws and the beam must be solved together (the step that makes the system
solve iterative, proposals.md s003). The beam is AxisForge's: v005 ``study.hertz_problem`` assembles its
stiffness K and load f on the mesh once and condenses them EXACTLY on the transverse DOFs u_k of the bearing
nodes (``K_c u - f_c`` = the force the shaft needs from the bearings, K_ss = the shaft on its bearings as pins,
non-singular). Each bearing is a radial law of its ring displacement d_k = u_k - e_k, the shaft centre relative
to its seat (e_k the seat offset), along the load plane:

    F_k = -sign(d_k) Fr_k(|d_k|)       [N, on the shaft along +Y]

the ball sums of ISO/TS 16281 at a prescribed delta_r = |d_k| (:func:`ball_loads`). The shaft centre moving
toward +Y compresses the balls on the +Y side, which push the shaft back toward -Y: the minus sign is physics,
the same as s004 ``DISPLACEMENT_SIGN = -1`` (the shaft centre moves along the force ON the bearing). Equilibrium
of the bearing nodes, with the unknowns d (so a stiff bearing does not lose its d to round-off of u = d + e):

    g(d) = K_c (d + e) - f_c - F(d) = 0      [N]

solved by ``numerical_solvers.newton`` (central finite-difference Jacobian, line search, ``lm`` fallback):
displacement-controlled, no nested ``solve_contact`` (its ``run_root`` tolerance 1e-6 would be noise to the
Jacobian, proposals.md). ``# OPEN R5-161``: proposals.md sketches a force method (two bearings as rigid
reference supports, the third bearing's reaction fed back as a ``RadialLoad``, n + 1 FEM solves per
iteration); the condensation is the same equilibrium in one assembly (v005 N3 / N4: exact to 1e-11).

Assumptions: one bending plane (XY; spur gear, every force along Y), so d is a signed scalar and the radial
direction is fixed (``phi`` never undefined; a two-plane load needs the vector law, ``# OPEN R5-165``). Rings
untilted in the law (psi = 0): the shaft slope at the bearing is recorded, not fed back (no bearing moment, no
tilt kinematics; s004 found the tilt coupling large in the kinematics, a later step). Rigid housing. Cage:
``phi_j`` is measured from the CURRENT load line with one ball on it (ISO/TS 16281 as AxisForge codes it): the cage
turns with the load direction. A cage fixed in space would, for odd Z, see a reversed load half a pitch off (Fr
-2 ... -3 % at s = 10 um, +0.6 % at s = 0; physics review; recorded per bearing as ``cage_half_pitch_Fr_change``; ``# OPEN R5-182``).

Conventions. ``F`` and the ``ShaftResults.R_xy`` of the expanded shaft are the force of the bearing ON the
shaft (AxisForge's sign of ``R``, fed back as +R: v005 F0b). A bearing solved by AxisForge at this force
(s004 ``ring_displacement``, s001 ``solve_bearing``) receives it as its node force, so its loaded zone sits
180 deg from the push of the shaft and its tilt projection ``psi . R_hat`` has the opposite sign of the tilt
seen from that zone (bearing-expert, s004 review S1): ``convention-dependent (R-003)``. Local azimuths and the
signed scalar law here are convention-free.

Units: mm, N, N/mm^1.5 (AxisForge); um only in record keys ``*_um``.
"""
from __future__ import annotations

import math

import numpy as np
from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281.contact_solver import ISO16281BallSolver

from numerical_solvers import NewtonOptions, newton

__all__ = ["ball_loads", "solve_shaft", "NEWTON", "K_MIN_N_PER_MM"]

# Newton on the residual scaled by P: residual_abs 1e-10 = 2e-7 N at P = 2 kN, above the round-off floor of
# K_c u (v005: ~1e-12 scaled, R5-155); step test as v005. Central differences: the law is C1 (delta^1.5 at
# contact onset), not C2, so a central step (eps^(1/3)) keeps the Jacobian error well below the step size
NEWTON = dict(residual_abs=1e-10, step_rel=1e-13, max_iter=50, jacobian_fd="central", fallback="lm")
# [N/mm] secant spring of an unloaded bearing given to v005.check (r01 needs k_r > 0): 1e-6 N/mm x 0.03 mm =
# 3e-8 N against reactions of 1e3 N, below the 1e-9 of the check
K_MIN_N_PER_MM = 1e-6


def ball_loads(bearing, delta_r_mm: float) -> tuple[float, np.ndarray]:
    """Radial force and ball loads of one ball bearing at a PRESCRIBED radial ring displacement (no axial load).

    ISO/TS 16281 (2008) 4.2.2, through AxisForge's own kinematics ``ISO16281BallSolver.elements`` (Eqs (12)-(15)),
    then the radial sum Eq. (16) as ``solve_contact`` writes it: ``Q_j = c_p delta_j^1.5``,
    ``Fr = sum_j Q_j cos(alpha_j) cos(phi_j)``. AxisForge has no public displacement-to-force function
    (``solve_contact`` is force-controlled), so the sum is written here (proposals.md: a D-0018 exception).

    Axial displacement: with Fa = 0 and untilted rings the axial balance Eq. (17) holds exactly at
    ``delta_a = -A sin(alpha_0)`` (every V_j = 0, so every alpha_j = 0; v003 check: the deep groove family
    carries the clearance in alpha_0 = arccos(1 - s / (2 A))). Then delta_j = max(delta_r cos(phi_j) - s/2, 0),
    Harris's radial-clearance law. Deep groove (symmetric groove) bearings only.

    Parameters
    ----------
    bearing : axisforge Bearing
        Deep groove ball bearing (s001 ``deep_groove``): ``A`` [mm], ``alpha_0`` [rad], ``phi_j`` [rad], ``cp``
        [N/mm^1.5].
    delta_r_mm : float
        Radial ring displacement along the load line [mm], >= 0.

    Returns
    -------
    Fr_N : float
        Magnitude of the radial force of the balls on the inner ring [N], >= 0 (0 inside the clearance,
        delta_r <= s/2); its sign on the shaft is set in :func:`solve_shaft` (against d).
    Q_j_N : ndarray (Z,)
        Ball loads [N] at the local azimuths ``bearing.phi_j`` (0 on the load line).
    """
    if delta_r_mm < 0.0:
        raise ValueError(f"delta_r must be >= 0, got {delta_r_mm} mm")
    A, a0 = bearing.A, bearing.alpha_0
    _, _, ca, _, cphi, d32 = ISO16281BallSolver.elements(bearing, delta_r_mm, -A * math.sin(a0),
                                                         np.zeros(len(bearing.phi_j)))
    Q = bearing.cp * d32
    return float(np.sum(Q * ca * cphi)), Q


def solve_shaft(problem: dict, bearings: dict, *, seat_mm: dict | None = None, scale: float = 1.0,
                k_linear_N_per_mm: float | None = None, label: str = "") -> dict:
    """Shaft on three bearings: Newton on the ring displacements d of the condensed beam (module docstring).

    Parameters
    ----------
    problem : dict
        v005 ``study.hertz_problem`` of the shaft (``K_c`` [N/mm], ``f_c`` [N], ``positions`` [mm], ``expand``,
        ``f_scale`` [N]); only the condensed beam is used, not its Hertz law.
    bearings : dict
        ``{x [mm]: Bearing}`` at every position of ``problem`` (construction ``bearings``).
    seat_mm : dict, optional
        ``{x [mm]: e [mm]}`` seat offsets along +Y; positions not listed sit at 0.
    scale : float
        Multiplies every bearing force (the stiff-bearing limit: scale -> inf gives the rigid supports).
    k_linear_N_per_mm : float, optional
        Replace the ISO law by a linear spring ``F = -k d`` at every bearing (the linear limit, judged by v005).
    label : str
        Name of the Newton solve.

    Returns
    -------
    dict
        ``newton`` (NewtonResult), ``positions`` [mm] (sorted), ``d_mm``, ``u_mm`` (shaft centre, d + e),
        ``e_mm``, ``F_N`` (bearing forces on the shaft along +Y), ``Q_j_N`` (per bearing), ``secant`` (``{x:
        (k [N/mm], 0)}``, v005 springs: ``k = |F / d|``, at least ``K_MIN_N_PER_MM``), ``settlements_mm``
        (``{"XY": {x: e}}``, the spring bases), ``result`` (ShaftResults of the whole shaft, AxisForge's
        post-processing). All arrays in position order.
    """
    pos = list(problem["positions"])
    if set(bearings) != set(pos):
        raise ValueError(f"bearings at {sorted(bearings)} mm, shaft bearings at {sorted(pos)} mm")
    seat_mm = seat_mm or {}
    e = np.array([seat_mm.get(x, 0.0) for x in pos])
    K_c, f_c = problem["K_c"], problem["f_c"]
    P = float(problem["f_scale"][0])
    blist = [bearings[x] for x in pos]

    def force(d):                                              # bearing forces on the shaft along +Y [N]
        if k_linear_N_per_mm is not None:
            return -k_linear_N_per_mm * np.asarray(d)
        return np.array([-math.copysign(scale * ball_loads(b, abs(dk))[0], dk) for b, dk in zip(blist, d)])

    def residual(d):
        return K_c @ (d + e) - f_c - force(d)

    # start: linear springs k0 = F0 / d_F0, F0 = P / 3, d_F0 = s/2 + (F0 / (scale cp Z))^(2/3) (the heuristic of
    # AxisForge's ISO16281BallSolver._initial_delta_r, not from the standard)
    F0 = P / len(pos)
    half_gap = np.array([b.A * (1.0 - math.cos(b.alpha_0)) for b in blist])     # s/2 [mm] (alpha_0 carries s)
    d_F0 = half_gap + np.array([(F0 / (scale * b.cp * b.Z)) ** (2.0 / 3.0) for b in blist])
    k0 = np.full(len(pos), k_linear_N_per_mm) if k_linear_N_per_mm is not None else F0 / d_F0
    # every bearing in contact at the start: the linear-spring solution, each d moved out by the half clearance
    # s/2 (the law's Jacobian is zero inside the clearance, and at d = 0 when s = 0; a shaft on one touching
    # bearing is a mechanism). Each solve starts here: no continuation in e was needed (``# OPEN R5-166``)
    x0 = np.linalg.solve(K_c + np.diag(k0), f_c - K_c @ e)
    if k_linear_N_per_mm is None:
        x0 = x0 + np.where(x0 >= 0.0, 1.0, -1.0) * half_gap
    res = newton(residual, x0, options=NewtonOptions(x_scale=d_F0, f_scale=np.full(len(pos), P), **NEWTON),
                 name="s003", label=label)
    d = np.asarray(res.x, dtype=float)
    F = force(d)
    Q = [ball_loads(b, abs(dk))[1] for b, dk in zip(blist, d)]
    k_sec = {x: (max(abs(Fk / dk) if dk != 0.0 else 0.0, K_MIN_N_PER_MM), 0.0) for x, Fk, dk in zip(pos, F, d)}
    order = np.argsort(pos)
    return {"newton": res, "positions": [pos[i] for i in order], "d_mm": d[order], "u_mm": (d + e)[order],
            "e_mm": e[order], "F_N": F[order], "Q_j_N": [Q[i] for i in order], "secant": k_sec,
            "settlements_mm": {"XY": {x: float(ek) for x, ek in zip(pos, e)}}, "result": problem["expand"](d + e)}
