"""The model of s005: the ISO/TS 16281 tilting moment of each bearing fed back to a two-bearing shaft.

Physics. AxisForge solves each bearing at a PRESCRIBED ring tilt psi and returns a moment Mz (ISO/TS 16281 4.2.2.2
Eq. (18), ball: ``Mz = (Dpw/2) c_p sum_j delta_j^1.5 sin(alpha_j) cos(phi_j)``; 5.2.4.2 Eq. (46), roller:
``Mz = sum_j cos(phi_j) sum_k x_k q_jk``); it never gives it back to the shaft (``contact_solver.py:780``,
``:1257``). Here the moment of the bearing ON the shaft is applied at the pin as an ``ExternalMoment``. It is minus
the moment the rings need to hold the tilt, the one conjugate to psi by virtual work (:func:`moment_on_shaft`):

* ball: the tilt enters Eq. (12) as ``+Ri sin(psi) cos(phi_j)`` in the axial term, so ``dU/dpsi = (2 Ri / Dpw) Mz18``:
  the moment on the shaft is ``-Mz18`` (the arm Dpw/2 instead of Ri is SP-16; ``2 Ri / Dpw`` = 1.007 here, its
  work-conjugate reading is a RECORD);
* roller: the tilt enters Eq. (40) as ``-x_k tan(psi_j)``, ``psi_j = psi cos(phi_j)``, so ``dU/dpsi = -Mz46``: the
  moment on the shaft is ``+Mz46`` (s005 review, bearing-expert M2 / physics SHOULD 3; it closes s002's R5-76 as a
  note: Eq. (46) is a lamina moment whose sense against psi is a convention of the clause). Fed back as ``-Mz46`` it
  would be a negative tilt stiffness: that run is kept as the RECORD ``roller_as_coded``.

On two pins the shaft stays statically determinate, but the moments enter its moment balance, so the reactions change
by ``(M_A + M_B) / L``, the slopes at the pins change, the bearing tilts change, and so Mz. With s004's rigid-body
field (ring displacements d_k along the force on the bearing) the unknowns are, per bearing, the tilt psi_k and the
moment m_k fed back, both along the load plane ``e``:

    psi_k - (theta_k(m) + (d_B - d_A) . e / L) = 0         [rad]   slope of the shaft at pin k = ring tilt
    m_k - feedback_k (G_k . e)                 = 0         [N mm]  moment fed back = moment of the bearing on the shaft

``theta_k(m) = theta_k(0) + sum_j F_kj m_j`` and ``R_k(m) = R_k(0) + sum_j C_kj m_j``: the beam is linear, so its
response to the two pin moments is taken from two unit-moment FEM solves (:func:`pin_flexibility`; exact
superposition, checked on the final FEM). ``d_k`` and ``Mz_k`` come from s004 ``ring_displacement`` at ``R_k(m)``, the
axial load ``Fa_k`` and the tilt ``psi_k``. Solved by ``numerical_solvers.newton`` (central finite-difference Jacobian,
line search, ``lm`` fallback; the Jacobian is not symmetric, SP-16). The nested bearing solve runs at
``BEARING_TOL`` = 1e-12 instead of AxisForge's default 1e-6 (a solver argument, s001 ``solve_bearing(tol=...)``): at
1e-6 the hybr root leaves Mz noise of 6e-5 relative (24.6108 vs 24.6094 N mm, 1e-4 N residual), larger than the Mz
change over a finite-difference step; at 1e-12 the residual is about 1e-11 N.

Planar. The mesh load lies on the line of action, so the reactions, slopes and moments of the two-pin shaft lie in
the plane of X and ``e`` (the unit support force at A on the rigid pins); the moments are applied along ``e`` and
the normal slope ``theta . n`` is recorded (it stays at round-off).

Conventions. The tilt and the loaded zone are AxisForge's (s004 ``ring_displacement``): the solver puts phi_j = 0
along the node force it is given, the support force on the shaft R, and its tilt is ``psi = theta . R_hat``
(``contact_solver.py:192-199``; the azimuth label of that zone is R-003). The axial load ``Fa`` is SIGNED in the
solver's axial sense (+ = the direction of ``A sin(alpha_0)``, V_j > 0, ``contact_solver.py:633``). For a symmetric
deep groove bearing, putting the zone at ``-R_hat`` with +Fa gives exactly the moments of the zone at ``+R_hat`` with
-Fa (review of s005: equal to every printed digit), so the "other zone reading" IS the other axial sense: the study
runs +Fa and -Fa and reports the effect per sense. ``# OPEN R5-179`` (R5-190, for José): which global X direction the
solver's +Fa is; only that sets which physical load case each sign is. The ring displacement is physics, s004's
``d_k = delta_r,k (-R_hat_k)`` (``DISPLACEMENT_SIGN``).
``ExternalMoment(magnitude, theta)`` is consumed by the FEM as the generalised force ``magnitude (cos theta, sin
theta)`` on the slope DOFs (dv_y/dx, dv_z/dx) (``core/loads.py`` ExternalMoment Warnings; D-0010 rows 21-22 open
for its NAMES only): a positive moment along ``e`` raises the slope along ``e`` at a pin (test ``test_moment_probe``).

Units: mm, N, N.mm, rad.
"""
from __future__ import annotations

import math

import numpy as np
from axisforge.core.loads import DistributedRadialLoad, ExternalMoment, LoadPlane

from numerical_solvers import NewtonOptions, newton
from studies.s000_axisforge.s004_bearing_compliance import (DISPLACEMENT_SIGN, rigid_line, ring_displacement,
                                                            support_force)
from studies.v000_axisforge.v001_shaft_uniform.check import node_value
from studies.v000_axisforge.v002_shaft_stepped_gear import THEORIES
from studies.v000_axisforge.v002_shaft_stepped_gear.study import solve      # OPEN R5-54 (s004): v002's solve

__all__ = ["moment_on_shaft", "with_moments", "pin_flexibility", "solve_feedback", "moment_balance", "NEWTON",
           "BEARING_TOL", "M_UNIT_NMM", "MZ_LABEL", "EB", "N_ELEMENTS"]

EB = THEORIES["euler_bernoulli"]
N_ELEMENTS = 40                 # s004's mesh: pins and the gear on nodes (Hermite nodal exactness, v001)
# OPEN R5-177: at L = 150 mm the 40 uniform nodes and the gear face edges (0.3 L +- 11.5 mm) leave a 0.25 mm element;
# the rigid FEM alone then misses v002's 1e-9 by round-off (2e-9); kept as in s004, the miss recorded (README 9)
# [-] tol of the nested ISO/TS 16281 root (AxisForge default 1e-6: module docstring). OPEN R5-178: at 1e-12 hybr can
# report failure (``ok`` False) at a root it has reached (residual about 1e-12 N); recorded
BEARING_TOL = 1e-12
M_UNIT_NMM = 1e4                # [N mm] probe moment of the flexibility solves (any value: the beam is linear)
MZ_LABEL = "s005:Mz"            # label prefix of the fed-back ExternalMoment (moment_balance leaves them out)
# Newton on residuals scaled by the rigid-pin slope and Fr Dpw/2: 1e-10 scaled = about 1e-14 rad and 3e-5 N mm,
# above the noise of the nested solve at BEARING_TOL (about 1e-5 N mm on Mz 8e4 N mm); central differences: the
# ball law is C1 at contact onset (delta^1.5) and the roller laminae switch on and off (piecewise smooth)
NEWTON = dict(residual_abs=1e-10, step_rel=1e-13, max_iter=50, jacobian_fd="central", fallback="lm")


def moment_on_shaft(bearing, ring: dict, R_yz) -> np.ndarray:
    """Moment of the bearing ON the shaft in the slope components (dv_y/dx, dv_z/dx) [N mm] (module docstring).

    ``G = -c Mz R_hat`` with ``c = +1`` for a ball bearing (Eq. (18): Mz is the moment the rings need to hold the
    tilt ``psi = theta . R_hat``) and ``c = -1`` for a roller bearing (Eq. (40) tilts the laminae by ``-x_k tan
    psi_j``, so the holding moment is ``-Mz46``). ``ring`` is s004 ``ring_displacement`` at the support force ``R_yz``.
    """
    R = np.asarray(R_yz, dtype=float)
    c = 1.0 if hasattr(bearing, "A") else -1.0      # point contact (ball, Eq. (18)) or line contact (roller, Eq. (46))
    # G = -c Mz R_hat: the reaction conjugate to psi_in = theta . R_hat (ball +Mz18, roller -Mz46 hold the tilt)
    return -c * ring["Mz_Nmm"] * R / np.linalg.norm(R)


def with_moments(system, x_mm, m_Nmm, e_yz):
    """The shaft with a bending moment ``m_k e`` at each station ``x_k`` (``ExternalMoment``, label ``MZ_LABEL``).

    ``m_k`` [N mm] is the generalised moment along the unit (Y, Z) direction ``e``: the FEM adds ``m_k e_y`` to the
    slope DOF dv_y/dx and ``m_k e_z`` to dv_z/dx (``ExternalMoment`` as consumed, ``theta`` = the angle of ``e`` from
    +Y toward +Z; a negative ``m_k`` is written as its modulus at ``theta + 180 deg``, as ``transmission.py:177``).
    Zero moments are left out.
    """
    ang = math.degrees(math.atan2(float(e_yz[1]), float(e_yz[0])))
    loads = [ExternalMoment(float(x), abs(float(m)), ang if m > 0.0 else ang + 180.0, label=f"{MZ_LABEL}:{k}",
                            source="bearing_reaction") for k, (x, m) in enumerate(zip(x_mm, m_Nmm)) if m != 0.0]
    return system.with_loads(loads) if loads else system


def _pin_state(system, result, x_mm):
    """Support force (Y, Z) [N] and slope (theta_xy, theta_xz) [rad] at each pin, in the order of ``x_mm``."""
    order = [next(i for i, b in enumerate(system.bearings) if b.position == x) for x in x_mm]
    R = [support_force(result, i) for i in order]
    th = [np.array([node_value(result, x, "theta_xy"), node_value(result, x, "theta_xz")]) for x in x_mm]
    return R, th


def pin_flexibility(system, rigid, x_mm, e_yz, *, settings=EB, n_elements: int = N_ELEMENTS) -> dict:
    """Response of the two-pin shaft to a unit moment along ``e`` at each pin (two FEM solves, superposition).

    Returns
    -------
    dict
        ``F`` (2 x 2) [rad / N mm]: ``F[k, j]`` = slope change along ``e`` at pin k per unit moment at pin j (end
        flexibility of the beam: L / (3 EI) at the loaded end of a uniform simply supported beam, L / (6 EI) at the
        other end, with the sign of the beam's own rotation); ``C`` (2, 2, 2) [1 / mm]: ``C[k, j]`` = change of the
        support force (Y, Z) at pin k per unit moment at pin j (statics: +-e / L); ``theta_n`` (2 x 2)
        [rad / N mm]: the slope change normal to ``e`` (0 for a planar problem).
    """
    R0, th0 = _pin_state(system, rigid, x_mm)
    e = np.asarray(e_yz, dtype=float)
    n = np.array([-e[1], e[0]])
    F, Fn, C = np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2, 2))
    for j in range(2):
        m = [M_UNIT_NMM if i == j else 0.0 for i in range(2)]
        R, th = _pin_state(system, solve(with_moments(system, x_mm, m, e), settings, n_elements), x_mm)
        for k in range(2):
            F[k, j] = float((th[k] - th0[k]) @ e) / M_UNIT_NMM
            Fn[k, j] = float((th[k] - th0[k]) @ n) / M_UNIT_NMM
            C[k, j] = (R[k] - R0[k]) / M_UNIT_NMM
    return {"F": F, "C": C, "theta_n": Fn}


def solve_feedback(system, pair, *, Fa_N=(0.0, 0.0), feedback=1.0, sign: float = DISPLACEMENT_SIGN,
                   label: str = "", settings=EB, n_elements: int = N_ELEMENTS) -> dict:
    """Tilts and fed-back moments of a shaft on two ISO/TS 16281 bearings: Newton on (psi_A, psi_B, m_A, m_B).

    Parameters
    ----------
    system : axisforge ShaftSystem
        The loaded shaft on exactly two pins, without the fed-back moments.
    pair : (Bearing, Bearing)
        Bearings at the pins, in position order (A, B).
    Fa_N : (float, float)
        Axial load of each bearing [N], signed in the solver's axial sense (module docstring), held fixed.
    feedback : float or (float, float)
        Multiplies the moment fed back, per bearing (A, B) or both: 1 = the moment of :func:`moment_on_shaft`;
        0 = no feedback (the s004 limit); ``2 Ri / Dpw`` = the work-conjugate arm of SP-16; -1 at a roller = Eq. (46)
        fed back as coded (RECORDs).
    sign : float
        Ring displacement sign (s004 ``DISPLACEMENT_SIGN``).
    label : str
        Name of the Newton solve.
    settings, n_elements
        Beam model and mesh (v002 ``solve``).

    Returns
    -------
    dict
        ``newton`` (NewtonResult), ``feedback`` (2,), ``G_fed_yz`` (the moment fed back per pin, ``feedback x G``
        [N mm], for :func:`moment_balance`), ``x_mm``, ``e_yz``, ``psi_rad`` (2, along ``e``), ``m_Nmm`` (2, fed
        back), ``R_yz`` (support forces with feedback, per pin), ``bearing`` (s004 ``ring_displacement`` per pin at
        the solution, plus ``G_yz``), ``rotation_yz`` [rad] ((d_B - d_A)/L), ``rigid`` (ShaftResults on rigid pins,
        no moments), ``R0_yz``, ``theta0_e_rad`` (rigid-pin slopes along e), ``flex`` (:func:`pin_flexibility`),
        ``final`` (ShaftResults of the shaft WITH the moments and the ring displacements as settlements: the check of
        the superposition), ``final_system``, ``settlements_mm``, ``superposition_theta_err_rad``,
        ``superposition_R_err`` (relative), ``theta_n_rad`` (normal slope at each pin, final FEM).

    Raises
    ------
    ValueError
        Unless the shaft has exactly two bearings and two bearing models are given.
    """
    if len(system.bearings) != 2 or len(pair) != 2:
        raise ValueError(f"two pins and two bearings only, got {len(system.bearings)} and {len(pair)}")
    fb = np.broadcast_to(np.asarray(feedback, dtype=float), (2,))
    x = sorted(b.position for b in system.bearings)
    rigid = solve(system, settings, n_elements)
    R0, th0 = _pin_state(system, rigid, x)
    e = R0[0] / np.linalg.norm(R0[0])                    # load plane: the support force at A on rigid pins
    flex = pin_flexibility(system, rigid, x, e, settings=settings, n_elements=n_elements)
    t0 = np.array([float(t @ e) for t in th0])

    def state(u):
        psi, m = u[:2], u[2:]
        R = [R0[k] + m @ flex["C"][k] for k in range(2)]          # sum_j m_j C[k, j]
        theta = t0 + flex["F"] @ m
        sol = []
        for k, b in enumerate(pair):
            ring = ring_displacement(b, R[k], psi[k] * e, sign=sign, Fa_N=Fa_N[k], tol=BEARING_TOL)
            sol.append({**ring, "G_yz": moment_on_shaft(b, ring, R[k])})
        _, rot = rigid_line([x[0]], x[0], x[1], sol[0]["d_yz"], sol[1]["d_yz"])      # s004: (d_B - d_A) / L
        return R, theta, sol, rot

    def residual(u):
        _, theta, sol, rot = state(u)
        r_psi = u[:2] - (theta + float(rot @ e))
        r_m = u[2:] - fb * np.array([float(s["G_yz"] @ e) for s in sol])
        return np.concatenate([r_psi, r_m])

    psi_ref = float(np.max(np.abs(t0)))
    M_ref = float(np.linalg.norm(R0[0])) * pair[0].Dpw / 2.0       # [N mm] Fr Dpw/2: the scale of a ball Mz
    opts = NewtonOptions(**NEWTON, x_scale=[psi_ref, psi_ref, M_ref, M_ref], f_scale=[psi_ref, psi_ref, M_ref, M_ref])
    nr = newton(residual, np.concatenate([t0, [0.0, 0.0]]), options=opts, name="s005_feedback", label=label)
    R, theta, sol, rot = state(nr.x)
    psi, m = nr.x[:2], nr.x[2:]
    settle = {"XY": (float(sol[0]["d_yz"][0]), float(sol[1]["d_yz"][0])),
              "XZ": (float(sol[0]["d_yz"][1]), float(sol[1]["d_yz"][1]))}
    final_system = with_moments(system, x, m, e)
    final = solve(final_system, settings, n_elements, settlements_mm=settle)
    Rf, thf = _pin_state(final_system, final, x)
    n = np.array([-e[1], e[0]])
    return {"newton": nr, "feedback": fb, "G_fed_yz": [fb[k] * sol[k]["G_yz"] for k in range(2)], "x_mm": x,
            "e_yz": e, "psi_rad": psi, "m_Nmm": m, "R_yz": R, "bearing": sol,
            "rotation_yz": rot, "rigid": rigid, "R0_yz": R0, "theta0_e_rad": t0, "flex": flex, "final": final,
            "final_system": final_system, "settlements_mm": settle,
            "superposition_theta_err_rad": max(abs(float(thf[k] @ e) - psi[k]) for k in range(2)),
            "superposition_R_err": max(float(np.linalg.norm(Rf[k] - R[k])) for k in range(2))
            / max(float(np.linalg.norm(r)) for r in R0),
            "theta_n_rad": [float(t @ n) for t in thf]}


def moment_balance(system, result, x_mm, G_yz) -> dict:
    """Moment equilibrium of the whole shaft about pin A, per plane, with the bearing moments ``G_yz``.

    Virtual work of a rigid rotation about A (v = (x - x_A) delta, slope = delta) in each plane: the forces of the
    loads (``RadialLoad``) and of the supports (``ShaftResults.R_xy``, ``R_xz``, on the shaft) times their arm,
    plus the bending moments (``ExternalMoment.component``, the slope-DOF force the FEM consumes) of the loads, plus
    the bearing moments on the shaft ``G_yz`` (from the bearing solves, :func:`bearing_state`, NOT the moments
    applied to the FEM, which carry ``MZ_LABEL`` and are left out): it closes only if the FEM equilibrium and the
    fed-back moments are both right. Torques about X and axial forces on the axis bend nothing.

    Returns
    -------
    dict
        ``sum_xy_Nmm``, ``sum_xz_Nmm`` (the residual moments), ``scale_Nmm`` (largest term), ``rel`` (largest
        residual over the scale).

    Raises
    ------
    ValueError
        If the shaft carries a distributed load (not summed here).
    """
    if any(isinstance(ld, DistributedRadialLoad) for ld in system.loads):
        raise ValueError("moment_balance sums point loads only")
    xA = x_mm[0]
    terms = {0: [], 1: []}
    for ld in system.radial_loads:
        terms[0].append(ld.Fy * (ld.position - xA))
        terms[1].append(ld.Fz * (ld.position - xA))
    for mo in system.external_moments:
        if not (mo.label or "").startswith(MZ_LABEL):
            terms[0].append(float(mo.component(LoadPlane.XY)))
            terms[1].append(float(mo.component(LoadPlane.XZ)))
    R, _ = _pin_state(system, result, x_mm)
    for k, xk in enumerate(x_mm):
        for p in (0, 1):
            terms[p].append(float(R[k][p]) * (xk - xA))
            terms[p].append(float(G_yz[k][p]))
    sums = [math.fsum(terms[p]) for p in (0, 1)]
    scale = max(abs(t) for p in (0, 1) for t in terms[p])
    return {"sum_xy_Nmm": sums[0], "sum_xz_Nmm": sums[1], "scale_Nmm": scale, "rel": max(map(abs, sums)) / scale}
