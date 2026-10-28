"""Load distribution of a ball bearing with any ball law Q(delta): AxisForge's kinematics, an own force balance.

The ring kinematics are AxisForge's ISO/TS 16281 ones, reused as they are (``ISO16281BallSolver.elements``, Eqs
(12)-(15)): for ring displacements ``delta_r``, ``delta_a`` [mm] every ball j has a total approach ``delta_j`` (both
contacts) [mm] and an operating angle ``alpha_j``. What changes is the ball's law ``Q_j = law(delta_j)``:

* ``cp``: AxisForge's ``Q = c_p delta^1.5`` (ISO Eq. (11) fit of the two contacts in series). With it this solver
  must give AxisForge's own result: the limit case;
* ``hertz``: the Hertz series of the two contacts, ``Q = K_s delta^1.5`` with
  ``K_s = (K_i^(-2/3) + K_o^(-2/3))^(-3/2)`` (Harris2007_vol1 Eqs 7.5-7.7, PDF p. 156), the K of each contact from
  the bridge's Hertz (judged against r02 by v105); it is the BEM series law with ``c = 0``;
* ``bem``: the s101 BEM laws of the two contacts in series at equal load (``series_law``).

The radial and axial force balance of the inner ring (ISO Eqs (16)-(17), the same as AxisForge's ``solve_contact``)

    F_r - sum_j Q_j cos(alpha_j) cos(phi_j) = 0,      F_a - sum_j Q_j sin(alpha_j) = 0

is solved for (delta_r, delta_a) by ``numerical_solvers.newton`` (never AxisForge's numerics). The rings are untilted
(psi = 0) and the azimuth is LOCAL (phi_j = 0 on the radial-load line): convention-free; a global reading is
``convention-dependent (R-003)``.

Units: mm, N, rad (AxisForge); the s101 laws are SI and are converted in :func:`law_mm`, the one place.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Callable

import numpy as np
from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281.contact_solver import ISO16281BallSolver

from numerical_solvers import NewtonOptions, newton
# OPEN R5-103: private; numerical_solvers keeps no Jacobian in NewtonResult, and this is its own central difference
from numerical_solvers.newton_raphson.newton import _fd_jacobian
from studies.s100_dry_contact.s101_bem_ball_raceway_law import BemLaw, fit_law

__all__ = ["series_law", "law_mm", "power_law", "solve_distribution"]

_M_PER_MM = 1e-3


def series_law(name: str, inner: BemLaw, outer: BemLaw) -> BemLaw:
    """The ball's law: the two contact laws in series at equal load, ``delta = delta_i(Q) + delta_o(Q)`` [m].

    Each contact's ``delta(Q)`` is the inverse of its smooth law (``BemLaw.delta``). The ball's law has the same
    smooth form, ``Q = K_s delta^1.5 (1 + c delta)`` with ``K_s = (K_i^(-2/3) + K_o^(-2/3))^(-3/2)`` the Hertz
    series, its ``c`` fitted on the union of the two load tables (``fit_law``; its rms is the series misfit).
    """
    Q = np.unique(np.concatenate([inner.Q_N, outer.Q_N]))
    d = inner.delta(Q) + outer.delta(Q)
    K_s = (inner.K_hertz_N_per_m1_5 ** (-2.0 / 3.0) + outer.K_hertz_N_per_m1_5 ** (-2.0 / 3.0)) ** -1.5
    c, rms = fit_law(d, Q, K_s)
    return BemLaw(name=name, delta_m=tuple(d), Q_N=tuple(Q), K_hertz_N_per_m1_5=K_s, c_per_m=c, fit_rms=rms)


def law_mm(law: BemLaw, *, hertz: bool = False) -> Callable[[np.ndarray], np.ndarray]:
    """A ball law in AxisForge units: ``Q [N] = law(delta [mm])``; 0 for delta <= 0 (no tension).

    The one unit conversion of the case (SI law, AxisForge mm). ``hertz=True`` drops the BEM ratio (``c = 0``):
    ``Q = K_H delta^1.5``, the Hertz law of the same contacts.
    """
    law = replace(law, c_per_m=0.0) if hertz else law
    return lambda delta_mm: law.Q(np.asarray(delta_mm, dtype=float) * _M_PER_MM)


def power_law(K_N_per_mm15: float) -> Callable[[np.ndarray], np.ndarray]:
    """``Q = K delta^1.5`` in AxisForge units [N, mm]; 0 for delta <= 0."""
    return lambda delta_mm: K_N_per_mm15 * np.maximum(np.asarray(delta_mm, dtype=float), 0.0) ** 1.5


def solve_distribution(bearing, law, Fr_N: float, Fa_N: float = 0.0, *, name: str = "", label: str = ""):
    """Force balance of one untilted ball bearing with the ball law ``law`` (``Q [N] = law(delta [mm])``).

    Newton settings (numerical_solvers README pattern (b)): ``x_scale`` [1e-2, 1e-1] mm (delta_r is 0.01-0.04 mm,
    delta_a up to 0.14 mm), ``f_scale`` [Fr, max(Fa, Fr)] (a residual of 1 is the load itself), central differences,
    line search, SciPy ``lm`` as the fallback. Converged ONLY when ``||F / f_scale|| <= 1e-10`` (1e-10 of the load):
    ``step_rel=None`` switches off the default step test, which could stop on a tiny step above that residual.

    Start: AxisForge's own heuristic (``ISO16281BallSolver._initial_delta_r`` / ``_initial_delta_a``,
    ``contact_solver.py:664-696``, reused as a formula: they are private): ``delta_r0 = A (1 - cos alpha_0) +
    (Fr / (c_p Z))^(2/3)``, ``delta_a0 = (Fa / (c_p Z))^(2/3)``; except at Fa = 0, where the exact answer for any law
    is known and used: ``delta_a = -A sin(alpha_0)`` (every ball back at alpha = 0, so sum Q_j sin(alpha_j) = 0).

    The tangent radial stiffness ``Kt = dFr / d delta_r`` at fixed Fa is the final Jacobian with delta_a condensed:
    ``Kt = S_rr - S_ra S_ar / S_aa``, ``S = -dF/dx`` (central differences at the solution).

    Returns
    -------
    summary : dict
        ``delta_r_mm``, ``delta_a_mm``, ``delta_r_el_mm`` (delta_r - A (1 - cos alpha_0): the clearance closed is
        taken out, Harris2007_vol1 Eq. (7.10), PDF p. 156; an elastic measure at Fa = 0 only), ``delta_max_mm``
        (approach of the most loaded ball), ``Q_max_N``, ``n_loaded``, ``alpha_max_deg``, ``Kr_N_per_mm`` (secant
        Fr / delta_r, R5-101), ``Kt_N_per_mm`` (tangent), and the Newton summary (``converged``, ``reason``,
        ``n_iter``, ...).
    result : numerical_solvers NewtonResult
        Its ``records()`` is the residual history.
    elements : list of dict
        ``j``, ``phi_local_deg``, ``delta_mm``, ``alpha_deg``, ``Q_N``.
    """
    Vpsi = np.zeros(bearing.Z)                       # untilted rings

    def state(x):
        delta_j, alpha_j, ca, sa, cp_j, _ = ISO16281BallSolver.elements(bearing, x[0], x[1], Vpsi)
        return delta_j, alpha_j, ca, sa, cp_j, law(delta_j)

    def residual(x):
        _, _, ca, sa, cp_j, Q = state(x)
        return np.array([Fr_N - np.sum(Q * ca * cp_j), Fa_N - np.sum(Q * sa)])

    # radial travel before the load-line ball touches; = s/2 for the deep groove (alpha_0 = arccos(1 - s / (2A)))
    gap_mm = bearing.A * (1.0 - math.cos(bearing.alpha_0))
    # Fa = 0: exact (alpha_j = 0 for every ball); Fa > 0: AxisForge's Hertz-size guess, the line search absorbs it
    x0 = [gap_mm + (Fr_N / (bearing.cp * bearing.Z)) ** (2.0 / 3.0),
          -bearing.A * math.sin(bearing.alpha_0) if Fa_N == 0.0 else (Fa_N / (bearing.cp * bearing.Z)) ** (2.0 / 3.0)]
    xs = np.array([1e-2, 1e-1])     # [mm] delta_r ~ 0.01-0.04, delta_a ~ 0.05-0.14: unit steps of the scaled problem
    options = NewtonOptions(x_scale=xs, f_scale=[Fr_N, max(Fa_N, Fr_N)], jacobian_fd="central",
                            globalisation="line_search", fallback="lm", residual_abs=1e-10, step_rel=None,
                            max_iter=50)
    res = newton(residual, x0, options=options, name=name, label=label)
    # dS/dx [N/mm] of the load the balls carry, S = applied - F (so the minus sign)
    S = -_fd_jacobian(residual, res.x, residual(res.x), xs, "central", None)
    Kt = S[0, 0] - S[0, 1] * S[1, 0] / S[1, 1]      # Fa held: dS_a = 0 eliminates d delta_a
    delta_j, alpha_j, _, _, _, Q = state(res.x)
    j_max = int(np.argmax(Q))
    summary = {"delta_r_mm": float(res.x[0]), "delta_a_mm": float(res.x[1]), "delta_r_el_mm": float(res.x[0]) - gap_mm,
               "delta_max_mm": float(delta_j[j_max]), "Q_max_N": float(Q.max()), "n_loaded": int(np.sum(Q > 0.0)),
               "alpha_max_deg": math.degrees(alpha_j[j_max]), "Kr_N_per_mm": Fr_N / res.x[0],
               "Kt_N_per_mm": float(Kt), **res.record()}
    elements = [{"j": j, "phi_local_deg": math.degrees(bearing.phi_j[j]), "delta_mm": float(delta_j[j]),
                 "alpha_deg": math.degrees(alpha_j[j]), "Q_N": float(Q[j])} for j in range(bearing.Z)]
    return summary, res, elements
