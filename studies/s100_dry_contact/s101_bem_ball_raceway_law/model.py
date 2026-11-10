"""BEM normal law Q(delta) of one ball-raceway contact, the model s102 imports through ``s101.__init__``.

One contact of the bridge (ball and groove with their true profiles, both elastic half-spaces) is pressed to a
given rigid approach ``delta`` (SlipPY BCCG under interference control, spatial kernel, ``slippy_bridge``
``solve``); the load it carries is the BEM law at that approach. Hertz (the bridge's ``hertz``, judged against
r02 by v105) is the same contact with the bodies replaced by their osculating paraboloids:
``Q_H(delta) = Q_ref (delta / delta_H(Q_ref))^(3/2)``, which holds for an elliptical contact at any reference load
because its eccentricity does not depend on the load (Johnson 1985, Sec. 4.2, Eq. (4.31), PDF p. 106). Both are
half-space models, so the difference is the GEOMETRY error of the paraboloid, growing with the size of the contact
against the groove, ``a / r_g`` (MG-08), not "BEM more accurate"; at a/R_ball ~ 0.4 the half-space error itself is
of the same order (Johnson 1985 Sec. 4.1, PDF p. 101): a finite-body judge is needed for a physical claim.

Grid (v104 rule): ``Grid.around(contact, hertz at Q_H(delta), h = b_H / b_over_h, span 2 x 2)``, minor semi-axis
along rolling (grid x). The exact gap grows faster than the paraboloid gap (quartic term
``y^4 (1/R^3 - 1/r_g^3) / 8 > 0`` for ``R < r_g``: the sphere falls away faster than the groove rises), so the
true contact is shorter than the Hertz ellipse and the grid has margin.

Units: SI (m, N, Pa), the bridge's.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from slippy_bridge.contact import Grid, hertz, solve

__all__ = ["BemLaw", "solve_at_approach", "fit_power_law", "fit_law", "bem_law"]


def solve_at_approach(contact, delta_m: float, results_dir, *, b_over_h: float = 16.0, span: float = 2.0,
                      h_m: float | None = None):
    """One BEM solve of ``contact`` at the rigid approach ``delta_m`` [m] (interference control, spatial kernel).

    The grid is sized on the Hertz ellipse of the load that gives ``delta_m`` in Hertz (the bridge's
    ``hertz`` at ``Q_H(delta)``); asked ``b/h = b_over_h`` (the grid's actual b/h is in [asked, 2 x asked),
    h a power of two in metres), half-length ``span`` semi-axes on both axes. ``h_m`` [m], a power of two, sets
    the spacing directly (grid-convergence levels).

    Returns
    -------
    result : slippy_bridge StaticResult
        ``load_found`` [N], ``a``, ``b`` [m], ``p0`` [Pa], ``converged``, ``edge_clear``; ``result.contact`` is the
        contact at ``Q_H(delta)`` (its ``load`` is the Hertz load at this approach).
    hertz_at_Q_H : bridge HertzResult
        The Hertz contact at ``Q_H(delta)`` (its ``delta`` equals ``delta_m``).
    seconds : float
        Wall time of the solve [s].
    """
    hertz_ref = hertz(contact)
    Q_H = contact.load * (delta_m / hertz_ref.delta) ** 1.5     # Hertz load at this approach (Johnson Eq. (4.31))
    contact_at_Q_H = replace(contact, load=Q_H)
    hertz_at_Q_H = hertz(contact_at_Q_H)
    grid = Grid.around(contact_at_Q_H, hertz_at_Q_H, h=h_m if h_m is not None else hertz_at_Q_H.b / b_over_h,
                       span_x=span, span_y=span)
    t = time.perf_counter()
    result = solve(contact_at_Q_H, grid, Path(results_dir) / "slippy", hertz_result=hertz_at_Q_H,
                   use_frequency_domain=False, interference=delta_m)
    return result, hertz_at_Q_H, time.perf_counter() - t


def fit_power_law(delta_m, Q_N) -> tuple[float, float]:
    """Least-squares fit of ``Q = K delta^n`` in log-log [K in N/m^n, n -]: a DESCRIPTOR of the table (Hertz gives
    n = 3/2 exactly); not the law (it misfits the table by up to 0.6 %: physics review of s101)."""
    n, lnK = np.polyfit(np.log(np.asarray(delta_m, dtype=float)), np.log(np.asarray(Q_N, dtype=float)), 1)
    return float(np.exp(lnK)), float(n)


def fit_law(delta_m, Q_N, K_hertz_N_per_m1_5: float) -> tuple[float, float]:
    """Least-squares ``c`` [1/m] of ``Q / Q_H - 1 = c delta`` (through the origin) and the rms residual [-].

    Why linear in delta: the departure is the quartic term of the gap, of relative size (a_H / r_g)^2, and for
    Hertz a_H^2 is proportional to delta at fixed ellipse shape; s101 measured slope 1.985 of |Q error| against
    a/r_g on log axes (2 expected).
    """
    d = np.asarray(delta_m, dtype=float)
    e = np.asarray(Q_N, dtype=float) / (K_hertz_N_per_m1_5 * d ** 1.5) - 1.0
    c = float(np.dot(d, e) / np.dot(d, d))
    return c, float(np.sqrt(np.mean((e - c * d) ** 2)))


@dataclass(frozen=True, kw_only=True)
class BemLaw:
    """The BEM normal law of one contact (or of a ball, two contacts in series), smooth everywhere.

    ``Q(delta) = K_H delta^1.5 (1 + c delta)``: the Hertz law of the same bodies times a ratio linear in the
    approach (:func:`fit_law`). Smooth, without kinks at table nodes (``dQ/ddelta`` continuous for Newton), equal
    to Hertz as delta -> 0, monotonic while ``delta < -0.6 / c`` (36 delta_H(Q_ref) here). Fitted on the table
    (``delta_m`` up to 1.5 delta_H(Q_ref)); beyond it the linear ratio is an extrapolation of the measured trend.

    Attributes
    ----------
    name : str
        The contact (e.g. "6210 inner") or the ball.
    delta_m, Q_N : tuple of float
        The table: approaches [m] and the loads the BEM found [N], accepted rows only (converged, edge clear).
    K_hertz_N_per_m1_5 : float
        Hertz constant of the same contact(s), ``Q_H / delta^1.5`` [N/m^1.5].
    c_per_m : float
        Ratio slope [1/m] (negative: the BEM contact is softer).
    fit_rms : float
        Rms of ``Q / Q_law - 1`` over the table [-].
    """

    name: str
    delta_m: tuple
    Q_N: tuple
    K_hertz_N_per_m1_5: float
    c_per_m: float
    fit_rms: float

    def Q(self, delta_m) -> np.ndarray:
        """Load [N] at approaches ``delta_m`` [m]; 0 for ``delta <= 0`` (a contact carries no tension)."""
        d = np.maximum(np.asarray(delta_m, dtype=float), 0.0)
        return self.K_hertz_N_per_m1_5 * d ** 1.5 * (1.0 + self.c_per_m * d)

    def delta(self, Q_N) -> np.ndarray:
        """Approach [m] at loads ``Q_N`` [N] (the inverse of :meth:`Q`): fixed point
        ``delta = (Q / (K_H (1 + c delta)))^(2/3)`` from the Hertz approach; |c delta| << 1 makes it a contraction
        (30 steps reach round-off)."""
        Q = np.maximum(np.asarray(Q_N, dtype=float), 0.0)
        d = (Q / self.K_hertz_N_per_m1_5) ** (2.0 / 3.0)
        for _ in range(30):
            d = (Q / (self.K_hertz_N_per_m1_5 * (1.0 + self.c_per_m * d))) ** (2.0 / 3.0)
        return d


def bem_law(name: str, contact, deltas_m, results_dir, *, b_over_h: float = 16.0, span: float = 2.0):
    """Solve ``contact`` at every approach of ``deltas_m`` [m]; return its :class:`BemLaw` and the solves.

    Returns
    -------
    law : BemLaw
        Fitted on the converged, edge-clear solves.
    solves : list of (delta_m, StaticResult, HertzResult, seconds)
        Every solve, for the records.
    """
    solves = [(d, *solve_at_approach(contact, d, results_dir, b_over_h=b_over_h, span=span)) for d in deltas_m]
    delta_m, Q_N = (np.array(v) for v in zip(*[(d, r.load_found) for d, r, _, _ in solves
                                                if r.converged and r.edge_clear]))
    hertz_ref = hertz(contact)
    K_H = contact.load / hertz_ref.delta ** 1.5
    c, rms = fit_law(delta_m, Q_N, K_H)
    return BemLaw(name=name, delta_m=tuple(delta_m), Q_N=tuple(Q_N), K_hertz_N_per_m1_5=K_H, c_per_m=c,
                  fit_rms=rms), solves
