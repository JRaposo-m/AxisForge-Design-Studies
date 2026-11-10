"""Static load zone of a single-row rolling bearing with rigid rings (r03).

Harris and Kotzalas (2007, vol. 1, ch. 7): every rolling element is a nonlinear spring ``Q = K_n delta^n``
between two rigid rings; a relative ring displacement (radial ``delta_r``, axial ``delta_a``) compresses
the elements on one side, the compression falls off with the azimuth ``psi`` and the elements beyond the
limiting angle ``psi_l`` are unloaded. The sum of the element forces balances the applied load. In the
continuous limit (many elements) the sums become the Sjovall integrals ``J_r(eps)``, ``J_a(eps)``
(Harris Tables 7.1, 7.4). The element spring constant ``K_n`` is an input: the Hertz series of the two
contact constants ``K_i``, ``K_o`` (:func:`ball_deflection`), which the caller takes from Hertz (r02 in a v
case). README sections 2-4.

An independent judge (D-0036 amendment rule 13): NumPy, SciPy and the standard library only, nothing from
``studies/``; of the library functions rule 13 allows (``R_VERIFIED``: SlipPY ``hertz_full``) it needs none.

Conventions:

* SI throughout: m, N, Pa; ``K_n`` in N/m^n.
* ``psi`` [rad] is the azimuth of an element measured from the direction of the radial ring displacement
  ``delta_r`` (Harris Fig. 7.1, Fig. 7.13: ``psi = 0`` at the most heavily loaded element). How this maps on
  the azimuth of AxisForge is the v case's business (convention-dependent, R-003; OPEN R5-08).
* ``delta_r >= 0`` is the radial displacement of one ring relative to the other along ``psi = 0``;
  ``delta_a`` the axial one, positive when it loads the contacts of contact angle ``alpha > 0``.
* ``clearance_m`` is the diametral clearance ``P_d`` (Harris Eq. 7.10): positive = clearance, negative =
  interference (preload, Harris Fig. 7.3c).
* The contact angle is the same for every element and does not change under load (Harris Sec. 7.5.1: "the
  contact angle alpha is assumed identical for all loaded balls or rollers").
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import beta

__all__ = ["N_POINT", "N_LINE", "sjovall_integrals", "stribeck_constant", "ball_deflection",
           "ring_load_distribution", "solve_ring_displacement"]

N_POINT = 1.5   # load-deflection exponent of a point (ball) contact, Q ~ delta^(3/2) (Harris Eq. 7.2)
# OPEN R5-01: Harris Eq. 7.4 states n = 10/9 for a line contact, but Tables 7.1 and 7.4 are computed
# with n = 1.1 (every printed line value is reproduced with 1.1, several miss with 10/9; Eq. 7.62 also
# prints n = 1.11, PDF p. 170, which misses like 10/9). r03 takes 1.1 so that its line integrals are those printed.
N_LINE = 1.1


def _check_n(n: float) -> None:
    if not (math.isfinite(n) and n >= 1.0):
        # n >= 1: Hertz gives 3/2 (point) and about 10/9 (line); n < 1 would be a softening contact
        raise ValueError(f"load-deflection exponent n must be finite and >= 1, got {n}")


def sjovall_integrals(eps: float, *, n: float = N_POINT) -> dict:
    """Sjovall radial and axial load-distribution integrals of a single-row bearing.

    With the element compression ``delta_psi = delta_max [1 - (1 - cos psi) / (2 eps)]`` (Harris Eq. 7.11,
    7.60) and ``Q = K_n delta^n``, the element load relative to the most loaded one is
    ``Q_psi / Q_max = [1 - (1 - cos psi) / (2 eps)]^n`` (Eq. 7.15, 7.62). Spreading the ``Z`` elements
    evenly around the circle (continuous limit) turns the force sums into

    ``J_r(eps) = 1/(2 pi) int_{-psi_l}^{psi_l} [1 - (1 - cos psi)/(2 eps)]^n cos psi dpsi`` (Eq. 7.20, 7.67)
    ``J_a(eps) = 1/(2 pi) int_{-psi_l}^{psi_l} [1 - (1 - cos psi)/(2 eps)]^n dpsi``          (Eq. 7.69)

    so that ``F_r = Z Q_max J_r cos alpha`` (Eq. 7.66; Eq. 7.19 with alpha = 0) and
    ``F_a = Z Q_max J_a sin alpha`` (Eq. 7.68). ``eps`` is the projection of the loaded arc on the
    diameter, as a fraction of it: ``eps = 0.5`` is half the circle loaded (zero clearance, radial load),
    ``eps >= 1`` all elements loaded.

    Parameters
    ----------
    eps : float
        Load distribution factor [-], ``>= 0``; ``math.inf`` is a pure axial load (all elements equal).
    n : float
        Load-deflection exponent [-]: :data:`N_POINT` (1.5) for balls, :data:`N_LINE` (1.1) for rollers.

    Returns
    -------
    dict
        ``J_r``, ``J_a`` [-]; ``psi_l_rad``: limiting angle of the load zone, ``arccos(1 - 2 eps)`` for
        ``eps < 1``, ``pi`` above; ``Fr_tan_alpha_over_Fa``: ``J_r / J_a``, the abscissa of Harris
        Table 7.4 (the ratio of the applied loads that produces this ``eps`` at zero clearance).

    Notes
    -----
    ``eps = 0`` returns the continuous limit ``J_r = J_a = 0`` with the ratio 1 (its limit). The printed
    "1/Z" of Tables 7.1 and 7.4 is the discrete one-element limit (only the element at ``psi = 0`` is
    loaded), reached through :func:`ring_load_distribution` (OPEN R5-02).

    References
    ----------
    Harris2007_vol1 Eqs 7.11-7.20 (PDF p. 156-158), Table 7.1 (PDF p. 158); Eqs 7.58-7.69, Table 7.4
    (PDF p. 169-171); Sjovall (1933) through Harris.
    """
    _check_n(n)
    if not eps >= 0.0:                       # also rejects nan
        raise ValueError(f"eps must be >= 0, got {eps}")
    if eps == 0.0:
        return {"J_r": 0.0, "J_a": 0.0, "psi_l_rad": 0.0, "Fr_tan_alpha_over_Fa": 1.0}
    if math.isinf(eps):
        # every element carries Q_max: J_a = 1, the radial components cancel round the circle
        return {"J_r": 0.0, "J_a": 1.0, "psi_l_rad": math.pi, "Fr_tan_alpha_over_Fa": 0.0}
    # the bracket vanishes at cos psi_l = 1 - 2 eps; for eps >= 1 it is positive all round
    psi_l = math.acos(1.0 - 2.0 * eps) if eps < 1.0 else math.pi

    # The integrands are even in psi: (1/2 pi) int_{-psi_l}^{psi_l} = (1/pi) int_0^{psi_l}. Two exact rewritings
    # make them friendly to quadrature (same integrals, not an approximation):
    # 1. J_r by parts (boundary terms vanish: sin 0 = 0 and the bracket is 0 at psi_l or sin pi = 0):
    #    J_r = n/(2 pi eps) int_0^{psi_l} [.]^(n-1) sin^2 psi dpsi, a positive integrand; Eq. 7.67 as written
    #    cancels its cos psi > 0 and < 0 lobes in a wide zone (eps = 1e6 leaves J_r ~ 4e-7).
    # 2. u = (1 - cos psi)/(2 eps), so [.] = 1 - u and sin psi = 2 sqrt(eps u (1 - eps u)): the end
    #    singularities (1 - u)^(n-1) at the zone edge, u^(-1/2) at psi = 0 become Jacobi weights that QUADPACK
    #    integrates exactly ('alg'), leaving a smooth function.
    opts = {"epsabs": 0.0, "epsrel": 1e-12, "limit": 200}    # far below the four printed digits
    if eps < 1.0:     # partial zone: u in [0, 1], the bracket vanishes at u = 1 (psi = psi_l)
        J_a = quad(lambda u: math.sqrt(eps / (1.0 - eps * u)), 0.0, 1.0, weight="alg", wvar=(-0.5, n), **opts)[0]
        J_r = 2.0 * n * quad(lambda u: math.sqrt(eps * (1.0 - eps * u)), 0.0, 1.0, weight="alg",
                             wvar=(0.5, n - 1.0), **opts)[0]
    elif eps == 1.0:  # the zone closes at psi = pi: both ends coincide, a Beta function
        J_a, J_r = beta(0.5, n + 0.5), 2.0 * n * beta(1.5, n + 0.5)
    else:             # full ring: u in [0, 1/eps], the bracket stays >= 1 - 1/eps > 0
        J_a = quad(lambda u: (1.0 - u) ** n, 0.0, 1.0 / eps, weight="alg", wvar=(-0.5, -0.5), **opts)[0]
        J_r = 2.0 * n * eps * quad(lambda u: (1.0 - u) ** (n - 1.0), 0.0, 1.0 / eps, weight="alg",
                                   wvar=(0.5, 0.5), **opts)[0]
    J_a, J_r = J_a / math.pi, J_r / math.pi
    # a zone too narrow for floating point (eps ~ 1e-300) has the one-element ratio, its limit
    return {"J_r": J_r, "J_a": J_a, "psi_l_rad": psi_l, "Fr_tan_alpha_over_Fa": J_r / J_a if J_a > 0.0 else 1.0}


def stribeck_constant(*, n: float = N_POINT) -> float:
    """Stribeck's constant ``Q_max Z cos(alpha) / F_r`` for a radial load at zero clearance.

    At zero clearance a radial load loads half the circle (``eps = 0.5``, Harris Fig. 7.3a), so
    ``Q_max = F_r / (Z J_r(0.5) cos alpha)`` and the constant is ``1 / J_r(0.5)``. Harris prints 4.37 for
    balls (Eq. 7.23, after Stribeck) and 4.08 for rollers (Eq. 7.25); the value 5 of Eq. 7.24 is a
    conservative approximation for a nominal clearance, not this constant.

    Parameters
    ----------
    n : float
        Load-deflection exponent [-] (1.5 balls, 1.1 rollers).

    Returns
    -------
    float
        ``1 / J_r(0.5)`` [-] in the continuous limit (``Z -> inf``). A finite ``Z`` changes the constant
        slightly (the discrete sum of Eq. 7.17, and where the elements sit): do not compare a ``Z = 9``
        ``Q_max`` with 4.37 at 1e-4.

    References
    ----------
    Harris2007_vol1 Eqs 7.23-7.25 (PDF p. 159).
    """
    return 1.0 / sjovall_integrals(0.5, n=n)["J_r"]


def ball_deflection(Q_N, *, K_i: float, K_o: float, n: float = N_POINT, clearance_m: float = 0.0) -> dict:
    """Total elastic approach of one rolling element between its two raceways, and the Hertz series constant.

    The element is squeezed at the inner and at the outer contact by the same normal load ``Q``; the two
    approaches add (Harris Eq. 7.5, ``delta_n = delta_i + delta_o``), so the element acts as one spring
    ``Q = K_n delta_n^n`` with ``K_n = [1 / ((1/K_i)^(1/n) + (1/K_o)^(1/n))]^n`` (Eq. 7.6, 7.7).

    Parameters
    ----------
    Q_N : float or array_like
        Normal element load [N], ``>= 0``.
    K_i, K_o : float
        Load-deflection constants of the inner and the outer contact [N/m^n], > 0, ``Q = K delta^n`` of
        each contact alone (a ball: from Hertz, r02 ``hertz(...).delta_m`` in a v case,
        ``K = Q / delta^1.5``; a roller: the caller's, OPEN R5-07).
    n : float
        Load-deflection exponent [-].
    clearance_m : float
        Diametral clearance ``P_d`` [m] (negative = interference), for ``delta_r_m`` only.

    Returns
    -------
    dict
        ``delta_i_m``, ``delta_o_m``, ``delta_n_m`` [m] (same shape as ``Q_N``); ``K_n_N_per_m_pow_n``
        [N/m^n]; ``delta_r_m`` [m]: the radial ring shift of a radial bearing (contact angle 0) for which
        ``Q_N`` is the load of the most loaded element, ``delta_r = delta_n + P_d/2`` (Eq. 7.21 inverted).
        A magnitude: the direction is that of the load. Contact angle 0 only: at ``alpha > 0`` the shift
        depends on ``delta_a`` as well; use :func:`solve_ring_displacement`.

    References
    ----------
    Harris2007_vol1 Eqs 7.4-7.7 (PDF p. 156), Eq. 7.21 (PDF p. 158).
    """
    _check_n(n)
    if not (K_i > 0.0 and K_o > 0.0):
        raise ValueError(f"contact constants must be > 0, got K_i = {K_i}, K_o = {K_o}")
    Q = np.asarray(Q_N, dtype=float)
    if np.any(~(Q >= 0.0)):
        raise ValueError("element loads must be >= 0")
    delta_i, delta_o = (Q / K_i) ** (1.0 / n), (Q / K_o) ** (1.0 / n)
    delta_n = delta_i + delta_o                                          # Eq. 7.5: approaches in series
    K_n = (1.0 / ((1.0 / K_i) ** (1.0 / n) + (1.0 / K_o) ** (1.0 / n))) ** n   # Eq. 7.6
    # psi = 0 element of a radial bearing: delta_max = delta_r - P_d/2 (Eq. 7.10 at psi = 0)
    return {"delta_i_m": delta_i, "delta_o_m": delta_o, "delta_n_m": delta_n, "K_n_N_per_m_pow_n": K_n,
            "delta_r_m": delta_n + clearance_m / 2.0}


def ring_load_distribution(delta_r_m: float, delta_a_m: float = 0.0, *, psi_rad, K_n: float,
                           n: float = N_POINT, alpha_rad: float = 0.0, clearance_m: float = 0.0) -> dict:
    """Element loads of a single-row bearing for a given rigid ring displacement.

    Each element at azimuth ``psi`` is compressed by the projection of the ring displacement on its
    contact normal, less half the diametral clearance:
    ``delta_psi = delta_a sin alpha + delta_r cos alpha cos psi - P_d/2``, and carries
    ``Q_psi = K_n max(delta_psi, 0)^n``. This joins Harris Eq. 7.10 (``alpha = 0``, with clearance) and
    Eq. 7.58 (combined, zero clearance); with both ``alpha != 0`` and ``P_d != 0`` it is r03's own
    combination (fixed contact angle, clearance along the normal; OPEN R5-06). It is identically
    ``delta_psi = delta_max [1 - (1 - cos psi)/(2 eps)]`` with
    ``delta_max = delta_a sin alpha + delta_r cos alpha - P_d/2`` and ``eps = delta_max / (2 delta_r cos alpha)``,
    which is Eq. 7.12 at ``alpha = 0`` and Eq. 7.61 at ``P_d = 0``.

    Parameters
    ----------
    delta_r_m : float
        Radial ring displacement [m], ``>= 0``, along ``psi = 0``.
    delta_a_m : float
        Axial ring displacement [m]; irrelevant at ``alpha = 0``.
    psi_rad : array_like
        Azimuths of the ``Z`` elements [rad], from the direction of ``delta_r`` (OPEN R5-08, R-003). For
        ``Z`` equally spaced elements: ``psi0 + 2 pi j / Z``.
    K_n : float
        Element load-deflection constant [N/m^n] (Hertz series of both contacts, :func:`ball_deflection`).
    n : float
        Load-deflection exponent [-].
    alpha_rad : float
        Contact angle [rad], ``0 <= alpha < pi/2``, the same for every element and fixed under load.
    clearance_m : float
        Diametral clearance ``P_d`` [m]; negative = interference.

    Returns
    -------
    dict
        Per element (arrays in the order of ``psi_rad``): ``psi_rad``, ``delta_psi_m`` (negative = gap),
        ``Q_N``. Scalars: ``delta_max_m``, ``Q_max_N`` (at ``psi = 0``, whether an element sits there or
        not), ``epsilon`` (``inf`` for ``delta_r = 0`` with a load; ``<= 0``: no element loaded),
        ``psi_l_rad``, ``J_r``, ``J_a``; ``Fr_N``, ``Fa_N``: the sums over the elements (Eqs 7.16,
        7.63-7.64); ``Fr_continuous_N``, ``Fa_continuous_N``: ``Z Q_max J cos/sin alpha`` (Eqs 7.66, 7.68).
        Forces are those the elements exert on the displaced ring, positive against ``delta_r``, ``delta_a``.

    References
    ----------
    Harris2007_vol1 Eqs 7.10-7.22 (PDF p. 156-158), 7.58-7.69 (PDF p. 169-171).
    """
    _check_n(n)
    if not (K_n > 0.0 and math.isfinite(K_n)):
        raise ValueError(f"K_n must be finite and > 0, got {K_n}")
    if not 0.0 <= alpha_rad < math.pi / 2.0:
        raise ValueError(f"contact angle must be in [0, pi/2), got {alpha_rad}")
    if not delta_r_m >= 0.0:
        raise ValueError(f"delta_r_m must be >= 0 (psi = 0 is its direction), got {delta_r_m}")
    psi = np.atleast_1d(np.asarray(psi_rad, dtype=float))
    if psi.ndim != 1 or psi.size == 0:
        raise ValueError("psi_rad must be a non-empty 1-D sequence of azimuths")
    s, c = math.sin(alpha_rad), math.cos(alpha_rad)
    delta_psi = delta_a_m * s + delta_r_m * c * np.cos(psi) - clearance_m / 2.0
    Q = K_n * np.clip(delta_psi, 0.0, None) ** n                     # an open contact carries nothing
    delta_max = delta_a_m * s + delta_r_m * c - clearance_m / 2.0
    radial = delta_r_m * c
    if delta_max <= 0.0:
        eps = delta_max / (2.0 * radial) if radial > 0.0 else 0.0   # no element touches
    else:
        eps = delta_max / (2.0 * radial) if radial > 0.0 else math.inf
    J = sjovall_integrals(max(eps, 0.0), n=n)
    Q_max = K_n * max(delta_max, 0.0) ** n
    Z = psi.size
    # each normal load Q acts along the contact line at alpha: radial part Q cos(a) cos(psi), axial Q sin(a).
    # OPEN R5-05: Eq. 7.70 prints Q_max = F_r / (J_r Z sin a); Eq. 7.66 (and this) gives cos a
    return {"psi_rad": psi, "delta_psi_m": delta_psi, "Q_N": Q,
            "delta_max_m": delta_max, "Q_max_N": Q_max, "epsilon": eps, "psi_l_rad": J["psi_l_rad"],
            "J_r": J["J_r"], "J_a": J["J_a"],
            "Fr_N": float(np.sum(Q * np.cos(psi))) * c, "Fa_N": float(np.sum(Q)) * s,
            "Fr_continuous_N": Z * Q_max * J["J_r"] * c, "Fa_continuous_N": Z * Q_max * J["J_a"] * s}


def solve_ring_displacement(Fr_N: float, Fa_N: float = 0.0, *, psi_rad, K_n: float, n: float = N_POINT,
                            alpha_rad: float = 0.0, clearance_m: float = 0.0, continuous: bool = False) -> dict:
    """Ring displacement ``(delta_r, delta_a)`` that balances an applied radial and axial load.

    Static equilibrium of the displaced ring: the element forces of :func:`ring_load_distribution` equal
    the applied ``F_r`` and ``F_a`` (Harris Eqs 7.16, 7.63-7.64). Harris solves it by trial and error
    (Eq. 7.22); this judge brackets and solves it with ``scipy.optimize.brentq``: at ``alpha = 0`` one
    unknown ``delta_r``; otherwise ``delta_a`` is solved for the axial balance inside each evaluation of the
    radial balance (``F_a`` grows monotonically with ``delta_a``, ``F_r`` with ``delta_r`` at fixed ``F_a``).

    Parameters
    ----------
    Fr_N : float
        Applied radial load [N], ``> 0``, along ``psi = 0``.
    Fa_N : float
        Applied axial load [N]; must be 0 at ``alpha = 0`` and ``> 0`` otherwise (a single row with
        ``alpha > 0`` carries no radial load without a thrust, Harris Sec. 7.5.1).
    psi_rad, K_n, n, alpha_rad, clearance_m
        As :func:`ring_load_distribution`.
    continuous : bool
        False: balance the sums over the elements at ``psi_rad`` (element by element, comparable with a
        discrete model). True: balance ``Z Q_max J_r cos alpha`` and ``Z Q_max J_a sin alpha``, the
        continuous form of Harris's tables (``Z = len(psi_rad)``; the azimuths then only report ``Q_N``).

    Returns
    -------
    dict
        :func:`ring_load_distribution` at the solution plus ``delta_r_m``, ``delta_a_m`` [m],
        ``converged`` (bool), ``n_radial_evaluations``, ``n_axial_iterations`` (total inner brentq
        iterations), ``residual_Fr_N``, ``residual_Fa_N`` [N] and ``continuous``.

    Raises
    ------
    ValueError
        Bad inputs, or a load no single row can carry: ``F_r tan(alpha) / F_a`` at or beyond the
        one-element limit (about 1, Harris Table 7.4 at ``eps = 0``).

    References
    ----------
    Harris2007_vol1 Eqs 7.16-7.22 (PDF p. 158), 7.63-7.71 (PDF p. 170-171).
    """
    if not (Fr_N > 0.0 and math.isfinite(Fr_N)):
        raise ValueError(f"Fr_N must be finite and > 0, got {Fr_N}")
    if not (0.0 <= alpha_rad < math.pi / 2.0):
        raise ValueError(f"contact angle must be in [0, pi/2), got {alpha_rad}")
    if alpha_rad == 0.0 and Fa_N != 0.0:
        raise ValueError("a zero contact angle carries no axial load: Fa_N must be 0")
    if alpha_rad > 0.0 and not (Fa_N > 0.0 and math.isfinite(Fa_N)):
        raise ValueError(f"a single row with alpha > 0 needs Fa_N > 0, got {Fa_N}")
    kr, ka = ("Fr_continuous_N", "Fa_continuous_N") if continuous else ("Fr_N", "Fa_N")
    Z = len(np.atleast_1d(psi_rad))
    s, c = math.sin(alpha_rad), math.cos(alpha_rad)
    # displacement scale: the uniform compression that would carry the larger load on all Z elements
    scale = (max(Fr_N, Fa_N) / (Z * K_n)) ** (1.0 / n) + abs(clearance_m)
    counts = {"radial": 0, "axial": 0}

    def state(dr, da):
        return ring_load_distribution(dr, da, psi_rad=psi_rad, K_n=K_n, n=n, alpha_rad=alpha_rad,
                                      clearance_m=clearance_m)

    def axial_for(dr):
        """delta_a balancing Fa_N at the radial displacement dr (alpha > 0)."""
        if alpha_rad == 0.0:
            return 0.0
        # lo: every contact open (delta_max = 0, nothing carried); hi: every contact closed by at least
        # the uniform compression that alone would carry Fa_N on all Z elements
        lo = (clearance_m / 2.0 - dr * c) / s
        hi = (clearance_m / 2.0 + dr * c + (Fa_N / (Z * K_n * s)) ** (1.0 / n)) / s
        da, info = brentq(lambda x: state(dr, x)[ka] - Fa_N, lo, hi, xtol=1e-14 * scale,
                          rtol=4.0 * np.finfo(float).eps, full_output=True)
        counts["axial"] += info.iterations
        return da

    def radial_residual(dr):
        counts["radial"] += 1
        return state(dr, axial_for(dr))[kr] - Fr_N

    # lower bracket with no radial force: at alpha = 0 the last displacement with every contact still open
    # (or 0 under interference, where the preload is symmetric); at alpha > 0 the centred ring (pure thrust)
    lo = max(clearance_m / 2.0, 0.0) if alpha_rad == 0.0 else 0.0
    if radial_residual(lo) > 0.0:
        raise ValueError("the load at the lower bracket already exceeds Fr_N (azimuths not balanced?)")
    hi = lo + scale
    for _ in range(80):                          # 2^80 x scale: far past any elastic displacement
        if radial_residual(hi) > 0.0:
            break
        hi = lo + 2.0 * (hi - lo)
    else:
        raise ValueError("no displacement carries Fr_N: Fr tan(alpha)/Fa beyond the single-row limit")
    dr, info = brentq(radial_residual, lo, hi, xtol=1e-14 * scale, rtol=4.0 * np.finfo(float).eps,
                      full_output=True)
    da = axial_for(dr)
    result = state(dr, da)
    # brentq always ends on a sign change; past the single-row limit (F_r tan(a)/F_a above the one-element
    # value) F_r(delta_r) only creeps towards F_r_N from below and the "sign change" is roundoff at a huge
    # delta_r. A balance worse than 1e-6 of the load (the solve reaches ~1e-15) is that case: refuse it
    if max(abs(result[kr] - Fr_N), abs(result[ka] - Fa_N)) > 1e-6 * max(Fr_N, Fa_N):
        raise ValueError("no displacement carries Fr_N: Fr tan(alpha)/Fa beyond the single-row limit "
                         f"(residuals {result[kr] - Fr_N:.3g} N, {result[ka] - Fa_N:.3g} N)")
    result.update({"delta_r_m": dr, "delta_a_m": da, "converged": bool(info.converged),
                   "n_radial_evaluations": counts["radial"], "n_axial_iterations": counts["axial"],
                   "residual_Fr_N": result[kr] - Fr_N, "residual_Fa_N": result[ka] - Fa_N,
                   "continuous": continuous})
    return result
