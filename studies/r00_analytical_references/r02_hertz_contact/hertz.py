"""Hertz point and line contact, written from the theory: the independent judge (r02).

It judges AxisForge (``PointContactStiffness``, ISO/TS 16281), SlipPY (``hertz_full``) and the
bridge (``slippy_bridge.contact.solvers.hertz``), so it shares no code with any of them: only
SciPy and the standard library are imported (D-0036 amendment ruling 8, SQ-05). Sources: Johnson, Contact
Mechanics (1985), ch. 4; Hamrock and Brewe (1983). Equation and page numbers: README section 2.

Axes and signs (README section 3): ``x`` is the rolling direction, ``y`` lies across it. A radius
is positive for a convex surface, negative for a concave one, ``math.inf`` for a flat. The
semi-axis ``b`` lies along ``x`` and ``a`` along ``y`` (Hamrock-Brewe: ``D_x = 2 b``,
``D_y = 2 a``); either may be the larger one, unlike Johnson's ``a >= b`` (OPEN MG-05: the bridge's
convention). SI units throughout.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.optimize import brentq
from scipy.special import ellipk, elliprd

__all__ = ["Validity", "HertzPoint", "HertzLine", "hertz_point", "hertz_line", "hamrock_brewe_fit",
           "SIZE_RATIO_MAX", "ALPHA_FIT_MAX"]

SIZE_RATIO_MAX = 0.1    # soft range: semi-axis / radius (P-02 policy; Johnson 1985 Sec. 4.2 only says 'much
                        # smaller than')
ALPHA_FIT_MAX = 100.0   # upper end of the Hamrock-Brewe fits, alpha = R_y / R_x (Tables 1-2)


@dataclass(frozen=True)
class Validity:
    """Soft validity range of a result (P-02 class ii; class iii clamps do not occur in r02).

    Attributes
    ----------
    in_range : bool
        ``ratio`` lies inside the range the formula is valid in.
    governing : str
        The quantity that sets ``ratio``, e.g. ``"a/R1y"``.
    ratio : float
        Its value [-].
    clamp_acted : bool
        A clamp printed in a standard changed the result (always False here).
    """

    in_range: bool
    governing: str
    ratio: float
    clamp_acted: bool = False


@dataclass(frozen=True)
class HertzPoint:
    """Hertz elliptical point contact.

    Attributes
    ----------
    a_m, b_m : float
        Semi-axes of the contact ellipse along ``y`` and ``x`` [m].
    p0_Pa, pm_Pa : float
        Maximum and mean contact pressure [Pa]; ``pm = 2 p0 / 3``.
    delta_m : float
        Approach of distant points of the two bodies (sum of both deformations) [m].
    k_n_N_per_m : float
        Tangent normal stiffness ``dF / d delta = 3 F / (2 delta)`` [N/m].
    E_star_Pa : float
        Contact modulus ``1/E* = (1 - nu1^2)/E1 + (1 - nu2^2)/E2`` [Pa].
    validity : Validity
        Largest semi-axis / radius ratio against ``SIZE_RATIO_MAX``.
    """

    a_m: float
    b_m: float
    p0_Pa: float
    pm_Pa: float
    delta_m: float
    k_n_N_per_m: float
    E_star_Pa: float
    validity: Validity


@dataclass(frozen=True)
class HertzLine:
    """Hertz line contact (plane strain).

    Attributes
    ----------
    b_m : float
        Semi-width of the contact strip along ``x`` [m] (Johnson's ``a``).
    p0_Pa : float
        Maximum contact pressure [Pa].
    w_N_per_m : float
        Load per unit length of the line [N/m], as given.
    validity : Validity
        Largest semi-width / radius ratio against ``SIZE_RATIO_MAX``.
    """

    b_m: float
    p0_Pa: float
    w_N_per_m: float
    validity: Validity


def _curvature(R_m: float, name: str) -> float:
    """``1/R`` [1/m]; a flat (``inf``) gives 0. Zero or NaN radius is a domain error."""
    if not (R_m != 0.0 and not math.isnan(R_m)):
        raise ValueError(f"{name} must be non-zero (inf for a flat), got {R_m}")
    return 1.0 / R_m


def _e_star(E1_Pa: float, nu1: float, E2_Pa: float, nu2: float) -> float:
    """Contact modulus ``1/E* = (1 - nu1^2)/E1 + (1 - nu2^2)/E2`` [Pa]; refuses non-physical E, nu."""
    for name, E in (("E1_Pa", E1_Pa), ("E2_Pa", E2_Pa)):
        if not (0.0 < E < math.inf):
            raise ValueError(f"{name} must be positive and finite, got {E}")
    for name, nu in (("nu1", nu1), ("nu2", nu2)):
        if not (-1.0 < nu < 0.5):
            raise ValueError(f"{name} must lie in (-1, 0.5), got {nu}")
    return 1.0 / ((1.0 - nu1**2) / E1_Pa + (1.0 - nu2**2) / E2_Pa)


def _size_validity(pairs) -> Validity:
    """Largest ``semi-axis / |R|`` over the finite radii; ``pairs`` = ((name, semi-axis, R), ...).

    The pairs hold the relative radius of the pair and each body's own radius, Johnson's conditions
    (i)-(iii) of Sec. 4.2 (OPEN MG-08: in a conforming groove the ball's own radius governs and puts a
    deep-groove contact out of range near 70 N; whether it sets the range or is only reported is José's).
    """
    ratio, name = max((s / abs(R), name) for name, s, R in pairs if math.isfinite(R))
    return Validity(ratio <= SIZE_RATIO_MAX, name, ratio)


def hertz_point(R1x_m, R1y_m, R2x_m, R2y_m, E1_Pa, nu1, E2_Pa, nu2, F_N) -> HertzPoint:
    """Hertz solution of an elliptical point contact between two elastic bodies.

    Two smooth bodies touch at a point and are pressed together by ``F_N``. Near the point each
    surface is a paraboloid, so the gap before loading is ``h = k_x x^2 / 2 + k_y y^2 / 2`` with
    the relative curvatures ``k_x = 1/R1x + 1/R2x`` and ``k_y = 1/R1y + 1/R2y``. The contact is
    an ellipse with semi-axes ``b`` (along ``x``) and ``a`` (along ``y``), and the pressure is
    ``p = p0 sqrt(1 - (x/b)^2 - (y/a)^2)``.

    Parameters
    ----------
    R1x_m, R1y_m, R2x_m, R2y_m : float
        Principal radii of curvature of body 1 and body 2 along ``x`` (rolling) and ``y`` [m].
        Convex > 0, concave < 0, flat ``math.inf``. The principal directions of the two bodies
        coincide; for bodies at an angle give the principal relative radii of the pair as body 1
        and a flat body 2 (Johnson 1985, Sec. 4.1): ``validity`` then sees the relative radii only.
    E1_Pa, E2_Pa : float
        Young's moduli [Pa].
    nu1, nu2 : float
        Poisson's ratios [-].
    F_N : float
        Normal load [N].

    Returns
    -------
    HertzPoint

    Raises
    ------
    ValueError
        Domain errors (P-02 class i): a zero radius, ``E <= 0``, ``nu`` outside (-1, 0.5),
        ``F_N <= 0``, or a relative curvature that is not positive (a line contact, ``k = 0``,
        or a concave body tighter than the convex one, ``k < 0``).

    Notes
    -----
    Assumptions (Hertz): smooth frictionless surfaces, linear elastic isotropic bodies, each
    treated as a half-space, normal load only, no adhesion. The half-space needs contact
    dimensions small against the radii: the ratio is returned in ``validity`` (soft range
    ``a/R <= 0.1``, P-02 class ii), not refused.

    With ``A >= B`` the semi-axes (Johnson's ``a >= b``), ``m = e^2 = 1 - (B/A)^2`` and ``K(m)``,
    ``E(m)`` the complete elliptic integrals, the ellipsoidal pressure displaces the surfaces by
    ``u = (L - M s^2 - N t^2) / (pi E*)``, ``s`` along ``A`` (Johnson 1985, Sec. 3.5(b), Eqs (3.61)-(3.63);
    Sec. 4.2(b), Eqs (4.25)-(4.28))::

        M = pi p0 B (K - E) / (m A^2)
        N = pi p0 B ((A/B)^2 E - K) / (m A^2)
        L = pi p0 B K

    Contact requires ``u = delta - h`` inside the ellipse, hence ``M = pi E* k_min / 2``,
    ``N = pi E* k_max / 2`` and ``delta = L / (pi E*) = p0 B K / E*``. ``N / M = k_max / k_min``
    fixes ``m`` (one scalar root; Hamrock-Brewe Eqs (8)-(12) solve the same equation by
    iteration) and the load ``F = (2/3) pi A B p0`` closes the system.

    ``(K - E) / m`` and ``(E - (1 - m) K) / m`` are 0/0 for a circular contact (``m = 0``), so
    they are evaluated as Carlson's symmetric integral ``R_D`` (DLMF 19.25.1), regular there::

        (K - E) / m                     = R_D(0, 1 - m, 1) / 3
        (E - (1 - m) K) / (m (1 - m))   = R_D(0, 1, 1 - m) / 3

    Both tend to ``pi / 4`` at ``m = 0``: the circular contact is the root ``m = 0`` of the same
    equation and needs no special case.

    Examples
    --------
    A 12.7 mm steel ball on a steel flat, 500 N (circular contact):

    >>> r = hertz_point(6.35e-3, 6.35e-3, math.inf, math.inf, 210e9, 0.3, 210e9, 0.3, 500.0)
    >>> round(r.p0_Pa / 1e9, 3), round(r.b_m * 1e3, 4)
    (3.173, 0.2743)
    """
    k_x = _curvature(R1x_m, "R1x_m") + _curvature(R2x_m, "R2x_m")      # relative curvatures [1/m]
    k_y = _curvature(R1y_m, "R1y_m") + _curvature(R2y_m, "R2y_m")
    e_star = _e_star(E1_Pa, nu1, E2_Pa, nu2)
    if not F_N > 0.0:
        raise ValueError(f"F_N must be positive, got {F_N}")
    k_min, k_max = min(k_x, k_y), max(k_x, k_y)
    if not k_min > 0.0:
        raise ValueError(f"relative curvatures must be positive for a point contact, got {k_x}, {k_y}")

    # m = e^2 from N/M = k_max/k_min in Carlson's form: k_min R_D(0, 1, 1-m) = k_max R_D(0, 1-m, 1).
    # The left side minus the right is <= 0 at m = 0 (0 for a circle: the root) and grows as
    # 1/(1 - m) towards m = 1, where K is infinite: the bracket stops at 1 - 1e-12 (A/B up to 1e6).
    m = brentq(lambda m: k_min * elliprd(0.0, 1.0, 1.0 - m) - k_max * elliprd(0.0, 1.0 - m, 1.0),
               0.0, 1.0 - 1e-12, xtol=1e-16)
    K = float(ellipk(m))
    D = float(elliprd(0.0, 1.0 - m, 1.0)) / 3.0                          # (K - E) / m

    A = (3.0 * F_N * D / (math.pi * k_min * e_star)) ** (1.0 / 3.0)    # major semi-axis, from M and F
    B = A * math.sqrt(1.0 - m)                                          # minor semi-axis
    p0 = 3.0 * F_N / (2.0 * math.pi * A * B)                            # F = (2/3) pi A B p0
    delta = p0 * B * K / e_star
    # the major semi-axis lies along the direction of the smaller relative curvature (flatter gap)
    b, a = (A, B) if k_x <= k_y else (B, A)
    validity = _size_validity((("b/Rx_rel", b, 1.0 / k_x), ("a/Ry_rel", a, 1.0 / k_y),
                               ("b/R1x", b, R1x_m), ("b/R2x", b, R2x_m), ("a/R1y", a, R1y_m), ("a/R2y", a, R2y_m)))
    return HertzPoint(a_m=a, b_m=b, p0_Pa=p0, pm_Pa=2.0 * p0 / 3.0, delta_m=delta,
                      k_n_N_per_m=1.5 * F_N / delta, E_star_Pa=e_star, validity=validity)


def hertz_line(R1_m, R2_m, E1_Pa, nu1, E2_Pa, nu2, w_N_per_m) -> HertzLine:
    """Hertz solution of a line contact between two elastic cylinders with parallel axes.

    Two smooth bodies, both straight along ``y`` (the line axis), touch along a line and are
    pressed together by ``w_N_per_m``. The gap before loading is ``h = x^2 / (2 R)`` with
    ``1/R = 1/R1 + 1/R2``; the contact is the strip ``|x| <= b`` and the pressure is
    ``p = p0 sqrt(1 - (x/b)^2)``.

    Parameters
    ----------
    R1_m, R2_m : float
        Radii of curvature of body 1 and body 2 along ``x`` [m]; convex > 0, concave < 0, flat
        ``math.inf``.
    E1_Pa, E2_Pa : float
        Young's moduli [Pa].
    nu1, nu2 : float
        Poisson's ratios [-].
    w_N_per_m : float
        Normal load per unit length of the line [N/m].

    Returns
    -------
    HertzLine
        No approach is returned: for a line contact it is not defined by the half-space solution,
        it depends on the size and support of the bodies (Johnson 1985, Sec. 4.2(c)).

    Raises
    ------
    ValueError
        Domain errors: a zero radius, ``E <= 0``, ``nu`` outside (-1, 0.5), ``w_N_per_m <= 0``,
        or ``1/R <= 0`` (the bodies do not form a Hertz contact).

    Notes
    -----
    Plane strain, otherwise the assumptions of :func:`hertz_point`. Johnson 1985 Eqs
    (4.43)-(4.45)::

        b  = (4 w R / (pi E*))^(1/2)
        p0 = 2 w / (pi b) = (w E* / (pi R))^(1/2)

    Examples
    --------
    A 10 mm steel roller on the inner raceway of a 70 mm pitch diameter, 5e4 N/m:

    >>> r = hertz_line(5e-3, 30e-3, 210e9, 0.3, 210e9, 0.3, 5e4)
    >>> round(r.b_m * 1e6, 2), round(r.p0_Pa / 1e6, 1)
    (48.63, 654.6)
    """
    k_x = _curvature(R1_m, "R1_m") + _curvature(R2_m, "R2_m")          # relative curvature 1/R [1/m]
    e_star = _e_star(E1_Pa, nu1, E2_Pa, nu2)
    if not k_x > 0.0:
        raise ValueError(f"relative curvature must be positive, got {k_x}")
    if not w_N_per_m > 0.0:
        raise ValueError(f"w_N_per_m must be positive, got {w_N_per_m}")
    b = math.sqrt(4.0 * w_N_per_m / (math.pi * k_x * e_star))          # Eq. (4.43)
    p0 = 2.0 * w_N_per_m / (math.pi * b)                                # Eq. (4.45), w = (pi/2) b p0
    return HertzLine(b_m=b, p0_Pa=p0, w_N_per_m=w_N_per_m,
                     validity=_size_validity((("b/R_rel", b, 1.0 / k_x), ("b/R1", b, R1_m), ("b/R2", b, R2_m))))


def hamrock_brewe_fit(alpha: float) -> dict:   # OPEN MG-04: dict as classes_and_workflow H.1, not a dataclass
    """Hamrock-Brewe curve fits of the ellipticity and the elliptic integrals, ``alpha >= 1``.

    Parameters
    ----------
    alpha : float
        Radius ratio ``R_y / R_x`` of the equivalent body (Hamrock-Brewe Eq. (6)) [-].

    Returns
    -------
    dict
        ``k``: ellipticity ``D_y / D_x`` (Eq. (29)); ``E``: second-kind integral (Eq. (30));
        ``F``: first-kind integral (Eq. (32)), with ``q = pi/2 - 1`` (Eq. (31)); ``validity``:
        ``alpha`` against the fitted range (soft, ``alpha <= 100``, Tables 1-2).

    Raises
    ------
    ValueError
        ``alpha < 1``: the branch ``0.01 <= alpha < 1`` (Table 4) is not implemented.
    """
    if not alpha >= 1.0:
        raise ValueError(f"alpha must be >= 1 (the alpha < 1 branch is not implemented), got {alpha}")
    q = math.pi / 2.0 - 1.0                                             # Eq. (31)
    return {"k": alpha ** (2.0 / math.pi),                              # Eq. (29)
            "E": 1.0 + q / alpha,                                       # Eq. (30)
            "F": math.pi / 2.0 + q * math.log(alpha),                   # Eq. (32)
            "validity": Validity(alpha <= ALPHA_FIT_MAX, "alpha", alpha)}
