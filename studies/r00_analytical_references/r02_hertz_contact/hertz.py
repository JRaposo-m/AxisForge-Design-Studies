"""Hertz point and line contact, computed by SlipPY ``hertz_full`` (r02).

r02 runs Hertz with SlipPY ``hertz_full`` (``slippy/contact/hertz.py``, branch ``local-fixes``), the one
library function an r case may import (D-0036 amendment rule 13: analytical formulas José verified against
Johnson and Hamrock-Brewe, 2026-10-27). This module checks the inputs, gives ``hertz_full`` the
equivalent body and maps its outputs to the r02 records; the own exact solution (Carlson ``R_D``) is the
check of ``hertz_full`` and lives in ``tests/test_r02.py``. Sources: Johnson, Contact Mechanics (1985),
ch. 4; Hamrock and Brewe (1983); Deeg (1996) inside ``hertz_full``. README section 2.

Axes and signs (README section 3): ``x`` is the rolling direction, ``y`` lies across it. A radius
is positive for a convex surface, negative for a concave one, ``math.inf`` for a flat. The
semi-axis ``b`` lies along ``x`` and ``a`` along ``y`` (Hamrock-Brewe: ``D_x = 2 b``,
``D_y = 2 a``); either may be the larger one, unlike Johnson's and SlipPY's ``a >= b`` (OPEN MG-05: the
bridge's convention). SI units throughout.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from slippy.contact.hertz import hertz_full

__all__ = ["Validity", "HertzPoint", "HertzLine", "hertz_point", "hertz_line", "hamrock_brewe_fit",
           "SIZE_RATIO_MAX", "ALPHA_FIT_MAX"]

SIZE_RATIO_MAX = 0.1    # soft range: semi-axis / radius (P-02 policy; Johnson 1985 Sec. 4.2 only says 'much
                        # smaller than')
ALPHA_FIT_MAX = 100.0   # upper end of the Hamrock-Brewe fits, alpha = R_y / R_x (Tables 1-2)
# OPEN RC-01: relative tolerance of the root for b/a in hertz_full (SlipPY default 1e-6 leaves up to 6.5e-8 on
# a, b, p0, delta against the exact solution; 1e-12 leaves 6e-12 up to k_max/k_min = 1e4, above it brentq's fixed
# xtol 2e-12 on b/a governs, ~6e-11). integration_error stays at the SlipPY default: quad does far better than the
# 1e-6 it is asked (measured, not guaranteed); below 1e-8 it warns of roundoff and the result does not change.
ROOT_ERROR = 1e-12


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


def _check_material(E1_Pa: float, nu1: float, E2_Pa: float, nu2: float) -> None:
    """Refuses non-physical E, nu (``hertz_full`` only refuses a zero modulus)."""
    for name, E in (("E1_Pa", E1_Pa), ("E2_Pa", E2_Pa)):
        if not (0.0 < E < math.inf):
            raise ValueError(f"{name} must be positive and finite, got {E}")
    for name, nu in (("nu1", nu1), ("nu2", nu2)):
        if not (-1.0 < nu < 0.5):
            raise ValueError(f"{name} must lie in (-1, 0.5), got {nu}")


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
        or a concave body tighter than the convex one, ``k < 0``). From ``hertz_full``: a
        curvature ratio ``k_max / k_min`` above about 7.9e4 (its root search for ``b/a`` in
        [1e-5, 1] fails, OPEN RC-02) and above 1e5 (SlipPY treats it as a line contact).

    Notes
    -----
    Assumptions (Hertz): smooth frictionless surfaces, linear elastic isotropic bodies, each
    treated as a half-space, normal load only, no adhesion. The half-space needs contact
    dimensions small against the radii: the ratio is returned in ``validity`` (soft range
    ``a/R <= 0.1``, P-02 class ii), not refused.

    Computed by SlipPY ``hertz_full`` (Johnson 1985 Sec. 4.2; the ratio of the semi-axes from
    Deeg 1996), called as the bridge calls it (``slippy_bridge.contact.solvers.hertz``): the
    equivalent body ``r1 = (1/k_max, 1/k_min)`` against a flat ``r2 = inf`` at ``angle = 0``, with
    both materials. The gap, ``E*`` and the approach are those of the pair; giving the two bodies
    themselves would send SlipPY's angle branch a body more curved across than along.
    ``root_error = ROOT_ERROR``. Mapping of the outputs::

        hertz_full            r02
        contact_radii         sorted -> (B, A), minor and major semi-axis (SlipPY: [major, minor])
                              A along the smaller relative curvature: b, a = (A, B) if k_x <= k_y
                              else (B, A)  (MG-05: b along x, a along y)
        max_pressure          p0_Pa
        mean_pressure         pm_Pa      (F / (pi a b) = 2 p0 / 3)
        total_deflection      delta_m    (sum of both bodies)
        e_star                E_star_Pa  (Johnson's E*, not E' = 2 E*; Q-15)
        -                     k_n_N_per_m = 3 F / (2 delta)  (from F ~ delta^(3/2))

    The circular contact (``k_x == k_y``) is SlipPY's sphere branch, Johnson 1985 Sec. 4.2(a).

    Examples
    --------
    A 12.7 mm steel ball on a steel flat, 500 N (circular contact):

    >>> r = hertz_point(6.35e-3, 6.35e-3, math.inf, math.inf, 210e9, 0.3, 210e9, 0.3, 500.0)
    >>> round(r.p0_Pa / 1e9, 3), round(r.b_m * 1e3, 4)
    (3.173, 0.2743)
    """
    k_x = _curvature(R1x_m, "R1x_m") + _curvature(R2x_m, "R2x_m")      # relative curvatures [1/m]
    k_y = _curvature(R1y_m, "R1y_m") + _curvature(R2y_m, "R2y_m")
    _check_material(E1_Pa, nu1, E2_Pa, nu2)
    if not F_N > 0.0:
        raise ValueError(f"F_N must be positive, got {F_N}")
    k_min, k_max = min(k_x, k_y), max(k_x, k_y)
    if not k_min > 0.0:
        raise ValueError(f"relative curvatures must be positive for a point contact, got {k_x}, {k_y}")

    sol = hertz_full(r1=(1.0 / k_max, 1.0 / k_min), r2=math.inf, moduli=(E1_Pa, E2_Pa), v=(nu1, nu2),
                     load=F_N, angle=0.0, root_error=ROOT_ERROR)
    B, A = sorted(float(r) for r in sol["contact_radii"])              # minor, major semi-axis
    # the major semi-axis lies along the direction of the smaller relative curvature (flatter gap)
    b, a = (A, B) if k_x <= k_y else (B, A)
    delta = float(sol["total_deflection"])
    validity = _size_validity((("b/Rx_rel", b, 1.0 / k_x), ("a/Ry_rel", a, 1.0 / k_y),
                               ("b/R1x", b, R1x_m), ("b/R2x", b, R2x_m), ("a/R1y", a, R1y_m), ("a/R2y", a, R2y_m)))
    return HertzPoint(a_m=a, b_m=b, p0_Pa=float(sol["max_pressure"]), pm_Pa=float(sol["mean_pressure"]),
                      delta_m=delta, k_n_N_per_m=1.5 * F_N / delta, E_star_Pa=float(sol["e_star"]),
                      validity=validity)


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
    Plane strain, otherwise the assumptions of :func:`hertz_point`. Computed by SlipPY
    ``hertz_full`` with ``line=True``, Johnson 1985 Eqs (4.43)-(4.45)::

        b  = (4 w R / (pi E*))^(1/2)
        p0 = 2 w / (pi b) = (w E* / (pi R))^(1/2)

    called as the bridge calls it: the equivalent cylinder ``r1 = (R, inf)``, straight along ``y``,
    against a flat ``r2 = inf``, with both materials. Mapping: the finite entry of
    ``contact_radii`` (SlipPY: ``[b, inf]``) -> ``b_m``; ``max_pressure`` -> ``p0_Pa``. SlipPY's
    ``mean_pressure`` (``w / (2 b)``) and ``e_star`` are not part of :class:`HertzLine`.

    Examples
    --------
    A 10 mm steel roller on the inner raceway of a 70 mm pitch diameter, 5e4 N/m:

    >>> r = hertz_line(5e-3, 30e-3, 210e9, 0.3, 210e9, 0.3, 5e4)
    >>> round(r.b_m * 1e6, 2), round(r.p0_Pa / 1e6, 1)
    (48.63, 654.6)
    """
    k_x = _curvature(R1_m, "R1_m") + _curvature(R2_m, "R2_m")          # relative curvature 1/R [1/m]
    _check_material(E1_Pa, nu1, E2_Pa, nu2)
    if not k_x > 0.0:
        raise ValueError(f"relative curvature must be positive, got {k_x}")
    if not w_N_per_m > 0.0:
        raise ValueError(f"w_N_per_m must be positive, got {w_N_per_m}")
    sol = hertz_full(r1=(1.0 / k_x, math.inf), r2=math.inf, moduli=(E1_Pa, E2_Pa), v=(nu1, nu2),
                     load=w_N_per_m, angle=0.0, line=True)
    b = min(float(r) for r in sol["contact_radii"])                    # [b, inf]: the half-width
    return HertzLine(b_m=b, p0_Pa=float(sol["max_pressure"]), w_N_per_m=w_N_per_m,
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
