"""Hertz elliptical point contact, written from the theory: the independent judge.

It judges both AxisForge (``PointContactStiffness``, ISO/TS 16281) and SlipPY (``hertz_full``),
so it shares no code with either: only NumPy and SciPy are imported, and the formulas are those of
Johnson, Contact Mechanics (1985), ch. 4 (Hertz).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import ellipk, elliprd

__all__ = ["hertz_point"]


def hertz_point(r1, r2, E, nu, load) -> dict:
    """Hertz solution of an elliptical point contact between two elastic bodies.

    Two smooth bodies touch at a point and are pressed together by ``load``. Near the point each
    surface is a paraboloid, so the gap between them before loading is
    ``h(x, y) = k_x x^2 / 2 + k_y y^2 / 2``, with the relative curvatures
    ``k_x = 1/r1_x + 1/r2_x`` and ``k_y = 1/r1_y + 1/r2_y``. The contact area is an ellipse with
    semi-axes ``s_x``, ``s_y`` and the pressure is ``p = p0 sqrt(1 - (x/s_x)^2 - (y/s_y)^2)``.

    Parameters
    ----------
    r1, r2 : (float, float) or float
        Principal radii of curvature ``(r_x, r_y)`` of body 1 and body 2 at the contact point [m].
        ``x`` is the rolling direction. Convex > 0, concave < 0, flat = ``math.inf``; a scalar
        means ``(r, r)``. The principal directions of the two bodies must coincide (angle 0).
    E : (float, float) or float
        Young's moduli ``(E1, E2)`` [Pa]; a scalar applies to both bodies.
    nu : (float, float) or float
        Poisson's ratios ``(nu1, nu2)`` [-]; a scalar applies to both bodies.
    load : float
        Normal load [N].

    Returns
    -------
    dict
        ``p0_Pa``: maximum contact pressure [Pa]; ``s_x_m``, ``s_y_m``: semi-axes of the contact
        ellipse along ``x`` (rolling) and ``y`` (across rolling) [m]; ``delta_m``: total approach
        of distant points of the two bodies [m] (the sum of the deformations of both).

    Raises
    ------
    ValueError
        If a relative curvature is not positive: a line contact (``k = 0``) or bodies that do not
        form a Hertz contact (``k < 0``). Point contact only.

    Notes
    -----
    Assumptions (Hertz): smooth frictionless surfaces, linear elastic isotropic bodies, each
    treated as a half-space, contact dimensions small against the radii of curvature, normal load
    only, no adhesion.

    With ``a >= b`` the semi-axes, ``m = e^2 = 1 - (b/a)^2`` and ``K(m)``, ``E(m)`` the complete
    elliptic integrals, the ellipsoidal pressure displaces the surfaces by
    ``u = (L - M x^2 - N y^2) / (pi E*)`` (``x`` along ``a``)::

        M = pi p0 b (K - E) / (m a^2)
        N = pi p0 b ((a/b)^2 E - K) / (m a^2)
        L = pi p0 b K

    Contact requires ``u = delta - h`` inside the ellipse, hence ``M = pi E* k_min / 2``,
    ``N = pi E* k_max / 2`` and ``delta = L / (pi E*)``. ``N / M = k_max / k_min`` fixes ``m``
    (one scalar root), and the load ``Q = (2/3) pi a b p0`` closes the system.

    ``(K - E) / m`` and ``(E - (1 - m) K) / m`` are 0/0 for a circular contact (``m = 0``), so
    they are evaluated as Carlson's symmetric integral ``R_D``, which is regular there::

        (K - E) / m                     = R_D(0, 1 - m, 1) / 3
        (E - (1 - m) K) / (m (1 - m))   = R_D(0, 1, 1 - m) / 3

    Both tend to ``pi / 4`` at ``m = 0``, so the circular contact is the root ``m = 0`` of the same
    equation and needs no special case.

    References
    ----------
    Johnson, Contact Mechanics (1985), ch. 3 (Boussinesq) / ch. 4 (Hertz). Equation numbers to be
    added from the book.

    Examples
    --------
    A 12.7 mm steel ball on a steel flat, 500 N (circular contact):

    >>> ref = hertz_point(6.35e-3, math.inf, 210e9, 0.3, 500.0)
    >>> round(ref["p0_Pa"] / 1e9, 3), round(ref["s_x_m"] * 1e3, 4)
    (3.173, 0.2743)
    """
    (r1_x, r1_y), (r2_x, r2_y) = np.broadcast_to(r1, 2), np.broadcast_to(r2, 2)
    (E1, E2), (nu1, nu2) = np.broadcast_to(E, 2), np.broadcast_to(nu, 2)
    k_x = 1.0 / r1_x + 1.0 / r2_x                 # relative curvatures [1/m]; 1/inf = 0 for a flat
    k_y = 1.0 / r1_y + 1.0 / r2_y
    k_min, k_max = min(k_x, k_y), max(k_x, k_y)
    if not k_min > 0.0:
        raise ValueError(f"relative curvatures must be positive for a point contact, got {k_x}, {k_y}")
    e_star = 1.0 / ((1.0 - nu1**2) / E1 + (1.0 - nu2**2) / E2)      # contact modulus E* [Pa]

    # Eccentricity m = e^2 from N/M = k_max/k_min, in Carlson's form (see Notes):
    # k_min R_D(0, 1, 1-m) = k_max R_D(0, 1-m, 1). Negative or zero at m = 0 (zero = circular, the
    # root itself) and positive towards m = 1, where the left side grows as 1/(1 - m). The upper
    # end stops short of 1 (K is infinite there); 1 - 1e-12 allows a/b up to 1e6.
    m = brentq(lambda m: k_min * elliprd(0.0, 1.0, 1.0 - m) - k_max * elliprd(0.0, 1.0 - m, 1.0),
               0.0, 1.0 - 1e-12, xtol=1e-16)
    K = ellipk(m)
    D = elliprd(0.0, 1.0 - m, 1.0) / 3.0          # (K - E) / m

    a = float(np.cbrt(3.0 * load * D / (math.pi * k_min * e_star)))  # major semi-axis, from M and Q
    b = a * math.sqrt(1.0 - m)                                        # minor semi-axis
    p0 = 3.0 * load / (2.0 * math.pi * a * b)                         # Q = (2/3) pi a b p0
    # the long semi-axis lies along the direction of the smaller relative curvature (flatter gap)
    s_x, s_y = (a, b) if k_x <= k_y else (b, a)
    return {"p0_Pa": float(p0), "s_x_m": s_x, "s_y_m": s_y, "delta_m": float(p0 * b * K / e_star)}
