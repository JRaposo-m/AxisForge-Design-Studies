"""Hertz point and line contact, written from the theory: the independent judge.

It judges both AxisForge (``PointContactStiffness``, ISO/TS 16281) and SlipPY (``hertz_full``),
so it shares no code with either: only NumPy and SciPy are imported, and the formulas are those of
Johnson, Contact Mechanics (1985), ch. 3-4 (Hertz).

Symbols. The bridge names are used: ``s_x`` is the semi-axis along rolling (``x``), ``s_y`` across
it (``y``), depth ``z`` into the body, and ``s_min = min(s_x, s_y)``. Johnson and Harris write ``a``
for the major and ``b`` for the minor semi-axis; for a line contact and for a circle ``s_min`` is
Johnson's ``a`` (semi-width, resp. radius), so ``z/s_min`` is his ``z/a``. On a ball-raceway contact
``s_x`` is the minor semi-axis ``b``.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import ellipk, elliprd

__all__ = ["hertz_point", "hertz_line", "hertz_pressure", "hertz_line_stress", "hertz_point_axis_stress"]


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


def hertz_line(r1, r2, E, nu, w) -> dict:
    """Hertz solution of a line contact between two elastic cylinders with parallel axes.

    Two smooth bodies, both straight along ``y`` (the line axis), touch along a line and are
    pressed together by the load ``w`` per unit length. Near the line the gap before loading is
    ``h(x) = x^2 / (2 R)`` with ``1/R = 1/r1_x + 1/r2_x``. The contact is a strip ``|x| <= s_x`` and
    the pressure is ``p = p0 sqrt(1 - (x/s_x)^2)``; ``s_x`` is Johnson's semi-width ``a``.

    Parameters
    ----------
    r1, r2 : (float, float) or float
        Principal radii of curvature ``(r_x, r_y)`` of body 1 and body 2 at the contact line [m].
        ``x`` is the rolling direction, ``y`` the line axis. Convex > 0, concave < 0, flat =
        ``math.inf``. ``r_y`` must be ``math.inf`` on both bodies, so a scalar is accepted only
        as ``math.inf`` (a flat).
    E : (float, float) or float
        Young's moduli ``(E1, E2)`` [Pa]; a scalar applies to both bodies.
    nu : (float, float) or float
        Poisson's ratios ``(nu1, nu2)`` [-]; a scalar applies to both bodies.
    w : float
        Normal load per unit length of the line [N/m].

    Returns
    -------
    dict
        ``s_x_m``: semi-width of the contact strip along ``x`` (rolling) [m]; ``s_y_m``: ``inf``
        (the strip is unbounded along the line axis); ``p0_Pa``: maximum contact pressure [Pa].
        The keys are those of :func:`hertz_point`. No approach is returned: for a line contact it is
        not defined by the local (half-space) solution, it depends on the size and support of the
        bodies (Johnson 1985, Sec. 4.2(c), PDF 110).

    Raises
    ------
    ValueError
        If a body is curved along ``y`` (not a line contact), if ``1/R <= 0`` (the bodies do not
        form a Hertz contact) or if ``w <= 0``.

    Notes
    -----
    Assumptions (Hertz, two-dimensional): plane strain, smooth frictionless surfaces, linear
    elastic isotropic bodies each treated as a half-space, strip width small against the radii,
    normal load only, no adhesion.

    With ``E' = 2 E*`` and ``1/E* = (1 - nu1^2)/E1 + (1 - nu2^2)/E2``::

        s_x = (8 w R / (pi E'))^(1/2) = (4 w R / (pi E*))^(1/2)
        p0  = 2 w / (pi s_x)          = (w E* / (pi R))^(1/2)

    the second line being the load tie ``w = (pi / 2) s_x p0`` of the elliptical pressure.

    References
    ----------
    Johnson, Contact Mechanics (1985), Sec. 4.2(c), Eqs (4.43)-(4.45) (book p. 101, PDF 111).

    Examples
    --------
    A 10 mm steel roller on the inner raceway of a 70 mm pitch diameter, 5e4 N/m:

    >>> ref = hertz_line((5e-3, math.inf), (30e-3, math.inf), 210e9, 0.3, 5e4)
    >>> round(ref["s_x_m"] * 1e6, 2), round(ref["p0_Pa"] / 1e6, 1)
    (48.63, 654.6)
    """
    (r1_x, r1_y), (r2_x, r2_y) = np.broadcast_to(r1, 2), np.broadcast_to(r2, 2)
    (E1, E2), (nu1, nu2) = np.broadcast_to(E, 2), np.broadcast_to(nu, 2)
    # a line contact needs both bodies straight along the line axis: 1/inf = 0
    if not (1.0 / r1_y == 0.0 and 1.0 / r2_y == 0.0):
        raise ValueError(f"both bodies must be straight along y for a line contact, got r_y {r1_y}, {r2_y}")
    k_x = 1.0 / r1_x + 1.0 / r2_x                 # relative curvature 1/R [1/m]
    if not k_x > 0.0:
        raise ValueError(f"relative curvature along x must be positive, got {k_x}")
    if not w > 0.0:
        raise ValueError(f"load per unit length must be positive, got {w}")
    e_star = 1.0 / ((1.0 - nu1**2) / E1 + (1.0 - nu2**2) / E2)      # contact modulus E* [Pa]

    s_x = math.sqrt(4.0 * w / (math.pi * k_x * e_star))             # Eq. (4.43), R = 1/k_x
    p0 = 2.0 * w / (math.pi * s_x)                                    # Eq. (4.45), w = (pi/2) s_x p0
    return {"s_x_m": s_x, "s_y_m": math.inf, "p0_Pa": p0}


def hertz_pressure(judge: dict, x_m, y_m=0.0) -> np.ndarray:
    """Hertz contact pressure at surface points, from a solved contact.

    ``p = p0 sqrt(1 - (x/s_x)^2 - (y/s_y)^2)`` inside the contact, 0 outside: the ellipsoidal
    pressure of a point contact (Johnson 1985, ch. 4); for a line contact (``s_y`` infinite) the
    ``y`` term vanishes and ``p = p0 sqrt(1 - (x/s_x)^2)`` (Eq. (4.41)-(4.45)).

    Parameters
    ----------
    judge : dict
        Result of :func:`hertz_point` or :func:`hertz_line`: ``p0_Pa`` [Pa], ``s_x_m``, ``s_y_m`` [m].
    x_m, y_m : array_like
        Surface coordinates along rolling and across it, from the contact centre [m]; broadcast.

    Returns
    -------
    ndarray
        Pressure [Pa], the broadcast shape of ``x_m`` and ``y_m``.

    Notes
    -----
    A separate function, not a ``pressure`` callable in the result dict: callers spread the dict
    into CSV records (``validation/bearings/iso16281_ball_contact/01_contacts_vs_hertz.py``).

    Examples
    --------
    >>> ref = hertz_point(6.35e-3, math.inf, 210e9, 0.3, 500.0)
    >>> float(hertz_pressure(ref, 0.0)) == ref["p0_Pa"], float(hertz_pressure(ref, 2 * ref["s_x_m"]))
    (True, 0.0)
    """
    p0, s_x, s_y = judge["p0_Pa"], judge["s_x_m"], judge["s_y_m"]
    x, y = np.broadcast_arrays(np.asarray(x_m, dtype=float), np.asarray(y_m, dtype=float))
    rho2 = (x / s_x) ** 2 + (0.0 if math.isinf(s_y) else (y / s_y) ** 2)   # 1/inf: line contact
    return p0 * np.sqrt(np.clip(1.0 - rho2, 0.0, None))


def hertz_line_stress(x_over_s_x, z_over_s_x) -> dict:
    """Stresses in the body under a Hertz line contact (McEwen's closed form), over ``p0``.

    The elliptical pressure ``p = p0 sqrt(1 - (x/a)^2)`` on an elastic half-space in plane strain,
    frictionless. With ``m``, ``n`` defined by
    ``m^2 = (sqrt((a^2 - x^2 + z^2)^2 + 4 x^2 z^2) + (a^2 - x^2 + z^2)) / 2`` and
    ``n^2 = (sqrt(...) - (a^2 - x^2 + z^2)) / 2`` (signs of ``m`` and ``n`` those of ``z`` and ``x``)::

        sigma_x = -(p0/a) [m (1 + (z^2 + n^2)/(m^2 + n^2)) - 2 z]
        sigma_z = -(p0/a) m (1 - (z^2 + n^2)/(m^2 + n^2))
        tau_xz  = -(p0/a) n (m^2 - z^2)/(m^2 + n^2)

    Parameters
    ----------
    x_over_s_x : array_like
        Position along rolling over the semi-width ``s_x`` (Johnson's ``a``) [-].
    z_over_s_x : array_like
        Depth into the body over ``s_x`` [-], ``>= 0``. Broadcast with ``x_over_s_x``.

    Returns
    -------
    dict of ndarray
        ``sigma_x_over_p0``, ``sigma_z_over_p0`` (compressive negative), ``tau_xz_over_p0``
        (negative for ``x > 0``, sign opposite to ``x``, as Johnson's Appendix 4 table 3 ``-/+``
        prints it; the field satisfies the equilibrium equations ``d sigma_x/dx + d tau_xz/dz = 0``,
        ``d tau_xz/dx + d sigma_z/dz = 0`` with this sign) and ``tau1_over_p0`` = in-plane maximum shear
        ``sqrt(((sigma_x - sigma_z)/2)^2 + tau_xz^2)``. The out-of-plane ``sigma_y = nu (sigma_x +
        sigma_z)`` is not returned; near the surface (``z < ~0.4 a``) it can be a principal stress
        that makes the 3-D maximum shear larger than ``tau1``.

    Raises
    ------
    ValueError
        If a depth is negative (the formula is for the body, ``z >= 0``).

    Notes
    -----
    On the axis ``x = 0``: ``sigma_z = -p0 a (a^2 + z^2)^(-1/2)`` (Eq. (4.46b) as corrected by
    Appendix 4; the print is dimensionally wrong) and ``tau1 = p0 (z/a - (z/a)^2 (1 + (z/a)^2)^(-1/2))``,
    maximum 0.300 ``p0`` at ``z = 0.786 a`` (Johnson Table 4.1 prints 0.785 ``a``). At the surface edge ``x = +-a``, ``z = 0`` the formula
    is 0/0; the limit, 0, is returned.

    References
    ----------
    McEwen (1949), as given by Johnson, Contact Mechanics (1985), Sec. 4.2(c): axis stresses
    Eqs (4.46)-(4.48a) (PDF 112), field Eqs (4.48b)-(4.49c) (PDF 113-114); Appendix 4 tables
    1-3 (PDF 439-440) for printed values.
    """
    x, z = np.broadcast_arrays(np.asarray(x_over_s_x, dtype=float), np.asarray(z_over_s_x, dtype=float))
    if np.any(z < 0.0):
        raise ValueError("hertz_line_stress: depth z/s_x must be >= 0 (into the body)")
    A = 1.0 - x**2 + z**2                                      # (a^2 - x^2 + z^2) / a^2
    root = np.sqrt(A**2 + 4.0 * x**2 * z**2)
    m = np.sqrt(0.5 * (root + A))                              # sign of z: z >= 0
    n = np.sign(x) * np.sqrt(np.maximum(0.5 * (root - A), 0.0))   # sign of x; max(): round-off below 0
    den = m**2 + n**2                                          # 0 only at the surface edge x = +-a, z = 0
    safe = np.where(den > 0.0, den, 1.0)
    q = np.where(den > 0.0, (z**2 + n**2) / safe, 0.0)
    sigma_x = -(m * (1.0 + q) - 2.0 * z)
    sigma_z = -m * (1.0 - q)
    tau_xz = np.where(den > 0.0, -n * (m**2 - z**2) / safe, 0.0)   # sign opposite to x (equilibrium)
    tau1 = np.sqrt((0.5 * (sigma_x - sigma_z)) ** 2 + tau_xz**2)
    return {"sigma_x_over_p0": sigma_x, "sigma_z_over_p0": sigma_z, "tau_xz_over_p0": tau_xz,
            "tau1_over_p0": tau1}


def hertz_point_axis_stress(z_over_s, nu) -> dict:
    """Stresses on the axis ``r = 0`` under a circular Hertz contact, over ``p0``.

    The ellipsoidal pressure ``p = p0 sqrt(1 - (r/a)^2)`` on an elastic half-space (circular
    contact of radius ``a``), frictionless. On the axis, with ``zeta = z/a``::

        sigma_r = sigma_theta = p0 [-(1 + nu)(1 - zeta atan(1/zeta)) + 1/(2 (1 + zeta^2))]
        sigma_z = -p0 / (1 + zeta^2)

    These are principal stresses there, so ``tau1 = |sigma_z - sigma_r| / 2``.

    Parameters
    ----------
    z_over_s : array_like
        Depth into the body over the contact radius ``a`` (= ``s_x`` = ``s_y`` = ``s_min``) [-],
        ``>= 0``.
    nu : float
        Poisson's ratio of the body [-].

    Returns
    -------
    dict of ndarray
        ``sigma_r_over_p0``, ``sigma_z_over_p0`` (compressive negative) and ``tau1_over_p0``.
        At the surface ``sigma_z = -1`` and ``sigma_r = -(1 + 2 nu)/2``.

    Raises
    ------
    ValueError
        If a depth is negative.

    Notes
    -----
    Circular contacts only: for an elliptical contact the axis stresses depend on the
    eccentricity (Johnson Sec. 4.2(b)) and are not given here (OPEN FG-09). For ``nu = 0.3`` the
    maximum of ``tau1`` is about 0.31 ``p0`` at ``z = 0.48 a``. ``zeta atan(1/zeta)`` is written
    with ``arctan2(1, zeta)`` so that the surface value is finite.

    References
    ----------
    Johnson, Contact Mechanics (1985), Eq. (3.45) (PDF 72).
    """
    zeta = np.asarray(z_over_s, dtype=float)
    if np.any(zeta < 0.0):
        raise ValueError("hertz_point_axis_stress: depth z/s must be >= 0 (into the body)")
    sigma_r = -(1.0 + nu) * (1.0 - zeta * np.arctan2(1.0, zeta)) + 0.5 / (1.0 + zeta**2)
    sigma_z = -1.0 / (1.0 + zeta**2)
    return {"sigma_r_over_p0": sigma_r, "sigma_z_over_p0": sigma_z, "tau1_over_p0": 0.5 * np.abs(sigma_z - sigma_r)}
