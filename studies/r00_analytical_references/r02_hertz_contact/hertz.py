"""Hertz contact of two bodies as given, computed by SlipPY ``hertz_full`` (r02).

r02 runs Hertz with SlipPY ``hertz_full`` (``slippy/contact/hertz.py``, branch ``local-fixes``), the one
library function an r case may import (D-0036 amendment rule 13: analytical formulas José verified against
Johnson and Hamrock-Brewe, 2026-10-27). :func:`hertz` passes each body's own principal radii, the materials,
the load and the angle between the bodies to ``hertz_full`` as given and maps its outputs to
:class:`HertzResult`; the own exact solution (Carlson ``R_D``) is the check of ``hertz_full`` and lives in
``tests/test_r02.py``. Sources: Johnson, Contact Mechanics (1985), ch. 4; Hamrock and Brewe (1983); Deeg
(1996) inside ``hertz_full``. README sections 2-4.

Conventions (``hertz_full``'s, README section 4): each body has its own principal axes ``x``, ``y``; a radius
is positive for a convex surface, negative for a concave one, ``math.inf`` for a flat. ``angle_rad`` is the
angle between the ``x`` axes of the two bodies. ``a >= b`` (Johnson). SI units throughout.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

from slippy.contact.hertz import hertz_full

__all__ = ["Validity", "HertzResult", "hertz", "hertz_point", "hamrock_brewe_fit", "SIZE_RATIO_MAX",
           "ALPHA_FIT_MAX"]

SIZE_RATIO_MAX = 0.1    # soft range: semi-axis / radius (P-02 policy; Johnson 1985 Sec. 4.2 only says 'much
                        # smaller than')
ALPHA_FIT_MAX = 100.0   # upper end of the Hamrock-Brewe fits, alpha = R_y / R_x (Tables 1-2)
# OPEN RC-01: relative tolerance of the root for b/a in hertz_full (SlipPY default 1e-6 leaves up to 6.5e-8 on
# a, b, p0, delta against the exact solution; 1e-12 leaves 6e-12 up to k_max/k_min = 1e4, above it brentq's fixed
# xtol 2e-12 on b/a governs, ~6e-11). integration_error keeps the SlipPY default 1e-6: hertz_full uses it in the
# root function only (its integrals for a, b and delta keep 1e-6 whatever is passed, OPEN RD-09), and quad does
# far better than asked (measured, not guaranteed).
ROOT_ERROR = 1e-12


@dataclass(frozen=True)
class Validity:
    """Soft validity range of a result (P-02 class ii; class iii clamps do not occur in r02).

    Attributes
    ----------
    in_range : bool
        ``ratio`` lies inside the range the formula is valid in.
    governing : str
        The quantity that sets ``ratio``, e.g. ``"a/r2"`` (largest semi-axis over the smallest finite
        ``|radius|`` of body 2).
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
class HertzResult:
    """Hertz contact of two bodies (``hertz_full`` outputs, README section 4).

    Attributes
    ----------
    shape : {"sphere", "elliptical", "line"}
        ``hertz_full``'s ``contact_shape``.
    a_m, b_m : float
        Major and minor semi-axis of the contact, ``a_m >= b_m`` [m]. Line: ``a_m = inf``, ``b_m`` the
        half-width.
    p0_Pa, pm_Pa : float
        Maximum and mean contact pressure [Pa] (point: ``pm = F / (pi a b) = 2 p0 / 3``; line:
        ``pm = w / (2 b)``).
    delta_m : float or None
        Approach of distant points of the two bodies (sum of both deformations) [m]; None for a line
        (not defined by the half-space solution, Johnson 1985 Sec. 4.2(c); ``hertz_full`` gives none).
    area_m2 : float or None
        Contact area ``pi a b`` [m^2]; None for a line.
    E_star_Pa : float
        Contact modulus ``1/E* = (1 - nu1^2)/E1 + (1 - nu2^2)/E2`` [Pa] (Johnson's E*, not E' = 2 E*).
    Rx_rel_m, Ry_rel_m : float
        Principal relative radii of the gap (``relative_radii``) [m]. ``hertz_full`` puts first the smaller
        one, along the minor semi-axis ``b`` (its ``alpha`` is the direction of the largest summed curvature),
        so ``Rx_rel_m <= Ry_rel_m`` and ``Ry_rel_m`` lies along ``a``; the names are ``hertz_full``'s, not
        the ``x`` axis of a body.
    R_e_m : float
        Equivalent radius ``sqrt(Rx_rel Ry_rel)`` [m] (``r_e``).
    alpha_rad, beta_rad : float
        Orientation of the major semi-axis ``a``: it lies at ``alpha_rad`` from the ``x`` axis of body 1
        and at ``-beta_rad`` from the ``x`` axis of body 2, angles in the sense of ``angle_rad``
        (``alpha_rad + beta_rad = angle_rad`` modulo ``pi``; Johnson 1985 Sec. 4.1). Established in
        ``tests/test_r02.py`` against the eigenvectors of the curvature sum. Sphere: no orientation
        (``hertz_full``'s value kept). Line: ``a`` lies along the line. A body given with its axes swapped
        (RD-03) moves the reference axis of its angle with it.
    k_n_N_per_m : float or None
        Tangent normal stiffness ``dF / d delta = 3 F / (2 delta)`` [N/m]; None for a line.
    validity : Validity
        Largest semi-axis over the smallest finite ``|radius|`` of each body, against ``SIZE_RATIO_MAX``.
    raw : dict
        The ``hertz_full`` dict, for its pressure and stress closures. Its ``alpha``, ``beta`` point along
        ``b`` (before its own ``pi/2`` turn). OPEN RD-04: ``raw["pressure_f"](x, y, transform=1)`` and
        ``raw["surface_displacement_b_f"][i](x, y, transform=1)`` are the mirror image (``y -> -y``) of the
        body-1 field the eigenvectors give; ``transform=2`` and ``transform=0`` (contact axes, ``a`` along
        x) agree with them.
    """

    shape: str
    a_m: float
    b_m: float
    p0_Pa: float
    pm_Pa: float
    delta_m: float | None
    area_m2: float | None
    E_star_Pa: float
    Rx_rel_m: float
    Ry_rel_m: float
    R_e_m: float
    alpha_rad: float
    beta_rad: float
    k_n_N_per_m: float | None
    validity: Validity
    raw: dict


def _pair(value, name: str) -> tuple[float, float]:
    """``value`` as (body 1, body 2) or (x, y): a number is used twice. Refuses anything else."""
    def real(v):
        return isinstance(v, numbers.Real) and not isinstance(v, bool)

    if real(value):
        return float(value), float(value)
    if isinstance(value, (tuple, list)) and len(value) == 2 and all(real(v) for v in value):
        return float(value[0]), float(value[1])
    raise TypeError(f"{name} must be a number or a pair of numbers, got {value!r}")


def _size_validity(a: float, b: float, bodies) -> Validity:
    """Largest finite semi-axis over the smallest finite ``|radius|`` of each body (P-02 class ii).

    Johnson's condition (iii) of Sec. 4.2 (each body small against the contact) only; (i) and (ii), against
    the relative radii, are not checked (OPEN RD-07, with MG-08: José). A concave radius counts but never
    sets the ratio: a positive relative curvature along it needs a tighter convex radius on the other body.
    """
    s, axis = (b, "b") if math.isinf(a) else (a, "a")      # a line: the half-width
    candidates = [(s / min(abs(r) for r in radii if math.isfinite(r)), f"{axis}/{name}")
                  for name, radii in bodies if any(math.isfinite(r) for r in radii)]
    ratio, governing = max(candidates)
    return Validity(ratio <= SIZE_RATIO_MAX, governing, ratio)


def hertz(r1_m, r2_m, E_Pa, nu, load, *, angle_rad: float = 0.0, line: bool = False,
          integration_error: float = 1e-6, root_error: float = ROOT_ERROR) -> HertzResult:
    """Hertz contact of two smooth elastic bodies, each given in its own principal axes.

    Near the point of first contact each surface is a paraboloid with its own principal radii. The gap
    between them is the sum of the two curvature tensors, its principal relative radii set the contact
    ellipse ``p = p0 sqrt(1 - (s/a)^2 - (t/b)^2)`` (Johnson 1985 Sec. 4.1, Eq. (4.4); Sec. 4.2). A line
    contact (``line=True``) is the plane-strain strip ``p = p0 sqrt(1 - (x/b)^2)``.

    Parameters
    ----------
    r1_m, r2_m : float or (float, float)
        Principal radii ``(R_x, R_y)`` of body 1 and body 2, each in its OWN principal axes [m]; a number
        is a body of revolution ``(R, R)``. Convex > 0, concave < 0, flat ``math.inf``.
    E_Pa : float or (float, float)
        Young's modulus of both bodies, or (body 1, body 2) [Pa].
    nu : float or (float, float)
        Poisson's ratio of both bodies, or (body 1, body 2) [-].
    load : float
        Normal load [N]; with ``line=True`` the load per unit length of the line [N/m].
    angle_rad : float, optional
        Angle between the ``x`` axis of body 1 and the ``x`` axis of body 2 [rad]; 0 when the axes
        coincide. Only ``angle_rad`` modulo ``pi`` matters.
    line : bool, optional
        True for a line contact (``hertz_full`` refuses to guess: one relative radius over 1e5 times the
        other is a line and must be flagged).
    integration_error, root_error : float, optional
        ``hertz_full``'s relative tolerances of the integrals inside its root function (only there,
        OPEN RD-09) and of the root ``b/a``; defaults 1e-6 (SlipPY's) and ``ROOT_ERROR = 1e-12`` (OPEN
        RC-01; SlipPY's default 1e-6 leaves up to 6.5e-8 relative on a, b, p0, delta).

    Returns
    -------
    HertzResult

    Raises
    ------
    TypeError
        A radius, modulus or Poisson's ratio that is not a real number (bool refused) or a pair of them.
    ValueError
        From r02: ``E <= 0`` or not finite, ``nu`` outside (-1, 0.5), a load that is not finite
        (``hertz_full`` checks none of them; its zero-modulus check never runs, OPEN RD-08).
        From ``hertz_full``, passed through: a zero radius, ``load <= 0``, a negative relative radius
        (a concave body tighter than the convex one), a conformal pair, a line contact without
        ``line=True`` and ``line=True`` on a point contact, a curvature ratio above about 7.9e4 (its
        root search for ``b/a`` in [1e-5, 1] fails, OPEN RC-02), and "There is a bug in this program"
        when both bodies are more curved along ``y`` than along ``x`` (``1/R_x < 1/R_y``), at any angle
        (OPEN RD-03: give one of them with its axes swapped and add ``pi/2`` to ``angle_rad``). A negative
        relative radius also warns (RuntimeWarning, ``r_e`` is computed before the check).

    Notes
    -----
    Assumptions (Hertz): smooth frictionless surfaces, linear elastic isotropic bodies, each treated as a
    half-space, normal load only, no adhesion. The half-space needs contact dimensions small against the
    radii: the ratio is returned in ``validity`` (soft range ``<= SIZE_RATIO_MAX``, P-02 class ii), not
    refused. Give the bodies as they are, not an equivalent body against a flat: the validity is judged
    per body.

    Mapping of the ``hertz_full`` outputs::

        hertz_full          HertzResult
        contact_shape       shape
        contact_radii       sorted -> a_m >= b_m      (line: a_m = inf, b_m = the smaller entry)
        max_pressure        p0_Pa
        mean_pressure       pm_Pa
        total_deflection    delta_m                   (line: none -> None)
        contact_area        area_m2                   (line: inf -> None)
        e_star              E_star_Pa
        relative_radii      Rx_rel_m, Ry_rel_m
        r_e                 R_e_m
        alpha, beta         alpha_rad, beta_rad       (+ pi/2 when Rx_rel < Ry_rel: on the major axis,
                                                       OPEN RD-05)
        -                   k_n_N_per_m = 3 F / (2 delta)
        (the dict)          raw

    Examples
    --------
    A 12.7 mm steel ball on a steel flat, 500 N; two steel cylinders R 10 mm crossed at 45 deg, 100 N:

    >>> r = hertz(6.35e-3, math.inf, 210e9, 0.3, 500.0)
    >>> r.shape, round(r.p0_Pa / 1e9, 3), round(r.a_m * 1e3, 4)
    ('sphere', 3.173, 0.2743)
    >>> r = hertz((10e-3, math.inf), (10e-3, math.inf), 210e9, 0.3, 100.0, angle_rad=math.pi / 4)
    >>> r.shape, round(r.p0_Pa / 1e6, 3), round(r.b_m * 1e3, 6)
    ('elliptical', 1178.488, 0.11269)
    """
    r1, r2 = _pair(r1_m, "r1_m"), _pair(r2_m, "r2_m")
    E, v = _pair(E_Pa, "E_Pa"), _pair(nu, "nu")
    if not all(0.0 < e < math.inf for e in E):
        raise ValueError(f"E_Pa must be positive and finite, got {E_Pa}")
    if not all(-1.0 < n < 0.5 for n in v):
        raise ValueError(f"nu must lie in (-1, 0.5), got {nu}")
    if not math.isfinite(load):                             # hertz_full lets a NaN load through
        raise ValueError(f"load must be finite, got {load}")

    sol = hertz_full(r1=r1, r2=r2, moduli=E, v=v, load=load, angle=angle_rad, line=line,
                     integration_error=integration_error, root_error=root_error)
    point = sol["contact_shape"] != "line"
    b, a = sorted(float(r) for r in sol["contact_radii"])
    if not point:   # hertz_full puts inf only where a relative radius is exactly inf, not above its 1e5 ratio
        a = math.inf
    Rx_rel, Ry_rel = (float(r) for r in sol["relative_radii"])
    # hertz_full's alpha, beta: the direction of the largest summed curvature (b); turned onto a (OPEN RD-05)
    turn = math.pi / 2 if Rx_rel < Ry_rel else 0.0
    delta = float(sol["total_deflection"]) if point else None
    return HertzResult(shape=sol["contact_shape"], a_m=a, b_m=b, p0_Pa=float(sol["max_pressure"]),
                       pm_Pa=float(sol["mean_pressure"]), delta_m=delta,
                       area_m2=float(sol["contact_area"]) if point else None, E_star_Pa=float(sol["e_star"]),
                       Rx_rel_m=Rx_rel, Ry_rel_m=Ry_rel, R_e_m=float(sol["r_e"]),
                       alpha_rad=float(sol["alpha"]) + turn, beta_rad=float(sol["beta"]) + turn,
                       k_n_N_per_m=1.5 * load / delta if point else None,
                       validity=_size_validity(a, b, (("r1", r1), ("r2", r2))), raw=sol)


def hertz_point(R1x_m, R1y_m, R2x_m, R2y_m, E1_Pa, nu1, E2_Pa, nu2, F_N) -> HertzResult:
    """Pre-R1d call kept for ``tests/test_slippy_bridge.py`` (RC-03): :func:`hertz` at ``angle_rad = 0``.

    Note ``a_m >= b_m`` now (before R1d ``b`` lay along ``x``).
    """
    return hertz((R1x_m, R1y_m), (R2x_m, R2y_m), (E1_Pa, E2_Pa), (nu1, nu2), F_N)


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
