"""Two elastic bodies and their contact: the problem every solver of the bridge reads.

* :class:`Body` - one body near its point of first contact: signed principal radii, elastic
  constants, and its exact surface profile.
* :class:`Contact` - the abstract base of every contact: two bodies, the load, the angle between
  them; it checks its inputs at creation and computes once what every solver needs, the
  principal relative curvatures of the gap, their orientation ``psi`` and the contact modulus
  ``E*``. The machine elements subclass it and set ``line`` (point or line contact).

Bridge axes: x is the rolling direction (array axis 0 of every grid), y across it, z along the
common normal. The origin is the point of first contact. SI units throughout. Numpy and the
standard library only: no SlipPY here.

References
----------
Johnson, K. L. (1985). *Contact Mechanics*. Cambridge University Press. Section 4.1, Eqs.
(4.1)-(4.5), p. 84-85 (the gap of two non-conforming bodies and its principal relative radii;
derivation Appendix 2, p. 425-426); p. 86 (two crossed cylinders at 45 deg); section 4.2(a),
p. 92 (``E*``, unnumbered).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

__all__ = ["DC", "Body", "Contact"]

# The AxisForge dataclass style (frozen, identity equality, keyword-only), defined here so that
# core does not depend on an AxisForge private module.
DC = dict(frozen=True, eq=False, kw_only=True)


@dataclass(**DC)
class Body:
    """One elastic body near its point of first contact.

    The surface is described by its two principal radii of curvature at the contact point, in
    the body's own frame (x_b, y_b). One exact shape covers every rolling-contact body: a circle
    of radius ``r_y`` (across) swept about an axis at distance ``r_x`` from the apex. This gives
    the sphere (``r_x = r_y``), the cylinder (``r_y = inf``), a raceway torus convex or concave in
    either direction, and a thrust raceway flat along rolling (``r_x = inf``). Near the apex the
    sag is ``x_b^2/(2 r_x) + y_b^2/(2 r_y)``, the paraboloid of Hertz (Johnson 1985, Eq. (4.2a)).

    How exact the shape is depends on the body it stands for: the torus is the exact raceway at
    a contact angle alpha = 0; at alpha != 0 the rolling radius ``r_x`` re-evaluated at alpha
    makes it exact to second order only (the true raceway is swept about the bearing axis, not
    about an axis at ``r_x``); a cylinder for a gear tooth approximates the involute by its
    radius of curvature at the contact point.

    Parameters
    ----------
    r : float or (float, float)
        Signed principal radii ``(r_x, r_y)`` [m] in the body frame; a scalar means ``(r, r)``.
        Convex > 0, concave < 0 (Johnson 1985 p. 85: a concave surface has a negative radius),
        flat ``math.inf``. Radii of curvature, not ellipsoid semi-axes.
    E : float
        Young's modulus [Pa].
    nu : float
        Poisson's ratio [-].

    Raises
    ------
    ValueError
        If a radius is zero, NaN or ``-inf`` (flat is ``+inf``), if ``E`` is not positive and
        finite, or if ``nu`` is outside (0, 0.5).
    """

    r: tuple[float, float]
    E: float
    nu: float

    # OPEN SB-02: body checks in Body.__post_init__, contact checks in Contact.__post_init__ (alternative: all in Contact)
    def __post_init__(self) -> None:
        r = (self.r, self.r) if np.ndim(self.r) == 0 else tuple(self.r)
        if len(r) != 2:
            raise ValueError(f"Body: r must be a number or (r_x, r_y), got {self.r!r}")
        r = tuple(float(v) for v in r)
        # a flat surface is +inf; -inf would be a "concave flat", the same surface with a sign that
        # the curvature sum would read as 0 anyway, so it is refused as a likely sign slip
        if any(v == 0.0 or math.isnan(v) or v == -math.inf for v in r):
            raise ValueError(f"Body: radii must be non-zero numbers, +inf for flat, got r = {r}")
        if not (self.E > 0 and math.isfinite(self.E)):
            raise ValueError(f"Body: E must be positive and finite [Pa], got {self.E}")
        # nu = 0.5 (incompressible) is excluded by policy: the bridge models steel-like solids, and
        # SlipPY Elastic does not check it
        if not 0.0 < self.nu < 0.5:
            raise ValueError(f"Body: nu must be in (0, 0.5), got {self.nu}")
        object.__setattr__(self, "r", r)

    def profile(self, X: np.ndarray, Y: np.ndarray, angle: float = 0.0) -> np.ndarray:
        """Exact surface height of the body on bridge coordinates, in the SlipPY sign (apex 0).

        Parameters
        ----------
        X, Y : ndarray
            Bridge coordinates [m] of the nodes (x = rolling, y across), same shape.
        angle : float, optional
            Angle of the body x axis from the bridge x axis [rad], positive from x toward y.
            Default 0 (body frame = bridge frame).

        Returns
        -------
        ndarray
            ``-sag`` [m], same shape as ``X``: 0 at the apex, negative away from it for a convex
            body (the surface falls away from the contact plane), positive for a concave one.

        Raises
        ------
        ValueError
            If a node lies beyond the body: outside the generating circle (``|y_b| >= |r_y|``),
            outside the swept circle (``|x_b| >= |rho|``), or past the axis of revolution (the
            local rolling radius ``rho`` changes sign). The grid is too large for the body.
        """
        X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
        c, s = math.cos(angle), math.sin(angle)
        # bridge -> body frame: projections on the body axes (x_b at `angle` from bridge x)
        x = X * c + Y * s
        y = -X * s + Y * c
        r_x, r_y = self.r
        with np.errstate(invalid="ignore"):                  # NaN beyond the body, caught below
            # sag of the generating circle across rolling; copysign keeps the concave branch for r_y < 0
            sag_y = np.zeros_like(y) if math.isinf(r_y) else r_y - math.copysign(1.0, r_y) * np.sqrt(r_y**2 - y**2)
            rho = r_x - sag_y                                 # local radius in the rolling direction
            z = -sag_y if math.isinf(r_x) else -(r_x - math.copysign(1.0, r_x) * np.sqrt(rho**2 - x**2))
        beyond = (np.isnan(z).any()
                  or (math.isfinite(r_y) and np.abs(y).max() >= abs(r_y))
                  # rho * r_x <= 0: far from the contact the local radius has crossed the axis of revolution
                  or (math.isfinite(r_x) and (np.any(np.abs(x) >= np.abs(rho)) or np.any(rho * r_x <= 0))))
        if beyond:
            raise ValueError(f"the grid reaches beyond the body of radii {self.r}: reduce span_x/span_y")
        return z


@dataclass(**DC)
class Contact:
    """Abstract normal contact of two elastic bodies; a machine element subclasses it.

    Body 1 has its frame on the bridge axes; body 2 is turned by ``angle``. Near the contact the
    gap between the undeformed surfaces is ``h = (x, y) K (x, y)^T / 2`` with ``K = K1 + K2`` the
    sum of the curvature tensors of the bodies on the bridge axes (Johnson 1985, section 4.1,
    Eqs. (4.1)-(4.3), p. 84-85; convex curvature positive, concave negative). The principal values
    of ``K`` are the principal relative curvatures ``kappa_max >= kappa_min``: ``kappa_max =
    1/R'' = 2B`` and ``kappa_min = 1/R' = 2A`` of Johnson Eqs. (4.3)-(4.5), p. 85 (his ``R'`` is
    the larger radius, ``A < B``; alpha there is the angle between the body x axes; derivation in
    Appendix 2, p. 425-426). ``psi`` is the direction of ``kappa_max``. In SlipPY terms,
    ``kappa_max, kappa_min = 1/relative_radii[0], 1/relative_radii[1]`` of ``hertz_full``, ``psi``
    its ``alpha`` (mod pi), and ``E_star`` its ``e_star`` (``E*``, not ``E'``). The contact ellipse
    has its minor semi-axis ``b`` along ``psi`` and its major semi-axis ``a`` across it (Johnson
    4.2(b), p. 98: ``a > b``, ``a`` along the smaller curvature). For a line contact the bridge
    keeps ``b`` for the half-width (Johnson calls it ``a``, 4.2(c), p. 101) and ``a = inf``.

    Assumptions: smooth, non-conforming surfaces, contact size small against the relative radii
    and the bodies, each body an elastic half-space, frictionless normal load (Johnson 1985,
    section 4.2, list (i)-(iv), p. 91-92).

    Parameters
    ----------
    body1, body2 : Body
        The two bodies; body 1 is the rolling element or pinion by convention of the subclasses.
    load : float
        Normal load: [N] for a point contact, per unit length [N/m] for a line contact.
    angle : float, optional
        Angle of the x axis of body 2 from the x axis of body 1 (= bridge x) [rad], positive from
        x toward y, in (-pi/2, pi/2]. The same meaning as ``angle`` of SlipPY ``hertz_full``
        (angle between the x axes of the two bodies); the sense is checked against
        ``hertz_full`` alpha in the Hertz tests. Default 0.
    label : str, optional
        Identifier used in error messages, file names and records.

    Attributes
    ----------
    line : bool
        Class attribute set by each subclass: True for a line contact (load per unit length, the
        line along bridge y), False for a point contact.
    kappa_max, kappa_min : float
        Principal relative curvatures [1/m], ``kappa_max > 0``, ``kappa_min >= 0`` (0 for a line).
    psi : float
        Direction of ``kappa_max`` (minor semi-axis ``b``) from bridge x [rad], in (-pi/2, pi/2],
        positive from x toward y. No meaning for a circular contact (``kappa_max = kappa_min``),
        where every direction is principal.
    E_star : float
        Contact modulus ``1/E* = (1 - nu1^2)/E1 + (1 - nu2^2)/E2`` [Pa] (Johnson 1985, section
        4.2(a), p. 92).

    Raises
    ------
    TypeError
        If ``Contact`` itself is created (abstract: a subclass sets ``line``).
    ValueError
        If the load is not positive and finite; if ``angle`` is outside (-pi/2, pi/2]; if the
        bodies do not form a Hertz contact (``kappa_max <= 0`` or ``kappa_min < 0``); if a point
        contact has ``kappa_min = 0``; if a line contact has ``kappa_min > 0`` or its line is not
        along bridge y (``psi != 0``); if a point ellipse is too long for SlipPY
        (``0 < kappa_min / kappa_max < 1.3e-5``: neither a point nor a line contact applies).
    """

    line: ClassVar[bool]          # declared, not set: each subclass sets it (point or line)

    body1: Body
    body2: Body
    load: float
    angle: float = 0.0
    label: str = ""
    kappa_max: float = field(init=False)
    kappa_min: float = field(init=False)
    psi: float = field(init=False)
    E_star: float = field(init=False)

    def __post_init__(self) -> None:
        if not hasattr(type(self), "line"):
            raise TypeError("Contact is abstract: create a subclass that sets the class attribute `line`")
        name = f"Contact {self.label!r}"
        if not (self.load > 0 and math.isfinite(self.load)):
            raise ValueError(f"{name}: load must be positive and finite, got {self.load}")
        # (-pi/2, pi/2] covers every relative orientation of two bodies once (a body turned by pi
        # is the same surface). SlipPY hertz_full reduces its angle to [-pi/2, pi/2) instead:
        # the two ends are the same orientation, so the ranges are physically the same.
        if not -math.pi / 2 < self.angle <= math.pi / 2:
            raise ValueError(f"{name}: angle must be in (-pi/2, pi/2] rad, got {self.angle}")

        # Curvature tensor of each body on the bridge axes, K = R diag(k_x, k_y) R^T with R the
        # rotation by the body angle (body 1 at 0, body 2 at `angle`); their sum is the gap.
        Kxx = Kyy = Kxy = 0.0
        for body, t in ((self.body1, 0.0), (self.body2, self.angle)):
            k_x, k_y = (0.0 if math.isinf(r) else 1.0 / r for r in body.r)      # flat: no curvature
            c, s = math.cos(t), math.sin(t)
            Kxx += k_x * c**2 + k_y * s**2
            Kyy += k_x * s**2 + k_y * c**2
            Kxy += (k_x - k_y) * c * s
        # principal values by Mohr's circle; psi = atan2(.)/2 lies in (-pi/2, pi/2]
        m, d = (Kxx + Kyy) / 2, (Kxx - Kyy) / 2
        rad = math.hypot(d, Kxy)
        kappa_max, kappa_min, psi = m + rad, m - rad, math.atan2(Kxy, d) / 2
        if not kappa_max > 0:
            raise ValueError(f"{name}: kappa_max = {kappa_max} 1/m <= 0: the bodies do not touch at one point "
                             "or line (conforming or both flat), not a Hertz contact")
        # A line computed as m - rad, or turned by pi/2 (cos(pi/2) = 6e-17), keeps a rounding residue
        # of order 1e-16 kappa_max; 1e-12 is far above it and far below the smallest ellipse SlipPY
        # solves (1.3e-5), so no real ellipse is snapped.
        if abs(kappa_min) <= 1e-12 * kappa_max:
            kappa_min = 0.0
        # OPEN SB-14: psi snapped when |Kxy| <= 1e-12 kappa_max
        if abs(Kxy) <= 1e-12 * kappa_max:
            psi = 0.0 if d >= 0 else math.pi / 2                # principal axes on the bridge axes
        if kappa_min < 0:
            raise ValueError(f"{name}: kappa_min = {kappa_min} 1/m < 0: the gap is a saddle, not a Hertz contact")

        # OPEN SB-03: point limit kappa_min/kappa_max >= 1.3e-5 (hertz_full root finding fails below ~1.267e-5, line above 1e5; D-0034 says b/a >= 1e-4)
        if 0 < kappa_min < 1.3e-5 * kappa_max:
            raise ValueError(f"{name}: kappa_min/kappa_max = {kappa_min / kappa_max:.3g} is in (0, 1.3e-5): neither "
                             "a point contact (SlipPY hertz_full fails on so long an ellipse) nor a line contact "
                             "(kappa_min is not 0) applies")
        if self.line:
            if kappa_min > 0:
                raise ValueError(f"{name}: a line contact needs kappa_min = 0, got {kappa_min} 1/m "
                                 "(the bodies form an ellipse: use a point contact)")
            # the bridge takes the line along y (the roller axis = grid axis 1, periodic there)
            if psi != 0.0:
                raise ValueError(f"{name}: the line of contact must lie along bridge y (psi = 0), got psi = "
                                 f"{psi} rad")
        elif kappa_min == 0:
            raise ValueError(f"{name}: a point contact needs kappa_min > 0, got 0 (the bodies touch along "
                             "a line: use a line contact)")

        b1, b2 = self.body1, self.body2
        E_star = 1.0 / ((1.0 - b1.nu**2) / b1.E + (1.0 - b2.nu**2) / b2.E)
        for key, value in (("kappa_max", kappa_max), ("kappa_min", kappa_min), ("psi", psi), ("E_star", E_star)):
            object.__setattr__(self, key, value)

    def profiles(self, grid) -> tuple[np.ndarray, np.ndarray]:
        """Exact profiles of the two bodies on a grid, body 2 turned by ``angle``.

        Parameters
        ----------
        grid : object with ``X``, ``Y``
            Node coordinates [m] on the bridge axes (``mesh.grid.Grid`` or any object with
            these arrays).

        Returns
        -------
        z1, z2 : ndarray
            ``-sag`` [m] of body 1 and body 2 (see :meth:`Body.profile`).
        """
        return self.body1.profile(grid.X, grid.Y), self.body2.profile(grid.X, grid.Y, self.angle)
