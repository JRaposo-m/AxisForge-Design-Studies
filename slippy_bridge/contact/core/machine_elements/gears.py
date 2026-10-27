"""Cylindrical gears: AxisForge cylindrical pair -> contacts of two involute flanks along the path.

A gear contact is the contact of two flanks at one point of the path of contact. As for the
bearings (D-0034 item 10) each AxisForge gear family has its own contact class with its own
bodies, and the caller picks the class; no dispatcher:

* :class:`CylindricalGearContact` <- AxisForge ``CylindricalFamily``, external and internal,
  spur and helical; bodies: two :class:`InvoluteFlank`, the true flanks with their ends.

The contact is a line contact: the line lies along the contact line (bridge y), the profile
direction is the rolling direction (bridge x, toward the tip of gear 1). It is quasi-steady (each
point solved as if it lasted for ever), the load sharing between the tooth pairs in contact is
AxisForge's rigid model (one load per unit length on all contact lines), and a helical contact is
a slice (see :class:`InvoluteFlank`). The equivalent cylinders (``SweptCircleBody``) are kept for
comparison only, through a plain ``Contact`` subclass.

The input of a contact is the diameter ``dy1`` of the flank of gear 1 at the contact point
(D-0034 item 9). At a given ``dy1`` the radii, the position on the path, the load share and the
load per unit length do not depend on which gear drives; the driver says which diameters are the
named points A..E and the sense of the entrainment. AxisForge is read here only: every value at a
point from one call of ``CylindricalFamily.pair_values_at``, the load from ``load_sharing``. Units
are converted here, once: mm -> m (``_MM``), N/mm -> N/m, MPa -> Pa (``_MPA``).

References
----------
ISO 21771:2007, 4.3 (involute helicoids), 4.3.8 Eq. (17) (radius of curvature of the involute at
a diameter), 5.4.5.3 Eq. (87) (the two radii add up to T1T2), 5.6 (sliding), through AxisForge
``CylindricalFamily.pair_values_at``, ``path_points``, ``curvature_radii`` and ``surface_speeds``;
AxisForge ``axisforge.solvers.machine_elements.gears.contact.load_sharing``. Johnson (1985),
section 4.2(c) (line contact).
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from axisforge.core.machine_elements.gears.cylindrical import CylindricalFamily
from axisforge.core.materials import available_materials, get_material
from axisforge.solvers.machine_elements.gears.contact import load_sharing

from ..contact import DC, Body, Contact
from ..submodels import Kinematics
from . import _MM, _MPA

__all__ = ["InvoluteFlank", "GearContact", "CylindricalGearContact"]

# the flank-end distances of a gear contact: (field name, body, attribute of the body)
_ENDS = (("d_end_tip_1_m", "body1", "d_end_tip_m"), ("d_end_root_1_m", "body1", "d_end_root_m"),
         ("d_end_tip_2_m", "body2", "d_end_tip_m"), ("d_end_root_2_m", "body2", "d_end_root_m"))

# OPEN R-024: rho2_n of a ring gear is concave < 0 (probe at C); José may close it.
# OPEN R-027: helical pairs are mid-face slices only; EHL on beta != 0 to be refused.
# OPEN R-118: E, nu of the FZG gears (20MnCr5): no held source; the case gives them explicitly.


@dataclass(**DC)
class InvoluteFlank(Body):
    """One involute flank of one gear at the contact point: the true tooth surface as a contact body.

    The flank is an involute helicoid of base radius ``rb`` and base helix angle ``beta_b``: the
    surface swept by a straight generator, inclined at ``beta_b`` to the axis in the plane tangent
    to the base cylinder; every transverse section is the involute of the base circle (ISO
    21771:2007, 4.3.1, Figure 7, p. 17; 4.3.10, p. 21). The contact line is that generator
    through the contact point.

    Body frame (see :class:`Body`), origin at the contact point:

    * z_b the outward normal of the flank (toward the mating flank): the line of action for a spur
      gear, in the plane of action at ``beta_b`` to it for a helical one;
    * y_b along the contact line (the generator): the flank is straight along it, so the profile
      does not depend on y_b;
    * x_b along the profile in the plane normal to the contact line, the rolling direction. It is
      the tangent of the transverse involute at the point, perpendicular to the plane of action.

    The profile is not even in x_b (the curvature of an involute grows from its base circle, and
    the flank ends at its tip and at its root). +x_b points toward the tip of this flank when
    ``toward_tip`` is True, toward its root otherwise. In a gear contact (bridge x toward the tip
    of gear 1, the side of g_y > 0) gear 1 has ``toward_tip = True`` and gear 2, external or ring,
    ``toward_tip = False``: moving along the common tangent toward the tip of gear 1 moves toward
    the root of gear 2 in both kinds of pair.

    Radii: ``r = (rho_n, inf)`` with ``rho_n = rho_t / cos(beta_b)`` (Meusnier: the transverse
    involute is an oblique section at ``beta_b`` to the normal section; AxisForge
    ``CylindricalFamily.curvature_radii``, derived there), signed as AxisForge: convex external
    flank > 0, concave ring flank < 0 (ISO 21771:2007, 5.4.5.3 NOTE, p. 36).

    Profile: the exact section of the helicoid by the plane through the contact point normal to
    the contact line, parametric in the transverse roll angle ``u`` (ISO 21771:2007, 4.3.7 Eq. (16);
    4.3.8 Eq. (17): the radius of curvature of the involute at roll angle ``u`` is ``rb u``, the
    length of roll). Derived here (no library source for the section): with ``t = |rho_t| / rb``
    the roll angle of the point, ``phi = u - t``, ``c, s = cos(beta_b), sin(beta_b)``, the
    transverse involute point at ``u`` has, along the transverse tangent ``T`` and the transverse
    normal ``n`` of the point (``n`` from the base-tangent point outward),
    ``D_T = rb (cos(phi) + u sin(phi) - 1)``, ``D_n = rb (u cos(phi) - sin(phi) - t)``. The
    generator through it (direction ``-s n(u) + c e_axis``) meets the normal plane after
    ``lam = s D_n / (s^2 cos(phi) + c^2)``, at::

        x = D_T - lam s sin(phi),    z = c D_n + lam s c (1 - cos(phi))

    on the axes of the involute (x along increasing roll angle, z away from the base-tangent
    point), and the section point lies on the transverse involute of roll angle ``u - lam s / rb``.
    Spur (``beta_b = 0``): ``lam = 0``, the transverse involute in its tangent frame, closed form.
    The node abscissa is inverted for ``u`` by Newton's method with the spur derivative (exact for
    spur, linear convergence for helical) on ``(u - u_c)^2``, ``u_c`` the roll angle whose section
    point lies on the base circle (``u_c = 0`` for spur), where ``dx/du = 0``: well conditioned at
    the base circle, to ``NEWTON_TOL rb`` on x. Second order: ``z = -x^2 cos(beta_b) / (2 |rho_t|) =
    -x^2 / (2 |rho_n|)``, the radius ``r``. On the body axes ``x_b = sigma x``, ``profile =
    sign(rho_t) z``, ``sigma = sign(rho_t) (+1 if toward_tip else -1)`` (the roll angle grows
    toward the tip of an external gear and toward the root of a ring).

    A helical contact is a SLICE: along a helical contact line the transverse roll angle, so
    ``rho_n``, varies linearly (the line crosses the flank from root to tip); this body is the
    flank of the contact point, straight along y_b with the ``rho_n`` of that point, and the
    bridge solves it as an infinite strip periodic along y_b, as each lamina of a roller (Q-13;
    the slices of D-0035 L3). The ends of a helical flank are not along y_b either: the tip and
    root cylinders cut the plane of action along lines parallel to the gear axis, at ``beta_b`` to
    the contact line, so the distance to an end changes along y_b by about ``tan(beta_b)`` per unit
    length (0.25 mm/mm for H501); the slice keeps the distances of the point.

    Flank ends (José 2026-10-27, option (b)): the flank exists between its tip (the tip circle;
    for a ring, its inner tip circle) and its root end ``r_root``. For an external gear the root
    end the bridge gives is the start of the involute, the base circle: an upper bound of the
    flank, whose involute ends at the root form circle ``d_Ff >= d_b`` (ISO 21771:2007, 7.6), with
    the fillet below (the ideal ``CylindricalFamily`` stores no ``d_Ff``;
    ``ManufacturedCylindricalFamily`` does). The tip is sharp (no chamfer, no tip relief, ISO
    21771 clause 6): a contact at an end (``d_end = 0``, the points A and E of the path) is an edge
    contact whose BEM pressure peak grows with grid refinement. Past an end there is no material:
    :meth:`profile` returns ``SAG_PAST_END`` below the lowest flank node there, far beyond any
    approach, so those nodes never touch. :attr:`d_end_tip_m`, :attr:`d_end_root_m` are the
    distances along x_b from the point to the two ends (the abscissae of the end points of the
    section, projected on the tangent plane as every profile node is, not arc lengths).

    Parameters
    ----------
    rb : float
        Base radius [m], > 0 (AxisForge ``Gear.rb``, a magnitude).
    rho_t : float
        Transverse radius of curvature at the point [m], signed: > 0 external, < 0 ring
        (AxisForge ``pair_values_at`` ``rho1_t_mm`` / ``rho2_t_mm``).
    beta_b : float
        Base helix angle [rad], ``|beta_b| < pi/2`` (AxisForge ``Gear.betab``); 0 for spur.
    r_tip, r_root : float
        Radii of the two ends of the flank [m], >= ``rb``: the tip radius (AxisForge ``Gear.ra``)
        and the root end (``rb`` for an external gear).
    toward_tip : bool
        True if +x_b points toward the tip of this flank, False toward its root.

    Other fields as :class:`Body` (``E``, ``nu``).

    Attributes
    ----------
    d_end_tip_m, d_end_root_m : float
        Distances along x_b from the contact point to the tip end and to the root end [m], >= 0.

    Raises
    ------
    ValueError
        If ``rb`` is not positive and finite, ``rho_t`` zero or not finite, ``|beta_b| >= pi/2``,
        an end radius below ``rb``, the ends in the wrong order for the kind of flank (a ring has
        its tip inside, ``r_tip < r_root``), or the point off the flank between its ends; and the
        checks of :class:`Body`.
    """

    # OPEN SB-67: a helical slice keeps the end distances of the point; the ends are oblique to y_b (~tan(beta_b) per unit length), wrong once an end lies in the grid
    # OPEN SB-65: past a flank end the profile is 1 mm below the lowest flank node (approaches are of um; sharp tip, edge singularity at A/E; alternative: a smooth fall-off or a tip chamfer)
    SAG_PAST_END: ClassVar[float] = 1e-3
    # Newton's method of the profile and fixed point of the ends: tolerance on x relative to rb [-] and iteration cap
    NEWTON_TOL: ClassVar[float] = 1e-14
    NEWTON_MAX_IT: ClassVar[int] = 60
    # a point on an end (A, E of the path) up to rounding: tolerance on its distance relative to rb [-]
    END_TOL: ClassVar[float] = 1e-9

    rb: float
    rho_t: float
    beta_b: float
    r_tip: float
    r_root: float
    toward_tip: bool
    d_end_tip_m: float = field(init=False)
    d_end_root_m: float = field(init=False)
    _rolls: tuple = field(init=False, repr=False)    # roll angles u_c (base circle), u_lo, u_hi (the ends)

    def __post_init__(self) -> None:
        name = type(self).__name__
        if not (self.rb > 0 and math.isfinite(self.rb)):
            raise ValueError(f"{name}: rb must be positive and finite [m], got {self.rb}")
        if not (self.rho_t != 0 and math.isfinite(self.rho_t)):
            raise ValueError(f"{name}: rho_t must be non-zero and finite [m], got {self.rho_t}")
        if not abs(self.beta_b) < math.pi / 2:
            raise ValueError(f"{name}: |beta_b| must be < pi/2, got {self.beta_b}")
        if not (self.r_tip >= self.rb and self.r_root >= self.rb):
            raise ValueError(f"{name}: the end radii must be >= rb = {self.rb} m, got r_tip = {self.r_tip}, "
                             f"r_root = {self.r_root}")
        external = self.rho_t > 0
        if (self.r_tip > self.r_root) != external:
            raise ValueError(f"{name}: an {'external' if external else 'internal (ring)'} flank needs r_tip "
                             f"{'>' if external else '<'} r_root, got r_tip = {self.r_tip}, r_root = {self.r_root}")
        # own axes: x along increasing roll angle (radius), so the external tip and the ring root are at x > 0
        u_tip, u_root = (self._roll_at_end(r) for r in (self.r_tip, self.r_root))
        x_tip, x_root = (float(self._section(u)[0]) for u in (u_tip, u_root))
        d_tip, d_root = (x_tip, -x_root) if external else (-x_tip, x_root)
        tol = self.END_TOL * self.rb
        if not (d_tip >= -tol and d_root >= -tol):
            raise ValueError(f"{name}: the point (rho_t = {self.rho_t} m) is off the flank between its ends "
                             f"(r_tip = {self.r_tip}, r_root = {self.r_root} m)")
        object.__setattr__(self, "d_end_tip_m", max(d_tip, 0.0))
        object.__setattr__(self, "d_end_root_m", max(d_root, 0.0))
        object.__setattr__(self, "_rolls", (self._roll_at_end(self.rb), *sorted((u_tip, u_root))))
        super().__post_init__()

    @property
    def r(self) -> tuple[float, float]:
        return (self.rho_t / math.cos(self.beta_b), math.inf)

    def _section(self, u):
        """``(x, z, u_t)`` of the section at roll angle ``u`` on the involute axes (see the class): abscissa,
        height [m] and the transverse roll angle of the section point [rad]."""
        rb, t = self.rb, abs(self.rho_t) / self.rb
        c, s = math.cos(self.beta_b), abs(math.sin(self.beta_b))   # the hand only mirrors the section
        phi = u - t
        cp, sp = np.cos(phi), np.sin(phi)
        D_T = rb * (cp + u * sp - 1.0)
        D_n = rb * (u * cp - sp - t)
        lam = s * D_n / (s * s * cp + c * c)                         # along the generator to the normal plane
        return D_T - lam * s * sp, c * D_n + lam * s * c * (1.0 - cp), u - lam * s / rb

    def _roll_at_end(self, r_end: float) -> float:
        """Roll angle ``u`` whose section point lies on the circle of radius ``r_end`` [rad] (fixed point)."""
        t_end = math.sqrt(max(r_end**2 - self.rb**2, 0.0)) / self.rb     # ISO 21771 Eq. (16)-(17): rb u = rho
        u = t_end
        for _ in range(self.NEWTON_MAX_IT):
            u_new = u + t_end - float(self._section(u)[2])               # u_t(u) = t_end; d u_t / d u ~ 1
            if abs(u_new - u) <= self.NEWTON_TOL:
                return u_new
            u = u_new
        raise ValueError(f"InvoluteFlank: no convergence at the end r = {r_end} m")

    def profile(self, X, Y, angle=0.0):
        """Exact height of the flank at each node (see :meth:`Body.profile` and the class).

        Independent of y_b. Nodes past an end of the flank get ``SAG_PAST_END`` below the lowest
        node on the flank (no material); nothing is refused.

        Raises
        ------
        ValueError
            If Newton's method does not converge (not met on a flank).
        """
        x_b, _ = self._body_frame(X, Y, angle)
        external = self.rho_t > 0
        sigma = (1.0 if external else -1.0) * (1.0 if self.toward_tip else -1.0)
        x = sigma * x_b                                                   # on the involute axes
        # the ends on the involute axes: x_lo (root of an external gear, tip of a ring) .. x_hi
        x_lo, x_hi = (-self.d_end_root_m, self.d_end_tip_m) if external else (-self.d_end_tip_m, self.d_end_root_m)
        on = (x >= x_lo) & (x <= x_hi)
        xs, inverse = np.unique(x[on], return_inverse=True)              # the profile depends on x_b only
        rb, t = self.rb, abs(self.rho_t) / self.rb
        u_c, u_lo, u_hi = self._rolls
        lo, hi = (u_lo - u_c) ** 2, (u_hi - u_c) ** 2
        v = np.clip((t - u_c) ** 2 + 2.0 * xs / rb, lo, hi)              # start: x ~ rb (u^2 - t^2) / 2
        for _ in range(self.NEWTON_MAX_IT):
            u = u_c + np.sqrt(v)
            res = self._section(u)[0] - xs
            if np.all(np.abs(res) <= self.NEWTON_TOL * rb):
                break
            # dx/dv = rb cos(phi) / 2 for the spur section (the helical term is of order s^2 phi^2 smaller);
            # cos(phi) > 0 on real gears (|phi| <= 0.41 rad on FZG C), unguarded for |phi| -> pi/2
            v = np.clip(v - res / (0.5 * rb * np.cos(u - t)), lo, hi)
        else:
            raise ValueError(f"InvoluteFlank: Newton's method did not converge (max |x residual| "
                             f"{np.abs(res).max():.3g} m)")
        z_on = (1.0 if external else -1.0) * self._section(u)[1][inverse]
        # past an end: SAG_PAST_END below the lowest flank node, so no material rises above the flank
        z = np.full(x.shape, min(0.0, z_on.min(initial=0.0)) - self.SAG_PAST_END)
        z[on] = z_on
        return z


@dataclass(**DC)
class GearContact(Contact):
    """Contact of the flanks of gear 1 (body 1) and gear 2 (body 2) at one point of the path.

    Abstract like :class:`Contact`: each family class (:class:`CylindricalGearContact`) sets
    ``line`` and ``family``. Field names and units as AxisForge (``pair_values_at``,
    ``path_points``, ``load_sharing``).

    Parameters
    ----------
    point : str
        Label of the point: "A".."E" or a label given by the caller (no ``:``, it goes into file
        names).
    dy1_mm, dy2_mm : float
        Diameter of the contact point on the flank of gear 1 and of gear 2 [mm].
    g_y_mm : float
        Position on the path from the pitch point C, positive toward the tip of gear 1 [mm].
    share : float
        Share of the normal force carried by this tooth pair [-]; nan when the load is given.
    l_mm : float
        Length of the contact line of this tooth pair [mm]; nan when the load is given.
    l_sum_mm : float
        Total length of the contact lines of all tooth pairs in contact [mm]; ``load = F_bn /
        l_sum``; nan when the load is given.
    n_pairs : int or float
        Number of tooth pairs in contact [-]; nan when the load is given.
    driver : {"gear1", "gear2"}
        The driving gear (fixes which diameters are A..E and the sense of the entrainment; not the
        geometry at ``dy1``).
    z2 : int
        Tooth number of gear 2, signed (ISO 21771: negative for an internal ring), for the sign of
        the flank of gear 2.
    material1, material2 : str
        Source of E and nu of gear 1 and gear 2: the AxisForge ``material_id``, or ``"given"`` /
        ``"given:<material_id>"`` for values given to ``from_pair``.

    Other fields as :class:`Contact`. Bridge y lies along the contact line: the face width for a
    spur pair, inclined at beta_b in the plane of action for a helical one; ``load`` is the load
    per unit length of that line [N/m].

    Raises
    ------
    ValueError
        If ``driver`` is not "gear1"/"gear2"; and every check of :class:`Contact`.
    """

    family: ClassVar[type]        # declared, not set: each family class sets its AxisForge family

    point: str
    dy1_mm: float
    dy2_mm: float
    g_y_mm: float
    share: float
    l_mm: float
    l_sum_mm: float
    n_pairs: int | float
    driver: str
    z2: int                       # OPEN SB-11: z2 stored on the contact for the sign check of rho2
    material1: str                # OPEN SB-12: E, nu source recorded on the contact
    material2: str

    def __post_init__(self) -> None:
        if self.driver not in ("gear1", "gear2"):
            raise ValueError(f"{type(self).__name__} {self.label!r}: driver must be 'gear1' or 'gear2', "
                             f"got {self.driver!r}")
        super().__post_init__()


@dataclass(**DC)
class CylindricalGearContact(GearContact):
    """Line contact of two involute flanks of an AxisForge ``CylindricalFamily`` pair at one point of the path.

    External and internal pairs (the ring's sign in its radii, as AxisForge), spur and helical (a
    helical contact is a slice: see :class:`InvoluteFlank`). Body 1 is the :class:`InvoluteFlank`
    of gear 1 (+x_b toward its tip), body 2 that of gear 2 (+x_b toward its root): bridge +x points
    toward the tip of gear 1, the side of g_y > 0; bridge y lies along the contact line; ``load`` is
    the load per unit length of that line [N/m]. Build the contacts of a pair with
    :meth:`from_pair`. The equivalent cylinders (``SweptCircleBody``) are not accepted as bodies
    here: compare with them through a plain :class:`Contact` subclass.

    Parameters
    ----------
    zeta1, zeta2 : float
        Specific sliding of flank 1 and flank 2 [-] (AxisForge ``pair_values_at``, ISO 21771:2007
        5.6.3 Eq. (114)-(115)).
    Kg : float
        Sliding factor [-] (``pair_values_at``, 5.6.2 Eq. (111)).
    kinematics : Kinematics or None, optional
        Surface speeds at the point with the meanings and signs of AxisForge ``surface_speeds``
        (``u1``, ``u2`` > 0 the rolling speeds across the contact line, ``vg = u1 - u2``,
        ``ue = (u1 + u2)/2``), when a speed is given to :meth:`from_pair`; None otherwise.

    Other fields as :class:`GearContact`.

    Attributes
    ----------
    beta_b_rad : float
        Base helix angle of the two flanks [rad] (AxisForge ``gear1.betab``).
    d_end_tip_1_m, d_end_root_1_m, d_end_tip_2_m, d_end_root_2_m : float
        Distances along bridge x from the point to the ends of the two flanks [m] (the bodies'
        ``d_end_tip_m``, ``d_end_root_m``): the tip of gear 1 and the root end of gear 2 lie at
        +x, the root end of gear 1 and the tip of gear 2 at -x. Divided by the Hertz half-width
        they are in the Hertz reference (``d_end_*_over_b``); below 1 that reference is not valid
        (``valid`` False, the reason recorded) and a solve still runs. The contact then reaches a
        sharp flank end: the half-space BEM sees a punch-like edge (Johnson 1985, 2.8, Eq. (2.61);
        a real tooth corner is a wedge, more compliant, and tip relief and tooth deflection are not
        modelled), so its p0, b, approach and ``err_*`` are numbers of the mesh, not of the gear (C40
        at K9, 0.30 b from the tip of gear 2: p0 3.08 / 4.16 / 5.87 GPa at h, h/2, h/4); b is half
        the truncated loaded segment. Read them only when ``hertz_valid`` is True.
    entrainment_sign_x : int
        Sense of the entrainment along bridge x, from the driver: -1 when gear 1 drives (the
        contact point moves along the line of action; over flank 1 it moves toward its tip and over
        flank 2 toward its root, both +x, so both surfaces move toward -x relative to it), +1 when
        gear 2 drives, at every point of the path, external or internal. In bridge x the speeds of
        ``kinematics`` are ``entrainment_sign_x`` times AxisForge's (``ue`` and ``vg`` alike).
        SlipPY's Reynolds solver needs the entrainment along +x (inlet at array index 0) and a
        positive speed: with gear 1 driving the arrays would have to be flipped along axis 0 (the
        contact cannot be mirrored: body 1 points toward its tip); no Reynolds step reads it yet.

    Raises
    ------
    ValueError
        If a body is not an :class:`InvoluteFlank`, body 1 not toward its tip or body 2 not
        toward its root; if the two flanks have different ``beta_b``; if ``body1.rho_t <= 0``; if
        the sign of ``body2.rho_t`` is not that of ``z2`` (convex external gear 2, concave ring;
        beyond the base-tangent point T2 of an external pair there is no involute, SB-59); and
        every check of :class:`GearContact`.
    """

    line = True
    family = CylindricalFamily

    zeta1: float
    zeta2: float
    Kg: float
    kinematics: Kinematics | None = None
    beta_b_rad: float = field(init=False)
    d_end_tip_1_m: float = field(init=False)
    d_end_root_1_m: float = field(init=False)
    d_end_tip_2_m: float = field(init=False)
    d_end_root_2_m: float = field(init=False)
    # OPEN SB-68: bridge +x toward the tip of gear 1 is against the entrainment when gear 1 drives (entrainment_sign_x -1)
    entrainment_sign_x: int = field(init=False)

    def __post_init__(self) -> None:
        name = f"{type(self).__name__} {self.label!r}"
        b1, b2 = self.body1, self.body2
        if not (isinstance(b1, InvoluteFlank) and isinstance(b2, InvoluteFlank) and b1.toward_tip
                and not b2.toward_tip):
            raise ValueError(f"{name}: body1 must be the InvoluteFlank of gear 1 toward its tip and body2 that of "
                             "gear 2 toward its root")
        if b1.beta_b != b2.beta_b:
            raise ValueError(f"{name}: the two flanks need the same beta_b, got {b1.beta_b}, {b2.beta_b}")
        if not b1.rho_t > 0:
            raise ValueError(f"{name}: rho1_t must be > 0, got {b1.rho_t} m")
        if np.sign(b2.rho_t) != np.sign(self.z2):
            raise ValueError(f"{name}: rho2_t = {b2.rho_t} m has not the sign of z2 = {self.z2} (convex external "
                             "gear 2, concave ring; beyond T2 of an external pair there is no involute, SB-59)")
        super().__post_init__()
        for key, value in (("beta_b_rad", b1.beta_b), *self.end_distances().items(),
                           ("entrainment_sign_x", -1 if self.driver == "gear1" else 1)):
            object.__setattr__(self, key, value)

    def end_distances(self) -> dict[str, float]:
        """The four flank-end distances [m] (see the attributes), by their field names, from the bodies."""
        return {key: getattr(getattr(self, body), attr) for key, body, attr in _ENDS}

    @classmethod
    def from_pair(cls, pair, *, points="path", driver: str, F_bn_N: float | None = None, load: float | None = None,
                  n1_rpm: float | None = None, E=None, nu=None, label: str | None = None
                  ) -> list[CylindricalGearContact]:
        """Contacts of an AxisForge cylindrical pair at points of the path of contact (SI units).

        Each contact is the two flanks at the point of the path where the flank of gear 1 has
        diameter ``dy1``: two :class:`InvoluteFlank` bodies with the values of that point, pressed
        by a load per unit length.

        Parameters
        ----------
        pair : GearPair
            AxisForge pair of ``CylindricalFamily`` (the exact class: another family, also the
            ``ManufacturedCylindricalFamily``, is refused). E and nu of each gear come from its
            ``material_id`` registered in ``axisforge.core.materials``, unless given below.
        points : "path", sequence of float, or mapping of str to float
            ``"path"``: the points A..E for ``driver`` (AxisForge ``path_points``, at
            ``dy1 = 2 sqrt((T1C + g_y)^2 + rb1^2)``, ISO 21771 Eq. (17) inverted; with gear 1 driving
            A is the active root ``2 rA1`` and E the tip ``da1``; B and D are evaluated exactly at
            the limits of single contact, so ``load_sharing`` gives the double-contact share there,
            R-305; a one-sided point is the caller's). Otherwise diameter(s) ``dy1`` of gear 1 [mm]
            on the path ``2 rA1 <= dy1 <= da1``: a mapping gives the labels, a sequence gets
            ``f"dy1={dy1:.6g}"``.
        driver : {"gear1", "gear2"}
            The driving gear: names A..E for ``"path"``, recorded, and fixes ``entrainment_sign_x``.
        F_bn_N : float, optional
            Total normal force of the mesh [N] (AxisForge ``MeshForces.F_bn_N``), > 0:
            ``load = F_bn / l_sum`` from ``load_sharing`` (the L0 of D-0035), ``share``, ``l``,
            ``l_sum``, ``n_pairs`` recorded.
        load : float, optional
            Load per unit length of the contact line [N/m], > 0, the same at every point;
            ``share``, ``l_mm``, ``l_sum_mm``, ``n_pairs`` nan. Exactly one of ``F_bn_N`` and
            ``load``.
        n1_rpm : float, optional
            Speed of gear 1 [rpm], > 0, whichever gear drives (a wheel speed is converted by the
            caller, ``n1 = n2 |z2 / z1|``): fills ``kinematics`` from ``pair_values_at``
            (``surface_speeds``). None: ``kinematics`` None.
        E, nu : (value or None, value or None), optional
            Young's modulus [MPa] and Poisson's ratio [-] of gear 1 and gear 2, given only for a
            gear without a registered AxisForge material; None for a gear that has one.
        label : str, optional
            Prefix of the labels; default ``pair.label`` or ``"gear"``.

        Returns
        -------
        list of CylindricalGearContact
            One per point, in the order given; label ``f"{prefix}_{point}"``. Every value at a
            point comes from one call of ``pair_values_at``: ``rho1_t_mm``, ``rho2_t_mm`` (the
            bodies, ``r = rho_t / cos(beta_b)`` = its ``rho1_n_mm``, ``rho2_n_mm``), ``g_y_mm``,
            ``dy2_mm``, ``zeta1``, ``zeta2``, ``Kg`` and, with ``n1_rpm``, ``u1``, ``u2``, ``ue``,
            ``vg``; the bridge recomputes none of them. Flank ends: gear 1 from its base circle to
            its tip circle; gear 2 external the same, a ring from its inner tip circle to its root
            circle. A helical point is the mid-face slice of its contact line, on ``[A, E]`` only:
            ``load_sharing`` also loads lines whose mid-face point lies up to ``b tan(beta_b)/2``
            outside the path, which are not translated (R-027).

        Raises
        ------
        TypeError
            If the pair is not of ``CylindricalFamily``.
        ValueError
            If not exactly one of ``F_bn_N`` and ``load`` is given, or it is not positive and
            finite; if ``points`` is a string other than "path"; if a label contains ``:``; if the
            path starts beyond the base-tangent point T1 (``T1A < 0``, a pair assembled with
            ``check=False``: the ``dy1`` input folds such a point onto its mirror, SB-59); if a
            ``dy1`` is off the path; if a point lies beyond T2 of an external pair (then
            ``points="path"`` fails at E: give the points before T2 as a mapping); the material
            errors of ``_materials``; and every check of the bodies and the contact.
        """
        tag = label or pair.label or "gear"
        if type(pair.family) is not cls.family:
            raise TypeError(f"{cls.__name__} takes a {cls.family.__name__} pair, got {type(pair.family).__name__} "
                            f"(pair {tag!r})")
        if (F_bn_N is None) == (load is None):
            raise ValueError(f"{tag}: give either F_bn_N or load, not both")
        # kept here: Contact would report the load per unit length, not the force given
        for key, value in (("F_bn_N", F_bn_N), ("load", load)):
            if value is not None and not (value > 0 and math.isfinite(value)):
                raise ValueError(f"{tag}: {key} must be positive and finite, got {value}")
        if isinstance(points, str):
            if points != "path":
                raise ValueError(f"{tag}: points must be 'path', diameters dy1 or a mapping label -> dy1, got {points!r}")
            rb1 = pair.gear1.rb
            points = {k: float(2.0 * math.hypot(pair.T1C + g_y, rb1))
                      for k, g_y in CylindricalFamily.path_points(pair, driver).items()}
        if isinstance(points, Mapping):
            labels, diameters = list(points), [float(v) for v in points.values()]
        else:
            diameters = [float(v) for v in np.atleast_1d(points)]
            labels = [f"dy1={d:.6g}" for d in diameters]
        bad = [lab for lab in labels if ":" in lab]
        if bad:
            raise ValueError(f"{tag}: point labels go into file names, no ':' (NTFS stream): {bad}")
        # OPEN SB-59: the whole-pair refusal keeps the T1 branch only (the dy1 input folds a point beyond T1 onto its
        # mirror, rho_y1 = L_y1 >= 0); a point beyond T2 of an external pair is refused point by point below
        if pair.T1A < 0.0:
            raise ValueError(f"{tag}: the path starts beyond the base-tangent point T1 (T1A = {pair.T1A:.6g} mm < 0): "
                             "a pair assembled with check=False is not translated")
        E_MPa, nu_gear, source = _materials(pair, tag, E, nu)

        values = CylindricalFamily.pair_values_at(pair, np.asarray(diameters), n1_rpm)
        g_y = values["g_y_mm"]                     # signed position on the path, + toward the tip of gear 1
        # on the path: distance from the stored A within [0, AE], with the end tolerance of load_sharing
        # (1e-9 p_bt) so that both agree at A and E; a dy1 below the base circle gives nan and fails too
        s, tol = g_y + pair.AC, 1e-9 * pair.gear1.pbt
        for lab, d, s_i in zip(labels, diameters, s):
            if not -tol <= s_i <= pair.AE + tol:
                raise ValueError(f"{tag}_{lab}: dy1 = {d} mm is off the path [{2.0 * pair.rA1:.6g}, "
                                 f"{pair.gear1.da:.6g}] mm (2 rA1 to da1)")
        sharing = load_sharing(pair, g_y) if F_bn_N is not None else None
        g1, g2 = pair.gear1, pair.gear2
        z2 = int(g2.z)
        # OPEN SB-66: root ends: the base circle of an external gear (upper bound of the involute; the ideal family
        # stores no d_Ff), the root circle of a ring (the fillet lies between d_Ff2 and d_f2)
        r_root2 = g2.rb if z2 > 0 else g2.rf
        out = []
        for i, lab in enumerate(labels):
            rho1_t, rho2_t = (float(values[k][i]) for k in ("rho1_t_mm", "rho2_t_mm"))
            if np.sign(rho2_t) != np.sign(z2):
                raise ValueError(f"{tag}_{lab}: the point is beyond the base-tangent point T2 (rho2_t = {rho2_t:.6g} "
                                 "mm): no involute of gear 2 there (SB-59)")
            # beta_b of gear 1 for both flanks, as curvature_radii (the hand only signs it)
            body1 = InvoluteFlank(rb=g1.rb * _MM, rho_t=rho1_t * _MM, beta_b=g1.betab, r_tip=g1.ra * _MM,
                                  r_root=g1.rb * _MM, toward_tip=True, E=E_MPa[0] * _MPA, nu=nu_gear[0])
            body2 = InvoluteFlank(rb=g2.rb * _MM, rho_t=rho2_t * _MM, beta_b=g1.betab, r_tip=g2.ra * _MM,
                                  r_root=r_root2 * _MM, toward_tip=False, E=E_MPa[1] * _MPA, nu=nu_gear[1])
            if sharing is None:
                w, share = load, dict(share=math.nan, l_mm=math.nan, l_sum_mm=math.nan, n_pairs=math.nan)
            else:
                w = F_bn_N / float(sharing["l_sum_mm"][i]) / _MM                  # [N/mm] -> [N/m]
                share = dict(share=float(sharing["share"][i]), l_mm=float(sharing["l_mm"][i]),
                             l_sum_mm=float(sharing["l_sum_mm"][i]), n_pairs=int(sharing["n_pairs"][i]))
            kinematics = None if n1_rpm is None else Kinematics(
                **{k: float(values[f"{k}_m_per_s"][i]) for k in ("ue", "vg", "u1", "u2")})
            out.append(cls(body1=body1, body2=body2, load=w, label=f"{tag}_{lab}", point=lab, dy1_mm=diameters[i],
                           dy2_mm=float(values["dy2_mm"][i]), g_y_mm=float(g_y[i]), **share, driver=driver, z2=z2,
                           material1=source[0], material2=source[1], zeta1=float(values["zeta1"][i]),
                           zeta2=float(values["zeta2"][i]), Kg=float(values["Kg"][i]), kinematics=kinematics))
        return out


def _materials(pair, tag: str, E, nu) -> tuple[list[float], list[float], list[str]]:
    """E [MPa], nu and their source per gear: the registered AxisForge material, or the values
    given for a gear without one."""
    for key, value in (("E", E), ("nu", nu)):
        if value is not None and not (isinstance(value, (tuple, list)) and len(value) == 2):
            raise ValueError(f"{tag}: {key} must be (gear1, gear2), each a value or None, got {value!r}")
    E, nu = (None, None) if E is None else tuple(E), (None, None) if nu is None else tuple(nu)
    E_MPa, nu_gear, source = [], [], []
    for i, name in enumerate(("gear1", "gear2")):
        material_id = getattr(pair, name).material_id
        given = E[i] is not None or nu[i] is not None
        if material_id != "" and material_id in available_materials():
            if given:
                raise ValueError(f"{tag}: {name} has material_id {material_id!r}; "
                                 "E and nu are given only for a gear without one")
            material = get_material(material_id)
            E_MPa.append(material.E)
            nu_gear.append(material.poisson_ratio)
            source.append(material_id)
        else:
            if E[i] is None or nu[i] is None:
                raise ValueError(f"{tag}: {name} has no AxisForge material (material_id {material_id!r} "
                                 "empty or not registered) and E, nu are not given for it")
            E_MPa.append(E[i])
            nu_gear.append(nu[i])
            # an id that AxisForge does not have stays visible
            source.append("given" if material_id == "" else f"given:{material_id}")
    return E_MPa, nu_gear, source
