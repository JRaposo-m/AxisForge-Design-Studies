"""Rolling bearings: AxisForge bearing (+ ISO/TS 16281 result) -> bearing contacts.

One contact is one rolling element (body 1) against one raceway (body 2). AxisForge reduces every
bearing family to the same descriptor: the principal radii of the rolling element
(``r_rolling_el``) and of each raceway (``r_inner``, ``r_outer``), two each in mm, with the
family tagged as point or line contact. This module converts that descriptor to the bridge
form and picks the contact class from the tag: :class:`BallRacewayContact` (point) or
:class:`RollerRacewayContact` (line). Radial and thrust bearings use the same two classes; a
thrust raceway is flat along rolling (``r_x = inf``).

Conversions made here, once, at the AxisForge boundary:

* units: mm -> m (``_MM``), MPa -> Pa (``_MPA``);
* signs: convex > 0, concave < 0. AxisForge stores the rolling radius of the outer raceway as a
  positive magnitude (``BallBearingGeometry.r_bx_outer``); the outer raceway is concave along
  rolling, so it is negated here. AxisForge is not changed (D-0019);
* contact angle: AxisForge evaluates the raceway radii at the family angle ``alpha_0``. When
  the ISO/TS 16281 contact solver has run, the rolling radius of a ball raceway is re-evaluated
  at the element's operating angle ``alpha_j``, with the formula of AxisForge
  ``BallBearingGeometry.r_bx_inner`` / ``r_bx_outer``. AxisForge's own contact constant ``c_p``
  stays at ``alpha_0`` (R-019, ruled), so the bridge radii and AxisForge's stiffness may differ
  slightly at ``alpha_j != alpha_0``. A roller raceway radius ``(Dpw -/+ Dwe)/2`` and a flat
  thrust raceway do not depend on the angle and are never re-evaluated;
* load: element load ``Q`` [N] for a point contact; for a line contact the mean load per unit
  length over the effective roller length ``Lwe`` [N/m]. Under tilt or edge load the mean
  understates the peak: use ``laminae=True`` (``q_jk`` over ``Lwe / n_s``).

A new AxisForge family needs no change here as long as it provides the same descriptor and its
point/line tag; a family that does not raises ``NotImplementedError`` naming what is missing.

:class:`DeepGrooveBallContact` and :class:`AngularContactBallContact` (D-0034 item 10) are contact
classes of one AxisForge family each, with their bodies and their reading of the bearing inside
them; their raceway is the true surface of revolution. The classes above give way to one class per
family.

References
----------
Harris, T. A., Kotzalas, M. N. (2007). *Rolling Bearing Analysis*, vol. 1, section 2.2 (ball and
raceway curvatures); AxisForge ``axisforge.core.machine_elements.bearings.families.geometry``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from axisforge.core.machine_elements.bearings.families import available_families
# the point/line predicates are not re-exported by axisforge.core
from axisforge.core.machine_elements.bearings.families.family import (AngularContactFamily, DeepGrooveBallFamily,
                                                                       is_line_contact_family,
                                                                       is_point_contact_family)
from axisforge.outputs.solvers.bearings.load_distribution.load_distribution_results import wrap_angle

from ..contact import DC, Body, Contact, SweptCircleBody
from . import _MM, _MPA

__all__ = ["BearingContact", "DeepGrooveBallContact", "AngularContactBallContact", "BallRacewayContact",
           "RollerRacewayContact", "bearing_contacts"]

# what bearing_contacts reads from an assembled AxisForge bearing: single-row geometry, then the
# elastic constants AxisForge only stores with contact=ContactAnalysis.ISO16281
_REQUIRED = ("r_rolling_el", "r_inner", "r_outer", "Dpw", "alpha_0")
_MATERIAL = ("e1", "e2", "nu1", "nu2")


@dataclass(**DC)
class BearingContact(Contact):
    """Contact of one rolling element (body 1) with one raceway (body 2) of a rolling bearing.

    Abstract like :class:`Contact`: each family class (:class:`DeepGrooveBallContact`,
    :class:`AngularContactBallContact`; until they
    are removed, :class:`BallRacewayContact` and :class:`RollerRacewayContact`) sets ``line`` and
    ``family`` (its AxisForge family class). The bridge axes are the bearing's own: x along rolling (the circumferential
    direction of the raceway), y across it (along the roller axis or the groove), so the body
    radii are principal on the bridge axes and the bodies are not turned (``angle = 0``). A
    spherical outer raceway (self-aligning bearing) would need its own handling: its gap can be
    near-circular and more curved across than along. Field names as the AxisForge element and
    lamina records (``element_records``, ``lamina_records``).

    Parameters
    ----------
    race : {"inner", "outer"}
        Raceway in contact.
    alpha_rad : float
        Contact angle of the element [rad]: the family angle ``alpha_0``, a given angle, or the
        element's solved ``alpha_j``. A ball raceway's rolling radius is evaluated at it; for
        rollers AxisForge repeats the nominal ``alpha_0`` there (not a result).
    row : int, optional
        Bearing row (from a result).
    j : int, optional
        Rolling element index (from a result).
    phi_rad : float, optional
        Element angle in the local frame of the row, 0 on the radial load line, wrapped to
        [-pi, pi) [rad] as AxisForge records it (``wrap_angle``).
    phi_global_rad : float, optional
        Element angle in the global frame [rad], [0, 2 pi), when the result is a
        ``BearingResult`` (AxisForge ``BearingResult.phi_j_global``; the global azimuth
        convention is open, R-003).

    Other fields as :class:`Contact` (``body1``, ``body2``, ``load``, ``angle``, ``label``).

    Raises
    ------
    ValueError
        If ``race`` is not "inner"/"outer"; if ``|alpha_rad| >= pi/2`` on a raceway curved along
        rolling; if ``angle != 0``, or ``psi != 0`` for a point contact (principal axes not on
        the bridge axes); and every check of :class:`Contact`.
    """

    family: ClassVar[type]        # declared, not set: each family class sets its AxisForge family

    race: str
    alpha_rad: float
    row: int | None = None
    j: int | None = None
    phi_rad: float | None = None
    phi_global_rad: float | None = None

    def __post_init__(self) -> None:
        name = f"BearingContact {self.label!r}"
        if self.race not in ("inner", "outer"):
            raise ValueError(f"{name}: race must be 'inner' or 'outer', got {self.race!r}")
        # The rolling radius (Dpw -/+ D cos a)/(2 cos a) is infinite at |a| = pi/2. A raceway flat
        # along rolling does not depend on alpha, and the solver may give a thrust ball alpha_j
        # slightly beyond 90 deg: no bound there.
        # OPEN SB-04: |alpha| < pi/2 only for a finite raceway r_x; no bound on a flat (thrust) raceway, alpha_j may pass 90 deg
        if math.isfinite(self.body2.r[0]) and not abs(self.alpha_rad) < math.pi / 2:
            raise ValueError(f"{name}: |alpha_rad| must be < pi/2 on a raceway curved along rolling, "
                             f"got {self.alpha_rad}")
        if self.angle != 0.0:
            raise ValueError(f"{name}: angle must be 0 (element and raceway share the bridge axes), got {self.angle}")
        super().__post_init__()
        # Contact already holds a line on bridge y. For a point, psi != 0 (with angle 0) means the
        # gap is more curved across rolling than along it: the minor semi-axis would not lie along rolling.
        if not self.line and self.psi != 0.0:
            raise ValueError(f"{name}: psi = 0 expected (principal axes on the bridge axes, minor semi-axis "
                             f"along rolling), got psi = {self.psi} rad")

    # ---- reading of AxisForge, shared by every family class (from_bearing of each class calls these)

    @classmethod
    def _refuse_other_family(cls, bearing) -> None:
        """Raise TypeError unless ``bearing`` is of the AxisForge family of this class (``cls.family``)."""
        if type(bearing.family) is not cls.family:
            raise TypeError(f"{cls.__name__} takes a {cls.family.__name__} bearing, got "
                            f"{type(bearing.family).__name__} (bearing {bearing.label or bearing.designation!r})")

    @staticmethod
    def _materials(bearing) -> tuple[float, float, float, float]:
        """``E1, nu1, E2, nu2`` [Pa, -] of the rolling element (1) and the raceways (2), MPa -> Pa.

        AxisForge stores them only on a bearing assembled with ``contact`` (ISO/TS 16281).
        """
        if not all(hasattr(bearing, a) for a in _MATERIAL):
            raise ValueError(f"{type(bearing.family).__name__}: the bearing has no elastic constants; assemble it "
                             "with contact=ContactAnalysis.ISO16281 and e1, e2, nu1, nu2")
        return bearing.e1 * _MPA, bearing.nu1, bearing.e2 * _MPA, bearing.nu2

    @classmethod
    def _loads(cls, bearing, result, *, load, alpha, elements, row):
        """The elements to build: ``(row_result, [(fields, Q [N]), ...])``.

        With ``load``: one item, no element index, ``alpha_rad`` = ``alpha`` or the family angle
        ``alpha_0``, ``row_result`` None. With ``result`` (ISO/TS 16281): one item per selected
        loaded element of the row, with its ``Q_j``, ``alpha_j``, ``phi_j`` (wrapped to [-pi, pi)
        as the AxisForge element records) and, from a ``BearingResult``, ``phi_j_global``.

        ``result`` may be what AxisForge returns: one value of the dict of ``SolverBase.solve``
        (its ``load_distribution`` is read), the ``BearingResult`` of ``solve_contact``, or one row
        of it. A ``BearingResult`` must be converged (``ok``; AxisForge does not enforce it, and
        its root tolerance is not a residual bound) and, when both carry a label, belong to
        ``bearing``. A bare row carries neither flag nor label and is taken as given. The row must
        be of the class's kind (point or line).
        """
        if (result is None) == (load is None):
            raise ValueError("give either result (ISO/TS 16281) or load, not both")
        if result is None:
            if row != 0:
                raise ValueError("row applies to result= only")
            return None, [({"alpha_rad": float(bearing.alpha_0 if alpha is None else alpha)}, float(load))]
        if alpha is not None:
            raise ValueError("alpha applies to load= only; with result= the solved alpha_j is used")
        result = getattr(result, "load_distribution", result)   # BearingAnalysisResult -> BearingResult
        if hasattr(result, "rows"):
            if not result.ok:
                raise ValueError(f"result {result.label!r} is not converged (ok=False, "
                                 f"residual {result.residual:.3g} N)")
            if result.label and bearing.label and result.label != bearing.label:
                raise ValueError(f"result {result.label!r} is not of bearing {bearing.label!r}")
            if not 0 <= row < len(result.rows):
                raise ValueError(f"row {row} is not a row of result {result.label!r} ({len(result.rows)} row(s))")
            phi_global, row_result = result.phi_j_global(row), result.rows[row]
        else:
            if row != 0:
                raise ValueError("row applies to a BearingResult; a single row object was given")
            phi_global, row_result = None, result
        if row_result.IS_LINE_CONTACT != cls.line:
            raise ValueError(f"the result row is a {'line' if row_result.IS_LINE_CONTACT else 'point'} contact "
                             f"row, {cls.__name__} a {'line' if cls.line else 'point'} contact")
        Q = row_result.Q_j
        if elements in ("loaded", "max"):
            if not np.any(Q > 0.0):
                raise ValueError(f"elements={elements!r}: no loaded element in the row (every Q_j <= 0)")
            js = np.flatnonzero(Q > 0.0) if elements == "loaded" else [np.argmax(Q)]
        else:
            if not all(0 <= j < len(Q) for j in elements):
                raise ValueError(f"elements {list(elements)} must be indices in [0, {len(Q)}) of the row")
            js = [j for j in elements if Q[j] > 0.0]               # unloaded elements of a list are skipped
        phi = wrap_angle(row_result.phi_j)
        return row_result, [({"row": row, "j": int(j), "phi_rad": float(phi[j]),
                              "phi_global_rad": None if phi_global is None else float(phi_global[j]),
                              "alpha_rad": float(row_result.alpha_j[j])}, float(Q[j])) for j in js]

    @staticmethod
    def _label(bearing, race: str, fields: dict) -> str:
        """``"<bearing label or designation>_<race>"``, plus ``_e<j>`` and ``_l<k>`` with a result."""
        suffix = "".join(f"_{tag}{fields[key]}" for key, tag in (("j", "e"), ("k", "l")) if key in fields)
        return f"{bearing.label or bearing.designation}_{race}{suffix}"


@dataclass(**DC)
class BallRacewayContact(BearingContact):
    """Ball on a raceway: a point contact, load ``Q`` [N] (see :class:`BearingContact`)."""

    line = False


@dataclass(**DC)
class RollerRacewayContact(BearingContact):
    """Roller on a raceway: a line contact along bridge y (the roller axis), load per unit length
    [N/m] (see :class:`BearingContact`). Straight rollers and raceways (no crowning, as AxisForge's
    ``r_ay = inf``).

    Parameters
    ----------
    k : int, optional
        Lamina index (``laminae=True``); None for the whole roller.
    x_mm : float, optional
        Mid-position of the lamina along the roller axis (AxisForge ``x_k``) [mm]; the roller axis
        is bridge y.
    """

    line = True

    k: int | None = None
    x_mm: float | None = None


@dataclass(**DC)
class DeepGrooveBallContact(BearingContact):
    """Ball on a raceway groove of an AxisForge ``DeepGrooveBallFamily`` bearing: a point contact,
    load ``Q`` [N].

    Body 1 is the :class:`Ball`, body 2 the true :class:`Raceway`, the groove circle swept about
    the bearing axis. Build the contacts of a bearing with :meth:`from_bearing`; the other fields
    as :class:`BearingContact`. The local torus with the same radii (``SweptCircleBody``) is not
    accepted as a body here: compare with it through a plain :class:`Contact` subclass.

    Raises
    ------
    ValueError
        If body 1 is not a :class:`Ball` or body 2 not a :class:`Raceway` of the same race and
        angle as the contact; and every check of :class:`BearingContact`.
    """

    line = False
    family = DeepGrooveBallFamily

    # OPEN SB-43: only the family's own Ball and true Raceway are accepted as bodies (alternative: also a SweptCircleBody of equal r)
    def __post_init__(self) -> None:
        if not (isinstance(self.body1, self.Ball) and isinstance(self.body2, self.Raceway)
                and (self.body2.race, self.body2.alpha) == (self.race, self.alpha_rad)):
            raise ValueError(f"{type(self).__name__} {self.label!r}: body1 must be its Ball and body2 its Raceway "
                             f"of race {self.race!r} at alpha {self.alpha_rad}")
        super().__post_init__()

    @dataclass(**DC)
    class Ball(Body):
        """The ball: a sphere of diameter ``Dw`` [m] (``r = (Dw/2, Dw/2)``).

        Other fields as :class:`Body` (``E``, ``nu``).
        """

        Dw: float

        def __post_init__(self) -> None:
            if not (self.Dw > 0 and math.isfinite(self.Dw)):
                raise ValueError(f"Ball: Dw must be positive and finite [m], got {self.Dw}")
            super().__post_init__()

        @property
        def r(self) -> tuple[float, float]:
            return (self.Dw / 2, self.Dw / 2)

        def profile(self, X, Y, angle=0.0):
            """Exact sphere ``-(R - sqrt(R^2 - x^2 - y^2))``, ``R = Dw/2`` (see :meth:`Body.profile`);
            ``angle`` has no effect on a sphere."""
            R = self.Dw / 2
            d2 = np.asarray(X, dtype=float)**2 + np.asarray(Y, dtype=float)**2
            if d2.max() >= R**2:
                raise ValueError(f"the grid reaches beyond the ball of diameter {self.Dw} m: reduce span_x/span_y")
            return -(R - np.sqrt(R**2 - d2))

    @dataclass(**DC)
    class Raceway(Body):
        """The true raceway groove: the groove circle of the meridian plane swept about the bearing axis.

        In the meridian plane through the contact point (axial coordinate along the bearing axis,
        ``rho`` the distance from it) the groove is a circle of radius ``r_g`` whose centre lies on
        the contact normal, ``r_g`` from the contact point toward the ball. The contact point is at
        ``R_P`` from the bearing axis, the normal inclined by ``alpha`` (the contact's
        ``alpha_rad``) to the radial direction. The raceway is that circle swept about the bearing
        axis, exact at any contact angle; it is taken whole, without the groove shoulders.

        Body frame (see :class:`Body`): y_b the meridian tangent ``cos(alpha) e_a - sin(alpha)
        e_r`` and z_b the outward normal ``sign (sin(alpha) e_a + cos(alpha) e_r)`` toward the
        ball, ``sign = +1`` inner, ``-1`` outer, with ``e_r`` radially outward and ``e_a`` the
        axial direction in which the outer contact point lies ahead of the inner one for ``alpha
        > 0``: AxisForge's direction of ``delta_a``, the axial component ``V_j`` of the groove-centre
        line (ISO/TS 16281 Eq. (15)). +y_b is the same direction on both raceways (+e_a at ``alpha
        = 0``). x_b along rolling, ``x_b = -sign e_t`` with ``e_t = e_a x e_r``: right-handed on the
        inner raceway, left-handed on the outer as the Body contract has it for body 2. The profile
        is even in x_b, not in y_b (the distance from the axis changes along the meridian):
        ``profile(x, y)`` at ``-alpha`` equals ``profile(x, -y)`` at ``alpha``.

        The parallel and the meridian are lines of curvature of a surface of revolution, so x_b,
        y_b are principal: ``r_y = -r_g`` (concave groove) and, by Meusnier's theorem, ``r_x = sign
        R_P / cos(alpha)`` (the normal section along rolling has curvature ``cos(alpha) / R_P``;
        Harris's raceway radius ``(Dpw -/+ D cos(alpha)) / (2 cos(alpha))``, AxisForge
        ``r_bx_inner`` / ``r_bx_outer`` at ``alpha_0``, when ``R_P = (Dpw -/+ D cos(alpha)) / 2``).

        Parameters
        ----------
        race : {"inner", "outer"}
        R_P : float
            Distance of the contact point from the bearing axis [m].
        r_g : float
            Groove radius [m], > 0 (AxisForge ``ri`` / ``re``).
        alpha : float
            Contact angle [rad], ``|alpha| < pi/2``.

        Other fields as :class:`Body` (``E``, ``nu``).
        """

        # Newton's method of the profile: step tolerance [m] and iteration cap
        NEWTON_TOL: ClassVar[float] = 1e-12
        NEWTON_MAX_IT: ClassVar[int] = 50

        race: str
        R_P: float
        r_g: float
        alpha: float

        def __post_init__(self) -> None:
            if self.race not in ("inner", "outer"):
                raise ValueError(f"Raceway: race must be 'inner' or 'outer', got {self.race!r}")
            if not abs(self.alpha) < math.pi / 2:
                raise ValueError(f"Raceway: |alpha| must be < pi/2, got {self.alpha}")
            sign = 1.0 if self.race == "inner" else -1.0
            # the groove circle must not reach the bearing axis: its centre at rho_G > r_g
            if not (self.r_g > 0 and math.isfinite(self.R_P)
                    and self.R_P + sign * self.r_g * math.cos(self.alpha) > self.r_g):
                raise ValueError(f"Raceway: need r_g > 0 and the groove centre farther than r_g from the bearing "
                                 f"axis [m], got r_g = {self.r_g}, R_P = {self.R_P}")
            super().__post_init__()

        @property
        def r(self) -> tuple[float, float]:
            sign = 1.0 if self.race == "inner" else -1.0
            return (sign * self.R_P / math.cos(self.alpha), -self.r_g)

        def profile(self, X, Y, angle=0.0):
            """Exact height of the swept groove at each node (see :meth:`Body.profile`).

            The node ``P + x x_b + y y_b + z z_b`` lies on the raceway when ``F(z) = (rho -
            rho_G)^2 + (ax - ax_G)^2 - r_g^2 = 0`` (``rho`` its distance from the bearing axis,
            ``ax`` its axial coordinate, P at ``ax = 0``). Solved for ``z`` node by node by Newton's
            method from the Hertz paraboloid (Johnson 1985, Eq. (4.2a)), to ``NEWTON_TOL``; the root
            on the contact side of the groove centre is kept.

            Raises
            ------
            ValueError
                If a node lies off the raceway: no root (beyond the groove, ``|y_b|`` about ``r_g``
                or more), no convergence, or the root on the far side of the groove.
            """
            x, y = self._body_frame(X, Y, angle)
            sign = 1.0 if self.race == "inner" else -1.0
            ca, sa = math.cos(self.alpha), math.sin(self.alpha)
            n_ax, n_r = sign * sa, sign * ca                     # z_b: axial and radial components
            ax_G, rho_G = self.r_g * n_ax, self.R_P + self.r_g * n_r   # groove centre
            r_x, r_y = self.r
            z = -(x**2 / (2 * r_x) + y**2 / (2 * r_y))            # Hertz paraboloid: start
            with np.errstate(invalid="ignore", divide="ignore"):  # NaN off the raceway, caught below
                for _ in range(self.NEWTON_MAX_IT):
                    ax = y * ca + z * n_ax                       # axial coordinate of the node
                    q = self.R_P - y * sa + z * n_r              # radial coordinate in the meridian of P
                    rho = np.hypot(q, x)                         # distance from the bearing axis
                    F = (rho - rho_G)**2 + (ax - ax_G)**2 - self.r_g**2
                    dF = 2 * (rho - rho_G) * (q / rho) * n_r + 2 * (ax - ax_G) * n_ax
                    step = F / dF
                    z = z - step
                    if not np.all(np.abs(step) <= self.NEWTON_TOL):   # NaN also continues, then fails
                        continue
                    ax, q = y * ca + z * n_ax, self.R_P - y * sa + z * n_r
                    rho = np.hypot(q, x)
                    # contact side: the node lies below the groove centre along the normal, as P does
                    # (axisymmetry: the test in the (rho, ax) half-plane holds off the meridian of P)
                    if np.all((rho - rho_G) * n_r + (ax - ax_G) * n_ax < 0):
                        return z
                    break
            raise ValueError(f"the grid reaches beyond the {self.race} raceway (r_g {self.r_g} m, alpha "
                             f"{self.alpha} rad): reduce span_x/span_y")

    @classmethod
    def from_bearing(cls, bearing, result=None, *, load: float | None = None, alpha: float | None = None,
                     races=("inner", "outer"), elements="loaded", row: int = 0) -> list[DeepGrooveBallContact]:
        """Contacts of one AxisForge single-row deep groove ball bearing.

        Two ways to give the load:

        * ``result``: the ISO/TS 16281 result (see :meth:`BearingContact._loads`). Each selected
          element j gives its ``Q_j`` and operating angle ``alpha_j``.
        * ``load`` (and optionally ``alpha``): one contact per race, no element index.

        Inner and outer contacts of an element carry the same load and angle: AxisForge has no
        centrifugal or gyroscopic term, so the ball is in equilibrium between two equal and
        opposite normal forces.

        Parameters
        ----------
        bearing : axisforge Bearing
            Assembled single-row (``i = 1``) ``DeepGrooveBallFamily`` bearing with the ISO/TS 16281
            contact constants.
        result : BearingAnalysisResult, BearingResult or BallLoadDistributionResult, optional
        load : float, optional
            Element normal load ``Q`` [N] when no ``result`` is given.
        alpha : float, optional
            Contact angle [rad] with ``load``; default the family angle ``alpha_0``.
        races : sequence of {"inner", "outer"}
        elements : "loaded" | "max" | sequence of int
            Which balls (with ``result``); unloaded balls in a list are skipped.
        row : int
            Row of the result (with ``result``; AxisForge gives one row).

        Returns
        -------
        list of DeepGrooveBallContact
            Element by element, races in the order given; labels ``"<bearing label or
            designation>_<race>"``, plus ``_e<j>`` with a result.

        Raises
        ------
        TypeError
            If ``bearing`` is not of ``DeepGrooveBallFamily``.
        NotImplementedError
            If the bearing has two rows (``i = 2``): AxisForge ignores ``i`` in the ISO/TS 16281
            solve (one row of ``Z`` balls at ``+alpha_0`` carries the whole load; only the rating
            reads ``i``).
        ValueError
            As :meth:`BearingContact._loads`; if the bearing has no elastic constants; and every
            check of the bodies and the contact.

        Notes
        -----
        The raceway is built at the element's angle (``alpha_j``, ``alpha_0`` or the given one):
        groove radius ``r_g = ri`` or ``re``; contact point at ``R_P = (Dpw -/+ Dw cos(alpha)) / 2``
        from the bearing axis, the ball centre kept on the pitch circle, so that ``r_x`` is
        AxisForge's ``BallBearingGeometry.r_bx_inner`` / ``r_bx_outer`` formula evaluated at
        ``alpha`` (AxisForge itself evaluates it at ``alpha_0`` only). AxisForge's ISO/TS 16281
        solver keeps instead the groove centres fixed in the rings (Eq. (13)-(15)): ``R_P = R_G -/+
        r_g cos(alpha)`` with ``R_G = Dpw/2 +/- (r_g - Dw/2) cos(alpha_0)``; the two agree at
        ``alpha = alpha_0`` and differ by ``(r_g - Dw/2)(cos(alpha) - cos(alpha_0))`` elsewhere
        (inner; outer opposite sign): 8.6 um, 3e-4 of ``R_P``, for the most loaded ball of the
        6210 case under its combined load (``alpha_j`` 17.95 deg).

        AxisForge's ``Q_j`` comes from its contact constant ``c_p`` at ``alpha_0`` (R-019), while
        these bodies are at ``alpha_j``: for that ball the effective ``R_x`` of the inner contact
        is 5.254 mm here against 5.198 mm at ``alpha_0`` (+1.1 %), so a bridge approach and
        AxisForge's ``delta_j`` differ at that order.
        """
        cls._refuse_other_family(bearing)
        if bearing.i != 1:
            raise NotImplementedError(f"{cls.__name__}: a {bearing.i}-row bearing is not translated (AxisForge's "
                                      "ISO/TS 16281 solve ignores i: one row of Z balls carries the whole load)")
        E1, nu1, E2, nu2 = cls._materials(bearing)
        ball = cls.Ball(Dw=bearing.Dw * _MM, E=E1, nu=nu1)
        _, items = cls._loads(bearing, result, load=load, alpha=alpha, elements=elements, row=row)
        out = []
        for fields, Q in items:
            alpha_el = fields["alpha_rad"]
            for race in races:
                sign = 1.0 if race == "inner" else -1.0
                # OPEN SB-41: contact point with the ball centre on the pitch circle (AxisForge r_bx formula at alpha), not the groove fixed in the ring
                R_P = (bearing.Dpw - sign * bearing.Dw * math.cos(alpha_el)) / 2               # [mm]
                r_g = bearing.ri if race == "inner" else bearing.re                          # [mm]
                raceway = cls.Raceway(race=race, R_P=R_P * _MM, r_g=r_g * _MM, alpha=alpha_el, E=E2, nu=nu2)
                out.append(cls(body1=ball, body2=raceway, load=Q, label=cls._label(bearing, race, fields),
                               race=race, **fields))
        return out

@dataclass(**DC)
class AngularContactBallContact(BearingContact):
    """Ball on a raceway groove of an AxisForge ``AngularContactFamily`` bearing: a point contact,
    load ``Q`` [N].

    The same bodies as :class:`DeepGrooveBallContact` (a ball in a groove of radius ``ri``/``re``,
    the raceway the true surface of revolution; AxisForge builds both families from the same
    ``BallBearingGeometry``); the family gives ``alpha_0`` (in (0, 45] deg) instead of the
    clearance. Build the contacts of a bearing with :meth:`from_bearing`; the other fields as
    :class:`BearingContact`.

    AxisForge models the bearing as a symmetric full groove with clearance: an axial load against
    the contact angle gives ``alpha_j <= 0`` (the ball on the other side of the groove, ``ok =
    True``). A real single-row angular contact bearing has a relieved shoulder there and carries
    no such load, so those contacts are refused. Neither AxisForge nor the bridge checks that the
    contact ellipse stays inside the shoulders (at 25 deg under Fr = Fa = 5 kN the major axis
    spans about 12 to 43 deg of the groove arc).

    The grooves are conformal (``a / r_g`` 0.27 in that case): the exact sphere and groove differ
    from the Hertz paraboloid by a y^4 term that stiffens the contact, so the BEM gives ``a`` about
    2 to 3 % below Hertz (falling as ``a^2`` with the load), a physical difference near the 3 %
    of ``rtol_reference``, not a numerical one (C10c slippy-expert).

    Raises
    ------
    ValueError
        If body 1 is not a :class:`Ball` or body 2 not a :class:`Raceway` of the same race and
        angle as the contact; and every check of :class:`BearingContact`.
    """

    line = False
    family = AngularContactFamily

    # OPEN SB-44: the bodies of DeepGrooveBallContact are reused, not copied (same groove geometry, as AxisForge's BallBearingGeometry for both families)
    Ball = DeepGrooveBallContact.Ball
    Raceway = DeepGrooveBallContact.Raceway

    # OPEN SB-43: only the family's own Ball and true Raceway are accepted as bodies (alternative: also a SweptCircleBody of equal r)
    def __post_init__(self) -> None:
        if not (isinstance(self.body1, self.Ball) and isinstance(self.body2, self.Raceway)
                and (self.body2.race, self.body2.alpha) == (self.race, self.alpha_rad)):
            raise ValueError(f"{type(self).__name__} {self.label!r}: body1 must be its Ball and body2 its Raceway "
                             f"of race {self.race!r} at alpha {self.alpha_rad}")
        super().__post_init__()

    @classmethod
    def from_bearing(cls, bearing, result=None, *, load: float | None = None, alpha: float | None = None,
                     races=("inner", "outer"), elements="loaded", row: int = 0) -> list[AngularContactBallContact]:
        """Contacts of one AxisForge single-row angular contact ball bearing.

        Two ways to give the load:

        * ``result``: the ISO/TS 16281 result (see :meth:`BearingContact._loads`). Each selected
          element j gives its ``Q_j`` and operating angle ``alpha_j``.
        * ``load`` (and optionally ``alpha``): one contact per race, no element index.

        Inner and outer contacts of an element carry the same load and angle: AxisForge has no
        centrifugal or gyroscopic term, so the ball is in equilibrium between two equal and
        opposite normal forces.

        Parameters
        ----------
        bearing : axisforge Bearing
            Assembled single-row (``i = 1``) ``AngularContactFamily`` bearing with the ISO/TS 16281
            contact constants.
        result : BearingAnalysisResult, BearingResult or BallLoadDistributionResult, optional
        load : float, optional
            Element normal load ``Q`` [N] when no ``result`` is given.
        alpha : float, optional
            Contact angle [rad] with ``load``; default the family angle ``alpha_0``.
        races : sequence of {"inner", "outer"}
        elements : "loaded" | "max" | sequence of int
            Which balls (with ``result``); unloaded balls in a list are skipped.
        row : int
            Row of the result (with ``result``; AxisForge gives one row).

        Returns
        -------
        list of AngularContactBallContact
            Element by element, races in the order given; labels ``"<bearing label or
            designation>_<race>"``, plus ``_e<j>`` with a result.

        Raises
        ------
        TypeError
            If ``bearing`` is not of ``AngularContactFamily``.
        NotImplementedError
            If the bearing has two rows (``i = 2``): AxisForge ignores ``i`` in the ISO/TS 16281
            solve (one row of ``Z`` balls at ``+alpha_0`` carries the whole load; only the rating
            reads ``i``).
        ValueError
            As :meth:`BearingContact._loads`; if a selected contact angle is on the relieved side
            (``alpha * alpha_0 <= 0``); if the bearing has no elastic constants; and every check of
            the bodies and the contact.

        Notes
        -----
        The raceway is built at the element's angle (``alpha_j``, ``alpha_0`` or the given one):
        groove radius ``r_g = ri`` or ``re``; contact point at ``R_P = (Dpw -/+ Dw cos(alpha)) / 2``
        from the bearing axis, the ball centre kept on the pitch circle, so that ``r_x`` is
        AxisForge's ``BallBearingGeometry.r_bx_inner`` / ``r_bx_outer`` formula evaluated at
        ``alpha`` (AxisForge itself evaluates it at ``alpha_0`` only). AxisForge's ISO/TS 16281
        solver keeps instead the groove centres fixed in the rings (Eq. (13)-(15)): ``R_P = R_G -/+
        r_g cos(alpha)`` with ``R_G = Dpw/2 +/- (r_g - Dw/2) cos(alpha_0)``; the two agree at
        ``alpha = alpha_0`` and differ by ``(r_g - Dw/2)(cos(alpha) - cos(alpha_0))`` elsewhere
        (inner; outer opposite sign): 4.9 um, 1.7e-4 (inner) / 1.2e-4 (outer) of ``R_P``, for the most loaded ball of a
        25 deg bearing (Dw 12.7, Dpw 70 mm, Z 15) under Fr = Fa = 5 kN (``alpha_j`` 27.49 deg).

        AxisForge's ``Q_j`` comes from its contact constant ``c_p`` at ``alpha_0`` (R-019), while
        these bodies are at ``alpha_j``: for that ball the effective ``R_x`` of the inner contact
        is 5.328 mm here against 5.306 mm at ``alpha_0`` (+0.4 %), so a bridge approach and
        AxisForge's ``delta_j`` differ at that order.
        """
        cls._refuse_other_family(bearing)
        if bearing.i != 1:
            raise NotImplementedError(f"{cls.__name__}: a {bearing.i}-row bearing is not translated (AxisForge's "
                                      "ISO/TS 16281 solve ignores i: one row of Z balls carries the whole load)")
        E1, nu1, E2, nu2 = cls._materials(bearing)
        ball = cls.Ball(Dw=bearing.Dw * _MM, E=E1, nu=nu1)
        _, items = cls._loads(bearing, result, load=load, alpha=alpha, elements=elements, row=row)
        # OPEN SB-45: alpha on the relieved side (alpha * alpha_0 <= 0) refused (alternative: build the mirrored raceway, documented)
        wrong_side = [f.get("j") for f, _ in items if not f["alpha_rad"] * bearing.alpha_0 > 0]
        if wrong_side:
            raise ValueError(f"{cls.__name__}: contact angle on the relieved side (alpha * alpha_0 <= 0; element(s) "
                             f"{wrong_side}): an axial load against the contact angle, which a single-row angular "
                             "contact bearing does not carry (AxisForge solves a symmetric groove)")
        out = []
        for fields, Q in items:
            alpha_el = fields["alpha_rad"]
            for race in races:
                sign = 1.0 if race == "inner" else -1.0
                # OPEN SB-41: contact point with the ball centre on the pitch circle (AxisForge r_bx formula at alpha), not the groove fixed in the ring
                R_P = (bearing.Dpw - sign * bearing.Dw * math.cos(alpha_el)) / 2               # [mm]
                r_g = bearing.ri if race == "inner" else bearing.re                          # [mm]
                raceway = cls.Raceway(race=race, R_P=R_P * _MM, r_g=r_g * _MM, alpha=alpha_el, E=E2, nu=nu2)
                out.append(cls(body1=ball, body2=raceway, load=Q, label=cls._label(bearing, race, fields),
                               race=race, **fields))
        return out

def _is_line_contact(bearing) -> bool:
    """True for a line contact family, False for a point contact one, after checking that the
    bearing can be translated.

    Raises
    ------
    NotImplementedError
        Naming the family: not an AxisForge point/line family, no single-row geometry (multi-row
        families keep it per row), or a thrust ball bearing away from 90 deg.
    ValueError
        If the bearing was assembled without the elastic constants.
    """
    family = type(bearing.family).__name__
    line = is_line_contact_family(bearing.family)
    if not line and not is_point_contact_family(bearing.family):
        raise NotImplementedError(f"{family} is not an AxisForge point or line contact family "
                                  f"(available_families(): {sorted(available_families())})")
    missing = [a for a in (*_REQUIRED, *(("Dwe", "Lwe") if line else ("Dw",))) if not hasattr(bearing, a)]
    if missing:
        raise NotImplementedError(f"{family}: the assembled bearing has no {', '.join(missing)} "
                                  "(single-row geometry); not translated")
    if not all(hasattr(bearing, a) for a in _MATERIAL):
        raise ValueError(f"{family}: the bearing has no elastic constants; assemble it with "
                         "contact=ContactAnalysis.ISO16281 and e1, e2, nu1, nu2")
    # AxisForge stores a flat thrust raceway (r_x = inf) for every alpha_0, exact only at 90 deg
    # (about 10 % in kappa_x at 60 deg). 1e-9 rad: the tolerance of AxisForge's own 90 deg test.
    # OPEN SB-15: thrust ball with alpha_0 != 90 deg refused (AxisForge stores a flat raceway; which washer is inner below 90 deg is José's call)
    if (not line and math.isinf(float(bearing.r_inner[0]))
            and not math.isclose(bearing.alpha_0, math.pi / 2, abs_tol=1e-9)):
        raise NotImplementedError(f"{family} at alpha_0 = {math.degrees(bearing.alpha_0):.6g} deg: AxisForge stores "
                                  "a raceway flat along rolling, exact only at 90 deg; not translated")
    return line


def _bodies(bearing, race: str, alpha: float | None) -> tuple[SweptCircleBody, SweptCircleBody]:
    """Rolling element (body 1) and raceway (body 2) in SI, outer rolling radius signed.

    ``alpha`` None keeps the radii as AxisForge stores them (at ``alpha_0``); otherwise the rolling
    radius of a ball raceway curved along rolling is re-evaluated at ``alpha``.
    """
    r1 = [float(r) for r in bearing.r_rolling_el]
    r2 = [float(r) for r in (bearing.r_outer if race == "outer" else bearing.r_inner)]
    if alpha is not None and math.isfinite(r2[0]):
        D_el = bearing.Dw
        sign = 1.0 if race == "outer" else -1.0              # AxisForge r_bx, at alpha
        r2[0] = (bearing.Dpw + sign * D_el * math.cos(alpha)) / (2.0 * math.cos(alpha))
    if race == "outer" and math.isfinite(r2[0]):
        r2[0] = -abs(r2[0])                                  # concave along rolling
    return (SweptCircleBody(r=tuple(r * _MM for r in r1), E=bearing.e1 * _MPA, nu=bearing.nu1),
            SweptCircleBody(r=tuple(r * _MM for r in r2), E=bearing.e2 * _MPA, nu=bearing.nu2))


def _row_of(result, row: int):
    """The ISO/TS 16281 row: ``result`` may be a BearingResult (``.rows``) or one row itself."""
    rows = getattr(result, "rows", None)
    if rows is None:
        if row != 0:
            raise ValueError("row applies to a BearingResult; a single row object was given")
        return result
    return rows[row]


def _select(Q: np.ndarray, elements) -> list[int]:
    if elements in ("loaded", "max"):
        if not np.any(Q > 0.0):
            raise ValueError(f"elements={elements!r}: no loaded element in the row (every Q_j <= 0)")
        return [int(j) for j in np.flatnonzero(Q > 0.0)] if elements == "loaded" else [int(np.argmax(Q))]
    return [int(j) for j in elements]


def bearing_contacts(bearing, result=None, *, load: float | None = None, alpha: float | None = None,
                     races=("inner", "outer"), elements="loaded", row: int = 0,
                     laminae: bool = False) -> list[BearingContact]:
    """Contacts of one AxisForge bearing (any family assembled with single-row geometry).

    Two ways to give the load:

    * ``result``: the ISO/TS 16281 result. Each selected element j gives its ``Q_j`` and operating
      angle ``alpha_j``; a roller gives its whole load as the mean line load ``Q_j / Lwe``, or with
      ``laminae=True`` one contact per loaded lamina (``q_jk / (Lwe / n_s)``).
    * ``load`` (and optionally ``alpha``): one contact per race, no element index.

    Inner and outer contacts of an element carry the same load and angle: AxisForge has no
    centrifugal or gyroscopic term, so the element is in equilibrium between two equal and
    opposite normal forces.

    Parameters
    ----------
    bearing : axisforge Bearing
        Assembled bearing (``axisforge.core.Bearing.assemble``) with the ISO/TS 16281 contact
        constants.
    result : BearingAnalysisResult, BearingResult or LoadDistributionResult, optional
        What AxisForge returns: one value of the dict of ``SolverBase.solve`` (its
        ``load_distribution`` is read), the ``BearingResult`` of ``solve_contact``, or one row of
        it. A ``BearingResult`` must be converged (``ok``) and, when both carry a label, belong to
        ``bearing``; it gives ``phi_global_rad``. A bare row carries neither flag nor label and is
        taken as given. The row must be of the bearing's kind (point or line).
    load : float, optional
        Element normal load ``Q`` [N] when no ``result`` is given.
    alpha : float, optional
        Contact angle [rad] with ``load``, for a ball raceway curved along rolling; default the
        family angle ``alpha_0`` with the radii as AxisForge stores them. Refused with ``result``
        (its own ``alpha_j`` is used), for line families and for flat raceways (no effect on the
        radii).
    races : sequence of {"inner", "outer"}
    elements : "loaded" | "max" | sequence of int
        Which rolling elements (with ``result``); unloaded elements in a list are skipped.
    row : int
        Row of a multi-row bearing (with ``result``).
    laminae : bool
        Roller bearings with ``result``: one contact per loaded lamina.

    Returns
    -------
    list of BearingContact
        :class:`BallRacewayContact` or :class:`RollerRacewayContact` from the AxisForge point/line
        tag, element by element (lamina by lamina), races in the order given. Labels
        ``"<bearing label or designation>_<race>"``, plus ``_e<j>`` and ``_l<k>`` with a result.

    Raises
    ------
    ValueError
        If both or neither of ``result`` and ``load`` are given; ``alpha`` with ``result``, a line
        family or a flat raceway; ``laminae`` or ``row`` with ``load``; the result is not
        converged, belongs to another bearing or its row is of the other kind; ``row`` with a bare
        row; ``laminae`` for a point contact; ``elements`` "loaded"/"max" on a row with no loaded
        element; the bearing has no elastic constants; and every check of the contact classes.
    NotImplementedError
        If the family is not translated (see ``_is_line_contact``).
    """
    if (result is None) == (load is None):
        raise ValueError("give either result (ISO/TS 16281) or load, not both")
    if result is not None and alpha is not None:
        raise ValueError("alpha applies to load= only; with result= the solved alpha_j is used")
    line = _is_line_contact(bearing)                           # the family tag, checked once per bearing
    if laminae and not line:
        raise ValueError("laminae applies to roller (line contact) bearings only")
    # OPEN SB-16: alpha= refused for line families and flat raceways (no effect on the radii)
    if alpha is not None and (line or math.isinf(float(bearing.r_inner[0]))):
        raise ValueError("alpha has no effect on a line family or a flat raceway (radii independent of the angle)")
    cls = RollerRacewayContact if line else BallRacewayContact
    length = bearing.Lwe if line else None                     # [mm], whole roller

    # (fields of the contact, load [N], angle to re-evaluate the ball raceway at or None, length [mm])
    items = []
    if result is None:
        if laminae or row != 0:
            raise ValueError("laminae and row apply to result= only")
        fields = {"alpha_rad": float(bearing.alpha_0 if alpha is None else alpha)}
        items.append((fields, load, alpha, length))
    else:
        result = getattr(result, "load_distribution", result)   # BearingAnalysisResult -> BearingResult
        if hasattr(result, "rows"):
            # AxisForge does not enforce convergence; its root tolerance is not a residual bound,
            # so the flag ``ok`` is the only test it documents.
            if not result.ok:
                raise ValueError(f"result {result.label!r} is not converged (ok=False, "
                                 f"residual {result.residual:.3g} N)")
            if result.label and bearing.label and result.label != bearing.label:
                raise ValueError(f"result {result.label!r} is not of bearing {bearing.label!r}")
        phi_global = result.phi_j_global(row) if hasattr(result, "rows") else None
        row_result = _row_of(result, row)
        if row_result.IS_LINE_CONTACT != line:
            raise ValueError(f"the result row is a {'line' if row_result.IS_LINE_CONTACT else 'point'} contact "
                             f"row, the bearing a {'line' if line else 'point'} contact family")
        phi = wrap_angle(row_result.phi_j)                      # [-pi, pi), as the AxisForge records
        lamina_length = bearing.Lwe / row_result.n_s if laminae else None   # [mm], equal slices
        for j in _select(row_result.Q_j, elements):
            alpha_j = float(row_result.alpha_j[j])
            alpha_eval = None if line else alpha_j              # roller raceways do not depend on alpha
            fields = {"row": row, "j": j, "phi_rad": float(phi[j]),
                      "phi_global_rad": None if phi_global is None else float(phi_global[j]),
                      "alpha_rad": alpha_j}
            if laminae:
                for k in np.flatnonzero(row_result.q_jk[j] > 0.0):
                    items.append(({**fields, "k": int(k), "x_mm": float(row_result.x_k[k])},
                                  float(row_result.q_jk[j, k]), None, lamina_length))
            elif row_result.Q_j[j] > 0.0:
                items.append((fields, float(row_result.Q_j[j]), alpha_eval, length))

    name = bearing.label or bearing.designation
    out = []
    for fields, Q, alpha_eval, length_mm in items:
        # line contact: mean load per unit length, mm -> m [N/m]
        load = Q if length_mm is None else Q / (length_mm * _MM)
        # label suffixes: element "_e<j>", lamina "_l<k>"
        suffix = "".join(f"_{tag}{fields[key]}" for key, tag in (("j", "e"), ("k", "l")) if key in fields)
        for race in races:
            body1, body2 = _bodies(bearing, race, alpha_eval)
            out.append(cls(body1=body1, body2=body2, load=load, label=f"{name}_{race}{suffix}", race=race,
                           **fields))
    return out
