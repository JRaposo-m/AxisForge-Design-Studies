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

References
----------
Harris, T. A., Kotzalas, M. N. (2007). *Rolling Bearing Analysis*, vol. 1, section 2.2 (ball and
raceway curvatures); AxisForge ``axisforge.core.machine_elements.bearings.families.geometry``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from axisforge.core.machine_elements.bearings.families import available_families
# the point/line predicates are not re-exported by axisforge.core
from axisforge.core.machine_elements.bearings.families.family import (is_line_contact_family,
                                                                       is_point_contact_family)
from axisforge.outputs.solvers.bearings.load_distribution.load_distribution_results import wrap_angle

from ..contact import DC, Body, Contact
from . import _MM, _MPA

__all__ = ["BearingContact", "BallRacewayContact", "RollerRacewayContact", "bearing_contacts"]

# what bearing_contacts reads from an assembled AxisForge bearing: single-row geometry, then the
# elastic constants AxisForge only stores with contact=ContactAnalysis.ISO16281
_REQUIRED = ("r_rolling_el", "r_inner", "r_outer", "Dpw", "alpha_0")
_MATERIAL = ("e1", "e2", "nu1", "nu2")


@dataclass(**DC)
class BearingContact(Contact):
    """Contact of one rolling element (body 1) with one raceway (body 2) of a rolling bearing.

    Abstract like :class:`Contact`: :class:`BallRacewayContact` and :class:`RollerRacewayContact`
    set ``line``. The bridge axes are the bearing's own: x along rolling (the circumferential
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


def _bodies(bearing, race: str, alpha: float | None) -> tuple[Body, Body]:
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
    return (Body(r=tuple(r * _MM for r in r1), E=bearing.e1 * _MPA, nu=bearing.nu1),
            Body(r=tuple(r * _MM for r in r2), E=bearing.e2 * _MPA, nu=bearing.nu2))


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
