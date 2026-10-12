"""Rolling bearings: AxisForge bearing (+ ISO/TS 16281 result) -> :class:`Contact` list.

:class:`BearingContacts` is registered for every AxisForge bearing family that exists today (point
and line contact, radial and thrust). One adapter serves them all, because AxisForge already
reduces each family to the same descriptor (``r_rolling_el``, ``r_inner``, ``r_outer``: two
principal radii each, in mm) and registers it as point or line contact; this module only converts
that descriptor to the Hertz/SlipPY form:

* units: mm -> m, MPa -> Pa;
* signs: convex > 0, concave < 0. AxisForge stores the rolling radius of the outer raceway as
  a positive magnitude (``geometry.py`` ``r_bx_outer``); the outer raceway is concave in the
  rolling direction, so it is negated here. AxisForge is left as it is (D-0019);
* contact angle: the radii AxisForge stores are evaluated at the family angle ``alpha_0``.
  When the ISO/TS 16281 contact solver has run, pass the element's operating angle
  ``alpha_j`` and the rolling radius of the raceway is re-evaluated at it. This repeats the
  one-line formula of AxisForge ``geometry.py`` (``r_bx_inner``/``r_bx_outer``) on purpose:
  ``alpha_j`` is a solver result read by design-studies (D-0019);
* load: N for a point contact; N/m for a line contact (load over the given length).

A new AxisForge family needs no change here as long as it provides the same three radii and
its point/line registration.
"""
from __future__ import annotations

import math

import numpy as np

from ...base import Contact, ContactCase, ElementAdapter, register

__all__ = ["contact_case", "rolling_radius", "entrainment_speed", "BearingContacts"]

_MM = 1e-3      # [m/mm]
_MPA = 1e6      # [Pa/MPa]


def rolling_radius(Dpw: float, D: float, alpha: float, race: str) -> float:
    """Raceway radius of curvature in the rolling direction, as a magnitude [same unit as Dpw].

    ``(Dpw - D cos(alpha)) / (2 cos(alpha))`` for the inner raceway,
    ``(Dpw + D cos(alpha)) / (2 cos(alpha))`` for the outer one, with ``D`` the ball diameter ``Dw`` or the
    roller diameter ``Dwe``. Same formula as AxisForge ``BallBearingGeometry.r_bx_inner/outer``
    (and, at ``alpha = 0``, the cylindrical roller geometry).

    Raises
    ------
    ValueError
        If ``race`` is not "inner"/"outer" or ``|alpha|`` is not below pi/2.
    """
    if race not in ("inner", "outer"):
        raise ValueError(f"race must be 'inner' or 'outer', got {race!r}")
    if not abs(alpha) < math.pi / 2:
        raise ValueError(f"|alpha| must be < pi/2 for a finite rolling radius, got {alpha}")
    sign = -1.0 if race == "inner" else 1.0
    return (Dpw + sign * D * math.cos(alpha)) / (2.0 * math.cos(alpha))


def _is_line_contact(bearing) -> bool:
    # The registry lives in the families module; it is not re-exported by axisforge.core.
    from axisforge.core.machine_elements.bearings.families.family import (
        is_line_contact_family, is_point_contact_family)
    if is_line_contact_family(bearing.family):
        return True
    if is_point_contact_family(bearing.family):
        return False
    raise ValueError(f"{type(bearing.family).__name__} is registered neither as point nor line contact")


def contact_case(bearing, race: str, load: float, alpha: float | None = None,
                 length: float | None = None, label: str | None = None) -> ContactCase:
    """Contact case of one rolling element against one raceway of an AxisForge bearing.

    Parameters
    ----------
    bearing : axisforge Bearing
        Assembled bearing (``af_c.Bearing.assemble``), any family.
    race : {"inner", "outer"}
        Raceway in contact.
    load : float
        Element normal load ``Q`` [N] (for a roller lamina: the lamina load ``q_jk`` [N]).
    alpha : float, optional
        Contact angle [rad]. None: the family angle ``alpha_0`` (radii as AxisForge stores them).
        Otherwise the operating angle ``alpha_j`` from the ISO/TS 16281 contact solver.
    length : float, optional
        Line contacts only: length carrying ``load`` [mm]. Default: the effective roller length
        ``Lwe`` (whole roller); pass ``Lwe / n_s`` for one lamina.
    label : str, optional
        Default ``"<bearing.label or bearing.designation>_<race>"``.

    Returns
    -------
    ContactCase
        Body 1 = rolling element, body 2 = raceway; SI units; ``x`` = rolling direction.

    Raises
    ------
    ValueError
        For an unknown race, a non-positive load or length, ``length`` on a point contact, or an
        angle outside (-pi/2, pi/2).
    """
    if race not in ("inner", "outer"):
        raise ValueError(f"race must be 'inner' or 'outer', got {race!r}")
    if not load > 0:
        raise ValueError(f"load must be positive, got {load}")
    line = _is_line_contact(bearing)

    r1 = [float(r) for r in bearing.r_rolling_el]
    r2 = [float(r) for r in getattr(bearing, f"r_{race}")]
    if alpha is not None and math.isfinite(r2[0]):          # thrust raceways are flat in x: stay inf
        D = bearing.Dwe if line else bearing.Dw
        r2[0] = rolling_radius(bearing.Dpw, D, alpha, race)
    if race == "outer" and math.isfinite(r2[0]):
        r2[0] = -abs(r2[0])                                  # concave in the rolling direction

    if line:
        length = bearing.Lwe if length is None else length
        if not length > 0:
            raise ValueError(f"length must be positive, got {length}")
        load_si = load / (length * _MM)                      # [N/m]
    else:
        if length is not None:
            raise ValueError("length applies to line contacts only")
        load_si = load

    if label is None:
        label = f"{bearing.label or bearing.designation}_{race}"
    return ContactCase(label=label,
                       r1=tuple(r * _MM for r in r1), r2=tuple(r * _MM for r in r2),
                       E=(bearing.e1 * _MPA, bearing.e2 * _MPA), nu=(bearing.nu1, bearing.nu2),
                       load=load_si, line=line)


def entrainment_speed(Dpw: float, D: float, alpha: float, n_inner_rpm: float, n_outer_rpm: float) -> float:
    """Entrainment (mean rolling) speed of a rolling element contact [m/s], pure rolling.

    ``u_e = (Dpw/4) (1 - gamma^2) |omega_i - omega_o|`` with ``gamma = D cos(alpha) / Dpw``
    (Harris & Kotzalas, cage kinematics; equal for the inner and outer contact). ``Dpw``, ``D``
    in mm. Spin and gyroscopic sliding are neglected; for a thrust bearing (alpha = pi/2) it is half
    the speed of the rotating washer at the pitch radius.
    """
    gamma = D * math.cos(alpha) / Dpw
    omega = abs(n_inner_rpm - n_outer_rpm) * 2.0 * math.pi / 60.0
    return (Dpw * _MM / 4.0) * (1.0 - gamma**2) * omega


def _families():
    """Every family AxisForge registers (``@register_family``): a new AxisForge family is picked up
    here without editing the bridge, as long as it provides the three radii and its point/line tag."""
    from axisforge.core.machine_elements.bearings.families.family import _FAMILY_REGISTRY
    return tuple(_FAMILY_REGISTRY.values())


def _row_of(result, row: int):
    """The ISO/TS 16281 row: ``result`` may be a BearingResult (``.rows``) or one row itself."""
    rows = getattr(result, "rows", None)
    if rows is None:
        if row != 0:
            raise ValueError("row applies to a BearingResult; a single row object was given")
        return result
    return rows[row]


def _select(Q: np.ndarray, elements) -> list[int]:
    if elements == "loaded":
        return [int(j) for j in np.flatnonzero(Q > 0.0)]
    if elements == "max":
        return [int(np.argmax(Q))]
    return [int(j) for j in elements]


@register(*_families())
class BearingContacts(ElementAdapter):
    """Adapter of every AxisForge rolling bearing family, point and line contact alike.

    One class only: AxisForge reduces each family to the same three pairs of radii and tags it as
    point or line contact (read by ``_is_line_contact``), so there is no second variant to subclass.
    """

    element = "bearing"

    @classmethod
    def contacts(cls, bearing, result=None, *, races=("inner", "outer"), elements="loaded",
                 row: int = 0, laminae: bool = False, load: float | None = None,
                 alpha: float | None = None, n_inner_rpm: float | None = None,
                 n_outer_rpm: float = 0.0) -> list[Contact]:
        """Contacts of one AxisForge bearing.

        Two ways to give the load:

        * ``result``: the ISO/TS 16281 result (``BearingResult`` or one of its rows). Each selected
          element j gives its ``Q_j`` and operating angle ``alpha_j``; rollers give their whole load
          over ``Lwe``, or with ``laminae=True`` one contact per loaded lamina (``q_jk`` over
          ``Lwe / n_s``).
        * ``load`` (and optionally ``alpha``): one contact per race, no element index.

        Parameters
        ----------
        bearing : axisforge Bearing
        result : BearingResult or LoadDistributionResult, optional
        races : sequence of {"inner", "outer"}
        elements : "loaded" | "max" | sequence of int
            Which rolling elements (with ``result``); unloaded elements in a list are skipped.
        row : int
            Row of a multi-row bearing (with ``result``).
        laminae : bool
            Roller bearings with ``result``: one contact per loaded lamina.
        load : float, optional
            Element load [N] when no ``result`` is given.
        alpha : float, optional
            Contact angle [rad] when no ``result`` is given (default: the family angle).
        n_inner_rpm, n_outer_rpm : float, optional
            Ring speeds [rpm]; if ``n_inner_rpm`` is given, ``kinematics`` gets ``u_e_m_per_s`` and
            ``slide_roll = 0`` (pure rolling).

        Returns
        -------
        list of Contact
            Inner and outer contacts carry the same element load (AxisForge has no centrifugal term).

        Raises
        ------
        ValueError
            If both or neither of ``result`` and ``load`` are given, or ``laminae`` is asked for a
            point contact.
        """
        if (result is None) == (load is None):
            raise ValueError("give either result (ISO/TS 16281) or load, not both")
        line = _is_line_contact(bearing)
        if laminae and not line:
            raise ValueError("laminae applies to roller (line contact) bearings only")
        subtype = type(bearing.family).__name__
        D = bearing.Dwe if line else bearing.Dw

        items = []          # (location, load [N], alpha [rad] or None, length [mm] or None)
        if result is None:
            items.append(({}, load, alpha, None))
        else:
            row_result = _row_of(result, row)
            phi_global = result.phi_j_global(row) if hasattr(result, "phi_j_global") else None
            for j in _select(row_result.Q_j, elements):
                loc = {"row": row, "element_j": j, "phi_local_rad": float(row_result.phi_j[j]),
                       **({} if phi_global is None else {"phi_global_rad": float(phi_global[j])}),
                       "alpha_rad": float(row_result.alpha_j[j])}
                if laminae:
                    lamina_length = bearing.Lwe / row_result.n_s
                    for k in np.flatnonzero(row_result.q_jk[j] > 0.0):
                        items.append(({**loc, "lamina_k": int(k), "x_k_mm": float(row_result.x_k[k])},
                                      float(row_result.q_jk[j, k]), float(row_result.alpha_j[j]),
                                      lamina_length))
                elif row_result.Q_j[j] > 0.0:
                    items.append((loc, float(row_result.Q_j[j]), float(row_result.alpha_j[j]), None))

        out = []
        for loc, Q, alpha_j, length in items:
            alpha_kin = bearing.alpha_0 if alpha_j is None else alpha_j
            kin = {} if n_inner_rpm is None else {
                "u_e_m_per_s": entrainment_speed(bearing.Dpw, D, alpha_kin, n_inner_rpm, n_outer_rpm),
                "slide_roll": 0.0}
            for race in races:
                suffix = "".join(f"_{k[0]}{v}" for k, v in loc.items() if k in ("element_j", "lamina_k"))
                case = contact_case(bearing, race, Q, alpha=alpha_j, length=length,
                                    label=f"{bearing.label or bearing.designation}_{race}{suffix}")
                out.append(Contact(cls.element, subtype, {"race": race, **loc}, case, kin))
        return out
