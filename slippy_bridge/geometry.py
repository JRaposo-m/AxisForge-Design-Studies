"""AxisForge bearing -> :class:`ContactCase` (no SlipPY import).

One function for every AxisForge bearing family. AxisForge already reduces each family to the
same descriptor (``r_rolling_el``, ``r_inner``, ``r_outer``: two principal radii each, in mm) and
registers it as point or line contact; this module only converts that descriptor to the
Hertz/SlipPY form:

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

from .case import ContactCase

__all__ = ["contact_case", "rolling_radius"]

_MM = 1e-3      # [m/mm]
_MPA = 1e6      # [Pa/MPa]


def rolling_radius(Dpw: float, D: float, alpha: float, race: str) -> float:
    """Raceway radius of curvature in the rolling direction, as a magnitude [same unit as Dpw].

    ``(Dpw - D cos a) / (2 cos a)`` for the inner raceway, ``(Dpw + D cos a) / (2 cos a)`` for the
    outer one, with ``D`` the ball diameter ``Dw`` or the roller diameter ``Dwe``. Same formula
    as AxisForge ``BallBearingGeometry.r_bx_inner/outer`` (and, at ``a = 0``, the cylindrical
    roller geometry).

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
                       moduli=(bearing.e1 * _MPA, bearing.e2 * _MPA), v=(bearing.nu1, bearing.nu2),
                       load=load_si, line=line)
