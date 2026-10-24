"""Inputs of the contact part of the bridge.

* :class:`ContactCase` - the Hertz inputs of one contact (radii, materials, load), SI units,
  ``x`` = rolling direction. Pure data; no SlipPY import.
* :class:`Kinematics` - the surface speeds of one contact, AxisForge names, SI units.
* :class:`Contact` - one contact of one machine element: the case plus *which* element, subtype
  and location it is, and (optionally) its kinematics. This is what the element functions
  (``elements.bearings.bearing_contacts``, ``elements.gears.gear_contacts``) hand to the surface,
  reference and step modules, and what the outputs record.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from axisforge.results._base import DC

if TYPE_CHECKING:
    from .elements.bearings import BearingLocation
    from .elements.gears import GearLocation

__all__ = ["ContactCase", "Kinematics", "Contact"]


def _pair(value, name: str) -> tuple[float, float]:
    """Return ``value`` as a 2-tuple of floats (a scalar is repeated), as ``hertz_full`` does."""
    if isinstance(value, (int, float)):
        value = (value, value)
    value = tuple(float(v) for v in value)
    if len(value) == 1:
        value = (value[0], value[0])
    if len(value) != 2:
        raise ValueError(f"{name} must be a number or a 2-element sequence, got {value}")
    return value


@dataclass(frozen=True)
class ContactCase:
    """Inputs of one Hertzian contact, in the form ``hertz_full`` expects.

    Parameters
    ----------
    label : str
        Short identifier used in file names and reports (no spaces).
    r1, r2 : float or tuple of float
        Principal radii of curvature ``(r_x, r_y)`` of body 1 (rolling element) and body 2
        (raceway) at the contact point [m]. ``x`` is the rolling direction. A scalar means
        ``(r, r)``. Sign convention: convex > 0, concave < 0, flat = ``math.inf``. These are
        radii of curvature, **not** ellipsoid semi-axes (compare ``slippy.surface.RoundSurface``).
    E : float or tuple of float
        Young's moduli ``(E1, E2)`` [Pa]; a scalar applies to both bodies (``moduli`` in SlipPY's
        ``hertz_full``).
    nu : float or tuple of float
        Poisson's ratios ``(nu1, nu2)`` [-]; a scalar applies to both bodies (``v`` in SlipPY's
        ``hertz_full``).
    load : float
        Normal load [N] for a point contact; load per unit length [N/m] for a line contact (the
        total load given to the step: ``surfaces.Grid.step_args``).
    angle : float, optional
        Angle between the x axes of the two bodies [rad]. Default 0.
    line : bool, optional
        True for a line contact (parallel cylinders), as the ``line`` flag of ``hertz_full``.

    Raises
    ------
    ValueError
        If a radius is zero or NaN, a modulus or the load is not positive, or a Poisson's ratio
        is outside (0, 0.5).

    References
    ----------
    Johnson, K. L. (1985). *Contact Mechanics*. Cambridge University Press, ch. 4.

    Examples
    --------
    A 12.7 mm steel ball on a flat, 500 N:

    >>> c = ContactCase("ball_flat", r1=6.35e-3, r2=math.inf, E=208e9, nu=0.3, load=500.0)
    >>> c.hertz_args()["r1"]
    (0.00635, 0.00635)
    """

    label: str
    r1: tuple[float, float]
    r2: tuple[float, float]
    E: tuple[float, float]
    nu: tuple[float, float]
    load: float
    angle: float = 0.0
    line: bool = False

    def __post_init__(self) -> None:
        for name in ("r1", "r2", "E", "nu"):
            object.__setattr__(self, name, _pair(getattr(self, name), name))
        for name in ("r1", "r2"):
            if any(r == 0 or math.isnan(r) for r in getattr(self, name)):
                raise ValueError(f"{name} contains a zero or NaN radius (use math.inf for flat)")
        if any(e <= 0 for e in self.E):
            raise ValueError(f"E must be positive, got {self.E}")
        if any(not 0.0 < n < 0.5 for n in self.nu):
            raise ValueError(f"nu must be in (0, 0.5), got {self.nu}")
        if not self.load > 0:
            raise ValueError(f"load must be positive, got {self.load}")

    def hertz_args(self) -> dict:
        """Return the keyword arguments for ``slippy.contact.hertz_full(**case.hertz_args())``."""
        return {"r1": self.r1, "r2": self.r2, "moduli": self.E, "v": self.nu,
                "load": self.load, "angle": self.angle, "line": self.line}

    def to_dict(self) -> dict:
        """Return the case as a JSON-serialisable dict (``inf`` written as the string ``"inf"``).

        :attr:`E` and :attr:`nu` are written under the keys ``moduli`` and ``v``. This dict is
        what :meth:`case_hash` hashes, so the keys are kept and the hash of a saved result stays
        the same.
        """

        def enc(x):
            if isinstance(x, tuple):
                return [enc(i) for i in x]
            if isinstance(x, float) and math.isinf(x):
                return "inf" if x > 0 else "-inf"
            return x

        return {"label": self.label, "r1": enc(self.r1), "r2": enc(self.r2),
                "moduli": list(self.E), "v": list(self.nu), "load": self.load,
                "angle": self.angle, "line": self.line}

    def case_hash(self) -> str:
        """Return a short SHA-256 hash of the physical content of the case (label excluded)."""
        d = self.to_dict()
        d.pop("label")
        return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


# =============================================================================
# One contact of one machine element
# =============================================================================

@dataclass(**DC)
class Kinematics:
    """Surface speeds of one contact, in the direction of rolling ``x`` [m/s].

    Names and signs are those of AxisForge ``CylindricalFamily.surface_speeds`` (gears), where
    ``u1``, ``u2`` are the speeds of the two surfaces relative to the contact point. Needed by
    lubrication, not by the static step.

    Parameters
    ----------
    ue_m_per_s : float
        MEAN entrainment speed ``(u1 + u2) / 2`` [m/s], never the sum ``u1 + u2`` (the rolling
        speed of SlipPY's Reynolds solver is this mean).
    vg_m_per_s : float
        Sliding speed ``u1 - u2`` [m/s], signed (body 1 referenced). For a bearing, pure rolling:
        ``vg = 0`` and ``u1``, ``u2`` not computed.
    u1_m_per_s, u2_m_per_s : float, optional
        Surface speeds of body 1 and body 2 [m/s], when the element gives them (gears); None
        for a bearing, where only the entrainment speed is formed.
    """

    ue_m_per_s: float
    vg_m_per_s: float
    u1_m_per_s: float | None = None
    u2_m_per_s: float | None = None


@dataclass(**DC)
class Contact:
    """One contact of one machine element, ready for the surface, reference and step modules.

    Parameters
    ----------
    element : str
        Machine element: ``"bearing"`` or ``"gear"``.
    subtype : str
        AxisForge class of the element subtype (e.g. ``"DeepGrooveBallFamily"``).
    location : BearingLocation or GearLocation
        Where the contact is, element-specific, units in the field names (see the two classes).
    case : ContactCase
        The Hertz inputs.
    kinematics : Kinematics, optional
        The surface speeds; None when the caller gave no speeds.
    """

    element: str
    subtype: str
    location: BearingLocation | GearLocation
    case: ContactCase
    kinematics: Kinematics | None = None

    def record(self) -> dict:
        """Flat record for the outputs: ``element``, ``subtype``, the location fields, ``label``,
        ``case_hash``, ``line`` and, when given, the kinematics fields (None = empty cell)."""
        return {"element": self.element, "subtype": self.subtype, **asdict(self.location),
                "label": self.case.label, "case_hash": self.case.case_hash(), "line": self.case.line,
                **(asdict(self.kinematics) if self.kinematics is not None else {})}
