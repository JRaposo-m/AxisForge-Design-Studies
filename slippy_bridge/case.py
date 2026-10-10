"""Contact case: the inputs of one Hertzian contact, as ``slippy.contact.hertz_full`` expects them.

Pure data in SI units (m, Pa, N). No AxisForge and no SlipPY import: a case can be written by
hand (closed-form references, didactic cases) or built from an AxisForge bearing by
:func:`slippy_bridge.geometry.contact_case`. Derived quantities (relative radii, reduced
modulus, contact semi-axes) are not computed here: they come from ``hertz_full``.

Moved from ``studies/bearings/contact/hertz_vs_bem/_common/case.py`` (the study constants stay
in the study).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

__all__ = ["ContactCase"]


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
    moduli : float or tuple of float
        Young's moduli ``(E1, E2)`` [Pa]; a scalar applies to both bodies.
    v : float or tuple of float
        Poisson's ratios ``(nu1, nu2)`` [-]; a scalar applies to both bodies.
    load : float
        Normal load [N] for a point contact; load per unit length [N/m] for a line contact.
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

    >>> c = ContactCase("ball_flat", r1=6.35e-3, r2=math.inf, moduli=208e9, v=0.3, load=500.0)
    >>> c.hertz_args()["r1"]
    (0.00635, 0.00635)
    """

    label: str
    r1: tuple[float, float]
    r2: tuple[float, float]
    moduli: tuple[float, float]
    v: tuple[float, float]
    load: float
    angle: float = 0.0
    line: bool = False

    def __post_init__(self) -> None:
        for name in ("r1", "r2", "moduli", "v"):
            object.__setattr__(self, name, _pair(getattr(self, name), name))
        for name in ("r1", "r2"):
            if any(r == 0 or math.isnan(r) for r in getattr(self, name)):
                raise ValueError(f"{name} contains a zero or NaN radius (use math.inf for flat)")
        if any(e <= 0 for e in self.moduli):
            raise ValueError(f"moduli must be positive, got {self.moduli}")
        if any(not 0.0 < n < 0.5 for n in self.v):
            raise ValueError(f"v must be in (0, 0.5), got {self.v}")
        if not self.load > 0:
            raise ValueError(f"load must be positive, got {self.load}")

    def hertz_args(self) -> dict:
        """Return the keyword arguments for ``slippy.contact.hertz_full(**case.hertz_args())``."""
        return {"r1": self.r1, "r2": self.r2, "moduli": self.moduli, "v": self.v,
                "load": self.load, "angle": self.angle, "line": self.line}

    def to_dict(self) -> dict:
        """Return the case as a JSON-serialisable dict (``inf`` written as the string ``"inf"``)."""

        def enc(x):
            if isinstance(x, tuple):
                return [enc(i) for i in x]
            if isinstance(x, float) and math.isinf(x):
                return "inf" if x > 0 else "-inf"
            return x

        return {"label": self.label, "r1": enc(self.r1), "r2": enc(self.r2),
                "moduli": list(self.moduli), "v": list(self.v), "load": self.load,
                "angle": self.angle, "line": self.line}

    def case_hash(self) -> str:
        """Return a short SHA-256 hash of the physical content of the case (label excluded)."""
        d = self.to_dict()
        d.pop("label")
        return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
