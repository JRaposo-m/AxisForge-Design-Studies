"""Contact case definition for the ``hertz_vs_bem`` question (SI units, no AxisForge import).

A :class:`ContactCase` is pure data that mirrors the inputs of ``slippy.contact.hertz_full``: the
principal radii of both bodies, their elastic constants, the normal load, the angle between the
principal axes and the line flag. Every build and the analytic reference start from the same
object, so they all describe the same contact. Derived quantities (relative radii, reduced
modulus, contact semi-axes) are NOT computed here: they come from ``hertz_full``.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

__all__ = ["ContactCase", "CASES", "N_POWERS", "N_POWERS_BY_CASE", "EXTENT_IN_A", "EXTENT_IN_A_BY_CASE"]


# =============================================================================
# Inputs  (SI: m, Pa, N)
# =============================================================================

# --- materials (one steel for every body) --------------------------------------
E_STEEL = 210e9                      # Young's modulus [Pa]
NU_STEEL = 0.3                       # Poisson's ratio [-]

# --- bearing geometry (deep groove ball bearing, typical 6210-type values, to confirm) ---
# Same definitions as the AxisForge ball-bearing geometry (Harris / Hamrock-Dowson notation),
# written here by hand so that this study does not import AxisForge.
DW = 12.7e-3                         # ball diameter [m]
DPW = 70.0e-3                        # pitch diameter [m]
F_I = 0.52                           # inner groove radius / ball diameter ri/Dw [-]
ALPHA_0 = 0.0                        # free contact angle [rad]

# --- bodies: principal radii (r_x, r_y) [m]; convex > 0, concave < 0, math.inf = flat ---
BALL = (DW / 2.0, DW / 2.0)
FLAT = (math.inf, math.inf)
INNER_RACEWAY = ((DPW - DW * math.cos(ALPHA_0)) / (2.0 * math.cos(ALPHA_0)),   # rolling radius
                 -F_I * DW)                                                    # groove radius

# --- roller (cylindrical roller bearing, same pitch diameter; placeholder size, to confirm) ---
DWE = 10.0e-3                        # roller diameter [m]
ROLLER = (DWE / 2.0, math.inf)       # (rx, ry): cylinder, no curvature along its axis
INNER_RACEWAY_ROLLER = ((DPW - DWE) / 2.0, math.inf)   # rolling radius of the inner raceway, straight in y

# --- loads ---------------------------------------------------------------------
LOAD_BALL_FLAT = 500.0               # [N]
LOAD_ROLLER_PER_LENGTH = 5.0e4       # [N/m] line contact; PLACEHOLDER (= 500 N over a 10 mm roller)
LOAD_BALL_RACEWAY = 500.0            # [N]  PLACEHOLDER: set the value of the S4 reference case

# --- grid (study constants) ----------------------------------------------------
N_POWERS = (6, 7, 8, 9)              # nodes per side N = 2**n (even N puts the apex on a node)
N_POWERS_BY_CASE = {                 # overrides N_POWERS for a case (the ellipse needs finer grids)
    "ball_raceway": (8, 9, 10),
}
EXTENT_IN_A_BY_CASE = {"ball_raceway": 3.0}   # the ball exists only up to r = 6.35 mm
EXTENT_IN_A = 6.0                    # half-width of the square domain in units of the semi-axis a [-]


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
        Principal radii of curvature ``(r_x, r_y)`` of body 1 and body 2 at the contact point
        [m]. A scalar means ``(r, r)``. Sign convention: convex > 0, concave < 0, flat =
        ``math.inf``. These are radii of curvature, **not** ellipsoid semi-axes (compare
        ``slippy.surface.RoundSurface``).
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

    Notes
    -----
    SI units throughout; ``hertz_full`` only requires consistent units. The geometry builds of
    this study assume ``angle == 0``; the analytic side accepts any angle.

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
        """Return a short SHA-256 hash of the physical content of the case (label excluded).

        Cases with the same radii, materials, load, angle and line flag share a hash whatever
        their label, so ``compare`` can check that all builds of a type describe the same contact.
        """
        d = self.to_dict()
        d.pop("label")
        return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


# =============================================================================
# Cases (one per type folder)
# =============================================================================

CASES = {
    "ball_flat": ContactCase("ball_flat", r1=BALL, r2=FLAT, moduli=E_STEEL, v=NU_STEEL,
                             load=LOAD_BALL_FLAT),
    "ball_raceway": ContactCase("ball_raceway", r1=BALL, r2=INNER_RACEWAY, moduli=E_STEEL,
                               v=NU_STEEL, load=LOAD_BALL_RACEWAY),
    "roller_raceway": ContactCase("roller_raceway", r1=ROLLER, r2=INNER_RACEWAY_ROLLER,
                                  moduli=E_STEEL, v=NU_STEEL, load=LOAD_ROLLER_PER_LENGTH, line=True),
}