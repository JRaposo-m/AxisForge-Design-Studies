"""The systems of s002: s001's 6210-like deep groove and NU210-like roller bearings, returned NOT solved.

Both at a diametral clearance of 20 um (mid of s001's sweep). The bearings and the load-distribution model
are s001's, imported through its public API (D-0036 amendment item 11). The tilt psi is the misalignment
of the inner ring against the outer ring in the plane of the radial load [rad], imposed (the solver's
``psi_override``), not taken from a shaft (s004 / s006 measure it).

Units: mm, N, rad (AxisForge); the records give psi in mrad.
"""
from __future__ import annotations

import math

from studies.s000_axisforge.s001_bearing_clearance_and_contact_angle import cylindrical_roller, deep_groove

__all__ = ["bearings", "S_MM", "FR_N", "FA_OVER_FR", "PSI_BALL_MRAD", "PSI_ROLLER_MRAD", "PSI_ISO_LIMIT_MRAD"]

S_MM = 0.020                                                    # diametral clearance [mm]
FR_N = (2500.0, 5000.0)                                         # two radial load levels [N], about 0.1 and 0.2 C
FA_OVER_FR = (0.0, 0.5)                                         # ball bearing only
PSI_BALL_MRAD = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)     # proposals.md s002: 0..2 mrad
PSI_ROLLER_MRAD = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)            # 0..1 mrad
# ISO/TS 16281 5.3.3, after Eq. (60) (PDF p. 21): the stress-riser approximation f[k] of Eq. (60), which AxisForge
# uses inside the roller L10r, holds for the reference profile, medium load and a total misalignment below 4 arc
# minutes; 4' = 4/60 deg = 1.1636 mrad. A limit of the roller life only: not of the lamina loads, not of ball bearings
PSI_ISO_LIMIT_MRAD = 1e3 * math.radians(4.0 / 60.0)


def bearings(s_mm: float = S_MM) -> dict:
    """The two bearings of s002 at diametral clearance ``s_mm`` [mm], unsolved: ``{"deep_groove": 6210-like,
    "cylindrical_roller": NU210-like (40 laminae)}`` (s001's builders)."""
    return {"deep_groove": deep_groove(s_mm), "cylindrical_roller": cylindrical_roller(s_mm)}
