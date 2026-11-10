"""The systems of s001: three single-row bearings of the 6210 envelope, returned NOT solved.

* deep groove ball bearing, 6210-like (Dw 12.7 mm, Dpw 70 mm, Z 9), diametral clearance s;
* angular contact ball bearing, 7210-like (same balls and pitch, Z 13), nominal contact angle alpha_0;
* cylindrical roller bearing, NU210-like (Dwe 11 mm, Lwe 12 mm, Dpw 70 mm, Z 14, 40 laminae), clearance s.

The internal geometries are didactic, not catalogue data (the designations are labels only;
`# OPEN R5-41`); they equal v003's builds (``v003 construction``: Dw 12.7, Dpw 70 mm, Z 9 / 13 / 14,
envelope 50/90/20, steel), copied rather than imported because a v exposes only check and literature
(D-0036 amendment 4). One difference: the roller has 40 laminae here, 30 in v003. The family needs at
least 30; 40 gives s002's lamina map a finer x_k, and v003 records the roller rows only (RECORD).
The clearance sweep goes past the normal group of a d = 50 mm bearing on purpose (method study).
Raceway groove radii, the
centre distance A and the spring constants cp, cL come from the AxisForge families (ISO/TS 16281,
4.2.2 and 4.2.3). Steel balls, rollers and rings, E 210 GPa, nu 0.3.

Clearance: ``s`` is the DIAMETRAL internal clearance [mm] of the AxisForge families (deep groove:
alpha_0 = arccos(1 - s/(2A)); roller: s/2 per side; R-105). The angular contact family takes
alpha_0 and derives s = 2A(1 - cos alpha_0) from it: for that type alpha_0 and s are one variable.
No preload: the families reject s < 0 (``family.py``), so this is a method study, not a preloaded bearing.

Units: mm, N, MPa (AxisForge).
"""
from __future__ import annotations

from axisforge.core.machine_elements.bearings import (AngularContactFamily, Bearing, BearingCatalog, ContactAnalysis,
                                                      CylindricalRollerFamily, DeepGrooveBallFamily)

__all__ = ["deep_groove", "angular_contact", "cylindrical_roller", "S_MM", "ALPHA_0_DEG", "FA_OVER_FR", "FR_N"]

# ISO/TS 16281 stiffness of steel on steel [MPa, -]: e1, nu1 rolling element, e2, nu2 both raceways
_CONTACT = dict(contact=ContactAnalysis.ISO16281, e1=210_000.0, e2=210_000.0, nu1=0.3, nu2=0.3)
_ENVELOPE = dict(d=50.0, D=90.0, b=20.0)           # 6210 / 7210 / NU210 bore, outside diameter, width [mm]

# --- sweep (proposals.md s001) -----------------------------------------------------------------
S_MM = (0.0, 0.005, 0.010, 0.020, 0.030, 0.040, 0.050, 0.060)   # diametral clearance [mm], 0..60 um, finer near 0
ALPHA_0_DEG = (15.0, 25.0, 40.0)                               # angular contact nominal angle [deg]
FA_OVER_FR = (0.0, 0.25, 0.5, 1.0)                             # ball bearings; 0 = radial only
# OPEN R5-40: proposals ask "one Fr level from C0"; AxisForge holds no static rating C0 (only the dynamic
# bearing.C, ISO 281 not held). Fr = 5000 N is about 0.2 C of the 6210 build (C 24.8 kN), one level for all three
FR_N = 5000.0


def deep_groove(s_mm: float = 0.0, *, label: str = "6210") -> Bearing:
    """6210-like deep groove ball bearing with diametral clearance ``s_mm`` [mm], unsolved (locating)."""
    return Bearing.assemble(family=DeepGrooveBallFamily(),
                            catalog=BearingCatalog(**_ENVELOPE, designation="6210-like", arrangement="locating",
                                                   label=label),
                            geometry=dict(Dw=12.7, Dpw=70.0, Z=9, s=s_mm, **_CONTACT))


def angular_contact(alpha_0_deg: float = 40.0, *, label: str = "7210") -> Bearing:
    """7210-like single-row angular contact ball bearing, nominal angle ``alpha_0_deg`` [deg], unsolved (locating).

    It carries axial load in one direction only, the direction of ``A sin(alpha_0)`` (Fa > 0 in the
    AxisForge solver, ``ISO16281BallSolver.elements``).
    """
    return Bearing.assemble(family=AngularContactFamily(),
                            catalog=BearingCatalog(**_ENVELOPE, designation="7210-like", arrangement="locating",
                                                   label=label),
                            geometry=dict(Dw=12.7, Dpw=70.0, Z=13, alpha_0_deg=alpha_0_deg, **_CONTACT))


def cylindrical_roller(s_mm: float = 0.0, *, label: str = "NU210") -> Bearing:
    """NU210-like cylindrical roller bearing with diametral clearance ``s_mm`` [mm], unsolved (non-locating).

    40 laminae per roller (the family requires at least 30); the ISO reference profile of the
    family is always applied (``contact_solver.py``, Eq. (41); R5-BE3).
    """
    return Bearing.assemble(family=CylindricalRollerFamily(),
                            catalog=BearingCatalog(**_ENVELOPE, designation="NU210-like", arrangement="non-locating",
                                                   label=label),
                            geometry=dict(Dwe=11.0, Lwe=12.0, Dpw=70.0, Z=14, n_s=40, s=s_mm, **_CONTACT))
