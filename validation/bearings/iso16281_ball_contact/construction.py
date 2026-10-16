"""
Construction -- 6210-like deep groove ball bearing and the load cases of
``validation/bearings/iso16281_ball_contact``.

Single source of the inputs of the case, written directly against the AxisForge API. The bearing
is the one of the case ``validation/contact/hertz_ellipse`` (same bearing, other loads); its
numbers are repeated here on purpose, a validation case shows its whole construction and imports
no study.

``build_bearing(s)`` returns the bearing assembled and NOT solved; the script chooses the solver.

Model
-----
Deep groove ball bearing with the boundary dimensions of a 6210 (d 50, D 90, B 20 mm), 9 balls of
12.7 mm on a pitch diameter of 70 mm, both groove radii 0.52 Dw (the AxisForge family default),
steel on steel. The radial clearance ``s`` sets the free contact angle ``alpha_0`` (0 for s = 0).
The number of balls and the groove radii are typical values, not catalogue data.

Units
-----
mm, N, MPa -- the AxisForge unit system. The conversion to SI is made only inside
``slippy_bridge``.
"""
from __future__ import annotations

import axisforge.core as af_c

__all__ = ["build_bearing", "LOADS"]

# ---- inputs -----------------------------------------------------------------------------------
E_STEEL = 210000.0      # Young's modulus [MPa]
NU_STEEL = 0.3          # Poisson's ratio [-]
DW = 12.7               # ball diameter [mm]
DPW = 70.0              # pitch diameter [mm]
Z = 9                   # balls [-]

# Load cases: radial clearance s [mm], radial load Fr [N] (applied in the XZ plane), axial load
# Fa [N]. Example values, easy to change: a moderate radial load without clearance (pressures at
# the level of the roughness and EHL work), a heavy radial load with clearance, and the same with
# an axial load (alpha_j rises above alpha_0 and differs from ball to ball).
LOADS = {
    "radial_moderate": dict(s=0.0, Fr=3300.0, Fa=0.0),
    "radial_heavy": dict(s=0.015, Fr=8900.0, Fa=0.0),
    "combined": dict(s=0.015, Fr=8900.0, Fa=4450.0),
}


def build_bearing(s: float):
    """6210-like deep groove ball bearing, assembled for ISO/TS 16281 and not solved.

    Parameters
    ----------
    s : float
        Radial internal clearance [mm], >= 0. It fixes the free contact angle ``alpha_0``.

    Returns
    -------
    axisforge Bearing
        With the ISO/TS 16281 contact data (``cp``, radii, ``alpha_0``); no load is applied.
    """
    catalog = af_c.BearingCatalog(d=50.0, D=90.0, b=20.0, designation="6210", position=0.0,
                                  arrangement="locating", label="6210")
    return af_c.Bearing.assemble(
        family=af_c.DeepGrooveBallFamily(), catalog=catalog,
        geometry=dict(Dw=DW, Dpw=DPW, Z=Z, s=s, contact=af_c.ContactAnalysis.ISO16281,
                      e1=E_STEEL, e2=E_STEEL, nu1=NU_STEEL, nu2=NU_STEEL))
