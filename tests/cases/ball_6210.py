"""
6210-like deep groove ball bearing and its load cases, built against the AxisForge API.

Copied from ``studies/v100_dry_contact/v104_iso16281_ball_contact/construction.py`` at the tag
``studies-pre-sweep`` (``build_bearing`` and ``LOADS`` only).

Model
-----
Boundary dimensions of a 6210 (d 50, D 90, B 20 mm), 9 balls of 12.7 mm on a pitch diameter of
70 mm, both groove radii 0.52 Dw (the AxisForge family default), steel on steel. The radial
clearance ``s`` sets the free contact angle ``alpha_0`` (0 for s = 0). The number of balls and the
groove radii are typical values, not catalogue data.

Units
-----
mm, N, MPa (AxisForge); the conversion to SI is made only inside ``slippy_bridge``.
"""
from __future__ import annotations

import axisforge.core as af_c

__all__ = ["build_bearing", "LOADS"]

E_STEEL = 210000.0      # Young's modulus [MPa]
NU_STEEL = 0.3          # Poisson's ratio [-]
DW = 12.7               # ball diameter [mm]
DPW = 70.0              # pitch diameter [mm]
Z = 9                   # balls [-]

# Load case: radial clearance s [mm], radial load Fr [N] (in the XZ plane), axial load Fa [N]; with the axial
# load alpha_j rises above alpha_0 and differs from ball to ball (the other two cases of v104 are not used here).
LOADS = {
    "combined": dict(s=0.015, Fr=8900.0, Fa=4450.0),
}


def build_bearing(s: float, *, position: float = 0.0, arrangement: str = "locating",
                  label: str = "6210"):
    """6210-like deep groove ball bearing, assembled for ISO/TS 16281 and not solved.

    Parameters
    ----------
    s : float
        Radial internal clearance [mm], >= 0. It fixes the free contact angle ``alpha_0``.
    position : float, optional (0.0)
        Axial position of the centre of the bearing width on a shaft [mm].
    arrangement : str, optional ("locating")
        Axial role in the shaft FEM: "locating" fixes v and u, "non-locating" v only.
    label : str, optional ("6210")
        Bearing label.

    Returns
    -------
    axisforge Bearing
        With the ISO/TS 16281 contact data (``cp``, radii, ``alpha_0``); no load is applied.
    """
    catalog = af_c.BearingCatalog(d=50.0, D=90.0, b=20.0, designation="6210", position=position,
                                  arrangement=arrangement, label=label)
    return af_c.Bearing.assemble(
        family=af_c.DeepGrooveBallFamily(), catalog=catalog,
        geometry=dict(Dw=DW, Dpw=DPW, Z=Z, s=s, contact=af_c.ContactAnalysis.ISO16281,
                      e1=E_STEEL, e2=E_STEEL, nu1=NU_STEEL, nu2=NU_STEEL))
