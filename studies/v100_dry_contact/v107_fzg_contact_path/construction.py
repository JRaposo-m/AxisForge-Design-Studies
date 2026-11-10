"""The systems of v107: the FZG test gears C40 (spur) and H501 (helical) of FEUP, returned NOT solved.

Ported from ``studies/v100_dry_contact/v105_fzg_geometry_ph/construction.py`` (tag ``studies-pre-sweep``,
as ``tests/cases/fzg_gears.py`` of the bridge tests). AxisForge units: mm, N, N.m, MPa, degrees.

FLAGS (v_prerequisites row 6): the wheel tip is built from x (Fernandes prints 115.54, a misprint of 118.54);
the torques are printed on the WHEEL (R-124), gear 1 gets ``T2 z1 / z2``; E and nu are not printed (R-118):
ISO 6336-2 generic steel 206 000 MPa, 0.3 assumed; the pinion drives (Q-09: FEUP does not say).
"""
from __future__ import annotations

from axisforge.core.machine_elements.gears import CylindricalFamily, CylindricalGearData, Gear, GearPair

__all__ = ["build_pair", "FAMILY", "PAIRS", "WHEEL_TORQUE_NM", "E_MPA", "NU", "DRIVER", "STEEL_ID"]

FAMILY = CylindricalFamily()
# Fernandes2015b Table 1 = Martins2015 Table 3 (PDF p. 4); alpha and m read as normal (R-120)
PAIRS = {
    "C40": dict(mn=4.5, z=(16, 24), alpha_n_deg=20.0, beta_n_deg=0.0, b=40.0, x=(0.1817, 0.1715), hand=(0, 0)),
    "H501": dict(mn=3.5, z=(20, 30), alpha_n_deg=20.0, beta_n_deg=15.0, b=23.0, x=(0.1381, 0.1319),
                 hand=(1, -1)),     # OPEN GL-38: hands not printed; opposite hands of an external pair
}
WHEEL_TORQUE_NM = {"K5": 104.97, "K7": 198.68, "K9": 323.27}   # Martins2015 Table 4, on the wheel (R-124)
E_MPA, NU = 206000.0, 0.3        # assumed (R-118): ISO 6336-2 generic steel
DRIVER = "gear1"                 # pinion drives (Q-09)
STEEL_ID = "steel_20MnCr5"       # not registered in AxisForge (R-118)


def build_pair(name: str) -> GearPair:
    """The FZG pair ``name`` ("C40" or "H501") with the FEUP profile shifts, zero backlash, NOT solved."""
    p = PAIRS[name]
    gears = [Gear.assemble(FAMILY, CylindricalGearData(p["mn"], z, x=x, b=p["b"], alpha_n_deg=p["alpha_n_deg"],
                                                       beta_n_deg=p["beta_n_deg"], hand=hand,
                                                       label=f"{name}_{member}", material_id=STEEL_ID))
             for z, x, hand, member in zip(p["z"], p["x"], p["hand"], ("pinion", "wheel"))]
    return GearPair.assemble(FAMILY, gear1=gears[0], gear2=gears[1], label=name)
