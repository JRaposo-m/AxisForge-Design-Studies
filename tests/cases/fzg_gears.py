"""
The FZG test gears C40 (spur) and H501 (helical) of FEUP and their load stages.

Copied from ``studies/v100_dry_contact/v105_fzg_geometry_ph/construction.py`` at the tag
``studies-pre-sweep`` (``build_pair`` and the constants the bridge tests read). The ``CSV L..``
line numbers refer to that case's ``reference_values.csv`` (header = line 1), also in the tag.

Model: involute spur or helical pair, rigid unmodified flanks, steel 20MnCr5 (E and nu not printed,
R-118). Torque is printed on the WHEEL (Martins2015 Table 4); the tests convert it to gear 1 (R-124).
Units: AxisForge (mm, N, N m, MPa); SI only inside ``slippy_bridge``.
"""
from __future__ import annotations

from axisforge.core.machine_elements.gears import CylindricalFamily, CylindricalGearData, Gear, GearPair

__all__ = ["build_pair", "FAMILY", "PAIRS", "STEEL_ID", "GEAR_E_MPA", "GEAR_NU", "WHEEL_TORQUE_NM", "DRIVER",
           "DELTA_S_MM"]

FAMILY = CylindricalFamily()

# --- gear geometry (Fernandes2015b Table 1 = Martins2015 Table 3) [mm, deg, -] -------------------
# OPEN R-120: alpha and m printed without saying normal or transverse; read as normal (mn, alpha_n).
PAIRS = {
    "C40": dict(mn=4.5,                    # module [mm]; CSV L4, L102
                z=(16, 24),                # teeth pinion, wheel [-]; CSV L2-3, L100-101
                alpha_n_deg=20.0,          # pressure angle [deg]; CSV L6, L104
                beta_n_deg=0.0,            # helix angle [deg]; CSV L7, L105
                b=40.0,                    # face width [mm]; CSV L8, L106
                x=(0.1817, 0.1715),        # profile shift coefficients [-]; CSV L9-10, L107-108
                hand=(0, 0),               # spur: no hand
                a_printed=91.5),           # centre distance [mm], compared, not imposed; CSV L5, L103
    "H501": dict(mn=3.5,                   # CSV L18, L118
                 z=(20, 30),               # CSV L16-17, L116-117
                 alpha_n_deg=20.0,         # CSV L20, L120
                 beta_n_deg=15.0,          # CSV L21, L121
                 b=23.0,                   # CSV L22, L122
                 x=(0.1381, 0.1319),       # FEUP x, never the GEARpie H501 entry (GL-11); CSV L23-24, L123-124
                 hand=(1, -1),             # OPEN GL-38: not printed; opposite hands of an external pair
                 a_printed=91.5),          # CSV L19, L119
}

# --- material (20MnCr5, CSV L114; no E or nu printed) ---------------------------------------
STEEL_ID = "steel_20MnCr5"       # material_id of both gears, not registered in AxisForge (OPEN R-118)
# E, nu given to the bridge: ISO 6336-2:2006 Table 1 generic steel, assumed (R-118)
GEAR_E_MPA = (206000.0, 206000.0)   # Young's modulus of gear 1, gear 2 [MPa]
GEAR_NU = (0.3, 0.3)                # Poisson's ratio of gear 1, gear 2 [-]

# --- load stages, on the WHEEL (Martins2015 Table 4) ------------------------------------------
WHEEL_TORQUE_NM = {"K5": 104.97,             # CSV L168
                   "K7": 198.68,             # CSV L178
                   "K9": 323.27}             # CSV L188
DRIVER = "gear1"                # pinion drives. OPEN Q-09 R-124: FEUP does not say which gear drives

DELTA_S_MM = 1e-6   # one-sided step either side of B and D [mm]: >> AxisForge end tolerance 1e-9 p_bt (contact.py:82)


def build_pair(name: str) -> GearPair:
    """The FZG pair ``name`` ("C40" or "H501"), assembled with the FEUP profile shifts and NOT solved.

    The working centre distance is the zero-backlash one that AxisForge derives from x1 + x2
    (``al=None``). Each gear carries ``material_id = STEEL_ID``, not registered in AxisForge: E and
    nu are ``GEAR_E_MPA``, ``GEAR_NU``.

    Parameters
    ----------
    name : str
        "C40" (spur, z 16/24, mn 4.5 mm) or "H501" (helical 15 deg, z 20/30, mn 3.5 mm).

    Returns
    -------
    GearPair
        AxisForge pair, gear 1 = pinion, mm and deg.

    Raises
    ------
    KeyError
        If ``name`` is not a key of ``PAIRS``.

    References
    ----------
    Fernandes2015b Table 1; Martins2015 Table 3.
    """
    p = PAIRS[name]
    gears = [Gear.assemble(FAMILY, CylindricalGearData(p["mn"], z, x=x, b=p["b"], alpha_n_deg=p["alpha_n_deg"],
                                                       beta_n_deg=p["beta_n_deg"], hand=hand,
                                                       label=f"{name}_{member}", material_id=STEEL_ID))
             for z, x, hand, member in zip(p["z"], p["x"], p["hand"], ("pinion", "wheel"))]
    return GearPair.assemble(FAMILY, gear1=gears[0], gear2=gears[1], label=name)
