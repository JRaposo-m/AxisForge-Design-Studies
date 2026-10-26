"""PROVISIONAL (scaffold 2026-10-17): structure agreed in debate_gears-lubrication-scaffold_2026-10-17; not validated.

Construction -- the FZG test gears C40 (spur) and H501 (helical) of FEUP and their load stages,
``validation/gears/fzg_geometry_ph``.

Named constants, each with its line in ``reference_values.csv`` (header = line 1), and the only two
functions of a construction file (plan sec 3.9; reason: ``02_ph`` and the tests both need them):

* :func:`build_pair` -- the AxisForge ``GearPair``, assembled and NOT solved;
* :func:`path_times` -- the times of the named points of the path of contact, the ``t_s`` that
  AxisForge ``contact_inputs`` takes.

Model: involute spur or helical pair, rigid unmodified flanks, steel 20MnCr5 (E and nu not printed,
R-118). Torque and speed are printed on the WHEEL (Martins2015 Table 4); the case converts them to
gear 1 (R-124). Units: AxisForge (mm, N, N m, MPa, rpm); SI only inside ``slippy_bridge``.
"""
from __future__ import annotations

import math
from pathlib import Path

from axisforge.core.machine_elements.gears import CylindricalFamily, CylindricalGearData, Gear, GearPair

__all__ = ["build_pair", "path_times", "FAMILY", "PAIRS", "STEEL_ID", "GEAR_E_MPA", "GEAR_NU", "WHEEL_TORQUE_NM",
           "WHEEL_SPEED_RPM", "RESULTS"]

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
# E, nu given to gear_contacts: ISO 6336-2:2006 Table 1 generic steel, assumed (README, R-118)
GEAR_E_MPA = (206000.0, 206000.0)   # Young's modulus of gear 1, gear 2 [MPa]
GEAR_NU = (0.3, 0.3)                # Poisson's ratio of gear 1, gear 2 [-]

# --- roughness: three printed values for C40 (record only) ------------------------------------
RA_C40_UM = {"Fernandes2015b": 0.7,          # CSV L15
             "Martins2015": 0.9,             # CSV L115
             "run-in axial": 0.35,           # CSV L148
             "run-in radial": 0.85}          # CSV L149. OPEN R-123: which one

# --- load stages and speed, on the WHEEL (Martins2015 Table 4) --------------------------------
WHEEL_TORQUE_NM = {"K5": 104.97,             # CSV L168
                   "K7": 198.68,             # CSV L178
                   "K9": 323.27}             # CSV L188 (K1 4.95 N m, CSV L158, not used)
WHEEL_SPEED_RPM = (200.0, 400.0, 1200.0)     # CSV L198-200
DRIVER = "gear1"                # pinion drives. OPEN Q-09 R-124: FEUP does not say which gear drives

# --- path points --------------------------------------------------------------------------------
DELTA_S_MM = 1e-6   # one-sided step either side of B and D [mm]: >> AxisForge end tolerance 1e-9 p_bt (contact.py:82)

# --- where the scripts write: results/ next to this case, git-ignored (D-0020) ----------------
RESULTS = Path(__file__).resolve().parent / "results"


def build_pair(name: str) -> GearPair:
    """The FZG pair ``name`` ("C40" or "H501"), assembled with the FEUP profile shifts and NOT solved.

    The working centre distance is the zero-backlash one that AxisForge derives from x1 + x2
    (``al=None``); ``01_geometry`` compares it with the printed 91.5 mm. Each gear carries
    ``material_id = STEEL_ID``, not registered in AxisForge: E and nu are ``GEAR_E_MPA``, ``GEAR_NU``.

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
    Fernandes2015b Table 1; Martins2015 Table 3 (``reference_values.csv`` L2-29, L100-131).
    """
    p = PAIRS[name]
    gears = [Gear.assemble(FAMILY, CylindricalGearData(p["mn"], z, x=x, b=p["b"], alpha_n_deg=p["alpha_n_deg"],
                                                       beta_n_deg=p["beta_n_deg"], hand=hand,
                                                       label=f"{name}_{member}", material_id=STEEL_ID))
             for z, x, hand, member in zip(p["z"], p["x"], p["hand"], ("pinion", "wheel"))]
    return GearPair.assemble(FAMILY, gear1=gears[0], gear2=gears[1], label=name)


def path_times(pair: GearPair, n1_rpm: float) -> dict[str, float]:
    """Times [s] at which the contact point of one tooth pair passes the named points of the path.

    With gear 1 driving, the contact point travels along the line of action from A (start of meshing)
    to E (end) at the speed of the base circle of gear 1, ``v_n = omega1 r_b1`` (ISO 21771 Eq. (105)),
    so ``t_P = abs(g_P - g_A) / v_n``. At B and D the load share jumps (double to single tooth contact
    and back); evaluated exactly there AxisForge gives the double-pair share, so the one-sided limits
    are taken ``DELTA_S_MM`` either side: B- and D+ are in double contact, B+ and D- in single contact
    (GL-09).
    B+ and D- are single-pair only for spur (eps_alpha < 2, eps_beta = 0); for a helical pair the labels are
    transverse positions only.

    Parameters
    ----------
    pair : GearPair
        Spur or helical pair with ``eps_alpha < 2``.
    n1_rpm : float
        Speed of gear 1 [rpm], > 0.

    Returns
    -------
    dict
        Labels "A", "B-", "B+", "C", "D-", "D+", "E" -> time since A [s], in path order.

    Raises
    ------
    ValueError
        If ``pair.epslon_alpha >= 2``: then B comes after D and there is no single-pair zone (GL-25).

    References
    ----------
    ISO 21771:2007, 5.6.1, Eq. (105); AxisForge ``CylindricalFamily.path_points``, ``contact.py:188``.
    """
    if pair.epslon_alpha >= 2.0:
        raise ValueError(f"{pair.label}: eps_alpha = {pair.epslon_alpha:.4f} >= 2, no single-pair zone B..D (GL-25)")
    g = CylindricalFamily.path_points(pair, DRIVER)                  # positions g_y from C [mm]
    v_n = n1_rpm * 2.0 * math.pi / 60.0 * pair.gear1.rb             # [mm/s]; duplicates contact.py:188, OPEN GL-24
    t = {k: abs(g[k] - g["A"]) / v_n for k in "ABCDE"}
    dt = DELTA_S_MM / v_n
    return {"A": t["A"], "B-": t["B"] - dt, "B+": t["B"] + dt, "C": t["C"], "D-": t["D"] - dt, "D+": t["D"] + dt,
            "E": t["E"]}
