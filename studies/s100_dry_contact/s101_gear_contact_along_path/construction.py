"""PROVISIONAL (scaffold 2026-10-17): structure agreed in debate_gears-lubrication-scaffold_2026-10-17; not validated.

Construction -- the FZG C40 spur pair, its path times and the FEUP oils, for the study
``studies/gears/contact_along_path`` (a demonstration: no published film along a gear path, R-145).

The pair and the path times are this study's OWN copy of ``validation/gears/fzg_geometry_ph/construction.py``
(a study imports no validation case, ``studies/README.md``); the oil data are the rows of this study's
``reference_values.csv`` (header = line 1), converted to SI here, once:

* density g/cm^3 -> kg/m^3 (x 1e3); kinematic viscosity cSt -> m^2/s (x 1e-6);
* pressure-viscosity coefficient: the printed number x 1e-8, read as 1/Pa (scale printed, unit not; R-295).

Spur only (R-027). AxisForge units for the pair (mm, N, N m, rpm); SI for the oils.
"""
from __future__ import annotations

import math
from pathlib import Path

from axisforge.core.machine_elements.gears import CylindricalFamily, CylindricalGearData, Gear, GearPair

__all__ = ["build_pair", "path_times", "FAMILY", "OIL_MARTINS2015", "OIL_FERNANDES2015A", "OIL", "RESULTS"]

FAMILY = CylindricalFamily()

_G_PER_CM3 = 1e3          # [kg/m^3 per g/cm^3]
_CST = 1e-6               # [m^2/s per cSt]
# eta0 [Pa s] = nu80 [m^2/s] x rho80 [kg/m^3] (dynamic = kinematic x density): None while rho80 is unknown. OPEN R-295
_ALPHA_SCALE = 1e-8       # printed "(alpha x 10^-8)": number x 1e-8, read as [1/Pa]. OPEN R-295 R-091

# --- C40 (Fernandes2015b Table 1 = Martins2015 Table 3; rows in validation/gears/fzg_geometry_ph) --------
C40 = dict(mn=4.5, z=(16, 24), alpha_n_deg=20.0, b=40.0, x=(0.1817, 0.1715))   # [mm, -, deg, mm, -]

STEEL_ID = "steel_20MnCr5"       # material_id of both gears, not registered in AxisForge (OPEN R-118)
# E, nu given to gear_contacts: ISO 6336-2:2006 Table 1 generic steel, assumed (README, R-118)
GEAR_E_MPA = (206000.0, 206000.0)   # Young's modulus of gear 1, gear 2 [MPa]
GEAR_NU = (0.3, 0.3)                # Poisson's ratio of gear 1, gear 2 [-]

# --- operating point (not in this study's sources) -----------------------------------------------
WHEEL_TORQUE_NM = None          # FZG stage torque on the wheel [N m]. OPEN GL-39
WHEEL_SPEED_RPM = None          # wheel speed [rpm]. OPEN GL-39
DRIVER = "gear1"                # pinion drives. OPEN Q-09 R-124
OIL_TEMPERATURE_C = 80.0        # oil injection temperature [deg C] (fzg_geometry_ph CSV L202); the oil rows are at 80 C
DELTA_S_MM = 1e-6               # one-sided step either side of B and D [mm] (GL-09)

# --- oils at 80 C: two sources, both kept (R-295) ------------------------------------------------
# eta0, the Roelands z and rho(80 C) are not printed: None until chosen. OPEN R-295
_UNSET = {"eta0_Pa_s": None, "roelands_z": None, "rho80_kg_per_m3": None}
OIL_MARTINS2015 = {   # Martins2015 Table 1
    "MINR": {"rho15_kg_per_m3": 0.902 * _G_PER_CM3,       # CSV L3
             "nu80_m2_per_s": 43.9 * _CST,                 # CSV L5
             "alpha80_per_Pa": 1.68 * _ALPHA_SCALE,        # CSV L7
             **_UNSET},
    "ESTR": {"rho15_kg_per_m3": 0.915 * _G_PER_CM3,       # CSV L11
             "nu80_m2_per_s": 56.1 * _CST,                 # CSV L13
             "alpha80_per_Pa": 1.14 * _ALPHA_SCALE,        # CSV L15
             **_UNSET},
    "PAOR": {"rho15_kg_per_m3": 0.859 * _G_PER_CM3,       # CSV L19
             "nu80_m2_per_s": 60.4 * _CST,                 # CSV L21
             "alpha80_per_Pa": 1.28 * _ALPHA_SCALE,        # CSV L23
             **_UNSET},
    "MINE": {"rho15_kg_per_m3": 0.893 * _G_PER_CM3,       # CSV L27
             "nu80_m2_per_s": 66.5 * _CST,                 # CSV L29
             "alpha80_per_Pa": 1.26 * _ALPHA_SCALE,        # CSV L31
             **_UNSET},
    "PAGD": {"rho15_kg_per_m3": 1.059 * _G_PER_CM3,       # CSV L35
             "nu80_m2_per_s": 78.9 * _CST,                 # CSV L37
             "alpha80_per_Pa": 1.05 * _ALPHA_SCALE,        # CSV L39
             **_UNSET},
}
OIL_FERNANDES2015A = {   # Fernandes2015a Tables 1, 4 (no 80 C viscosity; D341 m, n given, its equation not)
    "MINR": {"rho15_kg_per_m3": 0.902 * _G_PER_CM3,       # CSV L43
             "nu70_m2_per_s": 65.81 * _CST,                # CSV L46
             "nu100_m2_per_s": 22.33 * _CST,               # CSV L47
             "d341_m": 9.066, "d341_n": 3.473,             # CSV L48, L49
             "alpha_FTM80_per_Pa": 1.403 * _ALPHA_SCALE,   # CSV L110 (fitted to film; no unit printed)
             "alpha_Gold80_per_Pa": 1.667 * _ALPHA_SCALE,  # CSV L114
             **_UNSET},
    "ESTR": {"rho15_kg_per_m3": 0.915 * _G_PER_CM3,       # CSV L55
             "nu70_m2_per_s": 77.48 * _CST,                # CSV L58
             "nu100_m2_per_s": 34.85 * _CST,               # CSV L59
             "d341_m": 7.582, "d341_n": 2.880,             # CSV L60, L61
             "alpha_FTM80_per_Pa": 1.038 * _ALPHA_SCALE,   # CSV L111
             "alpha_Gold80_per_Pa": 1.131 * _ALPHA_SCALE,  # CSV L115
             **_UNSET},
    "PAOR": {"rho15_kg_per_m3": 0.859 * _G_PER_CM3,       # CSV L67
             "nu70_m2_per_s": 84.99 * _CST,                # CSV L70
             "nu100_m2_per_s": 33.33 * _CST,               # CSV L71
             "d341_m": 7.351, "d341_n": 2.787,             # CSV L72, L73
             "alpha_FTM80_per_Pa": 1.053 * _ALPHA_SCALE,   # CSV L112
             "alpha_Gold80_per_Pa": 1.262 * _ALPHA_SCALE,  # CSV L116
             **_UNSET},
    "MINE": {"rho15_kg_per_m3": 0.893 * _G_PER_CM3,       # CSV L79
             "nu70_m2_per_s": 93.19 * _CST,                # CSV L82
             "nu100_m2_per_s": 37.13 * _CST,               # CSV L83
             "d341_m": 7.048, "d341_n": 2.663,             # CSV L84, L85
             "alpha_FTM80_per_Pa": 0.653 * _ALPHA_SCALE,   # CSV L113
             "alpha_Gold80_per_Pa": 1.112 * _ALPHA_SCALE,  # CSV L117
             **_UNSET},
}
OIL = None                      # (source, oil, alpha key) of the run. OPEN R-295: which oil, which alpha

# --- EHL run settings ----------------------------------------------------------------------------
# Roelands p0 [Pa] is not independent: derived once as z (ln eta0 + 9.67) / alpha_printed when z and eta0 are set,
# so that SlipPY's initial slope equals the printed alpha that the Moes L uses. OPEN GL-40 R-295
ROELANDS_P0_PA = None
SPAN_X = None                   # (inlet, outlet) domain in units of the half-width b [-]. OPEN R-048
FEUP_C0 = None                  # C_0 of Fernandes2015a Eq. (3), not printed (CSV L176). OPEN GL-14

# --- where the scripts write: results/ next to this study, git-ignored (D-0020) -----------------
RESULTS = Path(__file__).resolve().parent / "results"


def build_pair() -> GearPair:
    """The FZG C40 spur pair (z 16/24, mn 4.5 mm, FEUP x), assembled and NOT solved.

    Own copy of ``validation/gears/fzg_geometry_ph`` ``build_pair("C40")`` (studies import no case). The
    gears carry ``material_id = STEEL_ID``, not registered: E and nu are ``GEAR_E_MPA``, ``GEAR_NU``.

    Returns
    -------
    GearPair
        AxisForge pair, gear 1 = pinion, mm and deg.

    References
    ----------
    Fernandes2015b Table 1; Martins2015 Table 3.
    """
    gears = [Gear.assemble(FAMILY, CylindricalGearData(C40["mn"], z, x=x, b=C40["b"], alpha_n_deg=C40["alpha_n_deg"],
                                                       label=f"C40_{member}", material_id=STEEL_ID))
             for z, x, member in zip(C40["z"], C40["x"], ("pinion", "wheel"))]
    return GearPair.assemble(FAMILY, gear1=gears[0], gear2=gears[1], label="C40")


def path_times(pair: GearPair, n1_rpm: float) -> dict[str, float]:
    """Times [s] at which the contact point passes A, B-, B+, C, D-, D+, E (gear 1 driving).

    Own copy of ``validation/gears/fzg_geometry_ph`` ``path_times``: the contact point moves along the
    line of action at ``v_n = omega1 r_b1`` (ISO 21771 Eq. (105)); B-/B+ and D-/D+ are the one-sided
    limits ``DELTA_S_MM`` either side of the load-share jumps (GL-09).
    B+ and D- are single-pair only for spur (eps_alpha < 2, eps_beta = 0); for a helical pair the labels are
    transverse positions only.

    Parameters
    ----------
    pair : GearPair
        Spur pair with ``eps_alpha < 2``.
    n1_rpm : float
        Speed of gear 1 [rpm], > 0.

    Returns
    -------
    dict
        Label -> time since A [s], in path order.

    Raises
    ------
    ValueError
        If ``pair.epslon_alpha >= 2`` (no single-pair zone, GL-25).
    """
    if pair.epslon_alpha >= 2.0:
        raise ValueError(f"{pair.label}: eps_alpha = {pair.epslon_alpha:.4f} >= 2, no single-pair zone B..D (GL-25)")
    g = CylindricalFamily.path_points(pair, DRIVER)                  # positions g_y from C [mm]
    v_n = n1_rpm * 2.0 * math.pi / 60.0 * pair.gear1.rb             # [mm/s]; duplicates contact.py:188, OPEN GL-24
    t = {k: abs(g[k] - g["A"]) / v_n for k in "ABCDE"}
    dt = DELTA_S_MM / v_n
    return {"A": t["A"], "B-": t["B"] - dt, "B+": t["B"] + dt, "C": t["C"], "D-": t["D"] - dt, "D+": t["D"] + dt,
            "E": t["E"]}
