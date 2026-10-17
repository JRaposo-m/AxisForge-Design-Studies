"""PROVISIONAL (scaffold 2026-10-17): structure agreed in debate_gears-lubrication-scaffold_2026-10-17; not validated.

Construction -- elliptical EHL contacts, film versus ellipticity D (Canzi2010 Tables 4-6),
``validation/ehl/ellipse_ratio``.

Named constants only, SI units, each with its line in ``reference_values.csv`` (header = line 1). Canzi
prints (alpha_bar, M2c, D) only; the SI point is derived in the script. The Hamrock-Dowson (1977) rows of
the same CSV (V-EHL-10, L138 onwards) have no script until mean or sum speed is settled (R-092).

Model: smooth steel ellipsoid on a steel flat, R_x along rolling (array axis 0) and R_y = R_x / D across
it, steady, isothermal, Newtonian, Roelands viscosity, Dowson-Higginson density, compressible.
"""
from __future__ import annotations

from pathlib import Path

# --- the D sweep, in the row order of every table block -----------------------------------------
D_VALUES = (1.0, 0.5, 0.2, 0.1, 0.05, 0.01)   # Canzi D = R_x / R_y [-]; CSV L2-7

# --- Tables 4 and 5: (alpha_bar, M2c) -> first CSV line of the H_c, H_m and H_c/H_m blocks (6 rows each, D order)
TABLE4 = (20.0, 500.0)           # alpha_bar [-] CSV L9; M2c = M2 at D = 1 [-] CSV L10
TABLE5 = (10.0, 100.0)           # alpha_bar [-] CSV L29; M2c [-] CSV L30
TABLE45_LINES = {TABLE4: {"H_c": 11, "H_m": 17, "H_c/H_m": 23}, TABLE5: {"H_c": 31, "H_m": 37, "H_c/H_m": 43}}

# --- Table 6: (alpha_bar, M2c) -> first CSV line of "H_Ccalc" (6 rows, D order); CELINE rows follow (+6)
TABLE6_CALC_LINES = {(10.0, 100.0): 50, (20.0, 500.0): 63, (5.0, 100.0): 76, (10.0, 500.0): 89,
                     (0.0, 100.0): 102}          # block headers CSV L49, L62, L75, L88, L101
# alpha_bar = 0 is isoviscous: ehl.run refuses z = 0 and alpha < 1e-9 (GL-17), so the block is recorded only.
TABLE6_RUN = ((10.0, 100.0), (20.0, 500.0), (5.0, 100.0), (10.0, 500.0))   # OPEN GL-17: isoviscous block

DENSITY = "dowson_higginson"     # Canzi: compressible, D-H (CSV L121); constants assumed standard

# --- bodies. SlipPY D-H (5.9e8 Pa, LM:216) and Roelands (9.67, LM:103-106)
# are dimensional: the SI point must reproduce the printed p_h as well as M, L, z. OPEN GL-34
E_PA = 210e9                     # Young's modulus of both steel bodies [Pa]
NU = 0.3                         # Poisson's ratio [-]

# --- assumed before the run: not printed by Canzi (R-114) --------------------------------------
ROELANDS_Z = None                # Roelands exponent z [-]. OPEN R-114
ROELANDS_P0_PA = None            # Roelands reference pressure [Pa]. OPEN R-114 GL-34 GL-18
RHO0_KG_PER_M3 = None            # ambient density [kg/m^3]. OPEN R-114 R-089
SPAN_X = None                    # (inlet, outlet) in units of a_x [-]. OPEN R-114 R-048
SPAN_Y = None                    # half-width across rolling in units of a_y [-]. OPEN R-114
R_X_M = None                     # R_x [m]. OPEN GL-34
U_MEAN_M_PER_S = None            # mean entrainment speed [m/s], along +x. OPEN GL-34 R-073

# --- where the scripts write: results/ next to this case, git-ignored (D-0020) ----------------
RESULTS = Path(__file__).resolve().parent / "results"
