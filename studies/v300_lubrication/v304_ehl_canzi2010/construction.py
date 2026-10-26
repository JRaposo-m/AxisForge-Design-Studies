"""PROVISIONAL (scaffold 2026-10-17): structure agreed in debate_gears-lubrication-scaffold_2026-10-17; not validated.

Construction -- the elliptical grid study (Table 1) and the circular contacts (Table 3) of Canzi2010,
``validation/ehl/canzi2010``.

Named constants only, SI units, each with its line in ``reference_values.csv`` (``GRID_`` names) or
``reference_values_circular.csv`` (``CIRC_`` names); header = line 1. Canzi prints the dimensionless
(alpha_bar, M2, D) only; the SI point is derived in the scripts. The rheology Canzi does not print is
assumed here before the run (R-114); ``None`` until decided.

Model: smooth steel body on a steel flat, steady, isothermal, Newtonian, Roelands viscosity,
Dowson-Higginson density, compressible (Canzi2010 p. 3).
"""
from __future__ import annotations

from pathlib import Path

# --- Table 1: elliptical grid study (reference_values.csv) -------------------------------------
GRID_ALPHA_BAR = 10.0                   # alpha p_h [-]; CSV L2
GRID_M2 = 2027.0                        # Moes 2-D load parameter [-] (SUM speed); CSV L3
GRID_D = 0.2                            # R_x / R_y [-] (Canzi convention); CSV L4
GRID_N = (128, 256, 512, 1024, 2048)    # nodes per side of Canzi's grids [-]; CSV L7-11
GRID_SMOKE_N = 128                      # the coarsest grid, the smoke run [-]; CSV L7

# --- Table 3: circular contacts (reference_values_circular.csv) --------------------------------
CIRC_ALPHA_BAR = (20.0, 10.0, 5.0)      # alpha p_h [-]; CSV L3-5
CIRC_M2 = (100.0, 500.0)                # Moes 2-D load parameter [-]; CSV L6-7
CIRC_D = 1.0                            # circular [-]; CSV L8
# (alpha_bar, M2) -> CSV line of "Hc calculated"; the "Moes formula" row is the next line
CIRC_CALCULATED_LINES = {(20.0, 100.0): 9, (20.0, 500.0): 11, (10.0, 100.0): 13, (10.0, 500.0): 15,
                         (5.0, 100.0): 17, (5.0, 500.0): 19}

# --- bodies. SlipPY D-H (5.9e8 Pa, LM:216) and Roelands (9.67, LM:103-106)
# are dimensional: the SI point must reproduce the printed p_h as well as M, L, z. OPEN GL-34
E_PA = 210e9                            # Young's modulus of both steel bodies [Pa]
NU = 0.3                                # Poisson's ratio [-]

# --- assumed before the run: not printed by Canzi (R-114) --------------------------------------
ROELANDS_Z = None                       # Roelands exponent z [-]. OPEN R-114
ROELANDS_P0_PA = None                   # Roelands reference pressure [Pa]. OPEN R-114 GL-34 GL-18
RHO0_KG_PER_M3 = None                   # ambient density [kg/m^3]. OPEN R-114 R-089
SPAN_X = None                           # (inlet, outlet) domain in units of a [-]. OPEN R-114 R-048
SPAN_Y = None                           # half-width across rolling in units of a_y [-]. OPEN R-114
R_X_M = None                            # R_x [m]. OPEN GL-34
U_MEAN_M_PER_S = None                   # mean entrainment speed [m/s]. OPEN GL-34
# Canzi: compressible, D-H (CSV L5; circular CSV L2, L28); the D-H constants are assumed standard
DENSITY = "dowson_higginson"

# --- where the scripts write: results/ next to this case, git-ignored (D-0020) ----------------
RESULTS = Path(__file__).resolve().parent / "results"
