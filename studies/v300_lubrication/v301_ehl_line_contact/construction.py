"""PROVISIONAL (scaffold 2026-10-17): structure agreed in debate_gears-lubrication-scaffold_2026-10-17; not validated.

Construction -- the EHL line contacts of Venner & Bos (1994, Tables 1-2) and Canzi2010 (Table 2),
``validation/ehl/line_contact``.

Named constants only, SI units, each with its line in ``reference_values.csv`` (header = line 1). Both
references are dimensionless; the SI point is derived in the script. ``None`` = not decided, with its OPEN id.

Model: smooth steel cylinder on a steel flat, infinitely long (periodic along the roller axis, array
axis 1), steady, isothermal, Newtonian, Roelands viscosity, density incompressible or Dowson-Higginson.
"""
from __future__ import annotations

from pathlib import Path

# --- VennerBos1994 Tables 1-2: dimensionless points (Moes, SUM speed, pp. 14-15) ---------------
M_LOW = 20.0                     # Moes line load parameter M [-]; CSV L50
M_HIGH = 200.0                   # Moes line load parameter M [-]; CSV L133
L_MOES = 10.0                    # Moes lubricant parameter L [-]; CSV L51, L134
ROELANDS_Z = 0.69                # Roelands exponent z [-]; CSV L72, L152
# OPEN GL-36: printed G/alpha = 2.78e11 Pa != E' of steel 2.31e11 Pa (E 210 GPa, nu 0.3);
# the G, U, p_h cross-check cannot all pass
ALPHA_PRINTED_PER_PA = 1.7e-8    # printed alpha, unit not printed (1/Pa assumed); CSV L71. OPEN R-091
# OPEN GL-36: printed G/alpha = 2.78e11 Pa != E' of steel 2.31e11 Pa (E 210 GPa, nu 0.3);
# the G, U, p_h cross-check cannot all pass
G_PRINTED = 4730.0               # materials parameter alpha E' [-]; CSV L53
U_PRINTED = 1.0e-11              # MEAN-speed group eta0 u_mean / (E' R) [-]; CSV L54
P_H_LOW_PA = 1.05e9              # Hertz pressure at M = 20 [Pa]; CSV L55
P_H_HIGH_PA = 3.3e9              # Hertz pressure at M = 200 [Pa]; CSV L138
SPAN_X_LOW = (4.0, 1.5)          # (inlet, outlet) in units of the half-width b [-]; CSV L56. OPEN R-048 R-073
SPAN_X_HIGH = (2.5, 1.5)         # (inlet, outlet) [-]; CSV L139. OPEN R-048 R-073
N_PRINTED_LOW = 28673            # nodes of Venner's finest grid, level 12 [-]; CSV L58 (record)
N_PRINTED_HIGH = 57345           # nodes of Venner's finest grid, level 13 [-]; CSV L141 (record)
DENSITIES = ("incompressible", "dowson_higginson")   # J-V rows recorded only: no SlipPY model

# --- Canzi2010 Table 2: one compared point ------------------------------------------------------
CANZI_ALPHA_BAR = 20.0           # alpha p_h [-]; CSV L160
CANZI_M1 = 17.97                 # Moes 1-D load parameter [-] (SUM speed); CSV L163
CANZI_DENSITY = "dowson_higginson"   # Canzi: compressible (CSV L159, L184); D-H constants assumed standard

# --- bodies. SlipPY D-H (5.9e8 Pa, LM:216) and Roelands (9.67, LM:103-106)
# are dimensional: the SI point must reproduce the printed p_h (P_H_LOW_PA, P_H_HIGH_PA) as well as M, L, z. OPEN GL-34
# OPEN GL-36: printed G/alpha = 2.78e11 Pa != E' of steel 2.31e11 Pa (E 210 GPa, nu 0.3);
# the G, U, p_h cross-check cannot all pass
E_PA = 210e9                     # Young's modulus of both steel bodies [Pa]
NU = 0.3                         # Poisson's ratio [-]

# --- the SI point and the SlipPY grid: not decided ---------------------------------------------
R_X_M = None                     # cylinder radius = R_x [m]. OPEN GL-34
U_MEAN_M_PER_S = None            # mean entrainment speed [m/s]. OPEN GL-34
ROELANDS_P0_PA = None            # Roelands reference pressure [Pa], not printed. OPEN GL-34 GL-18
RHO0_KG_PER_M3 = None            # ambient density [kg/m^3], not printed. OPEN R-089
N_X_RUN = None                   # SlipPY nodes along rolling [-]; Venner's 28673 is out of reach. OPEN R-115 GL-16
CANZI_SPAN_X = None              # Canzi's domain is not printed. OPEN R-114

# --- where the scripts write: results/ next to this case, git-ignored (D-0020) ----------------
RESULTS = Path(__file__).resolve().parent / "results"
