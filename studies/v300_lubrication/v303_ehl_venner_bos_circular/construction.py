"""PROVISIONAL (scaffold 2026-10-17): structure agreed in debate_gears-lubrication-scaffold_2026-10-17; not validated.

Construction -- the circular EHL contacts of Venner & Bos (1994), ``validation/ehl/venner_bos_circular``.

Named constants only, SI units, each with the line of ``reference_values.csv`` it comes from (header =
line 1). The reference is dimensionless (M, L); the SI point that reproduces it is derived in the script
by ``references.contact.ehl.moes_inverse``. A value not decided yet is ``None`` with its OPEN id.

Model: smooth steel ball on a steel flat, steady, isothermal, Newtonian, Roelands viscosity, density
incompressible or Dowson-Higginson (VennerBos1994 pp. 2-3, 9, 11).
"""
from __future__ import annotations

from pathlib import Path

# --- dimensionless point (Moes, SUM speed convention, VennerBos1994 pp. 14-15) --------------
M_LOW = 50.0                     # Moes load parameter M [-]; CSV L32
M_HIGH = 1000.0                  # Moes load parameter M [-]; CSV L75
L_MOES = 10.0                    # Moes lubricant parameter L [-]; CSV L33, L76

# --- printed lubricant data -------------------------------------------------------------------
ROELANDS_Z = 0.69                # Roelands pressure-viscosity exponent z [-]; CSV L44
# OPEN GL-36: printed G/alpha = 2.78e11 Pa != E' of steel 2.31e11 Pa (E 210 GPa, nu 0.3);
# the G, U, p_h cross-check cannot all pass
ALPHA_PRINTED_PER_PA = 1.7e-8    # printed alpha, unit not printed (1/Pa assumed); CSV L43. OPEN R-091
DH_C0_PA = 0.59e9                # Dowson-Higginson constant [Pa] (= SlipPY nd_dowson_higginson 5.9e8); CSV L45
DH_C1 = 1.34                     # Dowson-Higginson coefficient [-] (rho/rho0 -> 1.34 at high p); CSV L46

# --- printed groups, for the cross-check of the SI point (VennerBos U is the MEAN-speed group) --
# OPEN GL-36: printed G/alpha = 2.78e11 Pa != E' of steel 2.31e11 Pa (E 210 GPa, nu 0.3);
# the G, U, p_h cross-check cannot all pass
G_PRINTED = 4730.0               # materials parameter alpha E' [-]; CSV L35, L78
U_PRINTED = 1.0e-11              # speed parameter eta0 u_mean / (E' R_x) [-]; CSV L36, L79
P_H_LOW_PA = 0.79e9              # Hertz pressure at M = 50 [Pa]; CSV L37
P_H_HIGH_PA = 2.14e9             # Hertz pressure at M = 1000 [Pa] (the value used for rho_bar_c); CSV L81

# --- domain in units of the Hertz radius a (= s_x), inlet on -X ---------------------------------
SPAN_X_LOW = (4.5, 1.5)          # (inlet, outlet) [-]; CSV L39. OPEN R-048 R-073
SPAN_Y_LOW = 3.0                 # half-width across rolling [-]; CSV L39
SPAN_X_HIGH = (2.5, 1.5)         # (inlet, outlet) [-]; CSV L83. OPEN R-048 R-073
SPAN_Y_HIGH = 2.0                # half-width across rolling [-]; CSV L83
N_FINEST = 513                   # nodes per side of the finest grid [-]; CSV L38, L82
N_SMOKE = 65                     # nodes per side of the level-4 grid, the smoke run [-]; CSV L8

DENSITIES = ("incompressible", "dowson_higginson")   # J-V (CSV L30-31, L73-74) recorded only: no SlipPY model

# --- bodies. SlipPY D-H (5.9e8 Pa, LM:216) and Roelands (9.67, LM:103-106)
# are dimensional: the SI point must reproduce the printed p_h (P_H_LOW_PA, P_H_HIGH_PA) as well as M, L, z. OPEN GL-34
# OPEN GL-36: printed G/alpha = 2.78e11 Pa != E' of steel 2.31e11 Pa (E 210 GPa, nu 0.3);
# the G, U, p_h cross-check cannot all pass
E_PA = 210e9                     # Young's modulus of both steel bodies [Pa]
NU = 0.3                         # Poisson's ratio [-]

# --- the SI point: not decided ---------------------------------------------------------------
R_X_M = None                     # ball radius = R_x [m]. OPEN GL-34
U_MEAN_M_PER_S = None            # mean entrainment speed [m/s]. OPEN GL-34
ROELANDS_P0_PA = None            # Roelands reference pressure [Pa], not printed. OPEN GL-34 GL-18
RHO0_KG_PER_M3 = None            # ambient density [kg/m^3], not printed. OPEN R-089

# --- where the scripts write: results/ next to this case, git-ignored (D-0020) ----------------
RESULTS = Path(__file__).resolve().parent / "results"
