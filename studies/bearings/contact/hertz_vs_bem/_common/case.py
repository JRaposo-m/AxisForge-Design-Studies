"""Contact case definition for the ``hertz_vs_bem`` question (SI units, no AxisForge import).

A :class:`ContactCase` is pure data that mirrors the inputs of ``slippy.contact.hertz_full``: the
principal radii of both bodies, their elastic constants, the normal load, the angle between the
principal axes and the line flag. Every build and the analytic reference start from the same
object, so they all describe the same contact. Derived quantities (relative radii, reduced
modulus, contact semi-axes) are NOT computed here: they come from ``hertz_full``.
"""
from __future__ import annotations

import math

from slippy_bridge import ContactCase   # the class lives in the bridge; the study keeps its constants

__all__ = ["ContactCase", "CASES", "N_POWERS", "N_POWERS_BY_CASE", "EXTENT_IN_A", "EXTENT_IN_A_BY_CASE"]


# =============================================================================
# Inputs  (SI: m, Pa, N)
# =============================================================================

# --- materials (one steel for every body) --------------------------------------
E_STEEL = 210e9                      # Young's modulus [Pa]
NU_STEEL = 0.3                       # Poisson's ratio [-]

# --- bearing geometry (deep groove ball bearing, typical 6210-type values, to confirm) ---
# Same definitions as the AxisForge ball-bearing geometry (Harris / Hamrock-Dowson notation),
# written here by hand so that this study does not import AxisForge.
DW = 12.7e-3                         # ball diameter [m]
DPW = 70.0e-3                        # pitch diameter [m]
F_I = 0.52                           # inner groove radius / ball diameter ri/Dw [-]
ALPHA_0 = 0.0                        # free contact angle [rad]

# --- bodies: principal radii (r_x, r_y) [m]; convex > 0, concave < 0, math.inf = flat ---
BALL = (DW / 2.0, DW / 2.0)
FLAT = (math.inf, math.inf)
INNER_RACEWAY = ((DPW - DW * math.cos(ALPHA_0)) / (2.0 * math.cos(ALPHA_0)),   # rolling radius
                 -F_I * DW)                                                    # groove radius

# --- roller (cylindrical roller bearing, same pitch diameter; placeholder size, to confirm) ---
DWE = 10.0e-3                        # roller diameter [m]
ROLLER = (DWE / 2.0, math.inf)       # (rx, ry): cylinder, no curvature along its axis
INNER_RACEWAY_ROLLER = ((DPW - DWE) / 2.0, math.inf)   # rolling radius of the inner raceway, straight in y

# --- loads ---------------------------------------------------------------------
LOAD_BALL_FLAT = 500.0               # [N]
LOAD_ROLLER_PER_LENGTH = 5.0e4       # [N/m] line contact; PLACEHOLDER (= 500 N over a 10 mm roller)
LOAD_BALL_RACEWAY = 500.0            # [N]  PLACEHOLDER: set the value of the S4 reference case

# --- grid (study constants) ----------------------------------------------------
N_POWERS = (6, 7, 8, 9)              # nodes per side N = 2**n (even N puts the apex on a node)
N_POWERS_BY_CASE = {                 # overrides N_POWERS for a case (the ellipse needs finer grids)
    "ball_raceway": (8, 9, 10),
}
EXTENT_IN_A_BY_CASE = {"ball_raceway": 3.0}   # the ball exists only up to r = 6.35 mm
EXTENT_IN_A = 6.0                    # half-width of the square domain in units of the semi-axis a [-]


# =============================================================================
# Cases (one per type folder)
# =============================================================================

CASES = {
    "ball_flat": ContactCase("ball_flat", r1=BALL, r2=FLAT, moduli=E_STEEL, v=NU_STEEL,
                             load=LOAD_BALL_FLAT),
    "ball_raceway": ContactCase("ball_raceway", r1=BALL, r2=INNER_RACEWAY, moduli=E_STEEL,
                               v=NU_STEEL, load=LOAD_BALL_RACEWAY),
    "roller_raceway": ContactCase("roller_raceway", r1=ROLLER, r2=INNER_RACEWAY_ROLLER,
                                  moduli=E_STEEL, v=NU_STEEL, load=LOAD_ROLLER_PER_LENGTH, line=True),
}