"""Constants of the ``hertz_vs_bem`` study (SI units, no AxisForge import).

The three contacts (:data:`CASES`) are written here by hand as :class:`ContactCase`, so that their
``case_hash`` does not depend on AxisForge and stays the one of the saved results. Scripts 02 and
03 check that the AxisForge bearings of ``construction.py`` give the same contacts through the
bridge. Derived quantities (relative radii, semi-axes, Hertz pressure) are not computed here:
they come from ``slippy_bridge.contact``.
"""
from __future__ import annotations

import math
from pathlib import Path

from slippy_bridge.contact import ContactCase

__all__ = ["CASES", "H_LEVELS", "SPAN_X_0", "SPAN_X_LEVELS", "RESULTS"]


# =============================================================================
# Inputs  (SI: m, Pa, N)
# =============================================================================

# --- materials (one steel for every body) --------------------------------------
E_STEEL = 210e9                      # Young's modulus [Pa]
NU_STEEL = 0.3                       # Poisson's ratio [-]

# --- bearing geometry (deep groove ball bearing, typical 6210-type values, to confirm) ---
# Same definitions as the AxisForge ball-bearing geometry (Harris / Hamrock-Dowson notation).
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
ROLLER = (DWE / 2.0, math.inf)       # (r_x, r_y): cylinder, no curvature along its axis
INNER_RACEWAY_ROLLER = ((DPW - DWE) / 2.0, math.inf)   # rolling radius of the inner raceway, straight in y

# --- loads ---------------------------------------------------------------------
LOAD_BALL_FLAT = 500.0               # [N]
LOAD_ROLLER_PER_LENGTH = 5.0e4       # [N/m] line contact; PLACEHOLDER (= 500 N over a 10 mm roller)
LOAD_BALL_RACEWAY = 500.0            # [N]  PLACEHOLDER: set the value of the S4 reference case

# --- sweeps --------------------------------------------------------------------
# The reference grid is the default of the bridge: h0 = largest 2**k <= s_x / 10, +-4 s_x along
# rolling (``Grid.around``).
H_LEVELS = (2.0, 1.0, 0.5, 0.25)     # mesh sweep: grid spacing in units of h0 [-]
SPAN_X_0 = 4.0                       # half-length of the reference grid along rolling in units of s_x [-]
SPAN_X_LEVELS = (2.0, 4.0, 8.0)      # domain sweep: half-length along rolling in units of s_x [-]

# --- where the scripts write: results/ next to the study, git-ignored (D-0020) ------
RESULTS = Path(__file__).resolve().parents[1] / "results"


# =============================================================================
# Cases (one per script)
# =============================================================================

CASES = {
    "ball_flat": ContactCase("ball_flat", r1=BALL, r2=FLAT, E=E_STEEL, nu=NU_STEEL,
                             load=LOAD_BALL_FLAT),
    "ball_raceway": ContactCase("ball_raceway", r1=BALL, r2=INNER_RACEWAY, E=E_STEEL,
                               nu=NU_STEEL, load=LOAD_BALL_RACEWAY),
    "roller_raceway": ContactCase("roller_raceway", r1=ROLLER, r2=INNER_RACEWAY_ROLLER,
                                  E=E_STEEL, nu=NU_STEEL, load=LOAD_ROLLER_PER_LENGTH, line=True),
}
