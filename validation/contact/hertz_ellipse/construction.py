"""Construction -- ball on the inner raceway of a deep groove ball bearing, the contact of
``validation/contact/hertz_ellipse``.

Two independent descriptions of the same contact, both returned NOT solved:

* :data:`CASE`: the contact written by hand as a :class:`ContactCase`, in SI (m, Pa, N), so that
  its ``case_hash`` does not depend on AxisForge;
* :func:`ball_bearing`: the AxisForge bearing, in AxisForge units (mm, N, MPa). The conversion to
  SI is made only inside ``slippy_bridge``; the script checks that the bridge extracts
  :data:`CASE` from it.

The values are repeated on purpose (no construction shared between cases, ``validation/README.md``).

Model
-----
Elliptical point contact between a steel ball and the inner raceway of a deep groove ball bearing
at zero free contact angle (radial clearance 0). The groove conforms to the ball, so the ellipse is
long across rolling (``R_y / R_x`` about 32). Elastic, smooth, frictionless (Hertz; Johnson 1985,
ch. 4; Hamrock & Brewe 1983). The boundary dimensions (d, D, B) are those of a 6210 and the number
of balls is a typical value; AxisForge needs them to assemble a bearing, the contact of one ball
with the inner raceway does not depend on them.
"""
from __future__ import annotations

import math
from pathlib import Path

import axisforge.core as af_c

from slippy_bridge.contact import ContactCase

__all__ = ["CASE", "H_LEVELS", "SPAN_X_0", "SPAN_X_LEVELS", "RESULTS", "ball_bearing"]


# =============================================================================
# Contact case  (SI: m, Pa, N)
# =============================================================================

# --- material (one steel for both bodies) ----------------------------------------
E_STEEL = 210e9                      # Young's modulus [Pa]
NU_STEEL = 0.3                       # Poisson's ratio [-]

# --- bearing geometry (deep groove ball bearing, typical 6210-type values) ----------
# Same definitions as the AxisForge ball-bearing geometry (Harris / Hamrock-Dowson notation).
DW = 12.7e-3                         # ball diameter [m]
DPW = 70.0e-3                        # pitch diameter [m]
F_I = 0.52                           # inner groove radius / ball diameter ri/Dw [-]
ALPHA_0 = 0.0                        # free contact angle [rad]

# --- bodies: principal radii (r_x, r_y) [m]; convex > 0, concave < 0, math.inf = flat ---
BALL = (DW / 2.0, DW / 2.0)
INNER_RACEWAY = ((DPW - DW * math.cos(ALPHA_0)) / (2.0 * math.cos(ALPHA_0)),   # rolling radius
                 -F_I * DW)                                                    # groove radius: concave

# --- load ------------------------------------------------------------------------
# HamrockBrewe1983, Tables 2-3 caption (PDF p. 5). There it acts with R_x = 1 cm and
# E' = 2.1972e7 N/cm^2 (E ~ 199.9 GPa at nu = 0.3, not 210 GPa); here on the bearing geometry
# above, whose ratio R_y/R_x has no table row, so the table values are not compared directly:
# the judge is ``references.contact.hertz.hertz_point`` (wired in the case scripts, step 5).
LOAD = 4.448                         # normal ball load [N]

# --- sweeps --------------------------------------------------------------------
# The reference grid is the default of the bridge: h0 = largest 2**k <= s_x / 10, +-4 s_x along
# rolling (``Grid.around``).
H_LEVELS = (2.0, 1.0, 0.5, 0.25)     # mesh sweep: grid spacing in units of h0 [-]
SPAN_X_0 = 4.0                       # half-length of the reference grid along rolling in units of s_x [-]
SPAN_X_LEVELS = (2.0, 4.0, 8.0)      # domain sweep: half-length along rolling in units of s_x [-]

# --- where the scripts write: results/ next to this case, git-ignored (D-0020) ------
RESULTS = Path(__file__).resolve().parent / "results"

CASE = ContactCase("ball_raceway", r1=BALL, r2=INNER_RACEWAY, E=E_STEEL, nu=NU_STEEL, load=LOAD)


# =============================================================================
# AxisForge bearing  (mm, N, MPa)
# =============================================================================

E_STEEL_MPA = 210000.0  # Young's modulus [MPa]
DPW_MM = 70.0           # pitch diameter [mm]
DW_MM = 12.7            # ball diameter [mm]
Z = 9                   # rolling elements [-]


def ball_bearing():
    """Deep groove ball bearing, ``Dw`` 12.7 mm on ``Dpw`` 70 mm, assembled and NOT solved.

    Radial clearance ``s = 0``, so the free contact angle is zero and the rolling radius of the
    inner raceway is ``(Dpw - Dw) / 2``. The groove radius is the family default, ``0.52 Dw``.

    Returns
    -------
    axisforge.core.Bearing
        The bearing in AxisForge units (mm, N, MPa).
    """
    catalog = af_c.BearingCatalog(d=50.0, D=90.0, b=20.0, designation="6210", position=0.0,
                                  arrangement="locating", label="6210")
    return af_c.Bearing.assemble(family=af_c.DeepGrooveBallFamily(), catalog=catalog,
                                 geometry={"Dw": DW_MM, "Dpw": DPW_MM, "Z": Z, "s": 0.0,
                                           "contact": af_c.ContactAnalysis.ISO16281,
                                           "e1": E_STEEL_MPA, "e2": E_STEEL_MPA, "nu1": NU_STEEL, "nu2": NU_STEEL})
