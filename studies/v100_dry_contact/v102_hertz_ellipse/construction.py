"""Construction -- ball on the inner raceway of a deep groove ball bearing, the contact of
``validation/contact/hertz_ellipse``.

Two independent descriptions of the same contact, both returned NOT solved:

* :data:`CASE`: the contact written by hand as a :class:`ContactCase`, in SI (m, Pa, N), so that
  its ``case_hash`` does not depend on AxisForge;
* :func:`ball_bearing`: the AxisForge bearing, in AxisForge units (mm, N, MPa). The conversion to
  SI is made only inside ``slippy_bridge``; the script checks that the bridge extracts
  :data:`CASE` from it.

It also holds :data:`RATIO_CASES`, the printed case of Hamrock & Brewe (1983) Tables 1-3 for
``02_ratio``, converted here from cm and N/cm^2 to SI (:data:`CM`, :data:`N_PER_CM2`, the only
unit conversion of the case).

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

__all__ = ["CASE", "H_LEVELS", "SPAN_X_0", "SPAN_X_LEVELS", "RESULTS", "CM", "N_PER_CM2", "ALPHAS",
           "RATIO_CASES", "ball_bearing"]


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
# Hamrock & Brewe (1983) Tables 1-3 case, converted to SI here and only here
# =============================================================================

# --- the printed units -> SI ----------------------------------------------------------
CM = 1e-2                            # [m] per [cm]
N_PER_CM2 = 1e4                      # [Pa] per [N/cm^2]

# --- the printed case (Tables 2-3 caption, PDF p. 5; R_x also Table 1, PDF p. 4) ---------
E_PRIME_HB = 2.1972e7 * N_PER_CM2    # E' = 2/((1-nu_a^2)/E_a + (1-nu_b^2)/E_b) = 2 E* [Pa]
NU_HB = 0.3                          # not printed; any nu gives the same E*, chosen as steel [-]
# identical bodies: E* = E/(2(1 - nu^2)), so E' = 2E* gives E = E'(1 - nu^2)
E_HB = E_PRIME_HB * (1.0 - NU_HB**2) # Young's modulus of both bodies [Pa]
R_X_HB = 1.0 * CM                    # relative radius along rolling [m]
LOAD_HB = 4.448                      # normal load [N]
ALPHAS = (5.0, 6.0, 7.0, 8.0, 9.0, 10.0)   # alpha = R_y/R_x, the catalogue range of the tables [-]

# Body 1 carries both relative radii and body 2 is flat, so (R_x, R_y) = (R_X_HB, alpha R_X_HB).
RATIO_CASES = {alpha: ContactCase(f"hb_alpha_{alpha:g}", r1=(R_X_HB, alpha * R_X_HB),
                                  r2=(math.inf, math.inf), E=E_HB, nu=NU_HB, load=LOAD_HB)
               for alpha in ALPHAS}


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
