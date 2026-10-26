"""Construction -- roller on the inner raceway of a cylindrical roller bearing, the contact of
``validation/contact/hertz_line_subsurface``.

Two independent descriptions of the same contact, both returned NOT solved:

* :data:`CASE`: the contact written by hand as a :class:`ContactCase`, in SI (m, Pa, N/m), so
  that its ``case_hash`` does not depend on AxisForge;
* :func:`roller_bearing`: the AxisForge bearing, in AxisForge units (mm, N, MPa). The conversion
  to SI is made only inside ``slippy_bridge``; the script checks that the bridge extracts
  :data:`CASE` from it.

The values are repeated on purpose (no construction shared between cases, ``validation/README.md``).

Model
-----
Line contact (plane strain) between a steel cylinder and the inner raceway, both straight along
the roller axis ``y``: the load is per unit length. Elastic, smooth, frictionless (Hertz; Johnson
1985, Eqs (4.43)-(4.45)). The boundary dimensions (d, D, B) are those of an NU 210 and the number
of rollers is a typical value; AxisForge needs them to assemble a bearing, the contact of one
roller with the inner raceway does not depend on them.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import axisforge.core as af_c

from slippy_bridge.contact import ContactCase

__all__ = ["CASE", "H_LEVELS", "SPAN_X_0", "SPAN_X_LEVELS", "Z_OVER_A", "RESULTS", "roller_bearing"]


# =============================================================================
# Contact case  (SI: m, Pa, N/m)
# =============================================================================

# --- material (one steel for both bodies) ----------------------------------------
E_STEEL = 210e9                      # Young's modulus [Pa]
NU_STEEL = 0.3                       # Poisson's ratio [-]

# --- geometry (cylindrical roller bearing, typical values) --------------------------
DPW = 70.0e-3                        # pitch diameter [m]
DWE = 10.0e-3                        # roller diameter [m]

# --- bodies: principal radii (r_x, r_y) [m]; convex > 0, concave < 0, math.inf = flat ---
ROLLER = (DWE / 2.0, math.inf)       # cylinder: no curvature along its axis
INNER_RACEWAY_ROLLER = ((DPW - DWE) / 2.0, math.inf)   # rolling radius of the inner raceway, straight in y

# --- load ------------------------------------------------------------------------
# Chosen, not from a reference: the reference is scale-free (Johnson Eqs (4.43)-(4.45) and
# Appendix 4 are normalised by a and p0). 5.0e4 N/m = 500 N over a 10 mm roller.
LOAD = 5.0e4                         # normal load per unit roller length [N/m]

# --- sweeps --------------------------------------------------------------------
# The reference grid is the default of the bridge: h0 = largest 2**k <= s_x / 10, +-4 s_x along
# rolling (``Grid.around``).
H_LEVELS = (2.0, 1.0, 0.5, 0.25)     # mesh sweep: grid spacing in units of h0 [-]
SPAN_X_0 = 4.0                       # half-length of the reference grid along rolling in units of s_x [-]
SPAN_X_LEVELS = (2.0, 4.0, 8.0)      # domain sweep: half-length along rolling in units of s_x [-]

# --- subsurface stress: depths below the surface, in units of a ----------------------------
# 0 to 2a covers the Appendix 4 rows (z/a 0 ... 2.0) and tau1 max at 0.785a; step 0.02a
Z_OVER_A = np.linspace(0.0, 2.0, 101)
# the surface row sits at 1e-4 of the first step, as SlipPY's own default depths do
# (slippy/core/elastic_material.py:328): its kernels are singular at z = 0
Z_OVER_A[0] = 1e-4 * Z_OVER_A[1]

# --- where the scripts write: results/ next to this case, git-ignored (D-0020) ------
RESULTS = Path(__file__).resolve().parent / "results"

CASE = ContactCase("roller_raceway", r1=ROLLER, r2=INNER_RACEWAY_ROLLER, E=E_STEEL, nu=NU_STEEL,
                   load=LOAD, line=True)


# =============================================================================
# AxisForge bearing  (mm, N, MPa)
# =============================================================================

E_STEEL_MPA = 210000.0  # Young's modulus [MPa]
DPW_MM = 70.0           # pitch diameter [mm]
DWE_MM = 10.0           # roller diameter [mm]
LWE_MM = 10.0           # effective roller length [mm]
Z = 9                   # rolling elements [-]
N_S = 50                # laminae along the roller [-] (ISO/TS 16281; not used by the contact case)


def roller_bearing():
    """Cylindrical roller bearing, ``Dwe`` 10 mm and ``Lwe`` 10 mm on ``Dpw`` 70 mm, ``s = 0``,
    assembled and NOT solved.

    Returns
    -------
    axisforge.core.Bearing
        The bearing in AxisForge units (mm, N, MPa).
    """
    catalog = af_c.BearingCatalog(d=50.0, D=90.0, b=20.0, designation="NU210", position=0.0,
                                  arrangement="non-locating", label="NU210")
    return af_c.Bearing.assemble(family=af_c.CylindricalRollerFamily(), catalog=catalog,
                                 geometry={"Dwe": DWE_MM, "Lwe": LWE_MM, "Dpw": DPW_MM, "Z": Z, "n_s": N_S,
                                           "s": 0.0, "contact": af_c.ContactAnalysis.ISO16281,
                                           "e1": E_STEEL_MPA, "e2": E_STEEL_MPA, "nu1": NU_STEEL, "nu2": NU_STEEL})
