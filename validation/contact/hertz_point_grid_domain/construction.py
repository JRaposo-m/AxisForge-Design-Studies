"""Construction -- steel ball on a steel flat, the contact of ``validation/contact/hertz_point_grid_domain``.

The contact (:data:`CASE`) is written here by hand as a :class:`ContactCase`, so that its
``case_hash`` does not depend on AxisForge. It is returned NOT solved: the scripts choose the grid,
the surface build and the solver. Derived quantities (relative radii, semi-axes, Hertz pressure)
are not computed here: they come from ``slippy_bridge.contact`` and ``references.contact.hertz``.

Model
-----
Circular point contact: a 12.7 mm steel ball pressed on a steel half-space by a normal load. Both
bodies are elastic, smooth and frictionless (Johnson 1985, Eqs (4.22)-(4.24)). The load is
chosen, not taken from a reference: the closed forms (a, p0, delta) are exact for any load and
every graded quantity is a ratio to its closed form.

Units
-----
SI (m, Pa, N), the units of ``slippy_bridge``. Nothing here is in AxisForge units.
"""
from __future__ import annotations

import math
from pathlib import Path

from slippy_bridge.contact import ContactCase

__all__ = ["CASE", "H_LEVELS", "SPAN_X_0", "SPAN_X_LEVELS", "RESULTS"]


# =============================================================================
# Inputs  (SI: m, Pa, N)
# =============================================================================

# --- material (one steel for both bodies) ----------------------------------------
E_STEEL = 210e9                      # Young's modulus [Pa]
NU_STEEL = 0.3                       # Poisson's ratio [-]

# --- bodies: principal radii (r_x, r_y) [m]; convex > 0, concave < 0, math.inf = flat ---
DW = 12.7e-3                         # ball diameter [m]
BALL = (DW / 2.0, DW / 2.0)
FLAT = (math.inf, math.inf)

# --- load ------------------------------------------------------------------------
# Chosen, scale-free reference: Johnson Eqs (4.22)-(4.24) hold for any P.
LOAD = 500.0                         # normal load [N]

# --- sweeps --------------------------------------------------------------------
# The reference grid is the default of the bridge: h0 = largest 2**k <= s_x / 10, +-4 s_x along
# rolling (``Grid.around``).
H_LEVELS = (2.0, 1.0, 0.5, 0.25)     # mesh sweep: grid spacing in units of h0 [-]
SPAN_X_0 = 4.0                       # half-length of the reference grid along rolling in units of s_x [-]
SPAN_X_LEVELS = (2.0, 4.0, 8.0)      # domain sweep: half-length along rolling in units of s_x [-]

# --- where the scripts write: results/ next to this case, git-ignored (D-0020) ------
RESULTS = Path(__file__).resolve().parent / "results"


# =============================================================================
# Case
# =============================================================================

CASE = ContactCase("ball_flat", r1=BALL, r2=FLAT, E=E_STEEL, nu=NU_STEEL, load=LOAD)
