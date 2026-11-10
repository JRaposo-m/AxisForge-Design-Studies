"""The system of s003: a uniform shaft on three ball bearings, the middle seat offset by e, returned NOT solved.

The shaft is v005's (``three_support_shaft``: S355, L = 270 mm, d = 30 mm, zero-width pins at 10 (locating),
110 and 260 mm, spans l1 = 100, l2 = 150 mm; ``# OPEN R5-162``: s003 imports v005's construction and its
condensed beam, the model the later study needs, D-0036 amendment item 11). One gear load, spur (no axial force,
no couple, no J-16 tag): ``P_N`` along -Y at the middle of span 1 (x = 60 mm, proposals.md "gear at 1/2 span"),
so the whole problem lies in plane XY (proposals.md "first runs in one plane").

Construction rule (proposals.md s003): no load on a bearing node, so ``K d - f_ext`` at a bearing DOF is the
bearing force alone.

Bearings: three s001 6210-like deep groove ball bearings (ISO/TS 16281; ``# OPEN R5-163``: a 50 mm bore on a
30 mm shaft, as s004 R5-51: the FEM sees zero-width pins, the bearing model only its ring displacement, so the
assembly is fictitious and the numbers are not affected). Diametral clearance s (R-105), the same in all three.

Seat offset e: the housing bore of the middle bearing displaced by e along +Y (a radial misalignment of the
seats, the support "settlement" of v005 / r01 when the bearing is rigid). The outer seats stay at 0.

Load level: P = 2000 N gives reactions of about 1 kN, delta_r about 12 um per bearing (s = 0), of the order of
the shaft's own deflection (about 3e4 N/mm at x = 110), so neither the shaft nor the bearings dominate; and
e = -70 ... +50 um moves the middle reaction of the rigid shaft by about -2 ... +1.5 kN, through zero (``# OPEN R5-164``:
the proposals ask "one Fr"; this value is the case's own, AxisForge holds no C0, R5-40).

Units: mm, N, MPa (AxisForge).
"""
from __future__ import annotations

import numpy as np
from axisforge.core.loads import RadialLoad

from studies.s000_axisforge.s001_bearing_clearance_and_contact_angle import deep_groove
from studies.v000_axisforge.v005_shaft_three_supports import three_support_shaft
from studies.v000_axisforge.v005_shaft_three_supports.construction import SUPPORTS_MM

__all__ = ["shaft", "bearings", "SUPPORTS_MM", "X_MIDDLE_MM", "X_GEAR_MM", "P_N", "S_MM", "E_UM", "N_ELEMENTS"]

X_MIDDLE_MM = SUPPORTS_MM[1]             # the bearing whose seat is offset
X_GEAR_MM = 60.0                         # middle of span 1 (10 ... 110 mm), a node of every uniform mesh below
P_N = 2000.0                             # [N] spur gear force, along -Y (theta 180 deg from +Y toward +Z, D-0010)
S_MM = (0.0, 0.010, 0.020, 0.030)        # [mm] diametral clearance, proposals.md 0 ... 30 um
# [um] middle seat offset along +Y, step 10: -70 so that the zero crossings of the middle reaction of the rigid
# (about -45 um) and of the coupled shaft (about -56 um) both fall inside the sweep (physics review)
E_UM = tuple(float(e) for e in np.linspace(-70.0, 50.0, 13))
N_ELEMENTS = 54                          # uniform h = 5 mm: supports and gear on nodes (v005 Hertz mesh)


def shaft(*, P_N: float = P_N, label: str = "s003"):
    """The shaft on three zero-width pins with the gear force, NOT solved (v005 ``three_support_shaft``).

    Returns
    -------
    ShaftSystem
        Pins at ``SUPPORTS_MM``; one ``RadialLoad`` of ``P_N`` [N] at ``X_GEAR_MM`` along -Y.
    """
    return three_support_shaft(loads=[RadialLoad(X_GEAR_MM, P_N, theta_deg=180.0, label="gear")], label=label)


def bearings(s_mm: float) -> dict:
    """The three ISO/TS 16281 bearings by position, unsolved: s001 6210-like, diametral clearance ``s_mm`` [mm].

    Returns
    -------
    dict
        ``{x [mm]: Bearing}`` for x in ``SUPPORTS_MM``.
    """
    return {x: deep_groove(s_mm, label=f"6210_x{x:g}") for x in SUPPORTS_MM}
