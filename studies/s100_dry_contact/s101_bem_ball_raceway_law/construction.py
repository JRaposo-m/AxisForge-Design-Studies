"""The systems of s101: the most loaded ball-raceway contacts of two s001 bearings, the BEM contacts NOT solved.

Four smooth point contacts, built by ``slippy_bridge`` (``from_bearing(bearing, result, elements="max")``) from
AxisForge bearings solved by s001's model, with the true ball and groove profiles (not the Hertz paraboloid);
steel, E 210 GPa, nu 0.3, grooves f = 0.52:

* 6210-like deep groove (Dw 12.7 mm), zero clearance, Fr 5000 N: the ball on the load line, alpha 0,
  Q_max 2436 N (s001's case), inner and outer;
* 7210-like angular contact, alpha_0 40 deg, Fr = Fa = 5000 N: its most loaded ball, alpha_j 38.0 deg,
  Q_max 2071 N (s001's case), inner and outer.

The AxisForge solve fixes each contact's reference load ``Q_ref`` (its ``load``) and angle; the approach sweep is
relative to the Hertz approach ``delta_H(Q_ref)``. The bearings and the solve come from s001's public API (the
bearing model every bearing case imports); building the contacts from a solved bearing is what lets v105.check
judge them.

Units: the bridge's SI (m, N, Pa); AxisForge's mm stay inside the bearing objects.
"""
from __future__ import annotations

import numpy as np
from slippy_bridge.contact.core.machine_elements.bearings import AngularContactBallContact, DeepGrooveBallContact

from studies.s000_axisforge.s001_bearing_clearance_and_contact_angle import (angular_contact, deep_groove, load_node,
                                                                             solve_bearing)

__all__ = ["contacts", "DELTA_FACTORS", "B_OVER_H", "SPAN", "B_OVER_H_LEVELS", "FR_N"]

FR_N = 5000.0                                        # [N] radial load of both bearings (s001's level)
# approach sweep: delta / delta_H(Q_ref), 10 points, 0.05..1.5 (proposals.md s101); loads 0.011..1.8 Q_ref
DELTA_FACTORS = tuple(float(f) for f in np.geomspace(0.05, 1.5, 10))
B_OVER_H = 16.0                                      # asked b/h (v104 rule, spatial kernel); the grid has [16, 32)
SPAN = 2.0                                           # domain half-length in Hertz semi-axes on both axes (v104)
B_OVER_H_LEVELS = (8.0, 16.0, 32.0)                  # grid-convergence levels (asked), at delta_H(Q_ref)


def contacts() -> dict:
    """``{name: (bearing, load_distribution, contact)}`` of the four ball-raceway contacts, the BEM unsolved.

    Names ``"6210 inner"``, ``"6210 outer"``, ``"7210 a40 inner"``, ``"7210 a40 outer"``; each contact is the most
    loaded ball's (``elements="max"``) at its AxisForge ``Q_j`` and ``alpha_j``.
    """
    out = {}
    for tag, bearing, Fa, cls in (("6210", deep_groove(0.0), 0.0, DeepGrooveBallContact),
                                  ("7210 a40", angular_contact(40.0), FR_N, AngularContactBallContact)):
        analysis, _, error = solve_bearing(bearing, load_node(bearing, FR_N, Fa), postprocess=False)
        if analysis is None:
            raise RuntimeError(f"{tag}: AxisForge solve failed: {error}")
        ld = analysis.load_distribution
        for c in cls.from_bearing(bearing, ld, elements="max"):
            out[f"{tag} {c.race}"] = (bearing, ld, c)
    return out
