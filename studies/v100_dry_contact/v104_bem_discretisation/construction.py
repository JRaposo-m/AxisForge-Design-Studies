"""The systems and sweeps of v104, returned NOT solved: smooth Hertz contacts on many grids.

Two smooth point contacts, steel on steel, 500 N: the circular contact of v101 (ball 12.7 mm on a
flat) and an elliptical one (a body of radii 10 mm along x and 50 mm along y on a flat, R_y/R_x = 5,
the minor axis b along x so the ellipse sits on the grid lines). SI units.
"""
from __future__ import annotations

import math

from slippy_bridge.contact import SweptCircleBody

from studies.v100_dry_contact.v101_hertz_point import PointContact, ball_on_flat
from studies.v100_dry_contact.v101_hertz_point.construction import E_STEEL, LOAD, NU_STEEL

__all__ = ["systems", "ellipsoid_on_flat", "B_OVER_H", "SPANS", "B_OVER_H_SPAN_SWEEP", "KERNELS"]

B_OVER_H = (4.0, 8.0, 16.0, 32.0, 64.0)     # h sweep: minor semi-axis over spacing asked [-] (h rounded down)
SPANS = (1.25, 1.5, 2.0, 4.0, 8.0)          # domain sweep: half-length on both axes [semi-axes]
B_OVER_H_SPAN_SWEEP = 16.0                  # b/h of the domain sweep [-]
# kernel -> (use_frequency_domain, domain half-length of the h sweep [semi-axes]); 4 = the bridge's SPAN_X_0
KERNELS = {"spatial": (False, 2.0), "frequency": (True, 4.0)}


def ellipsoid_on_flat(load: float = LOAD) -> PointContact:
    """Body of radii (10 mm, 50 mm) on a steel flat, steel, ``load`` [N], unsolved (R_y/R_x = 5)."""
    steel = {"E": E_STEEL, "nu": NU_STEEL}
    return PointContact(body1=SweptCircleBody(r=(10e-3, 50e-3), **steel),
                        body2=SweptCircleBody(r=math.inf, **steel), load=load, label="ellipsoid_on_flat")


def systems() -> list[PointContact]:
    """The two contacts of the sweeps, circle first."""
    return [ball_on_flat(), ellipsoid_on_flat()]
