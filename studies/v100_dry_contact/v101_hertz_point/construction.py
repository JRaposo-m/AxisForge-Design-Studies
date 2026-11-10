"""The system of v101: a steel ball on a steel flat, returned NOT solved.

Circular point contact of two smooth, frictionless, linear elastic bodies (Johnson 1985, section
4.2(a), Eqs (4.19)-(4.24), PDF p. 102-103). The load is chosen: the closed forms hold for any
load and every judged quantity is a ratio to its reference. SI units (m, Pa, N), the units of
``slippy_bridge``; nothing here is in AxisForge units.
"""
from __future__ import annotations

import math
from pathlib import Path

from slippy_bridge.contact import Contact, SweptCircleBody

__all__ = ["PointContact", "ball_on_flat", "E_STEEL", "NU_STEEL", "DW", "LOAD", "A_OVER_H", "PUBLISHED"]

E_STEEL = 210e9          # Young's modulus [Pa]
NU_STEEL = 0.3           # Poisson's ratio [-]
DW = 12.7e-3             # ball diameter [m]
LOAD = 500.0             # normal load [N] (chosen)

# grid levels: the asked contact radius over spacing a/h [-]; Grid.around rounds h down to 2**k m,
# so the grid has a/h in [asked, 2 x asked)
A_OVER_H = (8.0, 16.0, 32.0)
PUBLISHED = Path(__file__).with_name("literature") / "published"   # the printed tables


# OPEN R2-03: slippy_bridge has no public point contact of two given bodies (its tests define ``_Point``);
# this one-line subclass is the v cases' until the bridge exports one
class PointContact(Contact):
    """A point contact of two given bodies (the bridge's ``Contact`` with ``line = False``)."""

    line = False


def ball_on_flat(load: float = LOAD, *, dw: float = DW, E: float = E_STEEL, nu: float = NU_STEEL,
                 label: str = "ball_on_flat") -> PointContact:
    """Ball of diameter ``dw`` [m] on a flat of the same material under ``load`` [N], unsolved."""
    return PointContact(body1=SweptCircleBody(r=dw / 2.0, E=E, nu=nu),
                        body2=SweptCircleBody(r=math.inf, E=E, nu=nu), load=load, label=label)
