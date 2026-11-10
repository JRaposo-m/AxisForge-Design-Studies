"""The system of v103: two steel rollers in line contact (radii 10 and 20 mm), returned NOT solved.

Plane-strain contact of two long cylinders with parallel axes along bridge y under a load per unit
length (Johnson 1985, section 4.2(c), Eqs (4.43)-(4.45), PDF p. 110-111). The BEM models one period of a
strip periodic along y (``slippy_bridge`` line contact). SI units (m, Pa, N/m).
"""
from __future__ import annotations

import math
from pathlib import Path

from slippy_bridge.contact import Contact, SweptCircleBody

from studies.v100_dry_contact.v101_hertz_point.construction import E_STEEL, NU_STEEL

__all__ = ["LineContact", "rollers", "R1", "R2", "LOAD_PER_LENGTH", "PUBLISHED"]

R1, R2 = 10e-3, 20e-3          # roller radii [m] (the line case of r02's tests)
LOAD_PER_LENGTH = 1e5          # [N/m] (chosen)
PUBLISHED = Path(__file__).with_name("literature") / "published"


# OPEN R2-03: slippy_bridge has no public line contact of two given bodies (its tests define ``_Line``)
class LineContact(Contact):
    """A line contact of two given bodies, the line along bridge y (the bridge ``Contact``, ``line = True``)."""

    line = True


def rollers(load: float = LOAD_PER_LENGTH, *, r1: float = R1, r2: float = R2) -> LineContact:
    """Two steel rollers of radii ``r1``, ``r2`` [m], axes along y, ``load`` [N/m], unsolved."""
    return LineContact(body1=SweptCircleBody(r=(r1, math.inf), E=E_STEEL, nu=NU_STEEL),
                       body2=SweptCircleBody(r=(r2, math.inf), E=E_STEEL, nu=NU_STEEL), load=load, label="rollers")
