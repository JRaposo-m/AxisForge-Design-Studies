"""Deep groove ball bearing (locating)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .base import BearingSpec


@dataclass(frozen=True)
class DeepGrooveSpec(BearingSpec):
    """Single-row deep groove ball bearing, used as the locating bearing.

    Attributes
    ----------
    s: float
        Total radial internal clearance Gr [mm] (diametral play, ISO 5753-1), s >= 0.
        The family derives the free contact angle from it.
    Dw: float
        Ball diameter [mm].
    Dpw: float
        Pitch diameter [mm].
    Z: int
        Number of balls.
    """

    kind: ClassVar[str] = "ball"
    arrangement: ClassVar[str] = "locating"

    designation: str = "deep_groove_didactic"
    s: float = 0.010
    Dw: float = 7.0
    Dpw: float = 31.0
    Z: int = 9

    def family(self):
        import axisforge.core as af_c
        return af_c.DeepGrooveBallFamily()

    def geometry(self) -> dict:
        return dict(Dw=self.Dw, Dpw=self.Dpw, Z=self.Z, s=self.s)
