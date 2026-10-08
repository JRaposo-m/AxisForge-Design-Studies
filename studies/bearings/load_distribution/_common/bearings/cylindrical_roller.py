"""Cylindrical roller bearing, NU/N type (non-locating)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .base import BearingSpec


@dataclass(frozen=True)
class CylindricalRollerSpec(BearingSpec):
    """Cylindrical roller bearing (NU/N type), used as the non-locating bearing.

    It has no axial capacity (Fa = 0 by definition). The ISO/TS 16281 solver discretises each
    roller in ``n_s`` laminae.

    Attributes
    ----------
    s: float
        Total radial internal clearance Gr [mm], s >= 0.
    Dwe: float
        Roller diameter [mm].
    Lwe: float
        Effective roller length [mm].
    Dpw: float
        Pitch diameter [mm].
    Z: int
        Number of rollers.
    n_s: int
        Number of laminae of the slice model (the family requires >= 30).
    """

    kind: ClassVar[str] = "roller"
    arrangement: ClassVar[str] = "non-locating"

    designation: str = "cylindrical_roller_didactic"
    s: float = 0.010
    Dwe: float = 5.0
    Lwe: float = 5.0
    Dpw: float = 31.0
    Z: int = 14
    n_s: int = 50

    def family(self):
        import axisforge.core as af_c
        return af_c.CylindricalRollerFamily()

    def geometry(self) -> dict:
        return dict(Dwe=self.Dwe, Lwe=self.Lwe, Dpw=self.Dpw, Z=self.Z, n_s=self.n_s, s=self.s)
