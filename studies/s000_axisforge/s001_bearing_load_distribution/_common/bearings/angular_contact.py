"""Single-row angular contact ball bearing (locating)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .base import BearingSpec


@dataclass(frozen=True)
class AngularContactSpec(BearingSpec):
    """Single-row angular contact ball bearing, used as the locating bearing.

    The family derives the radial internal clearance from the input contact angle
    (s = 2 A (1 - cos alpha_0), A = ri + re - Dw): alpha_0 and s are the SAME variable for this
    type, so it has no separate clearance parameter.

    An angular contact bearing needs an axial load: with spur gears (Fa = 0) it floats
    axially. Use it with a helical gear stage.

    Attributes
    ----------
    alpha_0_deg: float
        Input (free) contact angle [deg], 0 < alpha_0 <= 45.
    Dw: float
        Ball diameter [mm].
    Dpw: float
        Pitch diameter [mm].
    Z: int
        Number of balls.
    """

    kind: ClassVar[str] = "angular"
    arrangement: ClassVar[str] = "locating"

    designation: str = "angular_contact_didactic"
    alpha_0_deg: float = 25.0
    Dw: float = 7.0
    Dpw: float = 31.0
    Z: int = 9

    def family(self):
        # AngularContactFamily is not exported by axisforge.core yet (only DeepGroove / Roller are)
        from axisforge.core.machine_elements.bearings.families.family import AngularContactFamily
        return AngularContactFamily()

    def geometry(self) -> dict:
        return dict(Dw=self.Dw, Dpw=self.Dpw, Z=self.Z, alpha_0_deg=self.alpha_0_deg)
