"""Optional inputs of the contact sub-models, data only: :class:`Kinematics`.

Dry normal contact (Hertz, the static BEM) does not use them (D-0034 item 5). They are for the
later sub-models (friction direction, wear, flash temperature) and for lubrication.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

from .contact import DC

__all__ = ["Kinematics"]

# OPEN SB-22: the bearing entrainment formula of the old adapter is not carried over (the prompt: Kinematics only; D-0034 item 5 says it moves with the speeds)


@dataclass(**DC)
class Kinematics:
    """Surface speeds of one contact, along the rolling direction (bridge x) [m/s].

    Meanings and signs of AxisForge ``CylindricalFamily.surface_speeds`` (``u1``, ``u2`` the speeds
    of the two surfaces relative to the contact point); the names drop its ``_m_per_s`` suffix. Gear
    speeds will come from AxisForge (``surface_speeds``, ``contact_inputs``).

    Bridge +x is meant along the entrainment, so SlipPY's Reynolds solver (inlet at array index 0)
    needs ``ue > 0``; not checked here because nothing uses this class yet. No SlipPY Reynolds step
    reads a sliding speed; ``RollingSliding1D`` takes the creep ``vg / ue`` (its sign to be checked
    when wired). In SlipPY's sub-models ``direction='x'`` is array axis 1, so bridge x (axis 0) is
    SlipPY ``'y'`` there (R-156).

    Parameters
    ----------
    ue : float
        MEAN entrainment speed ``(u1 + u2) / 2`` [m/s], never the sum ``u1 + u2``: the rolling
        speed of SlipPY's Reynolds solver is this mean.
    vg : float
        Sliding speed ``u1 - u2`` [m/s], signed, body 1 referenced.
    u1, u2 : float, optional
        Surface speeds of body 1 and body 2 [m/s] when known; None otherwise.

    Raises
    ------
    ValueError
        If a given speed is not a finite real number (a bool is refused); if ``u1`` and ``u2`` are
        given and ``ue != (u1 + u2)/2`` or ``vg != u1 - u2`` (rel 1e-9, abs 1e-12 m/s): this catches
        the sum given as ``ue`` (R-075).
    """

    ue: float
    vg: float
    u1: float | None = None
    u2: float | None = None

    def __post_init__(self) -> None:
        # OPEN SB-23: ue > 0 (bridge +x along the entrainment) not enforced until a sub-model uses Kinematics
        for name in ("ue", "vg", "u1", "u2"):
            value = getattr(self, name)
            if value is None and name in ("u1", "u2"):
                continue
            if isinstance(value, bool) or not (isinstance(value, numbers.Real) and math.isfinite(value)):
                raise ValueError(f"Kinematics: {name} must be a finite real number [m/s], got {value!r}")
        if self.u1 is not None and self.u2 is not None:
            for name, value, expected in (("ue", self.ue, (self.u1 + self.u2) / 2), ("vg", self.vg, self.u1 - self.u2)):
                if not math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-12):
                    raise ValueError(f"Kinematics: {name} = {value} m/s is not {expected} m/s from u1 = {self.u1}, "
                                     f"u2 = {self.u2} (ue is the MEAN (u1 + u2)/2, vg = u1 - u2)")
