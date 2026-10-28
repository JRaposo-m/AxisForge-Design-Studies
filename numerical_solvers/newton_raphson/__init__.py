"""Newton-Raphson solver of a nonlinear system ``F(x) = 0`` (``newton``) and its result containers,
ported from AxisForge ``axisforge.solvers.numerics`` at d34ec7f. Eager re-exports.
"""
from __future__ import annotations

from .newton import NewtonError, NewtonOptions, newton
from .newton_results import IterationRecord, NewtonResult

__all__ = ["IterationRecord", "NewtonError", "NewtonOptions", "NewtonResult", "newton"]
