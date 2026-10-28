"""Generic numerical methods for the studies of this repository: Newton-Raphson for ``F(x) = 0``
(``newton``). Imports only NumPy, SciPy and the standard library. Eager re-exports; see README.md.
"""
from __future__ import annotations

from .newton_raphson import IterationRecord, NewtonError, NewtonOptions, NewtonResult, newton

__all__ = ["IterationRecord", "NewtonError", "NewtonOptions", "NewtonResult", "newton"]
