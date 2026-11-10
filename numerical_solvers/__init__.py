"""Generic numerical methods for the studies of this repository: Newton-Raphson for ``F(x) = 0``
(``newton``) and Richardson extrapolation / GCI over refinement levels (``richardson``). Imports only
NumPy, SciPy and the standard library. Eager re-exports; see README.md.
"""
from __future__ import annotations

from .newton_raphson import IterationRecord, NewtonError, NewtonOptions, NewtonResult, newton
from .richardson_extrapolation import (LevelRecord, MetricConvergence, RichardsonError, RichardsonOptions,
                         RichardsonResult, richardson, richardson_triple)

__all__ = ["IterationRecord", "NewtonError", "NewtonOptions", "NewtonResult", "newton",
           "LevelRecord", "MetricConvergence", "RichardsonError", "RichardsonOptions",
           "RichardsonResult", "richardson", "richardson_triple"]
