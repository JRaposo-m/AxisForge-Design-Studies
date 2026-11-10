"""Richardson extrapolation and Grid Convergence Index over refinement levels (``richardson``,
``richardson_triple``) and their result containers. Eager re-exports.

Named after the method, not ``richardson``: a subpackage named like the function it re-exports is
shadowed by it (pytest ``--import-mode=importlib`` resets the attribute). ``# OPEN R3-02``
"""
from __future__ import annotations

from .richardson import RichardsonError, RichardsonOptions, richardson, richardson_triple
from .richardson_results import LevelRecord, MetricConvergence, RichardsonResult

__all__ = ["LevelRecord", "MetricConvergence", "RichardsonError", "RichardsonOptions",
           "RichardsonResult", "richardson", "richardson_triple"]
