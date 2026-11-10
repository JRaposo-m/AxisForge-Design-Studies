"""s003 hyperstatic shaft on three ISO/TS 16281 bearings: reactions, slopes and loaded zones against rigid supports
over a seat offset and the clearance, solved by Newton on the ring displacements (README).

A later study (s005, the GRC steps) imports the model through this package (D-0036 amendment item 11).
"""
from .construction import bearings, shaft
from .model import ball_loads, solve_shaft

__all__ = ["ball_loads", "solve_shaft", "shaft", "bearings"]
