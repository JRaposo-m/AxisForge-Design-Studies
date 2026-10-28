"""s004 bearing compliance on a two-bearing shaft: the rigid-body field of yielding bearings and its tilt fixed point (README).

s003 may import ``ring_displacement`` and ``rigid_line`` through this package (D-0036 amendment item 11);
``coupled_tilt`` holds the reactions fixed and is valid for a determinate shaft on two pins only.
"""
from .construction import bearings, shaft
from .model import DISPLACEMENT_SIGN, MAX_PASSES, TOL_PSI_RAD, coupled_tilt, rigid_line, ring_displacement, support_force

__all__ = ["coupled_tilt", "ring_displacement", "rigid_line", "support_force", "DISPLACEMENT_SIGN", "TOL_PSI_RAD",
           "MAX_PASSES", "shaft", "bearings"]
