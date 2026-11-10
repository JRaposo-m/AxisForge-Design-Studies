"""The system of s005: s004's gear-loaded shaft on two ISO/TS 16281 bearings, over the span, NOT solved.

The shaft and the bearings are s004's (``shaft``, ``bearings``: s006's pinion shaft, uniform d 40 mm, pins at 0
(A, locating) and L (B, floating), spur H501 pinion at 0.3 L; s001's 6210-like deep groove ball bearing at A, at B
the same or the NU210-like cylindrical roller bearing), imported through s004's package API (D-0036 amendment item
11). Only the span and an axial load at the locating bearing are added here.

Axial load. The spur mesh gives no axial force, so ``Fa_A = FA_OVER_FR x |R_A|`` (``|R_A|`` the rigid-pin reaction
at A), SIGNED in the solver's axial sense (+ = the direction of ``A sin(alpha_0)``, V_j > 0, ``contact_solver.py:633``;
which global X direction that is stays open for José, R5-190), is given to bearing A only (the locating bearing carries all of it; the floating bearing B carries none). It is
NOT put on the shaft FEM: an axial force on the axis bends nothing (linear beam), so the bending problem is the same
with or without it; no couple F_a r_w arises (it is not a gear force), so no J-16 tag. ``# OPEN R5-175``: proposals.md
asks "Fa/Fr = 0 and 0.5" without saying where Fa comes from; an applied axial load at A is the simplest reading.

Units: mm, N, N.mm, rad (AxisForge); torque N.m.
"""
from __future__ import annotations

import numpy as np

from studies.s000_axisforge.s004_bearing_compliance import bearings, shaft, support_force

__all__ = ["build", "axial_loads", "SPANS_MM", "FA_OVER_FR", "S_KEYS", "S_MM", "FLOATING", "LOAD_FACTOR",
           "POSITION"]

SPANS_MM = (100.0, 150.0, 200.0, 250.0, 300.0)      # proposals.md s005: span 100-300 mm
# proposals.md s005: Fa/Fr = 0 and 0.5; both axial senses (review of s005: the sense sets the sign of the moment)
FA_OVER_FR = (0.0, 0.5, -0.5)
# diametral clearance [mm]: s004's values (OPEN R5-53 there: no ISO 5753); "CN-like" is the normal group, the
# "C0" of proposals.md (C0 is the old name of the normal group)
S_MM = {"0": 0.0, "CN-like": 0.010, "C3-like": 0.020}
S_KEYS = tuple(S_MM)
FLOATING = ("ball", "roller")                       # bearing at B; roller: its Mz is "diagnostic" in the code (SP-16)
# OPEN R5-176: one load level (s004's reference torque, factor 1) and the offset gear only (the centred gear with two
# equal bearings is s004's no-rotation limit; with Mz it adds nothing new)
LOAD_FACTOR = 1.0
POSITION = "offset"


def build(*, span_mm: float, floating: str, s_key: str, Fa_over_Fr: float) -> dict:
    """The loaded shaft, its two bearings and the axial load at A, unsolved.

    Parameters
    ----------
    span_mm : float
        Distance between the pins [mm]; the gear at 0.3 ``span_mm``.
    floating : str
        Bearing at B: "ball" (6210-like) or "roller" (NU210-like).
    s_key : str
        Clearance label, a key of ``S_MM``.
    Fa_over_Fr : float
        Axial load at A over the rigid-pin radial reaction at A [-], signed in the solver's axial sense (+: the
        direction of ``A sin(alpha_0)`` in the ball solver).

    Returns
    -------
    dict
        ``system`` (ShaftSystem, pins at 0 and ``span_mm``), ``bearings`` (A, B), ``s_mm`` and the inputs. The
        axial load itself needs the rigid-pin reaction: :func:`axial_loads`.
    """
    if s_key not in S_MM:
        raise ValueError(f"s_key must be one of {S_KEYS}, got {s_key!r}")
    return {"system": shaft(load_factor=LOAD_FACTOR, position=POSITION, span_mm=span_mm),
            "bearings": bearings(S_MM[s_key], floating), "Fa_over_Fr": Fa_over_Fr, "span_mm": span_mm,
            "floating": floating, "s_key": s_key, "s_mm": S_MM[s_key]}


def axial_loads(rigid, Fa_over_Fr: float) -> tuple[float, float]:
    """``(Fa_A, Fa_B)`` [N]: ``Fa_over_Fr`` x the rigid-pin reaction at A (``ShaftResults``; A is the pin at x = 0,
    the first bearing of s004's shaft), 0 at the floating bearing B. Held fixed while Mz changes the reactions."""
    return Fa_over_Fr * float(np.linalg.norm(support_force(rigid, 0))), 0.0
