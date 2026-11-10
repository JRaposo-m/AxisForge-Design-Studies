"""The system of s004: a gear-loaded shaft on two pins and the two bearings that will sit at the pins, NOT solved.

The shaft is s006's pinion shaft (``stage``: uniform d 40 mm, S355, pins at 0 and L, spur pinion of the
H501 pair at x_g; ``# OPEN R5-50``: s004 imports s006's construction through its package API, the model
shared by the later study, D-0036 amendment item 11). Span L = 200 mm; gear "offset" (x_g = 0.3 L: the two
reactions differ, 0.7 / 0.3 of the mesh force) or "centred" (equal reactions: the limit where two equal
bearings give no rigid-body rotation). The torque scales the load: ``load_factor`` x ``TORQUE_REF_NM``.

The bearings are s001's builds (6210-like deep groove ball, NU210-like cylindrical roller, d 50 mm;
``# OPEN R5-51``: the bore does not match the 40 mm shaft; the FEM sees zero-width pins, the bearing model
sees only the reaction, so the mismatch has no effect on the numbers, it only makes the assembly fictitious).
Pin A (x = 0, locating) carries a deep groove ball bearing; pin B (x = L, floating) a deep groove ball
bearing or, as the variant, a cylindrical roller bearing. Diametral clearance s (``S_MM``).

Construction rule (proposals.md s004): no load on a bearing node and no distributed load on an element next
to it (``K d = R + f_ext`` then gives the reaction alone at the pin): the gear sits at 0.3 L or 0.5 L, point loads.

Spur only (beta 0): no axial force, no couple, so no J-16 tag; the mesh force lies on the line of action l,
inclined to both principal planes, so both planes carry load.

Units: mm, N, MPa, N.mm, rad; torque N.m.
"""
from __future__ import annotations

from studies.s000_axisforge.s001_bearing_clearance_and_contact_angle import cylindrical_roller, deep_groove
from studies.s000_axisforge.s006_mesh_misalignment import stage

__all__ = ["shaft", "bearings", "SPAN_MM", "POSITIONS", "TORQUE_REF_NM", "LOAD_FACTORS", "S_MM", "FLOATING"]

SPAN_MM = 200.0
POSITIONS = ("offset", "centred")
# OPEN R5-52: proposals ask Fr 0.1 - 1 x C0r; AxisForge holds no static rating (R5-40). The reference torque
# 330 N.m puts about 10 kN on the line of action (F_bt = 2000 T / (d_w1 cos alpha_wt), d_w1 70.7 mm), so the
# larger reaction at factor 1 is about 7 kN = 0.28 C of the 6210 build (C 24.8 kN), s001's load level x 1.4
TORQUE_REF_NM = 330.0
LOAD_FACTORS = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
# OPEN R5-53: ISO 5753 (radial clearance groups) is not in the library: "0, CN-, C3-, C4-like" are s001 values
# that sit inside the usual DEEP GROOVE BALL groups of a 50 mm bore (normal about 6-23 um, C3 18-36, C4 30-51; not
# checked); ISO 5753-1 tabulates cylindrical roller bearings separately, the same labels are applied to the NU210 build
# only as names: 0, 10, 20, 40 um
S_MM = {"0": 0.0, "CN-like": 0.010, "C3-like": 0.020, "C4-like": 0.040}
FLOATING = ("ball", "roller")


def shaft(*, load_factor: float = 1.0, position: str = "offset", span_mm: float = SPAN_MM):
    """The loaded pinion shaft (``ShaftSystem``), unsolved: s006 ``stage`` at ``load_factor`` x ``TORQUE_REF_NM``.

    Returns
    -------
    ShaftSystem
        Pins at 0 (locating) and ``span_mm`` (non-locating), the mesh loads of the spur pinion at x_g,
        the input torque at x = 0.
    """
    return stage(span_mm=span_mm, position=position, torque_Nm=load_factor * TORQUE_REF_NM).shafts[0]


def bearings(s_mm: float, floating: str = "ball") -> tuple:
    """The two bearings, (at pin A, at pin B), unsolved: s001 builds with diametral clearance ``s_mm`` [mm].

    ``floating``: "ball" (6210-like at B too) or "roller" (NU210-like at B, 40 laminae).
    """
    if floating not in FLOATING:
        raise ValueError(f"floating must be one of {FLOATING}, got {floating!r}")
    b = deep_groove(s_mm, label="B_6210") if floating == "ball" else cylindrical_roller(s_mm, label="B_NU210")
    return deep_groove(s_mm, label="A_6210"), b
