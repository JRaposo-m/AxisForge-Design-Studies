"""The system of s104: the FZG C40 spur pair at K9, its contacts near the ends A and E of the path, NOT solved.

The pair is v107's (``build_pair("C40")``: FEUP test gears, z 16 / 24, m 4.5 mm, b 40 mm, x 0.1817 / 0.1715; E 206
GPa, nu 0.3 assumed, R-118; the pinion drives, Q-09), imported from v107 (OPEN R5-110, the R2-02 pattern). The load
is the K9 wheel torque 323.27 N m (Martins2015 Table 4, on the wheel, R-124), gear 1 torque ``T2 z1/z2``; the
normal force ``F_bn`` of AxisForge ``mesh_forces`` is shared between the lines in contact by AxisForge's rigid L0
rule (``load_sharing``: one load per unit length on every line; 1/2 near A and E, double contact).

Contacts: ``CylindricalGearContact.from_pair`` at the flank diameter ``dy1`` of the point, the two true involute
flanks with their ends (``InvoluteFlank``): the flank of a gear stops at its tip circle and the bridge puts the
surface ``SAG_PAST_END`` = 1 mm below past it (a sharp, unmodified tip; OPEN SB-65). Near A the edge is the tip of
gear 2 (on bridge -x), near E the tip of gear 1 (on bridge +x). At a ``dy1`` the contact does not depend on the
driver; the driver only names A and E (the other driver swaps which gear's tip is at which end).

Units: AxisForge mm, N, N m; the bridge converts to SI inside ``from_pair``.
"""
from __future__ import annotations

import math

from scipy.optimize import brentq
from slippy_bridge.contact import hertz
from slippy_bridge.contact.core.machine_elements.gears import CylindricalGearContact

# OPEN R5-110 (R2-02 pattern): the FZG pair, torque table and assumptions are v107's, imported
from studies.v100_dry_contact.v107_fzg_contact_path.construction import (DRIVER, E_MPA, FAMILY, NU, PAIRS,
                                                                        WHEEL_TORQUE_NM, build_pair)

__all__ = ["fzg_c40", "contact_at", "contacts_near_end", "edge_of", "TARGET_D_OVER_B", "STAGE", "PAIR", "Z"]

PAIR, STAGE = "C40", "K9"
Z = PAIRS[PAIR]["z"]
# d_end/b of the points near each end: 0 (the point on the end itself) ... 10 (Hertz expected)
TARGET_D_OVER_B = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0)
# the governing end near each end of the path, and its side on bridge x (tip of gear 1 at +x; gear 2 tip at -x)
_EDGE = {"A": "d_end_tip_2_m", "E": "d_end_tip_1_m"}
# side of each flank end on bridge x (x toward the tip of gear 1): gear 1 tip and gear 2 root on +x, the others on -x
_SIDE = {"d_end_tip_1_m": +1.0, "d_end_root_2_m": +1.0, "d_end_root_1_m": -1.0, "d_end_tip_2_m": -1.0}


def fzg_c40(stage: str = STAGE) -> dict:
    """The C40 pair at a FZG stage, unsolved: ``pair``, ``wheel_torque_Nm``, ``F_bn_N`` [N], ``path`` (``g_y`` of A..E
    [mm], pinion driving) and ``data`` (v107 ``PAIRS["C40"]``, for ``v107.check``)."""
    pair = build_pair(PAIR)
    torque = WHEEL_TORQUE_NM[stage]
    # gear 1 torque from the wheel torque (R-124); the sign and the rotation sense passed to mesh_forces follow v107
    # (D-0010 signs); only |F_bn| is used here
    T1 = -torque * Z[0] / Z[1]
    forces = FAMILY.mesh_forces(pair, T1, -1 if T1 < 0 else 1)
    return {"pair": pair, "wheel_torque_Nm": torque, "F_bn_N": forces.F_bn_N, "path": FAMILY.path_points(pair, DRIVER),
            "data": PAIRS[PAIR], "stage": stage}


def contact_at(system: dict, g_y_mm: float, label: str) -> CylindricalGearContact:
    """The bridge contact at the path coordinate ``g_y`` [mm] (from C), under the L0 share of ``F_bn``."""
    pair = system["pair"]
    dy1 = 2.0 * math.hypot(pair.T1C + g_y_mm, pair.gear1.rb)    # flank diameter of gear 1 (as from_pair)
    (c,) = CylindricalGearContact.from_pair(pair, points={label: dy1}, driver=DRIVER, F_bn_N=system["F_bn_N"],
                                            E=(E_MPA, E_MPA), nu=(NU, NU))
    return c


def edge_of(contact, end: str) -> dict:
    """The governing flank end near ``end``: ``name``, ``d_end_m`` [m] and its signed position ``x_edge_m`` on
    bridge x [m] (the contact point at 0). "A": the tip of gear 2 (-x); "E": the tip of gear 1 (+x); any other point
    ("C"): the nearest of the four ends, with its own side."""
    if end in _EDGE:
        name = _EDGE[end]
    else:
        name = min(contact.end_distances().items(), key=lambda kv: kv[1])[0]
    d = getattr(contact, name)
    return {"edge": name, "d_end_m": d, "x_edge_m": _SIDE[name] * d}


def contacts_near_end(system: dict, end: str, targets=TARGET_D_OVER_B) -> list:
    """The contacts whose distance to the end's tip is ``targets`` x the Hertz half-width ``b`` of that contact.

    ``g_y`` is found by ``brentq`` from the end inward (``d_end / b`` grows monotonically from 0 at the end).
    Returns ``(target, g_y_mm, contact)`` tuples; the contact is labelled ``"<end>_<target>"``.
    """
    g_end = float(system["path"][end])
    inward = 1.0 if end == "A" else -1.0                         # A is the start of the path (g_y < 0), E its end

    def ratio(offset_mm, target):
        c = contact_at(system, g_end + inward * offset_mm, "probe")
        return edge_of(c, end)["d_end_m"] / hertz(c).b - target

    out = []
    for t in targets:
        offset = 0.0 if t == 0.0 else brentq(ratio, 0.0, 3.0, args=(t,), xtol=1e-9)   # [mm] 3 mm > 10 b
        g_y_mm = g_end + inward * offset
        out.append((t, g_y_mm, contact_at(system, g_y_mm, f"{end}_{t:g}")))
    return out
