"""The systems of s102: s001's 6210-like deep groove ball bearing at two clearances, returned NOT solved.

The cases (proposals.md s102): diametral clearance s = 0 and 20 um, Fr = 2500 and 5000 N, Fa / Fr = 0 and 0.5; the
rings untilted. The bearing is s001's (``deep_groove``), through its public API. Units: mm, N (AxisForge).
"""
from __future__ import annotations

from studies.s000_axisforge.s001_bearing_clearance_and_contact_angle import deep_groove

__all__ = ["cases", "S_MM", "FR_N", "FA_OVER_FR", "LAW_DELTA_FACTORS"]

S_MM = (0.0, 0.020)
FR_N = (2500.0, 5000.0)
FA_OVER_FR = (0.0, 0.5)
# approach of each BEM law, per contact, as factors of its Hertz approach at the case's Q_max: loads 0.003..1.66
# Q_max, so every loaded ball of the case is inside the table (NR-09: 1.2 x the largest delta_j at least)
LAW_DELTA_FACTORS = (0.02, 0.04, 0.08, 0.16, 0.32, 0.6, 1.0, 1.4)


def cases() -> list[dict]:
    """``[{"name", "s_mm", "Fr_N", "Fa_N", "bearing"}]``, the bearing assembled, unsolved."""
    return [{"name": f"6210 s {1e3 * s:g} um, Fr {Fr:g} N, Fa/Fr {r:g}", "s_mm": s, "Fr_N": Fr, "Fa_N": r * Fr,
             "Fa_over_Fr": r, "bearing": deep_groove(s)} for s in S_MM for Fr in FR_N for r in FA_OVER_FR]
