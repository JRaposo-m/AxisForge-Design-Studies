"""PROVISIONAL (scaffold 2026-10-17): structure agreed in debate_gears-lubrication-scaffold_2026-10-17; not validated.

Lubrication part of the bridge: lubricant models, UnifiedReynoldsSolver and IterSemiSystem steps on
the surfaces of ``slippy_bridge.contact`` (``ehl.run``), with the conventions of
``Tese/80_Inbox/finding_composite_vs_two_profiles_2026-10-10.md`` (rolling along axis 0,
radius_in_rolling_direction = R_x, hertzian_half_width = s_x). Imports from ``contact``, never the
reverse.
"""
from .ehl import FILM_KEYS, run

__all__ = ["run", "FILM_KEYS"]
