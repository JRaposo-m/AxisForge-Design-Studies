"""
axisforge_bridge/construction/outputs/bearing_report.py

Report block for the `bearings` Construction domain -- mirrors
axisforge_bridge/construction/bearings/ one-to-one.

bearing_block() is a THIN wrapper -- Bearing.summary() (core) already
covers everything this domain's report needs (family, bearing_type,
duty, position, arrangement, d/D/b, C, enabled analyses). It exists
anyway so that bearings has the same one-call-per-domain shape as
shafts/gears/systems.

CONTENT ONLY -- this module builds a text block, it does not write any
file. text_report.py is the single place that assembles every domain's
blocks into the one combined construction report .txt and writes it.

Dependency (core only, read-only access):
  axisforge.core.machine_elements.bearings.bearing
      Bearing
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from axisforge.core.machine_elements.bearings.bearing import Bearing


def bearing_block(bearing: "Bearing") -> str:
    """ASCII block for one Bearing -- delegates entirely to
    Bearing.summary() (core already formats this well)."""
    return bearing.summary()
