"""
axisforge_bridge/construction/outputs/bearing_report.py

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
