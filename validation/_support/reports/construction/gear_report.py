"""
validation/_support/reports/construction/gear_report.py

Report block for the gears of a constructed system.

gear_block(ge) formats one GearElement -- its kinematic role/rotation
plus the underlying SpurHelicalGear's own geometry, which has no
summary() in core (only loose attributes + validate()). "spur" vs
"helical" is inferred the same way
core/mechanical_system/parallel_axis/spur_helical/gear_system.py's own
module docstring defines it: beta_n_deg == 0.0 -> spur, > 0.0 -> helical.

Dependency (core only, read-only access):
  axisforge.core.machine_elements.gears.parallel_axis.gear_properties.spur_helical_gear
      SpurHelicalGear
  axisforge.core.mechanical_system.parallel_axis.spur_helical.shaft_system
      GearElement
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from axisforge.core.mechanical_system.parallel_axis.spur_helical.shaft_system import (
        GearElement,
    )


def gear_block(ge: "GearElement") -> str:
    """
    ASCII block: role, rotation_dir, position, then the gear's own
    geometry (mn, z, x, b, alpha_n_deg, beta_n_deg, d, da, df,
    material_id). ge.gear is assumed to be a SpurHelicalGear.
    """
    g = ge.gear
    tag = ge.label or getattr(g, "label", "") or "gear"
    kind = "helical" if getattr(g, "beta_n_deg", 0.0) > 0.0 else "spur"
    header = f"  -- {tag} ({kind}, role={ge.role}) "
    lines = [
        header + "-" * max(0, 60 - len(header)),
        f"    rotation_dir   : {ge.rotation_dir!r}",
        f"    position       : {ge.position:.2f} mm",
        f"    mn / z / x     : {g.mn:.3f} mm / {g.z} / {g.x:.4f}",
        f"    alpha_n / beta_n: {g.alpha_n_deg:.2f} deg / {g.beta_n_deg:.2f} deg",
        f"    b (face width) : {g.b:.2f} mm",
        f"    d / da / df    : {g.d:.3f} / {g.da:.3f} / {g.df:.3f} mm",
        f"    material_id    : {g.material_id or '(none)'}",
    ]
    return "\n".join(lines)
