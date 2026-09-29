"""
axisforge_bridge/construction/outputs/system_report.py

Report blocks for the `systems` Construction domain -- mirrors
axisforge_bridge/construction/systems/ one-to-one.

Four functions, each "the systems report" for one facet of a
SpurHelicalGearSystem:

  topology_block(system)   -- delegates to SpurHelicalGearSystem.summary()
                              (core already formats topology + resolved
                              state -- source shaft, link count, and,
                              once resolved, per-shaft T_out/rotation_dir/
                              shaft_position).
  shaft_system_block(ss)   -- delegates to ShaftSystem.summary() (core
                              already formats length/speed/design life/
                              axis offset/bearing-gear-load counts).
  loads_block(ss)          -- ShaftSystem.loads has no dedicated core
                              summary(); this lists every Load
                              position-sorted, tagged with its own
                              source ("user"/"gear_mesh"/
                              "bearing_reaction") so a caller-declared
                              load is never confused with one resolve()
                              injected.
  mesh_block(link)         -- link-level info (shaft_a -> shaft_b,
                              phi_deg, torque_split, distribute_loads,
                              label) plus SpurHelicalMeshLink.meshing's
                              own .summary() (SpurHelicalGearMeshing,
                              core).

CONTENT ONLY -- this module builds text blocks, it does not write any
file. text_report.py is the single place that assembles every domain's
blocks into the one combined construction report .txt and writes it.

Dependency (core only, read-only access):
  axisforge.core.mechanical_system.parallel_axis.spur_helical.shaft_system
      ShaftSystem
  axisforge.core.mechanical_system.parallel_axis.spur_helical.gear_system
      SpurHelicalGearSystem, SpurHelicalMeshLink
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from axisforge.core.mechanical_system.parallel_axis.spur_helical.shaft_system import (
        ShaftSystem,
    )
    from axisforge.core.mechanical_system.parallel_axis.spur_helical.gear_system import (
        SpurHelicalGearSystem, SpurHelicalMeshLink,
    )


def topology_block(system: "SpurHelicalGearSystem") -> str:
    """Delegates to SpurHelicalGearSystem.summary()."""
    return system.summary()


def shaft_system_block(ss: "ShaftSystem") -> str:
    """Delegates to ShaftSystem.summary()."""
    return ss.summary()


def loads_block(ss: "ShaftSystem") -> str:
    """
    ASCII block listing every Load on ss, position-sorted (ss.loads is
    already sorted), tagged with its own source so caller-declared
    loads and resolve()'s own gear-mesh loads are never confused.
    """
    loads = ss.loads
    if not loads:
        return "  loads: (none)"
    lines = [f"  loads ({len(loads)} total):"]
    for ld in loads:
        lines.append(f"    [{ld.source}] {ld!r}")
    return "\n".join(lines)


def mesh_block(link: "SpurHelicalMeshLink") -> str:
    """
    ASCII block for one SpurHelicalMeshLink: the link-level line
    (endpoints, phi_deg, torque_split, distribute_loads, label) plus
    the underlying meshing object's own .summary() (SpurHelicalGearMeshing,
    core -- working centre distance, working pressure angle, contact
    ratios).
    """
    lines = [
        f"  {link.shaft_a.name} -> {link.shaft_b.name}   "
        f"phi_deg={link.phi_deg:.2f}   "
        f"torque_split={link.torque_split!r}   "
        f"distribute_loads={link.distribute_loads}   "
        f"label={link.label!r}",
        link.meshing.summary(),
    ]
    return "\n".join(lines)
