"""
axisforge_bridge/construction/outputs/text_report.py

THE principal writer for the whole Construction domain: assembles every
Construction domain's own content blocks --
  axisforge_bridge/construction/outputs/shaft_report.py   -> shaft_geometry_block()
  axisforge_bridge/construction/outputs/bearing_report.py -> bearing_block()
  axisforge_bridge/construction/outputs/gear_report.py    -> gear_block()
  axisforge_bridge/construction/outputs/system_report.py  -> topology_block(),
                                                               shaft_system_block(),
                                                               loads_block(),
                                                               mesh_block()
  axisforge_bridge/construction/outputs/fem_report.py     -> fem_results_block()
-- in Construction order (topology, then per shaft: geometry, bearings,
gears, shaft-system summary + loads [+ FEM results, if provided], then
meshes) into ONE combined .txt file. Those sibling modules build text
blocks only -- this is the only module in the package that actually
opens a file and writes.

write_construction_report(system, path, title="", fem_results=None) is
the only thing this module does. ONE combined .txt file, covering the
whole system, rather than several small per-domain files.

fem_results is OPTIONAL and keyed by ShaftSystem.name (e.g.
{"shaft_1_system": result_1, "shaft_2_system": result_2}), since FEM is
a separate solve step (RigidSupportFEMSolver.solve(ss)) the caller may
or may not have run yet -- the report must stay usable for a system
that's only been constructed, not solved. A shaft with no matching
entry simply gets no FEM section (not an error).

Writes with encoding="utf-8" explicitly: SpurHelicalGearSystem.summary()
and ShaftSystem.summary() (used by systems.topology_block/
shaft_system_block) use box-drawing / degree / middle-dot Unicode
glyphs, and Python's default text-file encoding on Windows is the
system codepage (commonly cp1252), which does not cover them and would
raise UnicodeEncodeError on write.

Dependency (core only, read-only access):
  axisforge.core.mechanical_system.parallel_axis.spur_helical.gear_system
      SpurHelicalGearSystem
  axisforge.results.fem_results.shaft_results
      ShaftResults
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .shaft_report import shaft_geometry_block
from .bearing_report import bearing_block
from .gear_report import gear_block
from .system_report import topology_block, shaft_system_block, loads_block, mesh_block
from .fem_report import fem_results_block

RULE = "=" * 72
SUB = "-" * 72

if TYPE_CHECKING:  # pragma: no cover
    from axisforge.core.mechanical_system.parallel_axis.spur_helical.gear_system import (
        SpurHelicalGearSystem,
    )
    from axisforge.results.fem_results.shaft_results import ShaftResults


def write_construction_report(
    system: "SpurHelicalGearSystem",
    path: "str | Path",
    title: str = "",
    fem_results: "dict[str, ShaftResults] | None" = None,
) -> str:
    """
    Write ONE .txt covering the whole Construction domain for `system`:
    topology/resolved state, then per shaft (in system.shafts order)
    its geometry, bearings, gears, ShaftSystem summary and loads (and,
    if `fem_results` has an entry keyed by that shaft's name, its FEM
    envelope + bearing reactions), then every mesh. Returns the written
    text, so a caller that wants to inspect/verify it doesn't have to
    re-open the file it just wrote.

    Parameters
    ----------
    system : SpurHelicalGearSystem
        Already built. Purely reads `system` -- does not validate or
        resolve anything.
    path : str | Path
        The .txt file to write (parent directory created if missing).
        A relative path resolves against the process's current working
        directory -- NOT this module's location, and not the calling
        script's location either. To have the report always land next
        to the script that built it, build an absolute path at the
        call site: Path(__file__).resolve().parent / "report.txt".
    title : str
        Header title. Defaults to system.label or "Construction report".
    fem_results : dict[str, ShaftResults] | None
        Optional map of ShaftSystem.name -> already-solved ShaftResults
        (e.g. from RigidSupportFEMSolver(settings).solve(ss)). Omit or
        leave a shaft's name out to skip the FEM section for that
        shaft -- this function never solves anything itself.
    """
    header_title = title or system.label or "Construction report"
    sections = [RULE, header_title.center(72), RULE, ""]

    sections.append("TOPOLOGY / RESOLVED STATE")
    sections.append(SUB)
    sections.append(topology_block(system))
    sections.append("")

    fem_results = fem_results or {}

    for ss in system.shafts:
        sections.append(RULE)
        sections.append(f"SHAFT: {ss.name}")
        sections.append(RULE)

        sections.append(shaft_geometry_block(ss.shaft))
        sections.append("")

        if ss.bearings:
            sections.append(f"  bearings ({len(ss.bearings)}):")
            for b in ss.bearings:
                sections.append(bearing_block(b))
        else:
            sections.append("  bearings: (none)")
        sections.append("")

        if ss.gears:
            sections.append(f"  gears ({len(ss.gears)}):")
            for ge in ss.gears:
                sections.append(gear_block(ge))
        else:
            sections.append("  gears: (none)")
        sections.append("")

        sections.append(shaft_system_block(ss))
        sections.append("")
        sections.append(loads_block(ss))
        sections.append("")

        fem_result = fem_results.get(ss.name)
        if fem_result is not None:
            sections.append(fem_results_block(fem_result))
            sections.append("")

    if system.links:
        sections.append(RULE)
        sections.append(f"MESHES ({len(system.links)})")
        sections.append(RULE)
        for link in system.links:
            sections.append(mesh_block(link))
            sections.append("")

    sections.append(RULE)
    text = "\n".join(sections) + "\n"
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text
