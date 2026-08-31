# 01 — Stepped shaft, static analysis

**Status: planned.** The capability declaration below is the target shape; `run.py` is not written yet. This file will be updated to "implemented" once it runs against real AxisForge output.

## Problem

A three-section stepped shaft (seat–body–seat), simply supported on two bearings, carrying an external radial load and torque applied directly — no gear mesh. The simplest problem that still exercises a full solve: build the shaft, solve it, read back deflection and report it.

## What it exercises

Construction (`shaft`) → Resolution (`shaft_fem`) → ElementAnalysis (`shaft_static_report`). No `MeshLoads` stage, because the loads are external rather than coming from a gear mesh — a deliberate choice to keep this first study to the shortest path through the pipeline.

## Target capability declaration

```python
from axisforge.fixtures.capabilities import (
    Capabilities, ConstructionCapabilities, MeshLoadsCapabilities,
    ResolutionCapabilities, ElementAnalysisCapabilities,
)

construction = ConstructionCapabilities(
    shaft=("Shaft", "ShaftSection", "Shoulder"),
)
mesh_loads = MeshLoadsCapabilities(construction=construction)
resolution = ResolutionCapabilities(mesh_loads=mesh_loads, shaft_fem=True)
analysis = ElementAnalysisCapabilities(
    resolution=resolution,
    shaft_static_report=True,
    outputs={"shaft_static_report": {"txt", "csv"}},
)

CAPS = Capabilities(
    element_analysis=analysis,
    loads=("TorqueLoad", "RadialLoad"),
)
CAPS.validate_or_raise()
globals().update(CAPS.resolve())
```

## Expected output

- Console: shaft construction validity, FEM solve confirmation.
- `report.txt`: deflection (`v_xz`, `v_xy`, `v`) at every mesh node.
- `results.csv`: the same table, one row per node.

See [`../../sample_outputs/01_stepped_shaft_static/`](../../sample_outputs/01_stepped_shaft_static/) once populated.
