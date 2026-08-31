# 03 — Single-stage helical gearbox drivetrain

**Status: planned.** The capability declaration below is the target shape; `run.py` is not written yet.

## Problem

A complete single-stage reduction: an input shaft and an output shaft connected by one helical gear mesh, each shaft carried on two deep-groove ball bearings. Power and speed are given at the input; everything downstream — mesh forces, shaft deflection, bearing reactions, bearing internal load distribution and life — is resolved from that one input. This is the study that exercises the whole pipeline end to end.

## What it exercises

All four stages, in full: `Construction` (shaft, bearings, bearing families, gears, system) → `MeshLoads` (`power_flow`, `gear_forces`) → `Resolution` (`shaft_fem`, `bearing_loads`) → `ElementAnalysis` (`point_contact`, `shaft_static_report`).

## Target capability declaration

```python
from axisforge.fixtures.capabilities import (
    Capabilities, ConstructionCapabilities, MeshLoadsCapabilities,
    ResolutionCapabilities, ElementAnalysisCapabilities,
)

construction = ConstructionCapabilities(
    shaft=("Shaft", "ShaftSection", "Shoulder"),
    bearings=("Bearing", "BearingCatalog"),
    bearing_families=("DeepGrooveBallFamily",),
    gears=("SpurHelicalGear", "SpurHelicalGearMeshing"),
    system=("ShaftSystem", "GearElement", "SpurHelicalMeshLink", "SpurHelicalGearSystem"),
)
mesh_loads = MeshLoadsCapabilities(
    construction=construction,
    power_flow=True,
    gear_forces=True,
    outputs={"gear_forces": {"txt"}},
)
resolution = ResolutionCapabilities(
    mesh_loads=mesh_loads,
    shaft_fem=True,
    bearing_loads=True,
)
analysis = ElementAnalysisCapabilities(
    resolution=resolution,
    point_contact=True,
    shaft_static_report=True,
    outputs={
        "point_contact": {"txt", "csv"},
        "shaft_static_report": {"txt", "csv"},
    },
)

CAPS = Capabilities(
    element_analysis=analysis,
    loads=("TorqueLoad", "RadialLoad"),
)
CAPS.validate_or_raise()
globals().update(CAPS.resolve())
```

## Expected output

One `report.txt` / `results.csv` pair covering: gear mesh forces per stage, shaft deflection for both shafts, and contact load distribution plus capacity/life for all four bearings. Console output confirms, in order: construction validity for every element, power-flow resolution, shaft FEM convergence (it is a direct linear solve, so this is closer to a sanity check than an iteration count), and bearing solver convergence with iteration count and residual for each of the four bearings.

See [`../../sample_outputs/03_helical_gearbox_drivetrain/`](../../sample_outputs/03_helical_gearbox_drivetrain/) once populated.
