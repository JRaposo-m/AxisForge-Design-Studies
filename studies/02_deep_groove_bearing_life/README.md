# 02 — Deep-groove ball bearing, life under combined load

**Status: planned.** The capability declaration below is the target shape; `run.py` is not written yet.

## Problem

A single deep-groove ball bearing (a catalogue size such as a 6210) under a combined radial and axial load applied directly, without a shaft in between — isolate the ISO/TS 16281 point-contact solve and ISO 281 life calculation from everything else in the pipeline.

## What it exercises

Construction (`bearings`, `bearing_families`) → ElementAnalysis (`point_contact`). No `MeshLoads` or shaft `Resolution` — the load is applied straight to the bearing, which is deliberate: this study is about the bearing solver on its own, not the shaft that would normally deliver its reactions.

## Target capability declaration

```python
from axisforge.fixtures.capabilities import (
    Capabilities, ConstructionCapabilities, MeshLoadsCapabilities,
    ResolutionCapabilities, ElementAnalysisCapabilities,
)

construction = ConstructionCapabilities(
    bearings=("Bearing", "BearingCatalog"),
    bearing_families=("DeepGrooveBallFamily",),
)
mesh_loads = MeshLoadsCapabilities(construction=construction)
resolution = ResolutionCapabilities(mesh_loads=mesh_loads)
analysis = ElementAnalysisCapabilities(
    resolution=resolution,
    point_contact=True,
    outputs={"point_contact": {"txt", "csv"}},
)

CAPS = Capabilities(element_analysis=analysis)
CAPS.validate_or_raise()
globals().update(CAPS.resolve())
```

Note this study does not enable `resolution.bearing_loads` — that stage exists to derive the bearing's reactions *from a solved shaft*. Here the load is given directly, so `point_contact` is computed straight from it. Whether `point_contact` should be allowed without `bearing_loads` at all, or whether this study should instead go through a minimal one-element shaft just to stay inside the normal chain, is an open question for when this study is actually built — see `docs/ARCHITECTURE.md` in the AxisForge repository for the exact prerequisite rules as they stand today.

## Expected output

- Console: bearing construction validity, solver convergence (iterations, residual).
- `report.txt`: contact load distribution (`phi_global`, `Q_j`), `Q_ci`/`Q_ce` capacity, `Q_ei`/`Q_ee` equivalent load.
- `results.csv`: the contact distribution table.

See [`../../sample_outputs/02_deep_groove_bearing_life/`](../../sample_outputs/02_deep_groove_bearing_life/) once populated.
