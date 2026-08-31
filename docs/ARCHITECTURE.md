# The capability pipeline

Every study in this repository is built the same way: it declares what it needs from AxisForge as a chain of four stages, and the chain enforces the order the physics actually requires.

```
Construction  →  MeshLoads  →  Resolution  →  ElementAnalysis
```

This lives in AxisForge itself, under `fixtures/capabilities/`. A study imports it, declares what it needs stage by stage, and gets back exactly those objects — nothing else is imported.

## The four stages

**1. Construction.** Instantiate the objects: the shaft and its sections, the bearings, the gears, the container that holds a shaft/gear system together. Nothing is solved yet — this is geometry and catalogue data only.

**2. MeshLoads.** A gearbox's mesh force cannot be known before its power flow is. `power_flow` resolves the torque and speed at every shaft in the system; `gear_forces`, which requires `power_flow`, turns that resolved torque into the tangential/radial/axial force at each gear mesh — the load the next stage needs as input.

**3. Resolution.** `shaft_fem` solves the shaft as a 1D Timoshenko beam under the loads assembled so far. `bearing_loads`, which requires `shaft_fem`, takes the resulting bearing reactions and solves each bearing's internal load distribution per ISO/TS 16281 — how load splits across the rolling elements.

**4. ElementAnalysis.** The per-element checks that only make sense once the system is resolved: `point_contact` and `line_contact` (which require `bearing_loads`) compute contact stress, capacity and equivalent dynamic load for ball and roller bearings respectively; `multirow_capacity` combines per-row life into a bearing life; `shaft_static_report` reads back the shaft's stress and deflection.

## Why the order is enforced, not just documented

Each stage is a small, immutable declaration — a set of names to import, or a boolean to switch an analysis on — and each one holds the previous stage as a field, so a `ResolutionCapabilities` cannot exist without a `MeshLoadsCapabilities` already built. Before anything is imported, every stage's `validate_chain()` walks back through all of them: asking for `bearing_loads` without `shaft_fem`, or `gear_forces` without `power_flow`, raises immediately and names the exact missing prerequisite.

The alternative — trusting a script to call things in the right order — fails silently. A bearing internal-load solve run against stale or default shaft reactions does not crash; it returns a number, and the number is wrong. Enforcing the order at the point where capabilities are declared turns that class of mistake into an exception raised before a single object is built.

## Console vs. report vs. data

Three separate things happen after an analysis runs, and none of them force the others:

- **Console** confirms the run was *sound*: construction validated, the bearing solver converged, in how many iterations, with what residual. It says nothing about whether the resulting numbers are good — only whether the calculation is trustworthy.
- **`report.txt`** is the same result, formatted for a person to read.
- **`results.csv`** is the same result again, formatted for a spreadsheet.

An analysis can run with any combination of these three switched on — solve `point_contact` and never print it, or print it but skip the CSV. The dependency runs one way: none of the three can be requested for an analysis that was not itself enabled.

> **Note on current status.** As of this writing, `outputs` in AxisForge's `fixtures/capabilities/` accepts a plain `True`/`False` per analysis — one switch for "pull in the console-facing reporter," full stop. The three-channel split described above (`{"txt", "csv"}` as a set of channels per analysis) is the agreed next step, not yet implemented. The study READMEs in this repository already show the target `outputs={"point_contact": {"txt", "csv"}}` shape; until that lands in AxisForge, treat those declarations as the plan, not a working example.

## Where this comes from

The pipeline, its validation rules and the report/CSV split are documented in full in AxisForge's own `fixtures/README.md`. This document is the short version, written for someone reading this repository without first reading that one.
