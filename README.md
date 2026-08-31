# AxisForge Design Studies

Worked examples that run [AxisForge](https://github.com/JRaposo-m/AxisForge-Shaft-Bearing-Gear-System-Analysis-Platform) — a deterministic, standards-based CAE engine for shafts, rolling bearings and parallel-axis gear systems — end to end on realistic drivetrain problems, and show exactly what it produces.

This repository is not the engine. It is the proof that the engine works: a set of self-contained studies, each declaring what it needs through AxisForge's capability pipeline, each producing a plain-text report and a CSV of its results.

← the engine itself lives in [AxisForge](https://github.com/JRaposo-m/AxisForge-Shaft-Bearing-Gear-System-Analysis-Platform)

---

## What AxisForge is

A modular platform for the mechanical design chain that shows up in every reduction gearbox: a shaft carrying bending and torsion, rolling bearings supporting it, and one or more gear meshes driving it. Every calculation traces to a published standard rather than a rule of thumb:

| Domain | Method |
|---|---|
| Shaft | 1D Timoshenko beam FEM |
| Rolling bearings | ISO 281 (basic rating life), ISO/TS 16281 (internal load distribution) |
| Parallel-axis gears | ISO 21771 (geometry), ISO 6336 (load capacity — in progress) |

Nothing in the engine is a black box: every solver is deterministic, GUI-independent, and exposes the intermediate objects it builds along the way, so a result can always be traced back to the geometry and loads that produced it.

## What this repository demonstrates

Each study under [`studies/`](studies/) is a realistic design problem — a stepped shaft, a deep-groove ball bearing selection, a full helical gearbox drivetrain — solved through AxisForge's **capability pipeline**: a script declares what it needs, in the order the physics actually requires it, and the pipeline refuses to run a step out of order.

```
Construction  →  MeshLoads  →  Resolution  →  ElementAnalysis
```

1. **Construction** — build the shaft, bearings and gears.
2. **MeshLoads** — resolve the gearbox's power flow, then the mesh forces it implies.
3. **Resolution** — solve the shaft (FEM) and, from its results, the bearings' internal load distribution.
4. **ElementAnalysis** — per-element checks that consume those resolved results: bearing contact and life, shaft stress and deflection.

A study cannot request a later stage without its prerequisite already declared — asking for a bearing's internal load distribution without first resolving the shaft raises immediately, naming exactly what is missing, rather than failing deep inside a solve or silently returning a wrong number. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design.

## What each study produces

Every study writes two things, deliberately kept apart:

- **Console** — a short, human-readable confirmation that the run is sound: did construction validate, did the bearing solver converge, in how many iterations, with what residual. Nothing about whether the resulting numbers are "good" — that judgement is the reader's.
- **A report file** (`report.txt`, plain ASCII) and a **data file** (`results.csv`) — the actual numbers: contact load distribution, capacity, life, shaft deflection. The text file for reading, the CSV for pulling into Excel or a notebook.

Sample output for each study is checked in under [`sample_outputs/`](sample_outputs/), so the shape of a run is visible without needing to execute anything.

## Status

| Study | Domain | Status |
|---|---|---|
| [`01_stepped_shaft_static`](studies/01_stepped_shaft_static) | Shaft | Planned |
| [`02_deep_groove_bearing_life`](studies/02_deep_groove_bearing_life) | Bearing | Planned |
| [`03_helical_gearbox_drivetrain`](studies/03_helical_gearbox_drivetrain) | Full drivetrain | Planned |

The capability pipeline these studies run through already exists in AxisForge (`fixtures/capabilities/`); the studies themselves, and the report/CSV writers they depend on, are being built one at a time. Each study's own `README.md` states exactly what stage it has reached.

## Setup

```bash
git clone https://github.com/JRaposo-m/AxisForge-Shaft-Bearing-Gear-System-Analysis-Platform.git axisforge
git clone <this-repo-url> axisforge-design-studies
cd axisforge-design-studies
pip install -r requirements.txt
pip install -e ../axisforge
```

Run a study from its own folder:

```bash
cd studies/01_stepped_shaft_static
python run.py
```

## Layout

```
axisforge-design-studies/
├── docs/
│   └── ARCHITECTURE.md      The capability pipeline, in depth
├── studies/                 One folder per worked example
│   ├── 01_stepped_shaft_static/
│   ├── 02_deep_groove_bearing_life/
│   └── 03_helical_gearbox_drivetrain/
├── sample_outputs/          Checked-in report.txt / results.csv per study
├── requirements.txt
└── LICENSE
```
