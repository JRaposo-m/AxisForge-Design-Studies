# axisforge/fixtures — Analysis Script Building Blocks

Reusable building blocks for assembling `design_xxx.py` analysis scripts — functional runs, pipeline verification and textbook validation. These are **not** pytest tests; unit and validation tests live in `tests/`.

The package is organised around the two stages a design script actually has: **Construction**, which builds objects and solves nothing, and **Studies**, which solves and records. Each depends only on the stage below it, and nothing in `axisforge` imports from `fixtures`.

← back to [project root](../../README.md)

---

## Table of Contents

- [Scope](#scope)
- [Status](#status)
- [Position in the architecture](#position-in-the-architecture)
- [Structure](#structure)
- [Capabilities](#capabilities)
- [Construction fixtures](#construction-fixtures)
- [Studies](#studies)
- [Result libraries](#result-libraries)
- [Report writers](#report-writers)
- [Script structure](#script-structure)
- [Design contracts](#design-contracts)
- [Extending this package](#extending-this-package)

---

## Scope

| In scope | Out of scope |
|---|---|
| Declaring what a script needs and resolving it to imports | Anything `axisforge` itself imports |
| Convenience wrappers that build domain objects from scalar parameters | The domain objects themselves (`core/`) |
| Driving a solver across a whole gearbox and recording the results | The numerical procedure (`solvers/`) |
| Registries keyed by shaft or bearing label | The shape of what they hold (`results/`) |
| Fixed-width text reports | Plotting, GUI, export formats |

---

## Status

| Stage | Module | Contents | Status |
|---|---|---|---|
| — | `capabilities/catalogue.py` | Capability metadata; no `axisforge` imports | Implemented |
| Construction | `construction/construction_capabilities.py` | `ConstructionCapabilities`, `CapabilityError` | Implemented |
| Construction | `construction/shafts/shaft_fixture.py` | `SectionSpec`, `ShaftFixture`, factories | Implemented |
| Construction | `construction/bearings/ball_radial_fixture.py` | Deep groove, angular contact, self-aligning | Implemented |
| Construction | `construction/bearings/ball_thrust_fixture.py` | Single-row and multi-row thrust ball | Implemented |
| Construction | `construction/bearings/roller_radial_fixture.py` | Cylindrical roller | Implemented |
| Construction | `construction/bearings/roller_thrust_fixture.py` | Cylindrical single/multi-row, needle | Implemented |
| Construction | `construction/gears/parallel_axis/fixed/` | External, internal, and both meshing fixtures | Implemented |
| Construction | `construction/systems/parallel_axis/spur_helical/linear_chain_fixture.py` | `ShaftSpec`, `StageSpec`, linear chain factory | Implemented |
| Construction | `construction/*/outputs/*_report.py`, `construction/outputs/text_report.py` | `write_construction_report` and its blocks | Implemented |
| Studies | `studies/study_capabilities.py` | `StudyCapabilities` | Implemented — now also resolves a `shaft_fem.euler_bernoulli_rigid` capability alongside `shaft_fem.timoshenko_rigid` |
| Studies | `studies/shafts/fem_studies/fem_simple.py` | `solve_system` | Implemented — **moved from `studies/shafts/fem_simple.py`**; now takes `theory`/`shear_theory`/`integration_method` (assembled into one `BeamModelSettings` internally) instead of a bare `theory` string |
| Studies | `studies/shafts/fem_studies/results_library.py` | `RigidSupportFEMResultsLibrary` | Implemented — **renamed from `RigidBearingFEMResultsLibrary`**, moved from `studies/shafts/results_library.py` |
| Studies | `studies/shafts/fem_studies/outputs/{resolution_report,comparison_report,plots}.py` | Per-shaft resolution report block, two-library comparison report, matplotlib figures | Implemented — **new since the last pass**, see [Report writers](#report-writers) |
| Studies | `studies/shafts/convergence_studies/{convergence_library,convergence_study}.py` | `ConvergenceResultsLibrary`, `run_convergence` | Present, **but see the open item below** — `convergence_study.py` calls the pre-rename `RigidBearingFEMSolver` API, which no longer exists under that name; confirm before relying on it |
| Studies | `studies/bearings/.../results_library.py` | `BearingResultBundle`, `BearingResultsLibrary` | Implemented |
| Studies | `studies/bearings/.../single_row/rolling_bearing_study.py` | `RollingBearingSolver` | Implemented |
| Studies | `studies/outputs/text_report.py` | `write_studies_report` | Implemented — **renamed from `studies/text_report.py`'s `write_resolution_report`**; now composes SHAFT_FEM, COMPARISON and MESH CONVERGENCE sections in one call, each optional (see [Report writers](#report-writers)) |
| — | full cross-domain integration pipelines (bearing + gear + fatigue in one report) | — | Planned |

> **Open item — `run_convergence()` calls a solver class that no longer exists under that name.** `studies/shafts/convergence_studies/convergence_study.py` does `from axisforge.solvers.machine_elements.shaft.fem_solvers.rigid_support import RigidBearingFEMSolver` and constructs it as `RigidBearingFEMSolver(theory=theory, distribute_gear_labels=distribute_gear_labels)`. The current module only defines `RigidSupportFEMSolver`, constructed from a `BeamModelSettings` instance, not a bare `theory` string (see [`solvers/README.md`](../solvers/README.md#status)). As written, `run_convergence()` cannot be imported. `fem_studies/fem_simple.py`'s `solve_system()` already uses the renamed class and signature correctly — use it as the reference when fixing `convergence_study.py`.

---

## Position in the architecture

```
   core/   mesh/   results/   solvers/          axisforge
                  ▲
                  │  reads only — never the reverse
                  │
        ┌─────────┴─────────┐
        │    fixtures/      │
        │                   │
        │  construction/  ──┼──▶  objects, unsolved
        │        │          │
        │        ▼          │
        │    studies/     ──┼──▶  results, recorded
        └───────────────────┘
```

Two rules hold this in place:

- **`fixtures` reads `axisforge`; `axisforge` never reads `fixtures`.** Capability resolution works by introspecting what each package publishes in its `__init__.py`. It reads only.
- **Studies depend on Construction, never the reverse.** `StudyCapabilities` carries the `ConstructionCapabilities` that built the system it is about to solve, because a study needs to know which capabilities were requested at build time.

This layer is where the run-level registries live. `RigidSupportFEMResultsLibrary`, `ConvergenceResultsLibrary` and `BearingResultsLibrary` are here, not in `solvers/`: a solver computes one thing, and collecting many results under labels is orchestration. `RollingBearingSolver` is here for the same reason — despite its name it is an orchestrator, dispatching a shaft's whole bearing set to the appropriate solvers and merging what comes back.

---

## Structure

```
fixtures/
├── capabilities/
│   └── catalogue.py                    Capability metadata — pure, no axisforge imports
│
├── construction/                       Stage 1 — build objects, solve nothing
│   ├── construction_capabilities.py    ConstructionCapabilities, CapabilityError
│   ├── shafts/
│   │   ├── shaft_fixture.py            SectionSpec, ShaftFixture
│   │   └── outputs/shaft_report.py
│   ├── bearings/
│   │   ├── ball_radial_fixture.py      deep groove · angular contact · self-aligning
│   │   ├── ball_thrust_fixture.py      thrust single · thrust multi-row
│   │   ├── roller_radial_fixture.py    cylindrical roller
│   │   ├── roller_thrust_fixture.py    cylindrical single/multi-row · needle
│   │   └── outputs/bearing_report.py
│   ├── gears/parallel_axis/fixed/
│   │   ├── external_gear_fixture.py    internal_gear_fixture.py
│   │   ├── spur_helical_meshing_fixture.py   internal_meshing_fixture.py
│   │   └── outputs/gear_report.py
│   ├── systems/parallel_axis/spur_helical/
│   │   ├── linear_chain_fixture.py     ShaftSpec, StageSpec, build_linear_system
│   │   └── outputs/system_report.py
│   └── outputs/text_report.py          write_construction_report — composes the four above
│
└── studies/                            Stage 2 — solve and record
    ├── study_capabilities.py           StudyCapabilities
    ├── outputs/text_report.py          write_studies_report — SHAFT_FEM + COMPARISON + CONVERGENCE, each optional
    ├── shafts/
    │   ├── fem_studies/
    │   │   ├── fem_simple.py           solve_system
    │   │   ├── results_library.py      RigidSupportFEMResultsLibrary
    │   │   └── outputs/
    │   │       ├── resolution_report.py    shaft_result_block (per-shaft text)
    │   │       ├── comparison_report.py    shaft_comparison_block, write_comparison_report
    │   │       └── plots.py                matplotlib figures — bending, shear, deflection, torsion, stress
    │   └── convergence_studies/
    │       ├── convergence_library.py  ConvergenceResultsLibrary
    │       └── convergence_study.py    run_convergence — see the open item under Status
    └── bearings/load_distribution/no_lubrication/
        ├── results_library.py          BearingResultBundle, BearingResultsLibrary
        └── single_row/rolling_bearing_study.py    RollingBearingSolver
```

The tree mirrors the domain twice — once under `construction/`, once under `studies/` — rather than being organised by domain with a stage subfolder inside each. The stage is the stronger separation: everything under `construction/` can run without a solver present, and that property is worth being able to see from the path alone.

`no_lubrication/` in the bearing study path is not a placeholder. It states the modelling assumption the study makes, so a later lubricated study is a sibling directory rather than a flag on this one.

---

## Capabilities

The capability layer lets a script declare **what it needs to do** and receive exactly the objects that job requires — instead of forty hand-written import lines whose dotted paths must be kept in step with the package tree.

It works by reading what each package already publishes in its `__init__.py`: the list of names it offers, and where each one lives. It reads only.

### The two stages

```
Construction  →  Studies
```

| Stage | Class | Does | Needs |
|---|---|---|---|
| 1 | `ConstructionCapabilities` | Instantiates the objects — shaft, bearings, bearing families, gears, meshing, the shaft and gear system containers. Nothing is solved. | Nothing |
| 2 | `StudyCapabilities` | Resolves the study entry points: `shaft_fem` (the rigid-support FEM solve — `shaft_fem.timoshenko_rigid` or `shaft_fem.euler_bernoulli_rigid` — read back into a `RigidSupportFEMResultsLibrary`) and `bearing_iso16281` (internal load distribution from those results). | The `ConstructionCapabilities` that built the system |

`StudyCapabilities` holds the `ConstructionCapabilities` as a field, so a study cannot be declared without the construction it is a study *of*. Its `has_capability()` walks both, which is how a study checks that the system it is about to solve was built with the capabilities that solve requires.

| `ConstructionCapabilities` member | Purpose |
|---|---|
| `validate` / `validate_or_raise` | Checks every requested capability string. Raises `CapabilityError`, naming the offending capability. |
| `validate_chain` | Walks the prerequisite chain, so a missing prerequisite is named before a single import happens. |
| `has_capability` | Whether a given capability was requested. |
| `resolve` | Imports exactly what was selected and returns it by name, raising on a collision across domains rather than shadowing. |
| `summary` | What was selected and what it pulled in. |

`StudyCapabilities` follows the same shape.

### One deliberate exception

`ConstructionCapabilities` resolves Construction-domain capabilities only — nothing it resolves is solved — **with one exception**: `"systems.parallel_axis_linear"` also calls `SpurHelicalGearSystem.resolve()` before returning. A system's whole purpose is to be resolved and positioned, so shipping it unresolved was judged an incomplete deliverable. Every other Construction capability stops at Construction. The FEM solve lives in `StudyCapabilities`.

This is documented here rather than hidden because it is the one place the stage boundary is crossed on purpose.

### `capabilities/catalogue.py`

Pure metadata: the full listing of capability strings with their descriptions, shared by both stage classes and importing nothing from `axisforge`. It is safe to introspect without pulling in any `core` or `solvers` code.

| Function | Purpose |
|---|---|
| `list_capabilities()` | Every selectable capability string. |
| `describe(capability)` | What one capability provides. |
| `chapter_of(capability)` | Which stage and domain it belongs to. |
| `requirements_of(capability)` | Its prerequisites, as capability strings. |
| `verify(capability, context)` | Whether a capability's prerequisites are satisfied in a given context. |
| `print_catalogue()` | An ASCII table of the whole catalogue. |

### Usage

```python
from axisforge.fixtures.construction.construction_capabilities import ConstructionCapabilities
from axisforge.fixtures.studies.study_capabilities            import StudyCapabilities

construction = ConstructionCapabilities(
    shaft  = ("shafts.stepped_3section",),
    system = ("systems.parallel_axis_linear",),
)
construction.validate_or_raise()
objs                  = construction.resolve()
ShaftFixture          = objs["ShaftFixture"]
make_stepped_3section = objs["factory"]

study = StudyCapabilities(
    construction     = construction,
    shaft_fem        = ("shaft_fem.timoshenko_rigid",),
    bearing_iso16281 = ("bearing_loads.single_row",),
)
study.validate_or_raise()
solve_system = study.resolve()["solve_system"]
library      = solve_system(system, construction)
```

Nothing else is imported: no roller post-processing, no planetary geometry, no report writer.

**Not** a plugin registry, not a dependency-injection container, and not a stability layer. It returns classes and functions, and a name that disappears from a package's surface fails loudly.

---

## Construction fixtures

Every fixture is a frozen dataclass that keeps the parameters that produced it, so a modified copy can be rebuilt without parsing the core object back apart. Modification always produces a new object through a `with_*` method.

### Shafts

`construction/shafts/shaft_fixture.py`.

**`SectionSpec`** — declarative descriptor for one section: length, diameter, material, surface roughness, label, and optional shoulders. It uses the core `Shoulder` directly; there is no intermediate spec layer.

**`ShaftFixture`** — immutable wrapper for a shaft, keeping the section specs.

| Member | Purpose |
|---|---|
| `total_length`, `n_sections` | Aggregate properties. |
| `with_sections` | A copy from a different section list, keeping the name. |
| `with_name` | A copy under a different name. |
| `validate`, `validate_or_raise` | Delegates to the core `Shaft`. |
| `summary` | Axial position and diameter per section, followed by a separate block listing every transition carrying a `Shoulder` (fillet radius, diameter range, r/d, D/d). Deliberately stops there: engineering properties and validation status belong to the reports that consume a `ShaftFixture`, not to its own summary. |

Shoulders are read from `shaft.transitions`, not from individual sections — `SectionSpec` carries no shoulder fields.

Surface roughness applies uniformly across sections; for section-specific treatment, build the spec list by hand. Keyways are not part of the spec — attach them to the core section after building.

### Bearings

`construction/bearings/`, one module per contact type and duty. All four call `Bearing.assemble()` with a family, so a bearing arrives already validated against the analyses it was requested for.

| Module | Families covered |
|---|---|
| `ball_radial_fixture.py` | `DeepGrooveBallFamily`, `AngularContactFamily`, `SelfAligningBallFamily` |
| `ball_thrust_fixture.py` | `SingleRowThrustBallFamily`, `MultiRowThrustBallFamily` |
| `roller_radial_fixture.py` | `CylindricalRollerFamily` |
| `roller_thrust_fixture.py` | `ThrustCylindricalRollerFamily`, `MultiRowThrustCylindricalRollerFamily`, `ThrustNeedleRollerFamily` |

Assembly raises immediately if the family does not support a requested analysis or if a required geometry field is missing, so the failure surfaces at the fixture rather than hundreds of lines into the solve.

```python
brg = Bearing.assemble(
    family=DeepGrooveBallFamily(),
    catalog=BearingCatalog(d=50, D=90, b=20, C=37_100, C0=23_200,
                           designation="6210", position=35.0, label="A"),
    geometry=dict(Dw=12.7, Dpw=70.0, Z=10, E=206_000, s=0.012),
    analyses={"point_contact": True},
)
```

The core bearing is immutable for a stronger reason than the other fixtures: it refuses every write once assembled. A fixture cannot patch an assembled bearing — it builds a new one.

### Gears

`construction/gears/parallel_axis/fixed/`, four modules: a single external gear, a single internal gear, an external (spur/helical) pair, and an internal pair.

The profile shift declared on a gear fixture is **provisional**. The working coefficients are resolved by the meshing object when the pair is built, and read back from the pair.

Tip and root relief and the finer roughness parameters are not exposed — a caller needing them instantiates the core gear directly.

### Systems

`construction/systems/parallel_axis/spur_helical/linear_chain_fixture.py`. Assembly only: it receives **already-built** domain objects — `Shaft`, `Bearing`, `SpurHelicalGear`, `Load` — creates the shaft containers, places gears and bearings, builds the mesh links and resolves the power flow. It constructs none of them itself.

| Type | Purpose |
|---|---|
| `ShaftSpec` | One shaft in the chain: the built shaft, its bearings, and that shaft's `ShaftSystem` keyword arguments. |
| `StageSpec` | One stage: the gear pair, the driver and driven axial positions, the line-of-centres angle, and a label. |

**Scope.** Linear chains only: *n* stages give *n*+1 shafts and *n* links. Every `StageSpec.torque_split` stays `None` — fan-out is simultaneous-driver machinery that a linear chain never exercises, and computing a torque split belongs to a dedicated pre-solver. Internal-gear stages are likewise deferred. Non-linear topologies are modelled as several independent systems; planetary arrangements use the planetary meshing class directly.

---

## Studies

### Shaft FEM

`studies/shafts/fem_studies/fem_simple.py` (moved from `studies/shafts/fem_simple.py`).

```python
solve_system(system, construction, library=None, *,
             theory, shear_theory, integration_method,
             distribute_gear_labels=None, extra_mandatory=None,
             kGA_override=None) -> RigidSupportFEMResultsLibrary
```

`theory`/`shear_theory`/`integration_method` are assembled into one `BeamModelSettings` internally and applied uniformly to every shaft — kept as three separate parameters (rather than one `BeamModelSettings` argument) specifically so `study_capabilities.py` can pin `theory` via `functools.partial` while a caller still supplies `shear_theory`/`integration_method` per call. There are no defaults, matching `BeamModelSettings.__post_init__`'s own requirement that `theory="euler_bernoulli"` pairs with `shear_theory=None, integration_method=None` and `theory="timoshenko"` requires both non-`None`.

Solves every `ShaftSystem` in `system.shafts`, **one fresh `RigidSupportFEMSolver` per shaft** — a solver's published attributes describe the last thing it solved, so reusing one across shafts would silently discard results. Each solve is read back through `ShaftResultsReader` (which also runs `TorsionSolver` — see [`solvers/README.md`](../solvers/README.md#torsion)) and stored in the library under the shaft's name via `library.store(result)`.

Passing an existing `library` adds to it rather than replacing it, so a study can be built up shaft by shaft.

### Mesh convergence

`studies/shafts/convergence_studies/convergence_study.py` — `run_convergence(system, construction, library=None, *, theory="timoshenko", regions=frozenset({"gears", "external_distributed"}), distribute_gear_labels=None, gci_threshold=0.01, safety_factor=1.25, max_levels=8) -> ConvergenceResultsLibrary`. Same "one fresh solver + one fresh `MeshConvergenceStudy` per shaft" rule as `solve_system()`, and the same `construction.has_capability("systems.parallel_axis_linear")` guard.

Bearing regions are excluded from the default `regions` set on purpose: studying convergence against the rigid-support bearing model, which is about to be replaced by the real ISO/TS 16281 internal load distribution, would measure convergence of a placeholder rather than of the physics that will actually ship. Bearing-region convergence stays reachable only by calling `MeshConvergenceStudy.intervals_from_shaft_system()` directly with `regions={"bearings", ...}`, never through this capability layer, until a bearing-aware convergence path exists.

**See the open item under [Status](#status):** this module currently imports the pre-rename `RigidBearingFEMSolver` name from `rigid_support.py` and calls it with the pre-`BeamModelSettings` constructor signature, neither of which matches that module's current content.

### Bearing internal load distribution

`studies/bearings/load_distribution/no_lubrication/single_row/rolling_bearing_study.py`.

**`RollingBearingSolver`** — the single entry point for a shaft's full bearing set, which may mix contact types and row counts.

| Member | Purpose |
|---|---|
| `solve` | Groups single-row bearings by resolved solver and dispatches each group; sends multi-row bearings individually to the multi-row solver registered on their row solver. Returns per-label results, in the order the caller supplied the bearings. |
| `postprocess_and_record` | The full pipeline — solve (or reuse an existing load distribution), then capacity, dynamic equivalent load and secant stiffness — recorded into a `BearingResultsLibrary`. |

`postprocess_and_record` takes a `catalog` mapping each bearing label to the arguments for its capacity and equivalent-load steps. Either sub-entry may be omitted to skip that step; stiffness is always computed. Capacity is obtained by calling the family through the bearing, so this orchestrator holds no ISO formulas of its own.

---

## Result libraries

Registries live here rather than in `solvers/`. Each holds results computed elsewhere and computes nothing itself. Each result type has exactly one slot per key; setters overwrite rather than accumulate history.

### `RigidSupportFEMResultsLibrary`

`studies/shafts/fem_studies/results_library.py` (renamed from `RigidBearingFEMResultsLibrary` in `studies/shafts/results_library.py`). Registry of `ShaftResults` keyed by `ShaftSystem.name` — its `store()` method reads the key straight off `results.name` and raises if it's empty.

This library stores **only** the results of the initial rigid-support FEM solve (plus the separately-run torsion pass — see [`solvers/README.md`](../solvers/README.md#torsion)). It is the canonical source of shaft internal forces, deflections, bearing reactions and section stresses, and every downstream consumer reads from it.

```
RigidSupportFEMSolver + TorsionSolver
      │
ShaftResultsReader.read()      → ShaftResults
      │
RigidSupportFEMResultsLibrary  ← stored here
      │
      ├── ISO16281BallSolver / ISO16281RollerSolver
      ├── ShaftPostProcessor
      └── (fatigue, static failure — planned)
```

Constraints: the key (`results.name`) is non-empty; re-storing under the same name overwrites, so a fresh solve replaces a stale one; no solver logic, no interface imports, no database calls.

### `ConvergenceResultsLibrary`

`studies/shafts/convergence_studies/convergence_library.py` — **new since the last pass**. Per-shaft registry of `MeshRefinementResult`, the same role `RigidSupportFEMResultsLibrary` plays for `ShaftResults`, keyed by `MeshRefinementResult.shaft_name`. `store()` overwrites an existing entry for the same shaft name, same "fresh solve replaces stale" convention. Populated by `run_convergence()` (see [Studies](#studies)) — one call per shaft in `system.shafts`.

`MeshRefinementResult.print_report()` was removed on the result's move into `results/`, and its replacement (an `outputs/` report block for this library, mirroring `comparison_report.py`'s own shape) has not been built yet — `run_convergence()` only solves and stores, it does not print.

### `BearingResultsLibrary`

`studies/bearings/load_distribution/no_lubrication/results_library.py`. Registry of `BearingResultBundle`, keyed by bearing label — the cross-type registry that lubrication, fatigue and life solvers will read from.

**`BearingResultBundle`** — every computed ISO/TS 16281 result for one bearing:

| Field | Contents |
|---|---|
| `label` | Bearing label. |
| `bearing_type` | Informational only; nothing dispatches on it. |
| `load_distribution` | Length 1 (single-row) or *i* (multi-row, index-aligned with `bearing.rows`). |
| `stiffness` | Always single, even for a multi-row bearing — the shared ring displacement. |
| `capacity` | `(Q_ci, Q_ce)`, always single, never row-indexed. |
| `dynamic_equivalent_load` | Same indexing as `load_distribution`. |
| `extra` | An open slot for analyses this package does not define. |

| Member | Purpose |
|---|---|
| `add_load_distribution_results` | Records a whole per-solver set at once, rather than label by label. |
| `load_distribution_results` | Retrieves that set back for one solver class. |
| `set_load_distribution` | Sets one label directly. |
| `set_capacity`, `set_dynamic_equivalent_load`, `set_stiffness`, `set_bearing_type`, `set_extra` | Per-label setters; each overwrites. |
| `get`, `labels` | Retrieval and enumeration; `get` raises for an unrecorded label. |

---

## Report writers

Fixed-width ASCII text reports, written to a file with an explicit UTF-8 encoding. Every writer is a pure consumer of already-computed objects: it reads and formats, it never solves and never mutates what it was given.

| Writer | Reports on |
|---|---|
| `construction/outputs/text_report.write_construction_report(system, path, title)` | The built, unsolved system — composes the shaft, bearing, gear and system report blocks |
| `studies/outputs/text_report.write_studies_report(system, path, title, shaft_fem_library=None, comparison=None, convergence_library=None)` | The solved state — **renamed from `studies/text_report.write_resolution_report`**, and now composes up to three optional sections in one call: SHAFT_FEM (per-shaft, via `shaft_fem_library`), COMPARISON (two libraries side by side, via `comparison`), and MESH CONVERGENCE (via `convergence_library`). Any combination, or none — an empty call still writes a valid report. |

Per-domain blocks live beside the fixtures they describe: `construction/shafts/outputs/shaft_report.py`, `construction/bearings/outputs/bearing_report.py`, `construction/gears/parallel_axis/fixed/outputs/gear_report.py`, `construction/systems/.../outputs/system_report.py`; on the studies side, `studies/shafts/fem_studies/outputs/{resolution_report,comparison_report}.py` and the still-unbuilt convergence report block noted under [Result libraries](#result-libraries).

### The resolution report's block structure

`studies/shafts/fem_studies/outputs/resolution_report.py` (moved from `studies/text_report.py`) exposes each block separately, so a caller can compose a partial report:

| Function | Contents |
|---|---|
| `summary_block` | Governing values and their locations: `M_max`, `v_max`, `sigma_b_max`, `tau_max`, and now `phi_max` (twist angle). |
| `bending_shear_table` | `x`, `M_xz`, `M_xy`, `M` [N·mm]; `V_xz`, `V_xy`, `V` [N]. |
| `deflection_torsion_table` | `x`, `v_xz`, `v_xy`, `v` [mm]; `T` [N·m]. `u` (axial displacement) is now available on `ShaftResults` alongside `theta_xz`/`theta_xy` but is not yet in this table — see [`results/README.md`](../results/README.md) for the full field list. |
| `section_stress_table` | `x`, `d` [mm]; `W`, `Wt` [mm³]; `sigma_b`, `tau` [MPa]. |
| `bearing_reactions_table` | From the reaction-vector-derived arrays, labelled by index-aligned lookup into `bearing_nodes`. |
| `bearing_node_kinematics_table` | Position, `u`, `v_xz`, `v_xy`, `theta_xz`, `theta_xy`, `psi_xz`, `psi_xy`. |
| `bearing_node_loads_table` | Position, `Fr_xz`, `Fr_xy`, `Fr`, `Fa`, `M_xz`, `M_xy` — from the **total** force vector. |
| `shaft_result_block` | Every block above, in order, for one shaft. |

A sibling `comparison_report.py` (`shaft_comparison_block`, `write_comparison_report`) compares two already-solved `RigidSupportFEMResultsLibrary` instances for the same `system` — e.g. a Timoshenko run against a Euler-Bernoulli run — one "SHAFT: `<name>`" section per shaft, with `label_a`/`label_b` column headers the caller supplies (no default, since the module makes no assumption about what the two sides are). A `plots.py` module in the same `outputs/` package produces the matching matplotlib figures (bending, shear, deflection with resultant + phase, torsion, stress); it deliberately does not include per-node geometry figures (`d(x)`, `W(x)`, `Wt(x)`) — those stayed in the text report only, on explicit feedback that they weren't of interest as standalone figures.

Three formatting decisions are deliberate and should be preserved by any new writer:

- **Per-node arrays are split into three tables, not one wide one.** Sixteen columns do not read in a fixed-width viewer.
- **Units follow the result container exactly, including the discontinuity.** `M_xz`/`M_xy`/`M` are N·mm and `T` is N·m; the column headers say so rather than normalising one to the other, so a reader cross-checking against the container's own docstring sees the same units in both places.
- **Bearing reactions and bearing node loads are two tables, not one.** They are computed from different arrays — the reaction vector and the total nodal force vector — and merging them would imply they are the same measurement read twice.

---

## Script structure

```python
# 1 — Declare and resolve capabilities
construction = ConstructionCapabilities(...)
construction.validate_or_raise()
globals().update(construction.resolve())

study = StudyCapabilities(construction=construction, shaft_fem=(...), bearing_iso16281=(...))
study.validate_or_raise()
globals().update(study.resolve())

# 2 — Parameters
shaft_specs = [ShaftSpec(...), ...]
stage_specs = [StageSpec(...), ...]
bearings    = [[Bearing.assemble(...), Bearing.assemble(...)], ...]

# 3 — Build and resolve the system
system = build_linear_system(shaft_specs, stage_specs, P_W, rpm_in)

# 4 — External loads
system.shafts[0].add_load(TorqueLoad(...))      # motor coupling
system.shafts[-1].add_load(RadialLoad(...))     # driven machine
write_construction_report(system, "construction.txt", "Drivetrain")

# 5 — Solve and record
library   = solve_system(system, construction, theory="timoshenko",
                          shear_theory="cowper", integration_method="exact")
bearing_results = RollingBearingSolver().postprocess_and_record(
    system, bearings, library, catalog={...})

# 6 — Report
write_studies_report(system, "resolution.txt", "Drivetrain", shaft_fem_library=library)
```

Adding a load after building is safe: `set_gear_loads()` replaces only loads tagged `"gear_mesh"` and leaves user loads untouched, so re-resolving is idempotent.

---

## Design contracts

**Immutability.** Every fixture object is a frozen dataclass; modification produces a new object through a `with_*` method.

```python
shaft2 = shaft.with_sections([...])
gear2  = gear.with_x(0.15)
```

**Named reference instances are never mutated.** Use a `with_*` method for a modified copy.

**Do not modify a fixture for a specific case.** Duplicate and rename it.

**Fixtures expose, they do not hide.** Every intermediate object produced during assembly is reachable from the result without reconstruction, and downstream solvers consume those objects directly.

**Construction solves nothing.** The single exception — `systems.parallel_axis_linear` calling `resolve()` — is documented above and is not a precedent.

**A writer never solves.** Report modules read objects handed to them; they never build one.

**Console and report output is ASCII only.**

**`capabilities/` reads what packages declare and is never imported by one.**

---

## Extending this package

**Adding a construction fixture.** Place it under `construction/<domain>/`, take already-built domain objects rather than raw geometry keyword arguments where the object is not this fixture's own to build, make it a frozen dataclass with `with_*` copies, and register its capability string in `capabilities/catalogue.py`.

**Adding a study.** Place it under `studies/<domain>/` in a directory that names its modelling assumption, as `no_lubrication/` does. Take the `ConstructionCapabilities` alongside the system so the study can check the system was built for the solve it is about to run. Return a library, not a bare result.

**Adding a report block.** Put it beside the fixture or study whose output it formats. Own your own rule constants and your own `write()` call rather than reaching for a shared IO helper — the existing writers deliberately do not share one, so a change to one report's layout cannot disturb another's.

**Adding a stage.** A third stage — mesh loads, element analysis — lands as its own capabilities class in its own directory, taking the previous stage as a field, exactly as `StudyCapabilities` takes `ConstructionCapabilities`.