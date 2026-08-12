# AxisForge

**Modular CAE-style platform for mechanical transmission system analysis.**

AxisForge is a deterministic, solver-centric engineering platform for the analysis of shaft–bearing–gear systems. It is built around physical first principles and traceable standards — every quantity is explainable, every result linked to an equation and a reference. Solvers are fully independent of any GUI layer, expose all intermediate quantities for inspection, and are validated against textbook and standard reference cases.

---

## Table of Contents

- [Overview](#overview)
- [Stack](#stack)
- [Repository Layout](#repository-layout)
- [Analysis Pipeline](#analysis-pipeline)
- [Module Reference](#module-reference)
  - [core/machine_elements — Shaft](#coremachine_elements--shaft)
  - [core/machine_elements — Bearings](#coremachine_elements--bearings)
  - [core/machine_elements — Gears](#coremachine_elements--gears)
  - [core/mechanical_system — Systems](#coremechanical_system--systems)
  - [core/mechanical_system — Gear Meshing](#coremechanical_system--gear-meshing)
  - [core — Loads](#core--loads)
  - [core — Materials](#core--materials)
  - [mesh — 1D Shaft Mesh](#mesh--1d-shaft-mesh)
  - [solvers — Shaft FEM](#solvers--shaft-fem)
  - [solvers — Bearings (ISO/TS 16281)](#solvers--bearings-isots-16281)
  - [solvers — Gears](#solvers--gears)
  - [models — Result Containers](#models--result-containers)
  - [ui — Interactive Runner](#ui--interactive-runner)
- [Design Principles](#design-principles)
- [Roadmap](#roadmap)
- [References](#references)

---

## Overview

The platform targets the complete analysis pipeline of a multi-shaft parallel-axis transmission:

- Static load distribution across multi-shaft gear trains
- 1D FEM shaft deflection and internal force recovery (two-plane)
- Internal rolling element load distribution in ball bearings
- Bearing rating life inputs (per-element capacities and equivalent loads)
- Spur / helical / internal gear geometry and mesh force integration
- Planetary (epicyclic) train kinematics and ideal torque distribution
- Fatigue analysis and failure susceptibility assessment *(future phases)*

---

## Stack

- **Python 3.11+**
- NumPy · SciPy · matplotlib
- PySide6 *(GUI — future)*
- SQLite *(data persistence — future)*
- pytest

Units are SI-consistent internally: **mm** for lengths, **N** for forces, **N·mm** for moments (torque propagates in **N·m** through the gear system and is converted at the boundary), **MPa** for stresses, **degrees** for input/output angles (**radians** internally).

---

## Repository Layout

```
axisforge/
├── core/
│   ├── loads.py
│   ├── materials.py
│   └── machine_elements/
│       ├── Shaft/
│       │   └── shaft.py                    # Shoulder, ShaftSection, Shaft
│       ├── Bearings/
│       │   ├── __init__.py                 # package exports
│       │   ├── bearing.py                  # Bearing (base)
│       │   ├── bearing_types.py            # BearingType
│       │   ├── bearing_factory.py          # make_bearing()
│       │   ├── geometry/
│       │   │   ├── __init__.py
│       │   │   ├── base.py                 # BearingGeometry (ABC)
│       │   │   ├── ball_geometry.py        # BallBearingGeometry
│       │   │   └── roller_geometry.py      # RollerBearingGeometry (Phase 2)
│       │   └── subtypes/
│       │       ├── __init__.py
│       │       ├── deep_groove_ball.py     # DeepGrooveBallBearing
│       │       ├── angular_contact_ball.py # AngularContactBallBearing (Phase 2)
│       │       ├── cylindrical_roller.py   # CylindricalRollerBearing  (Phase 2)
│       │       ├── tapered_roller.py       # TaperedRollerBearing      (Phase 2)
│       │       └── spherical_roller.py     # SphericalRollerBearing    (Phase 2)
│       └── Gears/
│           └── Parallel_Axis_gears/
│               ├── spur_helical_gear.py    # SpurHelicalGear
│               └── internal_gear.py        # InternalGear
│   └── mechanical_system/
│       └── Parallel_Axis_systems/
│           ├── systems/spur_helicoidal_system/
│           │   ├── shaft_system.py             # GearElement, ShaftSystem
│           │   └── SpurHelical_gear_system.py  # SpurHelicalMeshLink, SpurHelicalGearSystem
│           ├── gear_meshing/
│           │   ├── spur_helical_gear_meshing.py    # SpurHelicalGearMeshing
│           │   ├── internal_gear_meshing.py        # InternalGearMeshing
│           │   └── planetary_gear_meshing.py       # PlanetaryGearMeshing, PlanetaryKinematics
│           └── schematic.py                # assembly visualisation
├── mesh/
│   └── oneD/shaft/
│       ├── mesh_generation/
│       │   ├── mesh_1D.py                  # Mesh1D
│       │   ├── mesh_grade.py               # Grader
│       │   └── mesh_convergence_study.py   # RichardsonGCI, MeshConvergenceStudy
│       └── Elements/
│           ├── elem.py                     # Elem
│           └── Timoshenko_Selective_Integration/
│               └── timoshenko.py           # TimoshenkoBeam
├── solvers/
│   └── machine_elements/
│       ├── shaft/oneD_analysis/
│       │   ├── build_stiffness_matrix.py       # StiffnessMatrixBuilder
│       │   ├── FEM_solvers/
│       │   │   ├── simple_fem_solver.py        # SimpleFEMSolver
│       │   │   └── submodel_solver.py          # SubmodelSolver, SubmodelResult
│       │   └── static/
│       │       └── static_analysis.py          # BearingNodeData, ShaftResults,
│       │                                       #   SimpleFEMResultsLibrary, ShaftResultsReader
│       ├── bearings/
│       │   └── ISO_16281_ball_bearing.py       # IterativeBearingFEMSolver + capacity classes
│       └── gears/
│           ├── geometry.py                     # GearSolver
│           └── utils.py                        # geometry helpers
├── models/
│   ├── gear_result.py                      # GearGeometryResult, GearForceResult
│   └── stress_result.py                    # CriticalSection, StressResult
├── ui/
│   └── runner.py                           # interactive section runner (PySide6)
└── tests/
    └── ...                                 # unit / regression / validation tests
```

> **Note on legacy modules:** an earlier flat layout (`core/shaft.py`, `core/components.py`, `core/system.py`) coexists with the current `core/machine_elements/` package. New development targets the `machine_elements` structure; the flat modules are retained for backward compatibility during migration.

---

## Analysis Pipeline

The canonical solve sequence for one shaft:

```python
# 1. Assemble the multi-shaft gear system and resolve power flow
gearbox = SpurHelicalGearSystem(shafts, links, label="drivetrain")
gearbox.resolve(P_W, rpm_in, rotation_dir_source=1)

# 2. Mesh the shaft (mandatory nodes + optional grading at gears)
fem = SimpleFEMSolver()
fem.solve(shaft_system, extra_mandatory=gear_grade_nodes)

# 3. Post-process FEM into a results library
ShaftResultsReader(fem, shaft_system).read(library)

# 4. Solve internal bearing load distribution (ISO/TS 16281)
coupled = IterativeBearingFEMSolver()
load_dist = coupled.solve(shaft_system, bearings, library)

# 5. Compute per-element capacities and equivalent loads
cap   = RollingElementCapacity.radial(bearing, Cr=C_rating)
derel = DynamicEquivalentRollingElementLoad.from_distribution(bearing, result)
```

---

## Module Reference

### core/machine_elements — Shaft

**`shaft.py`**

| Class | Purpose |
|-------|---------|
| `Shoulder` | Fillet transition geometry between two adjacent sections (step change). |
| `ShaftSection` | Single uniform cylindrical segment with cross-section properties. |
| `Shaft` | Ordered sequence of `ShaftSection` objects forming a complete shaft. |

**`Shoulder`** — dataclass carrying `fillet_radius`, `diameter_large`, `diameter_small`. Validates that the fillet is positive and does not exceed the step height.
- `r_over_d` — r/d ratio (primary Peterson interpolation).
- `D_over_d` — D/d ratio (secondary Peterson interpolation).
- `validate()` — returns a list of geometry errors.

**`ShaftSection`** — dataclass with `length`, `diameter`, `inner_diameter`, `material_id`, `surface_finish_ra`, optional `shoulder_left`/`shoulder_right`, and keyways.
- `area` — cross-sectional area A [mm²].
- `second_moment_of_area` — I = π/64·(d⁴−dᵢ⁴) [mm⁴].
- `polar_moment` — J = π/32·(d⁴−dᵢ⁴) [mm⁴].
- `section_modulus` — W = I/(d/2) [mm³], for σ_b = M/W.
- `polar_section_modulus` — Wt = J/(d/2) [mm³], for τ = T/Wt.
- `validate()` — geometry errors, delegating to shoulders and keyways.

**`Shaft`** — datum x=0 is the left face of the first section; interior boundaries belong to the right section.
- `add_section(section)` — append a section to the right end.
- `total_length`, `n_sections` — aggregate properties.
- `axial_start(index)`, `axial_end(index)` — absolute face positions [mm].
- `section_at(x)` — returns `(ShaftSection, index)` containing position x.
- `diameter_at(x)`, `I_at(x)`, `J_at(x)`, `W_at(x)`, `Wt_at(x)` — section properties at any axial position.
- `shoulders()` — list of `(x_position, Shoulder)` for all steps.
- `validate()` / `validate_or_raise()` — full geometry consistency check, including adjacent shoulder–diameter matching.

---

### core/machine_elements — Bearings

The bearing package is structured in three layers: a catalogue base class, type-specific geometry classes, and concrete subtypes — one file per concern.

```
Bearings/
├── bearing.py          ← Bearing base: catalogue data, ISO 281, mounting
├── bearing_types.py    ← BearingType enum
├── bearing_factory.py  ← make_bearing() — type-dispatch factory
├── geometry/
│   ├── base.py         ← BearingGeometry ABC
│   ├── ball_geometry.py    ← BallBearingGeometry  (point contact, ISO/TS 16281)
│   └── roller_geometry.py  ← RollerBearingGeometry (line contact, Phase 2)
└── subtypes/
    ├── deep_groove_ball.py     ← DeepGrooveBallBearing
    └── ...                     ← other subtypes (Phase 2)
```

**`bearing_types.py`**

| Class | Purpose |
|-------|---------|
| `BearingType` | Enum of rolling bearing families. Determines life exponent p in ISO 281 (p=3 ball, p=10/3 roller). |

**`bearing.py` — `Bearing`**

Catalogue-level base class. Carries ISO 281 rating data, mounting arrangement, and geometry attribute slots (initialised to `None`; populated by subclass `setup_internal_geometry`). Solvers always access geometry via `bearing.Dw`, `bearing.ri`, `bearing.cp`, etc. — the interface is uniform regardless of subtype.

- Constructor: bore `d`, outer `D`, width `b`, ratings `C`/`C0`, factors `X`/`Y`, `arrangement` (`"locating"` / `"floating"` / `"non-locating"`), `contact_angle_deg`, `label`, `position`.
- `is_locating()` — True if the bearing restrains axial displacement.
- `equivalent_dynamic_load(Fr, Fa)` — P = X·Fr + Y·Fa (ISO 281).
- `has_internal_geometry()` — True if `setup_internal_geometry()` has been called.
- `validate()` / `validate_or_raise()` — catalogue-level geometry and factor checks.
- `summary()` / `__repr__()`.

**`bearing_factory.py` — `make_bearing(bearing_type, **kwargs)`**

Instantiates the correct subclass for a given `BearingType`. Useful when the type comes from a database, config file, or GUI — i.e. when the subclass is not known at write time.

```python
from axisforge.core.machine_elements.Bearings import make_bearing, BearingType

b = make_bearing(BearingType.DEEP_GROOVE_BALL, d=20, D=47, C=12700, ...)
```

**`geometry/base.py` — `BearingGeometry` (ABC)**

Abstract contract for internal geometry classes. Defines `setup(**kwargs)`, `hertz_spring_constant()`, and `load_deflection_exponent` (3/2 for ball, 10/9 for roller).

**`geometry/ball_geometry.py` — `BallBearingGeometry`**

Point contact geometry for ball bearings. ISO/TS 16281 eq.(2)–(11).

- `setup(ri, re, Dw, Dpw, Z, E, nu, *, s=None, alpha_0_deg=None)` — exactly one clearance input required. Computes A = ri+re−Dw, then either `alpha_0 = arccos(1 − s/2A)` or `s = 2A·(1 − cos α₀)`, then Ri and φ_j.
- `curvature_sum_inner()` / `curvature_sum_outer()` — Σρ, eq.(5)/(6).
- `curvature_diff_inner()` / `curvature_diff_outer()` — F(ρ), eq.(7)/(8).
- `hertz_spring_constant()` — c_p [N/mm^(3/2)], eq.(9)–(11) via elliptic integrals and `brentq`.

**`geometry/roller_geometry.py` — `RollerBearingGeometry`**

Line contact geometry placeholder. Phase 1 provides a Palmgren approximation for c_l; full ISO/TS 16281 line contact (crowning, tilt) is deferred to Phase 2.

**`subtypes/deep_groove_ball.py` — `DeepGrooveBallBearing`**

Concrete DGBB subtype. Delegates all internal geometry to `BallBearingGeometry` and mirrors computed attributes onto `self` for uniform solver access.

- `setup_internal_geometry(ri, re, Dw, Dpw, Z, E, nu=0.3, **kwargs)` — `**kwargs` passes the clearance specification (`s` or `alpha_0_deg`) straight through to `BallBearingGeometry.setup()`. Populates `self.ri`, `self.re`, `self.Dw`, `self.Dpw`, `self.Z`, `self.s`, `self.E`, `self.nu`, `self.A`, `self.alpha_0`, `self.Ri`, `self.phi_j`.
- `compute_hertz_point_contact()` — returns and caches `self.cp` [N/mm^(3/2)].
- `validate()` — extends base validation with geometry guards (ri > Dw/2, re > Dw/2, s ≥ 0).

```python
from axisforge.core.machine_elements.Bearings import DeepGrooveBallBearing

b = DeepGrooveBallBearing(d=20, D=47, b=14, C=12700, C0=6550,
                           designation="6204", position=20.0,
                           arrangement="locating", label="brg1a")

# via diametral clearance
b.setup_internal_geometry(ri=4.13, re=4.21, Dw=7.94, Dpw=33.5,
                           Z=8, E=206000, s=0.010)

# or via free contact angle
b.setup_internal_geometry(ri=4.13, re=4.21, Dw=7.94, Dpw=33.5,
                           Z=8, E=206000, alpha_0_deg=0.5)

cp = b.compute_hertz_point_contact()
```

---

### core/machine_elements — Gears

**`spur_helical_gear.py`**

**`SpurHelicalGear`** — geometry model for an external spur (β=0) or helical (β>0) gear. Carries module `mn`, teeth `z`, profile shift `x`, pressure angle `alpha_n`, helix `beta`, face width `b`, surface finish, and axial `position`. Exposes reference/base/tip geometry and the attribute interface consumed by the meshing classes.

**`internal_gear.py`**

**`InternalGear`** — ring (internal) gear model following the conventional KHK positive-z definition. Provides mesh-pair interference checks (`validate_mesh`) used by `InternalGearMeshing`.

---

### core/mechanical_system — Systems

**`shaft_system.py`**

**`GearElement`** — thin wrapper binding a gear geometry object to a kinematic role on a shaft; carries no geometry of its own.
- Constructor: `gear`, `role` ("driver"/"driven"), `rotation_dir` (±1, set only for the source gear), `label`.
- `position` — delegates to the underlying gear.
- `validate()` / `validate_or_raise()`.

**`ShaftSystem`** — autonomous single-shaft container of bearings, gears and loads. Has no knowledge of other shafts; its `shaft_position` (global y,z offset) is normally set by the gear system during resolve.
- `add_bearing(b)`, `add_gear(g)`, `add_load(ld)` — placement with axial-bounds checking (chainable).
- `set_gear_loads(loads)` — idempotent replacement of all `gear_mesh`-sourced loads (safe to re-resolve).
- Sorted accessors: `bearings`, `gears`, `loads`, `support_positions`.
- Type-filtered accessors: `radial_loads`, `axial_loads`, `torque_loads`, `external_moments`, `distributed_radial_loads`.
- `gear_extent(ge)`, `bearing_extent(b)` — axial [lo, hi] footprint (position is the centre of face/bearing width).
- `validate()` / `validate_or_raise()`.

**`SpurHelical_gear_system.py`**

**`SpurHelicalMeshLink`** — one directed mesh, shaft_a (driver) → shaft_b (driven), carrying a meshing model and the global line-of-centres angle `phi_deg`. Optional `torque_split` selects fan-out mode; `distribute_loads` toggles distributed vs point gear loads.
- `validate()` — self-mesh, torque_split range, and meshing-interface checks.

**`SpurHelicalGearSystem`** — single-source DAG of shafts connected by mesh links. Enforces exactly one source, no convergent merges, acyclicity, and fan-out torque_split consistency.
- `resolve(P, rpm, rotation_dir_source, source_position)` — propagates power from the source through the DAG in topological order, computing torque, rotation sense and shaft placement, then injects the resulting mesh loads onto every shaft. Torque propagates in N·m.
- `validate()` / `validate_or_raise()` — topology plus per-shaft validation.
- Internal graph helpers: incoming-count, source detection, Kahn topological order, driver grouping, axial-alignment checks.

---

### core/mechanical_system — Gear Meshing

**`spur_helical_gear_meshing.py`**

**`SpurHelicalGearMeshing`** — external spur/helical pair. Computes working centre distance and transverse working pressure angle, contact ratios (εα, εβ, εγ), interference checks, and mesh forces.
- Constructor accepts optional working centre distance `al`, profile-shift equalisation (`equalise_gs`), and addendum reduction.
- `forces(T_in, phi_deg, rotation_dir_in)` — returns Ft, Fr, Fa and per-side load angles, plus output torque and rotation sense.
- `al`, `u` — working centre distance and gear ratio, consumed by the gear system.
- `correction_for_gs_equilibrium()` — Henriot method for profile-shift balancing.

**`internal_gear_meshing.py`**

**`InternalGearMeshing`** — external + internal (ring) pair, using the difference form (z2−z1, x2−x1) throughout. Computes working geometry, path-of-contact points, Ohlendorf loss factor, contact ratios and mesh forces.
- `forces(T_in, phi_deg, rotation_dir_in)` — same interface as the external pair; note internal meshing does **not** reverse rotation sense.
- `gear_geometry(addendum_reduction)` — builds independent working-geometry copies of both gears.
- `validate()` / `validate_or_raise()`, `summary()`.

**`planetary_gear_meshing.py`**

**`PlanetaryGearMeshing`** — single-stage epicyclic train (sun + k planets + ring + carrier) by composition of one external pair (sun–planet) and one internal pair (planet–ring). Owns train-specific logic only.
- `planet_phi_deg(j)`, `planet_position(j)` — angular placement of planet j.
- Structural conditions (coaxiality, assembly, neighbouring), Willis kinematics (F=1 and F=2 modes), and ideal torque distribution over members and planets.

**`PlanetaryKinematics`** — frozen result of the Willis kinematic solve (ω per member in rad/s, operating-mode DoF, and Willis residual for numerical transparency).

**`PlanetaryMember`** / **`MeshTag`** — enums tagging train members (sun/planet/ring/carrier) and tooth contacts.

---

### core — Loads

**`loads.py`**

| Class | Purpose |
|-------|---------|
| `LoadPlane` | Enum of principal bending planes (XY, XZ) used as a decomposition key. |
| `Load` | Base class carrying `position`, `label`, `source` ("user"/"gear_mesh"/"bearing_reaction"). |
| `RadialLoad` | Transverse point force at angular position θ, decomposed into Fy/Fz. |
| `AxialLoad` | Force along the shaft axis (+X tensile). |
| `TorqueLoad` | Torque about the shaft axis [N·m]. |
| `ExternalMoment` | Applied bending moment at orientation θ, decomposed into My/Mz. |
| `DistributedRadialLoad` | Transverse load distributed over [x_lo, x_hi] with uniform or callable intensity/direction. |
| `LoadingProfile` | Fatigue cycle decomposition (stress ratio R → mean/amplitude factors). |

`RadialLoad`/`ExternalMoment` expose `Fy`/`Fz` (or `My`/`Mz`) and `component(plane)`. `DistributedRadialLoad` provides `resultant_component`, `centroid`, `bending_moment_contribution` (closed-form for uniform loads, quadrature otherwise), and `as_point_load` for the constant-θ case.

---

### core — Materials

**`materials.py`**

| Class | Purpose |
|-------|---------|
| `Material` | Shaft/structural material (Shigley-based): Sut, Sy, E, density, Poisson ratio, endurance limit. |
| `GearMaterial` | Gear material (ISO 6336-5): E, ν, ρ, thermal properties, σHlim, σFlim, quality class. |

`Material` exposes `endurance_limit` (Shigley §6-2: 0.5·Sut capped at 700 MPa) and `shear_yield_strength` (0.577·Sy). `GearMaterial` exposes `equivalent_modulus(other)` for the Hertzian reduced modulus of a pair.

Embedded shaft library: `S355`, `CrMo42` (42CrMo4), `AISI_1045`, `AISI_4340`. Embedded gear library: `GEAR_STEEL`, `GEAR_ADI`, `GEAR_POM`, `GEAR_PA66`.

Lookups: `get_material(id)`, `available_materials()`.

---

### mesh — 1D Shaft Mesh

**`mesh_1D.py` — `Mesh1D`**
Generates the 1D FEM node grid for one `ShaftSystem` from mandatory positions (section boundaries, bearings, gears, loads) plus optional `extra_mandatory` nodes, enforcing a minimum node spacing.

**`mesh_grade.py` — `Grader`**
Produces standardised mesh grades for a subdomain [x_lo, x_hi] by successive elementwise bisection (`grade_0` = base nodes, `grade_N` = N bisections). Consumed by the convergence study and by gear-face refinement.
- `get_grade("grade_N")` — sorted node positions at the requested refinement level.

**`Elements/elem.py` — `Elem`**
Single 1D beam element between two mesh nodes (length, E, I, A, Poisson ν, node indices).
- `from_mesh(mesh)` — builds the full element list from a `Mesh1D`, reading section properties and materials.
- `from_x_nodes(x_nodes, shaft_system)` — builds elements from an explicit node list (used by the submodel solver).
- `find_node_index(x_nodes, x)` — locate a node within tolerance.
- `validate()` — element sanity (positive length, plausible modulus units).

**`Elements/Timoshenko_Selective_Integration/timoshenko.py` — `TimoshenkoBeam`**
Timoshenko beam element with selective integration (shear factor 5/6).
- `stiffness_element(elem)` — 6×6 element stiffness (axial + shear + bending).
- `shape_functions`, `deformation_matrix`, `elasticity_matrix` — element interpolation and constitutive matrices.
- Natural-coordinate mapping helpers for distributed-load integration.

**`mesh_convergence_study.py` — `RichardsonGCI`, `MeshConvergenceStudy`**
Grid Convergence Index via Richardson extrapolation on the resultant transverse displacement, across ≥3 refinement levels. Produces per-load convergence records and the union of extra nodes to lock into production runs. Includes `print_report`.

---

### solvers — Shaft FEM

**`build_stiffness_matrix.py` — `StiffnessMatrixBuilder`**
Assembles the global stiffness matrix from element contributions. Currently wires the Timoshenko beam theory; extensible via the `_BEAM_THEORIES` registry.
- `build_stiffness_matrix(mesh, elements)` — global K (3 DOF/node: u, v, θ).

**`FEM_solvers/simple_fem_solver.py` — `SimpleFEMSolver`**
Orchestrates the full pipeline: `Mesh1D` → `Elem.from_mesh` → `StiffnessMatrixBuilder` → boundary conditions → two independent planar solves (XZ, XY) sharing the same K → superposition → torsion diagram on the same nodes. Every intermediate quantity is stored as a public attribute (numerical transparency).
- `solve(shaft_system, extra_mandatory)` — runs the full solve; stores `x_nodes`, `elements`, `free_dofs`, displacement vectors `d_total_xz`/`d_total_xy`, external force vectors, torsion arrays `T_total`/`tau_total`, and reactions.
- `return_values(x_nodes, [x_lo, x_hi])` — nodal solution quantities within an interval (for submodelling).
- Options: `theory`, `constraint_bearing`, `distribute_gear_labels` (which gear mesh loads are treated as distributed over face width).

**`FEM_solvers/submodel_solver.py` — `SubmodelSolver`, `SubmodelResult`**
Wraps `SimpleFEMSolver` and restricts metric evaluation to a subdomain [x_lo, x_hi], with Lagrange-multiplier boundary injection at the cut nodes. Used exclusively by the convergence study. `SubmodelResult` carries the subdomain displacement vectors and cut-node reaction multipliers.

**`static/static_analysis.py`**

| Class | Purpose |
|-------|---------|
| `BearingNodeData` | Complete FEM nodal state at a bearing position (displacements, reactions, seat misalignment ψ). |
| `ShaftResults` | Full FEM solution + post-processed engineering quantities for one shaft. |
| `SimpleFEMResultsLibrary` | Registry of `ShaftResults` keyed by shaft name — the canonical source all downstream solvers read from. |
| `ShaftResultsReader` | Post-processes a solved `SimpleFEMSolver` into a `ShaftResults` and stores it in the library. |

`BearingNodeData` carries `Fr_xz`, `Fr_xy`, `Fr`, `Fa`, moment reactions, displacements, and the seat-slope misalignment `psi_xz`/`psi_xy` (gradient of v across the seat, or nodal θ for zero-width seats). `ShaftResults` is organised into mesh, raw FEM solution, post-processed engineering quantities (internal forces, deflections, section stresses σ_b, τ), and per-bearing node data. The library provides `store`/`get`/`get_or_none`/`remove`/`clear`/`names`/`iter`/`all_results`. `ShaftResultsReader.read(library)` recovers internal forces, deflections, section properties, bearing reactions and node data in one pass.

---

### solvers — Bearings (ISO/TS 16281)

**`bearings/ISO_16281_ball_bearing.py`**

**`IterativeBearingFEMSolver`** — coupled shaft–bearing solver using the prescribed-ψ formulation. Per bearing it performs a single 2-DOF root solve (δr, δa) in the plane of the resultant radial force, with misalignment prescribed from the FEM seat slope projected onto that plane.
- `solve(shaft_system, bearings, library, Pd)` — returns `{label: LoadDistributionResult}` for all bearings, reading FEM data from the library via `BearingNodeData`.
- `minimum_axial_load(bearing, ...)` — smallest axial preload Fa_min such that δa ≥ 0, via `brentq`.
- Static post-processing helpers: `Q_j`, `phi_j_global`, `contact_distribution` (per-element φ and Q, global or local frame), `bearing_stiffness` (secant Kr_xz, Kr_xy, Ka).
- Solver core uses `scipy.optimize.root` (`hybr` with `lm` fallback).

**`LoadDistributionResult`** — output for one bearing: ring displacements δr/δa, prescribed misalignment ψ, resultant-force angle φ(Fr), per-element deflection δ_j and contact angle α_j, moment reaction Mz, solver diagnostics (iterations, residual, success flag).

**`BearingStiffnessState`** — secant stiffness decomposed onto the global XZ/XY axes plus axial, with an axial engagement regime (`no_load` / `engaged` / `closing_clearance`). Built via `from_load_distribution`.

**`RollingElementCapacity`** *(frozen dataclass)* — per-element dynamic capacity Q_ci / Q_ce, ISO/TS 16281 §4.3.1. Cr/Ca are supplied externally.
- `radial(bearing, Cr)` — radial ball bearings, eq.(19)/(20).
- `thrust_nonzero_alpha(bearing, Ca)` — thrust ball bearings α≠90°, eq.(21)/(22).
- `thrust_90deg(bearing, Ca)` — thrust ball bearings α=90°, eq.(23)/(24).

**`DynamicEquivalentRollingElementLoad`** *(frozen dataclass)* — dynamic equivalent rolling element loads Q_ei / Q_ee, ISO/TS 16281 §4.3.2, eq.(25)–(28), with inner/outer rotating convention. Built via `from_distribution`.

Two module-level helpers, `_geometry_bracket` and `_check_geometry`, encapsulate the shared §4.3.1 geometry factor and its validity guards.

---

### solvers — Gears

**`gears/geometry.py` — `GearSolver`**
Cylindrical gear geometry and mesh force calculation (MAAG / ISO 21771 / Shigley §13-7). Pure functions, no internal state.
- `compute_geometry(mn, z1, z2, alpha_n_deg, beta_deg, al, x1, x2, b)` — full pair geometry with optional profile shift; working centre distance from `al` or the involute equation (`brentq`); tip/root/working diameters; contact ratios. Returns `GearGeometryResult`.
- `compute_forces(T1_Nm, geometry)` — Ft, Fr, Fa from input torque. Returns `GearForceResult`.
- `to_gear_element(position, forces, geometry)` — assembles a `GearElement` for injection into the pipeline.

**`gears/utils.py`** — geometry helpers: rack constants (HAP, HFP), `solve_alpha_tw` (involute equation), `contact_ratio_alpha`/`contact_ratio_beta`, `undercut_z_min`, `validate_geometry_inputs`.

---

### models — Result Containers

**`gear_result.py`**
- `GearGeometryResult` *(frozen)* — module, teeth, angles, reference/base/tip/root/working diameters, centre distances, contact ratios (εα, εβ, εγ).
- `GearForceResult` *(frozen)* — Ft, Fr, Fa and derived torque quantities.

**`stress_result.py`**
- `CriticalSection` — per-section fatigue result (Goodman, ASME-elliptic, yield safety factors).
- `StressResult` — collection with `most_critical`, `min_nf_goodman`, `min_nf_asme`, `min_ny`.

---

### ui — Interactive Runner

**`runner.py`** — a PySide6-based section runner that executes user analysis scripts split into titled sections. Seeds a namespace with the core classes and solvers, captures stdout/stderr and validation errors per section, tracks names written, and renders matplotlib figures on demand. Includes `SectionParser`, `Section`, `SectionResult`.

---

## Design Principles

- **Low coupling, high cohesion** — solvers depend only on explicit result containers, never on each other's internals.
- **No hidden state** — every intermediate quantity is a public attribute; solvers expose their full working for inspection.
- **GUI-independent solvers** — the analysis core runs headless; the UI is a thin consumer.
- **Deterministic and explainable** — no black-box methods. Failure assessment (future) uses measurable physical drivers and traceable indices, not machine learning or probabilistic life prediction.
- **Composition over inheritance** — e.g. the planetary train composes two pair-meshing objects rather than subclassing them; bearing geometry is a component of the bearing subtype, not its identity.
- **Fail fast on geometry** — validation happens at construction; invalid geometry is never silently accepted.
- **One file per concern** — each bearing subtype, geometry class, and solver lives in its own module; imports are explicit and traceable.

---

## Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Shaft FEM · ISO/TS 16281 bearing load distribution · gear force integration | **Active** |
| 2 | Angular contact, cylindrical/tapered/spherical roller bearing subtypes · ISO 6336 gear strength | Planned |
| 3 | Fatigue analysis — Goodman / Morrow / Miner | Planned |
| 4 | Lubrication assessment — EHD film, grease | Planned |
| 5 | Failure susceptibility scoring · SQLite data layer | Planned |
| 6 | PySide6 GUI | Planned |

---

## References

- ISO/TS 16281:2008 — *Rolling bearings: Methods for calculating the modified reference rating life for universally loaded bearings*
- ISO 281:2007 — *Rolling bearings: Dynamic load ratings and rating life*
- ISO 21771:2007 — *Gears: Cylindrical involute gears and gear pairs*
- ISO 6336 — *Calculation of load capacity of spur and helical gears*
- ISO 6336-5 — *Strength and quality of materials*
- Harris & Kotzalas, *Rolling Bearing Analysis*, 5th ed., Wiley
- Palmgren, *Grundlagen der Wälzlagertechnik*, 3rd ed., Franckh
- Shigley, *Mechanical Engineering Design*, 10th ed.
- MAAG Gear Book, 2nd ed.
- Arnaudov & Karaivanov, *Planetary Gear Trains*
