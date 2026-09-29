# axisforge_bridge

Everything that sits **on top of** the AxisForge core to run an analysis and document its results: text reports and plots. It contains **no numerics** and defines **no domain objects** — those live in `axisforge` (`core/`, `mesh/`, `solvers/`, `results/`). Nothing in `axisforge` imports from the bridge.

Systems are built and solved **directly with the AxisForge core**; the bridge only formats what comes back. This README shows that workflow: build a system, solve it with the FEM solver, run a mesh-convergence study, write the outputs.

---

## Contents

- [Layout](#layout)
- [Setup](#setup)
- [1. Build a system](#1-build-a-system)
- [2. Solve with the FEM solver](#2-solve-with-the-fem-solver)
- [3. Mesh convergence](#3-mesh-convergence)
- [4. Write the outputs](#4-write-the-outputs)
- [Known caveats](#known-caveats)
- [Extending](#extending)

---

## Layout

```
axisforge_bridge/
├── outputs/
│   ├── construction/
│   │   ├── text_report.py          write_construction_report(system, path, title)
│   │   └── {shaft,bearing,gear,system}_report.py   content blocks (no I/O)
│   └── solver_results/
│       ├── fem_report.py           write_fem_report(system, results, path, title)
│       ├── comparison_report.py    write_comparison_report(results_a, results_b, system, path, ...)
│       ├── convergence_report.py   write_convergence_report(results, path, title, header_lines)
│       └── plots/fem_plots.py      write_fem_plots(system, results, base_dir)
├── studies/                        currently empty
└── validation/                     see below
```

**`validation/`** holds the runs that check AxisForge's FEM results against Abaqus and closed-form solutions. It is self-contained, and nothing in it is part of the workflow described here.

Design rules:

- **Content blocks return strings, writers write files.** Every `*_report.py` block function returns text; only the `write_*` functions open a file (UTF-8).
- **Results are read, never built, by the bridge.** `ShaftResults`, `ConvergenceRecord`, `MeshRefinementResult` are imported for typing only.
- **`_table()` is duplicated** across report modules on purpose (no shared `_io.py`).

## Setup

1. `axisforge` must be importable on its own: `pip install -e <AxisForge repo>` (or `PYTHONPATH` at its root).
2. `axisforge_bridge` needs no install, only its **parent folder** on `sys.path`:

```python
import sys
sys.path.insert(0, r"<folder that contains axisforge_bridge/>")
```

3. Python 3.12, NumPy, SciPy, matplotlib.

---

## 1. Build a system

Everything below is `axisforge` core; the bridge is not involved until the outputs.

**Material.** The core material registry (`axisforge.core.materials`) starts **empty**. Register what the shaft sections use — `ShaftSection`'s default `material_id` is `"S355"`:

```python
from axisforge.core.materials.base import (
    IsotropicElastic, Material, StrengthProperties, available_materials, register,
)

if "S355" not in available_materials():
    register(Material(
        material_id="S355", density=7850.0,
        elastic=IsotropicElastic(E=210_000.0, poisson_ratio=0.3),      # MPa
        strength=StrengthProperties(Sut=590.0, Sy=355.0),
        description="EN 10025-2 S355",
    ))
```

Missing material → `KeyError: Material 'S355' not found`.

**Shafts, bearings, gears, loads, system:**

```python
from axisforge.core.loads import RadialLoad, TorqueLoad
from axisforge.core.machine_elements.shaft.shaft import Shaft, ShaftSection
from axisforge.core.machine_elements.bearings.base import BearingCatalog
from axisforge.core.machine_elements.bearings.bearing import Bearing
from axisforge.core.machine_elements.bearings.families.family import (
    CylindricalRollerFamily, DeepGrooveBallFamily,
)
from axisforge.core.machine_elements.gears.parallel_axis.gear_properties.spur_helical_gear import SpurHelicalGear
from axisforge.core.machine_elements.gears.parallel_axis.gear_meshing.spurhelical_meshing import SpurHelicalGearMeshing
from axisforge.core.mechanical_system.parallel_axis.spur_helical.shaft_system import GearElement, ShaftSystem
from axisforge.core.mechanical_system.parallel_axis.spur_helical.gear_system import (
    SpurHelicalGearSystem, SpurHelicalMeshLink,
)

# --- shaft geometry, wrapped in a ShaftSystem (shaft + bearings + gears + loads)
shaft = Shaft(label="shaft1")
shaft.add_section(ShaftSection(length=200.0, diameter=20.0, label="shaft1"))
ss1 = ShaftSystem(shaft, name="shaft1", speed_rpm=1000.0)
# ss2 built the same way (name="shaft2", speed_rpm=500.0)

# --- bearings: Bearing.assemble(family, catalog, geometry)
ball = Bearing.assemble(
    family=DeepGrooveBallFamily(),
    catalog=BearingCatalog(d=20.0, D=42.0, b=12.0, designation="6204",
                           position=10.0, arrangement="locating", label="shaft1_ball"),
    geometry=dict(Dw=7.0, Dpw=31.0, Z=9, s=0.02),
)
roller = Bearing.assemble(
    family=CylindricalRollerFamily(),
    catalog=BearingCatalog(d=20.0, D=47.0, b=14.0, designation="NU204",
                           position=190.0, arrangement="non-locating", label="shaft1_roller"),
    geometry=dict(Dwe=6.5, Lwe=6.0, Dpw=33.5, Z=12, s=0.015, n_s=40, i=1),
)
ss1.add_bearing(ball)
ss1.add_bearing(roller)

# --- gears + mesh
g_driver = SpurHelicalGear(mn=2.0, z=20, b=15.0, position=100.0, label="g_driver")
g_driven = SpurHelicalGear(mn=2.0, z=40, b=15.0, position=100.0, label="g_driven")
meshing  = SpurHelicalGearMeshing(g_driver, g_driven, label="stage1")
ge_driver = GearElement(g_driver, role="driver", label="g_driver"); ss1.add_gear(ge_driver)
ge_driven = GearElement(g_driven, role="driven", label="g_driven"); ss2.add_gear(ge_driven)

# --- extra loads (theta from +Y towards +Z: XY plane <-> Fy = F cos(theta), XZ plane <-> Fz = F sin(theta))
ss1.add_load(RadialLoad(60.0, 500.0, theta_deg=270.0, label="radial_load"))

# --- link the shafts, validate, resolve the gear-mesh loads
link = SpurHelicalMeshLink(ss1, ge_driver, ss2, ge_driven, meshing, phi_deg=0.0, label="stage1")
system = SpurHelicalGearSystem([ss1, ss2], [link], label="my_system")
system.validate_or_raise()
system.resolve(P=10471.9755, rpm=1000.0, rotation_dir_source=1)   # adds Ft/Fr/torque loads, source="gear_mesh"

# --- the driven shaft only receives the mesh reaction torque: close it with a sink
net = sum(ld.magnitude for ld in system.shafts[-1].torque_loads)
system.shafts[-1].add_load(TorqueLoad(position=200.0, magnitude=-net, label="output_coupling", source="user"))
```

Things that bite:

| Item | Detail |
|---|---|
| Bearing family | `CylindricalRollerFamily` rejects `arrangement="locating"` (no flange to react axial load) → a roller is always `"non-locating"`. |
| Bearing width `b` | `Mesh1D` adds nodes at `position ± b/2` (`b=12` at x=10 → nodes at 4 and 16 mm). `b=0` adds none. It does not change the FEM result itself (supports are point BCs). |
| Bearing data | `C`/`C0` are not catalogue data (computed by the family from the assembled geometry); `E` only matters with `contact=ContactAnalysis.ISO16281`. |
| Supports | Every bearing constrains `v = 0` only (`u = 0` also at the locating one); **none constrains theta** → true pin supports, no moment reaction. |
| Loads | `RadialLoad.magnitude` must be ≥ 0; direction is `theta_deg`. |
| Torque equilibrium | Without a closing `TorqueLoad` on the driven shaft the torsion DOF is unconstrained. |

### Shoulders (stepped shafts)

A shaft is an ordered list of `ShaftSection`s (uniform cylinders, `z = 0` at the left face of the first one). A **shoulder** is not a section: it is a `Shoulder` attached to the *boundary* between two adjacent sections with `Shaft.set_transition(i, shoulder)`, where `i` is the boundary between `sections[i]` and `sections[i+1]`.

```python
from axisforge.core.machine_elements.shaft.shaft import Shaft, ShaftSection, Shoulder

shaft = Shaft(label="stepped")
shaft.add_section(ShaftSection(length=60.0, diameter=20.0, label="journal_left"))    # sections[0]: z = 0 ... 60
shaft.add_section(ShaftSection(length=80.0, diameter=26.0, label="hub_seat"))        # sections[1]: z = 60 ... 140
shaft.add_section(ShaftSection(length=60.0, diameter=20.0, label="journal_right"))   # sections[2]: z = 140 ... 200

# call AFTER all the sections involved have been added (the index is checked against the current section count)
shaft.set_transition(0, Shoulder(fillet_radius=1.5, diameter_large=26.0, diameter_small=20.0))   # at z = 60
shaft.set_transition(1, Shoulder(fillet_radius=1.5, diameter_large=26.0, diameter_small=20.0))   # at z = 140

shaft.validate_or_raise()
ss = ShaftSystem(shaft, name="stepped", speed_rpm=1000.0)     # then bearings, gears and loads as above
```

- **Rules** (checked by `Shoulder` and by `Shaft.validate()`): `fillet_radius > 0`, `diameter_large > diameter_small`, `fillet_radius <= (D - d)/2`, and the two diameters must be exactly those of the two adjacent sections (`Shoulder mismatch at boundary i/i+1` otherwise). A boundary without `set_transition` is a plain step with no fillet data.
- **Section fields:** `diameter`, `inner_diameter` (hollow if > 0), `material_id` (default `"S355"`; every id used must be registered), `surface_finish_ra`, `keyways`, `label`. Positions are absolute: bearing, gear and load `position` refer to the shaft's `z`.
- **In the FEM:** every section boundary is a mandatory node (`Mesh1D`), and each element takes the `A`, `I`, `J` of its section — the step appears as a jump in stiffness and in `d(x)`, `W(x)`, `Wt(x)`, hence in `sigma_b`/`tau`. The FEM does **not** apply any stress concentration: `ShaftResults.sigma_b` is nominal.
- **Stress concentration** (`Kt`, notch sensitivity `q`, `Kf = 1 + q(Kt - 1)`) is a separate post-processing step that reads `Shaft.shoulders()` (`[(z, Shoulder), ...]`) and the keyways:

```python
from axisforge.solvers.machine_elements.shaft.static_solvers.postprocessing import ShaftPostProcessor

ppr = ShaftPostProcessor(ss, results[ss.name]).process()      # PostProcessedResults
ppr.features                        # one StressConcentration per shoulder/keyway: Kt, q, Kf, r, D, d (traceable)
ppr.Kf_b, ppr.Kf_t                  # node-aligned factors (1.0 where there is no feature)
ppr.sigma_b_corrected, ppr.tau_corrected, ppr.sigma_b_corr_max
```

  `Kf` is placed on the **nearest node** of the shoulder. Caveat: the `Kt` curve fits are a simplified Shigley Table 6-2 fit (D/d ≈ 1.5 baseline, conservative), not a full Peterson interpolation in `r/d` and `D/d`; the module itself says to replace them when precision matters. This step was not exercised for this README.

A single `ShaftSystem` (shaft + bearings + loads, no gears) is enough for a stand-alone shaft; `SpurHelicalGearSystem` is only needed to get gear-mesh loads resolved.

---

## 2. Solve with the FEM solver

```python
from axisforge.mesh.shaft.beam_model_settings import BeamModelSettings
from axisforge.solvers.machine_elements.shaft.fem_solvers.global_solver.rigid_support import RigidSupportFEMSolver

# Euler-Bernoulli: no shear term -> shear_theory and integration_method MUST be None (raises otherwise)
eb = BeamModelSettings(beam_theory="euler_bernoulli", shear_theory=None, integration_method=None)
# Timoshenko
ts = BeamModelSettings(beam_theory="timoshenko", shear_theory="cowper",        # or "hutchinson"
                       integration_method="single_point")                       # or "exact" (shows shear locking)

results = {ss.name: RigidSupportFEMSolver(ts).solve(ss) for ss in system.shafts}   # {name: ShaftResults}
```

- One `RigidSupportFEMSolver` per shaft (cheap; no instance reuse).
- `solve(ss, extra_mandatory=[x, ...])` adds nodes — this is how a refined mesh is applied.
- `RigidSupportFEMSolver(settings, kGA_override=...)` replaces `κGA` for every Timoshenko element (e.g. to test against an Abaqus `*Preprint` value).
- Each `solve()` validates the system, meshes (`Mesh1D`), assembles K, applies the BCs, solves both bending planes independently and always runs the torsion solver.

`ShaftResults` (per shaft, node-aligned arrays): `x_nodes`, `u`, `v_xz`, `v_xy`, `v`, `theta_xz`, `theta_xy`, `M_xz`, `M_xy`, `M`, `V_xz`, `V_xy`, `V`, `T`, `phi`, `d`, `W`, `Wt`, `sigma_b`, `tau`; envelopes `M_max`, `v_max`, `sigma_b_max`, `tau_max`, `phi_max` (+ `x_*`); `bearing_nodes[]` (`label`, `Fr`, `Fa`, …); `bearing_positions`, `R`, `R_xz`, `R_xy`, `R_axial`.

---

## 3. Mesh convergence

There is no single `run_convergence()` call: the grade loop is written by the caller, on top of four core pieces.

| Piece | Where | Role |
|---|---|---|
| `Grader(x_lo, x_hi, x_nodes)` | `mesh/shaft/mesh_generation/mesh_grade.py` | `grade_0` = base nodes in the span; `grade_N` = elementwise bisection of `grade_{N-1}` (constant ratio r = 2) |
| `SubmodelSolver(x_lo, x_hi).solve(base_results, ss, settings, grade)` | `solvers/machine_elements/shaft/fem_solvers/submodel_solver/lagrange_multipliers.py` | Re-solves the span at that grade with the global solution imposed at the two cut nodes (Lagrange multipliers); returns a `SubmodelResult` |
| `MeshConvergenceStudy(requests, gci_threshold, safety_factor)` | `solvers/mesh/convergence_solver.py` | Richardson GCI bookkeeping over grade levels |
| `ConvergenceRecord` / `MeshRefinementResult` | `results/convergence_results/convergence_results.py` | Data: metric history, GCI history, `x_final`, `all_extra_nodes` |

```python
from axisforge.solvers.machine_elements.shaft.fem_solvers.submodel_solver.lagrange_multipliers import SubmodelSolver
from axisforge.solvers.mesh.convergence_solver import MeshConvergenceStudy, build_mesh_refinement_result

ss = system.shafts[0]
base = results[ss.name]                                   # global solve on the reference mesh
positions = sorted(b.position for b in ss.bearings)
x_lo, x_hi = positions[0], positions[-1]                  # bearing-to-bearing span

study = MeshConvergenceStudy([("v_xz", "max"), ("v_xy", "max")],
                             gci_threshold=0.01, safety_factor=1.25)
rec = study.new_record("global_v", x_lo, x_hi)
for lvl in range(8):
    grade = f"grade_{lvl}"
    sub = SubmodelSolver(x_lo, x_hi).solve(base, ss, ts, grade)      # SubmodelResult
    if study.add_level(rec, grade, sub):                  # True once converged (needs >= 3 grades)
        break
else:
    study.finalize_unconverged(rec, sub)                  # publishes the last grade's nodes; rec.converged stays False

refinement = build_mesh_refinement_result(ss.name, {rec.label: rec})
refined = RigidSupportFEMSolver(ts).solve(ss, extra_mandatory=refinement.all_extra_nodes)
```

- A **request** is `(field, eval_form)`. `field` is read with `getattr` from `SubmodelResult`: `u, v_xz, v_xy, theta_xz, theta_xy, M_xz, M_xy, V_xz, V_xy`. Eval forms: `max` (max|·|), `mean`, `max_minus_mean`, `max_minus_mean_relative`, `at_point` (needs a named point). New forms: `@register_eval_form`.
- `SubmodelResult` has **no resultant fields** (`v_res`, `M_res`): requests are per plane.
- One `(gci_threshold, safety_factor)` per `MeshConvergenceStudy`. Different thresholds per quantity → one study per group, all fed the same `SubmodelResult`.
- `RichardsonGCI` needs a **constant refinement ratio** between consecutive grades; bisection guarantees it. A non-monotonic change, a degenerate order or a metric exactly 0 falls back to `_DummyGCI` (`converged=False`; `p`/`f_h0` print as `n/a`).
- `all_extra_nodes` includes `x_final` even for records that did not converge — check `rec.converged` before trusting it.

---

## 4. Write the outputs

```python
from axisforge_bridge.outputs.construction.text_report import write_construction_report
from axisforge_bridge.outputs.solver_results.fem_report import write_fem_report
from axisforge_bridge.outputs.solver_results.plots.fem_plots import write_fem_plots
from axisforge_bridge.outputs.solver_results.comparison_report import write_comparison_report
from axisforge_bridge.outputs.solver_results.convergence_report import write_convergence_report

write_construction_report(system, out / "construction_report.txt", title="...")   # geometry, bearings, gears, loads (no solve needed)
write_fem_report(system, results, out / "report_resolution.txt", title="...")     # summary + per-node tables per shaft
write_fem_plots(system, results, out)                                             # out/plots/<shaft>/*.png
write_comparison_report(results_a, results_b, system, out / "comparison.txt",
                        label_a="timoshenko", label_b="euler_bernoulli")
write_convergence_report({ss.name: refinement}, out / "convergence.txt",
                         title="...", header_lines=["domain = bearing_to_bearing", "..."])
```

- **`write_fem_report`** — one `SHAFT_FEM` section per shaft: envelope summary, bending & shear, deflection & torsion, section & stress, bearing reactions, bearing-node kinematics and loads.
- **`write_fem_plots`** — 9 figures per shaft: `bending_moment_{components,resultant}`, `shear_{components,resultant}`, `deflection_{components,resultant}`, `torsion`, `bending_stress`, `shear_stress`. Resultant figures have a phase panel, `atan2(xz, xy)` at every node, **unmasked**: it is meaningless where the resultant is ~0 (e.g. the overhangs outside the bearings).
- **`write_comparison_report`** — two `{name: ShaftResults}` sets of the same `system` (e.g. two theories, or base vs refined mesh). Per-node tables pair nodes **by x** (nodes present in both meshes only) and state how many were paired.
- **`write_convergence_report`** — one block per shaft: per-grade metric history, GCI detail (`r`, `p`, raw deltas, `e`, GCI, `f_h0`), Richardson-extrapolated estimates and the union of extra nodes. Units come from the metric name (`v_*` mm, `M_*` N·mm, `V_*` N, `theta_*` rad).
- There is **no CSV writer** in the bridge yet.

---

## Known caveats

| # | Item | Effect |
|---|---|---|
| 1 | Nodal `M`/`V` are **element-constant** (2-node Timoshenko element): the nodal value equals the element midpoint value. | `M_max` comes out ≈ 4 % below the statics value on a 200 mm shaft with a 500 N load; can also make the `M` GCI non-monotonic. |
| 2 | `element_postprocessing.py` `TODO(owner)`: Euler-Bernoulli `shear_force()` is negated, Timoshenko's is not. | `V` can differ in sign between the two theories (≈ −200 % in `comparison_report`). |
| 3 | In the results `dM/dx = −V`, so the phase of `M` sits ~180° from the load/`V` direction. | Visible in the phase panel of `bending_moment_resultant`. |
| 4 | Core `family.py`: `if contact in (ContactAnalysis.ISO16281):` (missing comma). | `TypeError` when that path is hit; unrelated to the shaft FEM. |
| 5 | Core `postprocessing.py`: `mat.Su` falls back to 700 MPa if absent. | Silent for materials registered without `Sut`. |

## Extending

- **New report block:** add a content function to the relevant `*_report.py`, call it from the writer; keep I/O in `write_*` only.
- **New plot:** one function plus one `_FIGURE_REGISTRY` entry in `plots/fem_plots.py`.
- **New convergence metric:** add an eval form with `@register_eval_form`, or add the field to `SubmodelResult` (core change) and request `(field, form)`.