# fem_stepped_distributed_gear

**Validation case — shaft FEM (AxisForge `RigidSupportFEMSolver`), distributed loads.**
Formerly `case_distributed_gear_parabolic`.

## Objective

Show that the AxisForge 1D beam FEM reproduces the exact solution of a stepped
shaft on two pin supports loaded only by distributed loads, and that the
Timoshenko solution converges to it under mesh refinement.

Stepped shaft (five sections, four filleted shoulders, L = 200 mm, S355),
6204 + NU204; gear mesh forces spread uniformly over the face width
(`distribute_loads=True`) and a parabolic user load (2000 N over 40 mm):
`shaft1` on [40, 80] mm, `shaft2` on [130, 170] mm. Full model in `construction.py`.

| | |
|---|---|
| AxisForge | `secondary`, commit `c90c74b` (last full run: 2026-10-06) |
| Reference | `references/beams/stepped_beam.py` — piecewise exact integration, EB and Timoshenko |
| Criteria | Notebook 02: GCI v ≤ 1 % (Fs 1.25), M ≤ 2 % (Fs 3.0) — **provisional**, to be confirmed |
| Result | _to record_ |

Notebook 02 is **verification** (GCI, order p, inherited bias of the submodel)
as well as validation (convergence to the exact solution).

## Prerequisites in AxisForge

- Corrected `assembly/load_assembly/distributed_loads.py` (Timoshenko θ slots
  zero; Jacobian of the overlap). Without it the Timoshenko solution does not
  converge to the exact one under these loads.
- Corrected `RichardsonGCI` (`f_h0` from the fine grade; `p <= 0` rejected).
- `RefinementFloorReached` (not present in `main` on 2026-10-06; present in `secondary`).

## Files

| File | Role |
|---|---|
| `construction.py` | All inputs (top of file) + `segments()` + `parabolic_intensity()` + `build_system()` (loads applied, **not** solved) + `uniform_nodes()` / `solve()` (thin solver call) |
| `01_shear_effect_euler_bernoulli_vs_timoshenko.ipynb` | Geometry and loads; minimal mesh: EB vs Timoshenko (Cowper, Hutchinson), checked against the exact solution |
| `02_verification_timoshenko_vs_analytical.ipynb` | Timoshenko vs exact, equilibrium check, GCI convergence, GCI vs true error, inherited bias of the submodel, converged-mesh solve |
| `*.py` next to each `.ipynb` | jupytext pair (`py:percent`) |

Shared, outside this folder: `references/beams/`, `validation/_support/`,
`style/axisforge.mplstyle`.

## Running

As in `../fem_uniform_radial_load/README.md`: from this folder, clean kernel.

## Open points

- Shoulders are kept clear of the bearing faces because
  `ShaftSystem._shoulder_coincidence_errors` reports an overlap for a shoulder
  that abuts a bearing face exactly (AxisForge — to confirm whether intended).
