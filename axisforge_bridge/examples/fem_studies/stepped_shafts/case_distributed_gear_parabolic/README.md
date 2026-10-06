# case_distributed_gear_parabolic — provisional

Stepped shaft (five sections, four filleted shoulders, L = 200 mm) on two
pin supports, loaded only by distributed loads: gear mesh forces spread
uniformly over the face width and a parabolic user load (2000 N over 40 mm).
Solved with Euler–Bernoulli and Timoshenko.

Suggested location: `validation/fem_studies/resolution/stepped_shafts/distributed_load/`.

| File | Role |
|---|---|
| `case_definition.py` | **All inputs** (top of file) + `segments()` + `build_system()` + `solve(system, settings, n_elements)` |
| `analytical_solution.py` | Exact EB/Timoshenko solution for a stepped two-pin beam under point and polynomial distributed loads (piecewise exact integration) |
| `01_shear_effect_euler_bernoulli_vs_timoshenko.ipynb` | Study on the minimal mesh: geometry and loads, EB vs Timoshenko (Cowper, Hutchinson), checked against the exact solution |
| `02_verification_timoshenko_vs_analytical.ipynb` | Verification: Timoshenko vs exact, equilibrium check, GCI convergence (`v_res:max`, `M_res:max_minus_mean`) with the AxisForge `MeshConvergenceStudy`, GCI vs true error, inherited bias of the submodel, converged-mesh solve |
| `figures.py` | Pure figure functions (data → `Figure`), incl. `shaft_layout` — provisional, moves with `outputs/` |
| `axisforge.mplstyle` | Figure appearance — provisional location |
| `*.py` next to each `.ipynb` | jupytext pair (`py:percent`) — version this one |

## Prerequisites

- AxisForge with the corrected `assembly/load_assembly/distributed_loads.py`
  (Timoshenko theta slots zero; Jacobian of the overlap). Without it the
  Timoshenko solution does not converge to the exact one under these loads.
- AxisForge with the corrected `RichardsonGCI` (`f_h0` from the fine grade;
  `p <= 0` rejected).
- Notebook 02 also needs `axisforge_bridge` importable (prints the GCI tables
  with `outputs/solver_results/convergence_report.py`): `pip install -e .` at the
  design-studies root, or start Jupyter with
  `set PYTHONPATH=<AxisForge-Design-Studies root>`.

Start Jupyter **from this folder**; run with a clean kernel (*Restart & Run All*).
Doctests: `pytest --doctest-modules case_definition.py analytical_solution.py`.

## Provisional points

- `figures.py` and `axisforge.mplstyle` are duplicated from the point-load case;
  they should move to a shared module before a third case is added.
- GCI thresholds and safety factors (v: 1 %, Fs 1.25; M: 2 %, Fs 3.0) to be confirmed.
- Shoulders are kept clear of the bearing faces because
  `ShaftSystem._shoulder_coincidence_errors` reports an overlap for a shoulder
  that abuts a bearing face exactly.
