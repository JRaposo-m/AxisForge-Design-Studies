# validation/_support

Helpers shared by the validation cases — and only by them. They **read**
AxisForge objects and results; they never build or solve a system.

| Module | What |
|---|---|
| `style.py` | Locates `style/axisforge.mplstyle`; `use_style()`, `INK`, `MUTED` |
| `figures/shaft_studies.py` | Report figures of the shaft FEM cases (field comparison, convergence, shaft layout) |
| `figures/fem_plots.py` | Generic FEM result plots |
| `reports/construction/` | Text blocks describing a constructed system (shaft, bearings, gears, topology, loads) |
| `reports/solver_results/` | Text blocks for FEM results, two-solve comparison, mesh convergence |
| `convergence.py` | `ResultantView` — resultant fields over a `SubmodelResult` for `MeshConvergenceStudy` |

Not imported by `studies/` or `examples/`: a study that needs one of these
copies it, so that a change here can never silently change a study.
