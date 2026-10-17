# validation/_support

Helpers shared by the validation cases — and only by them. They **read**
AxisForge objects and results; they never build or solve a system.

| Module | What |
|---|---|
| `style.py` | Locates `style/axisforge.mplstyle` (`STYLE_PATH`); `use_style()` |
| `figures/shaft_studies.py` | Report figures of the shaft FEM cases (field comparison, convergence, shaft layout, fields along a shaft) |
| `figures/contact_plots.py` | Contact cases: error against Hertz along a sweep, pressure cuts |
| `figures/bearing_plots.py` | Bearing cases: A2 sign check, element loads and contact ellipses around the bearing (`A2_NOTE`) |
| `reports/construction/` | Text blocks describing a constructed system (shaft, bearings, gears, topology, loads) |
| `reports/solver_results/` | Text blocks for FEM results, two-solve comparison, mesh convergence |
| `convergence.py` | `ResultantView` — resultant fields over a `SubmodelResult` for `MeshConvergenceStudy`; `sweep`, `RECORD_COLUMNS` — mesh and domain sweep of a smooth SlipPY contact |

Not imported by `studies/` or `examples/`: a study that needs one of these
copies it, so that a change here can never silently change a study.
