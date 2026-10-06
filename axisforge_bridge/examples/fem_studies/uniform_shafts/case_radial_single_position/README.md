# case_radial_single_position — provisional

Uniform shaft (L = 200 mm, d = 20 mm), one radial point load at two
positions, solved with Euler–Bernoulli and Timoshenko. Replaces the
previous `euler_bernoulli/`, `timoshenko/`, `analytical/` and Abaqus
folders of this case.

| File | Role |
|---|---|
| `case_definition.py` | **All inputs** (top of file) + `build_system()` + `solve(system, settings, n_elements)` |
| `analytical_solution.py` | Closed-form Macaulay reference (EB and Timoshenko) |
| `01_shear_effect_euler_bernoulli_vs_timoshenko.ipynb` | Study on the minimal mesh: EB vs Timoshenko (Cowper, Hutchinson), checked against the closed form |
| `02_verification_timoshenko_vs_analytical.ipynb` | Verification: Timoshenko vs analytical, GCI convergence (`v_res:max`, `M_res:max_minus_mean`) with the AxisForge `MeshConvergenceStudy`, GCI vs true error, converged-mesh solve |
| `figures.py` | Pure figure functions (data → `Figure`) — provisional, moves with `outputs/` |
| `axisforge.mplstyle` | The only place where figure appearance is defined — provisional location |
| `*.py` next to each `.ipynb` | jupytext pair (`py:percent`) — version this one; diffs stay readable |

## Running

The notebooks import the modules in this folder by name, so start Jupyter
**from this folder** (or open the notebook in VS Code, which uses the
notebook's folder as working directory). `axisforge` must be importable
(`pip install -e` of the AxisForge repo), and notebook 02 also needs
`axisforge_bridge` (it prints the GCI tables with
`outputs/solver_results/convergence_report.py`): install the design-studies
repo (`pip install -e .` at its root) or start Jupyter with
`set PYTHONPATH=<AxisForge-Design-Studies root>`. No `sys.path` manipulation
in the code and no `common/`.

Run the notebook top to bottom with a clean kernel (*Restart & Run All*);
a notebook only counts if it runs that way. Export with
`jupyter nbconvert --to html <notebook>.ipynb`.

Doctests: `pytest --doctest-modules case_definition.py analytical_solution.py`.

## Provisional points

- `figures.py` and `axisforge.mplstyle` belong with the presentation layer
  (`outputs/`, which is moving to AxisForge), not inside a case folder.
- The GCI thresholds and safety factors in notebook 02 §3.1 (v: 1 %, Fs 1.25; M: 2 %, Fs 3.0) are carried over from the previous refined-mesh script and are to be confirmed.
- Outputs are not written to disk; the notebook (or its HTML export) is the report.