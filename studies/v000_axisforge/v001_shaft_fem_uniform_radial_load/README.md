# fem_uniform_radial_load

**Validation case — shaft FEM (AxisForge `RigidSupportFEMSolver`).**
Formerly `case_radial_single_position`.

## Objective

Show that the AxisForge 1D beam FEM (Euler–Bernoulli and Timoshenko) reproduces
the closed-form solution of a uniform shaft on two pin supports, and that the
Timoshenko solution converges to it under mesh refinement.

Uniform shaft (L = 200 mm, d = 20 mm, S355), 6204 locating + NU204 non-locating,
spur gear background load, one extra radial point load (500 N) at two positions:
`shaft1` x = 60 mm, `shaft2` x = 140 mm. Full model in `construction.py`.

| | |
|---|---|
| AxisForge | `secondary`, commit `c90c74b` (last full run: 2026-10-06) |
| Reference | `references/beams/uniform_beam.py` — Macaulay, EB and Timoshenko (Cowper / Hutchinson κ) |
| Criteria | Notebook 02 §3.1: GCI v ≤ 1 % (Fs 1.25), M ≤ 2 % (Fs 3.0) — **provisional**, carried over from the old refined-mesh script, to be confirmed |
| Result | _to record_ |

Notebook 02 is **verification** (the discrete model converges: GCI, order p)
as well as validation (it converges to the reference). Keep the two readings
separate when recording the result.

## Files

| File | Role |
|---|---|
| `construction.py` | All inputs (top of file) + `section_properties()` + `build_system()` (loads applied, **not** solved) + `uniform_nodes()` / `solve()` (thin solver call; the notebook chooses beam model and mesh) |
| `01_shear_effect_euler_bernoulli_vs_timoshenko.ipynb` | Minimal mesh: EB vs Timoshenko (Cowper, Hutchinson), checked against the closed form |
| `02_verification_timoshenko_vs_analytical.ipynb` | Timoshenko vs analytical, GCI convergence (`v_res:max`, `M_res:max_minus_mean`) with AxisForge's `MeshConvergenceStudy`, GCI vs true error, converged-mesh solve |
| `*.py` next to each `.ipynb` | jupytext pair (`py:percent`) — the one to review in diffs |

Shared, outside this folder: `references/beams/` (the judge),
`validation/_support/` (figures, convergence report, `ResultantView`),
`style/axisforge.mplstyle`.

## Running

From this folder (the notebooks import `construction` as a sibling module):
Jupyter started here, or VS Code (uses the notebook's folder), or
`python 01_….py`. Needs `pip install -e <AxisForge repo>` and `pip install -e .`
at the design-studies root (for `references` and `validation._support`).

Run top to bottom with a clean kernel (*Restart & Run All*). Outputs are not
written to disk; the notebook (or its HTML export) is the report.
