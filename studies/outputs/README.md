# studies/outputs

Plot and report models of the studies (D-0036 amendment items 3 and 10): no case draws or formats by
itself; it builds records and calls a function listed here. Tables are printed and written with
`axisforge.outputs._format` (`format_table`, `write_records_csv`) directly; a report model is added here
only when a case needs more than that.

```text
outputs/
  common/style/      paper.mplstyle (house paper style, ported from figures/paper-style 19cc689), PAPER
  common/figures.py  figure_path, save, MARKERS: file naming and saving shared by every plot module
  common/plots/      convergence.py (Newton residual history)
  l000_axisforge/    plots/shaft.py, plots/bearing_loads.py
  l100_dry_contact/  plots/contact.py
  tests/             the index test below
```

- Every figure is drawn inside `matplotlib.pyplot.style.context(PAPER)` and written as
  `<case>_<file>_<function>[_<variant>].png` into the calling case's `results/` (git-ignored, D-0020).
- Semantics of the style: numerical result = filled markers; formula / theory = lines (own solid,
  others dashed); published value = open markers. Colour never carries identity alone. No titles.
- A test (`tests/test_index.py`) fails when a public function of a model module is missing from the
  index below.

## Index
| function | file | what it shows | record fields | called by |
|---|---|---|---|---|
| `deflection_along_shaft` | `l000_axisforge/plots/shaft.py` | deflection v [mm] and rotation theta [mrad] along x [mm] of one bending plane: FEM nodes (filled markers) on the reference curves (lines; a second reference of a series dotted), supports, loads and section steps marked | `series`, `source`, `x_mm`, `v_mm`, `theta_rad` | v001, v002, s006, v005, s004, s003 |
| `error_vs_elements` | `l000_axisforge/plots/shaft.py` | `abs(rel_err)` [-] vs number of elements (log-log), tolerance line with the pass band shaded, optional `n^-p` guides | `n_elements`, `series`, `rel_err` | v001, v002, v005 |
| `ratio_sweep` | `l000_axisforge/plots/shaft.py` | a dimensionless ratio along a sweep: FEM (filled), published (open), closed form (line), limit line; residual panel when FEM rows carry `rel_err`; opt. legend outside the axes | `<x>`, `y`, `series`, `source`, opt. `rel_err` | v001, v002, s006, v005, s004, s003 |
| `element_loads` | `l000_axisforge/plots/bearing_loads.py` | rolling-element load Q [N] vs azimuth phi [deg] (Cartesian -180..180 or polar): computed elements (filled), published (open), judge's continuous load zone and zero-clearance cos^n shape (lines); load directions dash-dotted; frame and R-003 tag in the caption | `series`, `source`, `phi_deg`, `Q_N` | v003, s001, s002, s102, s003 |
| `load_zone_sweep` | `l000_axisforge/plots/bearing_loads.py` | load-zone quantities (eps, Stribeck `Z Q_max / F_r`) vs a swept parameter (clearance), one panel per quantity: computed (filled) on the judge's curves (lines), limit values dotted; opt. vertical marks (validity limits), hatched spans (kept points outside a validity range), open-marker sources (another case's point) and filled-marker sources (a BEM result) | `<x>`, `quantity`, `y`, `series`, `source` | v003, s001, s002, v105, s104, s103, s102 |
| `lamina_map` | `l000_axisforge/plots/bearing_loads.py` | roller lamina loads q_jk [N] in colour, element azimuth phi [deg] by lamina x_k [mm], one panel per value of a swept key (tilt) | `<panel>`, `phi_deg`, `x_mm`, `q_N` | s002 |
| `newton_history` | `common/plots/convergence.py` | scaled residual norm [-] (log) vs Newton iteration, one line per solve, stopping tolerance dashed; opt. axis labels for another iteration (s004: fixed-point max abs(delta psi) [rad] vs pass) | `label`, `k`, `residual_norm` (`NewtonResult.records()`) | s102, v005, s004, s003 |
| `relative_errors` | `l100_dry_contact/plots/contact.py` | `abs(rel_err)` [%] of BEM quantities vs a swept parameter, one panel per series, tolerances dashed, optional slope guide | `<x>`, `quantity`, `series`, `rel_err`, opt. `tol`, `verdict`; opt. caption | v101, v102, v103, v104, v003, v105, v002, v005, s101, s103, s003 |
| `pressure_cut` | `l100_dry_contact/plots/contact.py` | BEM `p/p0_H` along cuts through the centre vs the Hertz profile `sqrt(1 - xi^2)` | `axis`, `xi`, `p_over_p0_ref` | v101, v102, v103, s104 |
| `load_approach` | `l100_dry_contact/plots/contact.py` | load Q [N] vs approach delta [um], log-log: BEM points (filled) on each contact's Hertz law (line), dotted slope guide (1.5) | `series`, `source`, `delta_m`, `Q_N` | s101 |
