# hertz_vs_bem

**Study: SlipPY static BEM against Hertz, through `slippy_bridge.contact` (D-0019).**

## Question

Does the dry, smooth, static contact that the bridge builds and SlipPY solves reproduce Hertz
(maximum pressure, semi-axes, approach) for the three contacts of a rolling bearing, and how much
grid and how much domain does it take?

| Script | Contact | Hertz type |
|---|---|---|
| `01_ball_flat.py` | ball, 12.7 mm, on a flat, 500 N | circular point contact |
| `02_ball_raceway.py` | the same ball on the inner raceway of a deep groove ball bearing (`Dpw` 70 mm, groove 0.52 `Dw`, contact angle 0), 500 N | elliptical point contact, long across rolling |
| `03_roller_raceway.py` | roller, 10 mm, on the inner raceway of a cylindrical roller bearing (`Dpw` 70 mm), 5e4 N/m | line contact (plane strain) |

Steel on steel (E = 210 GPa, nu = 0.3). Assumptions: smooth surfaces, linear elastic half-spaces,
frictionless, static normal load, principal axes aligned. Each contact is solved in the two builds
of `slippy_bridge.contact.pair`: `composite` (equivalent body on a flat, what Hertz assumes) and
`two_profiles` (each body with its exact profile).

The reference is SlipPY's own `hertz_full` (with the local fixes), read through
`slippy_bridge.contact.hertz`. It is not an independent judge: `references/contact` is still
empty. This is why the folder is a study and not a validation case.

## Provenance

| | |
|---|---|
| design-studies | `10c3832` + uncommitted working tree (`slippy_bridge` and this study) |
| AxisForge | `b3f624a` |
| SlipPY | `df5916c` (branch `local-fixes`) |
| Run | 2026-10-10, **provisional**: to be repeated from a committed state |

## Files

| File | Role |
|---|---|
| `_common/case.py` | The study constants in SI: `CASES` (three `ContactCase`), the mesh levels and the domain levels |
| `_common/sweep.py` | `sweep` (the mesh and domain sweep, bridge calls only) and `criteria` (the criteria below) |
| `_common/plots.py` | `convergence` and `pressure_cuts`, with `style/axisforge.mplstyle` |
| `construction.py` | The AxisForge bearings (mm, N, MPa), assembled and not solved |
| `01_`, `02_`, `03_` | One per contact: sweep, CSV, figures, criteria |
| `04_compare_saved.py` | Temporary: new results against the `data/results.csv` saved before the bridge |
| `<type>/<build>/data/` | Results of the first version of the study (tracked); kept until the comparison is accepted |

Grid, surfaces, static step, measurements, Hertz reference and checks come from
`slippy_bridge.contact`; the CSV and the tables from `axisforge.outputs._format`.

**Sweeps.** The reference grid is the default of the bridge: spacing `h0` = largest `2**k` not
above `s_x / 10`, half-length `4 s_x` along rolling and `1.3 s_y` across it (16 nodes on the
periodic axis of the line contact). Mesh sweep: `2 h0`, `h0`, `h0/2`, `h0/4`. Domain sweep:
half-length `2`, `4`, `8` `s_x` at `h0`.

**AxisForge link (02 and 03).** The case of the sweep is written by hand in SI, so that its
`case_hash` is independent of AxisForge and equal to the saved one (`9ccc2ddabd085997`,
`67f4d0603a4d1dcc`, `8f21f492364adfda`). The script then assembles the bearing of
`construction.py`, asks the bridge for its inner-race contact (`contacts(bearing, load=...)`) and
checks radii, moduli, Poisson's ratios and load against the study case to 1e-12 relative. The two
are compared by value and not by hash, because the mm -> m conversion can change the last bit. That
contact is solved once on the default grid; its record is the row `sweep = axisforge` of the CSV,
with `element`, `subtype` and `race`.

## Criteria (fixed before running)

On the default grid (`h0`, half-length `4 s_x`), smooth surfaces, per case and per build:

1. the six checks of the record are true (`converged`, `load_ok`, `edge_clear`, `margin_ok`,
   `resolution_ok`, `within_reference`);
2. |err p0| <= 1 %; |err s_x|, |err s_y| <= 3 %; |err delta| <= 2 % (point contacts only);
3. the two builds agree in p0 within 1 %;
4. domain sweep: |err p0| at half-length 8 <= at 4 <= at 2;
5. mesh sweep: |err p0| at `h0/4` <= 0.5 %.

The scripts print PASS or FAIL per criterion and build; the limits are not adjusted.

## Running

```bat
python studies\bearings\contact\hertz_vs_bem\01_ball_flat.py
python studies\bearings\contact\hertz_vs_bem\02_ball_raceway.py
python studies\bearings\contact\hertz_vs_bem\03_roller_raceway.py
python studies\bearings\contact\hertz_vs_bem\04_compare_saved.py
```

`--build composite` or `--build two_profiles` runs one build. Needs `pyfftw`. The whole study
takes about two minutes; the finest `ball_raceway` grid (558 x 1660 nodes) takes 20 to 30 s.
Output goes to `results/<type>/` inside this folder, git-ignored (`results.csv`,
`convergence.png`, `pressure_cuts.png`, `slippy/`) and `compare_saved.txt` beside them.

## Result (provisional run of 2026-10-10)

Errors against Hertz on the default grid, in %:

| Case | Build | p0 | s_x | s_y | delta | p0 at `h0/4` | p0 at half-length 2 / 4 / 8 |
|---|---|---|---|---|---|---|---|
| ball_flat | composite | +0.272 | +0.422 | +0.137 | +1.260 | +0.322 | +0.458 / +0.272 / +0.224 |
| ball_flat | two_profiles | +0.272 | +0.422 | +0.137 | +1.260 | +0.322 | +0.458 / +0.272 / +0.224 |
| ball_raceway | composite | -0.361 | -1.064 | +0.772 | +0.470 | -0.380 | -2.664 / -0.361 / +0.069 |
| ball_raceway | two_profiles | -0.056 | -0.834 | -0.628 | +0.893 | -0.073 | -2.318 / -0.056 / +0.366 |
| roller_raceway | composite | +0.349 | +0.557 | - | - | +0.323 | +1.162 / +0.349 / +0.127 |
| roller_raceway | two_profiles | +0.349 | +0.559 | - | - | +0.322 | +1.162 / +0.349 / +0.127 |

| Criterion | ball_flat | ball_raceway | roller_raceway |
|---|---|---|---|
| 1 six checks | PASS, PASS | PASS, PASS | PASS, PASS |
| 2 errors on the default grid | PASS, PASS | PASS, PASS | PASS, PASS |
| 3 builds agree in p0 | PASS (0.000 %) | PASS (0.306 %) | PASS (0.000 %) |
| 4 domain | PASS, PASS | PASS, **FAIL** (two_profiles) | PASS, PASS |
| 5 finest mesh | PASS, PASS | PASS, PASS | PASS, PASS |

(composite, two_profiles.) AxisForge link: PASS for both bearings (largest relative difference
1.3e-16).

What the numbers say:

* **The error depends on the domain, not on the mesh.** Refining from `2 h0` to `h0/4` does not
  reduce the p0 error: it stays near 0.3 % in magnitude for five of the six runs, and for
  ball_raceway / two_profiles it stays between -0.01 and -0.07 %. Extending the domain along
  rolling changes it. At half-length 2 the pressure is zero on the boundary and p0 is still 0.5 to
  2.7 % off.
* **Cause: the elastic kernel, not the grid.** The bridge builds `slippy.contact.Elastic` with its
  default `use_frequency_domain=True` (`slippy/core/elastic_material.py:81`, kernel in
  `slippy/core/materials.py:156-164`). That kernel repeats the load with period twice the domain
  in each direction (image loads) and corrects only the mean value. A half-space BEM should not
  depend on the domain once the contact is inside it; the domain sweep of this study measures the
  image error. The saved results (square domains of +-6 a) used the same kernel on a larger cell
  and were biased too (ball_flat approach -0.64 %).
* **Criterion 4 fails for ball_raceway / two_profiles**: the p0 error is -2.318 %, -0.056 %,
  +0.366 % at half-length 2, 4, 8. It changes sign, so its magnitude is not monotonic although
  the value moves in one direction. Left as FAIL.
* **The approach is the quantity most sensitive to the domain** (ball_flat: -1.6 %, +1.3 %, +6.3 %
  at half-length 2, 4, 8; ball_raceway composite: +8.1 %, +0.5 %, -1.2 %) and does not converge
  in this sweep.
* **Roller half-width:** its error oscillates with h (-1.6, +0.6, -0.7, +0.02 %). The oscillation
  is sub-node (all below 0.11 h; the BEM contact edge sits on a node) and no trend should be read
  into it.
* Against the saved results (`04_compare_saved.py`): the Hertz reference is identical to the last
  digit for the three cases; the finest-mesh p0 differs by 0.2 to 0.4 %. The old square domains
  (+-6 a, +-3 a for ball_raceway) were closer to Hertz in p0 than the new +-4 s_x by 1.3 s_y in
  four of the five runs (not in ball_raceway / two_profiles).

**Reviewer check, outside the study.** The physics reviewer repeated the runs at `h0` with a
modified material, `use_frequency_domain=False` (spatial kernel). These numbers are not produced
by the scripts of this folder and are to be reproduced in the committed run:

* the results no longer depend on the domain, to all printed digits, at half-length 2, 4 and 8:
  ball_flat p0 +0.035 %, delta +0.004 %; ball_raceway composite p0 +0.007 %, delta -0.002 %;
* ball_raceway / two_profiles: p0 +0.307 %, s_y -1.36 %, delta +0.404 %, independent of domain
  and mesh. This is a real effect of the exact profiles beyond the Hertz paraboloid, and it is
  the offset of that build against composite seen above;
* the mesh sweep then converges: ball_flat p0 +0.099, +0.035, +0.021, +0.018 % from `2 h0` to
  `h0/4`;
* with the frequency-domain kernel, widening the domain across rolling does not remove the
  dependence: ball_raceway composite delta +0.47, +4.04, +8.57 % at half-width 1.3, 2, 3 `s_y`.

The results recorded above use the bridge as it is (frequency-domain kernel).

## Open

* **Kernel of the bridge material.** Whether `slippy_bridge.contact` should build the material
  with the spatial kernel is an open decision for José; nothing was changed.
* **Criterion 4', proposed, not applied:** p0 and delta change by less than 0.05 % between
  half-lengths 2, 4 and 8. It asks for independence of the domain instead of a monotonic error.
* **Judging two_profiles:** proposed, not applied: compare it with the composite build within a
  stated band, since its difference from Hertz is physical and not an error of the solution.
