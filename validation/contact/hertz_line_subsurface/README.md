# hertz_line_subsurface

Validation case: SlipPY static BEM, through `slippy_bridge.contact`, against the Hertz line
contact (plane strain) and its subsurface stress field. V-DRY-03, V-SUB-02.

## Question

Does SlipPY's static BEM reproduce the Hertz line contact (semi-width, maximum pressure)? Does
SlipPY's `SubsurfaceStress` reproduce the stress field beneath it: `tau1` on the axis, the
orthogonal shear `tau_zx`, and the Appendix 4 tables?

## Sources

| Key | Item | PDF page |
|---|---|---|
| Johnson1985 | Sec. 4.2(c), Eq. (4.36): `R`; `delta` not defined by the local solution | 110 |
| Johnson1985 | Eqs (4.41)-(4.45): `a = (4PR/(pi E*))^(1/2)`, `p0 = 2P/(pi a)` | 111 |
| Johnson1985 | Eqs (4.46)-(4.48a), Fig. 4.5: axis stresses, `tau1` max 0.30 `p0` at 0.78`a` (Eq. 4.47) | 112 |
| Johnson1985 | Eqs (4.48b)-(4.49c): McEwen's field at any point | 113-114 |
| Johnson1985 | Table 4.1: `tau1` max 0.300 `p0` at 0.785 `b` (line limit) | 109 |
| Johnson1985 | Sec. 9.2(b): `tau_zx` max 0.25 `p0` at (+-0.87`a`, 0.50`a`) | 299 |
| Johnson1985 | Appendix 4: `sigma_x`, `sigma_z`, `tau_zx` for `p` and `q`, 8 `z/a` x 8 `x/a` | 439-440 |

The printed rows are in `reference_values.csv`, copied unchanged from the vault
(`80_Inbox/2026-10-16/reference_values/contact/hertz_line_subsurface/`).

## Conventions

- Johnson's semi-width `a` is the catalogue's `b` and the bridge's `s_x`.
- `P` is the load per unit length; `1/R = 1/R1 + 1/R2`; `E*` as for a point contact.
- `x` is the rolling direction, array axis 0. The domain is periodic along the roller axis
  (array axis 1).
- The axis stresses do not depend on nu, except `sigma_y`: in plane strain
  `sigma_y = nu(sigma_x + sigma_z)`.
- Subsurface depths: `Z_OVER_A[0]` is 1e-4 of the first depth step, SlipPY's own default
  (`slippy/core/elastic_material.py:328`), because its kernels are singular at `z = 0`. The printed
  `z/a = 0` rows of Appendix 4 are read at that depth.
- Appendix 4 prints `-(sigma_x)_p/p0`, `-(sigma_z)_p/p0` and `-/+(tau_zx)_p/p0` as magnitudes.
  The sign of `tau_zx` follows the sign of `x`, and the columns are `+-x/a`. The BEM is compared on
  `x >= 0` as `-sigma_x/p0`, `-sigma_z/p0`, `abs(tau_zx)/p0`.
- Eq. (4.46b) is printed `sigma_z = -(p0/a)(a^2 + z^2)^(-1/2)`, which is dimensionally wrong.
  Appendix 4 matches `sigma_z = -p0 a (a^2 + z^2)^(-1/2)`.
- Units SI (m, Pa, N/m) throughout the case.

## Reference

The judge for the contact is `references/contact/hertz.py::hertz_line` (Eqs (4.43)-(4.45)). The
subsurface references are Johnson's printed constants, the axis formula
`tau1/p0 = z/a - (z/a)^2 (1 + (z/a)^2)^(-1/2)` evaluated directly, and the Appendix 4 tables (three
decimals). SlipPY's own line-contact stress functions are not used. The bridge record's `err_*`
are relative to SlipPY's `hertz_full`. They are recorded and never graded.

## Case

- Roller: 10 mm, on the inner raceway (`Dpw` 70 mm, rolling radius 30 mm), steel (E = 210 GPa,
  nu = 0.3).
- Load: 5.0e4 N/m. The load is chosen, not taken from a reference: Johnson Eqs (4.43)-(4.45) and
  Appendix 4 are normalised by `a` and `p0`.
- Assumptions: smooth, frictionless, linear elastic half-spaces, plane strain, static normal load.

## Criteria

Fixed before running. The [proposed] tolerances are accepted provisionally by José, 2026-10-16.

| ID | Quantity | Criterion | Script |
|---|---|---|---|
| L1 | Half-width `a` and `p0` against `hertz_line` | within 1 % [proposed]; `delta` not compared | `01_roller_raceway` |
| L2 | `tau1` max and depth on the axis | `abs(tau1max/p0 - 0.300) <= 0.005`, `abs(z/a - 0.785) <= 0.01` [proposed]; the same against the axis formula | `02_subsurface` |
| L3 | `tau_zx` max and position | within 0.005 `p0` and 0.01 `a` of 0.25 `p0` at (0.87`a`, 0.50`a`) [proposed] | `02_subsurface` |
| L4 | Appendix 4 p-tables, 3 x 64 points | `abs(BEM - table) <= 0.001 p0` | `02_subsurface` |
| L5 | Appendix 4 q-tables | not graded in phase 1 (no tangential load) | - |

Review risk (2026-10-16), José decides before the full run:

- L2, L3 positions: the tolerance 0.01`a` is at most half a node step (depth step 0.02`a`;
  along `x` the default spacing is about 0.08`a`, so `h/(2a)` is about 0.04 > 0.01). A node-wise
  maximum can miss it from the grid alone.
- L2 profile: SlipPY's `tau1` is the 3-D `(sigma_1 - sigma_3)/2`. For `z` below about 0.4`a`,
  `sigma_y = nu(sigma_x + sigma_z)` is the largest principal stress, so near the surface the field
  ratio against the 2-D axis formula differs without any defect. The peak at 0.785`a` is not affected.
- L4: 0.001 `p0` is the print resolution of the tables. BEM discretisation and linear
  interpolation near `x = a` exceed it regardless of SlipPY. Feeding Johnson's exact field
  (Eq. (4.49)) on the default grid to the L4 code (offline test, 2026-10-16) leaves 16 of 192
  points beyond 0.001, all near `x = a` with `z <= 0.2a`; the worst is 0.085 at `x/a = 1`, `z/a = 0`.

## Scripts

Every script is a jupytext pair (`.py` and `.ipynb`, `py:percent`). It prints the criteria table and
writes `results/<script>.csv` and `results/<script>_criteria.csv` (git-ignored). `--smoke` runs the
coarsest grid only. In a notebook the full grids run.

| Script | Content |
|---|---|
| `construction.py` | The contact (`CASE`, SI), the grid levels, the depths, `roller_bearing()` (AxisForge, mm/N/MPa); nothing solved |
| `01_roller_raceway` | Mesh and domain sweep (two builds); the AxisForge bearing through the bridge, checked to be the same contact and solved; L1 |
| `02_subsurface` | `SubsurfaceStress`, built with SlipPY directly (José, Q2): principal stresses for L2, the tensor for L3-L4 |

Run from this folder (the scripts import `construction` as a sibling module):
`python 01_roller_raceway.py [--smoke]`.

## Provenance

| | |
|---|---|
| design-studies | to be filled by the run |
| AxisForge | `c328999` (`refactor/gears-mechanical-system`) |
| SlipPY | `df5916c` (`local-fixes`), unmodified |

The study `studies/bearings/contact/hertz_vs_bem` gave provisional results on 2026-10-10 (its
README is in git at `9d53ae0`).

## Known SlipPY issues

Results of `SubsurfaceStress` are recorded with the label "suspected SlipPY defect, not fixed
(José, 2026-10-16)". SlipPY is used unmodified (`df5916c`). `02_subsurface` records each solve
attempt in `results/02_subsurface.csv`, with its traceback and the SlipPY log beside it.

| # | Defect | Where (SlipPY) | Effect here | Traceback |
|---|---|---|---|---|
| 1 | The data check mutates `step.provides` (`steps.py:88` `current_state = self.provides`, `:170` `.update(model.provides)`; 16 -> 18 entries), so the provides check at `steps.py:275-278` raises; `skip_data_check` avoids it | `contact/steps.py:88`, `:170`, `:275-278` | attempt 1 `solve()` raises | `02_subsurface_<model>_traceback_1.txt` |
| 2 | `SubsurfaceStress.solve` returns the whole state, not only its own outputs; the step requires the two sets to be equal | `contact/sub_models/sub_surface_stress.py:151`; `contact/steps.py:287-294` | attempt 2 `solve(skip_data_check=True)` raises | `02_subsurface_<model>_traceback_2.txt` |
| 3 | Tensor components (`'all'`, `'xx'`, ...) are passed to `get_derived_stresses`, which accepts only `'1'`, `'2'`, `'3'`, `'vm'`, against its own docstring | `contact/sub_models/sub_surface_stress.py:147`; `core/_stress_utils.py:265-267` | no stress tensor from the sub-model; only principal stresses | `02_subsurface_tensor_traceback_3.txt` |
| 4 | `plan_multi_convolve` called with the default `fft_ims=True` on spatial kernels | `contact/sub_models/sub_surface_stress.py:133` | field about 1e-5 of the true one (measured tau1 ratio 1.2e-5 at 0.24a, 5e-6 at 0.48a, 4e-6 at 0.96a); recorded as value, expected, ratio | - |

Attempts 1 and 2 are José's protocol. Attempt 3 repeats attempt 2 with
`slippy.ERROR_IF_MISSING_SUB_MODEL = False`, restored afterwards: "attempt 3: beyond the ruling's
two attempts; SlipPY's own switch named in its error message; no SlipPY change; José to confirm
whether it counts". A criterion graded on its field is marked "graded on attempt 3 (diagnostic)".

Evidence outside the repository, not used by any file here: the slippy-expert's reproduction
`scratchpad/s5/repro.py` (2026-10-16) convolves the same kernels with `fft_ims=False` and gives
`sigma_z` within 0.25 % of Johnson.

The three-attempt protocol stays local to `02_subsurface` on purpose: it is José's procedure for
this run, temporary until the SlipPY defects are settled.

L3 and L4 need the tensor and are recorded as "blocked: suspected SlipPY defect, not fixed
(José, 2026-10-16)". `<model>` is `principal` (L2) or `tensor` (L3-L4).

## Result

Not run. Smoke results are in step 7.
