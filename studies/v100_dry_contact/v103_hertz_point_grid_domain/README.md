# hertz_point_grid_domain

Validation case: SlipPY static BEM, through `slippy_bridge.contact`, against the Hertz sphere on
a flat; grid, domain and kernel convergence; subsurface principal shear. V-DRY-01, V-DRY-04.

## Question

Does SlipPY's static BEM reproduce the Hertz circular contact (contact radius, maximum and mean
pressure, approach, axis stresses), and which grid and domain does it need to do so?

## Sources

| Key | Item | PDF page |
|---|---|---|
| Johnson1985 | Sec. 4.2(a), Eqs (4.17)-(4.24): `a`, `p0`, `delta`, `p_m` | 102-103 |
| Johnson1985 | Surface stresses at the edge, `(1 - 2nu) p0/3` | 104 |
| Johnson1985 | `tau1` max 0.31 `p0` at 0.48`a` (nu = 0.3); shear 0.10 `p0` at the origin, 0.13 `p0` at the edge | 105 |
| Johnson1985 | Eqs (3.43)-(3.47): surface and axis stresses | 72 |
| Johnson1985 | Table 4.1 (circular limit), Appendix 3(b) | 109, 437-438 |
| PolonskyKeer1999 | Error measures and trends, `eps_r < 1/M`, `eps_p ~ M^-3/2` | 9-11 |
| Muser2014 | Resolution rule `Delta_a/a_c = 1/32`, `a_c/L = 1/8`, area error < 0.1 % | 7 |

The printed rows are in `reference_values.csv`, copied unchanged from the vault
(`80_Inbox/2026-10-16/reference_values/contact/hertz_point_grid_domain/`).

## Conventions

- `1/E* = (1 - nu1^2)/E1 + (1 - nu2^2)/E2`, `1/R = 1/R1 + 1/R2`, `P` the total load, `delta` the
  approach of distant points, `p_m = P/(pi a^2)`, `p0 = (3/2) p_m`.
- `x` is the rolling direction, array axis 0. Units SI (m, Pa, N) throughout the case.
- `M = 2a/h`, the number of nodes across the contact diameter.
- P1: the measured contact radius is `sqrt(s_x s_y)`, with the semi-axes that
  `slippy_bridge.contact.measure` places below one node. Polonsky & Keer's area is `N_c h^2` (the
  count of loaded nodes), so P1's `< 1/M` on the measured radius is looser than theirs.
- P2 uses Polonsky & Keer's own measures on the pressure field: `eps_r = abs(sqrt(N_c h^2/pi) - a)/a`
  and `eps_p = sum_Ic abs(p_ij - p_H(x_i, y_j)) / sum_Ic p_H(x_i, y_j)`, the L1 error over the loaded
  nodes `I_c`, with `p_H` from `hertz_point`. Their `M^-3/2` bounds `eps_p`, not the error of the
  peak pressure.
- Subsurface depths: `Z_OVER_A[0]` is 1e-4 of the first depth step, SlipPY's own default
  (`slippy/core/elastic_material.py:328`), because its kernels are singular at `z = 0`. The surface
  criteria (P5) are evaluated at that depth.
- Johnson's stress constants hold for nu = 0.3, the value of the case.

## Reference

The judge is `references/contact/hertz.py::hertz_point`. Johnson's closed forms (4.22)-(4.24)
are exact. The subsurface reference is `tau1` max = 0.31 `p0` at `z` = 0.48`a`, and Eq. (3.45)
evaluated directly in `02_subsurface.py`. Eq. (3.46) prints 0.57`a`; that value is a misprint and
is not used. The bridge record's `err_*` are relative to SlipPY's own `hertz_full`. They are
recorded and never graded.

## Case

- Ball: 12.7 mm, steel (E = 210 GPa, nu = 0.3), on a steel flat.
- Load: 500 N. The load is chosen: the reference scales with it, and every graded quantity is
  scaled by `a` and `p0`.
- Assumptions: smooth, frictionless, linear elastic half-spaces, static normal load.

## Criteria

Fixed before running. The [proposed] tolerances are accepted provisionally by José, 2026-10-16.

| ID | Quantity | Criterion | Script |
|---|---|---|---|
| P1 | `a`, `p0`, `delta`, `p_m` on the default grid against Eqs (4.22)-(4.24) | `abs(eps_a) < 1/M` and `abs(eps_p0) <= M^-3/2` (P&K's bounds applied to the measured radius and the peak pressure) [proposed]; `delta` within 2 % [proposed]; `p_m` recorded | `01_ball_flat` |
| P2 | Mesh sweep, composite build, P&K's measures | at each level `eps_r < 1/M`, `eps_p <= M^-3/2`; observed order of `eps_p` >= 1.5 [proposed reading of "~"] | `01_ball_flat` |
| P3 | Muser rule: `h = a/32`, cell `8a` | `abs(err contact area) < 0.1 %`, area `pi s_x s_y` | `01_ball_flat` |
| P4 | `tau1` max and depth on the axis | `abs(tau1max/p0 - 0.31) <= 0.005`, `abs(z/a - 0.48) <= 0.01`; the same against Eq. (3.45) | `02_subsurface` |
| P5 | Shear at the origin; surface shear and radial tension at the first node with `r >= a` | 0.10 `p0`; `(1 - 2nu) a^2/(3 r^2) p0` (Eq. (3.44), 0.13 `p0` at `r = a`) at that node's `r`; within 0.005 `p0` [proposed] | `02_subsurface` |
| P6 | Domain half-length 2, 4, 8 `a`, spatial kernel | `p0` and `delta` change < 0.05 % between half-lengths [proposed]; frequency kernel recorded | `03_kernel` |

## Scripts

Every script is a jupytext pair (`.py` and `.ipynb`, `py:percent`). It prints the criteria table and
writes `results/<script>.csv` and `results/<script>_criteria.csv` (git-ignored). `--smoke` runs the
coarsest grid only. In a notebook the full grids run.

| Script | Content |
|---|---|
| `construction.py` | The contact (`CASE`, SI) and the grid levels; returns nothing solved |
| `01_ball_flat` | Mesh sweep (two builds), Muser grid; P1-P3; convergence and pressure-cut figures |
| `02_subsurface` | `SubsurfaceStress` on the axis, built with SlipPY directly (José, Q2); P4-P5 |
| `03_kernel` | Domain sweep with the frequency-domain and the spatial-domain kernel; P6 |

Run from this folder (the scripts import `construction` as a sibling module):
`python 01_ball_flat.py [--smoke]`.

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
| 1 | The data check mutates `step.provides` (`steps.py:88` `current_state = self.provides`, `:170` `.update(model.provides)`; 16 -> 18 entries), so the provides check at `steps.py:275-278` raises; `skip_data_check` avoids it | `contact/steps.py:88`, `:170`, `:275-278` | attempt 1 `solve()` raises | `02_subsurface_traceback_1.txt` |
| 2 | `SubsurfaceStress.solve` returns the whole state, not only its own outputs; the step requires the two sets to be equal | `contact/sub_models/sub_surface_stress.py:151`; `contact/steps.py:287-294` | attempt 2 `solve(skip_data_check=True)` raises | `02_subsurface_traceback_2.txt` |
| 3 | Tensor components (`'all'`, `'xx'`, ...) are passed to `get_derived_stresses`, which accepts only `'1'`, `'2'`, `'3'`, `'vm'`, against its own docstring | `contact/sub_models/sub_surface_stress.py:147`; `core/_stress_utils.py:265-267` | no stress tensor from the sub-model; only principal stresses | - (principal stresses only are requested) |
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

## Result

Not run. Smoke results are in step 7.
