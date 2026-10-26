# hertz_ellipse

Validation case: SlipPY static BEM, through `slippy_bridge.contact`, against the Hertz
elliptical contact; the judge self-checked against Hamrock & Brewe (1983). V-DRY-02.

## Question

Does SlipPY's static BEM reproduce the Hertz elliptical contact (semi-axes, maximum pressure,
approach, orientation) over the catalogue range `R_y/R_x = 5-10` and for the ball on the inner
raceway of a deep groove ball bearing?

## Sources

| Key | Item | PDF page |
|---|---|---|
| HamrockBrewe1983 | Eqs (2), (4)-(7), (13)-(17): definitions and exact forms | 2-3 |
| HamrockBrewe1983 | Table 1: `k`, `F_int`, `E` exact, alpha 1-100 | 4 |
| HamrockBrewe1983 | Table 2: `D_y`, `D_x`, `sigma_max`, `delta` exact; the case constants in the caption | 5 |
| HamrockBrewe1983 | Table 3: `t`, `x_0`, `z_0`, `tau_0` (not graded) | 5 |
| HamrockDowson1977 | Eqs (29)-(31), `k = 1.03 alpha^0.64` (formula rows only) | 10 |

The printed rows are in `reference_values.csv`, copied unchanged from the vault
(`80_Inbox/2026-10-16/reference_values/contact/hertz_ellipse/`).

## Conventions

- `alpha = R_y/R_x`; `1/R_x = 1/r_ax + 1/r_bx`, `1/R_y = 1/r_ay + 1/r_by`. Convex radii are
  positive, concave negative. `x` is the rolling direction, array axis 0.
- `k = D_y/D_x` with diameters. For alpha >= 1, `k >= 1` and the major axis lies along `y`, on
  array axis 1.
- `E' = 2/((1 - nu_a^2)/E_a + (1 - nu_b^2)/E_b) = 2E*`. For identical bodies `E = E'(1 - nu^2)`;
  nu = 0.3 is chosen and does not change `E*`.
- `F_int`, `E` are the complete elliptic integrals of the first and second kind at
  `m = e^2 = 1 - 1/k^2`.
- Units SI inside the case. The Hamrock & Brewe case is converted from cm and N/cm^2 in
  `construction.py` only (`CM`, `N_PER_CM2`).
- The printed percent errors use `(z - z_bar)/z`, the opposite sign of Eq. (33). They are not used.

## Reference

The judge is `references/contact/hertz.py::hertz_point`. `02_ratio` checks it against the exact
columns of Tables 1-2 at alpha = 5-10 (E1, E2). The BEM is graded against the judge (E3-E5). The
bearing contact (`R_y/R_x` about 32) has no table row; its judge is `hertz_point`. The bridge
record's `err_*` are relative to SlipPY's own `hertz_full`. They are recorded and never graded.

## Case and loads

| Script | Contact | Load | Source |
|---|---|---|---|
| `01_ball_raceway` | 12.7 mm ball on the inner raceway: `Dpw` 70 mm, groove 0.52 `Dw`, contact angle 0; steel 210 GPa, nu = 0.3 | 4.448 N | HamrockBrewe1983, Tables 2-3 caption, PDF p. 5 |
| `02_ratio` | `R_x` = 1.0 cm, `R_y = alpha R_x`, alpha = 5-10; `E'` = 2.1972e7 N/cm^2 | 4.448 N | the same caption: exactly the printed case |

Table 5 (ball bearing applications, PDF p. 6) is not used. On two of its load rows the CSV value
is 222.4111 N, while the note in the same row says printed "1.00x10^5 N". The CSV is kept
unchanged.

Assumptions: smooth, frictionless, linear elastic half-spaces, static normal load, aligned
principal axes.

## Criteria

Fixed before running. The [proposed] tolerances are accepted provisionally by José, 2026-10-16.

| ID | Quantity | Criterion | Script |
|---|---|---|---|
| E1 | Judge: `k`, `F_int`, `E` against Table 1 exact, alpha 5-10 | rel 1e-4 | `02_ratio` |
| E2 | Judge: `D_y`, `D_x`, `sigma_max`, `delta` against Table 2 exact, alpha 5-10 | about 0.5 % (three printed digits), made explicit as `max(0.5 %, half unit of the last printed digit)` | `02_ratio` |
| E3 | BEM `s_x`, `s_y`, `p0` against the judge, alpha 5-10 and the bearing contact | `abs(err) <= 1 %` | `02_ratio`, `01_ball_raceway` |
| E4 | BEM `delta` against the judge | 2 % [proposed] | `02_ratio`, `01_ball_raceway` |
| E5 | Axis map: `x` rolling on axis 0, major axis on axis 1 | `s_y > s_x` (BEM and judge) | `02_ratio`, `01_ball_raceway` |

Table 3 is not graded in phase 1 (it needs the subsurface field). Table 2 prints `D_x` at alpha
= 10 as 0.0087 cm, where the half unit of the last digit is 0.57 %: E2's limit there is 0.57 %.
`02_ratio` prints the print resolution next to each E1/E2 error.

## Scripts

Every script is a jupytext pair (`.py` and `.ipynb`, `py:percent`). It prints the criteria table and
writes `results/<script>.csv` and `results/<script>_criteria.csv` (git-ignored). `--smoke` runs the
coarsest grid (`2 h0`) and the composite build only. In a notebook the full grids run.

| Script | Content |
|---|---|
| `construction.py` | The bearing contact (`CASE`, SI), `RATIO_CASES` (SI), the grid levels, `ball_bearing()` (AxisForge, mm/N/MPa); nothing solved |
| `01_ball_raceway` | Mesh and domain sweep of the bearing contact (two builds); the AxisForge bearing through the bridge, checked to be the same contact and solved; E3-E5 |
| `02_ratio` | Judge self-check E1-E2; BEM on the default grid per alpha; E3-E5 |

Run from this folder (the scripts import `construction` as a sibling module):
`python 01_ball_raceway.py [--smoke]`.

## Provenance

| | |
|---|---|
| design-studies | to be filled by the run |
| AxisForge | `c328999` (`refactor/gears-mechanical-system`) |
| SlipPY | `df5916c` (`local-fixes`), unmodified |

The study `studies/bearings/contact/hertz_vs_bem` gave provisional results on 2026-10-10 with a
500 N placeholder load (its README is in git at `9d53ae0`).

## Known SlipPY issues

Results of `SubsurfaceStress` (Table 3, not graded here) would be recorded with the label
"suspected SlipPY defect, not fixed (José, 2026-10-16)". This case runs no subsurface solve.

## Result

Not run. Smoke results are in step 7.
