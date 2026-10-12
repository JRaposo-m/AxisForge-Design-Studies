# iso16281_ball_contact

**Validation case — ball bearing contacts (AxisForge `ISO16281BallSolver`, `slippy_bridge`).**
**Implementation check, same theory** — verification in the sense of `validation/README.md`
(the code reproduces the equations of its model), not validation of the model.

## Question

Do the contacts of a solved AxisForge ball bearing (element load `Q_j`, operating angle
`alpha_j`, inner and outer raceway), passed through `slippy_bridge`, agree with a Hertz solution
written independently from the theory?

## What this case checks, and what it does not

AxisForge's load-deflection constant `cp` *is* Hertz, and the judge is Hertz. Agreement therefore
shows that the theory is **coded** consistently in three places; it says nothing about whether
Hertz holds for this contact.

Checked:

- `cp` as coded in AxisForge (`contact_models/iso16281.py`), including the factor 2 in
  `E' = 2 / ((1 - nu1^2)/E1 + (1 - nu2^2)/E2)` (D-0007) (identical materials only; the
  dissimilar-material form of E' is not exercised);
- the radii AxisForge stores for the ball and the two raceways;
- the bridge conversion: mm -> m, MPa -> Pa, and the negative rolling radius of the outer
  raceway (concave along rolling);
- that the bridge hands on `Q_j` and `alpha_j` of a real solve to both raceways.

Not checked: that Hertz is valid here (the job of the BEM study `hertz_vs_bem` and its successor).

| | |
|---|---|
| AxisForge | commit `b3f624a` |
| Reference | `references/contact/hertz.py::hertz_point` — elliptical point contact from Johnson (1985), ch. 4; imports neither AxisForge nor SlipPY |
| Bearing | 6210-like: Dw 12.7 mm, Dpw 70 mm, Z 9, both grooves 0.52 Dw, steel 210000 MPa / 0.3 (`construction.py`) |
| Loads | `radial_moderate` s 0, Fr 3300 N; `radial_heavy` s 0.015 mm, Fr 8900 N; `combined` s 0.015 mm, Fr 8900 N, Fa 4450 N — example values |

## Criteria (fixed before running; all in `tests/`)

The script asserts nothing. The pass/fail criteria are these tests; the tolerances follow from the
algebra and were fixed before the first run.

| Test | Criterion | Tolerance (rel) |
|---|---|---|
| `test_contact_reference.py::test_hertz_pressure_closes_the_gap` | judge: the Boussinesq integral of its own pressure equals `delta - k_x x^2/2 - k_y y^2/2` at 4 interior points; curvatures and `E*` built from the raw inputs | 1e-8 |
| `test_contact_reference.py::test_pressure_carries_the_load` | judge: `(2/3) pi s_x s_y p0 = Q` | 1e-12 |
| `test_contact_reference.py::test_circular_limit_matches_johnson` | judge: sphere on flat against `a`, `p0`, `delta` of Johnson | 1e-9 |
| `test_slippy_bridge.py::test_independent_hertz_agrees_with_slippy` | judge against SlipPY `hertz_full` on the bridge case, inner and outer: `p0`, `s_x`, `s_y`, `delta` | 1e-6 |
| `test_slippy_bridge.py::test_axisforge_cp_is_hertz_with_a_rounded_constant` | `cp * (pi sqrt(2)/3) / 1.48 / cp_judge = 1`, with `cp_judge = Q / (delta_i + delta_e)^1.5` at `alpha_0`; per raceway, curvature sum and `chi` of `PointContactStiffness` against the bridge radii and the judge's semi-axis ratio; at s = 0 and s = 0.015 mm | 1e-6 |
| `test_slippy_bridge.py::test_contacts_of_a_real_iso16281_solve` | solver converged; two contacts per loaded ball; inner **and** outer carry `Q_j` and `alpha_j` | exact |

## Reported only (no pass/fail)

`01_contacts_vs_hertz.py` writes, per load case, loaded ball and raceway: `Q_j`, `alpha_j`, Hertz
`p0`, semi-axes along and across rolling, approach per raceway, the same `p0` and approach with
the raceway radii at `alpha_0` and their relative difference (F1), and the ellipse edge angle
`alpha_j + s_y / |r_groove|`. Per ball: AxisForge's `delta_j` against the summed Hertz approach at
`alpha_0` (F0).

## Findings (flagged, nothing fixed)

- **F0 — constant 1.48 against 1.48096.** The Hertz approach of a ball between two raceways
  rearranges to `Q = (pi sqrt(2)/3) E' (T_i + T_e)^(-3/2) delta^1.5`, `pi sqrt(2)/3 = 1.48096`.
  AxisForge has 1.48 (`contact_models/iso16281.py:104`): `cp` is 0.065 % low and the approach
  0.043 % high (`delta_j / delta_Hertz = 1.000433`), for every load. **Open** until the `cp`
  equation of ISO/TS 16281 is read: if the standard prints 1.48, AxisForge follows the standard.
- **F1 — curvatures at `alpha_0`, not at `alpha_j`.** `cp` uses the raceway curvatures at the free
  contact angle `alpha_0`; the operating angle `alpha_j` differs from it under load. The effect is
  reported per raceway, because the inner and the outer differences have opposite signs and
  largely cancel in the sum, which therefore understates it. It also appears under a purely radial
  load with clearance: there `alpha_0 = 9.86 deg` while the loaded balls sit at `alpha_j = 0`.
  Whether ISO/TS 16281 defines gamma at the nominal angle has not been read either.

## Limits

- Both grooves are 0.52 Dw, so a swap of the inner and outer groove radii would be invisible.
- At the heavy load the semi-axis across rolling reaches about 0.38 of the groove radius: Hertz's
  assumption of a contact small against the radii is strained. Judging that is the BEM study's job.
- AxisForge has no shoulder geometry; the ellipse edge angle is reported and compared with nothing.
  s_y is taken as an arc; read as a tangent-plane half-width (asin) the angle is about 0.5 deg
  larger. Measured about the groove centre of curvature. Indicative only: at s_y/r = 0.38 Hertz
  is strained.
- The `cp` check uses identical materials only; the dissimilar-material form of E' is not
  exercised.
- p0 up to 3.1 GPa is treated as elastic; no plasticity or static-capacity statement is made.
- The path bearing geometry -> principal radii is checked only by two codes (AxisForge, bridge)
  agreeing on the ISO curvature formula. A worked example from Harris & Kotzalas would close it;
  none is included because no numbers have been typed from the book.
- No ISO 76 check: AxisForge has no C0, and f0 has not been typed from the standard.

## Result

**Provisional — not from a committed state** (design-studies working tree on top of `ad825e8`,
AxisForge `b3f624a`, run of 2026-10-11). To be replaced by a run from a commit.

- All six tests above pass.
- F0: `delta_j / delta_Hertz(alpha_0)` = 1.000433 for all 17 loaded balls of the three load cases.
- F1, value at `alpha_0` relative to value at `alpha_j`, most loaded ball. Baseline is alpha_0
  (what AxisForge uses). Against the radii at alpha = 0 the same ball gives +0.098 % / -0.072 %
  (approach) and +0.49 % / -0.34 % (p0); these are the +0.10 / -0.07 % of the debate note, which
  used a bearing with alpha_0 = 0.

  | Load case | `alpha_j` | `p0` inner / outer | approach inner / outer |
  |---|---|---|---|
  | `radial_moderate` (`alpha_0` = 0) | 0 deg | 0 / 0 | 0 / 0 |
  | `radial_heavy` (`alpha_0` = 9.86 deg) | 0 deg | -0.15 % / +0.10 % | -0.030 % / +0.022 % |
  | `combined` (`alpha_0` = 9.86 deg) | 17.95 deg | +0.34 % / -0.24 % | +0.068 % / -0.050 % |

- Most loaded ball: `radial_moderate` 1608 N, `p0` 2194 / 1857 MPa (inner / outer);
  `radial_heavy` 4528 N, 3098 / 2623 MPa, edge angle 22.0 deg; `combined` 4048 N,
  2970 / 2535 MPa, edge angle 39.1 deg.

## Files

| File | Role |
|---|---|
| `construction.py` | Inputs (top of file), `LOADS`, `build_bearing(s)` (assembled, **not** solved) |
| `01_contacts_vs_hertz.py` | Solve, contacts through the bridge, Hertz on each; prints the tables, writes `results/contacts.csv` and `results/approach.csv` (git-ignored) |

## Running

From this folder (the script imports `construction` as a sibling module):
`python 01_contacts_vs_hertz.py`. Needs AxisForge and this repository installed
(`pip install -e`). No BEM solve is run.
