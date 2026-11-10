# r03_load_zone: static load zone of a single-row rolling bearing

## 1. Reference
The static distribution of the load among the rolling elements of a single-row bearing with rigid rings
(Harris and Kotzalas 2007, vol. 1, ch. 7): each element is a nonlinear spring `Q = K_n delta^n` between the
rings; a relative ring displacement (`delta_r`, `delta_a`) compresses the elements on one side, the
compression falls off with the azimuth `psi` and stops at the limiting angle `psi_l`; the element forces
balance the applied `F_r`, `F_a`. It holds the Stribeck constant, the Sjovall integrals `J_r(eps)`,
`J_a(eps)` (point and line contact, Tables 7.1 and 7.4), the element loads `Q_j` at given azimuths for a
given displacement, the displacement that balances given loads (element by element or in the continuous
form of the tables), and the Hertz series of the two contacts of an element: `K_n` from the contact
constants `K_i`, `K_o`, the total approach `delta_i + delta_o`, and the ring shift `delta_r` from `Q_max`
(radial bearing). The contact constants are inputs: r03 computes no Hertz contact (a v case takes them
from r02 `hertz(...).delta_m`, `K = Q / delta^1.5`). Reference of v003 (load zone: `Q_j`, `Q_max`, `eps`,
`delta_r`, `delta_a`), v004 (stiffness), v105 (the `c_p` row: series of the two contacts, with r02's
contacts) and s004 (`delta_r` from `Q_max`). An independent judge (D-0036 amendment rule 13): NumPy,
SciPy and the standard library; it imports no other r. Of the library functions rule 13 allows
(`R_VERIFIED` in `tests/test_imports.py`: SlipPY `hertz_full`) it needs none.

## 2. Sources
| Key | What | Equation / table | PDF page |
|---|---|---|---|
| Harris2007_vol1 | load-deflection `Q = K delta^n`, n = 3/2 and 10/9; Hertz series of two contacts | Eqs 7.1-7.7 | 155-156 |
| Harris2007_vol1 | radial load with diametral clearance: `delta_psi`, `eps`, `psi_l`, `Q_psi` | Eqs 7.10-7.15, Fig. 7.1 | 156-157 |
| Harris2007_vol1 | radial equilibrium, `J_r(eps)`, `Q_max = K_n (delta_r - P_d/2)^n` | Eqs 7.16-7.22, Table 7.1 | 158 |
| Harris2007_vol1 | Stribeck 4.37 (balls), 5 (with clearance, approximate), 4.08 (rollers) | Eqs 7.23-7.25 | 159 |
| Harris2007_vol1 | combined load, zero clearance: `delta_psi`, `eps`, `psi_l`, `J_r`, `J_a`, `Q_max` | Eqs 7.58-7.71, Fig. 7.13, Table 7.4 | 169-171 |
| Harris2007_vol1 | angular-contact ball bearing under thrust: contact angle changing with load (cited, not used) | Eqs 7.27-7.36 | 160-163 |
| Hamrock1983 | ball-bearing relative radii `R_x`, `R_y` (for a v case building `K_i`, `K_o`; not used here) | (23)-(26) | 16-17 |
Every value of Tables 7.1 and 7.4 read from the page image at 130 dpi (the text layer of Table 7.1
interleaves its two halves; Table 7.4 text and image agree). Pages are PDF pages (the Harris
text layer carries no printed page numbers; PDF pages are used throughout).

## 3. Scope and validity
- Rigid rings, rigid housing and shaft; static (no centrifugal or gyroscopic load, Harris Sec. 7.6); the
  contact angle is the same for every element and **does not change under load** (Harris Sec. 7.5.1: "the
  values of the integrals are approximate"). A model whose contact angle follows the load (AxisForge
  ISO/TS 16281) differs from r03 by that effect: a v case states it or keeps `alpha` small.
- Element compression: `delta_psi = delta_a sin(a) + delta_r cos(a) cos(psi) - P_d/2`. It is Harris Eq.
  7.10 at `a = 0` and Eq. 7.58 at `P_d = 0`; **both at once is r03's own combination** (clearance along
  the contact normal at a fixed angle; Harris treats clearance at `a != 0` by the change of the contact
  angle, Eqs 7.27-7.36, PDF p. 160-163). OPEN R5-06.
- The Sjovall integrals are the continuous limit (many elements). The printed `eps = 0` rows ("1/Z") are
  the discrete one-element limit and are checked with the element sums (OPEN R5-02).
- No Hertz contact in r03: `K_i`, `K_o`, `K_n` are inputs (balls: r02 in the v case; rollers: the caller's,
  Harris Eq. 7.9 Palmgren is steel-only and empirical, OPEN R5-07).
- `delta_r` from `Q_max` (`ball_deflection`) is for a radial bearing (contact angle 0) only; at `a > 0` the
  shift depends on `delta_a` too, use `solve_ring_displacement`.
- **What r03 judges exactly, and what it does not** (bearing-expert review, for v003):
  - **Deep-groove ball bearing** under `F_r` (any clearance, `F_a = 0`): r03 at `alpha = 0` with
    `clearance_m` = AxisForge `s` (diametral, the same `P_d`) and `K_n` = AxisForge `c_p` is the exact judge
    of `Q_j`, `delta_j`, `delta_r`, `eps`. Not `delta_a`: AxisForge measures it from the `alpha_0` touching
    state (`-A sin(alpha_0)` at `F_a = 0`), r03 has none.
  - **Angular-contact ball bearing**: AxisForge's contact angle follows the load (curvature-centre model);
    r03 keeps it fixed (its first-order linearisation, gap of order `delta/A`: a few % in `delta_r`, more at
    the zone edge). Outside r03's exact scope: v003 records it (criterion on `Q_max`, `delta_r` with a
    tolerance from `delta/A`). Pass `clearance_m = 0` for it (AxisForge's derived `s` is not a clearance).
  - **Cylindrical roller bearing**: AxisForge applies the ISO/TS 16281 reference profile in laminae; r03
    has no laminae and no profile (one spring `K_n delta^n` per roller). Outside r03's exact scope: v003
    records it, with `n = 10/9` and `K_n` = AxisForge `c_L` (whole roller), not `N_LINE`.
  - Clearance is always the diametral `P_d` (Harris Eq. 7.10).
- Domain errors raise `ValueError`: `eps < 0` or nan, `n < 1`, non-positive contact constants,
  `delta_r < 0`, an angle outside `[0, pi/2)`, an axial load at `a = 0`, no axial load at `a > 0`, a load
  beyond the single-row limit (`F_r tan(a)/F_a` at or above about 1; a balance worse than 1e-6 of the load
  is refused), an empty azimuth list.

## 4. Public API
```python
from studies.r00_analytical_references.r03_load_zone import (
    N_POINT, N_LINE, sjovall_integrals, stribeck_constant, ball_deflection, ring_load_distribution,
    solve_ring_displacement)
sjovall_integrals(eps, *, n=N_POINT) -> dict          # J_r, J_a, psi_l_rad, Fr_tan_alpha_over_Fa
stribeck_constant(*, n=N_POINT) -> float              # 1 / J_r(0.5): 4.370 balls, 4.077 rollers
ball_deflection(Q_N, *, K_i, K_o, n=N_POINT, clearance_m=0.0) -> dict
    # delta_i_m, delta_o_m, delta_n_m, K_n_N_per_m_pow_n (Eq. 7.6), delta_r_m = delta_n + P_d/2 (a = 0)
ring_load_distribution(delta_r_m, delta_a_m=0.0, *, psi_rad, K_n, n=N_POINT, alpha_rad=0.0,
                       clearance_m=0.0) -> dict
    # psi_rad, delta_psi_m, Q_N (per element); delta_max_m, Q_max_N, epsilon, psi_l_rad, J_r, J_a,
    # Fr_N, Fa_N (element sums), Fr_continuous_N, Fa_continuous_N (Z Q_max J cos/sin a)
solve_ring_displacement(Fr_N, Fa_N=0.0, *, psi_rad, K_n, n=N_POINT, alpha_rad=0.0, clearance_m=0.0,
                        continuous=False) -> dict
    # ring_load_distribution at the solution + delta_r_m, delta_a_m, converged, n_radial_evaluations,
    # n_axial_iterations, residual_Fr_N, residual_Fa_N, continuous
```
**Conventions.** SI throughout (m, N, Pa; `K_n` in N/m^n). `psi` is measured from the direction of
`delta_r` (`psi = 0` = most loaded element, Harris Figs 7.1, 7.13); `delta_r >= 0`; `delta_a > 0` loads
contacts of `a > 0`. `clearance_m` is the diametral clearance `P_d`, negative for interference. Forces are
those the elements exert on the displaced ring. How `psi` maps on AxisForge's azimuth and load direction is
the v case's (convention-dependent, R-003; OPEN R5-08). AxisForge gives mm and MPa: a caller multiplies
lengths by 1e-3, moduli by 1e6, and `K_n` [N/mm^n] by 1e3^n.

**Solver.** `scipy.optimize.brentq` on bracketed monotone functions (no Newton: an r imports nothing from
`numerical_solvers`, rule 13). After the root, a balance worse than 1e-6 of the load raises: past the
single-row limit brentq would otherwise return a roundoff root at a huge `delta_r` (bearing-expert review). At `a = 0` one unknown `delta_r`; at `a > 0` the axial balance is solved
for `delta_a` inside each radial evaluation. Tolerances `xtol = 1e-14` times the displacement scale,
`rtol = 4 machine epsilon`; residuals below 1e-11 N in the tests.

## 5. Data
`published/harris2007_vol1_table7_4.csv` (Table 7.4, PDF p. 171: `eps`, `F_r tan(a)/F_a`, `J_r`, `J_a`, point
and line, 15 rows each) and `published/harris2007_vol1_table7_1.csv` (Table 7.1, PDF p. 158: `eps`, `J_r`,
point and line, 16 rows each), values as printed (strings, "1/Z" kept), with `source_key`, `location`,
`pdf_page`, `read_from`, `note`. One row per `eps` and contact (wide), not the long schema of r04: the
columns are the table's. Table 7.1 repeats the `J_r` of Table 7.4 for the same `eps` (only `eps = 0.1` is
new); it is kept as a check of the two printings.

Findings on the tables (never tuned; the tests carry the reason):
- **`eps = 1.67` is `5/3`** in both tables: all six values match `5/3` to half a unit, none match 1.67
  (OPEN R5-03).
- **Line contact is computed with `n = 1.1`**, not the `10/9` of Eq. 7.4 (1.1: 4 misses of 26 values, 10/9:
  26; a scan 1.09-1.12 has its minimum at 1.1; Eq. 7.62, PDF p. 170, prints `n = 1.11`, which misses as 10/9
  does; librarian count over all 39 printed line values: 1.1 misses 12, 10/9 and 1.11 miss 38). `N_LINE = 1.1`
  (OPEN R5-01).
- **Table 7.4, point, `eps = 1.25`: `F_r tan(a)/F_a` printed 0.4338, a misprint of 0.4538** (`J_r/J_a` =
  0.2289/0.5044 = 0.4538; integral 0.453778).
- Table 7.4, point, `eps = 0.3`: ratio printed 0.8964, integral 0.896470 (truncated, not rounded).
- Table 7.4, line: the ratio column is the ratio of the printed (rounded) `J_r`, `J_a` (8 rows miss the
  integral by 0.8-3.1 units; all 13 match the printed ratio, `test_line_ratio_column`).
- Line `J`: Tables 7.1/7.4 `J_r(0.2)` 0.1737 (0.173637), `J_r(1.25)` 0.2078 (0.207867), Table 7.4
  `J_a(0.8)` 0.4098 (0.409876), `J_a(0.9)` 0.4439 (0.444059): 0.6-1.6 units of the last digit.
  OPEN R5-04 for all of these.
- **Eq. 7.70 prints `Q_max = F_r / (J_r Z sin a)`; Eq. 7.66 gives `cos a`.** r03 uses `cos a`
  (tested both ways round with Eq. 7.71; OPEN R5-05).

## 6. Criteria (`tests/test_r03.py`), fixed before running
| ID | Quantity | Role | Tolerance | Result |
|---|---|---|---|---|
| C1 | Table 7.4: every printed `J_r`, `J_a`, `F_r tan(a)/F_a`, point and line (eps 0 in C2) | published | half a unit of the last printed digit | 70 of 84 pass; 14 misses of section 5, xfail strict |
| C1b | Table 7.1: every printed `J_r` | published | same | 28 of 30; 2 line misses (section 5) |
| C1c | all printed `J` within 2 units; line ratio = printed `J_r/J_a` | record bound | 2e-4 / half unit | pass |
| C2 | `eps = 0` rows: one element loaded, `F_r/(Z Q_max) = 1/Z`, ratio 1 (radial with clearance, combined) | discrete limit | rel 1e-12 | pass |
| C3 | Stribeck 4.37 (Eq. 7.23) and 4.08 (Eq. 7.25, n = 1.1 and 10/9); radial zero-clearance solve: `eps = 0.5`, `psi_l = 90 deg`, `Q_max Z / F_r` = constant | published / identity | half a unit / 1e-10 | pass (4.3701, 4.0766) |
| C4 | `n = 1` closed form of `J_r`, `J_a` (partial and full zone); `eps -> inf`: `J_r eps -> n/4`, `(1 - J_a) eps -> n/2`; `eps -> 0` limits; line vs point | closed form / limit | rel 1e-11 / 1e-5 | pass |
| C5 | Eqs 7.12-7.13, 7.22 (clearance and interference), 7.61, 7.65 (combined), 7.66 and 7.71 at the combined solve, zone edge | identity | rel 1e-12 to 1e-14 | pass |
| C6 | discrete solves balance `F_r`, `F_a` (5 cases: clearance, interference, offset azimuths, combined); Z = 360 discrete vs continuous | identity / limit | 1e-9 F / rel 1e-4 | pass |
| C7 | just beyond the single-row limit (`F_r tan(a)/F_a` = 1.05, 1.2), discrete and continuous: raises | validity | exact | pass |
| C8 | Hertz series (Eqs 7.5-7.7), `delta_r` from `Q_max` round trip through the radial solve | identity | rel 1e-12 | pass |
| C9 | domain errors raise | validity | exact | pass |

## 7. Depends on
NumPy, SciPy (`quad` with Jacobi weights, `brentq`, `beta`) and the standard library. Nothing from
`studies/`, AxisForge, SlipPY or the bridge (rule 13, `tests/test_imports.py`). Written at
design-studies `84344df` (branch `layout/studies-levels`). Blockers: none.

## 8. How to run
```bat
python -m pytest studies\r00_analytical_references\r03_load_zone
```
No `study.py` and no `results/` (SQ-08): the tests against the published tables are the case.
