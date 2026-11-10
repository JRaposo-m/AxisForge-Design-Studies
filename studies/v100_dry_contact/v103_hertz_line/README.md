# v103 line

## 1. Question
Does SlipPY's static BEM, through `slippy_bridge.contact` (a line contact is one period of a strip periodic
along the line), reproduce the plane-strain Hertz contact of two cylinders: half-width `b`, peak `p0`, profile
`p(x)`? One validation: **line contact on a periodic strip vs r02 line** (v_prerequisites row 4). Ported in
spirit from `v100_dry_contact/v101_hertz_line_subsurface/01_roller_raceway` (tag `studies-pre-sweep`); the
subsurface part waits for v108 (Q-03).

## 2. Sources
| key | item | PDF page |
|---|---|---|
| Johnson1985 | Sec. 4.2(c): load P per unit length; `1/R = 1/R1 + 1/R2` (4.36); `a^2 = 4PR/(pi E*)` (4.43); `p(x)` (4.44); `p0 = 2P/(pi a) = (4/pi) p_m` (4.45) (Johnson's `a` is our `b`) | 110-111 |
| Johnson1985 | the approach "cannot be found from local contact considerations" | 110 |
Rows: `literature/published/johnson1985_line.csv` (vault extraction 2026-10-16 through the tag; the `4/pi` row
added from Eq. (4.45)).

## 3. Model
Two steel rollers, radii 10 and 20 mm (`E` 210 GPa, `nu` 0.3), axes along bridge y, 1e5 N/m (r02's line test
case: `p0` 742.240 MPa, `b` 0.08577 mm). `LineContact` (a one-line `Contact` subclass, `line = True`, OPEN
R2-03). `Grid.around(..., n_y_line=16)`: 16 nodes along the periodic axis; the frequency kernel (the only one
the bridge allows for a line: the spatial kernel would be a finite strip) repeats the load along x every
2 n h (images). A load uniform along y has only `k_y = 0` components, so `n_y` does not change `p` (it reaches only the
zero-frequency term, i.e. the approach): `n_y` is run time only (slippy-expert). **Load control only**: an interference given to a line is a number of its grid (the
approach is not defined locally), so J-7 does not apply here.
- h sweep: `b/h` 8, 16, 32, 64 asked (actual 11.2, 22.5, 45, 90) at half-length 4 half-widths.
- Domain sweep: half-length 2, 4, 8, 16 at `b/h` 16 asked.

## 4. Public API
```python
check(result, *, case="v103") -> list[dict]       # a SOLVED line contact vs r02 hertz(line=True)
literature.cases() -> [LiteratureCase(...)]; LineContact; rollers(load=1e5, *, r1, r2)
```

## 5. Inputs
As section 3. SI (load per unit length N/m).

## 6. Criteria (fixed before the run)
Judged at `b_H/h >= 16` (v104) and a domain half-length `>= 4 b` (the bridge's `SPAN_X_0`, where v104 found the
point-contact `p0` within 0.05 %), converged, edge clear, load balanced, Hertz reference in range; otherwise
RECORD. `M = 2 b_H/h`.
| id | quantity | tolerance | source |
|---|---|---|---|
| C3 | `p0` | `M^(-3/2)` | as v101 C3 |
| C4 | `b` (half-width) | `1/M` | as v101 C4 |
| C5 | Polonsky-Keer `eps_p` against `p_H(x)` (4.44) | RECORD | |
| C6 | load per unit length carried, `sum p h^2 / period` | RECORD | a balance |
| L1 | `p0/p_m` vs Johnson's printed `4/pi`, `p_m = w/(2b)` from the BEM | RECORD | implied by C3-C4 |
| - | approach | not compared | Johnson PDF p. 110; r02 gives none |

## 7. How to run
`python -m studies.v100_dry_contact.v103_hertz_line.study` (about 9 s) writes `results/v103_{bem,check}.csv`,
`v103_contact_relative_errors_{h,span}.png`, `v103_contact_pressure_cut_bh16_asked.png`. `notebook.ipynb` runs
the same, committed with its outputs. Tests: `python -m pytest studies/v100_dry_contact/v103_hertz_line`.

## 8. Result (run of 2026-10-28)
**PASS: C4 everywhere judged; C3 at `b/h` 22.5 (half-length 4) and at half-length 4, 8, 16 (`b/h` 22.5).
MISS: C3 at `b/h` 45 and 90 (half-length 4): p0 +0.32 % against `M^-3/2` = 0.12 % and 0.04 %.**
| sweep | actual b/h | half-length | p0 | b | eps_p | p0/p_m vs 4/pi |
|---|---|---|---|---|---|---|
| h | 11.2 (RECORD) | 4 | +0.29 % | -0.93 % | 0.75 % | -0.64 % |
| h | 22.5 | 4 | +0.31 % PASS (tol 0.33 %, marginal) | -0.76 % PASS | 0.49 % | -0.45 % |
| h | 45 | 4 | +0.32 % **MISS** | +0.07 % PASS | 0.32 % | +0.39 % |
| h | 90 | 4 | +0.32 % **MISS** | -0.33 % PASS | 0.37 % | -0.01 % |
| span | 22.5 | 2 (RECORD) | +1.22 % | -1.55 % | 1.66 % | -0.34 % |
| span | 22.5 | 8 | +0.075 % PASS | -0.48 % PASS | 0.20 % | -0.41 % |
| span | 22.5 | 16 | +0.016 % PASS | -0.41 % PASS | 0.13 % | -0.39 % |
Cause of the miss (found): the p0 error does not fall with `h` at half-length 4 (+0.29 to +0.32 %) and falls
4x each time the domain doubles (1.22 / 0.31 / 0.075 / 0.016 % at 2 / 4 / 8 / 16): the frequency kernel's
image strips along x, not the discretisation. Estimate (physics-reviewer; derived from Johnson's surface
gradient of a strip load, Eq. (2.25b), page not read here): images of the same load every `L = 2 n_x h`
(about 4 x span x b) add a curvature that raises p0 by `pi^2 b^2/(12 L^2)` = 0.0514/span^2: 1.29 / 0.32 / 0.080 /
0.020 %, as measured. The judged domain of section 6 (from v104's point contacts) is too small for a line,
whose images are whole strips. **Grid rule for a line (frequency kernel): the image error must stay inside
the C3 tolerance, `0.0514/span^2 <= M^-3/2`, i.e. half-length `>= 0.23 M^0.75` half-widths (3.9 at M 45, 6.8 at
M 90, 11.4 at M 180), with `b/h >= 16`.** The half-width: its image part falls like p0 (about -0.02 % at
half-length 16); the rest (about +-0.4 %, changing sign with the grid phase: -0.93 / -0.76 / +0.07 / -0.33 %) is
the edge measurement at the threshold 1e-6 p0 on a square-root edge, within `1/M`. OPEN R2-08: at half-length 4 the image error (0.32 %) exceeds `M^-3/2` from `b/h` about 23, so C3
as fixed cannot pass under refinement there; for José: judge a line at half-length >= 16, or take
`M^-3/2 + 0.051/span^2` (criteria revision, not applied). C6 is a consistency check
only (under load control SlipPY scales the pressure to the given load); L1 is C3 minus C4 (no independent
information).

## 9. Depends on
v101 (`reference`, `bem_load`, `pressure_l1_error`, `pressure_cuts`, `LiteratureCase`), v104 (`fc76048`, grid
rule), r02 (`407273d`), `slippy_bridge.contact`, `studies/outputs`. AxisForge `d34ec7f`, SlipPY `df5916c` (read
only). Not covered: approach, subsurface (v108), finite rollers and edges.
