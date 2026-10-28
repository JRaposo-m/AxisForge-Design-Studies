# v102 ellipses

## 1. Question
Does SlipPY's static BEM, through `slippy_bridge.contact`, reproduce the Hertz ellipse (approach, p0, semi-axes,
ellipticity, orientation) of two bodies given with their own radii and the angle between them, over the
ellipticity range `alpha = R_y/R_x` 1-100, including a ball in a concave groove? One validation: **elliptical
BEM contacts vs r02 Hertz over alpha 1-100** (v_prerequisites row 3). Ported in spirit from
`v100_dry_contact/v102_hertz_ellipse` (tag `studies-pre-sweep`: ball-raceway and ratio scripts), rebuilt on the
r02 input convention (each body as given, the angle between them).

## 2. Sources
| key | item | PDF page |
|---|---|---|
| HamrockBrewe1983 | Table 1: exact ellipticity `k` (Hamrock-Anderson 1973 numerics) vs `alpha`, `R_x` 1 cm | 4 |
| HamrockBrewe1983 | Table 5: wheel on rail, ball on plane, ball on the outer ring; `E' = 2.197e7 N/cm^2`; printed `D_x`, `D_y`, `delta`, `sigma_max`, `k` (exact columns) | 6 |
| Johnson1985 | crossed cylinders, section 4.1, p. 86 (book) | 96 |
Rows with the printed strings: `literature/published/hamrock_brewe_1983_t1_t5.csv` (vault extraction of
2026-10-16 through the tag; Table 1 rows as in r02's published table).

## 3. Model
- **Alpha sweep:** two steel cylinders (`E` 210 GPa, `nu` 0.3) of radius 10 mm, body 1 curved along bridge x,
  body 2 the same cylinder turned by `theta = arccos((alpha - 1)/(alpha + 1))` (90 deg at alpha 1), 50 N, at the
  Table 1 alphas 1, 2, 5, 10, 25, 50, 100. The ellipse lies at `psi` from bridge x, along neither body.
- **Table 5:** the three contacts as printed (cm, N, `E'`) converted in `construction.py` (x 1e-2, x 1e4,
  `E = E'(1 - nu^2)` with `nu = 0.3`; only `E'` enters Hertz). The ball on the outer ring has body 2 concave in
  both directions (`R_x = -38.9 mm`, `R_y = -6.6 mm`).
- SlipPY BEM through `slippy_bridge.contact.solve` on a principal grid (`Grid.around(..., angle="principal")`:
  the ellipse axes on the grid lines; the bodies' profiles are turned by their angles to the grid), spatial
  kernel, half-length 2 semi-axes, `b/h` 16 and 32 asked (v104's rule), interference control (J-7) at r02's
  approach for the nominal load.

## 4. Public API
```python
check(result, *, published=None, case="v102") -> list[dict]   # a SOLVED point contact; v101's rows + E1, E2, V1 (+ T5)
orientation(result) -> float                                  # BEM major-axis direction from bridge x [rad]
literature.cases() -> [LiteratureCase(...)]                   # 7 Table 1 alphas + 3 Table 5 contacts
crossed_cylinders(alpha, *, R, load), table5() -> {application: contact}   # construction, unsolved
```

## 5. Inputs
As section 3; r02 at 50 N: alpha 1 `a` 0.148 mm (`a/R` 0.015) to alpha 100 `a` 0.883 mm, `b` 0.0485 mm
(`a/R` 0.088, the largest of the sweep, inside r02's soft limit 0.1).

## 6. Criteria (fixed before the run)
Judged only on the spatial kernel at `b_H/h >= 16`, converged, edge clear, Hertz reference in range (per-body
size ratio <= 0.1); otherwise RECORD. `M = 2 b_H / h`.
| id | quantity | tolerance | source of the tolerance |
|---|---|---|---|
| C1-C5 | v101's rows (approach 0.5 %; p0 `M^-3/2`; `a`, `b` `1/M`; load and `eps_p` RECORD) | as v101 | v101 README section 6 |
| E1 | `k = a/b` vs r02 | `2/M` | C4 on both semi-axes |
| E2 | orientation of the BEM major axis (largest second moment of `p`) vs r02 `alpha_rad`, mod pi | `h/a_H` rad | author's choice: one node at the end of the major semi-axis; RECORD on a circle. On a principal grid it tests that the bodies' profiles are turned as r02's `alpha_rad` says and that the right axis is the major one (exact by symmetry when right); a grid not aligned with the ellipse would be a stronger test |
| V1 | r02 per-body size ratio and governing body | RECORD | MG-08: record only |
| T1 | BEM `a/b` vs Table 1 printed `k` (alpha sweep) | `2/M` + half a unit of the printed digit | C4; the print |
| T5 | Table 5 `2b`, `2a` (smaller / larger printed diameter), `delta`, `sigma_max` (= p0), `k` from the BEM, taken to the printed load by the Hertz exponents (sizes, p0 `~ P^(1/3)`, approach `~ P^(2/3)`; under interference the BEM approach is the one that carries the load found) | v101 tolerance of the quantity (`2/M` for k) + half a unit of the printed digit | v101; the print |
| T5 r02 | the same from r02 at the printed load | RECORD | closed form vs print |

## 7. How to run
`python -m studies.v100_dry_contact.v102_hertz_ellipses.study` (about 2 min) writes `results/` (git-ignored):
`v102_bem.csv`, `v102_check.csv`, `v102_contact_relative_errors.png`,
`v102_contact_pressure_cut_ball_outer_ring.png`. `notebook.ipynb` runs the same, committed with its outputs.
Tests: `python -m pytest studies/v100_dry_contact/v102_hertz_ellipses` (about 3 s).

## 8. Result (run of 2026-10-28)
**Alpha sweep: PASS at every alpha 1-100, both grids (C1, C3, C4, E1, E2, T1). Table 5: ball on plane PASS;
wheel on rail MISS on 8 rows (a source error, below); ball on the outer ring RECORD (out of Hertz range).
138 judged, 130 PASS, 8 MISS.**
| alpha (actual b/h, 16 asked) | 1 (19.4) | 2 (31.2) | 5 (24.4) | 10 (20.6) | 25 (16.9) | 50 (29.2) | 100 (25.5) |
|---|---|---|---|---|---|---|---|
| approach vs r02 | -0.015 % | -0.002 % | -0.003 % | -0.003 % | -0.002 % | -0.001 % | -0.001 % |
| p0 vs r02 | +0.008 % | +0.004 % | +0.004 % | +0.005 % | +0.018 % | +0.002 % | +0.001 % |
| b vs r02 | -0.61 % | -0.30 % | -0.49 % | -0.24 % | +0.77 % | -0.32 % | -0.47 % |
| BEM a/b vs printed Table 1 k (T1) | 0 (1.0000) | +0.08 % (1.5858) | +0.32 % (2.8902) | +0.31 % (4.4994) | -0.76 % (7.9440) | +0.35 % (12.0711) | +0.49 % (18.1871) |
| orientation (E2) | (circle) | < 1e-14 rad | < 1e-14 | < 1e-14 | < 1e-14 | < 1e-14 | < 1e-14 |
At `b/h` 32 asked the semi-axis errors halve (T1 within 0.22 %).

**Table 5** (BEM at the printed load, `b/h` 16-56):
- ball on plane: `2a = 2b` +0.31 / +0.04 %, approach +0.06 %, `sigma_max` +0.19 %; r02 itself -0.07 % / +0.06 % /
  +0.17 % against the print (within 1.5 printed digits). PASS.
- **wheel on rail: MISS, and r02 misses by the same amounts**: diameters -15.8 %, approach +19.0 %, `sigma_max`
  +41 %, while `k` agrees (BEM -0.005 %, r02 +0.003 %). Cause (librarian and physics-reviewer, page image
  2026-10-28): the printed outputs are one Hertz contact of the printed F, `E'` and k but of an effective
  radius R = 31.41 cm, while the printed radii give R = 18.78 cm (`1/R = 1/R_x + 1/R_y`; 50.19 cm taken as
  R_y would give 31.41 cm). A source error: not a BEM or r02 defect. The criteria were fixed before the
  run, so the MISS stands (OPEN R2-07: José may turn these rows to RECORD with this reason).
- ball on the outer ring: per-body size ratio 0.145 (the ball governs; the groove 0.14) > 0.1, so RECORD
  (MG-08). r02 vs print within 0.07 %. BEM vs print: approach +0.29 %, `sigma_max` +0.15 %, `2a` -0.84 / -0.81 %,
  `2b` -0.49 / +0.65 %, k -0.3 / -1.4 % (`k` does not converge on refinement: the long thin ellipse at `b/h` 16-33).
  The BEM (exact torus gap) and r02 (paraboloids) differ by +0.23 / +0.24 % in the approach at both grids, a
  stable offset: the non-quadratic gap (profile) effect of a ball in a close groove, recorded.

## 9. Depends on
v101 (`87c3442`, `fc76048`: `check`, `reference`, `bem_load`, `pressure_cuts`, `PointContact`, `LiteratureCase`),
v104 (`fc76048`: the grid rule), r02 (`407273d`), `slippy_bridge.contact`, `studies/outputs`. AxisForge
`d34ec7f`, SlipPY `df5916c` (read only).
