# v104 BEM discretisation

## 1. Question
Which grid spacing `h`, domain half-length and SlipPY kernel (spatial or frequency) give a stated error in the
approach `delta` and the peak pressure `p0` of a smooth Hertz point contact solved through `slippy_bridge`?
One validation: **the BEM discretisation on smooth Hertz contacts against r02, no edge series**
(v_prerequisites row 2). Output: the grid rule this repository uses (v102, v103, R2 Part C, s101, s104).
Ported from the mesh and domain parts of `v100_dry_contact/v103_hertz_point_grid_domain` (tag
`studies-pre-sweep`, its `01_ball_flat` and `03_kernel`).

## 2. Sources
| key | item | PDF page |
|---|---|---|
| PolonskyKeer1999 | Sec. 4.1: `eps_r = abs(sqrt(A/pi) - r_H)/r_H`, `A = N_c h^2`; `eps_p = sum abs(p - p_H) / sum p_H` over the contact nodes; "eps_r < M^-1 for all M considered" (M = 64-1024 grid rows, grid slightly larger than the contact diameter); eps_p "on the order of M^-3/2", growing slightly faster | 10 (Fig. 1 on 9, plotted only) |
| Muser2014 | "Delta_a/a_c = 1/32 and a_c/L = 1/8 ... within less than 0.1 % error if the contact area is determined through a fit of the radial pressure profile" (GFMD, periodic cell L) | 7 |
| Johnson1985 | Hertz, through r02 (v101) | 102-103 |
Rows with pages: `literature/published/polonsky_keer_1999_muser_2014.csv` (vault extraction 2026-10-16,
through the tag; checked against the PDFs by the librarian, `_evidence/v101_librarian.md`).

## 3. Model
v101's ball on a flat (circle) and an ellipsoid of radii (10 mm, 50 mm) on a flat (`R_y/R_x` 5, `b/a` 0.346,
minor axis along grid x), steel, 500 N, **load control** (the errors read at a fixed load). Grids from
`Grid.around(contact, hertz, h=b/(b/h), span_x=span, span_y=span)`, `h` rounded down to a power of two in
metres (so `b/h` in [asked, 2 x asked)), the span in Hertz semi-axes on each axis. Kernels: spatial
(zero-padded, no images) and frequency (periodic with period 2 n h, images; bridge default).
- h sweep: `b/h` asked 4, 8, 16, 32, 64; half-length 2 (spatial) and 4 (frequency, the bridge's `SPAN_X_0`).
- Domain sweep: half-length 1.25, 1.5, 2, 4, 8 at `b/h` 16 asked.
- Muser grid (circle): `b/h` 32 asked; frequency kernel at half-length 2 (period 2 n h = 8a, his `a_c/L = 1/8`),
  spatial kernel at half-length 4 (grid 8a; no cell, an analogy).

## 4. Public API
```python
check(result, *, case="v104") -> dict        # one discretisation record of a SOLVED point contact on its grid
judge(records) -> list[dict]                  # D1, L1-L3 over the sweep records (one schema)
literature.cases() -> [LiteratureCase(...)]   # Polonsky-Keer, Muser (rows with pages)
systems() -> [ball_on_flat(), ellipsoid_on_flat()]
```
`check` uses v101's `reference`, `bem_load`, `pressure_l1_error` (OPEN R2-02).

## 5. Inputs
As section 3. SI. Hertz (r02): circle `a = b` 0.27430 mm; ellipse `b` 0.23734 mm, `a` 0.68595 mm.

## 6. Criteria (fixed before the run)
| id | quantity | tolerance | judged where |
|---|---|---|---|
| D1 | change of `delta` and `p0` against the largest domain (8) | 0.05 % | spatial kernel, both systems (old case P6) |
| L1 | Polonsky-Keer `eps_r < 1/M`, `M = 2b/h` | `1/M` (printed) | circle, each kernel, `M >= 64` (their range); below RECORD |
| L2 | order of `eps_p` between the two finest grids | RECORD | no constant printed |
| L3 | Muser: area `pi a b` from a fit of `p^2 = p0^2 (1 - x^2/b^2 - y^2/a^2)` | 0.1 % (printed) | circle, `b/h` 32 asked; frequency kernel at period 8a (judged; set to Muser's cell after the physics review, tolerance unchanged), spatial RECORD |
| - | every error of the sweeps; the frequency kernel's domain change | RECORD | |
The grid rule is a result, not a criterion: the coarsest `b/h` of the h sweep from which `abs(err)` stays within
0.5 % and within 0.1 % on every finer grid, in the ACTUAL `b/h` of the grids solved.

## 7. How to run
`python -m studies.v100_dry_contact.v104_bem_discretisation.study` (about 4 min; the frequency kernel at
`b/h` 124 on the ellipse takes 80 s) writes `results/v104_{check,judged,grid_rule}.csv` and
`v104_contact_relative_errors_{h,span}.png` (`v104_judged.csv` holds D1, L1-L3). `notebook.ipynb` runs the same, committed with its outputs.
Tests: `python -m pytest studies/v100_dry_contact/v104_bem_discretisation` (about 2 s).

## 8. Result (run of 2026-10-28)
**PASS: D1 (8 rows; spatial kernel: the change is 0 to the printed digits, it does not see the domain), L1 at
M 72 and 144 (both kernels). MISS: L3 on the frequency kernel at Muser's period 8a, area -0.244 % (limit
0.1 %; RECORD on the spatial kernel +0.014 %).** The L3 miss: Muser's 0.1 % is for a GFMD periodic cell whose
reference is his own periodic solution; here the reference is the isolated Hertz contact, so the image loads
of SlipPY's frequency kernel at period 8a (the same effect as the approach error below) enter the area.

| contact | kernel, half-length | actual b/h | delta | p0 | b | eps_r | eps_p | area fit |
|---|---|---|---|---|---|---|---|---|
| circle | spatial, 2 | 4.5 / 9.0 / 18 / 36 / 72 | -0.32 / -0.012 / +0.004 / +0.011 / +0.012 % | +0.13 / +0.10 / +0.035 / +0.021 / +0.018 % | -1.3 / +0.45 / +0.26 / +0.17 / +0.14 % | 4.3 / 1.4 / 0.87 / 0.10 / 0.07 % | 2.3 / 1.0 / 0.43 / 0.13 / 0.06 % | -1.7 / +0.32 / +0.09 / +0.014 / -0.013 % |
| circle | frequency, 4 | same | -1.08 / -0.91 / -0.94 / -0.95 / -0.95 % | -0.05 / +0.14 / +0.037 / +0.046 / +0.047 % | -0.79 / +0.50 / +0.27 / +0.18 / +0.14 % | 4.3 / 1.4 / 1.3 / 0.20 / 0.07 % | 2.0 / 0.87 / 0.40 / 0.14 / 0.09 % | |
| ellipse | spatial, 2 | 7.8 / 16 / 31 / 62 / 124 | -0.023 / -0.006 / +0.001 / +0.002 / +0.003 % | +0.059 / +0.012 / +0.007 / +0.005 / +0.005 % | +2.2 / -0.57 / -0.13 / -0.15 / -0.11 % (a: -0.47 / +0.14 / +0.11 / +0.05 / -0.02 %) | 0.17 / 0.46 / 0.32 / 0.14 / 0.07 % | 0.67 / 0.29 / 0.11 / 0.04 / 0.02 % | |
| ellipse | frequency, 4 | same | +0.40 / +0.49 / +0.54 / +0.57 / +0.57 % | -0.04 / -0.003 / +0.006 / +0.004 / +0.004 % | +2.9 / -0.48 / -0.15 / -0.19 / -0.19 % | 0.17 / 0.64 / 0.34 / 0.16 / 0.07 % | 0.57 / 0.31 / 0.17 / 0.14 / 0.13 % | |

Frequency kernel against the domain (actual b/h 18 circle, 31 ellipse; RECORD): approach circle -2.28 / -2.11 /
-1.73 / -0.94 / -0.48 % at half-length 1.25 / 1.5 / 2 / 4 / 8; ellipse +0.86 / +0.85 / +0.78 / +0.54 / +0.30 %.
From half-length 2 to 8 the error falls about as 1/span, as Muser's c/r displacement of a periodic cell implies
(Muser2014 Eq. (29), PDF p. 7: the mean displacement converges slowly with periodic boundaries); between 1.25
and 2 it is nearly flat. Its sign differs between the two systems; the cell is square for the circle and 2.9:1
for the ellipse (span counted in semi-axes on each axis), and both the lattice sum and SlipPY's zero-frequency
term depend on the cell's aspect ratio: a likely cause, not verified. p0 is hardly affected (<= 0.05 % at
half-length >= 4). L2: the two-point order of `eps_p` (1.12 spatial, 0.63 frequency) is noisy (edge nodes);
the magnitude meets Polonsky and Keer's `M^-3/2`: spatial 0.13 % at M 72 (0.16 %) and 0.06 % at M 144 (0.058 %).

### Grid rule (this repository; tested scope below)
| kernel | domain half-length | `abs(err delta)`, `abs(err p0)` <= 0.5 % | <= 0.1 % | semi-axes |
|---|---|---|---|---|
| **spatial (default)** | >= 1.25 semi-axes tested (the domain must contain the contact; no images) | actual `b/h >= 4.5` (circle), `>= 7.8` (ellipse: none coarser solved) | actual `b/h >= 9.0` (circle: p0 +0.10 %, at the limit), `>= 7.8` (ellipse) | `b` within 0.6 % from actual `b/h` 16 (ellipse), 0.26 % (circle) |
| frequency | 4 (`SPAN_X_0`) | p0 from actual `b/h >= 4.5`; approach not at half-length 4 (-0.95 % circle, +0.57 % ellipse); half-length 8: -0.48 %, +0.30 % | p0 from actual `b/h >= 18` (circle) | as spatial (table) |
Scope: two smooth contacts (circle, `R_y/R_x = 5`), steel on steel, 500 N, load control, ellipse axes on the grid
lines (`psi = 0`), quantities `delta`, `p0`, `b` (not traction, not subsurface). With the power-of-two rounding, an
asked `b/h` gives an actual one in [asked, 2 x asked): ask `b/h >= 16` (actual 16-32) for 0.1 % with margin.
**For R2: spatial kernel, `b/h >= 16` asked, half-length 2 for smooth contacts**; a rough contact needs a finer
`h` set by its roughness (R2 Part C), not by `b`.

## 9. Depends on
v101 (`87c3442`: `reference`, `bem_load`, `pressure_l1_error`, `ball_on_flat`, `PointContact`, `LiteratureCase`),
r02 (`407273d`), `slippy_bridge.contact`, `studies/outputs` (`relative_errors`). AxisForge `d34ec7f`, SlipPY
`df5916c` (read only). Not covered: edge contacts (no edge series, by the prompt), line contacts (v103), subsurface.
