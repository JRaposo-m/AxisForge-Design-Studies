# s104 tip edge zone

## 1. Question
On the rigid L0 path A..E of the FZG C40 spur pair (K9) with sharp, unmodified tips, over what length near A and E is
the contact an edge contact, and how do the converged integral measures of the BEM pressure (centroid offset, width,
mean pressure, load within `b` of the edge) depart from Hertz there? `p0` is recorded and flagged, never a result:
next to a sharp edge it is singular and grows with the grid.

## 2. Why
SB-65 / SB-70: the bridge reports the edge (`end_distances`, `HertzResult.ends_over_b`, `valid`) but does not refuse
it; v107 records `d_end/b` and excludes `d_end/b < 1`. s104 measures the extent and size of the effect, which has no
published reference (librarian NOT FOUND), so it cannot be a v case (J-8).

## 3. Inputs / sweep
- FZG C40 (v107 `build_pair("C40")`: z 16/24, m 4.5 mm, b 40 mm, x 0.1817/0.1715; E 206 GPa, nu 0.3 assumed, R-118);
  K9 wheel torque 323.27 N m (Martins2015 Table 4, on the wheel, R-124); pinion driving (Q-09); `F_bn` from AxisForge
  `mesh_forces`, shared by L0 (`load_sharing`: 1/2 near A and E).
- Points: near A (edge: tip of gear 2) and near E (edge: tip of gear 1), at `d_end/b` = 0, 0.25, 0.5, 0.75, 1, 1.5, 2,
  3, 5, 10 (`b` the Hertz half-width of the point; `g_y` found by `brentq`), and C (no edge).
- Grids: `b/h` asked 16, 32, 64 (h, h/2, h/4). `Grid.around` floors `h` to a power of two, so the actual `b/h` lies
  in [asked, 2 x asked): 16.1-31.9 / 32.1-63.9 / 64.2-127.8 (OPEN R5-118); the ratio between levels stays exactly 2.
  Domain half-length, the same at every level of a point: `max(10, 2 x 0.23 M^0.75)` half-widths at the `M = 2 b/h`
  actually solved on the finest grid (R2-08 line rule with a margin 2, so the image error 0.0514/span^2 is a quarter
  of v103's C3 tolerance; OPEN R5-111): 17.6-29.4 `b`. 16 nodes along the periodic axis; SlipPY frequency kernel,
  load control (as v103).
- Near A the domain also reaches past gear 1's root end on -x (`d_end_root_1/b` 4.7-6.7 < span): the bridge puts
  `SAG_PAST_END` there too (an upper bound of the flank at the base circle, SB-66); that region is unloaded.
- Not swept (OPEN): K5 (R5-112), the h/2-shifted grid (R5-113), the other driver (the same contacts with A and E
  swapped; R5-114), H501 (helical slice ends oblique, SB-67; R5-115).

## 4. Model (`model.py`, public API in `__init__.py`)
```python
fzg_c40(stage="K9") -> dict                 # pair, torque, F_bn_N, path (g_y of A..E), data (construction.py)
contact_at(system, g_y_mm, label)           # bridge CylindricalGearContact at a path point
contacts_near_end(system, end, targets)     # [(d_end/b target, g_y, contact)] near "A" or "E"
edge_of(contact, end) -> dict               # governing tip: name, d_end_m, signed x_edge_m on bridge x
measures(result, x_edge_m, b_m) -> dict     # integral measures of one BEM solve (model docstring)
hertz_measures(x_edge_m, b_m) -> dict       # the same measures of Hertz without the edge
grid_series(contact, x_edge_m, output_dir)  # three grids, richardson (flat_atol 0, max_p 2), finest result
```
The flank of each gear stops at its tip circle; past it the bridge sets the surface `SAG_PAST_END` = 1 mm below (a
sharp corner on a half-space: a step, not a wedge; the free-edge compliance of the tooth is not modelled, OPEN SB-65).
Measures from the pressure profile across the line `p(x)` (the mean over the periodic axis): centroid offset `x_c/b`
(+ away from the edge), equivalent width `b_eq/b` (`b_eq = 2 sqrt` of the second moment; 1 for Hertz), mean pressure
`p_m/p_m,H = b/b_eq`, edge share (load within `b` of the edge over `w`, against Hertz's share of the same window).
Departure = the largest of `|b_eq/b - 1|`, `|x_c/b|`, `|edge share - Hertz|`. Two more definitions were **added after
the run (R5-116)**: `departure_edge` (without `x_c/b`) and `departure_baseline` (`x_c/b` minus its value at `d_end/b`
= 10 at the same end, which removes the curvature-gradient offset and keeps the most sensitive measure).
The field is piecewise constant per node (SlipPY BEM): its material edge is at the last on-flank node + h/2, not at
`x_edge_m`, so the edge window and the `singular` flag jitter by one node with `h` (the oscillatory Richardson
judgements; K1 is not affected).

## 5. v first
- v107 `check(pair, 323.27, data=PAIRS["C40"])`: the pair, its path points and forces against r04 at K9.
- v103 `check(result)`: every point with `d_end/b >= 3` (a plain line contact, v103's own conditions: frequency kernel,
  `b/h >= 16`, half-length >= 0.23 M^0.75 (R2-08)) on its finest grid. Every point is judged at 2 x that rule (Sec. 3);
  v103 also assumes constant curvature, which near A does not hold (R5-116).
- v104 (smooth contacts only; the edge refinement series is s104's own).

## 6. Criteria (fixed before the run)
| ID | Quantity | Tolerance | Why |
|---|---|---|---|
| K1 | change of each measure (`x_c/b`, `b_eq/b`, edge share) from `b/h` 32 to 64, every point | <= 0.01 (absolute, dimensionless) | proposals s104 "< 1 %"; the edge pressure ~ r^(-1/2) is integrable, so the integrals converge |
| K2 | the measures at C and at `d_end/b` = 10 equal Hertz | departure <= 0.01 | BEM = Hertz for `d_end/b >> 1` |
| V | v107 judged rows; v103 judged rows of the far points | their own tolerances | validation of the system at its conditions |
| O | `d_end/b` (and mm) at 1 % and 5 % departure, per end; also for `departure_edge` and `departure_baseline` (added after the run, R5-116) | RECORD | onset of the geometric edge effect only (no tooth compliance) |
| P | `p0/p0_H` per grid, `singular` flag (peak on the node next to the edge) | RECORD | not a result |
Richardson (`numerical_solvers.richardson`, `kind="size"`, `flat_atol=0`, `max_p=2`): status, observed `p`, GCI recorded
for every measure and point.

## 7. How to run
```bat
python -m studies.s100_dry_contact.s104_tip_edge_zone.study
python -m pytest studies\s100_dry_contact\s104_tip_edge_zone
```
Results in `results/` (git-ignored) and copied to `Vault/80_Inbox/2026-10-28/r5_studies/s104/`.

## 8. Risks
- Half-space with a step, not a wedge: the free-edge compliance of a real tooth tip is not modelled; the measures are
  the geometric edge effect only.
- L0 rigid sharing: no corner contact before A / after E (needs LTCA, T3); the share is 1/2 at every point here.
- SB-66 (root ends at the base circle, not the true root form), SB-71, Q-09 (driver).
- The R2-08 line rule was set on smooth contacts (R5-111); the span is 2 x the rule at the actual `M` (17.6-29.4 b).
- Effort M.

## 9. Result (run of 2026-10-28, session 2: span 2 x R2-08 rule)
- **Scope:** rigid-geometry edge zone (L0, share 1/2 held to the ends, sharp half-space step). The loaded extension of
  the path (corner contact before A / after E, tooth deflection about 6 um vs `b` 58-83 um) and the share ramp are not
  included (LTCA, T3); they act at the same scale as the result.
- **Edge effect (finest grid):** the measures meet Hertz from `d_end/b` = 1 on. Inside, the zone narrows and its load
  moves away from the tip: at `d_end/b` = 0.75 / 0.5 / 0.25 / 0 the width `b_eq/b` is 0.96 / 0.86 / 0.75 / 0.65 and the
  centroid `x_c/b` 0.04 / 0.11 / 0.24 / 0.39. A and E agree within 0.01; near A the curvature-gradient shift (about
  0.01 b, R5-116) has the SAME sign as the edge shift, so A's `x_c/b` includes it (0.389 at A vs 0.384 at E, `d_end/b`
  0). Onset (`departure_edge`, and the same with `departure_baseline`): 1 % at `d_end/b` 0.98 (A, `d_end` 0.057 mm of
  arc on gear 2's tip flank) / 0.96 (E, 0.079 mm on gear 1's); 5 % at 0.72 (A, 0.042 mm) / 0.74 (E, 0.061 mm). Along the
  path (`d_end/b` grows about 10 per mm of `g_y` near A, 8 near E) the edge zone `d_end < b` is the first 0.10 mm of
  path length `g_y` after A and the last 0.12 mm before E (of 19.4 mm).
- **Departure with the centroid (README 6 definition):** reaches 1 % already at `d_end/b` 3.9 near A, but not from the
  edge: far from it (`d_end/b` 2-10) `x_c/b` stays at 0.008-0.011 near A (0.002 near E, 0.001 at C). Near A the contact
  lies 4.7-6.7 `b` of ARC LENGTH from the base circle of gear 1 (its root end, SB-66); there rho1 = 4.3 mm (74 b) and
  d rho1/ds = r_b1/rho1 = 7.9, so rho1 varies by about 11 % across the contact and shifts the load (OPEN R5-116).
  All three definitions are in `s104_onset.csv`.
- **K1 (grid):** 21/21 PASS; largest change of a measure from h/2 to h/4 0.0045 (edge share, E 0.25). Richardson
  (`kind="size"`, `flat_atol=0`, `max_p=2`, r = 2): of 63 judgements 27 monotonic, 26 oscillatory, 9 flat, 1
  non-convergent (`x_c/b` at E 2, change 5e-5); oscillatory = changes of 1e-5 to 3e-3 (edge window snapping, Sec. 4).
  Observed `p` 0.86-1.04 next to the edge (monotonic `x_c`, `b_eq` at `d_end/b` 0, 0.5, 0.75; GCI 0.001-0.035), 2 (=
  max_p) far from it. With p near 1 the error left in the finest level is about the h/2 -> h/4 change, so K1 bounds
  it. `p0/p0_H` at `d_end/b` 0 (A) grows 6.5 -> 9.2 -> 13.0 (about `sqrt(2)` per halving: the `r^(-1/2)` corner);
  `singular` on every point with `d_end/b < 1`, never far from it.
- **K2:** `departure_edge` at C 0.0002 and at `d_end/b` 10 0.0004 (A) / 0.0001 (E): PASS. With the centroid: 0.0009 (C)
  / 0.0079 (A) / 0.0020 (E): PASS (<= 0.01).
- **V:** v107 at K9: G, P, F rows PASS (H RECORD by v107's design). v103 on the far points (span 2 x R2-08): C4 (`b`) 7/7
  PASS; C3 (`p0` within `M^-1.5`) PASS at C (1.8e-4 vs 6.9e-4), at E 3/5/10 (1.2-1.4e-4 vs 4.2-4.3e-4) and at A 10
  (4.8e-4 vs 6.5e-4); MISS at A 3 (5.7e-4 vs 2.5e-4) and A 5 (4.9e-4 vs 2.4e-4). First session-2 run at 1 x the rule
  (span 10-14.7): 6 MISS, E 4.3-4.6e-4 just over 4.2-4.3e-4, because at the rule the image error equals the tolerance
  (OPEN R5-117: the session-1 "kernel floor" was the R2-08 image error 0.0514/span^2 at span 10, below the rule at the
  actual `M`). The remaining p0 error near A (about 5e-4, against 1.2-1.8e-4 at E and C, image part 6e-5) is put on the
  curvature gradient near gear 1's base circle (R5-116): those points lie outside v103's constant-curvature conditions.
  The measures of s104 are integrals and do not use `p0`.
- Runtime: 63 BEM solves, about 150 s.

Figures (`results/`, copied to OUT/s104): `s104_contact_pressure_cut_near_A` (BEM `p/p0_H` across the line near A at
`d_end/b` 0, 0.5, 1, 3 and C against Hertz: the tip cuts the ellipse and the peak grows at the edge; from 1 on the BEM
lies on Hertz), `s104_bearing_loads_load_zone_sweep_measures` (width, centroid and edge share against `d_end/b`, A and
E, Hertz limits 1 and 0, the `d_end = b` mark: the departure starts below 1), `s104_bearing_loads_load_zone_sweep_grid`
(the measures and `p0/p0_H` on the three grids for A 0, 0.5, 3 and C with the Richardson extrapolation: integrals flat,
`p0` diverging next to the edge). The figure files carry `bearing_loads_load_zone_sweep` because they reuse that plot
model (D-0018); they show gear contacts.

## 10. Depends on
v107 (`build_pair`, `PAIRS`, `WHEEL_TORQUE_NM`, `DRIVER`, `E_MPA`, `NU`, `FAMILY`, `check`: imported, OPEN R5-110), v103
`check`, `slippy_bridge` (`CylindricalGearContact.from_pair`, `Grid.around`, `solve`, `hertz`), SlipPY (frequency
kernel), `numerical_solvers.richardson`, `studies/outputs` (`l000_axisforge/plots/bearing_loads.py::load_zone_sweep`
with the new `filled_sources`, `l100_dry_contact/plots/contact.py::pressure_cut`). AxisForge `d34ec7f`, SlipPY
`local-fixes` (read only); design-studies `layout/studies-levels` `103c045` + this case (uncommitted at the run).
OPEN: R5-110 (v107 import), R5-111 (span rule on an edge contact), R5-112 (K5 not run), R5-113 (h/2-shifted grid
not run), R5-114 (other driver not run: same contacts, A and E swapped), R5-115 (H501 not run, SB-67), R5-116
(curvature gradient near the base circle; onset definitions added after the run), R5-117 (v103 C3 MISS at span 10 =
R2-08 image error at the actual `M`; span now 2 x the rule; A 3 / A 5 still MISS, curvature gradient), R5-118 (dyadic
`h`: actual `b/h` in [asked, 2 x asked), 16.1-127.8).
