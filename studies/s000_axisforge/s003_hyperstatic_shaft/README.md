# s003 hyperstatic shaft on three ISO/TS 16281 bearings

## 1. Question
On a shaft with three ball bearings, how much do the reactions, the slope and deflection at the gear and Q_max move
from rigid supports (ladder steps 1-2) to ISO/TS 16281 bearings solved together with the shaft by Newton (step 3), as
a function of a radial seat offset e of the middle bearing and of the diametral clearance s?

## 2. Why
The step that makes the system solve iterative (thesis concept v3, ladder step 3); every later GRC step reuses it.
On three supports the reactions depend on how much each bearing yields (unlike s004's determinate shaft, where only the
kinematics change). proposals.md s003 (rank 8, T2).

## 3. Inputs / sweep
Shaft: v005's (`three_support_shaft`: uniform d 30 mm, S355, L 270 mm, zero-width pins at 10 (locating), 110, 260 mm;
`# OPEN R5-162`: v005's construction, condensed beam `study.hertz_problem` and settled solve `study.solve` used by an s,
as s004 R5-54). Spur gear force P = 2000 N along -Y at x = 60 mm (middle of span 1; `# OPEN R5-164`: the load level is
the case's own: reactions about 1 kN, delta_r about 12 um, of the order of the shaft's own deflection). One plane (XY),
no couple, no J-16 tag. Bearings: three s001 6210-like deep groove ball bearings (`# OPEN R5-163`: 50 mm bore on a 30 mm
shaft, fictitious assembly as s004 R5-51). Folder `s003_hyperstatic_shaft` (`# OPEN R5-160`: no s003 at the tag
`studies-pre-sweep`; name from proposals.md).
| swept | values |
|---|---|
| middle seat offset e (along +Y) | -70 ... +50 um, step 10 (extended from -50 at review: both zero crossings of R_2 inside) |
| diametral clearance s (all three, R-105) | 0, 10, 20, 30 um |
| limits | linear law k = 3e4 N/mm (e -50, 0, 50 um); bearing force x 1, 1e2, 1e4, 1e6 at s = 0 |
Euler-Bernoulli, 54 uniform elements (h 5 mm; supports and gear on nodes, Hermite nodal exactness, v001 / v005).
CRB middle bearing (proposals "RECORD row") not run (`# OPEN R5-168`: the line-contact sums would be a second
D-0018 exception; next run).

## 4. Model
`model.py` (exported; a later study imports it):
```python
ball_loads(bearing, delta_r_mm) -> (Fr_N, Q_j_N)            # ISO/TS 16281 Eqs (12)-(16) at a PRESCRIBED delta_r, Fa = 0
solve_shaft(problem, bearings, *, seat_mm=None, scale=1.0, k_linear_N_per_mm=None, label="") -> {newton, F_N, d_mm, u_mm, Q_j_N, secant, settlements_mm, result, ...}
```
- **Beam:** AxisForge's K and f on the mesh, condensed EXACTLY on the transverse DOFs u of the bearing nodes (v005
  `hertz_problem`, extended in R5 to return `K_c`, `f_c`; v005 N3 / N4 showed this condensation exact to 1e-11).
- **Bearing law:** F_k = -sign(d_k) Fr_k(|d_k|) on the shaft along +Y, d_k = u_k - e_k the shaft centre relative to its
  seat. Fr(delta_r) = sum_j c_p delta_j^1.5 cos(alpha_j) cos(phi_j) with AxisForge's `ISO16281BallSolver.elements` at
  delta_a = -A sin(alpha_0) (with Fa = 0 and untilted rings Eq. (17) holds there exactly: every alpha_j = 0, v003), so
  delta_j = max(delta_r cos(phi_j) - s/2, 0). The sums are written in `model.py` (proposals.md: a D-0018 exception,
  AxisForge has no displacement-to-force function). The minus sign is physics: the shaft centre moving toward +Y
  compresses the +Y balls, which push it back (s004 `DISPLACEMENT_SIGN = -1`).
- **Solve:** g(d) = K_c (d + e) - f_c - F(d) = 0 [N] by `numerical_solvers.newton` (central FD Jacobian, line search,
  `lm` fallback, residual scaled by P, `residual_abs` 1e-10 = 2e-7 N, `step_rel` 1e-13). Unknowns d, not u, so a stiff
  bearing's tiny d is not lost to round-off of u = d + e. Start: linear springs at the secant stiffness of P/3, each d
  moved out by s/2 so every bearing touches (inside the clearance the law has zero stiffness; at s = 0 and d = 0 too).
  `# OPEN R5-161`: proposals.md sketches a force method (two bearings rigid, the third's reaction fed back as a
  `RadialLoad`, n + 1 FEM solves per iteration); the condensation is the same equilibrium in one assembly.
  `# OPEN R5-166`: no continuation in e was needed (every solve converged from its own start).
- **Rigid reference:** `RigidSupportFEMSolver` on the three pins, the middle one settled by e (v005 `study.solve`).
- **Assumptions:** one plane (d a signed scalar, phi never undefined; two planes need the vector law, `# OPEN R5-165`);
  rings untilted in the law (the slope at each bearing is recorded, not fed back; s004 found tilt coupling large in
  the kinematics); rigid housing; no bearing moment fed back (s005). Cage: `phi_j` is measured from the CURRENT load
  line with one ball on it (ISO/TS 16281 as AxisForge codes it), so the cage turns with the load direction; a cage
  fixed in space would, for odd Z = 9, see a reversed load half a pitch off (RECORD `cage_half_pitch_Fr_change`).
- **Conventions, `convention-dependent (R-003)`.** F and `ShaftResults.R_xy` are the force of the bearing ON the shaft
  (fed back as +R, v005 F0b). Each loaded bearing is also solved force-controlled by AxisForge at that force (s004
  `ring_displacement`, node force = support force on the shaft): its loaded zone sits 180 deg from the push of the
  shaft, and its tilt projection psi . R_hat (recorded as `psi_in_axisforge_mrad`) has the opposite sign of the tilt
  seen from that zone: physically the tilt is -psi_in (bearing-expert, s004 review S1). Per bearing the zone centre is
  recorded both ways (D-0010 theta from +Y toward +Z): `zone_theta_deg` physical (0 if d > 0, 180 if d < 0: the balls
  on the side the centre moves toward) and `zone_theta_axisforge_deg` / `phi_Fr_axisforge_deg` (AxisForge's node
  force +R); they differ by 180 deg in every row, which is R-003 made explicit. Local azimuths (load zone figure) and
  the signed law are convention-free.

## 5. v first
Through the v's checks at s003's own conditions, never an r directly:
- **v005.check** of every converged coupled shaft, with the secant springs k_k = |F_k / d_k| of the converged state
  and the seat offsets as spring-base settlements (an unloaded bearing: k = 1e-6 N/mm, r01 needs k > 0; 3e-8 N);
  v005 (r01 extended) judges the field and the reactions at its TOL_NODAL 1e-9.
- **v003.check** of every loaded bearing, solved by AxisForge (s001 `solve_bearing` via s004 `ring_displacement`) at
  its coupled force, psi = 0 (r03; deep groove judged).
- v001 / v002 enter through v005 (same comparator family).

## 6. Criteria (S0-S5, S7 fixed before the first run, proposals.md s003; S6, S8, S9 added at review)
| id | criterion | tolerance | source |
|---|---|---|---|
| S0 | every coupled Newton solve converged (rows used only if converged) | all | proposals.md |
| S1 | `ball_loads` at the `solve_contact` (tol 1e-12) delta_r gives Fr; delta_a - (-A sin alpha_0) RECORD | 1e-9 rel | proposals.md |
| S2 | v005.check of every coupled shaft (secant springs): aggregate of v005's verdicts (number of MISS) | 0 MISS (v005 TOL_NODAL 1e-9) | v first |
| S3 | linear law (k 3e4 N/mm) through the same Newton: v005.check, and forces vs v005 `study.solve` | 1e-8 rel | proposals.md "v005 closed form" |
| S4 | sum F = P, sum F (x - x_1) = P (x_P - x_1), AxisForge R_xy of the coupled field = F | 1e-8 rel | proposals.md |
| S5 | sign of s004 `ring_displacement` d (force-controlled AxisForge solve, the same -1 sign) = the model's d: sign consistency with s004 (R5-58); delta_r is invariant under R -> -R, so the R-003 azimuth is NOT tested here | exact | consistency |
| S6 | stiff limit, s = 0, scale 1 ... 1e6: forces and d vs the exact power-law judge (`hertz_judge`), S6v at e = 0 vs v005 `solve_hertz` | 1e-8 of P / of max abs(d) | **added at review** |
| S6r | the same solves vs the rigid three-support FEM, slope vs scale (-2/3 expected) | RECORD | R5-167 |
| S7 | v003.check of every loaded bearing: aggregate of v003's verdicts (number of MISS) | 0 MISS (v003 1e-5; c_p 5e-3) | v first |
| S8 | e = 0, s >= 10 um (bearing 3 inside its clearance): F = (P/2, P/2, 0), d_3 - d_1 = P l1^2 l_ov / (16 EI) | 1e-8 | **added at review** |
| S9 | every s = 0 point: forces and d vs the exact power-law judge | 1e-8 of P / of max abs(d) | **added at review** |
| Z, Sc | e at R_2 = 0 (rigid, coupled); the clearance s_c above which bearing 3 is unloaded at e = 0 | RECORD | physics review |
| R | reactions rigid / coupled, d, d / (s/2), Q_max, n_loaded, unloaded, slopes at bearings and gear, psi_in, zone azimuths, cage half-pitch effect, u - rigid line | RECORD | - |
**S6, S8, S9 were added at review, after the first run** (physics review MUST 1, 2; as R5-104): they are exact judges
(at s = 0 the ISO law IS v005's Hertz law F = K abs(d)^1.5, K = scale c_p sum_{cos > 0} cos^2.5 phi_j = 7.71e5 N/mm^1.5
at scale 1, because delta_j = abs(d) cos phi_j; S8 is two-support statics), so their tolerances are not tuned to a
result, but they are not pre-run criteria. The first run's "1e-4 of P at scale 1e6" for the stiff limit (read from a
prototype, `# OPEN R5-167`) is dropped; the distance to the rigid FEM stays RECORD (S6r). Scale 1e9 is not run: the
curvature-centre kinematics sqrt(U^2 + V^2) - A loses digits at delta / A ~ 1e-8 (R5-169).

## 7. How to run
`python -m studies.s000_axisforge.s003_hyperstatic_shaft.study` (about 13 s) writes `results/` (git-ignored):
`s003_law.csv`, `s003_points.csv` (per bearing and (s, e); gear rows `bearing = 0`), `s003_v005.csv`, `s003_v003.csv`,
`s003_limits.csv`, `s003_criteria.csv`, `s003_newton_history.csv`, `s003_elements.csv`, figures
`s003_shaft_ratio_sweep_reactions_s{0,30}um.png` (R_i / P vs e, ISO bearings vs rigid; residual panel: s = 0 vs the
power-law judge, s = 30 um v005.check C3), `s003_shaft_ratio_sweep_clearance.png` (d / (s/2) of bearings 2, 3 vs e,
clearance band shaded, S8 closed form), `s003_convergence_newton_history.png`,
`s003_bearing_loads_element_loads_middle_e0.png` (Q_j of the middle bearing vs the zone shape, local azimuth),
`s003_shaft_deflection_along_shaft_s20um_e+50um.png`, `s003_contact_relative_errors_stiff_limit{,_judge}.png` (vs the
rigid FEM with the -2/3 guide; vs the power-law judge with the 1e-8 band).
Tests: `python -m pytest studies/s000_axisforge/s003_hyperstatic_shaft`.

## 8. Risks
Non-smooth law at contact onset (delta^1.5: C1, central FD fine) and inside the clearance (zero stiffness: a shaft on
one touching bearing is a mechanism; the start puts every bearing in contact). A sign error is not cancelled (the
reaction on the shaft is opposite to the force on the ring): S4 (AxisForge R = F) and S5 (AxisForge's own bearing solve
gives the same d, sign consistency only: it cannot see R-003) guard it. Tilt not in the law; one plane only; rigid
housing; the lift-off of bearing 3 at s = 10 um holds by 0.3 um (R5-181).

## 9. Result (run of 2026-10-28, after the review fixes)
**1577 PASS, 0 MISS, 21 RECORD.** S0 52/52 coupled solves converged (Newton 3-5 iterations, quadratic, no fallback); S1
12/12 (worst 2.3e-14); S2 0 MISS in 156 v005 records (worst 1.9e-10); S3 6/6 (6.9e-11); S4 156/156 (1.4e-10); S5
148/148 loaded bearings (abs(d) of the force-controlled AxisForge solve within 5e-7 um of the model); S6 24/24 and S6v
4/4 (6.7e-11); S7 0 MISS in 1036 v003 records (worst 6.5e-4, the c_p row, tol 5e-3); S8 6/6 (1.5e-10); S9 26/26
(1.2e-10). S6r RECORD: 0.26 / 1.6e-2 / 7.8e-4 / 3.6e-5 of P at scale 1, 1e2, 1e4, 1e6 (e +50 um), slope -0.603, -0.662,
-0.666 (expected -2/3).
- **The bearings move the load.** e = 0, s = 0: R = 952 / 1080 / -32 N (rigid 850 / 1250 / -100): the middle bearing
  sheds 14 % of its rigid load to bearing 1. e = +50 um: R_2 2114 N (s 0) ... 1823 N (s 30) against 2642 N rigid.
- **The zero crossing of the middle reaction moves.** Rigid R_2 = 0 at e = -44.9 um; with ISO bearings at s = 0 at
  e = -57.6 um (so at e = -50 um R_2 = +112 N pushes where the rigid shaft needs -142 N). With clearance R_2 does not
  cross: the middle bearing sits inside its clearance (R_2 = 0) at e = -60 (s 10) and -70 / -60 um (s 20, 30).
- **Clearance can make the shaft determinate.** At e = 0 bearing 3 sits inside its clearance for s >= s_c = 9.72 um
  (Sc; s = 10 um: d_3 = 4.70 um, 0.30 um inside s/2 = 5 um; Euler-Bernoulli, rigid housing: the margin is
  model-sensitive), and then R = (P/2, P/2, 0), d_3 - d_1 = 22.46 um, the two-support statics (S8). Away from e = 0
  bearing 3 is loaded again (e -10 / +10 um at s 30: R_3 = +48 / -8 N): in `reactions_s30um` it is a single point at
  R_3 = 0, not a plateau; `clearance` shows the d_3 / (s/2) points crossing the band.
- **Gear:** v at the gear -16.6 um (s 0, e 0) vs -3.9 um rigid; slope at the gear (e +50) 0.39-0.42 mrad vs 0.55
  rigid (-23 ... -28 %). Q_max of the middle bearing (e 0) 526 N (s 0) ... 636 N (s 30); 5 balls loaded at s = 0, 3
  with clearance.
- **Tilt left out of the law:** max abs(psi_in) = 0.78 mrad (2.7 arcmin; bearing 1, s 0, e -70 um; 0.63 mrad at
  e -50); AxisForge's sign, seen from the physical zone it is -psi_in. Its effect on Q_max is not estimated here
  (s002 / s005).
- **Cage phase (RECORD):** a cage fixed in space, half a pitch off, changes Fr at the same abs(d) by +0.6 % (s 0), -3.7
  ... +15 % (s 10), down to -100 % at contact onset (s 30, one ball barely touching): the "ball on the load line"
  assumption matters with clearance at light load (`# OPEN R5-182`).
- Middle centre relative to the line through the outer centres (s004 `rigid_line`): -6.1 um (s 0, e 0) ... +31 um
  (e +50); the rigid supports impose e itself.

## 10. Depends on
v005 (`three_support_shaft`, `SUPPORTS_MM`, `check`, `along_shaft`, `study.hertz_problem` (+ `K_c`, `f_c` returned,
R5), `study.solve`, `study.EB`), s001 (`deep_groove`, `loaded_zone`), s004 (`ring_displacement`, `rigid_line` only),
v003 (`check`), v001 (`node_value`), `numerical_solvers.newton`, AxisForge `ISO16281BallSolver.elements` /
`.solve_contact`, `studies/outputs` (`shaft.ratio_sweep`, `shaft.deflection_along_shaft`,
`bearing_loads.element_loads`, `convergence.newton_history`, `contact.relative_errors`). AxisForge
`feature/newton-solver` `d34ec7f` (read only); design-studies `layout/studies-levels` at `479525f` + this case
(uncommitted at the run).
OPEN: R5-160 (folder name), R5-161 (condensation instead of the force method), R5-162 (v005 construction / study used by
an s), R5-163 (bore 50 on a 30 mm shaft), R5-164 (load level), R5-165 (one plane), R5-166 (no continuation), R5-167
(stiff-limit tolerance from a prototype, dropped; S6r RECORD), R5-168 (CRB row not run), R5-169 (round-off floor of
`elements`), R5-180 (s = 0 exact judge added at review), R5-181 (lift-off margin 0.3 um, s_c 9.72 um), R5-182 (cage
follows the load line), R5-183 (zone azimuths recorded both ways), R5-184 (S5 is sign consistency, not R-003), R5-185
(v005 `hertz_problem` with a force law and seat, minimal-review SHOULD 2: issue row only).
