# s004 bearing compliance on a two-bearing shaft

## 1. Question
On a statically determinate two-bearing shaft with a gear load, how much do the ISO/TS 16281 ring displacements change
the bearing tilt psi and the deflection and slope at the gear, against the rigid-support FEM, as a function of clearance
and load? Decision: if |psi_coupled / psi_rigid - 1| exceeds 10 % over the range, s002 and s006 must use coupled psi.

## 2. Why
The reactions do NOT change (determinate): the case shows kinematics only and says so. The rigid-body slope
(d_B - d_A)/L is of the order of the shaft slope (10 um over 100 mm = 1e-4 rad). The result decides whether s002 / s006
need coupled psi. This case owns the rigid-body-field code (`model.py`, exported in `__init__.py`; s003 may reuse
`ring_displacement` and `rigid_line`; D-0036 amendment item 11). proposals.md s004 (rank 2, T1).

## 3. Inputs / sweep
Shaft: s006's pinion shaft (`stage`: uniform d 40 mm, S355, pins at 0 (A, locating) and L = 200 mm (B, floating), spur
pinion of the H501 pair; `# OPEN R5-50`: imported through s006's package API). Gear "offset" (x_g = 0.3 L: reactions
0.7 / 0.3 of the mesh force) and "centred" (0.5 L: the limit where equal bearings give no rotation). Bearings: s001's
6210-like deep groove ball bearing at A; at B the same or, as the variant, the NU210-like cylindrical roller bearing
(`# OPEN R5-51`: 50 mm bore on a 40 mm shaft; the FEM sees zero-width pins, the bearing model only the reaction, so
the numbers are unaffected; the assembly is fictitious).
| swept | values |
|---|---|
| diametral clearance s | 0, 10, 20, 40 um, labelled 0, CN-, C3-, C4-like (`# OPEN R5-53`: ISO 5753 not held; s001 values) |
| load | torque factor 0.1, 0.2, 0.3, 0.5, 0.7, 1 x 330 N.m (F_bt about 10 kN at 1; larger reaction 7 kN = 0.28 C of the 6210 build; `# OPEN R5-52`: proposals ask 0.1-1 x C0r, AxisForge holds no C0, R5-40) |
| floating bearing | ball (6210-like) or roller (NU210-like) |
| gear position | offset, centred |
| displacement sign | statics (-1, all criteria); +1 once, unphysical, as a sensitivity RECORD (CSV only) |
Euler-Bernoulli, 40 uniform elements (pins and gear on nodes; Hermite nodal exactness, v001). Timoshenko not run
(`# OPEN R5-55`: the in-span slopes do not depend on kGA on two pins, s006).

## 4. Model
`model.py` (exported; s003 may import `ring_displacement` and `rigid_line`, NOT `coupled_tilt`):
```python
support_force(result, k) -> R (Y, Z) [N]                       # support force on the shaft at pin k (AxisForge R_xy, R_xz)
ring_displacement(bearing, R_yz, psi_yz, *, sign=-1) -> {d_yz, delta_r_mm, psi_in_rad, psi_n_rad, Q_max_N, ...}
rigid_line(x, x_A, x_B, d_A_yz, d_B_yz) -> (v (n, 2) [mm], rotation (2,) [rad])     # v001 rigid_body per plane
coupled_tilt(system, result, bearings, *, sign=-1, tol=1e-10, max_passes=50) -> {converged, passes, history, psi_yz, d_yz, R_yz, ...}
```
**`coupled_tilt`: two pins, statically determinate only.** It holds the reactions at the rigid-pin values and never
updates them (exact only when statics fixes them); it raises `ValueError` for any other number of supports or bearings.
A hyperstatic shaft (s003) changes its reactions with d and needs its own coupled solve.
Physics: on two pins the reactions follow from statics; bearings that yield only move the shaft as a rigid body on top
of the rigid-pin field (exact superposition of the linear beam): v = v_FEM + d_A + (d_B - d_A)(x - x_A)/L. The tilt of
bearing k is the shaft slope there (outer ring held by the housing): psi_k = theta_FEM(x_k) + (d_B - d_A)/L. delta_r
depends weakly on psi, so a plain fixed point runs from the rigid-pin slope: psi_k <- theta_FEM(x_k) + (d_B(psi_B) -
d_A(psi_A))/L until max |delta psi| < 1e-10 rad (no Newton: the map is a strong contraction, 3-4 passes).
**Conventions.** Each bearing is solved by s001's `solve_bearing` at |R_k| with the node force along R_k, the support
force on the shaft, as AxisForge feeds it (v003 sign check: Q_max lands 180 deg from the load on the shaft). The tilt
fed to the solver is AxisForge's own projection, psi_in = psi . R_hat (`contact_solver.py:194-201`); the normal part
psi_n, which the solver drops (`:201`), is recorded. **The shaft centre moves ALONG THE FORCE ON THE BEARING: d_k =
delta_r,k (-R_hat_k)**: statics (the shaft pushes the inner ring with -R_k; rigid housing), not a convention; AxisForge
never maps delta_r back to the shaft. The only `convention-dependent (R-003)` part is the feed: AxisForge places its
loaded zone 180 deg off and so sees the tilt with the opposite sign relative to the zone. For Fa = 0 and symmetric
bearings delta_r and Q_max are invariant under R -> -R and psi -> -psi (bearing-expert check, both builds), so **no
s004 number depends on R-003**; it matters in s003 / s005 (Fa != 0, Mz fed back). The tilts in the records are the
signed components along R_hat (s006 `bearing_tilts` `psi_load_rad`), so a sign reversal would show in the ratio.
Vectors are (Y, Z): v (v_xy, v_xz), slopes (dv_y/dx, dv_z/dx). Load angles: the bearing solver measures phi_Fr from +Z
toward +Y, the AxisForge README (D-0010) from +Y toward +Z.
Closed-form estimate (no fixed point, no AxisForge number): ratio_k = 1 + (d_B - d_A) . R_hat_k / (L theta_r01,k .
R_hat_k), with the r01 slopes (v001 `reference`) and the r03 delta_r at psi = 0 (v003 C4 reference).
The settled field is also solved by v002 `study.solve` (prescribed pin displacements on AxisForge's K; `# OPEN R5-54`:
a v's study function used by an s), which gives the reactions identity and the v002.check of the coupled field.

## 5. v first
v001 / v002 (r01) through `v002.check` (v001's comparator) at the study's own conditions: the rigid FEM, and the
settled FEM with the coupled d as settlements (r01 + rigid-body field). v003 (r03) through `v003.check(bearing,
analysis.load_distribution)` for both bearings at the rigid reactions and psi = 0 (r03 has no tilt; deep groove judged,
roller RECORD as in v003). Never an r directly.

## 6. Criteria (fixed before the run)
| id | criterion | tolerance | source |
|---|---|---|---|
| S1 | v002.check PASS for every judged record of the rigid and the settled FEM | v002 (rel 1e-9) | v first |
| S2 | v003.check PASS for every judged bearing record at psi = 0 | v003 | v first |
| S3 | fixed point converged | max |delta psi| < 1e-10 rad | proposals.md |
| S4 | reactions of the settled FEM = rigid FEM (statically determinate) | rel 1e-10 | proposals.md |
| S5 | every bearing solve ok and balanced (s001 rule) | residual <= 1e-6 max(Fr, Fa) | s001 |
| D | decision: max |psi_c / psi_r - 1| over the offset sweep > 10 % -> s002 / s006 use coupled psi | 10 % | proposals.md |
| R | psi_n; 4' limit (1.1636 mrad, ISO/TS 16281 5.3.3) on hypot(psi_in, psi_n), rollers only (R5-77); the unphysical +1 sign; gear deflection and slope; Q_max; S4 error relative to max k d; closed-form ratio | RECORD | - |

## 7. How to run
`python -m studies.s000_axisforge.s004_bearing_compliance.study` (about 20 s) writes `results/` (git-ignored):
`s004_points.csv`, `s004_v002.csv`, `s004_v003.csv`, `s004_history.csv`, `s004_criteria.csv` and the figures
`s004_shaft_deflection_along_shaft_roller_C3_x1.png` (rigid vs coupled, with the rigid-body line, component along the
mesh load), `s004_shaft_ratio_sweep_psi_ratio_{ball,roller}.png` (psi_c / psi_r vs load for s = 0 and C4-like per pin,
FEM markers on the closed-form lines, the centred-gear limit, the +-10 % band), `s004_convergence_newton_history_fixed_point.png`
(fixed-point passes, max |delta psi| [rad] vs pass). Tests: `python -m pytest studies/s000_axisforge/s004_bearing_compliance`.

## 8. Risks
Clearance at light load (a bearing nearly unloaded): delta_r includes s/2 at every load; it cancels in the rotation for
any two bearings with the same s whose reactions are collinear and of the same sense (two pins with the gear between
them, ball or roller at B), and adds s/L of rotation when the reactions are opposite (overhung gear). The fixed-point
stopping rule sits near the solver's own tolerance (delta_r to about 1e-8 mm, i.e. 5e-11 rad over 200 mm). The R-003
feed (zone 180 deg off, tilt sign relative to the zone) costs nothing here (Fa = 0, symmetric bearings) but does in
s003 / s005. Rigid housing; no bearing tilt stiffness fed back to the shaft (Mz, s005).

## 9. Result (run of 2026-10-28, session 2)
**S1, S2, S3, S5 PASS; S4 MISS (round-off, evidenced); D: coupled psi needed (both variants).**
- S1 1152/1152 (largest 7.4e-10). S2 1008/1008 (6.5e-4). S3 192/192 converged in 3-4 passes (1 at the centred
  equal-ball limit, where the rotation is 0; last change <= 8.3e-11 rad). S5 192/192 (residual <= 1.3e-3 N).
- **S4 MISS: 76/192 within 1e-10; largest 6.0e-10.** The rigid-body displacement enters the FEM through K d, whose terms
  (12 EI/h^3 x |d| = about 8e7 N) cancel to reactions of 700-7000 N. Relative to max |k d| the error is <= 2.4e-16
  (`reaction_identity_err_kd`, RECORD): machine round-off, not physics. v002.check C3 of the settled FEM passes at 1e-9
  and the settled slope at the pins equals the fixed point's psi to 5.9e-14 rad. The tolerance stays (proposals.md);
  R5-56 proposes judging relative to k d.
- **D: coupled psi is needed.** Offset gear, ball at B: psi_c / psi_r 0.76-0.78 (A) and 1.29-1.31 (B) at factor 0.1,
  0.90 / 1.14 at factor 1 (max |ratio - 1| 31 %); the closed form r01 + r03 reproduces every ball ratio to 7e-5.
  Roller at B: 0.50-0.58 / 1.54-1.65 at 0.1, 0.80 / 1.26 at 1 (max 65 %). **Roller numbers: delta_r of the NU210 build
  is not judged** (v003 RECORD: AxisForge's roller delta_r is 7-209 % above r03's, so the roller closed-form line differs
  by up to 0.11 and is no judge); D already holds on the ball variant. The coupling grows at LIGHT load: the shaft slope
  scales with F, delta_r with F^(2/3) (balls, Hertz point contact) or about F^0.9 (rollers, lamina law; exponent not
  checked against the page), plus the load-independent s/2, so the rotation is relatively largest at low load. It moves
  the tilt from the heavier-loaded pin A to pin B (A's centre moves more along the force on it). Clearance matters
  little in the ratio: ball 0.896 -> 0.895 at factor 1 from s 0 to C4-like.
- **Roller tilt at B rises:** rigid 0.69 -> coupled 0.87 mrad at factor 1, C3-like (+26 %); the largest coupled tilt is
  1.068 mrad (C4-like, centred gear, factor 1; rigid 0.95) = 92 % of 4' (none over). Rigid pins understate the roller
  misalignment; a heavier load or a longer span would cross 4' (R5-59, for s002 / s006).
- delta_r depends weakly on psi here: coupled vs psi = 0, at most -0.14 % for the 6210 build and -3.8 % for the NU210
  build at B (s003 must not assume it for crowned rollers at larger tilt).
- Limit: centred gear with two equal ball bearings gives ratio 1 exactly at every load and clearance; with the roller at
  B it does not (0.68-1.32).
- Gear (factor 1, offset; component along the mesh load): deflection 0.045 mm on rigid pins -> 0.083 mm (ball) / 0.078
  mm (roller) at s = 0, the bearing compliance; the clearance adds a rigid shift of about s/2 (0.089 / 0.095 / 0.106 mm
  ball at CN- / C3- / C4-like), which leaves the rotation unchanged. The slope at the gear falls from 0.426 to 0.331
  (ball) / 0.245-0.248 mrad (roller): -22 % / -42 %. **This holds for this pinion shaft and layout only:** the rigid
  rotation opposes the bending slope at x_g = 0.3 L because pin A (larger reaction) is on the gear side; for a gear
  nearer the lighter-loaded or the floating bearing, an overhung gear, or the mating shaft it can add, and the mesh
  misalignment is the relative tilt of both gears; the centre displacement also moves the mesh. s006 on rigid pins is
  NOT a bound in general (R5-57: rerun s006 with the coupling on both shafts).
- RECORD: psi_n <= 2.6e-14 mrad (all load in the plane of l). The unphysical +1 sign (shaft centre moved against the
  force on the bearing; the gear then deflects LESS than on rigid pins, 0.007 mm or even -0.017 mm at factor 1) gives
  ratios 0.69-1.24 (ball) / 0.35-1.50 (roller): a sensitivity check of the formula, not a reading of R-003. Q_max at
  factor 1, C3-like: A 3648 N, B 1623 N (ball) / 1220 N (roller).

## 10. Depends on
s001 (`load_node`, `solve_bearing`, `records`, `deep_groove`, `cylindrical_roller`), s006 (`stage`, `gear_station`,
`line_of_action`, `bearing_tilts`), v001 (`rigid_body`, `node_value`, `reference`, `section`), v002 (`check`,
`along_shaft`, `THEORIES`, `study.solve`), v003 (`check`), `studies/outputs` (`l000_axisforge/plots/shaft.py`,
`common/plots/convergence.py`). AxisForge `feature/newton-solver` `d34ec7f` (read only); design-studies
`layout/studies-levels` at `09b58f6` + this case (uncommitted at the run).
OPEN: R5-50 (s006 construction and model imported), R5-51 (bore 50 on a 40 mm shaft), R5-52 (load level, no C0), R5-53
(clearance labels without ISO 5753; ball groups applied to the roller), R5-54 (v002 study.solve used by an s), R5-55 (no
Timoshenko), R5-56 (S4 tolerance vs cancellation), R5-57 (s006 rigid-pin misalignment is not a bound; rerun with the
coupling), R5-58 (R-003 affects only the zone azimuth and the tilt sign relative to it), R5-59 (roller tilt at B 92 % of
4' with coupling).
