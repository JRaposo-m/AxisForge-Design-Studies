# s005 bearing tilting-moment feedback

## 1. Question
When the ISO/TS 16281 tilting moment Mz(psi) of each bearing is fed back to the shaft as a support moment, how much
do the bearing tilts psi, the reactions and Q_max change on a two-bearing shaft, over the span and with an axial
load? (proposals.md s005.)

## 2. Why
AxisForge computes Mz (ball Eq. (18), `contact_solver.py:780`; roller Eq. (46), `:1257`, "diagnostic") and never
returns it to the shaft. s004 found the tilt kinematics large; the moment is the other half of the coupling. Known
limits stated, not fixed (SP-16): Mz uses the arm Dpw/2 while the tilt enters through Ri (`:760`), so Mz and psi are
not exactly work-conjugate (the Jacobian is not symmetric). Mz has no published judge: r03 has no tilted-ring moment
(Harris2007_vol2 Sec. 1.2, Eqs 1.22-1.24, PDF p. 24) and no value for this layout is published (R-103); a closed-form
bound is checked instead (R0, added at review). The couple plane and sign are unruled (D-0010 rows 21-22, U1).

## 3. Inputs / sweep
s004's system (`construction.py` imports s004 `shaft`, `bearings`): s006's pinion shaft (uniform d 40 mm, S355,
pins at 0 (A, locating) and L (B, floating), spur H501 pinion at 0.3 L, torque 330 N.m: s004 factor 1); s001's
6210-like deep groove ball bearing at A; at B the same (judged) or the NU210-like cylindrical roller bearing
(its checks RECORD). Euler-Bernoulli, 40 uniform elements.
| swept | values |
|---|---|
| span L | 100, 150, 200, 250, 300 mm |
| axial load at A | Fa/Fr = 0, +0.5, -0.5 of the rigid-pin reaction at A, held fixed, SIGNED in the solver's axial sense (+ = the direction of A sin(alpha_0), V_j > 0, `contact_solver.py:633`; which global X direction that is: `# OPEN R5-179`, for José, R5-190). `# OPEN R5-175`: an applied axial load at the locating bearing, not on the shaft FEM (an axial force on the axis bends nothing; no gear couple, so no J-16). The -0.5 sense was added at review |
| diametral clearance s | 0, 10 (CN-like, the "C0" of proposals.md), 20 um (C3-like) (s004 values, its OPEN R5-53) |
| floating bearing | ball (judged), roller |
| feedback | 0 (no moment: the s004 limit), 1 (the moment of the bearing on the shaft, Sec. 4) |
| RECORD readings | work-conjugate arm 2 Ri/Dpw (SP-16, ball pair); roller Eq. (46) fed back as coded (`-Mz46`) |
`# OPEN R5-176`: one load level (factor 1) and the offset gear only.

## 4. Model (`model.py`)
```python
moment_on_shaft(bearing, ring, R_yz) -> G_yz [N mm]                    # -Mz18 R_hat (ball), +Mz46 R_hat (roller)
with_moments(system, x_mm, m_Nmm, e_yz) -> ShaftSystem               # ExternalMoment m_k e at each pin
pin_flexibility(system, rigid, x_mm, e_yz) -> {F, C, theta_n}          # two unit-moment FEM solves (linear beam)
solve_feedback(system, pair, *, Fa_N, feedback=1, sign=-1) -> {newton, psi_rad, m_Nmm, R_yz, bearing, final, ...}
moment_balance(system, result, x_mm, G_yz) -> {sum_xy_Nmm, sum_xz_Nmm, scale_Nmm, rel}
```
Each bearing is solved by s004 `ring_displacement` (extended at review with `Fa_N`, `tol` and the returned `Mz_Nmm`;
its defaults give s004's numbers, test `test_ring_displacement_defaults_unchanged_by_the_s005_keywords`) through
s001 `solve_bearing` (extended with `tol`, default unchanged, test `test_solve_bearing_tol_keyword_keeps_the_default`).
Physics: the moment of the bearing ON the shaft is put at its pin as an `ExternalMoment`. By virtual work it is minus
the moment conjugate to the tilt: ball `-Mz18` (Eq. (12) tilts the axial term by `+Ri sin psi cos phi_j`, so
`dU/dpsi = (2 Ri/Dpw) Mz18`; the arm Dpw/2 is SP-16), roller `+Mz46` (Eq. (40) tilts the laminae by `-x_k tan psi_j`,
so `dU/dpsi = -Mz46`; review of s005). The shaft stays determinate on two pins, but the moments change the reactions
by (M_A + M_B)/L and the pin slopes. With s004's rigid-body field (ring displacement along the force on the bearing)
the unknowns per bearing are the tilt psi_k and the moment m_k fed back, along the load plane e (the support force at
A on rigid pins; the problem is planar, the normal slope recorded): `psi_k = theta_k(m) + (d_B - d_A).e / L`,
`m_k = feedback G_k.e`, with `theta(m)`, `R(m)` from the beam's flexibility (exact superposition; checked on a final
FEM with the moments and the ring displacements as settlements). `numerical_solvers.newton`, central-difference
Jacobian, line search, `lm` fallback; residuals scaled by the rigid-pin slope and Fr Dpw/2. The nested bearing solve
runs with `tol=1e-12` (at the default 1e-6 the Mz noise, 6e-5 relative, exceeds the Mz change over a difference step).
**Conventions.** The tilt and the loaded zone are AxisForge's (s004's feed): the solver puts its loaded zone along the
node force it is given, the support force on the shaft R, and its tilt is `psi = theta . R_hat` (the azimuth label of
that zone is R-003). For a symmetric deep groove bearing, the zone at `-R_hat` with +Fa gives exactly the moments of
the zone at `+R_hat` with -Fa (review of s005, equal to every printed digit): the "other zone reading" of the first run
was the other axial sense. So the study runs +Fa and -Fa and reports the effect per axial sense; what stays open is
only which global X direction the solver's +Fa is (R5-190). `ExternalMoment(M, theta)` is used as the FEM consumes
it, the generalised force M (cos theta, sin theta) on the slope DOFs (dv_y/dx, dv_z/dx), checked by a probe test;
D-0010 rows 21-22 concern its names.

## 5. v first
v002 `check` (r01) of the final FEM of every solve (moments + settlements: r01 with couples and pin settlements); v003
`check` (r03) of both bearings at psi = 0 at the final reactions (judged: deep groove, Fa = 0; the rest RECORD by v003's
rule). s004 `coupled_tilt` as the limit Mz -> 0. Never an r directly.

## 6. Criteria (fixed before the run)
Judged: ball at B, feedback 0 and 1, every span, Fa/Fr and clearance (at review: both axial senses). The same checks of
the roller at B are recorded (RECORD).
| id | criterion | tolerance | source |
|---|---|---|---|
| F1 | v002.check PASS for every judged record of the final FEM | v002 (rel 1e-9) | v first |
| F2 | v003.check PASS for every judged bearing record at psi = 0 | v003 | v first |
| F3 | Newton converged | `residual_abs` 1e-10 (scaled) | numerical_solvers |
| F4 | moment equilibrium of the shaft about A including the bearing Mz (from the bearing solves, not the applied moments), both planes | rel 1e-8 of the largest term | lead / proposals.md |
| F5 | superposition: final FEM slope along e at each pin = Newton psi; reactions = flexibility reactions | 1e-10 rad; rel 1e-8 | model |
| L1 | limit Mz -> 0 (feedback 0) equals s004 `coupled_tilt` (Fa = 0) | 1e-10 rad (s004's stop rule) | lead |
| S5 | every bearing solve ok and balanced (s001 rule) | residual <= 1e-6 max(Fr, Fa) | s001 |
| R0 | (added at review, RECORD) Mz0 / (Fa Dpw/2) in (0, 1]: by Eqs (17)-(18) the mean of cos(phi_j) weighted by the axial ball forces | (0, 1] | bearing-expert |
| R | change of psi, R, Q_max with feedback vs span; tilt stiffness; Mz at psi = 0 and its tilt part; normal slope; Newton behaviour; roller; 2 Ri/Dpw arm; roller as coded | RECORD | - |
Expected (hypotheses, proposals.md): at Fa = 0 the ball tilt stiffness is small against the shaft (change << 1 %); with
Fa = 0.5 Fr the moment at psi = 0 dominates and the change is large, its sign set by the axial sense.

## 7. How to run
`python -m studies.s000_axisforge.s005_bearing_moment_feedback.study` (about 1 min) writes `results/` (git-ignored).
Tests: `python -m pytest studies/s000_axisforge/s005_bearing_moment_feedback`.

## 8. Risks
ExternalMoment sign (probe test); lamina switching of the roller against a finite-difference Jacobian (central
differences, recorded); which global axial direction the solver's +Fa is (R5-190); Mz not exactly work-conjugate
(SP-16); Mz without a published judge (closed-form bound only). Effort M.

## 9. Result (run of 2026-10-28, after review)
**F2, F3, F4, F5, L1 PASS; F1 MISS (span 150 only, round-off of a 0.25 mm element); S5 MISS (1 solve, ok flag).**
- F1 493/540: the 47 misses are all at L = 150 mm (largest 9.6e-9 against 1e-9); every other span passes. Cause: at
  L = 150 the 40 uniform nodes and the gear face edges (0.3 L +- 11.5 mm) leave a 0.25 mm element, and the RIGID FEM
  alone (no moment, no settlement) already gives 2.0e-9 there; 41 elements (smallest 0.57 mm) give 3.9e-10. A mesh
  round-off of the FEM set up by v002 `uniform_nodes`, not of s005; mesh and tolerance kept (`# OPEN R5-177`). Figure
  `s005_contact_relative_errors_v002_F1`.
- F2 450/450 (6.5e-4). F3 180/180 (scaled residual <= 8.8e-11). F4 180/180: moment equilibrium of the shaft with the
  bearings' own Mz <= 1.5e-9 of the largest term. F5 slope 7.9e-13 rad, reactions 8.7e-9. L1 30/30 (3.2e-11 rad).
  R0 60/60: Mz0 / (Fa Dpw/2) = 0.52-0.71.
- S5 179/180: one solve (ball, s = 0, L 200, Fa 0, pin A) has AxisForge's root flag `ok = False` at a residual of 2.7e-12 N (balanced): at
  `tol = 1e-12` hybr can report "not converged" at a root it has reached (`# OPEN R5-178`).
- **Fa = 0 (ball pair): the feedback is negligible.** psi changes by at most 0.009 mrad, R and Q_max by <= 0.13 %: the
  ball tilt stiffness (2.5e5 N mm/rad at psi = 0; 1e5-1.3e6 at the solution) is two to three orders below the end
  stiffness of the shaft (3 EI / L about 4e8 N mm/rad at L 200).
- **Fa/Fr = 0.5 at A (ball pair): the feedback is large; its direction is set by the axial sense.** Mz at psi = 0 is
  65-87 kN mm (the axial ball forces sit on the loaded side). The tilt stiffness at A is 1.14e7 N mm/rad at psi = 0
  (both senses), 1.5-2.9e7 (+Fa) and 6.1e6-1.2e7 (-Fa) at the solution tilt, i.e. 3-7 % of the shaft end stiffness;
  the tilt part of Mz is 5-43 kN mm (+Fa) and 1-17 kN mm (-Fa) (corrected at review: the first README said 0.2 kN mm).
  - +Fa (solver sense): R_A -1 to -10 %, R_B +3 to +23 %, Q_max at B +3 to +21 %, psi_A -0.10 to -0.17 mrad.
  - -Fa: R_A +4 to +13 %, R_B -10 to -29 %, Q_max at B -10 to -28 %, psi_A +0.19 to +0.39 mrad.
  Largest at the SHORT span (the moment over L changes the reactions by M / L). Which of the two is the physical case
  of a given axial load waits for R5-190 (the solver's +Fa against global X). Fa/Fr at the solution: 0.51-0.55 / 0.44-0.48.
- SP-16 (RECORD): the work-conjugate arm `2 Ri / Dpw` of the 6210-like build is 1.007, so the conjugate reading moves
  every number by < 0.3 %.
- **Roller at B (moment on the shaft +Mz46, virtual work):** restoring, tilt stiffness 2.6-4.2e6 N mm/rad at B; at
  Fa = 0 psi_B falls by 0.004-0.024 mrad and R_B rises by 0.4-0.7 %. Fed back as coded (`-Mz46`, RECORD) the stiffness
  is negative and psi_B rises by 0.004-0.020 mrad (the first run's result; it closes s002's R5-76 as a note).
- Newton (every solve, 270): converged on `residual_abs` in 2-3 iterations, at most 28 residual evaluations, no
  line-search cut, no fallback; quadratic tail (e.g. 2.8e-1, 1.3e-3, 2.2e-8, 1.1e-15); scaled Jacobian condition
  1.8-16. Roller laminae switching did not show. The normal slope stays <= 6.3e-13 rad (planar). Nothing odd in the
  logic; the stop test is on the scaled residual only (`step_rel` 1e-13 never fired).

Figures (`results/`, OUT/s005):
- `s005_bearing_loads_load_zone_sweep_feedback_{ball,roller}_{Fa0,Fa05}`: psi, m (moment on the shaft), |R|, Q_max vs
  span, C3-like, no-feedback lines and fed-back points per axial sense: at Fa = 0 the points sit on their lines; at
  Fa = +-0.5 they leave them on opposite sides, most at the short span.
- `s005_bearing_loads_load_zone_sweep_change_{ball,roller}_{Fa0,Fa05}`: delta psi [mrad], delta R/R, delta Q_max/Q_max
  vs span against 0 (no ratio of a near-zero rigid tilt).
- `s005_bearing_loads_load_zone_sweep_moment_curve`: moment on the shaft vs tilt at L 200: falling lines (restoring)
  for the ball at every Fa and for the roller with +Mz46; the as-coded roller rises; the Fa offsets are opposite.
- `s005_contact_relative_errors_v002_F1`: the F1 errors vs span with the 1e-9 line: only L = 150 is above.
- `s005_convergence_newton_history_main_{ball,roller}`: Newton residuals, quadratic.

## 10. Depends on
s004 (`shaft`, `bearings`, `coupled_tilt`, `ring_displacement` with `Fa_N`/`tol`/`Mz_Nmm`, `rigid_line`, `support_force`,
`DISPLACEMENT_SIGN`), s001 (`solve_bearing` with `tol`, through s004), v001 (`node_value`), v002 (`check`, `THEORIES`,
`study.solve`: s004's OPEN R5-54), v003 (`check`), `numerical_solvers.newton`, `studies/outputs` (`load_zone_sweep`,
`relative_errors`, `newton_history`). AxisForge `d34ec7f` (read only); design-studies `layout/studies-levels` `2c33c0a` +
this case and the s001 / s004 keywords (uncommitted at the run).
OPEN: R5-175 (axial load applied at A), R5-176 (one load level, offset gear), R5-177 (L 150 sliver element, F1),
R5-178 (`ok` flag at tol 1e-12, S5), R5-179 (Fa signed in the solver's axial sense; its global X direction, R5-190).
