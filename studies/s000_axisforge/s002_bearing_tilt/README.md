# s002 bearing tilt

## 1. Question
How does an imposed tilt psi of the inner ring change Q_j, alpha_j, Mz and Q_max of a deep groove ball bearing, and the
lamina loads q_jk of a cylindrical roller bearing (AxisForge ISO/TS 16281)?

## 2. Why
The shaft slope at a bearing couples shaft and bearing (ladder step 3; ruled by José, rule 12). This case measures
the bearing's answer to a given psi; s004 / s006 measure the psi a shaft produces. The model is s001's, imported
through `studies.s000_axisforge.s001_bearing_clearance_and_contact_angle` (no `model.py` here).

## 3. Inputs / sweep
s001's 6210-like deep groove and NU210-like roller bearings (40 laminae), diametral clearance 20 um. Two load levels
Fr = 2500 and 5000 N: 0.10 and 0.20 C of the deep groove build (C 24.8 kN), 0.043 and 0.086 C_r of the roller
(C_r 58.1 kN), which is the "medium load" ISO 5.3.3 asks for Eq. (60) (the standard does not define it). Deep groove: Fa/Fr = 0 and 0.5; psi = 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5,
2 mrad. Roller: Fa = 0; psi = 0, 0.125, 0.25, 0.5, 0.75, 1 mrad. Each series also at psi = -1 mrad, which T4 uses.
ISO/TS 16281 5.3.3 (after Eq. (60), PDF p. 21) limits the stress-riser approximation of Eq. (60), used in the roller
L10r, to a total misalignment below 4' = 1.1636 mrad. Roller rows above it would be flagged `psi > 4' (ISO 5.3.3,
Eq. (60))`; the roller sweep stops at 1 mrad, so none is. The limit is not one of the lamina loads, nor of ball bearings,
so the ball rows up to 2 mrad are not flagged (R5-73, R5-74). The 4' is on the TOTAL misalignment: a shaft case
(s004, s006) must test hypot(psi_in, psi_n), not the in-plane projection the solver uses (R5-77).

## 4. Model
s001 `solve_bearing(bearing, load_node(...), psi_rad=psi)` is the AxisForge solver with `psi_input=True` and
`psi_override`. Ball: `Vpsi_j = Ri sin(psi) cos(phi_j)` is added to the axial centre distance, Eq. (12). The moment
Mz = (Dpw/2) cp sum Q... sin(alpha_j) cos(phi_j), Eq. (18). Roller: `psi_j = arctan(tan psi cos phi_j)`, laminae
`delta_jk = delta_j - x_k tan(psi_j) - 2 P(x_k)` with the ISO reference profile, Eqs (39)-(44), and Mz = sum
cos(phi_j) sum x_k q_jk, Eq. (46). psi is the misalignment in the plane of the radial load. The tilt normal to that
plane is dropped by the solver. Mz is recorded in each solver's own sign. Its axis in the shaft frame follows
phi_Fr and the sign of the fed-back force, so it is `convention-dependent (R-003)` and is not mapped here (D-0010).
Local azimuth only, as in s001. Eq. (60) (`stress_riser_factor`) is not used in the lamina map: q_jk are lamina
loads (independent nonlinear springs, Eq. (36), no shear coupling), not edge pressures. Eq. (60) is inside
AxisForge's roller L10r (`contact_postprocessing.py`), valid here because the roller sweep stays below 4'.
Mz of Eqs (18) and (46) is the external moment on the rings that holds the imposed tilt (from equilibrium,
ISO 4.2.2 / 5.2.4). Mz psi > 0 means a positive tilt stiffness; the bearing's own restoring moment is -Mz.

## 5. v first
v003, through `v003.check(bearing, analysis.load_distribution, case="s002")`, at psi = 0 (r03 has no tilt).
v006 (roller tilt) is blocked, so every roller result is RECORD.

## 6. Criteria (fixed before the run)
| id | criterion | tolerance |
|---|---|---|
| T1 | psi = 0 imposed through `psi_override` equals s001's path without an override (psi from the node slopes): Q_j, delta_r | 1e-6 relative (solver tolerance) |
| T2 | v003.check at psi = 0: every judged record PASS (deep groove, Fa = 0, both Fr) | v003's |
| T3 | every case ok and balanced, residual <= 1e-6 max(Fr, Fa) | - |
| T4 | deep groove, Fa = 0: positive tilt stiffness, Mz psi > 0 for psi > 0 (Mz = applied moment, the solver's sign), and Mz(-1 mrad) = -Mz(+1 mrad) | 1e-6 of Mz(+1 mrad) |
| R | roller T4 (RECORD, v006 blocked); everything at Fa > 0; L10r | RECORD |

## 7. How to run
`python -m studies.s000_axisforge.s002_bearing_tilt.study` (about 6 s) writes `results/` (git-ignored):
`s002_cases.csv`, `s002_elements.csv`, `s002_laminae.csv`, `s002_v003.csv`, `s002_psi0_vs_s001.csv`,
`s002_criteria.csv`, and the figures `s002_bearing_loads_element_loads_{deep_groove,cylindrical_roller}.png`,
`s002_bearing_loads_load_zone_sweep_tilt.png` and `s002_bearing_loads_lamina_map_roller.png`
(`studies/outputs/l000_axisforge/plots/bearing_loads.py`). Tests: `python -m pytest studies/s000_axisforge/s002_bearing_tilt`.

## 8. Risks
q_jk are not edge pressures. The roller L10r holds below 4' (ISO 5.3.3). R-003 for any shaft-frame reading, and
for the sign of psi against Fa. A deep groove bearing with clearance under a pure radial load is free axially here:
the ring floats, see section 9. In a shaft with a locating bearing (s004, s006) delta_a is constrained, and the
bearing picks up Fa and Mz instead.

## 9. Result (run of 2026-10-28)
**T1, T2, T3 PASS; T4 MISS; the roller T4 is RECORD.**
- T1: the two paths agree exactly (0.0, 6 cases).
- T2: 16/16 PASS.
- T3: 50/50 balanced, worst residual 2.8e-4 N.
- T4 (deep groove, Fa = 0): positive tilt stiffness at every psi > 0 (Mz 91 / 178 N.mm at +1 mrad for Fr 2500 /
  5000 N). The
  oddness test misses by a hair: |Mz(-1) + Mz(+1)| / |Mz(+1)| = 1.6e-6 against 1e-6. The cause is the near-zero
  moment: 1.5e-4 N.mm absolute, which is 8e-10 of Fr Dpw/2. A tolerance taken relative to Mz itself is tighter than
  the root tolerance passes through to a quantity this small. The miss is kept; the remedy (normalise by Fr Dpw/2)
  is R5-71.
- Roller (RECORD): Mz is -4901 N.mm at +1 mrad (Fr 5000 N), against +178 N.mm for the ball. The opposite sense is
  in ISO itself: each solver matches its printed equation. With Eq. (12) (+Ri sin psi cos phi_j), Eq. (18) gives
  Mz psi > 0; with Eqs (40)-(41) (-x_k tan psi_j), Eq. (46) gives Mz psi < 0. If x_k points along the ball's +V,
  both psi are the same rigid rotation, so the ball Mz is work-conjugate to psi and the roller sign is the odd one
  (-Eq. (46) would be). ISO never relates the two clauses. Neither Mz can be placed in D-0010 axes until the sign
  of psi from a shaft slope is traced. Mz is not fed back today (diagnostic), so no current result is wrong; it is
  a risk for s005 (R5-76, convention, relates to R-208).

Answer:
- Deep groove, pure radial load, s 20 um, inner ring free axially: a tilt up to 2 mrad changes Q_max by less than
  0.01 %. The ring floats axially by -0.031 mm per mrad (0.87 Ri psi; delta_a -0.100 -> -0.162 mm), and the
  moment stays about 91 / 178 N.mm per mrad (Fr 2500 / 5000 N). The clearance absorbs the tilt only because nothing holds the ring axially.
- Deep groove with Fa = 0.5 Fr: there is already a moment at psi = 0 (Mz 54.9 kN.mm at Fr 5000 N; combined load).
  The sign of psi against Fa matters (`convention-dependent (R-003)`, R5-75). At -1 mrad, the worse side, Q_max
  rises 7.4 % and 3.9 % and L10r falls 17 % and 12 % (Fr 2500 / 5000 N). At +1 mrad Q_max rises 1.4 % and 2.2 % and
  L10r rises 8 % and 3 %: the ball L10r rises with psi > 0 up to about 1.25 mrad while Q_max rises, because
  the cubic mean over the balls, not Q_max, sets the life. At Fr 2500 N and psi >= 1.25 mrad, delta_r is negative: the axial shift and the tilt
  together already give the approach near phi = 0.
- Roller, Fr 5000 N: at 1 mrad the load moves to one roller end on the loaded side (lamina map), Q_max falls 0.5 %,
  and the reference life L10r falls 15 % (7993 -> 6771 Mrev). At Fr 2500 N it falls 19 %. This uses Eq. (60), valid
  below 4'.
Tables and figures are copied to the vault, `80_Inbox/2026-10-28/r5_studies/s002/`.

## 10. Depends on
s001 (`load_node`, `solve_bearing`, `records`, `deep_groove`, `cylindrical_roller`; R5, uncommitted at this run), v003
(`check`), `studies/outputs/l000_axisforge/plots/bearing_loads.py` (`element_loads`; `load_zone_sweep` with the new
`x_marks` and `open_sources`; the new `lamina_map`). AxisForge `feature/newton-solver` `d34ec7f` (read only):
`lamina_records` of `axisforge.outputs`. OPEN: R5-71 (T4 tolerance for a near-zero moment),
R5-73 (4' limit: ISO 5.3.3, roller Eq. (60) only), R5-74 (Eq. (60) is in the roller L10r; Fr/C_r stated),
R5-75 (both tilt senses at Fa > 0, R-003), R5-76 (Mz senses of Eq. (18) vs (46), convention; was R5-70),
R5-77 (in-plane tilt only; the 4' test is on hypot(psi_in, psi_n); was R5-72).
