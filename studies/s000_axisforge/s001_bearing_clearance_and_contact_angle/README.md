# s001 bearing clearance and contact angle

## 1. Question
How do the diametral clearance s (deep groove ball, cylindrical roller) and the nominal contact angle alpha_0 (angular
contact ball) change the element loads Q_j, the loaded zone (psi_l, epsilon), the ring displacements delta_r / delta_a
and Q_max of one bearing at a fixed Fr, Fa (AxisForge ISO/TS 16281 load distribution)?

## 2. Why
The bearing half of ladder step 3. s001 owns the load-distribution model (`model.py`) that s002, s003, s004 and s102
import through `s001.__init__` (D-0036 amendment item 11). Ported from `s001_bearing_load_distribution/_common/`
at the tag `studies-pre-sweep` (`solve.py`, `rows.py`, the bearing specs), without the shaft FEM build, plots,
reports, tilt and stiffness (stiffness: v004; tilt: s002).

## 3. Inputs / sweep
Three single-row bearings of the 6210 envelope (d 50, D 90, b 20 mm), steel, E 210 GPa, nu 0.3; internal geometry
didactic, not catalogue, equal to v003's builds except 40 laminae per roller against 30 (`# OPEN R5-41`): 6210-like deep groove ball bearing (Dw 12.7, Dpw 70 mm, Z 9), 7210-like angular
contact (same balls, Z 13), NU210-like roller (Dwe 11, Lwe 12, Dpw 70 mm, Z 14, 40 laminae).
| bearing | swept | load |
|---|---|---|
| deep groove | s = 0, 5, 10, 20, 30, 40, 50, 60 um (diametral) | Fr 5000 N; Fa/Fr = 0, 0.25, 0.5, 1 |
| angular contact | alpha_0 = 15, 25, 40 deg (s = 2A(1 - cos alpha_0) derived) | Fr 5000 N; Fa/Fr = 0 (flagged R-137), 0.25, 0.5, 1, Fa in the contact direction |
| cylindrical roller | s = 0 ... 60 um, as above | Fr 5000 N, Fa 0 |
Fr = 5000 N is about 0.2 C of the 6210 build (`# OPEN R5-40`: proposals ask a level from C0; AxisForge has no static
rating). Untilted rings (psi = 0). No preload: the families reject s < 0, so this is a method study.

## 4. Model
AxisForge's ISO/TS 16281 solvers (`ISO16281BallSolver`, `ISO16281RollerSolver`, dispatched by `resolve_solver_cls`):
radial and axial force balance of the inner ring for delta_r, delta_a, ball contact angle following the load
(curvature-centre model, Eqs (12)-(18)), roller laminae with the ISO reference profile (Eqs (38)-(41)).
`model.py` (public API, `__init__.py`):
```python
load_node(bearing, Fr_N, Fa_N=0.0, *, phi_Fr_rad=0.0) -> BearingNodeData   # a known load, no shaft FEM needed
solve_bearing(bearing, node, *, psi_rad=0.0, postprocess=True) -> (BearingAnalysisResult | None, postprocessed, error)
loaded_zone(bearing, delta_r_mm, delta_a_mm=0.0) -> (psi_l_rad, epsilon)     # from the solver's own kinematics
records(bearing, analysis, *, postprocessed, error, **ident) -> (case record, element records)
deep_groove(s_mm), angular_contact(alpha_0_deg), cylindrical_roller(s_mm)   # construction.py, unsolved
```
`loaded_zone`: the zone edge where Eq. (12) (balls) or `delta_r cos(phi) = s/2` (rollers) closes, epsilon =
(1 - cos psi_l)/2 (Harris2007_vol1 Eqs 7.10-7.15, PDF p. 156-157, written with psi_l; capped at 1); AxisForge has no epsilon field. A shaft study passes its
FEM `ShaftResults.bearing_nodes` entry instead of `load_node` (the support force on the shaft: R-003).
Azimuth: every record is in the LOCAL azimuth phi_j (0 on the radial-load line, the most loaded element); where the
zone sits in the shaft frame is `convention-dependent (R-003)` and is not used here. Clearance s diametral (R-105).

## 5. v first
v003 (ISO/TS 16281 load zone vs r03): every solved bearing goes through `v003.check(bearing,
analysis.load_distribution, case="s001")` (`results/s001_v003.csv`); v003 judges the deep groove bearing at Fa = 0
(any s) and records the angular contact and roller rows (OPEN R5-12 of v003). r03 is never called here.

## 6. Criteria (fixed before the run)
| id | criterion | source |
|---|---|---|
| S1 | v003.check PASS for every judged record of the deep groove bearing at Fa = 0 (s = 0 is the r03 limit, eps = 0.5) | proposals.md s001 |
| S2 | radial load (Fa = 0), deep groove and roller: psi_l strictly decreasing and Q_max strictly increasing with s | proposals.md s001 |
| S3 | every case `ok` and balanced, residual <= 1e-6 max(Fr, Fa) (AxisForge does not enforce `ok`) | proposals.md caveat |
| S4 | SB-45 guard: alpha_j alpha_ref > 0 for every loaded ball when Fa > 0 (alpha_ref = alpha_0, or Fa at alpha_0 = 0) | proposals.md s001 |
| R | angular contact at Fa = 0 flagged R-137, kept in the tables, not drawn; L10r RECORD ("no judge: ISO 281:2007 not held"); every Fa > 0 and roller trend RECORD | proposals.md s001 |

## 7. How to run
`python -m studies.s000_axisforge.s001_bearing_clearance_and_contact_angle.study` (about 5 s) writes `results/`
(git-ignored): `s001_cases.csv`, `s001_elements.csv`, `s001_v003.csv`, `s001_criteria.csv` and the figures
`s001_bearing_loads_element_loads_{deep_groove_s,angular_contact_alpha0}.png`,
`s001_bearing_loads_load_zone_sweep_{zone_vs_s,displacements_vs_s,angular_contact}.png`
(`studies/outputs/l000_axisforge/plots/bearing_loads.py`) and `s001_contact_relative_errors_v003.png` (the judged
errors with v003's tolerance band, `l100_dry_contact/plots/contact.py::relative_errors`, as v003 does). Tests:
`python -m pytest studies/s000_axisforge/s001_bearing_clearance_and_contact_angle`.

## 8. Risks
- R-137: the angular contact bearing is solved at Fa = 0 and reported ok. AxisForge drives every alpha_j to 0, a
  deep-groove-like state.
- SB-45: the deep groove family has a symmetric groove, so alpha < 0 under Fa < 0 is legitimate there. The guard
  takes Fa as its reference for that family and alpha_0 only for angular contact (R5-44).
- Each angular contact bearing is a SINGLE bearing, axially free. Past r03's single-row limit (Fr tan(alpha_0) / Fa
  above about 1.25; here 15/25/40 deg at Fa/Fr 0.25 and 40 deg at 0.5) the ring slides back until the bearing acts as
  a deep groove bearing with the built-in clearance 2A(1 - cos alpha_0) (0.238 mm at 40 deg). A mounted 7210 has a
  partner bearing or a preload that holds alpha near alpha_0. That is not modelled, so those 4 rows are not a service
  state. r03, a fixed-angle model, refuses them: they are flagged `unvalidated (R5-42)` and drawn as open markers.
- Roller zone edge from rigid kinematics: the ISO profile moves it inward by less than 5e-4 in cos(psi_l). `n_loaded`
  gives the rollers with Q_j > 0.
- eps is capped at 1 (all elements loaded). Harris's eps is above 1 there.

## 9. Result (run of 2026-10-28)
**S1-S4 PASS.**
- S1: 56/56 judged v003 records PASS. The largest |error| is 3.0e-8 against 1e-5 (figure `relative_errors_v003`).
  c_p against the Hertz series: -0.065 %.
- S2: deep groove psi_l 90.0 -> 64.4 deg, Q_max 2436 -> 2955 N (+21 %) from s 0 to 60 um. Roller psi_l 90.0 ->
  49.2 deg, Q_max 1553 -> 2362 N (+52 %).
- S3: 52/52 balanced, largest residual 4.3e-4 N.
- S4: 33/33.
- Stribeck Z Q_max / Fr at s = 0: deep groove 4.385 against 4.37 (Harris Eq. 7.23); roller 4.35 against r03's 4.09.
  The roller difference is the ISO profile.

Answer:
- Radial load, Fr 5000 N: 60 um of diametral clearance narrows the deep groove zone from eps 0.50 to 0.28 and raises
  Q_max by 21 %. The roller goes from eps 0.50 to 0.17 with +52 %: its stiffer line contact (delta_r 0.012 mm at
  s = 0 against 0.035 mm for the balls) makes the same clearance a larger share of its approach.
- Roller against r03 (RECORD): AxisForge's eps is 8-23 % above r03 for s > 0 and delta_r 10-54 % above. Both come from
  the ISO reference profile of the laminae, which r03 does not have: a shorter loaded length, a softer roller.
- Combined load, deep groove: the effect of clearance depends on Fa/Fr.
  - At Fa/Fr 0.25, Q_max still rises with s (2258 -> 2793 N, +24 %).
  - At 0.5 it is non-monotone, with a minimum near s 20 um (2269 N).
  - At Fa/Fr 1 the trend reverses: all 9 balls are loaded at every s and Q_max falls (3006 -> 2637 N), because
    clearance raises the operating angle (18.0 -> 24.8 deg).
- Angular contact (RECORD), Fa/Fr 1: alpha_0 15 / 25 / 40 deg gives Q_max 1940 / 1778 / 2071 N and eps 1.0 / 1.0 /
  0.57.
- Below Fr tan(alpha_0) / Fa = 1 the model has no axial partner: the 13 deg operating angle and delta_r 0.14 mm of the
  40 deg bearing at Fa/Fr 0.25 belong to that unmounted state, unvalidated (v003).
- L10r (RECORD, ISO/TS 16281 reference life, no ISO 281 judge): deep groove at Fa = 0 falls from 126 to 100 Mrev over s.
Tables and figures are copied to the vault, `80_Inbox/2026-10-28/r5_studies/s001/`.

## 10. Depends on
v003 (`studies.v000_axisforge.v003_iso16281_load_zone.check`, R5 run, uncommitted at the time of this run: fill its
commit when it lands), `studies/outputs/l000_axisforge/plots/bearing_loads.py`, `l100_dry_contact/plots/contact.py`.
AxisForge `feature/newton-solver` `d34ec7f` (read only): `resolve_solver_cls`, `ISO16281BallSolver` (`elements`
reused for the dense Q(phi) curve of the figures), `BearingNodeData`, `axisforge.outputs.solvers.bearings` record
readers, `_format.write_records_csv`. The one-bearing solve at given loads also exists in v003 (`study.solve_bearing`)
and will be needed by s002: an AxisForge public entry point is an issue row (R5-45). Until then s001's is the
reference one.
