# v005 shaft on three supports

## 1. Question
Does the AxisForge shaft FEM solve a statically INDETERMINATE shaft right: three rigid supports, linear radial and
rotational support springs, prescribed settlements (FEM vs r01 extended, `solve_supported`), and three nonlinear
(Hertz) bearings, stiff shaft (vs Harris2007_vol2 Eq. 9.48) and flexible shaft (Newton from `numerical_solvers`,
self-consistent with `scipy.optimize.root`, with r01 at the secant springs and with AxisForge itself through a
fed-back `RadialLoad`)? One validation (v_prerequisites row 13). **First check: the sign of a bearing force fed back
to the shaft** (two pins, one known `RadialLoad`): `convention-dependent (R-003)`.

## 2. Sources
| key | item | PDF page |
|---|---|---|
| Harris2007_vol2 | ch. 9 sign senses: y and delta_r DOWN (along P), F up on the shaft (Eqs 9.6-9.10, Fig. 9.4): inferred from the figures and C2 = EI delta_r1, not printed in words; traced at R5-141 (r01) | 278-279 |
| Harris2007_vol2 | sec. 9.3.2, Fig. 9.9: three bearings, P in span 1; printed: no moment loads, bearing deflection differences negligible ("rigid bearings", uniform EI: the case's reading, one EI in Eqs 9.36-9.40); Eqs 9.36-9.41 | 284 |
| Harris2007_vol2 | Eqs 9.42-9.44: F_1, F_2, F_3. **FLAG: the text cites "Equation 9.4" for Eq. 9.41** | 285 |
| Harris2007_vol2 | sec. 9.3.2.1, Fig. 9.10: rigid shaft; Eq. 9.45 (similar triangles; also Eqs 9.30-9.35 with I -> inf), Eq. 9.46 F_r = K delta_r^n, Eq. 9.47 | 285 |
| Harris2007_vol2 | Eq. 9.48 "valid for bearings that support a radial load only", solved with the equilibrium equations; "See Example 9.5" (worked numbers on the CD-ROM, J-14: **no printed digits**) | 286 |
| Onate2013_vol2 | elimination with prescribed displacements, 1.3.4 Eq. (1.22) (as v002) | via v001 / v002 |
Rows in `literature/published/harris2007_vol2.csv` (read from the page images, 2026-10-28). No numbers are printed for
Fig. 9.9 or 9.10: the inputs are the case's own (those of r01 C17), the judges are the printed equations.

## 3. Model
Uniform S355 shaft L 270 mm, d 30 mm (EI 8.35e9 N.mm^2), supports at 10 (locating), 110, 260 mm (l1 100, l2 150 mm);
P = 1000 N at 50 mm along -Y (Harris's P "down"; a = 40 mm), Q = 600 N at 200 mm along +Z (plane XZ). AxisForge
`RigidSupportFEMSolver` takes three bearings natively (v = 0 at each, rotation free); each support is a zero-width 6204
(v001 R5-21; `# OPEN R5-151`: v001 `add_pins` takes two positions, the third pin repeats its constants).
- **Linear (`study.solve`, `# OPEN R5-150`).** AxisForge has no spring or settlement input. `solve` rebuilds AxisForge's
  K on the same mesh with `StiffnessMatrixBuilder.add_nodal_stiffness` (radial spring: slot 1, the transverse DOF
  freed; rotational spring: slot 2, the cross-section rotation), settles rigid supports by elimination (as v002) and a
  spring's base by `k s` on its load, then runs AxisForge's `build_shaft_result` with the BEAM K, so `R_*` and the
  rotation rows of `f_*_reaction` are the forces and couples of the supports on the shaft. Base case: middle radial
  spring 3e4 N/mm (about the shaft's own stiffness there), rotational spring 2e8 N.mm/rad at 260, settlements XY
  (0.01, -0.02, 0.015) mm, XZ (-0.01, 0, 0.02) mm (the middle one is the spring base).
- **Judge r01 (`check`)**, under the physics reviewer's conditions (r01ext review): the same EI / kGA per section
  (v001 `segments`, AxisForge's kappa; one-point Timoshenko on a uniform mesh at kGA_eff, v001), supports as point
  springs at the bearing centres, linear stiffness, two planes = two r01 calls with the D-0010 map (forces,
  deflections, settlements along +v; couples, theta from +x toward +v), `R` = force on the shaft.
- **Hertz bearings (`study.hertz_problem`, `solve_hertz`).** Each bearing `F = -K sign(v) |v|^n` on the shaft along +Y
  (Harris Eq. 9.46 with delta = -v, a sign convention of the case, not printed; signed: a radial bearing with ZERO
  clearance resists both ways with the same K, loaded arc +-90 deg; with clearance it does not, unchecked against
  Harris vol. 1 ch. 7), K 1e6 N/mm^1.5,
  n 1.5 (`# OPEN R5-152`: the case's own K). AxisForge's beam K and load, condensed exactly on the three bearing
  DOFs; `numerical_solvers.newton` (analytic Jacobian, line search, residual scaled by P, `residual_abs` 1e-10 =
  1e-7 N, above the measured round-off floor ~1e-12 scaled at EI x 1e6), start = linear springs at the secant stiffness of P/3. Plane XY only (P only, radial load only as Harris).
- **Stiff shaft:** E x s, s = 1 ... 1e6 (Harris: I -> inf), each solved as above, vs Eq. 9.48 + equilibrium solved by
  `brentq` (`literature.harris_eq_9_48`).
- **Fed-back sign (first check, R-003).** `ShaftResults.R_*` and `bearing_nodes.Fr_*` are the support force on the
  shaft. Feeding a bearing force back to the shaft as a `RadialLoad` therefore uses +R (magnitude `hypot`, theta =
  `atan2(R_z, R_y)` from +Y toward +Z). `phi_Fr = atan2(Fr_xy, Fr_xz)` (from +Z toward +Y), the direction the bearing
  solver reads, lies 180 deg from the push of the shaft on the bearing (R-003, unruled; recorded, code not changed).

## 4. Public API (`__init__.py`)
```python
check(system, result, settings, *, springs=None, settlements_mm=None, case="v005", label="", tag="") -> list[dict]
reference(system, settings, *, springs=None, settlements_mm=None, h_mm=None) -> {"XZ": SupportedSolution, "XY": ...}
supports(system, plane, *, springs=None, settlements_mm=None) -> [r01 Support]
along_shaft(system, result, settings, *, springs=None, settlements_mm=None, series="", plane="XY") -> records
three_support_shaft(*, supports_mm=SUPPORTS_MM, loads=None, material_id=None, label=...) -> ShaftSystem (unsolved)
stiff_material(scale) -> material id;   MODELS, SPRINGS, SETTLEMENTS_MM
literature.cases(); literature.harris_eqs_9_42_9_44(*, l1, l2, a, P); literature.harris_eq_9_48(*, l1, l2, a, P, K, n)
study.solve(system, settings, n, *, springs, settlements_mm) -> ShaftResults
study.solve_hertz(system, settings, n, *, K, n) -> (ShaftResults, NewtonResult, problem)
```
`springs = {x [mm]: (k_r [N/mm] or inf, k_theta [N.mm/rad])}`, `settlements_mm = {plane: {x [mm]: s [mm]}}`. s003
can pass its system to `check` with the secant springs of its converged bearings.

## 5. Inputs
Base above. Meshes n 27, 54, 108, 216 (h 10 ... 1.25 mm). Middle spring sweep k / 3e4 N/mm = 1e-4 ... 1e4 (17
points, EB, n 27, P only, ends rigid). Hertz: n 54. First check: pins at 10, 260 mm, 1000 N at 50 mm, theta 210 deg.

## 6. Criteria (fixed before the run)
Field errors: largest nodal deviation over the largest absolute r01 value of the field in the plane.
| id | quantity | tolerance | source of the tolerance |
|---|---|---|---|
| F0a, F0b | statics of the two-pin shaft; reaction at B after feeding back +R_B (zero), at A (unchanged) | 1e-9 | statics (v001 TOL_NODAL) |
| F0c, F0d | feeding back -R_B (R_B doubles); phi_Fr - load direction | RECORD | R-003 |
| L1-L3 | three rigid pins (EB) vs Harris Eqs 9.42-9.44 | 1e-9 of P | printed equations, no printed digits |
| C1, C2 | EB nodal v, theta vs r01 | 1e-9 | nodal exactness (v001) |
| C3, C3c | support forces, spring couples vs r01 (EB) / r01 kGA_eff (one-point Timoshenko) | 1e-9 | as C1 |
| C4 | one-point Timoshenko v, theta vs r01 kGA_eff | 1e-9 | v001 |
| C5 / C6 | Richardson of v_xy at P: EB flat; Timoshenko monotonic and extrapolated within 1e-9 of r01 | as v002 | carried over from v002 |
| K0, Kinf, T1 | sweep ends vs two pins / Harris 9.42-9.44; Timoshenko vs r01 at kGA | RECORD | - |
| N2 | Newton vs scipy root (hybr), bearing forces | 1e-8 of P | two stopping tests (1e-7 N, xtol) |
| N3 | converged shaft vs r01 at the secant springs (C1-C3) | 1e-9 | nodal exactness |
| N4 | F_2 fed back as `RadialLoad`, pins settled to v_1, v_3: v_2, R_1, R_3 | 1e-9 | exact re-solve |
| N5 | sum F = P, sum F (x - x_1) = P a | 1e-9 | statics |
| H1 | EI x 1e5 vs Harris Eq. 9.48 | 1e-4 of P: **RECORD in this run**, criterion from the next | first order in the shaft compliance |
| H2 | err x s at 1e5 over that at 1e3 | [0.8, 1.25]: **RECORD in this run**, criterion from the next | O(1/s) |
| N1, H0, N6 | Newton summary; other scales; flexible shaft vs Eq. 9.48 | RECORD | - |
H1 / H2 were set after a scratch prototype of the stiff series (coder evidence says so), so they are RECORD in this
run (`study.CRITERIA_NEXT_RUN`; the record's `verdict_next_run` holds what they would say) and fixed as criteria for
the next run; the others were fixed before any run.

## 7. How to run
`python -m studies.v000_axisforge.v005_shaft_three_supports.study` (about 17 s) writes `results/` (git-ignored):
`v005_first_check.csv`, `v005_literature.csv`, `v005_check.csv`, `v005_richardson.csv`, `v005_stiffness_sweep.csv`,
`v005_hertz.csv`, `v005_newton_history.csv`, figures `v005_shaft_deflection_along_shaft_{XY,XZ}_n27.png`,
`..._hertz.png`, `v005_shaft_error_vs_elements.png`, `v005_shaft_ratio_sweep_{stiffness,rigid_limit}.png`,
`v005_contact_relative_errors_rigid_limit.png`, `v005_convergence_newton_history.png`. `notebook.ipynb` (pair of
`notebook.py`) runs the same, committed with outputs. Tests: `python -m pytest studies/v000_axisforge/v005_shaft_three_supports`.

## 8. Result (run of 2026-10-28)
**129 PASS, 3 MISS, 46 RECORD** (session 2, after the reviews; before: 131 / 3 / 44, H1 and H2 counted as PASS).
First check, Harris 9.42-9.44, every r01 comparison on n 27 / 54 and the Hertz part PASS; the rigid limit is RECORD
(H1, H2 would pass: `verdict_next_run` PASS).
| id | largest error | note |
|---|---|---|
| F0a / F0b | 1.9e-12 / 1.8e-16 | +R fed back empties pin B; -R doubles it (F0c); phi_Fr - load = 180.000 deg (F0d): R-003 |
| L1-L3 | 1.3e-13 | F = 532.8 / 512.0 / -44.8 N |
| C1 / C2 / C3 (EB) | 9e-13 (n 27) ... 2.6e-10 / 2.4e-10 / 8.8e-10 (n 108-216) | round-off growth as v002 (R5-33) |
| **C3c EB, XZ** | 1.6e-11 (27), 2.2e-10 (54), **3.5e-9 (108), 1.1e-9 (216): MISS** | EB round-off on the couple `K d - f` at the rotation row (cancellation of 4 EI/h terms, R5-33), the smaller XZ couple as scale; Timoshenko C3c <= 2.3e-11 |
| C4 / C3 Timoshenko | 3.3e-12 / 2.6e-12 | every mesh |
| C5 | flat (1.1e-9 off r01) | |
| **C6** | **3.1e-9: MISS** | r01 at kGA_eff ITSELF extrapolates to 3.1e-9 off r01 at kGA here (4e-16 for the two-pin shaft): in a hyperstatic beam the shear flexibility enters the reactions, so v(h) is not exactly a + c h^2. The FEM is exact (C4); the tolerance carried over from v002 does not hold for a hyperstatic beam (`# OPEN R5-154`) |
| stiffness sweep C1-C3 | 4.6e-12 | K0 5.5e-5 at k = 1e-4 k_mid, Kinf 4.7e-5 at 1e4 (the limits approached as k, 1/k) |
| N1 / N2 | 4 iterations, quadratic, residual 9e-14 / 8.5e-14 | scipy `root` reports success=False (xtol 1e-14 unreachable) at the same point |
| N3 / N4 / N5 | 1.8e-11 / 1.0e-11 / 1.7e-11 | F = 567.8 / 453.6 / -21.4 N (bearing 3 pulls down) |
| H1 / H2 (RECORD) | 4.0e-6 at EI x 1e5 / ratio 1.012 | errors 0.12, 3.2e-2, 3.9e-3, 4.0e-4, 4.0e-5, 4.0e-6, 2.2e-6 (s = 1 ... 1e6: round-off floor at 1e6); Harris: 639.8 / 333.7 / 26.5 N |

## 9. Depends on
r01 (`solve_supported`, `Support`, R5 G1), v001 (`segments`, `theory_name`, `node_value`, `richardson_verdict`,
`PLANES`, `TOL_NODAL`, `uniform_shaft`, `uniform_nodes`, `ensure_material`, `E_MPA`), v101 `LiteratureCase`
(`# OPEN R5-153`), `numerical_solvers.newton`, `numerical_solvers.richardson`, `scipy.optimize.root` / `brentq`,
`studies/outputs` (`shaft.deflection_along_shaft`, `error_vs_elements`, `ratio_sweep`; `contact.relative_errors`;
`common.plots.convergence.newton_history`). AxisForge `feature/newton-solver` `d34ec7f` (read only;
`RigidSupportFEMSolver`, `StiffnessMatrixBuilder.add_nodal_stiffness`, `Mesh1D`, `build_shaft_result`).
Design-studies `layout/studies-levels` + this case (uncommitted at the run).
OPEN: R5-150 (springs / settlements imposed on AxisForge's K), R5-151 (third pin repeats v001's constants), R5-152
(Hertz K the case's own), R5-153 (LiteratureCase), R5-154 (C6 tolerance for hyperstatic beams), R5-155 (note, not a defect:
with `residual_abs` below the round-off floor `newton` ends `line_search_failed` at the solution, Armijo cannot hold
on round-off; physics-review rerun at 1e-12: s = 1, 1e3 converge, only s = 1e6 stalls at 1.2e-12; hence 1e-10 here),
R5-156 (v002 / v005 `study.solve` duplicate the settlement elimination: merge into one shared `solve`, issue row).
