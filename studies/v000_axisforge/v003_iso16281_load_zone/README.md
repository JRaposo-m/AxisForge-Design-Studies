# v003 ISO/TS 16281 load zone

## 1. Question
Does AxisForge's ISO/TS 16281 load distribution of one rigid-ring bearing (`ISO16281BallSolver`, roller solver)
put the right load on each rolling element? Judged for the deep groove ball bearing under a radial load, at
any diametral clearance: AxisForge's `Q_j`, `Q_max`, `delta_r`, `eps` against r03 (Harris ch. 7, the same `c_p`
and clearance), and `c_p` against the Hertz series of the two contacts (r02 `hertz_full` + r03 Eq. 7.6). The
angular contact and cylindrical roller bearings are compared the same way but RECORD only (section 3). First
check (R-003): where does `Q_max` land in the global frame against the direction of the load on the shaft?

## 2. Sources
| Key | What | Equation / table | PDF page |
|---|---|---|---|
| r03 (`5019e10`) | load zone of a rigid-ring bearing: `ring_load_distribution`, `solve_ring_displacement`, `ball_deflection` | Harris2007_vol1 Eqs 7.10-7.22, 7.58-7.71 | 156-171 |
| r02 (`hertz_full`) | Hertz contact of the ball with each groove | Johnson1985 Sec. 4.2 | 105-107 |
| ISO_TS_16281_2008 | AxisForge's model: curvature-centre approach, `alpha_j`, `c_p` | 4.2.2, Eqs (11)-(15) | (AxisForge docstrings) |
| Hamrock1983 | DGBB example: inputs, `F_max` 4513 N, `delta_i`, `delta_o`, `delta` 5.438e-5 m, Z 4.564 | sec. 14.2, Eqs (1)-(3), (30), (37)-(39), (42), (57) | 44-46 |
| Hernot2000 | 218 ACBB: Z 16, k 423000 N/mm^1.5, a 40 deg, `F_r = F_a`; `Q_max` 6544 N; Table 1 `delta_a`, `delta_r`, `eps` | sec. 2.3, Eqs (1)-(9), Table 1 | 2-3 |
| Harris2007_vol1 | load distribution for clearance / interference; `delta_a` and `alpha` charts | Figs 7.3, 7.6 | 160, 163 (charts, not used) |
Hamrock: PDF page = printed page + 5. Hernot: journal page = PDF page + 82. Values read from the page images.

## 3. Model
- AxisForge (ISO/TS 16281 4.2.2): each ball between the groove curvature centres, distance `A = ri + re - Dw`;
  `delta_j = |(A cos a0 + delta_r cos phi_j, A sin a0 + delta_a)| - A`, `alpha_j` its angle, `Q_j = c_p delta_j^1.5`;
  rollers `Q_j` from laminae with the reference profile. `s` is the **diametral** clearance (`alpha_0 =
  arccos(1 - s/(2A))`), Harris's `P_d`.
- r03: `delta_psi = delta_a sin a + delta_r cos a cos psi - P_d/2`, `Q = K_n delta^n`, fixed contact angle.
- **Deep groove, `F_a = 0` (judged):** AxisForge drives `delta_a` to `-A sin a0` and every `alpha_j` to 0, which
  is Harris Eq. 7.10 = r03 at `alpha = 0` with `clearance_m = s` and `K_n = c_p` (bearing-expert review of r03):
  the two must agree to the root tolerance. `delta_a` is not compared (AxisForge measures it from the
  touching state at `alpha_0`).
- **Angular contact (RECORD, OPEN R5-12):** AxisForge's angle follows the load; r03 is its fixed-angle
  linearisation, gap of order `delta/A`. r03 gets `clearance_m = 0` (AxisForge's `s` of an ACBB is derived from
  `alpha_0`, not a clearance).
- **Cylindrical roller (RECORD, OPEN R5-12):** AxisForge applies the ISO/TS 16281 reference profile in laminae;
  r03 is one spring per roller, `n = 10/9` (AxisForge's, not r03's `N_LINE = 1.1`), `K_n = c_L`.
- Azimuth: the element comparison uses AxisForge's LOCAL `phi_j` (0 on the load line) = r03's `psi`: no
  convention enters it. The global azimuth (sign check) is convention-dependent (R-003).
- One bearing is driven through the solver's shaft-result node list (`study.solve_bearing`: an AxisForge
  `BearingNodeData`, the same form as s001 `model.solve_bearing`; no documented one-bearing entry point, OPEN
  R5-11), `psi = 0` (no tilt; r03 has none).

## 4. Public API (`__init__.py`)
```python
from studies.v000_axisforge.v003_iso16281_load_zone import check, elements, r03_inputs, hertz_series_cp, literature
check(bearing, result, *, case="v003", label="") -> list[dict]     # C1-C7 records
elements(bearing, result, *, case="v003", label="") -> list[dict]  # Q_j per element, AxisForge and r03
r03_inputs(bearing) -> dict      # kind, n, K_n [N/m^n], alpha_rad, clearance_m: the one SI conversion
hertz_series_cp(bearing) -> float  # c_p [N/mm^1.5] from r02 (each contact) and r03 Eq. 7.6
deep_groove_6210(s_mm), angular_contact_7210(alpha_0_deg), cylindrical_nu210(s_mm), sign_check_shaft(theta_deg)
literature.cases()               # Hamrock 1983 sec. 14.2 (AxisForge system), Hernot 2000 (r03 inputs only)
```
`study.solve_bearing(bearing, Fr_N, Fa_N=0.0)` solves one bearing (`F_r` along +Z of the node, `psi = 0`).

## 5. Inputs
- 6210-like DGBB: 9 balls of 12.7 mm, `Dpw` 70 mm, `f_i = f_o = 0.52`, steel 210 GPa / 0.3; clearance sweep
  `s` = 0, 10, 20, 30, 40, 60 um; `F_r` = 1000, 2000, 5000, 10000 N.
- 7210-like ACBB (RECORD): 13 balls, `alpha_0` 40 deg; `F_r` 3000 N, `F_a` = 4000 ... 40000 N; `F_a > 0` is the
  load ON the bearing and seats the row (`alpha_j` grows); a FEM node carries the opposite sign (R-003)
  (`F_r tan a0 / F_a` = 0.63 ... 0.063).
- NU210-like CRB (RECORD): 14 rollers 11 x 12 mm, 30 laminae, `s` 20 um, `F_r` 8000 N.
- Sign check: shaft L 200 mm, d 50 mm, two 6210 (s = 0) at 10 / 190 mm, `RadialLoad` 6600 N at 100 mm,
  `theta_deg` 270 (-Z) and 300 (`F_y` +0.5 F, `F_z` -0.866 F); `RigidSupportFEMSolver`, Timoshenko.
- Literature: Hamrock's bearing (`Dpw` 65 mm, `s` 15 um, E 200 GPa, `F_r` 8900 N); Hernot's 218 (r03 only).

## 6. Criteria (fixed before the run)
| ID | Quantity | Judge | Tolerance | Why |
|---|---|---|---|---|
| C1 | `Q_j` of every element at AxisForge's `delta_r` (max `abs(dQ_j)/Q_max`) | r03 `ring_load_distribution` | 1e-5 | AxisForge root tolerance 1e-6 (relative step, `SOLVER_TOLERANCE`), x1.5 through `Q ~ delta^1.5`; one order above |
| C2 | `Q_j` of every element at the same `F_r` | r03 `solve_ring_displacement` (element sums) | 1e-5 | same |
| C3 | `Q_max` | same | 1e-5 rel | same |
| C4 | `delta_r` | same | 1e-5 rel | same |
| C5 | `eps = (1 - s/(2 delta_r))/2` from AxisForge's `delta_r` (AxisForge has no eps): **restates C4**, identically 0.5 at `s = 0` | r03 solve | 1e-5 rel | same |
| C6 | `sum Q_j cos(a_j) cos(phi_j) = F_r` | AxisForge against its own input: **self-consistency, not a judge** | 1e-5 rel | same |
| C7 | `c_p` (every ball bearing) | Hertz series: r02 per contact, r03 Eq. 7.6 | 0.5 % | ISO/TS 16281 Eq. (11) fits the Hertz integrals (bearing-expert review) |
| C1-C6 ACBB, CRB | as above | r03 | RECORD | fixed-angle / profile-free judge (section 3) |
| L1 | Hamrock `F_max` 4513 N | AxisForge, his bearing | 1 % | his Stribeck factor from the approximate Eq. (30) and fits with non-self-consistent printed values (OPEN R5-09) |
| L2 | Hamrock `delta` 5.438e-5 m of the most loaded ball | AxisForge | 2 % | his contact fits (R5-09: delta_o of the fits vs exact Hertz 1.7 %) |
| L3 | Hamrock Z 4.564 | AxisForge `Q_max Z / F_r` | RECORD | = L1 |
| L4-L7 | Hernot `Q_max` 6544 N, `delta_a` -0.00926 mm, `delta_r` 0.08882 mm, `eps` 0.45627 | r03, continuous form (his Eq. (7)), `F_r = F_a` = 17800 N (printed 178000, R5-13) | half a unit of the last printed digit | printed digits; judges the judge (r03's continuous branch, not the discrete `alpha = 0` branch of C1-C6) |
| S1 | sign check: offset of `Q_max` from the load direction, `abs(r)/F_r` (radial load only) | physics of the inner ring | RECORD | evidence on R-003 / R-396; convention-dependent (R-003); recorded, not ruled |

## 7. How to run
```bat
python -m studies.v000_axisforge.v003_iso16281_load_zone.study
python -m pytest studies\v000_axisforge\v003_iso16281_load_zone
jupytext --sync studies\v000_axisforge\v003_iso16281_load_zone\notebook.py
jupyter nbconvert --to notebook --execute --inplace studies\v000_axisforge\v003_iso16281_load_zone\notebook.ipynb
```
Results in `results/` (git-ignored): `v003_check.csv`, `v003_elements.csv`, `v003_literature.csv`,
`v003_solver.csv`, `v003_sign_check.csv` and the figures.

## 8. Result (run of 2026-10-28)
**Judged: 174 PASS, 0 MISS** (24 deep-groove solves x C1-C7, Hamrock L1-L2, Hernot L4-L7). Of these, 126 are
judges (C1-C4, C7, L1-L2, L4-L7); 24 C5 restate C4 and 24 C6 are AxisForge's self-consistency.
- C1-C6 (6210, `s` 0-60 um, `F_r` 1-10 kN): largest error 2.9e-8 (tolerance 1e-5); C1 (r03 at AxisForge's own
  `delta_r`) exact to 6e-15: AxisForge's ISO/TS 16281 ball model with `F_a = 0` is Harris Eq. 7.10. The rest is
  AxisForge's root tolerance (residual up to 1.7e-4 N).
- `s = 0`: `eps = 0.5` and `Z Q_max / F_r` = 4.384 at every load (9 balls; 4.37 of Harris Eq. 7.23 is
  `1/J_r(0.5)`, the continuous form: calling it the many-ball limit is r03's interpretation, not printed by Harris).
- C7 `c_p`: -0.065 % against the Hertz series of r02 (same for the 7210).
- Hamrock: `F_max` 4522.5 N vs 4513 N (+0.21 %), `delta` 5.4275e-5 m vs 5.438e-5 m (-0.19 %), Z 4.573 vs 4.564.
- Hernot (r03, continuous): `Q_max` 6543.90 N (6544), `delta_a` -0.009257 mm (-0.00926), `delta_r` 0.088816 mm
  (0.08882), `eps` 0.456271 (0.45627): all to the printed digit, with `F_r = F_a = 17800 N` (printed 178000 N:
  with 178000 N r03 gives `Q_max` 65439 N; OPEN R5-13).
- RECORD 7210 (`F_r` 3000 N, `F_a` 4-40 kN, `alpha_j` 40.2-46.4 deg): `Q_max` -8.8 % ... +0.5 %, `delta_r` +3.9 ...
  +17.9 %, `eps` -3.5 ... -23.7 % against fixed-angle r03; the gap grows with the thrust (the angle grows).
  NU210 (`s` 20 um, 8000 N): `Q_max` +2.2 %, `delta_r` +18.9 % (profile in laminae).
- **Sign check S1 (R-003), RECORD:** for both load directions and both bearings the most loaded ball is **180 deg
  from the load on the shaft** (offset -180 / +180 deg: -Z load, `Q_max` at 0 deg, load at 180 deg; 300 deg load,
  `Q_max` at 330 deg, load at 150 deg) and the inner-ring balance gives **`|r| / F_r = 2.000`**. These two numbers are
  the evidence; `Q_max` at `phi_Fr` holds by construction (the element set turns with `phi_Fr`, R-396). The node force
  fed to the solver is the support force on the shaft. Evidence on R-003 and R-396 (José, 2026-10-24: not a solver
  defect; a change of convention must flip `F_a` too). Radial only (`F_a = 0`): the axial half of R-003 is not
  exercised (OPEN R5-15). Convention-dependent (R-003); AxisForge not changed.
- AxisForge's solver (SciPy root hybr/lm): 30 solves, all ok, 11-29 function evaluations, residual <= 1.7e-4 N.

Figures (`results/`): `v003_bearing_loads_element_loads_6210` (markers on the r03 line, zone edge at +-90 / about
70 deg), `..._load_zone_sweep_6210` (eps and Stribeck vs `s`, limits 0.5 and 4.37), `v003_contact_relative_errors_6210`
(C1-C6 errors under the tolerance), `..._element_loads_hamrock1983` (printed `F_max` open marker on AxisForge's ball
at 0 deg), `..._element_loads_7210` and `v003_contact_relative_errors_7210` (RECORD), `..._element_loads_sign_check_270`,
`_300` (polar, radius Q [N], global frame, load direction and `phi_Fr` dash-dot, offset and `|r|/F_r` in the
caption, R-003).

## 9. Depends on
r03 (`5019e10`: `ring_load_distribution`, `solve_ring_displacement`, `ball_deflection`, `stribeck_constant`), r02
(`hertz`, SlipPY `hertz_full`), `studies/outputs` (`l000_axisforge/plots/bearing_loads.py`,
`l100_dry_contact/plots/contact.py::relative_errors`), v101 `LiteratureCase`. AxisForge `feature/newton-solver`
`d34ec7f` (read only); design-studies `layout/studies-levels` `95bcaa9` + this case (uncommitted at the run).
OPEN: R5-10 (LiteratureCase), R5-11 (one-bearing drive through a duck-typed node), R5-12 (ACBB, CRB RECORD),
R5-13 (Hernot loads printed 178000 N, corrected in `hernot_218`), R5-15 (sign check radial only), R5-16 (no
independent variable-angle / roller-profile judge), R5-09 (Hamrock fits, from r03).
