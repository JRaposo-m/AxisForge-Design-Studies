# v002 shaft stepped gear

## 1. Question
Does the AxisForge shaft FEM (`RigidSupportFEMSolver`) reproduce the closed-form stepped beam of r01 on two pins
loaded by a gear mesh **as AxisForge produces it** (`Transmission.resolve` -> `mesh_forces_to_shaft_loads`: F_c, F_t,
F_a, the torque and, for a helical pair, the couple F_a r_w), with prescribed support settlements, and are those mesh
loads right? One validation (v_prerequisites row 10): **FEM vs r01, nodal deflection and rotation at machine
precision for Euler-Bernoulli; the gear-load identities F_tw = 2000 T / d_w, F_r, F_a, the couple r x F and the
`Transmission` closure checks; Harris's two-bearing equations.** Ported from
`v000_axisforge/v002_shaft_fem_stepped_distributed_gear` at the tag `studies-pre-sweep` (sections, fillets,
material, gear pair, power, the two-shaft `Transmission`; changes in `construction.py`).

## 2. Sources
| key | item | PDF page |
|---|---|---|
| Harris2007_vol2 | sec. 9.2.2, Fig. 9.3: two-bearing shaft, loads P^k and couples T^k at a^k, bearing deflections and slopes | 278 |
| Harris2007_vol2 | Eqs 9.8-9.15: EI y'' = -M, integration on both sides of the load. **FLAG Eq. 9.14 prints M_1 x^3/2 for M_1 x^2/2** (not used) | 278-279 |
| Harris2007_vol2 | Eqs 9.20-9.23: F_1, M_1, F_2, M_2, one load | 280 |
| Harris2007_vol2 | Eqs 9.24-9.27: n loads by superposition; text: pins (M = 0) give the determinate reactions. **FLAG Eqs 9.25, 9.27 print 1/l^3 on the P^k sum where 9.21, 9.23 print 1/l^2** (a moment: 1/l^2 is right and used; the printed form is L6 RECORD) | 281 |
| Harris2007_vol2 | p. 281 prints only "See Example 9.3." (the examples are recorded as on the book's CD-ROM, J-14; not checked on these pages): **no printed digits to judge** | 281 |
| Harris2007_vol2 | Fig. 9.4 (left of the load, Eqs 9.10-9.14), Fig. 9.5 (right of it, Eqs 9.15-9.19) | 279, 280 |
| ISO_21771_2007 | working geometry through r04 `pair_geometry`: inv(alpha_wt) Eq. (55) (zero backlash) and d_w Eq. (56), doc p. 29 / PDF p. 33; beta_w from the lead, 4.3.2 Eq. (4), doc p. 18 / PDF p. 22 (pages read by the gear-expert, s006 review); the hand (4.1.4) not transcribed | 22, 33 |
| ISO_6336-1_2019 | F_t = 2000 T / d (4.2.2, Formula (1)) | **no page read in this case** |
| - | **Stepped shaft between supports: no published table held** (Roark, Pilkey not in the library); r01 is the only judge of the stepped case | - |
The rows are in `literature/published/harris2007_vol2.csv`, read from the page images on 2026-10-28 (the library note
`06_Library/bearings/Harris2007_vol2.md` has no content pass). r01 holds the stepped closed form (its README 2).

## 3. Model
Two identical stepped S355 shafts (L 200 mm; d 20 / 28 / 25 / 28 / 20 mm, steps at 20, 85, 115, 180 mm, fillets
1 mm) on ideal pins at 10 mm (locating) and 190 mm, joined by a cylindrical pair at x = 100 mm (m_n 2 mm, z 20/40,
alpha_n 20 deg, b 20 mm, no profile shift), T = 100 N.m on the pinion at 1000 rpm, sink on shaft2 at x = L.
`RigidSupportFEMSolver`, Euler-Bernoulli and Timoshenko one-point (Cowper, Hutchinson), uniform meshes n = 40 ... 320
(h = 5 ... 0.625 mm; every support, step, load and face limit on a node).
- **Stepped r01 judge** (`check.reference`): r01 `solve_stepped`, one `Segment` per `ShaftSection` (EI, kGA from r01
  `section_stiffness`). One-point Timoshenko on a uniform mesh with point loads: the exact element at
  `1/kGA_eff = 1/kGA - h^2/(12 EI)` **per segment** (v001 `kga_reduced_integration`; each element lies in one section).
- **Couples (CW-01, J-16).** The FEM adds `ExternalMoment.component(XY)` (M cos theta) to the rotation DOF of XY
  and `component(XZ)` (M sin theta) to that of XZ, in N.mm; theta is the section rotation (from +x toward +v), so this is an r01 couple of the same sign.
  Physically a moment vector acts in XY as C = +M_Z and in XZ as C = -M_Y (r01 README trap). I5 checks that what the
  FEM consumes equals r x F of the axial force at the pitch point; T5 records how wrong the reading of the names
  ("My about +Y") would be. D-0010 rows 21-22, U1, J-16 are open: **every number that depends on the couple is
  tagged `convention-dependent (J-16)`** (column `tag`, figure captions).
- **Settlements** (`# OPEN R5-31`): AxisForge has no prescribed support displacement. `study.solve` redoes the
  elimination on AxisForge's own K, f_ext and DOF partition with d_c = delta at each pin (`K_ff d_f = f_f - K_fc d_c`)
  and passes the result through AxisForge's `build_shaft_result`. Judge: r01 plus the rigid-body field
  `delta_A + (delta_B - delta_A)(x - x_A)/(x_B - x_A)` (determinate beam: no strain, reactions unchanged;
  `check.rigid_body`; r01 has no settlement until part G1).
- Planes and signs as v001 (forces along +v, theta = section rotation, `R_*` = force of the support on the shaft along +v).
- **One shaft comparator** (v002 minimal review): `check`, `reference`, `segments`, `rigid_body`, `along_shaft` live in
  `v001_shaft_uniform/check.py` (stepped, couples, settlements, C7 axial statics) and v002 re-exports them; v002's own
  `check.py` holds the gear side: `mesh_judge` (the mesh forces WITH SIGNS from the inputs: torque, hand, rotation,
  driver; geometric derivation of the axial sign, no library source, R5-36), `physical_couples` (r x F),
  `gear_identities`.
- **Distributed variant**: F_c and F_t are spread uniformly over the face width, but F_a and the couple stay point loads
  at x = 100 mm (AxisForge's choice, `transmission.py:166-187`).

## 4. Public API (`__init__.py`)
```python
check, reference, segments, rigid_body, along_shaft                  # re-exported from v001 check (one comparator)
mesh_judge(*, torque_Nm, beta_deg, mn_mm, z1, z2, alpha_n_deg, b_mm, x1=0, x2=0, hand1=0, rot1=1, gear1_drives=True)
# magnitudes from r04 pair_geometry + mesh_forces (the v calls its r), signs from hand, rotation, driver
physical_couples(F_x_N, rho_mm, theta_c_deg) -> (C_XY, C_XZ) [N.mm]
gear_identities(resolved, judge, *, theta_c_deg) -> list[dict]       # I1-I3, I3s, I4-I8, T5
literature.cases() -> [LiteratureCase]; literature.harris_eqs_9_24_9_27(...), harris_pin_slopes(...)
gear_system(*, beta_deg=20, hand=+1, phi_deg=30, distribute=False, torque_Nm=100) -> ResolvedTransmission (unsolved)
stepped_shaft(name, material_id=None) -> Shaft;   uniform_nodes, THEORIES (v001)
study.solve(system, settings, n_elements, *, settlements_mm=None, kGA_override=None) -> ShaftResults
```
`check` takes any shaft on two pins (stepped or uniform; forces, distributed forces, couples, settlements), so s004 /
s006 / s003 can pass theirs. `LiteratureCase` from v101 (`# OPEN R5-22`); v001 comparator, pins (`add_pins`),
`node_value`, `richardson_verdict` imported (`# OPEN R5-30`).

## 5. Inputs
Base: beta 20 deg, hand +1 (pinion right-hand, wheel left), phi 30 deg (both planes carry force and couple),
point loads, settlements XY (A 0.02, B -0.03) mm, XZ (A -0.01, B 0.04) mm. Sweeps (n = 40, rigid pins unless said):
helix beta 0 ... 30 deg x hand +-1 at phi 0 (couple in XY only); phi 0, 45, ..., 270 deg at beta 20; settlement
delta_B 0 ... 0.5 mm, delta_A = -delta_B/2, both planes; distributed variant (F_c, F_t uniform over the face width),
base settlements, n 40, 80. Literature: uniform shaft d 28 mm, pins at 0 and l = 180 mm, P = 4000 N and T = 3.6e4 N.mm
at 60 mm, P = -1500 N and T = -2.5e4 N.mm at 130 mm (plane XY, +v = Harris's y: the case's own reading, Harris states no y direction), delta_r1 0.02, delta_r2 -0.03 mm,
EB, n 36. `# OPEN R5-32`: face width 15 -> 20 mm and no parabolic user load (old case), so a 5 mm grid exists.

## 6. Criteria (fixed before the run)
Field errors: largest nodal deviation over the largest absolute r01 value of the field in the plane.
| id | quantity | tolerance | source of the tolerance |
|---|---|---|---|
| C1, C2 | Euler-Bernoulli nodal v, theta vs r01 (+ settlement field) | rel 1e-9 | Hermite nodal exactness (v001, brief) |
| C3 | reactions, every model, vs r01 | rel 1e-9 | statics (determinate) |
| C4 | one-point Timoshenko v, theta vs r01 at per-segment kGA_eff, uniform mesh, point loads | rel 1e-9 | Onate (2.35) vs (2.40) (v001) |
| C5 / C6 | Richardson of v_xy at the gear (shaft1): EB `flat` (level differences <= flat_rtol 1e-8 relative, recorded as tol); Timoshenko `monotonic` and extrapolation within 1e-9 of r01 | as stated | nodal exactness; error exactly c h^2 |
| C7 | axial reaction at the locating pin = -sum(AxialLoad), none at the other (added at review) | rel 1e-9 | statics |
| I1-I6, I3s, I8 | F_tw (from the input torque), F_r, F_a; signed F_1,x (I3s); transverse load per shaft; couple (consumed vs r x F of the signed judge); couple modulus vs F_a r_w of that shaft; AxialLoad per shaft vs the signed F_x (I8). I3s, I8 added at review | rel 1e-12 | float identities (brief) |
| I7 | `Transmission` closure checks | their own flag (rel 1e-9) | AxisForge |
| L1-L5 | Harris Eqs 9.24-9.27 at the FEM slopes: F_1, F_2 = -R; M_1 = M_2 = 0 (pins); pin slopes | rel 1e-9 | printed equations, no printed digits (p. 281: "See Example 9.3." only) |
| T1, T5, R1, L6, K1 | Timoshenko vs r01 at kGA; named reading of the couple; nodal M; Harris as printed (1/l^3); reaction shift of the couple vs statics | RECORD | - |

## 7. How to run
`python -m studies.v000_axisforge.v002_shaft_stepped_gear.study` (about 25 s) writes `results/` (git-ignored):
`v002_check.csv`, `v002_identities.csv`, `v002_richardson.csv`, `v002_couple.csv`, `v002_literature.csv`, and
`v002_shaft_deflection_along_shaft_{XY,XZ}_n40.png`, `v002_shaft_error_vs_elements.png`,
`v002_shaft_ratio_sweep_couple_shift.png` (`studies/outputs/l000_axisforge`), `v002_contact_relative_errors_{settlement,
helix}.png` (`l100_dry_contact` `relative_errors`, which floors errors at 1e-10 %). `notebook.ipynb` (jupytext pair of
`notebook.py`) runs the same and is committed with its outputs. Tests: `python -m pytest
studies/v000_axisforge/v002_shaft_stepped_gear` (about 10 s).

## 8. Result (run of 2026-10-28)
**908 of 919 judged check records PASS; 11 MISS, all Euler-Bernoulli round-off at the finest meshes (the
conditioning of the problem, see C1); every identity (399), Harris L1-L5 and Richardson C5 / C6 PASS.** (Re-run after
the review: the same 11 MISS; C7, I3s, I8 added, all PASS.)
| id | largest error | note |
|---|---|---|
| C1 / C2 | 5.6e-12 (n 40), 1.1e-10 (80), 7.4e-10 (160), **5.7e-9 / 6.0e-9 (320): MISS** | round-off from the h^-4 conditioning of the Hermite beam stiffness (cond x16 per halving, measured 4.1e9 at n 320 by the physics review); the one-point Timoshenko element is h^-2 (1.4e8). The error grows x8 per halving, inside the cond eps bound; not a defect (R5-33) |
| C3 | 1.2e-11 (40) ... **1.0e-9 (160, one record), 7.4e-9 (320): MISS** | as C1 |
| C4 | 4.2e-13 ... 3.0e-12, every mesh, both shafts, sweeps | h^-2 conditioning: no visible loss |
| C5 / C6 | flat at every triple (flat_rtol 1e-8; the n 320 value is 2.4e-9 off r01, inside it, while C1 MISSes) / p = 2.0000000, extrapolation 1.0e-12 (Cowper), 3.0e-12 (Hutchinson) | |
| C7 | 1.4e-13 | axial reaction, every helical solve |
| settlement, helix, phi, distributed sweeps (C1-C4) | 7.6e-11 / 2.3e-11 / 1.5e-11 / 1.6e-10 | delta 0 ... 0.5 mm; beta 0 ... 30 deg, both hands |
| I1-I6, I3s, I8 | 7.3e-14 | signed judge from the inputs; J-16 on I5, I6 |
| L1 / L2 | F_1 2188.888889 N, F_2 311.111111 N; 1.6e-12 / 4.1e-12 | |
| L3 / L4 / L5 | 2.2e-13 / 8.8e-13 / 1.3e-12 | Harris's pins = AxisForge's pins, settlements included |
RECORD: T1 Timoshenko vs r01 at kGA 9.5e-4 (n 40) -> 1.5e-5 (320), x4 per halving; distributed variant C4 (not
judged: point-load identity) 1.0e-3 at n 40, O(h^2). T5 the named reading of the couple differs from r x F by 1.00 ... 1.41
of that shaft's couple F_a r_w (largest component; a 90 deg turn): the names `My`, `Mz` do not describe what the FEM consumes,
the consumption is physically right (I5). L6 Harris as printed: M_1 = -9.1e4 N.mm, M_2 = -1.4e4 N.mm instead of 0
(64 % / 10 % of the moment scale): the misprint is real. K1 couple share of R_A = C / span within 1.5e-10 for every
beta and hand (`v002_couple.csv`). R1 nodal M up to 0.98: at the couple node only, where M jumps by -C (r01
right-continuous, `ShaftResults` holds the left element's value).
**First run (same day), changed before this one**: C4 was also applied to the distributed variant and gave 32 MISS
(1.0e-3 at n 40, 2.5e-4 at 80: O(h^2)). The kGA_eff identity needs nodal loads (v001 README 3); the consistent load
vector of the linear element under a distributed load is not the exact element's. C4 is now restricted to point
loads (`check.py`); the distributed Timoshenko records are T1. No tolerance was changed.

## 9. Depends on
r01 (`solve_stepped`, `solve_uniform`, `section_stiffness`), v001 (the shaft comparator of `check.py`, `section`,
`node_value`, `richardson_verdict`, `add_pins`, `ensure_material`, `uniform_nodes`, `uniform_shaft`, `THEORIES`; changed
in this run for v002), v101 `LiteratureCase`, `numerical_solvers.richardson`, `studies/outputs` (`l000_axisforge/plots/shaft.py`,
`l100_dry_contact/plots/contact.py` `relative_errors`). AxisForge `feature/newton-solver`
`d34ec7f` (read only); design-studies `layout/studies-levels` `95bcaa9` + this case (uncommitted at the run).
OPEN: R5-21 (pin as a zero-width bearing, v001), R5-22 (LiteratureCase), R5-30 (v002 imports v001 helpers),
R5-31 (settlements imposed on AxisForge's K), R5-32 (face width 20 mm, no parabolic load), R5-33 (numerics note:
the nodal-exactness tolerance ignores the h^-4 conditioning), R5-34 (Harris Eqs 9.25 / 9.27 misprint), R5-36 (axial-force
sign vs hand: judged against a geometric derivation, no library source), R5-37 (`ExternalMoment` docstring: "no producer").
The `ExternalMoment` names vs consumption (CW-01) is evidence on the existing R-174 / R-565, not a new row.
Next run (physics review): judge the distributed Timoshenko variant by a C6-type Richardson criterion (p = 2, to r01
at kGA).
