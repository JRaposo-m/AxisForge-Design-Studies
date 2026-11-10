# r02_hertz_contact: Hertz contact of two bodies as given

## 1. Reference
The Hertz solution of two smooth elastic bodies in contact, each given as it is (its own principal radii,
its material) with the load and the angle between them: elliptical or circular point contact and line
contact (plane strain), computed by SlipPY `hertz_full` (D-0036 amendment rule 13: the one library
function an r may import, analytical formulas José verified against Johnson and Hamrock-Brewe on
2026-10-27; SQ-05 reversed for Hertz). The own exact solution of R1b (Carlson `R_D`, with its own
curvature-tensor sum since R1d) is a helper of `tests/test_r02.py`, the check of `hertz_full`. Reference
of v101, v102, v103 (`K_n` for v002) and of the AxisForge ISO/TS 16281 contact stiffness. The v101
criterion "bridge hertz vs r02" is no longer a check: the bridge (`slippy_bridge.contact.solvers.hertz`)
and r02 both run `hertz_full`. Not the same call: r02 passes `root_error = 1e-12`, the bridge keeps
SlipPY's 1e-6, so the two differ by up to 6.5e-8 relative (OPEN RC-01); a v101 tolerance below about
1e-7 between them would test that setting, not Hertz.

## 2. Sources
| Key | What | Equation / table | PDF page |
|---|---|---|---|
| Johnson1985 | geometry of non-conforming surfaces, principal relative curvatures | Sec. 4.1, (4.3)-(4.5) | 95 |
| Johnson1985 | two cylinders crossed at an angle (example) | Sec. 4.1 | 96-97 |
| Johnson1985 | Boussinesq point load on a half-space (test C11) | Sec. 3.2, (3.22b) | 62 |
| Johnson1985 | pressure p0 (1 - r^2/a^2)^(1/2) on an ellipse: displacement L, M, N | Sec. 3.5(b), (3.61)-(3.63) | 75 |
| Johnson1985 | solids of revolution (sphere): a, p0, delta | Sec. 4.2(a), (4.19)-(4.24) | 102-103 |
| Johnson1985 | elliptical contact: semi-axes, approach from K(e), E(e) | Sec. 4.2(b), (4.25)-(4.33) | 105-107 |
| Johnson1985 | line contact: gap, semi-width, p0; approach not defined locally | Sec. 4.2(c), (4.36), (4.43)-(4.45) | 110-111 |
| HamrockBrewe1983 | curvature sum, R_x, R_y, alpha = R_y/R_x | (2), (4)-(6) | 2 |
| HamrockBrewe1983 | k = D_y/D_x; exact D_y, D_x, delta, E', sigma_max; iteration for k | (7)-(17) | 3 |
| HamrockBrewe1983 | fits k_bar (29), E_bar (30), q = pi/2 - 1 (31), F_bar (32), alpha >= 1 | (29)-(32) | 4 |
| HamrockBrewe1983 | Table 1: exact and fitted k, F, E, alpha 1-100 | Table 1 | 4 |
| HamrockBrewe1983 | Table 2: exact and simplified D_y, D_x, sigma_max, delta | Table 2 | 5 |
| (DLMF) | Carlson's R_D form of (K - E)/m and (E - k'^2 K)/(m k'^2) (test helper) | DLMF 19.25.1 | not in the library |
| SlipPY `hertz_full` | the computation: Johnson Sec. 4.2, b/a from Deeg (1996); `slippy/contact/hertz.py`, branch `local-fixes`, commit `df5916c` | - | - |
Every equation number and page checked on the scans (R1b librarian, `_evidence/R1b_librarian.md`); DLMF
19.25.1 unverified (the identities hold numerically to about 1e-13).
Book page = PDF page - 10; HamrockBrewe1983 journal page = PDF page + 170.

## 3. Scope and validity
- Elastic isotropic half-spaces, smooth and frictionless surfaces, normal load only, no adhesion. Each
  body is a paraboloid near the first contact with its own principal axes; the two may be turned by any
  angle (Johnson Sec. 4.1). Line contact: plane strain, both bodies straight along the line.
- **Validity (P-02).** The 0.1 limit is P-02 policy (Johnson Sec. 4.2 asks "much smaller than"). Soft
  range, returned in `validity` (`in_range`, `governing`, `ratio`), not refused: for each body, the
  largest semi-axis `a` (line: the half-width `b`) over the smallest finite `|radius|` of THAT body
  (concave radii count, flats are skipped), the larger of the two against `SIZE_RATIO_MAX = 0.1`;
  `governing` is `"a/r1"`, `"a/r2"`, `"b/r1"` or `"b/r2"`. A v criterion outside it is RECORD. This is
  Johnson's condition (iii) (each body); (i) and (ii), against the relative radii, are not checked (up to
  2x lower for two equal balls; OPEN RD-07). With the bodies given as they are, every radius of a
  conforming groove pair enters, the groove radius included; before R1d a pair at an angle had to be given
  as its equivalent body, which hid the radii of both bodies. The groove radius can never set the ratio:
  a positive relative curvature along a concave radius needs a tighter convex radius on the other body,
  so in a groove the ball radius governs. A ball in a groove leaves the range early (6.35 mm ball, groove
  0.52 D_w: near 70 N): whether the per-body ratio sets the range or is only reported is OPEN MG-08
  (José). Fits: `alpha <= 100` (`ALPHA_FIT_MAX`) soft; `alpha < 1` raises (branch of Table 4 not
  implemented). No clamp: `clamp_acted` is always False.
- **Domain errors** raise. From r02: a radius, modulus or Poisson's ratio that is not a number or a pair
  of them (`TypeError`; NumPy scalars are numbers, `bool` is not), `E <= 0` or not finite, `nu` outside
  (-1, 0.5), a load that is not finite (`hertz_full` checks none of them; its own zero-modulus check never
  runs, OPEN RD-08).
  From `hertz_full`, passed through as they are (`ValueError`): a zero radius, a load `<= 0`, a negative
  relative radius (a concave body tighter than the convex one), two flats (conformal), a line contact
  without `line=True` (one relative radius over 1e5 times the other) and `line=True` on a point contact.
- **Range of `hertz_full`:** a relative curvature ratio above about 7.9e4 raises (its root search for
  b/a is bracketed in [1e-5, 1] and loses the sign change, though the true b/a is about 1.4e-3; OPEN
  RC-02, SlipPY), above 1e5 it is a line contact. The R1b solution reached A/B 1e6; nothing within the
  Hamrock-Brewe range (alpha <= 100) is affected. Two bodies that are both more curved along `y` than
  along `x` raise "There is a bug in this program please report" at any angle (its two `arctan2` land on
  the same branch, `alpha + beta = angle +/- pi`, and the check is not taken modulo `pi`; OPEN RD-03,
  SlipPY): give one of them with its axes swapped and add `pi/2` to `angle_rad` (tested); `alpha_rad` or
  `beta_rad` then counts from that body's swapped axes. A line contact (`line=True`) always returns
  `a_m = inf`, also where `hertz_full` leaves a finite relative radius above its 1e5 ratio.

## 4. Public API
```python
from studies.r00_analytical_references.r02_hertz_contact import hertz, hamrock_brewe_fit
hertz(r1_m, r2_m, E_Pa, nu, load, *, angle_rad=0.0, line=False,
      integration_error=1e-6, root_error=ROOT_ERROR) -> HertzResult      # ROOT_ERROR = 1e-12
    # shape, a_m >= b_m, p0_Pa, pm_Pa, delta_m, area_m2, E_star_Pa, Rx_rel_m, Ry_rel_m, R_e_m,
    # alpha_rad, beta_rad, k_n_N_per_m (= 3 F / (2 delta)), validity, raw (the hertz_full dict)
hamrock_brewe_fit(alpha) -> dict      # k, E, F, validity (Eqs (29)-(32))
Validity(in_range, governing, ratio, clamp_acted)   # frozen; dataclasses.asdict for records
hertz_point(R1x_m, R1y_m, R2x_m, R2y_m, E1_Pa, nu1, E2_Pa, nu2, F_N)   # pre-R1d call, = hertz at angle 0;
    # kept only for tests/test_slippy_bridge.py (RC-03)
```
`hertz` passes its inputs to `hertz_full` as given (only the type, material and load checks of section 3)
and maps the outputs; the mapping is in its docstring. Results are frozen dataclasses. `Validity` is defined
here; an r imports no other r (each defines the same four fields).

**Conventions (those of `hertz_full`).**
- **Bodies.** `r1_m`, `r2_m`: `(R_x, R_y)` of each body in its OWN principal axes, or one number for a body
  of revolution. A radius is **positive for a convex surface, negative for a concave one**, `math.inf` for
  a flat. Do not pre-reduce the pair to an equivalent body against a flat: the result is the same, but the
  per-body validity then sees only the relative radii.
- **Angle.** `angle_rad` is the angle between the `x` axis of body 1 and the `x` axis of body 2; only its
  value modulo `pi` matters (the result is the same for `-angle_rad` and for a swap of the bodies).
- **Semi-axes and orientation.** `a_m >= b_m` (Johnson; before R1d `b` lay along `x`). The major
  semi-axis `a` lies at `alpha_rad` from the `x` axis of body 1 and at `-beta_rad` from the `x` axis of
  body 2, in the sense of `angle_rad` (`alpha_rad + beta_rad = angle_rad` modulo `pi`, Johnson Sec. 4.1),
  as the tests establish against the eigenvectors of the curvature sum. `hertz_full`'s own `alpha`,
  `beta` (in `raw`) are the direction of the largest summed curvature, along `b`; r02 turns them by `pi/2`
  when `Rx_rel_m < Ry_rel_m`, as `hertz_full` does inside its pressure closure (OPEN RD-05: r02's choice).
  `Rx_rel_m <= Ry_rel_m` are `hertz_full`'s `relative_radii`: the smaller along `b`, the larger along `a`,
  not along the `x` axis of a body. OPEN RD-04 (SlipPY): `raw["pressure_f"](x, y, transform=1)` and
  `raw["surface_displacement_b_f"][i](x, y, transform=1)` are the mirror image (`y -> -y`) of that body-1
  field; `transform=0` (contact axes) and `transform=2` (body 2) agree with it.
- **Load and units.** SI throughout: radii [m], moduli [Pa], load [N] for a point contact or [N/m]
  with `line=True`. `E_Pa`, `nu`: one number for both bodies or `(body 1, body 2)`.
- **Tolerances.** `integration_error = 1e-6` (SlipPY's default; `hertz_full` uses it in its root function
  only, OPEN RD-09) and `root_error = ROOT_ERROR = 1e-12` (SlipPY's default 1e-6 leaves up to 6.5e-8
  relative on a, b, p0, delta; 1e-12 leaves 6e-12: OPEN RC-01).

### Inputs by hand and from AxisForge
Every row: SI, radii signed as above. AxisForge gives mm and MPa: the caller multiplies radii by 1e-3
and moduli (`e1`, `e2` on a bearing assembled with ISO/TS 16281 contact) by 1e6.

| Contact | `r1_m` | `r2_m` | `angle_rad` | `line` | `load` |
|---|---|---|---|---|---|
| ball on a flat | `R` | `inf` | 0 | False | `F` [N] |
| two balls | `R1` | `R2` | 0 | False | `F` |
| crossed cylinders | `(R1, inf)` | `(R2, inf)` | the angle between their axes (90 deg with equal R: a sphere R on a flat) | False; True only at 0 | `F`; `w` [N/m] at 0 |
| ball in a groove, `x` along rolling | `R_ball` | `(R_race, -R_groove)`, `R_race > 0` inner, `< 0` outer | 0 | False | `F` |
| same, groove axes in the other order | `R_ball` | `(-R_groove, R_race)` | 0 (same contact, `alpha_rad` turned by `pi/2`) | False | `F` |
| gear flanks, spur (`CylindricalFamily.pair_values_at`) | `(rho1_t_mm * 1e-3, inf)` | `(rho2_t_mm * 1e-3, inf)`, negative on an internal ring | 0 | True | normal load per unit length of contact line [N/m]: `F_bn / (l_sum_mm * 1e-3)` (`load_sharing`; one pair: `F_bn / b`) |
| gear flanks, helical | `(rho1_n_mm * 1e-3, inf)` | `(rho2_n_mm * 1e-3, inf)`, normal plane, `rho_t / cos(beta_b)` | 0 | True | `F_bn / (l_sum_mm * 1e-3)`, the bridge's L0 (D-0035); OPEN RD-06 |
| ball bearing (`DeepGrooveBallFamily`, `AngularContactFamily`) | `r_rolling_el[0] * 1e-3` = `Dw / 2` | inner `(r_inner[0], r_inner[1]) * 1e-3` = `(r_bx_inner, -ri)`; outer `(-r_outer[0], r_outer[1]) * 1e-3` = `(-r_bx_outer, -re)` | 0 | False | `Q_j` [N] |
| cylindrical roller bearing (`CylindricalRollerFamily`) | `(r_rolling_el[0] * 1e-3, inf)` = `(Dwe / 2, inf)` | inner `(r_inner[0] * 1e-3, inf)`; outer `(-r_outer[0] * 1e-3, inf)` | 0 | True | `Q_j / (Lwe * 1e-3)` [N/m]; per lamina `q_jk / (Lwe * 1e-3 / n_s)` |

AxisForge keys: `pair_values_at` returns `rho1_t_mm`, `rho2_t_mm` (transverse plane), `rho1_n_mm`,
`rho2_n_mm` (plane normal to the contact line), `rho_red_t_mm`, `rho_red_n_mm`, `g_y_mm` [mm], signed (the
concave flank of a ring negative, D-0010 item 8); `load_sharing(pair, g_y)` returns `share`, `l_mm`,
`l_sum_mm`, `n_pairs`. A bearing carries `r_rolling_el`, `r_inner`, `r_outer` [mm] from
`BallBearingGeometry` / `CylindricalRollerBearingGeometry`: `r_by_*` is already negative (concave groove),
but **`r_bx_outer` is a positive magnitude**: the caller puts the minus sign of the concave outer raceway
(as `slippy_bridge.contact.core.machine_elements.bearings` does). The ball rows cover the deep-groove and
angular-contact families only (a thrust ball bearing stores `r_inner = [inf, r_by]`). The rolling radii are
at `alpha_0` (the bridge evaluates them at `alpha_j`, SB-48). Helical: OPEN RD-06. The ball-in-groove rows
are checked in the tests (both axis orders; the Hamrock-Brewe Table 2 rows as a ball in a straight groove).

## 5. Data
`published/hamrock_brewe_1983_t1_t2.csv`: the 549 rows of HamrockBrewe1983 Tables 1-2 (exact,
fitted and percent-error columns, 26 values of alpha, and the parameters E' = 2.1972e7 N/cm^2,
F = 4.448 N, R_x = 1.0 cm), moved unchanged from `v100_dry_contact/v102_hertz_ellipse/reference_values.csv`
(schema of that file; `case_id` still `V-DRY-02`; one file for both tables as structure_proposal B.4, not
one per table as B.1: OPEN MG-03). Notes in the rows: the printed percent error is
(z - z_bar) 100 / z, opposite in sign to Eq. (33); several Table 1 cells were read from the image
(OCR prints 1 for 4).
Rounding: Table 2 `sigma_max_bar` at alpha 1.25 is printed 0.440e+05 N/cm^2, Eqs (13)-(17) with the
fits give 0.439499e+05 (0.0003 % below the rounding boundary 0.4395e+05): checked to one unit of the last digit.
Not held: Table 5 and the 0.01 <= alpha < 1 branch.

## 6. Criteria (`tests/test_r02.py`)
| ID | Quantity | Role | Tolerance |
|---|---|---|---|
| C1 | Table 1 exact k, F, E (26 alpha) from `hertz` | published | half a unit of the last printed digit |
| C2 | Table 1 fitted k, F, E from `hamrock_brewe_fit` | published | same |
| C3 | Table 2 exact D_y, D_x, sigma_max, delta, as the equivalent body on a flat and (R1d) as a ball in a straight conforming groove (25 alpha > 1) | published | same |
| C4 | Table 2 simplified columns, Eqs (13)-(17) with the fits | published | same (one unit at C3's one cell, section 5) |
| C5 | Eqs (13)-(17) with r02's own k, E, F give its a, b, p0, delta | identity | rel 1e-10 |
| C6 | crossed cylinders R 10 / 10 mm at 45 deg, steel, 100 N, as two bodies at 45 deg and as the principal relative radii: p0 1178.488 MPa, b 0.112690 mm, delta 2.97689 um (cowork artifact A10) | reference | half a unit of the last printed digit |
| C7 | line R 10 / 20 mm, 1e5 N/m: p0 742.240 MPa (A10); a line without `line=True` and `line=True` on a point raise | reference | same |
| C8 | the sphere on a flat, a, p0, delta, area; equal cylinders crossed at 90 deg are a sphere R on a flat | closed form | rel 1e-12 |
| C9 | invariances: body swap, angle -> -angle, angle -> angle + pi; the groove in either axis order; a concave radius is the equivalent convex body 1/R_ball - 1/abs(R) | identity | rel 1e-12 / 1e-14 |
| C10 | load tie F = (2/3) pi a b p0, pm, k_n against a finite difference | identity | rel 1e-12 / 1e-7 |
| C11 | Boussinesq: the pressure closes the gap at four points | self-check | rel 1e-8 |
| C12 | line: Johnson (4.43)-(4.45), both forms of p0, pm | closed form | rel 1e-12 |
| C13 | domain and type errors raise (hertz_full's pass through; its 'bug' message at three angles and the swapped-axes workaround, RD-03); a line has `a_m = inf` (line along x, ratio above 1e5); the per-body size ratio is a record | validity | exact / rel 1e-10 |
| C14 | `hertz` (`hertz_full`) against the exact solution of the tests (own curvature-tensor sum, Carlson `R_D`), a, b, p0, delta, 19 contacts (12 of R1c at angle 0, 5 at an angle, 2 in a groove); `Rx_rel`, `Ry_rel`, `R_e` against the eigenvalues | check of `hertz_full` | rel 1e-10 (reached 6.0e-12) / 1e-12 |
| C15 | orientation: `alpha_rad`, `-beta_rad` on the eigenvector of the smaller relative curvature (8 contacts); `raw["pressure_f"]` axes for transform 0, 1, 2 (OPEN RD-04) | check of `hertz_full` | 1e-12 / 1e-4 (angle grid) |

## 7. Depends on
SlipPY `slippy.contact.hertz.hertz_full` (branch `local-fixes`, commit `df5916c`, `pip install -e ..\slippy`)
and the standard library; the tests also use NumPy and SciPy. Nothing in `studies/`, AxisForge or the bridge
(rule 13, checked by `tests/test_imports.py` with its allow-list); the AxisForge names of section 4 are
documentation only. Blockers: none.

## 8. How to run
```bat
python -m pytest studies\r00_analytical_references\r02_hertz_contact
```
No `study.py` and no `results/` (SQ-08): the tests against the published tables are the case.
