# r02_hertz_contact: Hertz point and line contact

## 1. Reference
The Hertz solution of two smooth elastic bodies in contact: elliptical point contact from the exact
elliptic integrals, and line contact (plane strain). It is the independent judge of the bridge
(`slippy_bridge.contact.solvers.hertz`, i.e. SlipPY `hertz_full`) and of the AxisForge ISO/TS 16281
contact stiffness, so it re-implements the formulas and imports none of them (SQ-05, D-0036 amendment
ruling 8). Judges: v101, v102, v103 (`K_n` for v002).

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
| (DLMF) | Carlson's R_D form of (K - E)/m and (E - k'^2 K)/(m k'^2) | DLMF 19.25.1 | not in the library |
Every equation number and page checked on the scans (R1b librarian, `_evidence/R1b_librarian.md`); DLMF
19.25.1 unverified (the identities hold numerically to about 1e-13).
Book page = PDF page - 10; HamrockBrewe1983 journal page = PDF page + 170.

## 3. Scope and validity
- Elastic isotropic half-spaces, smooth and frictionless surfaces, normal load only, no adhesion;
  principal directions of the two bodies coincide (bodies at an angle: give the principal relative
  radii of the pair as body 1 against a flat, Johnson Sec. 4.1). Line contact: plane strain, both
  bodies straight along `y`.
- **Axes and signs.** `x` is the rolling direction, `y` lies across it. A radius is **positive for a
  convex surface, negative for a concave one**, `math.inf` for a flat (an outer raceway along `x` and a
  groove along `y` are negative). `b` is the semi-axis along `x`, `a` along `y` (Hamrock-Brewe
  `D_x = 2 b`, `D_y = 2 a`); either may be the larger one (not Johnson's `a >= b`: OPEN MG-05). SI units, in the argument names.
- **Validity (P-02).** The 0.1 limit is P-02 policy (Johnson Sec. 4.2 asks "much smaller than"). Domain errors raise `ValueError`: a zero radius, `E <= 0`, `nu` outside
  (-1, 0.5), a load `<= 0`, a relative curvature `<= 0` (line contact given to `hertz_point`, or a
  concave body tighter than the convex one). Soft range: semi-axis / radius `<= 0.1`
  (`SIZE_RATIO_MAX`) over the relative radii of the pair and every finite radius of each body (Johnson's
  conditions (i)-(iii)), returned in `validity` (`in_range`, `governing`, e.g.
  `"a/R1y"`, `ratio`); a v criterion outside it is RECORD. A ball in a groove leaves the range early
  (6.35 mm ball, groove 0.52 D_w: near 70 N; on the 6210 at 1000 N a/R_ball is 0.24, a/R_relative 0.009):
  whether the per-body ratio sets the range or is only reported is OPEN MG-08. Fits: `alpha <= 100` (`ALPHA_FIT_MAX`) soft;
  `alpha < 1` raises (branch of Table 4 not implemented). No clamp: `clamp_acted` is always False.

## 4. Public API
```python
from studies.r00_analytical_references.r02_hertz_contact import hertz_point, hertz_line, hamrock_brewe_fit
hertz_point(R1x_m, R1y_m, R2x_m, R2y_m, E1_Pa, nu1, E2_Pa, nu2, F_N) -> HertzPoint
    # a_m, b_m, p0_Pa, pm_Pa, delta_m, k_n_N_per_m (= 3 F / (2 delta)), E_star_Pa, validity
hertz_line(R1_m, R2_m, E1_Pa, nu1, E2_Pa, nu2, w_N_per_m) -> HertzLine
    # b_m, p0_Pa, w_N_per_m, validity (no approach: it depends on the bodies, Johnson Sec. 4.2(c))
hamrock_brewe_fit(alpha) -> dict      # k, E, F, validity (Eqs (29)-(32))
Validity(in_range, governing, ratio, clamp_acted)   # frozen; dataclasses.asdict for records
```
Results are frozen dataclasses. `Validity` is defined here; an r imports no other r (each defines the
same four fields).

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
| C1 | Table 1 exact k, F, E (26 alpha) from `hertz_point` | published | half a unit of the last printed digit |
| C2 | Table 1 fitted k, F, E from `hamrock_brewe_fit` | published | same |
| C3 | Table 2 exact D_y, D_x, sigma_max, delta | published | same |
| C4 | Table 2 simplified columns, Eqs (13)-(17) with the fits | published | same (one unit at C3's one cell, section 5) |
| C5 | Eqs (13)-(17) with the judge's own k, E, F give its a, b, p0, delta | identity | rel 1e-10 |
| C6 | crossed cylinders R 10 / 10 mm at 45 deg, steel, 100 N: p0 1178.488 MPa, b 0.112690 mm, delta 2.97689 um (cowork artifact A10) | judge | half a unit of the last printed digit |
| C7 | line R 10 / 20 mm, 1e5 N/m: p0 742.240 MPa (A10) | judge | same |
| C8 | alpha = 1: the sphere on a flat, a, p0, delta | closed form | rel 1e-12 |
| C9 | x-y swap swaps a and b; body swap (concave raceway as body 1) gives the same contact; a concave radius is the equivalent convex body 1/R_ball - 1/abs(R) | identity | rel 1e-14 |
| C10 | load tie F = (2/3) pi a b p0, pm, k_n against a finite difference | identity | rel 1e-12 / 1e-7 |
| C11 | Boussinesq: the pressure closes the gap at four points | self-check | rel 1e-8 |
| C12 | line: Johnson (4.43)-(4.45), both forms of p0 | closed form | rel 1e-12 |
| C13 | domain errors raise; the size ratio is a record | validity | exact |

## 7. Depends on
SciPy (`ellipk`, `elliprd`, `brentq`) and the standard library; nothing in this repository, AxisForge, SlipPY or the
bridge (checked by `tests/test_imports.py`). Blockers: none.

## 8. How to run
```bat
python -m pytest studies\r00_analytical_references\r02_hertz_contact
```
No `study.py` and no `results/` (SQ-08): the tests against the published tables are the case.
