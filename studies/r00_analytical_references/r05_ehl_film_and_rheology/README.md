# r05_ehl_film_and_rheology: EHL film formulas and lubricant rheology

**PROVISIONAL scaffold.** `ehl.py` is moved here unchanged (SW3); the content of this case is run R3. This
README states what the case is for (r_catalogue C1) and what it holds today; nothing below is validated yet.

## 1. Reference
The published dimensionless groups, film-thickness formulas and lubricant laws of elastohydrodynamic
lubrication, re-implemented as an independent judge (D-0036 amendment ruling 8, SQ-05) of the SlipPY EHL solve
through the bridge (wrapper not written yet) and of the system-model formulas. Judges: v301 ehl_circular,
v302 ehl_line (rows, blocked by the bridge EHL wrapper).
- **Today (`ehl.py`):** reduced modulus `E'` and `E*`; the groups U, G, W and Moes M, L (sum convention);
  Canzi (alpha_bar, M) to Moes (M, L); the inverse of Moes; Moes film (line and circular, compressible);
  CELINE; film scalings between Venner and Moes. Only `e_prime` is implemented: the six others are scaffolds
  that raise `NotImplementedError` after their input checks (OPEN R-091, R-093, R-114, GL-12, GL-13, GL-18).
- **R3 adds (r_catalogue C1):** Barus; Roelands and alpha0; alpha(T) (Gold or Roelands S0, P-20); Blok
  alpha*; Walther D341; Dowson-Higginson density; groups with `_mean` / `_sum` suffixes; Hamrock-Dowson
  H_min, H_c; Dowson and Dowson-Higginson line; Moes asymptotes; Nijenbanning.

## 2. Sources
| Key | What | Equation / table | PDF page |
|---|---|---|---|
| VennerBos1994 | Eq. (4); E', G (PDF 14); Moes L, M, U, H in the sum convention (Appendix, PDF 15) | (4); Appendix | 2, 14-15 |
| Canzi2010 | Moes line and circular film, compressibility, CELINE, Canzi parameters | (1) PDF 1; (3), (4), (11), (15) PDF 2; (24) PDF 6; Table 3 PDF 4 | 1-4, 6 |
| HamrockDowson1977 | point-contact film H_min, H_c; ellipticity | (1)-(5) PDF 2; (28) PDF 9; (33), Table 5 PDF 10; Table 6 PDF 11 | 2, 9-11 |
| Harris2007_vol2 | pressure-viscosity, film formulas | (4.18)-(4.19), (4.56)-(4.63) | 117-118, 127-128 |
| Hamrock1983 | elliptical film formulas | (69)-(74) | 35-36 |
| Nijenbanning1994 | elliptical film formula | (9)-(10) | 11 |
| Fernandes2015a | oil data | Table 1 PDF 2; Eq. 2 PDF 3; Table 4 PDF 5 | 2-5 |
| Johnson1985 | contact modulus E* | ch. 4 (scan: equation not checked) | 94- |
Equation numbers and PDF pages checked by the librarian on 2026-10-27 (`80_Inbox/2026-10-27/studies_sweep/_evidence/SW4_librarian.md`); Harris2007_vol2 from the text layer; Hamrock1983 by OCR; Johnson1985 unverified (scan).
**MISSING (not in the library):** Hamrock and Dowson 1976 Part I (and NASA RP-1255; the speed convention of
its U, R-092, Q-11); Roelands 1966 (Roelands law, alpha0, S0, Q-10); Blok 1963 (alpha*, Q-10).

## 3. Scope and validity
Newtonian, isothermal steady films; Roelands to about 1 GPa (R3); constants and every argument in SI, units in
the argument names. Speeds: every function takes the MEAN entrainment speed `u_mean = (u1 + u2)/2` (R-075);
Moes M and L in the SUM convention. `a` is the Hertz semi-axis along rolling (`s_x` in the bridge).
Soft fit ranges (R3, P-02): H-D k 1-8, G 2310-6785; Moes tabulated alpha_bar <= 20; equivalent line M > 20.
Domain errors raise `ValueError` (non-positive speed, viscosity, radius, load or pressure-viscosity
coefficient; not exactly one load; a compressible film without `p_h_Pa`, or with `p_h_Pa` given in MPa).

## 4. Public API
```python
from studies.r00_analytical_references.r05_ehl_film_and_rheology import (
    e_prime, groups, canzi_moes, moes_inverse, moes, celine, film_scalings)
e_prime(E1_Pa, nu1, E2_Pa, nu2) -> dict(E_prime_Pa, E_star_Pa)
# scaffolds below: input checks, then NotImplementedError
groups(*, u_mean_m_per_s, eta0_Pa_s, E_prime_Pa, R_x_m, alpha_per_Pa, load_N=None, w_N_per_m=None) -> dict
canzi_moes(*, alpha_bar, M_canzi, line, D=None) -> dict(M_moes, L_moes)
moes_inverse(*, M, L, line, E_prime_Pa, R_x_m, u_mean_m_per_s, roelands_z, roelands_p0_Pa) -> dict
moes(*, M, L, line, compressible, p_h_Pa=None) -> dict(H_c, H_m)
celine(*, M, L, D) -> dict(H_c, H_m)
film_scalings(*, h_m, R_x_m, s_x_ref_m, U_mean) -> dict
```
The results are dicts (not frozen dataclasses, structure_proposal B.2): kept as they are until R3.

## 5. Data
None held yet. R3 puts in `published/` the formula columns of HamrockDowson1977 Tables 5-6 (34 cases; case 33
prints G 3491, read 3591) and Canzi2010 Tables 2-3, 6 (their rows are in the tag `studies-pre-sweep`, old
v301-v304 `reference_values.csv`).

## 6. Criteria (`tests/test_r05.py`)
| ID | Quantity | Role | Tolerance |
|---|---|---|---|
| C1 | E' = 2 E* = E / (1 - nu^2) for equal steels; E* the Hertz modulus | identity | rel 1e-15 |
| C2 | input checks of `e_prime`, `groups`, `moes` (p_h in MPa refused) | validity | exact |
| C3 | Moes compressible film vs Canzi2010 Tables 2-3 "Hc Moes formula" (line 4.77, circular 4.05) | published (formula column) | abs 0.005; xfail strict until R3 |
| C4 | SlipPY Dowson-Higginson density tends to 1.34, constant in Pa (unit check of the solver, not of r05) | identity | rel 1e-12 / 1e-5 |

## 7. Depends on
NumPy and the standard library only (`tests/test_imports.py`); nothing in this repository, AxisForge, SlipPY or
the bridge. Only C4 imports SlipPY, in the test, to check the solver's density law. Blockers for R3: the
missing sources of section 2.

## 8. How to run
```bat
python -m pytest studies\r00_analytical_references\r05_ehl_film_and_rheology
```
No `study.py` and no `results/` (SQ-08).
