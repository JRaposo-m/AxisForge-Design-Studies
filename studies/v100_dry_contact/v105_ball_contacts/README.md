# v105 ball contacts

## 1. Question
Does `slippy_bridge` turn a solved AxisForge ball bearing into the right ball-raceway contacts? This is a check of
the TRANSFER AxisForge -> bridge (`DeepGrooveBallContact.from_bearing`, `AngularContactBallContact.from_bearing`),
not of Hertz: both the bridge and r02 run SlipPY `hertz_full` (Hertz itself is judged in v101/v102). For every loaded
ball, inner and outer: the element's load, azimuth and angle against AxisForge's result; the bridge's Hertz (`a`, `b`,
`p0`, `delta`) of the contact it builds against r02 of the ball and groove built from AxisForge's own attributes (v003
`r02_bodies`) at AxisForge's `Q_j`; the size of each contact (r02's per-body ratio and the bridge's, RECORD); the
Hertz series of the two contacts against r02's series. The independent judges of the contact geometry are Hamrock's
printed radii.

## 2. Sources
| Key | What | Equation / table | PDF page |
|---|---|---|---|
| r02 (`hertz`, SlipPY `hertz_full`) | Hertz of two bodies as given; per-body size ratio | Johnson1985 Sec. 4.2 | 105-107 |
| Johnson1985 | Hertz assumptions: `a << R` (relative) and `a << R_1,2` (each body); conforming surfaces | Sec. 4.2 (i)-(iv); Sec. 5.3 | 101-102; 124-125 |
| `slippy_bridge` | `from_bearing` (raceway at `alpha_j`, groove centre fixed in its ring), `solvers.hertz` | `contact/core/machine_elements/bearings.py`, `contact/solvers/hertz.py` | - |
| v003 (`r02_bodies`, `hertz_series_cp`) | r02 inputs of a ball contact; r02 Hertz series of one ball (Harris Eq. 7.6) | Harris2007_vol1 Eqs 7.5-7.7 | 156 |
| Hamrock1983 | sec. 14.2 contacts: `R_x`, `R_y` (Eqs (23)-(26)), curvature sums (19), radius ratios, `k` fits (37), `delta_i`, `delta_o` at `F_max` (42) | sec. 14.2 | 44-46 |
| HamrockBrewe1983 | Table 5, ball on an outer ring | Table 5 | 6 (judged in v102, not repeated) |
Hamrock: PDF page = printed page + 5. Values read from the page images.

## 3. Model
- AxisForge solves the bearing (ISO/TS 16281, `psi = 0`, v003 `study.solve_bearing`); the bridge builds each
  loaded ball's two contacts at the element's angle `alpha_j` with the groove centre fixed in its ring.
- What the comparison can catch: the bridge's raceway rolling radius (recomputed from `Dw, Dpw, ri, re, alpha_0`)
  against AxisForge's stored `r_bx`, the concave outer sign, the mm/MPa -> SI conversion, the equivalent-body
  reduction, and (C0) the element's `Q`, `phi`, `alpha`. What it cannot: a `hertz_full` error (v101/v102),
  AxisForge's `Q_j` itself (v003).
- r02 gets the ball `Dw/2` and the groove `(r_bx_inner, -ri)` / `(-r_bx_outer, -re)` from AxisForge's attributes, at
  `alpha_0` (v003 `r02_bodies`, r02 README section 4), and the element's `Q_j` from the AxisForge result.
- Judged whenever the element runs at `alpha_0` (deep groove at zero clearance, `alpha_j = alpha_0 = 0`): the two
  describe the same contact. Elsewhere the bridge's radii at `alpha_j` and AxisForge's at `alpha_0` differ by
  design (SB-48): RECORD. The size of the contact does not gate the transfer (its error does not depend on it).
- Size (C5, RECORD, OPEN R5-88): r02's per-body ratio `validity.ratio`, governed by the ball (`a / r1`, 3.8 %
  above `a / r_g`), crosses 0.1 near 71 N per ball; the bridge's ratio on the relative radii (SB-07) stays far below.
  Johnson1985 Sec. 4.2 asks both `a << R` (relative) and `a << R_1,2` (each body); in a 0.52 groove the per-body one
  governs. Above 0.1 means "outside the P-02 policy", not "Hertz wrong" (bearing practice uses Hertz there); the
  policy is José's (SB-07, RD-07).
- No BEM solve: one loaded ball-groove contact through the bridge's static solve is s101's question (OPEN R5-81).

## 4. Public API (`__init__.py`)
```python
from studies.v100_dry_contact.v105_ball_contacts import check, series_records, load_cases, literature
check(contacts, bearing, result, *, case="v105", label="") -> list[dict]  # C0-C5 per contact
series_records(contacts, bearing, *, case="v105", label="") -> list[dict] # S1-S2 per element
load_cases() -> list[dict]             # bearing builder (v003), Fr, Fa, bridge contact class, role
```

## 5. Inputs
- 6210 (v003's: 9 balls 12.7 mm, `Dpw` 70 mm, grooves 0.52, steel 210 GPa / 0.3), zero clearance,
  `F_r` = 100, 200, 500, 1000, 2000, 5000, 10000 N (size sweep); 20 um clearance at 5000 N (RECORD).
- 7210 (`alpha_0` 40 deg, 13 balls), `F_r` 3000 N, `F_a` 4000 N (RECORD).
- Hamrock's bearing (v003 literature: `Dpw` 65 mm, `s` 15 um, E 200 GPa) at `F_max` = 4513 N, `alpha` = 0.

## 6. Criteria (fixed before the run; revised after review, see section 8)
| ID | Quantity | Judge | Tolerance | Why |
|---|---|---|---|---|
| C0 | `Q`, `phi`, `alpha` of each bridge contact | AxisForge `element_records` | exact | the bridge copies them |
| C1-C4 | `a`, `b`, `p0`, `delta` of every contact | r02 at AxisForge's radii and `Q_j` | 1e-7 rel, judged at `alpha_j = alpha_0`, else RECORD | both `hertz_full`; the bridge's root_error 1e-6 leaves <= 6.5e-8 (RC-01) |
| C5 | r02 per-body size ratio (`a/r1`), `a/r_g`, the bridge's ratio and flag | - | RECORD | OPEN R5-88 (SB-07) |
| S1 | `K_n = Q / (delta_i + delta_o)^1.5` of the bridge contacts | r02 series (v003 `hertz_series_cp`) | 1e-7 rel at `alpha_0`, else RECORD | as C1-C4 |
| S2 | same | AxisForge `c_p` | RECORD | judged in v003 C7 (0.5 %) |
| L1-L4 | Hamrock `R_x`, `R_y` inner and outer | the bridge's relative radii | half a unit of the printed digit (v102 `half_unit`) | printed digits: the independent judge |
| L5-L12 | curvature sums, radius ratios, exact `k` (bridge Hertz at `alpha = 0`) vs Hamrock's `alpha^(2/pi)` fit, `delta` at `F_max` vs his fits | bridge | RECORD | Hamrock's fits and printed values not self-consistent (R5-09, R5-87) |
Before review C1-C4 were judged only at `a/r_g <= 0.1` and r02 got the bridge's own load; the SlipPY-expert and
physics reviews moved the gate to `alpha_j = alpha_0` only and the load to AxisForge's `Q_j` (with C0).

## 7. How to run
```bat
python -m studies.v100_dry_contact.v105_ball_contacts.study
python -m pytest studies\v100_dry_contact\v105_ball_contacts
jupytext --sync studies\v100_dry_contact\v105_ball_contacts\notebook.py
jupyter nbconvert --to notebook --execute --inplace studies\v100_dry_contact\v105_ball_contacts\notebook.ipynb
```

## 8. Result (run of 2026-10-28, after review)
**Judged: 613 PASS, 0 MISS** (C0 294, C1-C4 280, S1 35, Hamrock L1-L4).
- C0: the bridge's `Q`, `phi`, `alpha` of every contact equal AxisForge's element records exactly.
- C1-C4: bridge vs r02 within 1.3e-10 (inner) and 2.5e-12 (outer) at every load (RC-01 root tolerance), judged at
  every load of the zero-clearance sweep (100 N - 10 kN).
- S1: 4.7e-11. S2 (`c_p`) RECORD: -0.065 % (judged in v003 C7).
- Size (RECORD): r02's per-body ratio (`a/r1`, the ball governs every contact) of the most loaded ball is 0.088 at
  `F_r` 100 N and 0.41 at 10 kN; `a/r_g` is 3.8 % lower. The bridge's ratio on the relative radii is 0.012-0.055 and
  calls every contact valid: about 7x looser (evidence on SB-07; policy OPEN R5-88).
- RECORD, elements away from `alpha_0` (SB-48): 6210 with 20 um at 5000 N (`alpha_j = 0`, radii at 11.4 deg):
  `b` 2.3e-3, `p0` 2.0e-3, series 9.1e-5; 7210 at 3000 / 4000 N: 2.7e-3, 2.4e-3, 8.5e-5.
- Hamrock: the bridge's `R_x` 5.109 / 7.591 mm, `R_y` 165.1 mm match his printed 0.00511 / 0.00759 / 0.165 m (L1-L4
  PASS). RECORD: curvature sums 201.785 / 137.799 (printed 201.76 / 137.81), ratios 32.31 / 21.75 (32.35 / 21.74);
  exact `k` at `alpha = 0` 9.285 / 7.295 against his `alpha^(2/pi)` fit 9.42 (FLAG: Eq. (37) gives 9.15) / 7.09;
  compressions at 4513 N 2.759e-5 / 2.659e-5 m against his 2.731e-5 / 2.707e-5, and his `delta_i` inherits the printed
  `k_i` (with 9.15 it would be 2.785e-5; his `E_bar_i`, `F_bar_i` are also off Eqs (38)-(39): R5-87).

Figures (`results/`): `v105_bearing_loads_load_zone_sweep_around_6210` (bridge open markers on the r02 lines for `a`
and `p0` around the bearing, inner and outer), `v105_contact_relative_errors_conformity` (C1-C4 error flat at 1e-10 to
1e-12 against r02's size ratio, tolerance dashed), `v105_bearing_loads_load_zone_sweep_validity` (r02's `a/r1` crosses
0.1 near 150 N of bearing load; the bridge's ratio stays below 0.06).

## 9. Depends on
r02 (`hertz`); v003 (`r02_bodies`, `hertz_series_cp`, `MM`, `deep_groove_6210`, `angular_contact_7210`,
`literature.hamrock_dgbb`, `study.solve_bearing`: imported, OPEN R5-80); v102 `half_unit` (R5-80); `slippy_bridge`
(`from_bearing`, `solvers.hertz`); `studies/outputs` (`l000_axisforge/plots/bearing_loads.py::load_zone_sweep`,
`l100_dry_contact/plots/contact.py::relative_errors`); v101 `LiteratureCase`. AxisForge `d34ec7f`, SlipPY
`local-fixes` (read only); design-studies `layout/studies-levels` at the v003 commit `39bccfc` + this case (uncommitted
at the run).
