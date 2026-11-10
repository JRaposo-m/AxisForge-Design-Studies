# v107 FZG contact along the path

## 1. Question
Does AxisForge give the geometry, the flank radii and specific sliding at the points A..E of the path, and the
mesh forces, of the FZG test gears C40 (spur) and H501 (helical) as the independent judge r04 does and as
Martins 2015 Table 4 prints; and which reading of FEUP's printed Hertzian pressure p_H holds (Q-09 / SQ-06)?
One validation: **FZG contact along the path vs r04 + r02 and the literature** (v_prerequisites row 6).
Ported from `v100_dry_contact/v105_fzg_geometry_ph` (tag `studies-pre-sweep`), C40 first.

## 2. Sources
| key | item | PDF page |
|---|---|---|
| Fernandes2015b | Table 1: geometry of C40, H501 (r04's published data) | 4 |
| Martins2015 | Table 3 (geometry), Table 4: wheel torques K5, K7, K9, F_r, F_a, p_H for C40 and H501/951 | 4 |
| ISO_21771_2007 | through r04 (Eqs (55), (77)-(87), (114)-(115)) | 33-48 |
Rows: `literature/published/martins2015_t4.csv` (vault extraction 2026-10-16 through the tag).

## 3. Model
AxisForge `GearPair` of `construction.build_pair` (FEUP profile shifts, zero backlash, pinion driving). FLAGS:
the wheel tip from x (Fernandes prints 115.54, a misprint of 118.54); torques printed on the wheel (R-124:
gear 1 gets `T2 z1/z2`); E 206 000 MPa and nu 0.3 assumed (R-118, ISO 6336-2 generic steel); hands of H501
not printed (GL-38). r04 is built from the same basic data (`construction.PAIRS`), never from the AxisForge
object. p_H (spur only): r02 `hertz(line=True)` with AxisForge `rho1_t_mm`, `rho2_t_mm` (x 1e-3) and the load
per unit length `share F_bn / b` (`load_sharing`), as the r02 README section 4 table says.

## 4. Public API
```python
check(pair, wheel_torque_Nm, *, data, published=None, case="v107") -> list[dict]
literature.cases() -> [LiteratureCase(...)]     # C40, H501 x K5, K7, K9 with the printed Fr, Fa, p_H
build_pair(name), PAIRS                         # construction, unsolved
```

## 5. Inputs
As section 3; stages K5 (104.97 N.m), K7 (198.68), K9 (323.27) on the wheel.

## 6. Criteria (fixed before the run)
| id | quantity | tolerance |
|---|---|---|
| G | `a_w`, T1A..T1E | 1e-10 relative (P-12) |
| P | at A..E: `rho1_t`, `rho2_t` (relative), `zeta1`, `zeta2` (absolute) | 1e-10 |
| F | `F_tw`, `F_r`, `F_a`, `F_bn` vs r04 at the wheel torque | 1e-10 |
| L | AxisForge `F_r`, `F_a` vs Martins Table 4 | half a printed unit + the torque's half unit carried |
| H, L | p_H: Hertz at C and A (`load_sharing` share: 1, 1/2), at B and E (full load, set) vs the printed p_H | RECORD (Q-09) |
Not done (OPEN R2-13): the H501 slice consistency and p_H (needs the contact-line length along the path,
Fernandes Eqs A1-A16, R2-10); d_end/b flags per point (the bridge's `GearContact.end_distances`); K6, K8 (not
extracted); the wheel as driver.

## 7. How to run
`python -m studies.v100_dry_contact.v107_fzg_contact_path.study` (about 2 s) writes `results/v107_check.csv`;
`notebook.ipynb` (committed with outputs). Tests: `python -m pytest studies/v100_dry_contact/v107_fzg_contact_path`.

## 8. Result (run of 2026-10-28)
**PASS: G, P, F for C40 and H501 at K5-K9 (largest difference 1.4e-14); Martins F_r and F_a PASS except H501
K7 F_a 979.47 vs 980 N (MISS by 0.005 N past the tolerance; also recorded in r04: a torque printed with more
digits, 198.7 N.m, would give 979.57). 192 judged, 1 MISS.**

**FEUP p_H reading (RECORD, C40; E 206 GPa assumed):**
| stage | printed p_H | C, full load | A, share 1/2 | B, full load (Z_B point) | E, full load |
|---|---|---|---|---|---|
| K5 | 498 MPa | 471.5 (-5.3 %) | 497.3 (-0.13 %) | 504.6 (+1.3 %) | 494.8 (-0.6 %) |
| K7 | 685 | 648.6 (-5.3 %) | 684.2 (-0.11 %) | (+1.3 %) | (-0.6 %) |
| K9 | 874 | 827.4 (-5.3 %) | 872.8 (-0.14 %) | (+1.3 %) | (-0.6 %) |
The printed values scale as sqrt(T2) (685/498 = 1.3755 vs 1.3758), so the three stages are one data point.
Consistent with Hertz at A with half the load; **not unique** (gear-expert): B at full load with E 200 GPa
(-0.2 %), E at full load (-0.6 %) and Hertz at the unshifted pitch point (about +0.9 %) fall in the same +-1 %,
which is also the E uncertainty (200-210 GPa, R-118). Ruled out: Hertz at C with the full load (-5.3 %), the
ISO nominal Z_B Z_eps sigma_C (-6.8 %), C / Z_eps (+2.9 %). A at half load has no mechanism behind it (under
rigid load sharing B at full load is the largest pressure on the path). Q-09 stays open until the FEUP method
(Fernandes Eqs A1-A16, Martins section 2) gives the point and E (OPEN R2-14).

## 9. Depends on
r04 (`a24514b`), r02 (`407273d`), AxisForge `feature/newton-solver` `d34ec7f` (read only), v101
`LiteratureCase`. No SlipPY BEM here (r02 runs SlipPY `hertz_full`).
