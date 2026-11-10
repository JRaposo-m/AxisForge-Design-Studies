# s103 profile shift split

## 1. Question
For the FZG pairs C40 (spur) and H501 (helical) with FEUP's sum x1 + x2 kept, how does the split x1 move the path
points A..E, the transverse contact ratio eps_alpha, the specific sliding at the two roots (zeta_f1 at A, zeta_f2 at E),
the flank radii and the Hertz pressure p0 along the path (L0 load sharing), and the tooth thickness at the tips? Where
do FEUP's split, AxisForge's equal-specific-sliding split and x1 = x2 sit in that range, and which part of it can a
rack cutter make?

## 2. Why
The profile shift is the one geometric design variable AxisForge exposes (ISO_21771_2007 5.6.3, J-19). Proposals.md
"s103" (later, S-M); a design-variable sweep for the optimisation stage, whose key criterion is the equal-sliding
identity through r04.

## 3. Inputs / sweep
v107's pairs (`PAIRS`, imported, `# OPEN R5-130`): C40 z 16/24, m_n 4.5, b 40, x 0.1817/0.1715 (sum 0.3532); H501
z 20/30, m_n 3.5, beta 15 deg, b 23, x 0.1381/0.1319 (sum 0.2700); E 206 GPa, nu 0.3 assumed (R-118); pinion drives
(Q-09); hands of H501 GL-38. Stage K9: 323.27 N.m on the wheel (Martins2015 Table 4, R-124). Sweep: 25 values of x1
strictly inside the ideal range of `model.bounds` + FEUP's x1 + x1 = x2 + AxisForge's equal-specific-sliding x1 +
H501's second published split x1 0.1809 / x2 0.0891 (GEARpie / Gear-Calculation, AxisForge `baseline_cases.py:89-90`,
R-307).

Tips, two variants (`# OPEN R5-138`; R-306 is José's question "did FEUP shorten the tips?"): **k = 0** (v107: tips
from x alone) and **k (R-306)**: AxisForge's tip alteration `addendum_reduction=True` (Q23, k = (a_w - a)/m_n -
(x1 + x2), the same on both gears and constant at a fixed sum: C40 k -0.0198, H501 k -0.0090). Only k reproduces
FEUP's printed tips and eps_alpha (Fernandes2015b Table 1): C40 82.457 / 118.365 mm, eps 1.438 (printed 82.46 /
115.54, 1.44; FZG type C 118.36); H501 80.373 / 116.564, eps 1.463 (printed 80.37 / 116.57, 1.45). The printed C40
wheel tip 115.54 reads as a misprint of 118.36 (k) rather than of 118.54 (k = 0, v107's FLAG): `# OPEN R5-171`, not
decided here.

## 4. Model (`model.py`, public API in `__init__.py`)
- `construction.split_pair(name, x1, *, equal_sliding, addendum_reduction)`: v107 `build_pair` with x replaced in the
  gear data (`dataclasses.replace`; v107 has no x argument and is not changed: `# OPEN R5-131`). With the sum fixed,
  a_w, alpha_wt, T1T2 and C are fixed (ISO_21771_2007 5.2 Eqs (55), (58), PDF p. 32-35); only the tips move (Eq. (33),
  PDF p. 28). F_bn is fixed.
- `construction.as_cut(pair)`: the same pair in AxisForge `ManufacturedCylindricalFamily` (`from_ideal`, zero
  allowances), cut by AxisForge's default tool, the counterpart of the ISO 53 basic rack
  (`GeneratingRackProfile.from_basic_rack`: h_aP0 1.25 m_n, rho_aP0 0.38 m_n; FEUP prints no tool: `# OPEN R5-132`).
  AxisForge gives x_Emin (7.7 Eq. (135), PDF p. 63; h_FaP0 with sin(alpha_t) as printed, MG-01), d_Ff (7.6 Eq. (128),
  PDF p. 61-62) and d_Nf; its `validate_pair` reports undercut, a pointed tip and d_Nf < d_Ff.
- `model.split_point(pair, T)`: AxisForge `GearPair` path T1A..T1E, eps_alpha, zeta_f1, zeta_f2 (5.6.3 Eqs (116),
  (117), PDF p. 45-48); bridge `CylindricalGearContact.from_pair` + `hertz` at A..E with the L0 load F_bn / l_sum
  (`load_sharing`, l_sum = summed contact-line length at that instant, 5.4.8, PDF p. 41-44). Spur: share 1/2 in double
  contact, also AT B and D, so B, D are added with the full load (their single-contact side, where the largest L0 p0
  sits). Helical: the share follows the contact lines and moves with the split (0.66-0.67 at B / D, 0.40-0.99 at C in
  `s103_points.csv`), set by the fractional parts of eps_alpha and eps_beta, not by the curvature. C is left out when
  the path lies on one side of it (`C_on_path` False; R5-135). Tip thickness s_an / m_n from AxisForge
  `gear_values_at(gear, d_a)` (4.7.1 Eq. (38), PDF p. 29; 7.9, PDF p. 63). `p0_max` is taken over the points where
  `hertz` is valid (A, E sit on a tip edge: s104's question). As-cut flags per split: `undercut_pinion`,
  `undercut_wheel`, `fillet_pinion`, `fillet_wheel` (d_Nf < d_Ff), `made_valid`.
- `model.bounds(name, *, addendum_reduction)`: **ideal range** (the sweep): T1A >= 0, T2E >= 0 (AxisForge
  `validate_pair`), no pointed tip (s_a > 0), tip outside the reference circle (h_a > 0, AxisForge `validate_gear`).
  **Range as cut**, inside it: also x >= x_Emin for both gears and d_Nf >= d_Ff for both gears (T1A >= T1Ff1,
  T2E >= T2Ff2; 5.4.1 Eq. (66), (67), PDF p. 37). Splits between the two are kept and tagged (`made_valid` False),
  hatched in the figures: ideal flanks that a rack cutter does not make. On the undercut side (x < x_Emin) Eq. (128)
  does not hold (the form circle is the intersection of involute and trochoid, 7.6 last sentence, PDF p. 62; not
  computed): **the active root of an undercut gear is not checked**, so the zeta_f1 and p0(A) branches there are on
  flank that may not exist. The ideal `CylindricalFamily` accepts a pointed tip (`# OPEN R5-133`) and has no
  d_Nf >= d_Ff check (`# OPEN R5-137`). 7.9 fixes no minimum tip thickness (reported only, `# OPEN R5-134`).
- Validation: v107 `check(pair, T, data=pair_data(name, x1))` at every split (an s calls its v, never r04 / r02).
  r04 inside v107 builds the tips from x with k = 0, so for the k variant the tip-dependent G records (T1A, T1B, T1D,
  T1E) are not judged; a_w, T1C, every P record (rho, zeta at the AxisForge points) and F are.
- Helical: geometry and sliding judged as for spur; p0 is the mid-face slice of the bridge (R-027), RECORD (the
  contact-line length along the path is R2-10 / s106). **J-16 does not enter**: no shaft, F_a r_w is not used.

## 5. v first
v107 (FZG contact along the path vs r04 + r02; PASS 1.4e-14 on G, P, F at FEUP's split). s103 runs v107's `check`
at each of its own splits.

## 6. Criteria (fixed before the run)
| id | quantity | tolerance | source |
|---|---|---|---|
| S0/G, S0/P, S0/F | every v107 judged record (a_w, T1A..T1E; rho, zeta at A..E; F_tw, F_r, F_a, F_bn vs r04) at every split (k variant: G on a_w and T1C only, r04 has k = 0) | 1e-10 (v107 `TOL_CLOSED_FORM`) | P-12 |
| S1 | at AxisForge's equal-sliding split: r04's zeta1 at A minus zeta2 at E (through v107 P records) | 1e-9 abs | brentq xtol 1e-12 on x1 |
| S2/A, S2/C, S2/B full (spur) | the model's bridge p0 vs v107's r02 p0 at the same point and load: a **consistency** check (both call SlipPY `hertz_full` with the same inputs), not a physics judge | 1e-9 rel | same SlipPY `hertz_full`, same inputs |
| sweep values, bounds, p0_max | RECORD | - | - |

The physics reviewer hand-checked p0 = sqrt(w E* / (pi R)) (line contact) at FEUP and at equal sliding for C40 to
5 digits (review of 2026-10-28; not a criterion of this run).

## 7. How to run
`python -m studies.s100_dry_contact.s103_profile_shift_split.study` (about 10 s) writes `results/` (git-ignored):
`s103_sweep.csv`, `s103_points.csv`, `s103_v107.csv`, `s103_criteria.csv` (column `tips`: "k = 0" / "k (R-306)"),
`s103_bearing_loads_load_zone_sweep_{C40,H501}.png` (k = 0) and `..._{C40,H501}_k.png` (k): T1Y, zeta, eps_alpha,
p0, s_an vs x1; FEUP, x1 = x2, equal sliding, GEARpie, undercut and fillet limits marked; not-as-cut intervals
hatched; AxisForge filled, Hertz-invalid p0 open, r04 / r02 through v107 as lines; and
`s103_contact_relative_errors_vs_v107.png`. Tests: `python -m pytest studies/s100_dry_contact/s103_profile_shift_split`.

## 8. Risks
L0 is rigid load sharing (no tooth stiffness): the largest p0 sits at B or D by construction. The range as cut uses
zero allowances and AxisForge's default rack cutter (R5-132); the tips are ideal (no chamfer, no tip relief) with
k = 0 or AxisForge's k (R-306, R5-138). Helical p0 is one slice.

## 9. Result (run of 2026-10-28, session 2)
**514 PASS, 0 MISS** (both tips variants). S0 worst 1.1e-14 (C40) / 3.2e-12 (H501); S1 2.5e-12 / 1.7e-14 (k = 0),
1.6e-12 / 4.0e-15 (k); S2 4.4e-16.

| | C40 k = 0 | C40 k (R-306) | H501 k = 0 | H501 k (R-306) |
|---|---|---|---|---|
| ideal x1 range (binding) | -0.336 (T1A = 0) ... 1.034 (pinion tip pointed) | -0.356 ... 1.075 (same) | -0.752 (T1A = 0) ... 1.270 (wheel h_a = 0) | -0.761 ... 1.261 (same) |
| as cut (binding) | 0.064 (pinion undercut) ... 0.757 (wheel undercut) | same | -0.283 (pinion undercut) ... 1.010 (wheel tip on pinion fillet) | -0.283 ... 1.113 (same) |
| other limits inside the ideal range | wheel tip on pinion fillet 0.961 | 1.070 | wheel undercut 1.197 | 1.197 |
| FEUP x1 0.1817 / 0.1381 | eps 1.462, zeta_f1 -3.75, zeta_f2 -2.18, p0max 885.5 MPa at B | 1.438, -3.54, -2.11, 889.3 at B | 1.474, -2.21, -1.53, 1065 at C | 1.463, -2.17, -1.51, 1077 at C |
| x1 = x2 0.1766 / 0.1350 | 1.463, -3.81, -2.16, 886.5 | 1.438, -3.60, -2.10, 890.4 | 1.475, -2.23, -1.52, 1065 | 1.463, -2.18, -1.51, 1076 |
| equal sliding | x1 0.3094: 1.452, -2.63 both, 864.0 | x1 0.3035: 1.428, -2.53 both, 867.9 | x1 0.2456: 1.466, -1.76 both, 1015 | x1 0.2434: 1.455, -1.74 both, 1021 |
| GEARpie x1 0.1809 (R-307) | - | - | 1.472, -2.02, -1.62, 1049 | 1.460, -1.98, -1.60, 1054 |

Every number of the table carries `(R-306)`: it depends on the tips variant. All marked splits lie inside the range
as cut (`made_valid` True).
- The session-1 upper ends 1.034 (C40) and 1.331 (H501) were ideal ones; H501's was wrong as well (the wheel tip
  reaches its reference circle at x1 1.270 before the pinion tip gets pointed; no swept point lay beyond it). As cut
  the ranges shrink to 0.064-0.757 (C40) and -0.283-1.010 (H501). The physics reviewer's hand value of H501's fillet
  limit, 0.971, differs from AxisForge's 1.010 (the gear-expert reads > 1.0: d_Nf1 72.575 < d_Ff1 72.582); C40 0.961
  agrees with AxisForge (0.9606).
- zeta_f1 rises and zeta_f2 falls monotonically with x1; eps_alpha has a flat maximum near x1 = 0 and falls toward the
  pointed-tip end. k lowers eps_alpha by 0.025 (C40) / 0.012 (H501).
- C40 p0_max (L0, single contact): inside the range as cut it falls from 906 MPa (x1 0.09) to a minimum of about
  834 MPa near x1 0.56-0.61, where the governing point moves from B to D, then rises to 848 MPa at 0.72. The 1037 MPa
  of session 1 (x1 -0.28) lies on an undercut pinion (not as cut). FEUP's split sits near x1 = x2, 6 % above the
  minimum; the equal-sliding split lies between. With k the same picture, about +0.5 % in p0.
- Strongly unequal splits push C out of the path (H501, x1 > about 1.1, not as cut) and, for C40, into a
  double-contact zone (p0 at C drops to half load).
- FEUP's split is close to x1 = x2 for both pairs (0.1817 vs 0.1766; 0.1381 vs 0.1350), not to equal sliding. For H501
  a second published split of the same sum, 0.1809 (R-307), lies between FEUP's and equal sliding: the FEUP sources
  split 0.27 differently, so "FEUP splits near x1 = x2" is a reading of one source, not a design rule.
- H501 p0 1065 / 1015 MPa (mid-face slice, L0) is not comparable with the printed p_H 1382 MPa (Martins2015 Table 4;
  R-027, R-370): not a 23 % miss.

## 10. Depends on
v107 (`build_pair`, `check`, `PAIRS`, `FAMILY`, `WHEEL_TORQUE_NM`, `E_MPA`, `NU`, `DRIVER`), AxisForge
`feature/newton-solver` `d34ec7f` (read only: `GearPair.assemble(x_split=..., addendum_reduction=...)`,
`ManufacturedCylindricalFamily.from_ideal` / `validate_pair` (x_Emin, d_Ff, d_Nf), `gear_values_at`, `path_points`,
`mesh_forces`), slippy_bridge (`CylindricalGearContact.from_pair`, `hertz`), `studies/outputs`
(`bearing_loads.load_zone_sweep` with `x_spans`, `contact.relative_errors`). OPEN R5-130..R5-139, R5-170, R5-171.
