# s101 BEM ball-raceway law

## 1. Question
What is the BEM normal law Q(delta) of one ball-raceway contact on the true groove (inner and outer), and how far
are it, a, b and p0 from Hertz at equal approach, as the contact grows against the groove (a/r_g)?

## 2. Why
It is the measured piece of "a bearing with a BEM contact law" and the table s102 imports. It is framed as the
GEOMETRY error of the Hertz paraboloid, not "BEM is more accurate": both are elastic half-spaces, and the ring's
bulk compliance is in neither.

## 3. Inputs / sweep
- Contacts: the most loaded ball of two s001 bearings, built by `slippy_bridge` from the AxisForge solve
  (`from_bearing(bearing, result, elements="max")`). Ball 12.7 mm, grooves f = 0.52, steel.
  - 6210-like deep groove at Fr 5000 N, s = 0: alpha 0, Q_ref 2436 N, inner and outer.
  - 7210-like angular contact (alpha_0 40 deg) at Fr = Fa = 5000 N: alpha_j 38.0 deg, Q_ref 2071 N, inner and
    outer.
- Approach: delta = (0.05 ... 1.5) delta_H(Q_ref), 10 points, geometric. The loads span 0.011 to 1.8 Q_ref, and
  a/r_g runs 0.064 to 0.38.
- Out of this run: the other osculations (f 0.515, 0.54) and angles of proposals.md, and the dense s102 table
  (OPEN R5-90). s102 builds its own table through `bem_law` at its alpha_j.

## 4. Model (`model.py`, public API in `__init__.py`)
```python
solve_at_approach(contact, delta_m, results_dir, *, b_over_h=16, span=2, h_m=None) -> (StaticResult, HertzResult, s)
bem_law(name, contact, deltas_m, results_dir, *, b_over_h=16, span=2) -> (BemLaw, solves)
BemLaw(name, delta_m, Q_N, K_hertz_N_per_m1_5, c_per_m, fit_rms)    # Q = K_H delta^1.5 (1 + c delta), smooth
BemLaw.Q(delta_m) -> Q_N;   BemLaw.delta(Q_N) -> delta_m                # the law and its inverse
fit_law(delta_m, Q_N, K_hertz) -> (c, rms);   fit_power_law(delta_m, Q_N) -> (K, n)   # law; descriptor only
contacts() -> {name: (bearing, load_distribution, contact)}                 # construction.py
```
- Solve: SlipPY BCCG under interference control (the rigid approach given, the load found), through `slippy_bridge`
  `solve`. Spatial kernel; grid by the v104 rule (b/h asked 16, actual in [16, 32), half-length 2 semi-axes on both
  axes), sized on the Hertz ellipse at Q_H(delta).
- Hertz at equal approach: Q_H = Q_ref (delta / delta_H(Q_ref))^1.5 (Johnson 1985 Sec. 4.2, Eq. (4.31), PDF p. 106: the ellipse's eccentricity does not depend on the load, so
  Q ~ delta^1.5 at any reference load), from the bridge's
  `hertz`, which v105 judges against r02. Units SI throughout; AxisForge's mm stay in the bearing.

## 5. v first
- v105.check (the bridge's contacts against the AxisForge element and r02): runs on every contact built
  (`s101_v105.csv`).
- v101 / v104: the BEM grid rule and the 0.5 % approach accuracy on the spatial kernel at b/h >= 16.

## 6. Criteria (fixed before the run)
| id | criterion | tolerance |
|---|---|---|
| B1 | Q_BEM / Q_H at equal approach, where a/r_g <= 0.1 (Hertz's range, MG-08), b/h >= 16, converged, edge clear | 0.75 % (v101's 0.5 % on delta, times 1.5, v101 C2) |
| B2 | per contact, the departure Q_BEM/Q_H - 1 is monotonic and sign-stable in a/r_g; a/r_g at 1 % and 3 % reported | - |
| B3 | every solve converged with the pressure clear of the domain edge (`load_ok` is no check under interference, SB-57) | - |
| B4 | grid convergence of Q at delta_H(Q_ref), 6210 inner and outer: `numerical_solvers.richardson` over h, h/2, h/4 (b/h asked 8, 16, 32; actual 14.7 / 29.4 / 58.8 inner, 8.9 / 17.9 / 35.7 outer), flat_atol 0, max_p 2; status monotonic or flat and GCI_fm <= 0.75 % | 0.75 % |
| V | v105.check PASS on its judged records for the contacts built | v105's |
| R | the fitted (K, n); a, b, p0 against Hertz; the series of the two 6210 laws against AxisForge's c_p; everything at a/r_g > 0.1 | RECORD |

## 7. How to run
- Run: `python -m studies.s100_dry_contact.s101_bem_ball_raceway_law.study`, about 4 min (46 BEM solves).
- Output in `results/` (git-ignored):
  - tables: `s101_points.csv`, `s101_laws.csv`, `s101_v105.csv`, `s101_richardson.csv`, `s101_criteria.csv`;
  - figures `s101_contact_{load_approach, relative_errors_vs_conformity, relative_errors_richardson,
    pressure_cut_6210_inner_Qref}.png` (`studies/outputs/l100_dry_contact/plots/contact.py`; `load_approach` is
    new).
- Tests: `python -m pytest studies/s100_dry_contact/s101_bem_ball_raceway_law`.

## 8. Risks
- The half-space is shared with Hertz, so neither model has the ball's own size (a/R_ball, MG-08, J-11).
- No published conforming judge (Kang2006 to re-triage).
- The groove shoulder is not modelled (the bridge's raceway).
- The grid is sized on the Hertz ellipse, which is longer than the true contact (margin, not a cut).

## 9. Result (run of 2026-10-28, after review)
**B1-B4 and V PASS.**
- B1: 11/11 judged points, largest |err| 0.18 %. B4 shows the grid part is about 1e-4 %, so B1 judges geometry.
- B2: the departure is monotonic and negative in all four contacts. It goes from -0.08 % at a/r_g 0.07 to -2.5 %
  (6210 inner, a/r_g 0.38). The BEM carries LESS load than Hertz at equal approach.
  - Why: the exact gap exceeds the paraboloid gap by y^4 (1/R^3 - 1/r_g^3) / 8 > 0. The sphere falls away faster
    than the groove rises, so at equal approach the contact is shorter along the groove.
  - |Q_BEM/Q_H - 1| reaches 1 % at a/r_g = 0.235 to 0.237 in all four contacts, and never reaches 3 % in the sweep.
  - Slope of log |Q error| against log a/r_g: 1.985 / 1.989 / 1.993 / 1.992 (2 expected for the quartic gap;
    dotted guide on `relative_errors_vs_conformity`).
- B3: 40/40.
- B4: monotonic. 6210 inner: p 1.33, GCI_fm 1.6e-5. 6210 outer: p 1.64, GCI_fm 1.4e-5. Q_extrapolated / Q_H - 1 =
  -1.70 % / -1.64 % at delta_H(Q_ref): the departure is in the gap, not the grid.
- V: 21/21 judged v105 records PASS. That includes S1, the 6210 ball's Hertz series of the two bridge contacts
  against r02 (`series_records`).
- The law (what s102 imports): Q = K_H delta^1.5 (1 + c delta).
  - c = -954 / -949 / -958 / -954 1/m (c delta_H(Q_ref) = -1.7 % / -1.6 % / -1.5 % / -1.5 %).
  - Rms misfit over the table: 1.2-1.5e-4.
  - Smooth (no kink at table nodes), equal to Hertz as delta -> 0.
  - The former log-log table with a power-law fit outside it (misfit up to 0.6 %, jumps at the table ends) is
    dropped.
- RECORD:
  - Power-law descriptor n = 1.4935 / 1.4938 / 1.4942 / 1.4944: not the law.
  - At Q_ref the BEM law carries 0.983-0.985 of Hertz.
  - The semi-axis a (along the groove) is the most affected: -6.2 % at a/r_g 0.38. p0: +0.44 %. b: within 0.7 % (sub-node noise).
  - The 6210 ball as a series of its two BEM laws at Q_ref: K = 0.9839 of AxisForge's c_p (the Hertz series is
    1.0006 of c_p, v003 C7; S2 in `s101_v105.csv`).
Answer, WITHIN THE ELASTIC HALF-SPACE (both models): on the true 0.52 groove of a 12.7 mm ball, the BEM law departs
from Hertz by about -0.18 % at a/r_g 0.1, -1 % at 0.24 and -2.5 % at 0.38 (1.8 Q_ref). At the 6210's Q_max it is
1.7 % softer. This is the gap error of the paraboloid. At a/R_ball about 0.4 the half-space error itself is of the
same order (Johnson 1985 Sec. 4.1, PDF p. 101). The softening is therefore not a judged physical correction: a
finite-body judge (FE; Kang2006 to re-triage) is needed (R5-94).
Tables and figures are copied to the vault, `80_Inbox/2026-10-28/r5_studies/s101/`.

## 10. Depends on
- s001 (`deep_groove`, `angular_contact`, `load_node`, `solve_bearing`); v003 (`r02_bodies`, for r_g); v105
  (`check`); v101 (`pressure_cuts`).
- `slippy_bridge.contact` (`Grid`, `hertz`, `solve`, `DeepGrooveBallContact`, `AngularContactBallContact`);
  `numerical_solvers.richardson`; `studies/outputs/l100_dry_contact/plots/contact.py`.
- AxisForge `d34ec7f`, SlipPY `local-fixes` `df5916c` (read only); design-studies `60ea589` plus the working tree.
- OPEN: R5-90 (osculation and angle sweep, and the dense s102 table, not in this run); R5-91 (Q_ref of the 7210 is its
  alpha_j 38 deg contact, so v105 records it, SB-48); R5-92 (`max_p = 2` given by the brief as the formal order; the
  observed p is 1.3-1.6, max_p not active); R5-93 (the bridge
  docstring says the y^4 term "stiffens" the contact; it softens Q(delta); bridge frozen, issue row); R5-94 (no
  finite-body judge of the softening).
