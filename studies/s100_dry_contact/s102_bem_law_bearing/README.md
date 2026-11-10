# s102 BEM-law bearing

## 1. Question
If each ball's Q(delta) is the s101 BEM law (inner and outer contacts in series at equal load) instead of c_p delta^1.5,
how much do delta_r, delta_a, Q_max, epsilon and the radial stiffness of a 6210 move? How much of that is the BEM
(geometry) effect and how much the ISO-fit effect?

## 2. Why
It is the only case that closes AxisForge -> SlipPY -> AxisForge on a bearing ("s1xx with BEM per element"). It is
the first bearing whose balls carry a measured BEM law.

## 3. Inputs / sweep
- Bearing: s001's 6210-like deep groove ball bearing, diametral clearance s = 0 and 20 um.
- Loads: Fr = 2500 and 5000 N, Fa/Fr = 0 and 0.5; rings untilted.
- Three ball laws:
  - c_p: AxisForge's `c_p delta^1.5` (ISO Eq. (11) fit);
  - Hertz series: `K_s delta^1.5`, the bridge's Hertz of the two contacts in series;
  - BEM series: the s101 laws of the two contacts in series (`series_law`).
- One BEM law per operating angle: the angle of the case's most loaded ball in AxisForge's solve (0 deg for
  Fa = 0, 12.7-17.3 deg for Fa/Fr 0.5).
  - Each contact's law: 8 approaches, 0.02-1.4 delta_H(Q_max), i.e. loads 0.003-1.66 Q_max (NR-09).
  - s101 grid rule: spatial kernel, b/h >= 16, half-length 2.

## 4. Model (`model.py`)
```python
solve_distribution(bearing, law, Fr_N, Fa_N=0, *, name, label) -> (summary, NewtonResult, elements)
series_law(name, inner: BemLaw, outer: BemLaw) -> BemLaw     # delta = delta_i(Q) + delta_o(Q)
law_mm(BemLaw, *, hertz=False) -> Q[N](delta[mm]);   power_law(K [N/mm^1.5]) -> Q(delta)
```
- Kinematics: AxisForge's own (`ISO16281BallSolver.elements`, Eqs (12)-(15)).
- Balance: ISO Eqs (16)-(17), solved for (delta_r, delta_a) by `numerical_solvers.newton`:
  - x_scale [1e-2, 1e-1] mm (delta_r, delta_a), f_scale [Fr, max(Fa, Fr)];
  - central differences, line search, `lm` fallback;
  - stop only on residual_abs 1e-10 (`step_rel=None`: the default step test could stop above it).
- Start: AxisForge's own heuristic (`ISO16281BallSolver._initial_delta_r/_initial_delta_a`), the clearance
  closed plus a Hertz-size approach, so the ball on the load line is in contact (NR-04); at Fa = 0 the exact
  delta_a = -A sin(alpha_0) (every ball back at alpha = 0).
- Outputs besides delta_r, Q_max, eps: delta_max (approach of the most loaded ball), delta_r - s/2 (Fa = 0; the
  closed clearance taken out) and the tangent stiffness Kt = dFr/d delta_r at fixed Fa (final Jacobian, delta_a
  condensed).
- Hertz series = the BEM series law with c = 0 (same K_s): the BEM effect is the (1 + c delta) factor alone.
- Azimuth: local (phi_j = 0 on the radial-load line). Any global reading is `convention-dependent (R-003)`.
- s101 and s001 are imported through their `__init__` APIs.

## 5. v first
- v003.check on AxisForge's own solve of every case (judged at Fa = 0).
- The limit case of this solver: with the c_p law it must give AxisForge's result.
- s101: the laws, with its B1-B4.

## 6. Criteria (fixed before the run)
| id | criterion | tolerance |
|---|---|---|
| N1 | limit case: the c_p law in this solver reproduces AxisForge's ISO/TS 16281 solve, Q_j and delta_r, every case | 1e-6 rel (AxisForge's SOLVER_TOLERANCE) |
| N2 | v003.check of AxisForge's solve: every judged record PASS | v003's |
| N3 | every Newton solve converged (residual_abs 1e-10) | - |
| N4 | BEM effect (BEM series / Hertz series - 1) and ISO-fit effect (Hertz series / c_p - 1) of delta_r, delta_r - s/2, delta_max, Q_max, eps, Kt reported apart | RECORD |
| N5 | limit case s = 0, Fa = 0 (Stribeck invariance): Q_j / Fr of the c_p and Hertz-series laws equal (K drops out) | 1e-8 (100 x the Newton residual) |

N5 was added at review (physics review item 5), after the first run and before the session-2 run; its tolerance
follows from the residual, not from the result (OPEN R5-104).
| R | Newton iterations, evaluations, step lengths, fallback (`s102_newton.csv`); the BEM laws (`s102_laws.csv`) | RECORD |

## 7. How to run
- Run: `python -m studies.s100_dry_contact.s102_bem_law_bearing.study`, about 2 min (80 BEM solves for 5 laws).
- Output in `results/` (git-ignored):
  - tables: `s102_{cases, elements, effects, newton, laws, v003, criteria}.csv`;
  - figures `s102_bearing_loads_element_loads_{s0_Fa0, s20_Fa05}.png` and
    `s102_bearing_loads_load_zone_sweep_effects.png` (`outputs/l000_axisforge/plots/bearing_loads.py`), and
    `s102_convergence_newton_history.png` (new `outputs/common/plots/convergence.py`).
- Tests: `python -m pytest studies/s100_dry_contact/s102_bem_law_bearing`.

## 8. Risks
- NR-09: loads outside a law table. Each table reaches 1.66 Q_max; beyond it the smooth s101 law extrapolates its
  linear ratio.
- NR-04: a start out of contact. The start puts the load-line ball in contact.
- A ball entering or leaving the zone: Q = K delta^1.5 (1 + c delta) is C1 at delta = 0 (dQ/d delta -> 0), so a
  central difference that straddles 0 errs by O(h^1.5), about 6e-8 mm here: harmless. (The jump hazard of the
  old table law is gone with the smooth s101 law.)
- One law per case, at the angle of the most loaded ball. The other balls of an Fa > 0 case run at other angles (the
  groove radii change little with alpha: R5-100).
- The half-space is shared by both laws (s101 section 2).

## 9. Result (session-2 run of 2026-10-28, smooth s101 law of commit 09b58f6)
**N1, N2, N3, N5 PASS.**
- N1 (limit case): this solver with the c_p law reproduces AxisForge to 2.8e-8 in Q_j and 1.8e-8 in delta_r.
- N2: 32/32 v003 records PASS.
- N3: 24/24 Newton solves converged on residual_abs, 4-6 iterations, final ||F / f_scale|| <= 1.7e-14, no fallback.
- N5: Q_j / Fr of c_p and Hertz series equal to 2.2e-15.

Answer (N4, RECORD; BEM effect / ISO-fit effect, over the 8 cases):
| quantity | BEM effect | ISO-fit effect |
|---|---|---|
| delta_r | +0.43 to +1.48 % | -0.07 to -0.02 % |
| delta_r - s/2 (Fa = 0) | +0.61 to +1.00 % | -0.04 % |
| delta_max | +0.61 to +1.00 % | -0.05 to -0.04 % |
| Q_max | -0.32 to -0.11 % | below 0.01 % |
| eps | -0.46 to +0.21 % (0 where s = 0 or the whole ring is loaded) | about 0.01 % |
| tangent Kt | -2.03 to -1.06 % | +0.04 to +0.07 % |
- The BEM ball is softer: the most loaded ball sits 0.6-1.0 % deeper and the tangent radial stiffness drops
  1.1-2.0 % (largest at Fr 5000 N). At s = 20 um the closed clearance (s/2 = 10 um) dilutes the delta_r effect
  (+0.43 % against +0.61 % on delta_r - s/2 at Fr 2500 N).
- Why Q_max hardly moves: Stribeck invariance. At s = 0, Fa = 0 with Q = K delta^1.5 the sharing Q_j / Fr does not
  depend on K (N5: 2e-15), and at fixed Fr the balance pins the loads, so a pure K change shows up only as
  displacement. Only the load-growing part (1 + c delta), c < 0, moves load from the heavy balls to the light ones:
  Q_max goes down in every case (-0.11 to -0.32 %), and eps goes up at s = 20 um, Fa = 0 (+0.18, +0.21 %), as
  expected. At s = 20 um, Fr 5000 N, Fa/Fr 0.5 eps goes down (-0.46 %): there the axial shift changes too; recorded,
  not explained.
- The ISO Eq. (11) fit of c_p is 14-23 times smaller an effect on delta_r than the BEM geometry.
- The proposals' "1-2 % in Q" holds at a fixed approach (s101); at a fixed Fr it appears as displacement and
  stiffness (delta_max +0.6-1.0 %, Kt -1.1-2.0 %), not as Q_max (below 0.35 %).
- The effects are those of the half-space geometry of s101 (not a finite-body judge; s101 section 2).
Tables and figures are copied to the vault, `80_Inbox/2026-10-28/r5_studies/s102/`.

## 10. Depends on
- s001 (`deep_groove`, `load_node`, `solve_bearing`, `records`, `loaded_zone`); s101 at `09b58f6` (`BemLaw`,
  `BemLaw.delta`, `fit_law`, `bem_law`); v003 (`check`).
- AxisForge `d34ec7f` (`ISO16281BallSolver.elements`), `numerical_solvers.newton`, `slippy_bridge`
  (`DeepGrooveBallContact`, `hertz`), `studies/outputs`. SlipPY `df5916c`.
- OPEN: R5-100 (one BEM law per case at the angle of Q_max); R5-101 (Kr is the secant Fr / delta_r, clearance
  included, as AxisForge; the effects use the tangent Kt); R5-103 (Kt uses numerical_solvers' private
  `_fd_jacobian`: NewtonResult keeps no Jacobian); R5-104 (N5 added at review). R5-102 (laws before the s101
  review fixes) is closed by this run.
