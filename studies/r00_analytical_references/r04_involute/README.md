# r04 involute

## 1. Reference
Involute geometry of an external cylindrical gear pair (spur or helical, profile-shifted), its path of contact
(points A..E, contact ratios), the flank radii of curvature and specific sliding along the path, and the mesh
forces at the working pitch circle. The independent judge of v107 (FZG contact along the path): own formulas,
NumPy / SciPy only (amendment rule 13); it never calls AxisForge `pair_values_at`, `path_points` or `load_sharing`.

## 2. Sources
ISO 21771:2007 read from the page images (text layer garbled; librarian 2026-10-28; PDF page = document page + 4):
| relation | ISO 21771 | PDF |
|---|---|---|
| `inv(alpha) = tan(alpha) - alpha` | 4.3.9 Eq. (18) | 25 |
| `tan(alpha_n) = tan(alpha_t) cos(beta)` (printed this way round) | 4.3.6 Eq. (14) | 24 |
| `d = z m_n / cos(beta)`; `d_b = d cos(alpha_t)` | Eqs (1), (19) | 18, 25 |
| `tan(beta_b) = tan(beta) cos(alpha_t)`, `sin(beta_b) = sin(beta) cos(alpha_n)` | Eqs (5), (6) | 22 |
| `inv(alpha_wt) = inv(alpha_t) + 2 tan(alpha_n)(x1 + x2)/(z1 + z2)` (zero backlash only) | 5.2.4 Eq. (55) | 33 |
| `d_w = d_b / cos(alpha_wt)`; `a_w = (d_w1 + d_w2)/2` (the form `a cos(alpha_t)/cos(alpha_wt)` is not printed) | Eqs (56)-(58) | 33-34 |
| `d_a = d + 2 (x m_n + h_aP + k m_n)` (h_aP a length; k tip alteration; r04 takes `h_aP = m_n`, `k = 0`) | 4.5.3 Eq. (33) | 28 |
| `p_bt = pi d_b / z` | 4.4.5 Eq. (28) | 27 |
| `g_alpha`, points A..E from T1 (`rho_C1 = d_b1 tan(alpha_wt)/2`, `rho_B1 = rho_E1 - p_et`, `T1T2 = a_w sin(alpha_wt)`) | 5.4.5.2-5.4.5.3 Eqs (77)-(87) | 39-40 |
| `eps_alpha = g_alpha / p_et`, `p_et = p_bt`; `eps_beta = b sin(beta)/(pi m_n)` | Eqs (90), (93) | 41, 42 |
| `rho_y = r_b tan(alpha_yt)` (transverse section) | 4.3.8 Eq. (17) | 25 |
| `zeta1 = 1 - rho2/(u rho1)`, `zeta2 = 1 - u rho1/rho2`, `u = z2/z1` | 5.6.3 Eqs (114)-(115); (52) | 48; 32 |
| `tan(beta_y) = tan(beta) d_y/d` (gives `beta_w`) | 4.3.3 Eq. (8) | 23 |
Not in ISO 21771: the normal-plane radius `rho_n = rho_t / cos(beta_b)` (the convention of AxisForge and the
bridge, D-0035; OPEN R2-11: source to cite), and the mesh forces (`F_t = 2T/d_w`, `F_r = F_t tan(alpha_wt)`,
`F_a = F_t tan(beta_w)`, `F_bn = F_t/(cos(alpha_wt) cos(beta_b))`: the usual ISO 6336-1 / textbook forms, checked
here against Martins 2015 Table 4). Misprints of ISO 21771 noticed (none changes a result): Eq. (90) text calls
p_et the "transverse normal base pitch"; Eqs (91)-(92) use b_w, Eq. (93) b; Eq. (107) lower-case rho_c.
Tables: Fernandes2015b Table 1 (PDF p. 4), Martins2015 Tables 3-4 (PDF p. 4), in
`published/fernandes2015b_t1_martins2015_t3_t4.csv` (vault extraction 2026-10-16 through the tag).

## 3. Scope and validity
External pairs; ideal involutes, rigid teeth, no flank or tip modification, zero backlash. The tips are given
or built from `x` with `k = 0`. Left out: internal pairs, H_V (Ohlendorf, until v401), the L0 share (OPEN R2-10).

## 4. Public API
```python
from studies.r00_analytical_references.r04_involute import inv, inv_inverse, pair_geometry, path_point, mesh_forces
pair_geometry(z1, z2, m_n, alpha_n_deg, beta_deg, x1, x2, b, *, d_a1=None, d_a2=None) -> PairGeometry
    # alpha_t, beta_b, d, d_b, d_a, alpha_wt, a, a_w, d_w, beta_w, p_bt, T1T2, T1A..T1E, g_alpha, eps_alpha, eps_beta
path_point(geometry, T1Y) -> PathPoint     # on_path, single_pair, rho1_t, rho2_t, rho1_n, rho2_n, rho_red_n, zeta1, zeta2
mesh_forces(geometry, torque_Nm, *, on=2) -> MeshForces   # F_t (at d_w), F_r, F_a, F_bn [N]; couple_k = F_a r_wk [N.m],
                                                          # the tilting moment of F_a (not about the gear axis)
tip_diameter(z, m_n, beta_deg, x, h_a_star=1.0)
```
mm, N, N.m; angles in degrees in, radians out (`_rad`); force magnitudes (directions are the caller's, D-0010).

## 5. Data
`published/fernandes2015b_t1_martins2015_t3_t4.csv`: Fernandes Table 1 (7 pairs: z, m, a, alpha, beta, b, x, d_a,
eps_alpha, eps_beta), Martins Table 3 (C40, H501) and Table 4 (wheel torques K1-K9, F_r, F_a).

## 6. Criteria (`tests/test_r04.py`), to half a unit of the last printed digit
| test | against | result |
|---|---|---|
| `a_w` from `x1 + x2` (Eq. 55) | printed `a`, 7 pairs | PASS (e.g. Transfer 1 105.011 vs 105.0) |
| `eps_beta` (Eq. 93) | 6 pairs | PASS |
| `eps_alpha` from the printed tips | H951, Transfer 1, Transfer 2, m=10 | PASS |
| C40 tips from x (Eq. 33, k = 0) | 82.64 (Fernandes), 118.54 (Martins) | PASS |
| F_r, F_a from the wheel torque at `d_w2` (+ the torque's printed half unit carried) | Martins Table 4, C40 and H501, K1, K5, K7, K9 (K6, K8 not extracted) | PASS, except H501 K7 F_a 979.47 vs 980 (0.53 N, 0.005 N past the tolerance): recorded |
| identities: Eq. (87), (81), (85)-(86), zero sliding at C, Eq. (114), Eqs (5)-(6), inv inverse | | PASS (1e-10 to 1e-14) |
**Printed values that do not follow (recorded, not judged):** C40 `eps_alpha` from its tips 1.4625, printed 1.44
(C3 item 3); H501 1.4631, printed 1.45; Fernandes C40 wheel `d_a` 115.54 is a misprint of 118.54 (Martins Table
3, and the tip from x; with 115.54 `eps_alpha` would be 1.236); the tips of H501, H951, Transfer 1-2, m=10 include a
tip alteration (not from x alone; e.g. H951 from x 78.27, printed 76.23). The two eps_alpha gaps would close with
tips about 0.16 mm (C40) and 0.07 mm (H501) smaller in diameter: a tip chamfer (active tip d_Na < d_a) is
plausible, not settled (OPEN R2-12). H501 K7 F_a: a torque printed with more digits (198.7 N.m gives 979.57)
would explain it (gear-expert). For v107: H501 needs the contact-line length along the path (Fernandes
Eqs A1-A16, R2-10); a spur pair off the single-pair zone needs a load share.

## 7. Depends on
SciPy (`brentq`) and the standard library. Nothing from `studies/` or the libraries. No `study.py` and no
`results/` (SQ-08).

## 8. How to run
`python -m pytest studies/r00_analytical_references/r04_involute`
