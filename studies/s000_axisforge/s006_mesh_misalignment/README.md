# s006 relative mesh misalignment from shaft deflection

## 1. Question
For one parallel cylindrical stage, how do the bearing span, the gear position (centred, offset, overhung) and the
helix angle (with hand and driver) change the relative slope of the two gear axes in the plane of action of the loaded
flank over the face width, i.e. **the shaft-bending part of f_sh** (f_sh_bend; ISO_6336-1_2019 Table 2, doc p. 6 /
PDF p. 14: f_sh is the "component of equivalent misalignment due to deformations of pinion and wheel shafts"; it is NOT
f_ma, the misalignment from manufacturing deviations, doc p. 5 / PDF p. 13), and the shaft slopes psi at the bearings?

## 2. Why
Ladder step 2 of the thesis concept v3 ("deflection and slope drive bearing tilt and mesh misalignment"); it sets the
psi range of s002 and is the bending part of f_sh for a later K_Hbeta study (blocked), which must add f_ma (manufacture),
f_be (bearings), f_ca (case) and the torsion on its own. proposals.md s006 (rank 3, T1).

## 3. Inputs / sweep
Pair: m_n 3.5 mm, z 20/30, x 0.1809/0.0891, b 23 mm, alpha_n 20 deg (GEARpie H501 of AxisForge's baseline cases; its
hand is not in that record, both hands are run), centre distance from x (88.41 mm spur). Shafts: uniform d 40 mm, S355,
rigid pins at 0 and L on both; one torque, 60.75 N.m on the pinion, 1000 rpm. Line of centres along +Y (results are
frame-free).
| swept | values |
|---|---|
| span L | 100, 150, ..., 400 mm (spur) |
| gear position x_g | centred 0.5 L, offset 0.3 L, overhung 1.2 L (0.2 L outside pin B; shaft 1.2 L + b/2) |
| driver | pinion shaft or wheel shaft as the source (two transmissions; same pinion torque and sense) |
| helix angle beta x hand | 0, 5, ..., 30 deg x pinion right / left, at L = 200 mm, all positions, both drivers (`convention-dependent (J-16)` for beta > 0) |
(`# OPEN R5-60`: proposals ask 0 / 15 / 30 deg; the 5 deg steps only make the line figure readable.) The x of H501 is
kept at every beta, so a_w moves (88.4 -> 91.5 -> about 95 mm) and the force falls at fixed torque: the column
`f_sh_bend_per_F_bt_um_per_kN` separates geometry from force.

## 4. Model
`construction.stage` (unsolved `ResolvedTransmission`, v001 `uniform_shaft`, AxisForge `GearPair`, `PairStage`,
`Transmission.resolve`), solved with `RigidSupportFEMSolver`, Euler-Bernoulli, on a piecewise-uniform mesh (about 40
elements; stations: ends, pins, gear, face edges); Timoshenko one-point (Cowper, about 400 elements) as RECORD.
`model.py` (no existing function projects shaft slopes on the plane of action: `face_load_factors` in
`axisforge/solvers/machine_elements/gears/iso6336.py` is a placeholder):
```python
line_of_action(wheel_shaft) -> l (Y, Z)            # the RadialLoad sum AxisForge put on the wheel shaft, normalised
mesh_misalignment(theta_p, theta_w, v_p, v_w, l, b) -> {f_sh_bend_um, f_perp_um, dtheta_rad, u_rel_um}
bearing_tilts(system, result) -> [{psi_rad, psi_load_rad, psi_n_rad, ...}]     # slope at each pin
pinion_twist(result, x_g, b, r_b) -> {twist_rad, twist_um}                     # RECORD
stage(*, span_mm, position, beta_deg=0, hand=1, driver="pinion") -> ResolvedTransmission (construction.py)
```
**f_sh_bend = b (theta_wheel - theta_pinion) . l**, theta = (theta_xy, theta_xz) = (dv_y/dx, dv_z/dx) of each shaft
at x_g (rigid gear bodies turn with the cross-section), l the transverse line of action of the loaded flank (the
plane of action contains l and X; ISO 6336-1 Table 2 footnote b: the components in the plane of action are
determinant). **Sign: f_sh_bend > 0 when the wheel moves away from the pinion along l toward +X: the flanks open at the
+X end of the face, the load concentrates at the -X end.** l follows driver and rotation by itself (it is the force
direction), so the sign needs no convention of this study. **Definition: the tangent at mid-face times b**; ISO compares
the face ends (F_betax, 7.5.3.4, doc p. 54 / PDF p. 62): the chord differs by (V_L + V_R)(b/2)^3/(6 EI) per shaft, at
most 1.6 % here (offset, L 100; 0.1 % at L 400; overhung 1.3 % at L 100). f_perp (across the plane of action: a change
of centre distance along the face) and u_rel (relative approach at x_g) are RECORD. psi per pin: modulus, in the plane
of the bearing force (psi_load) and normal to it (psi_n, `# OPEN R5-61`: proposals' "psi_n" read so). Pinion twist
over b: phi(x_g + b/2) - phi(x_g - b/2) times r_b, signed as AxisForge's phi. A point torque at x_g twists the half face
on the loaded side by T (b/2)/GJ; a torque spread uniformly over b gives the same end-to-end twist T b/(2 GJ), only
the shape along b differs (R5-62 closed). Whether the twist adds to or offsets f_sh_bend depends on the side the
torque enters (ISO combines them with sign): recorded, not combined (R5-63).
Assumptions: linear elastic shafts; rigid pins (no bearing compliance or tilt stiffness: s004 says whether it
matters); **the d 40 shaft section runs through the face and the gear bodies add no stiffness** (the pinion body,
root diameter about 62.5 mm, is about 6 times stiffer in torsion and stiffer in bending over b): the twist is an upper
bound, the bending effect is small for in-span slopes; the gear bodies are rigid for the projection; mesh forces
independent of deflections; no housing; no f_ma, f_be, f_ca.

## 5. v first
v002 (and through it v001's comparator, r01, r04) at the study's own conditions: every solved shaft goes through
`v002.check(shaft, result, EB)` (couples, C7 axial statics), every stage through `v002.gear_identities(resolved,
v002.mesh_judge(...))`: the mesh forces with their signs from the inputs (r04 `pair_geometry` / `mesh_forces` for the
profile-shifted working geometry; hand, sense, which gear drives), and the `Transmission` closure. The r01 slopes
enter only through `v002.reference` (the v's API). Helical points: `convention-dependent (J-16)` (D-0010 rows 21-22, U1).

## 6. Criteria (fixed before the run)
| id | criterion | tolerance | source |
|---|---|---|---|
| S1 | v002.check PASS for every judged record of every solved shaft | v002 (rel 1e-9) | proposals.md (v first) |
| S2 | v002.gear_identities PASS (I1-I6, I3s, I8 rel 1e-12, I7 `Transmission` closure) | as v002 | proposals.md |
| S3 | couple share of the reaction at pin A = C / span, per plane and shaft, C = r x F of the signed axial force of `mesh_judge` (v002 `physical_couples`) | 1e-9 of F_a r_w,max / span | proposals.md (identity), J-16 |
| S4 | f_sh_bend from the FEM nodes = f_sh_bend from the r01 slopes of the same system (spur judged; helical computed, J-16) | 1e-9 (v001 `TOL_NODAL`) of b max|theta| | Hermite nodal exactness |
| R | f_perp, u_rel, psi, twist, Timoshenko f_sh_bend: RECORD; ISO 6336-1:2019 sec. 7 f_sh rule: **not done** (not quick: the standard is a scan, read from page images, J-15; `# OPEN R5-63`) | - | - |
No GCI: the Euler-Bernoulli nodal values are exact (v001, v002).

## 7. How to run
`python -m studies.s000_axisforge.s006_mesh_misalignment.study` (about 2 min) writes `results/` (git-ignored):
`s006_points.csv`, `s006_bearings.csv`, `s006_v002.csv`, `s006_identities.csv`, `s006_criteria.csv`, and the figures
`s006_shaft_deflection_along_shaft_{XY,XZ}_offset_L200.png` (the face edges drawn with the "section step" marker),
`s006_shaft_ratio_sweep_{misalignment_vs_span,misalignment_vs_beta,psi_vs_span}.png`
(`studies/outputs/l000_axisforge/plots/shaft.py`; lines in place of the proposed contour, `# OPEN R5-64`). Tests:
`python -m pytest studies/s000_axisforge/s006_mesh_misalignment` (about 10 s).

## 8. Risks
J-16 (couple plane, sign and unit unruled: every helical number is tagged); rigid pins ignore the bearing's own tilt
and compliance; the gear bodies add no stiffness (twist an upper bound); the plane-of-action orientation flips with
driver and rotation (handled by taking l from the force). The long spans (L/d = 10) are beyond a realistic stage: the
relative approach u_rel reaches 187 um (centred, L 400), more than a typical backlash, and the assumption of mesh
forces independent of the deflections is shakier there.

## 9. Result (run of 2026-10-28, after review)
**S1-S4 PASS**: S1 1512/1512 (largest 6.3e-11; includes C7, the axial reaction), S2 2394/2394 (6.3e-14; signed judge,
I3s, I8), S3 72/72 (5.0e-10), S4 114/114 (2.4e-11).
**Answer.** For this pair and 40 mm shafts the shaft-bending part of f_sh grows as L^2: offset gear (0.3 L) 0.90 um at
L 100 to 14.4 um at L 400; overhung gear (0.2 L) three times more, 2.8 to 44.6 um; centred gear 0 at every span
(symmetry: the slope at mid-span vanishes on both shafts). It is positive at every off-centre point: the load moves to
the face end nearer the closer bearing. The offset value matches the simply supported slope
2 b F_bt a b'(b' - a)/(3 L EI) to 1e-10 (test). The driver does not change f_sh_bend (l and the loads mirror about the
line of centres together). Helix (J-16, L 200, offset): f_sh_bend falls from 3.61 um (spur) to 3.03 or 3.34 um at
30 deg. Most of the fall is the force on the line of action: F_bt -11.7 % (F_tw -13.3 % at fixed torque on a larger pitch
diameter, partly offset by alpha_wt 21.56 -> 24.00 deg; r04; f_sh_bend / F_bt 1.952 -> 1.857 um/kN, -4.9 %, hand +1 with
the pinion driving: geometry and couple); the couple F_a r_w adds +-0.15 um (+-4 % of the spur value), its
sign set by the pair (hand, driver) (hand +1 with the pinion driving = hand -1 with the wheel driving); centred gear
0.105 um of either sign at 30 deg, from the couple alone (the two couples have the same sense, only their difference
F_a (r_w2 - r_w1) tilts the mesh); f_perp +-0.35 um (offset). psi at the pins 0.02 - 0.75 mrad over the grid (the
range for s002). Shear (Timoshenko RECORD) leaves the in-span gear slopes unchanged (the section rotation follows
M/EI, and with M = 0 at both pins the shear deflection adds no rigid rotation); for the overhung gear it adds a
constant +0.189 um at every span, 2 b F_bt (0.2 L)/(kGA L), independent of L (+6.8 % at L 100, +0.4 % at L 400). Pinion
twist over the face 1.13 um (spur), 1.28 um at 30 deg (d 40 shaft under the face: an upper bound; RECORD).
**First run (same day), changed before this one:** the mesh overlaid 40 uniform nodes on the stations, leaving
0.1 - 0.25 mm elements beside 4 - 8 mm ones; Euler-Bernoulli lost digits there: S1 84 MISS (largest 4.7e-9, overhung
L 250 - 400 and reaction records), S3 4 MISS (1.9e-9), S4 6 MISS (6.8e-9). The mesh is now piecewise uniform between
stations (`study.solve`); no tolerance changed. Ordinary conditioning: a 0.1 mm element beside 5 mm ones raises the
bending stiffness ratio by (h_max/h_min)^3 about 1e5 (R5-65, note).
**Review (2026-10-28):** the quantity was first named f_ma; renamed f_sh_bend (physics, gear-expert MUST); the twist
text corrected; values unchanged.

## 10. Depends on
v002 (uncommitted at the run: `check`, `reference`, `along_shaft` re-exported from v001, `mesh_judge` (r04),
`gear_identities`, `physical_couples`, `THEORIES`), v001 `uniform_shaft`, `node_value`, `TOL_NODAL`,
`studies/outputs/l000_axisforge/plots/shaft.py`. AxisForge `feature/newton-solver` `d34ec7f` (read only); design-studies
`layout/studies-levels` `95bcaa9` + v001 / v002 changes + this case.
OPEN: R5-60 (beta steps), R5-61 (psi_n), R5-63 (f_sh of ISO 6336-1 not done; record the twist sign relative to
f_sh_bend for it), R5-64 (lines, not contour), R5-65 (conditioning note, sliver elements), R5-66 (nearest-node lookup:
a public AxisForge nodal accessor). R5-62 closed (twist text).
