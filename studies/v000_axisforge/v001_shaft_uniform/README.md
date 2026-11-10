# v001 shaft uniform

## 1. Question
Does the AxisForge shaft FEM (`RigidSupportFEMSolver`, Euler-Bernoulli and Timoshenko) reproduce the
closed-form beam of r01 on a uniform shaft on two rigid pins, and how? One validation: **FEM vs r01, nodal
deflection and rotation at machine precision for Euler-Bernoulli (Hermite nodal exactness), and the known
O(h^2) error of the one-point Timoshenko element, exactly** (v_prerequisites row 9). Ported from
`v000_axisforge/v001_shaft_fem_uniform_radial_load` at the tag `studies-pre-sweep` (shaft, material, mesh
helper, beam-model presets; the gear stage left to v002, the old GCI thresholds dropped: section 6).

## 2. Sources
| key | item | PDF page |
|---|---|---|
| Onate2013_vol2 | Sec. 1.5: the 2-node Hermite stiffness (and the nodal force vector "for some loads") equals standard matrix analysis; nodal exactness follows from it (Onate does not state it in these words) | 66 |
| Onate2013_vol2 | Fig. 1.17 (simply supported beam, L 4, P -1, Q -1, E 47619.05, I 2.133e-7, SI) and Table 1.1 (error of the CCB / CVB rotation-free elements, uniform meshes) | 65-66 |
| Onate2013_vol2 | Fig. 2.3: k_z = 5/6 (rectangle), 6/7 (circle, no nu) | 73 |
| Onate2013_vol2 | Eqs (2.35)-(2.36): exact Timoshenko cantilever flexibility | 81 |
| Onate2013_vol2 | text after Eq. (2.33): rectangular example nu = 0.25, k_z = 5/6 | 80 |
| Onate2013_vol2 | Eqs (2.38)-(2.42): one-point shear integration, its flexibility (2.40), r_w = (3 lambda^2 + 3)/(4 lambda^2); Table 2.1: r_w for 1-16 elements, lambda -> inf | 83-84 |
| Onate2013_vol2 | Fig. 2.17 (square beam, L 4, q 1) and Table 2.3 (slender and thick closed-form central deflections, h/L 0.01-0.20) | 124, 126 |
Book page = PDF page - 31. The rows are in `literature/published/onate2013_vol2.csv`, read from the page images on
2026-10-28 (the library note `06_Library/system/Onate2013_vol2.md` has no content pass yet). r01 holds the
closed forms (its README section 2).

## 3. Model
AxisForge `RigidSupportFEMSolver` (frame elements, two bending planes solved apart, supports by elimination:
v = 0 at every bearing, u = 0 at the locating one, rotation free, so ideal pins). Four beam models
(`construction.THEORIES`): Euler-Bernoulli (cubic Hermite); Timoshenko 2-node linear, one-point shear integration,
Cowper and Hutchinson kappa; Timoshenko full ("exact") integration, Cowper (RECORD: locks). Uniform meshes of
n = 8, 16, 32, 64, 128 elements over the shaft (`uniform_nodes`), every support and load on a node.

Why the one-point Timoshenko element is judged by an identity and not by r01 directly: its flexibility (Onate
(2.40)) differs from the exact element's (2.35) only by h^3/(4 EI) in place of h^3/(3 EI). It is therefore the
exact element of a beam with `1/kGA_eff = 1/kGA - h^2/(12 EI)` (`check.kga_reduced_integration`), and on a
uniform mesh with point loads on nodes its nodal values equal r01 at `kGA_eff` (not with a distributed load: the
AxisForge Timoshenko load vector q h/2 lacks the end moments +-q h^2/12 of the kGA_eff element, so `check` makes C4
RECORD then, `# OPEN R5-25`); against r01 at the true kGA its error is
second order in h, the FEM too stiff. For a cantilever of n elements this is Onate's `r_w = 1 - 1/(4 n^2)` (Table 2.1).
A side result: in the XZ plane (load inside the span, M = 0 at both pins) the nodal rotations of the one-point
element are exact (shear adds only -M/kGA plus a linear term, which vanishes there); in the XY plane (load on the
overhang, M(x_B) != 0) they are not.

Planes and signs: AxisForge v_xy along +Y, v_xz along +Z, nodal rotation theta = rotation of the cross-section in
the plane, positive from +x toward +v (dv/dx for Euler-Bernoulli, dv/dx - gamma for Timoshenko, gamma = V/kGA; the
AxisForge docstrings that write dv/dx are register R-584), reactions
`R_*` = force of the support on the shaft along +v; r01 uses one plane (x, v) with the same sign for forces,
deflections, reactions and theta from +x toward +v. With forces only (no couple) each plane maps as it is: no
convention tag (R-003, J-16 do not apply); the couple trap CW-01 is v002's.

## 4. Public API (`__init__.py`)
```python
check(system, result, settings, *, settlements_mm=None, kGA_override=None, case="v001", label="") -> list[dict]
reference(system, settings, *, kGA_override=None, h_mm=None, couples=True) -> {"XZ": ..., "XY": ...}  # r01 of the loads
segments(system, settings, *, kGA_override=None, h_mm=None) -> [Segment]; section(...) -> (EI, kGA) of one section
plane_couples(system, plane) -> [(x, C)] (J-16);  rigid_body(x, x_A, x_B, delta_A, delta_B) -> (v, theta)
along_shaft(system, result, settings, *, settlements_mm=None, ...) -> records for plots.shaft.deflection_along_shaft
node_value(result, x_mm, key);  richardson_verdict(name, fin, exact, ...) -> C5 / C6 record
kga_reduced_integration(kGA, EI, h) -> kGA_eff [N]
literature.cases() -> [LiteratureCase(name, build, published, source)]   # Onate Tables 2.1, 1.1, 2.3
uniform_shaft(*, length_mm, diameter_mm, supports_mm, loads, material_id, label) -> ShaftSystem (unsolved)
uniform_nodes(n_elements, length_mm) -> list[float];   THEORIES                      # mesh, beam models
```
`check.py` is the one shaft comparator of v001 and v002 (merged at the v002 review, R5): any shaft on two pins,
uniform or stepped, with forces, distributed forces, couples (`ExternalMoment`, tagged J-16) and pin settlements
(rigid-body field), plus C7 (axial reaction at the locating pin vs statics) when an axial load is present; v002, s004,
s006 pass theirs. `add_pins(system, supports_mm)` (construction) places the two zero-width pins. `LiteratureCase` is imported from v101 (`# OPEN R5-22`, as R2-02).

## 5. Inputs
Shaft L 200 mm, d 20 mm, S355 (E 210 GPa, nu 0.3); pins at 25 mm (locating) and 175 mm (non-locating), two
25 mm overhangs; F1 500 N at 75 mm along -Z (plane XZ), F2 200 N at 200 mm (overhang tip) along +Y (plane XY).
L/d = 10. Shear sweep: the same shaft with d = L/(L/d), L/d = 2 ... 100 (below span/d ~ 3, i.e. L/d 2-3 here, neither
beam theory is a valid shaft model: Saint-Venant end zones, local contact; there the ratio compares two models only). Supports are 6204 bearings of zero width
(`# OPEN R5-21`: AxisForge has no point support; the solver reads position and arrangement only). Literature:
Onate's SI beams converted once in `literature/__init__.py` (m -> mm, Pa -> MPa, N/m -> N/mm), a square section
replaced by the circle of the same I, its shear stiffness given as `kGA_override`. Table 2.1 by symmetry: span 2L,
force 2P at mid-span, n elements per half; lambda -> inf as `kGA_override = 1e6 EI/L^2` (shear part 3e-6).

## 6. Criteria (fixed before the run)
Field errors: largest nodal deviation over the largest absolute r01 value of the field in the plane, each loaded plane.
| id | quantity | tolerance | source of the tolerance |
|---|---|---|---|
| C1 | Euler-Bernoulli nodal v vs r01 | rel 1e-9 | Hermite nodal exactness; machine precision with two decades for cond(K) ~ n^4 (brief) |
| C2 | Euler-Bernoulli nodal theta vs r01 | rel 1e-9 | as C1 |
| C3 | reactions, every model, vs r01 | rel 1e-9 | statics (determinate) |
| C4 | one-point Timoshenko nodal v, theta vs r01 at `kGA_eff` (uniform mesh) | rel 1e-9 | Onate (2.35) vs (2.40) |
| C5 | Euler-Bernoulli load-point deflection over n = 8 ... 128: Richardson status `flat` | flat (rel 1e-8) | nodal exactness |
| C6 | one-point Timoshenko load-point deflection: Richardson extrapolation vs r01 at kGA, status `monotonic` | rel 1e-9 | error exactly c h^2 |
| L1 | Table 2.1 r_w, n = 1 ... 16 | half a unit of the printed digit (5e-4), inclusive | printed digits (0.9375 sits on the rounding boundary of 0.938) |
| L2 | C1 on Onate's system (Fig. 1.17 / Table 1.1): Euler-Bernoulli central deflection error, point and uniform load, 4 ... 32 elements per half (the 0.00 is ours: Onate prints only the CCB / CVB errors; covers the unit conversion and the consistent load) | 0.005 % (prints 0.00) | printed digits of the % column |
| L3 | Table 2.3 slender central deflection, Euler-Bernoulli, 2 elements per half | half a unit of the printed digit | printed digits |
| T1 | Timoshenko vs r01 at kGA (fields); T2 full-integration series; R1 nodal M; L4 Table 2.3 thick (nu, k_z not printed); shear ratio FEM vs r01 | RECORD | - |
The old case's GCI thresholds (v 1 %, M 2 %, "provisional") are not carried: with an exact judge the nodal
error itself is judged. Richardson (`numerical_solvers.richardson`, kind "count", `flat_atol=0`, `max_p` = 4 for
Hermite, 2 for the linear Timoshenko element, every level) runs on the load-point deflection (F1, plane XZ). No
between-node quantity is evaluated: `ShaftResults` is nodal, and an interior value would be this case's own
interpolation, not AxisForge's output.

## 7. How to run
`python -m studies.v000_axisforge.v001_shaft_uniform.study` (about 10 s) writes `results/` (git-ignored):
`v001_check.csv`, `v001_richardson.csv`, `v001_shear.csv`, `v001_literature.csv`, and the figures
`v001_shaft_deflection_along_shaft_{XZ,XY}_n8.png`, `v001_shaft_error_vs_elements.png`,
`v001_shaft_ratio_sweep_shear.png`, `v001_shaft_ratio_sweep_onate_table_2_1.png` (`studies/outputs/l000_axisforge`).
`notebook.ipynb` (jupytext pair of `notebook.py`) runs the same and is committed with its outputs.
Tests: `python -m pytest studies/v000_axisforge/v001_shaft_uniform` (about 5 s).

## 8. Result (run of 2026-10-28)
**PASS: every judged record (103 of 204 records; C1-C6, L1-L3), no miss.**
| id | largest error | note |
|---|---|---|
| C1 / C2 | 1.2e-11 / 1.0e-11 | grows with n (round-off, cond(K)); 2e-15 at n = 8 |
| C3 | 1.8e-11 | all four models |
| C4 | 8.2e-13 | Cowper and Hutchinson, both planes, v and theta |
| C5 | flat at every triple | |
| C6 | 3.8e-13 (Cowper), 5.1e-14 (Hutchinson) | observed p = 2.000000000 |
| L1 | r_w 0.750003, 0.937503, 0.984378, 0.996097, 0.999026 vs 0.750, 0.938, 0.984, 0.996, 0.999 | largest 4.97e-4 (n = 2: exact 0.9375 prints 0.938 only by rounding half up; the finite SLENDER_KGA_FACTOR adds 3e-6 the right way) |
| L2 | 8.2e-9 % (32 per half; CCB 0.19 %, CVB 0.05 % printed) | |
| L3 | -328.125, -0.525, -0.0328125, -0.00205078 m vs -3.28E+02, -5.25E-01, -3.28E-02, -2.05E-03 | |
RECORD: T1 one-point Timoshenko 3.9e-2 (n = 8) to 1.5e-4 (n = 128), ratio 4 per halving; full integration 0.42
(n = 8) to 2.8e-3 (n = 128), observed p 1.29 -> 1.94, not converged by GCI 1 % at n = 128 (locking); R1 nodal M:
Euler-Bernoulli 1.8e-11, Timoshenko O(h) (nodal M from one adjacent element, off by about V h/2: registers R-224,
R-391, not a new finding). Shear ratio v_T/v_EB at F1: 2.10 (L/d 2),
1.044 (L/d 10), 1.0004 (L/d 100), FEM at n = 128 within 1.2e-4 of r01. L4 (Table 2.3 thick, with nu 0.25,
k_z 5/6 assumed): r01 -0.03360 vs printed -0.0335 (h/L 0.10), -0.002248 vs -0.00222 (0.20): Onate's thick column
is not reproduced with these inputs (`# OPEN R5-24`; h/L 0.05 fits nu 0.25, 0.10 and 0.20 need about 0.1: estimate,
not used). Librarian (2026-10-28): Table 2.3's E, nu, k_z are not printed anywhere in the PDF; the text sends the
reader to [ZO3] (Zarate and Onate, CIMNE PI385, 2012), which is not in the library: R5-23 and R5-24 stay open.

## 9. Depends on
r01 (`028c34e`, `solve_uniform`, `solve_stepped`, `section_stiffness`), `numerical_solvers.richardson`,
`studies/outputs` (`l000_axisforge/plots/shaft.py`), v101 `LiteratureCase`. AxisForge `feature/newton-solver`
`d34ec7f` (read only); design-studies `layout/studies-levels` `84344df` + this case (uncommitted at the run).
OPEN: R5-21 (pin as a zero-width bearing), R5-22 (LiteratureCase), R5-23 (Table 2.3 E taken from Fig. 1.17),
R5-24 (Table 2.3 nu, k_z; [ZO3] not held), R5-25 (C4 identity for point loads only; RECORD with a distributed
load); the level folder name is R5-00 (agents_runner). Beam-model names: the `THEORIES` keys
(`euler_bernoulli`, `timoshenko/<kappa>/<integration>`) are the `theory` of the records (`check.theory_name`).
