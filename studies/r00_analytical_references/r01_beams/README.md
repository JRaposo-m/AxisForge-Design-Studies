# r01_beams: Euler-Bernoulli and Timoshenko beams on two pins

## 1. Reference
Closed-form deflection, rotation, bending moment and shear force of a beam on two pin supports, in
one bending plane: a uniform section under point loads (Macaulay) and a stepped section under point
and polynomial distributed loads (piecewise exact integration), Euler-Bernoulli and Timoshenko. The
bending and corrected shear stiffness are computed here from the section data (P-01). Judge of the
AxisForge shaft FEM (v001, which absorbs today's v002 stepped case). The uniform solver is a
special case of the stepped one, kept as an independent cross-check (C7).

## 2. Sources
| Key | What | Equation / figure | PDF page |
|---|---|---|---|
| Onate2013_vol2 | Euler-Bernoulli beam: kinematics, M = E I w'' (1.5), virtual work (1.6) | (1.1)-(1.6) | 34-36 |
| Onate2013_vol2 | Timoshenko beam: rotation and shear strain | (2.1)-(2.2) | 69 |
| Onate2013_vol2 | M = EI d(theta)/dx, Q = GA gamma | (2.4b)-(2.5b) | 70-71 |
| Onate2013_vol2 | gamma = Q / (k_z G A); k_z by shear strain energy | (2.6c), (2.7) | 72 |
| Onate2013_vol2 | shear factors; solid circle 6/7 at nu = 0; hollow circle k_z = 6/(7 + 20 K^2) without nu | Fig. 2.3 | 73 |
Onate writes Q and k_z for V and kappa. V = -dM/dx (equilibrium) and the closed forms of tests C3-C5 are
textbook statics with no source held (unverified).
| Cowper1966 | shear correction factor, solid circle, 6 (1 + nu) / (7 + 6 nu); J. Appl. Mech. 33(2) 335-340 | Table 1 (unverified) | MISSING (not in the library) |
| Hutchinson2001 | shear correction factor, solid circle, 6 (1 + nu)^2 / (7 + 12 nu + 4 nu^2); J. Appl. Mech. 68(1) 87-92 | (to confirm) | MISSING |
Onate2013_vol2: book page = PDF page - 31. The kappa formulas are kept as they were before R1; their
equation numbers wait for the two papers.

## 3. Scope and validity
Linear elastic, small slopes, statically determinate (two ideal pins: v = 0, rotation free); section
constant per segment with abrupt steps (fillets do not change EI or kGA); kappa constant per
segment; one bending plane (a circular shaft loaded in two planes is two calls, resultant by
`hypot`). Point loads may lie outside the span (overhangs). Distributed loads: polynomial intensity
of degree <= 6 is integrated exactly. Sign convention: forces and deflections share one positive
direction per plane; `M = sum F_k <x - x_k>`, `V = -dM/dx`; `M` and `V` are right-continuous at a
point load. Units: mm, N, MPa, N.mm, rad.
Domain errors raise: supports out of order; `E <= 0`, `nu` outside (-1, 0.5), not `0 <= d_i < d_o`.
**Hollow section with shear (Timoshenko): `NotImplementedError` (`# OPEN MG-02`)**: the hollow
Cowper / Hutchinson factors need Cowper 1966 and Hutchinson 2001, not in the library. Onate Fig. 2.3 prints a
hollow-circle k_z = 6/(7 + 20 K^2) without nu: not the nu-dependent factor of the solid one. Hollow
Euler-Bernoulli (`kappa=None`) is available.

## 4. Public API
```python
from studies.r00_analytical_references.r01_beams import (section_stiffness, solve_uniform, solve_stepped,
                                                         Segment, DistributedLoad, KAPPA)
section_stiffness(E, nu, d_o, d_i=0.0, kappa="cowper" | "hutchinson" | None) -> (EI [N.mm^2], kGA [N] or None)
solve_uniform(loads, x_A, x_B, EI, kGA) -> PlaneSolution             # loads: [(x [mm], F [N])]; kGA None: EB
solve_stepped(point_loads, distributed, x_A, x_B, segments, length) -> SteppedPlaneSolution
Segment(x_lo, x_hi, EI, kGA=None)          DistributedLoad(x_lo, x_hi, q)      # q(x) [N/mm]
# both solutions: .v(x), .M(x), .V(x), .R_A, .R_B; uniform also .v_bending(x); stepped also .theta(x)
kappa_cowper(nu), kappa_hutchinson(nu), KAPPA = {"cowper": ..., "hutchinson": ...}
```
The solutions are callable objects built once, not frozen dataclasses (structure_proposal B.2 asks
frozen results; their integration constants are set while building: OPEN MG-07).
Removed in R1: `solve_shaft(shaft_system, ...)` and `ShaftSolution` of both modules (they read the
loads from an AxisForge `ShaftSystem` through `LoadPlane`); the caller (a v `check`) passes the loads
of each plane and the section design data.

## 5. Data
None held: no published beam table is in the library (Roark, Cowper 1966 Table 1, Hutchinson 2001
are MISSING). No `published/` until one is.

## 6. Criteria (`tests/test_r01.py`)
| ID | Quantity | Role | Tolerance |
|---|---|---|---|
| C1 | EI, kGA from E, nu, d (solid; hollow EB); kappa ratio | identity | rel 1e-15 |
| C2 | kappa at nu 0.3 (doctest values), both 6/7 at nu = 0 | identity | 6 digits / rel 1e-15 |
| C3 | simply supported, central load: v(L/2) = F L^3/(48 EI) + F L/(4 kGA), R, M | closed form | rel 1e-12 |
| C4 | overhang tip load: F a^2 (L + a)/(3 EI) + F a (L + a)/(kGA L) (cantilever part F a^3/(3 EI)) | closed form | rel 1e-12 |
| C5 | simply supported, uniform q (stepped solver): 5 q L^4/(384 EI) + q L^2/(8 kGA) | closed form | rel 1e-12 |
| C6 | Euler-Bernoulli = Timoshenko as kGA -> inf; shear makes the beam softer | limit | rel 1e-12 |
| C7 | uniform = stepped with three equal segments and overhangs: v, M, V, R | identity | 1e-12 of max v / 1e-10 |
| C8 | reactions balance a point and a linear distributed load | identity | rel 1e-13 |
| C9 | hollow kappa, bad section, supports out of order raise | validity | exact |
A cantilever is outside the two-pin scope: C4 holds its term.

## 7. Depends on
NumPy only; nothing in this repository, AxisForge, SlipPY or the bridge (checked by
`tests/test_imports.py`). Blocker for the published-table test: Cowper 1966, Hutchinson 2001.

## 8. How to run
```bat
python -m pytest studies\r00_analytical_references\r01_beams
```
No `study.py` and no `results/` (SQ-08).
