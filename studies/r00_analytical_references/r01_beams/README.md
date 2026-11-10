# r01_beams: Euler-Bernoulli and Timoshenko beams on two pins, one fixed end, or several rigid or elastic supports

## 1. Reference
Closed-form deflection, rotation, bending moment and shear force of a beam on two pin supports or
one fixed end (cantilever), in one bending plane: a uniform section under point forces and point
couples (Macaulay) and a stepped section under point forces, point couples and polynomial
distributed loads (piecewise exact integration), Euler-Bernoulli and Timoshenko. The bending and
corrected shear stiffness are computed here from the section data (P-01). Judge of the AxisForge
shaft FEM: v001 (uniform shaft) and v002 (stepped shaft with a gear load; the couple F_a r_w of a
helical mesh, CW-01, CW-02). The uniform solver is a special case of the stepped one, kept as an
independent cross-check (C7, C15), except for the statics: both take the reactions from the shared
`_reactions` (equilibrium is tested on its own, C8, C14, and the closed forms C3-C5, C10-C13 catch a sign).
**Extension (R5, SW-03):** `supported_beam.solve_supported` solves the statically INDETERMINATE beam on any number of
supports (rigid or linear radial springs, rotational springs up to a clamp, prescribed settlements and rotations,
per-span EI through the segments) by the force method on the exact determinate solver: judge of v005 (shaft on three
supports and springs).

## 2. Sources
| Key | What | Equation / figure | PDF page |
|---|---|---|---|
| Onate2013_vol2 | Euler-Bernoulli beam: kinematics, M = E I w'' (1.5), virtual work (1.6) | (1.1)-(1.6) | 34-36 |
| Onate2013_vol2 | Timoshenko beam: rotation and shear strain | (2.1)-(2.2) | 69 |
| Onate2013_vol2 | M = EI d(theta)/dx, Q = GA gamma | (2.4b)-(2.5b) | 70-71 |
| Onate2013_vol2 | gamma = Q / (k_z G A); k_z by shear strain energy | (2.6c), (2.7) | 72 |
| Onate2013_vol2 | shear factors; solid circle 6/7 at nu = 0; hollow circle k_z = 6/(7 + 20 K^2) without nu | Fig. 2.3 | 73 |
| Onate2013_vol2 | Timoshenko clamp: w1 = 0, theta1 = 0 (the section rotation) | (2.29) | 79 |
| Onate2013_vol2 | Timoshenko cantilever: flexibility matrix (tip force and tip moment), w2 = (L/GA* + L^3/(3 EI)) P, theta2 = L^2 P/(2 EI) | (2.35)-(2.36) | 81 |
Onate writes Q and k_z for V and kappa. V = -dM/dx (equilibrium) and the closed forms of tests C3-C5 and
C10-C14 are textbook statics with no source held: no beam handbook (Roark, Gere, Timoshenko) is in the
library. The cantilever cases C11-C12 are printed in Onate (2.35)-(2.36) (tip force: deflection and rotation,
EB and Timoshenko; tip moment: the flexibility entries L^2/(2 EI), L/EI); the clamp of C11-C13 is Onate (2.29). The
couple on a simply supported span (C10), the mirrored and interior clamps and the cantilever under uniform q (C13)
are derived in the test docstrings by Macaulay integration of EI v'' = M (no print held).
| Harris2007_vol2 | sign senses of ch. 9: y and delta_r DOWN (along P), theta = dy/dx, F up on the shaft, M_1 CCW, M_2 CW; r01 map delta = -settlement, theta_H = -rotation, F = R, M_1 = C[0], M_2 = -C[1] (R5-141 answered) | Eqs 9.6-9.10, Fig. 9.4 | 278-279 |
| Harris2007_vol2 | two-bearing slope-deflection equations: support forces and moments from prescribed end rotations and deflections | Eqs 9.24-9.27 | 281 |
| Harris2007_vol2 | three-bearing shaft: the same per span, Eq. 9.30 prints 3 (delta_1 - delta_2)/l where Eq. 9.24 has 2 (FLAG: use 9.24) | Eqs 9.28-9.35 | 282-284 |
| Harris2007_vol2 | three bearings, point load in span 1, rigid bearings (no moment loads, bearing deflection differences negligible, p. 284): F_1, F_2, F_3 | Eqs 9.42-9.44, Fig. 9.9 | 284-285 |
| Cowper1966 | shear correction factor, solid circle, 6 (1 + nu) / (7 + 6 nu); J. Appl. Mech. 33(2) 335-340 | Table 1 (unverified) | MISSING (not in the library) |
| Hutchinson2001 | shear correction factor, solid circle, 6 (1 + nu)^2 / (7 + 12 nu + 4 nu^2); J. Appl. Mech. 68(1) 87-92 | (to confirm) | MISSING |
Onate2013_vol2: book page = PDF page - 31. The kappa formulas are kept as they were before R1; their
equation numbers wait for the two papers.

## 3. Scope and validity
Linear elastic, small slopes. `solve_uniform` and `solve_stepped`: statically determinate only, two ideal pins (v = 0, rotation free) at
`x_A < x_B`, or one ideal fixed end at `x_A`, at an end or inside the beam (`x_B=None`: v = 0 and theta = 0, theta the rotation of the
cross-section, so the shear strain is free at the clamp: the Timoshenko cantilever gains P L / kGA). A third
support, a pin with a fixed end or two fixed ends make the beam hyperstatic: out of scope, and the API has
no argument for them in `solve_uniform` / `solve_stepped`; `solve_supported` (below) takes them (SW-03 answered
by the R5 extension). Section
constant per segment with abrupt steps (fillets do not change EI or kGA); kappa constant per
segment; one bending plane (a circular shaft loaded in two planes is two calls, resultant by
`hypot`). Point loads may lie outside the span (overhangs). Distributed loads: polynomial intensity
of degree <= 6 is integrated exactly. Units: mm, N, MPa, N.mm, rad.

**Sign convention (one bending plane with axes x along the beam, v transverse).** Forces and deflections
share one positive direction, +v. A **couple is positive when it turns from +x toward +v** (counterclockwise
with x to the right and v up; right-handed about the axis e_x x e_v). The rotation theta has the same sense.
Bending moment and shear force: `M = sum F_k <x - x_k> - sum C_k <x - c_k>^0`, `EI theta' = M`,
`V = -dM/dx` between loads; a couple lowers M by C_k across c_k and adds no shear force (so no shear
deflection). `M` and `V` are right-continuous at a load. Equilibrium, with a force F at x giving the moment
x F about x = 0: `sum F + R_A + R_B = 0`, `sum x F + sum C + x_A R_A + x_B R_B + C_A = 0` (R_B = 0, C_A the
reaction couple at a fixed end; C_A = 0 on two pins). r01 imports no AxisForge and knows nothing of its
axes: the v case that uses it (v002) maps AxisForge `ExternalMoment` and the plane XY / XZ to this
convention, and writes the mapping there (CW-01, D-0010). Trap for that mapping: the positive axis of a couple
is e_x x e_v; in a plane (x, y) it is +z, but in a plane (x, z) it is e_x x e_z = -e_y, so a moment M_y about +y
enters r01 as C = -M_y.
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
solve_uniform(loads, x_A, x_B, EI, kGA, *, couples=()) -> PlaneSolution
solve_stepped(point_loads, distributed, x_A, x_B, segments, length, *, couples=()) -> SteppedPlaneSolution
solve_supported(point_loads, distributed, supports, segments, length, *, couples=()) -> SupportedSolution
Support(x, k_r=inf, k_theta=0.0, settlement=0.0, rotation=0.0)   # [mm], [N/mm], [N.mm/rad], [mm], [rad]
# SupportedSolution: .v(x), .theta(x), .M(x), .V(x), .R [N] and .C [N.mm] per support (forces and couples on the beam)
# loads: [(x [mm], F [N])]; couples: [(x [mm], C [N.mm])]; x_B=None: fixed end at x_A; kGA None: EB
Segment(x_lo, x_hi, EI, kGA=None)          DistributedLoad(x_lo, x_hi, q)      # q(x) [N/mm]
# both solutions: .v(x), .theta(x), .M(x), .V(x), .R_A, .R_B (None at a fixed end), .C_A (0 on two pins);
# uniform also .v_bending(x)
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
| C10 | couple C0 at a on pins at 0, L: R_A = -R_B = C0/L; M jump -C0; v(x <= a) = C0 x (x^2 + 3 (L - a)^2 - L^2)/(6 L EI), v(a) = C0 a (L - a)(L - 2a)/(3 L EI); v unchanged by shear, theta(0) + C0/(L kGA); V = -C0/L | closed form (Macaulay) | rel 1e-11 |
| C11 | cantilever, tip force P: v(L) = P L^3/(3 EI) + P L/kGA, theta(L) = P L^2/(2 EI), R_A = -P, C_A = -P L, M = P (L - x) | closed form | rel 1e-12 |
| C12 | cantilever, tip couple C0: v(L) = C0 L^2/(2 EI), theta(L) = C0 L/EI (shear adds nothing), C_A = -C0 | closed form | rel 1e-12 |
| C13 | cantilever fixed at the right end (mirror of C11); fixed inside the beam with a force at each end (two cantilevers); uniform q on a cantilever (stepped): q L^4/(8 EI) + q L^2/(2 kGA) | closed form | rel 1e-12 |
| C14 | equilibrium of forces and couples, pins and fixed end, both solvers; M = V = 0 past the last load | identity | 1e-12 / 1e-9 |
| C15 | uniform = stepped with forces, couples, pins or fixed end: v, theta, M, V, C_A | identity | 1e-12 of max v / 1e-10 |
| C16 | EB = Timoshenko as kGA -> inf with couples, pins and fixed end (v, theta) | limit | rel 1e-12 |
| C17 | three rigid supports, point load in span 1: F_1, F_2, F_3 of Harris2007_vol2 Eqs 9.42-9.44 (PDF p. 285); inputs l1 100, l2 150, a 40 mm, P 1000 N are this case's own (Harris prints no numbers) | printed equations | rel 1e-12 |
| C18 | Clapeyron: M over the middle support -P a b (l1 + a) / (2 l1 (l1 + l2)) | closed form (three-moment equation) | rel 1e-10 |
| C19 | two equal spans under q: reactions 3/8, 5/4, 3/8 q L, M = -q L^2/8 over the middle | closed form | rel 1e-12 / 1e-9 |
| C20 | middle support settled by s: R = 6 EI s / L^3, ends -R/2, v = s there | closed form (pin-pin 2L) | rel 1e-12 |
| C21 | spring limits: k_r = 1e15 equals rigid; k_r = 1e-9 equals the two-pin `solve_stepped` | limit | 1e-9 |
| C22 | two radial springs (determinate): reactions of the pins, ends sink by R/k, M unchanged | identity | rel 1e-12 |
| C23 | clamp (k_theta = inf) and pin under q: 5/8, 3/8 q L, clamp couple q L^2/8, theta(0) = 0 | closed form | rel 1e-12 |
| C24 | clamped-clamped span, downward P, both ends settled and rotated: SIGNED F_1, M_1, F_2, M_2 of Harris Eqs 9.24-9.27 (PDF p. 281) with delta = -settlement, theta_H = -rotation, F = R, M_1 = C[0], M_2 = -C[1] (senses Eqs 9.6-9.10, Fig. 9.4, PDF p. 278-279); prescribed end state; equilibrium | printed equations (signed) | rel 1e-12 |
| C25 | per-span EI keeps equilibrium; Timoshenko kGA -> inf equals EB; shear makes it softer | identity / limit | rel 1e-12 / 1e-7 |
| C26 | clamp with a vanishing spring is the cantilever (P, P a) | limit | rel 1e-9 |
| C27 | fewer than two supports, coincident, outside, k_r <= 0, k_theta < 0 raise | validity | exact |
C17-C27 are in `tests/test_r01_supports.py`; `test_r01.py` is unchanged.
Test values (C10-C16): steel shaft E 210 GPa, nu 0.3, d 30 mm (EI = 8.3498e9 N.mm^2, Cowper kGA = 5.0605e7 N),
L = 200 mm, P = 1000 N, C0 = 5e4 N.mm at a = 60 mm, q = 5 N/mm.

## 7. Depends on
NumPy only; nothing in this repository, AxisForge, SlipPY or the bridge (checked by
`tests/test_imports.py`). Blocker for the published-table test: Cowper 1966, Hutchinson 2001.

## 8. How to run
```bat
python -m pytest studies\r00_analytical_references\r01_beams --doctest-modules
```
`--doctest-modules` also runs the docstring examples of `uniform_beam`, `stepped_beam` and `supported_beam`; the
default `python -m pytest` does not. `SupportedSolution.M` / `.V` return the right-hand side of a jump at a support
(compare at x_s +/- eps).
No `study.py` and no `results/` (SQ-08).
