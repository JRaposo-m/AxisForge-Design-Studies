# v101 Hertz point

## 1. Question
Does SlipPY's static BEM, driven through `slippy_bridge.contact`, reproduce the Hertz contact of a smooth
point contact (approach, peak pressure, semi-axes, load), and from which grid? One validation: **BEM point
contact vs r02 Hertz, approach within 0.5 % on the spatial kernel at b/h >= 16** (v_prerequisites row 1).
Ported from `v100_dry_contact/v103_hertz_point_grid_domain` at the tag `studies-pre-sweep` (its mesh part;
the domain and kernel sweep went to v104, the subsurface part waits for v108, Q-03).

## 2. Sources
| key | item | PDF page |
|---|---|---|
| Johnson1985 | Sec. 4.2(a), Eqs (4.19)-(4.24): `a`, `delta`, `p0`; text after (4.21): `p0 = (3/2) p_m`; Eq. (3.39) quoted: `p = p0 sqrt(1 - r^2/a^2)` | 102-103 |
| Johnson1985 | Table 4.1 (elliptical contacts, circular limit): subsurface only, not used here | 109 |
| PolonskyKeer1999 | error measures and bounds: `eps_r < 1/M`, `eps_p ~ M^-3/2`, `M` nodes across the contact | 9-10 |
The printed rows are in `literature/published/johnson1985_ch4.csv` (vault extraction of 2026-10-16, through
the tag). Library notes: `06_Library/contact/Johnson1985.md` (scanned book), `PolonskyKeer1999.md`.

## 3. Model
Two smooth, frictionless, linear elastic half-spaces with their exact profiles (`SweptCircleBody`); SlipPY
`StaticStep` through `slippy_bridge.contact.solve` on a `Grid.around` grid (h a power of two in metres, rounded
down from the `a/h` asked, so the grid has `a/h` in [asked, 2 x asked)). r02 and the bridge's own Hertz both
run SlipPY `hertz_full`: the comparison is the BEM against SlipPY's Hertz (verified by José, rule 13).
Kernel: spatial (`use_frequency_domain=False`; this case's choice, the bridge default is True; SQ-19, Q-06
open): linear convolution on a span of 2n nodes, zero-padded, no images; domain half-length 2 contact radii.
Frequency kernel (RECORD) at the bridge's `SPAN_X_0` = 4 radii: the kernel is periodic with period 2 n h and
its zero-frequency term is the sum of the spatial kernel (SlipPY `elastic_material.py:161-167`,
`materials.py:158-162`), so the contact feels images of its load. Load control: **interference** (rigid
approach given, the load is found; J-7) primary, `normal_load` the cross-check.

## 4. Public API (`__init__.py`)
```python
check(result, *, case="v101") -> list[dict]     # a SOLVED point contact (StaticResult: construction -> solve)
reference(contact, load) -> r02 HertzResult     # the two bodies as given, at load [N] (line: [N/m])
bem_load(result) -> float                       # load the BEM carries (nominal, or found); OPEN R2-04
pressure_cuts(result), pressure_l1_error(result)   # p/p0_H on the principal cuts; Polonsky-Keer eps_p
literature.cases() -> [LiteratureCase(name, build, published, source)]
PointContact, ball_on_flat(load=500.0, *, dw, E, nu, label)   # construction, unsolved
```
`PointContact` is a one-line `Contact` subclass (`line = False`): the bridge exports no generic point contact
(OPEN R2-03, issue row). `LiteratureCase`, `reference`, `bem_load`, `pressure_cuts`, `pressure_l1_error` are
public because v102-v104 import them (OPEN R2-02).

## 5. Inputs
Steel ball `Dw` 12.7 mm on a steel flat, `E` 210 GPa, `nu` 0.3, `P` 500 N (chosen: every judged quantity is
a ratio). Hertz (r02): `a` 0.274296 mm (`a/R` 0.043), `p0` 3173.02 MPa, `delta` 11.8485 um. Grids `a/h`
asked 8, 16, 32 (got 8.99, 17.98, 35.95 on the spatial kernel). SI throughout.

## 6. Criteria (fixed before the run; C2, C5, L1-L2 made RECORD after the review, tolerances unchanged)
Judged only on the spatial kernel, `b_H/h >= 16`, converged, edge clear, load balanced (load control), Hertz
reference in range; otherwise RECORD. `M = 2 b_H / h`, about Polonsky and Keer's M (their grid rows, the grid
slightly larger than the contact diameter, PDF p. 10). C3 and C4 are a proposed reading of their bounds: they
bound an L1 pressure error and the radius of the counted contact area, here applied to the peak and to the
semi-axes the bridge measures below one node (C4 is about 10x loose); and they tested M = 64-1024, so at
`b/h` about 18 (M about 36) the bound is extrapolated (`_evidence/v101_librarian.md`).
| id | quantity | tolerance | source of the tolerance |
|---|---|---|---|
| C1 | approach `delta` (interference: given vs Hertz at the load found) | 0.5 % | v_prerequisites row 1 |
| C2 | load found at the given approach vs `P_H = P (delta/delta_H)^(3/2)` (interference only) | RECORD | C1 read the other way (`-1.5 x` its error) |
| C3 | `p0` | `M^(-3/2)` | author's choice after Polonsky-Keer `eps_p ~ M^-3/2` (PDF p. 10) |
| C4 | `a`, `b` (the same on this circle) | `1/M` | Polonsky-Keer `eps_r < 1/M` (PDF p. 10) |
| C5 | Polonsky-Keer `eps_p = sum abs(p - p_H) / sum p_H` over the loaded nodes | RECORD | no constant printed |
| L1 | `p0/p_m` vs Johnson's printed 3/2, `p_m = P/(pi a b)` from the BEM alone | RECORD | implied by C1-C4 |
| L2 | `delta R/(a b)` vs 1 (Eq. (4.23)) from the BEM alone | RECORD | implied by C1-C4 |

## 7. How to run
`python -m studies.v100_dry_contact.v101_hertz_point.study` (about 7 s) writes `results/` (git-ignored):
`v101_bem.csv` (bridge records), `v101_check.csv`, figures `v101_contact_relative_errors.png`,
`v101_contact_pressure_cut_spatial_ah16_asked.png` (`studies/outputs`). `notebook.ipynb` (jupytext pair of
`notebook.py`) runs the same and is committed with its outputs.
Tests: `python -m pytest studies/v100_dry_contact/v101_hertz_point` (about 3 s).

## 8. Result (run of 2026-10-28)
**PASS: C1, C3, C4 at `a/h` 17.98 and 35.95 under both controls (3 independent quantities x 2 grids x 2
controls), no miss.**
| a/h (spatial) | control | C1 delta | C2 load | C3 p0 | C4 a = b | C5 eps_p | L1 p0/p_m | L2 |
|---|---|---|---|---|---|---|---|---|
| 8.99 (RECORD) | interference | -0.012 % | +0.018 % | +0.099 % | +0.445 % | 1.04 % | +0.99 % | -0.90 % |
| 17.98 | interference | +0.004 % | -0.006 % | +0.035 % | +0.256 % | 0.43 % | +0.55 % | -0.51 % |
| 35.95 | interference | +0.011 % | -0.016 % | +0.021 % | +0.175 % | 0.13 % | +0.37 % | -0.34 % |
| 17.98 | load | +0.004 % | - | +0.035 % | +0.255 % | 0.43 % | +0.55 % | -0.50 % |
Frequency kernel (RECORD, span 4): approach -0.91 to -0.95 % at every grid (the image loads; C3 item 5 of the
vault's `studies_structure_review/r_catalogue.md`), load found +1.4 % under interference; p0 +0.04 % at
`a/h` 36 (does not converge below its image error).

## 9. Depends on
r02 (`407273d`), `slippy_bridge.contact` (`solve`, `Grid`, `hertz` to size the grid), `studies/outputs`
(`l100_dry_contact/plots/contact.py`). AxisForge `feature/newton-solver` `d34ec7f`, SlipPY `local-fixes`
`df5916c` (read only). Not covered: subsurface stresses (Johnson Table 4.1; SlipPY `SubsurfaceStress`
defects of the old case, v108 blocked by Q-03). Reviews: `80_Inbox/2026-10-28/r2_report_v_cases/_evidence/v101_*`.
