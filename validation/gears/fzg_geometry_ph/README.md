# fzg_geometry_ph

Validation case: AxisForge cylindrical gear geometry of the FEUP FZG test gears (C40 spur, H501 helical)
against the printed tables, and the Hertz pressure of C40 at the load stages K5, K7, K9 (AxisForge
`contact_inputs` -> `slippy_bridge` gear adapter -> Hertz) against Martins2015 Table 4. V-GEAR-01 ..
V-GEAR-04.

**Status: SCAFFOLD (2026-10-17).** Structure agreed in `debate_gears-lubrication-scaffold_2026-10-17`;
not validated, not run.

## Question

PROVISIONAL. Does AxisForge reproduce the printed tip diameters, centre distance and contact ratios of
C40 and H501, and does the AxisForge -> bridge -> Hertz chain scale p_H between load stages as the
printed table does? OPEN R-117, Q-09.

## Sources

PROVISIONAL. Rows in `reference_values.csv` (header = line 1). OPEN R-120, R-121.

| Key | Item | PDF page | CSV lines |
|---|---|---|---|
| Fernandes2015b | Table 1: geometry C40, H501 (z, m, a, alpha, beta, b, x, d_a, eps, Ra) | 4 | L2-29 (C40 L2-15) |
| Martins2015 | Table 3: geometry C40, H501, H951 incl. eps_gamma, material 20MnCr5 | 4 | L100-147 (C40 L100-115) |
| Martins2015 | text 2.3.1: Ra after run-in | 3 | L148-150 |
| Fernandes2015b | text 3: lever arm, K5/K7/K9 "applied on wheel" | 3 | L151-157 |
| Martins2015 | Table 4: wheel torque, input power, F_r, F_a, p_H per stage; wheel speeds | 4 | L158-200 |

## Conventions

PROVISIONAL. OPEN R-120, R-124, Q-09.

- Torque and speed are printed on the WHEEL; the case uses `T1 = T2 z1/z2`, `n1 = n2 z2/z1` (R-124).
- The driver is gear 1 (the pinion), assumed: FEUP does not say (Q-09 open).
- alpha and m are printed without saying normal or transverse; read as normal (R-120).
- E and nu are not printed (20MnCr5 only, R-118).
- The p_H convention (point, load sharing) is not stated. The criterion is therefore the load-ratio
  test; p_H at A, B+ and C is a record.
- The p_H computed here is the Hertz p0 of the local equivalent cylinders at each point, under AxisForge's
  equal-line-load sharing (w = F_bn / l_sum); it is not the ISO 6336-2 sigma_H (no Z_eps, Z_beta, and its
  nominal load differs). Martins2015 prints one p_H per stage, compared here with every point; which point
  it refers to is open (R-117, Q-09).
- AxisForge units (mm, N, N m, MPa, rpm) in the case. SI only inside `slippy_bridge`. The CSV symbol
  column is not printed (GL-23).

## Reference

PROVISIONAL. Geometry: Martins2015 Table 3 for the C40 wheel d_a (118.54, L110); the 115.54 of
Fernandes2015b (L12) is very likely a typo and is reported beside it (R-121). The tables are otherwise
identical. Load: Martins2015 Table 4 wheel torques (L168, L178, L188) and p_H (L174, L184, L194).
OPEN R-121, R-117.

## Case

PROVISIONAL. `construction.build_pair("C40" | "H501")`: the FEUP z, mn, alpha_n, beta, b and x (H501 x from
L23-24, never the GEARpie entry, GL-11), zero-backlash centre distance from x. H501 hand of helix is not
printed (OPEN GL-38). `construction.path_times` gives the times of A, B-, B+, C, D-, D+, E with one-sided
limits 1e-6 mm either side of B and D (GL-09). It refuses eps_alpha >= 2 (GL-25). The gears carry
`material_id` `steel_20MnCr5`, not registered in AxisForge; E and nu are given (R-118 below). Assumptions: rigid unmodified involute flanks, rigid load
sharing, local equivalent cylinders, smooth frictionless elastic half-spaces. OPEN R-118, GL-09, GL-11,
GL-25.

## Criteria

PROVISIONAL. Fixed before running; all **[proposed]**. OPEN R-117, R-118, GL-10, GL-30, GL-41.

| ID | Quantity | Criterion | CSV lines | Script |
|---|---|---|---|---|
| G1 | C40 d_a pinion, d_a wheel | within 0.01 mm [proposed]; wheel against L110, L12 reported (R-121) | L11, L109; L110; L12 | `01_geometry` |
| G2 | C40 working centre distance | within 0.01 mm of 91.5 [proposed] | L5 | `01_geometry` |
| G3 | eps_alpha, eps_beta, eps_gamma (C40, H501); H501 d_a | record only: AxisForge 1.4624 vs 1.44 is a known gap (GL-10) | L13-14, L111-113; L25-28, L129 | `01_geometry` |
| P0 | C40 radial force F_r per stage | within 0.5 N of the printed integer [proposed]; checks R-124 (wheel -> pinion), d_w, alpha_wt (OPEN GL-41) | L172, L182, L192 | `02_ph` |
| P1 | p_H(K7)/p_H(K5), p_H(K9)/p_H(K5), C40, every point | within 0.2 % of 1.3755 and 1.7550 [proposed]. Jose's V-GEAR-04 criterion: it proves the stage torques and p ~ sqrt(w) only, blind to the geometry, the point, E and the driver; 0.2 % is set by the rounding of the printed integers (0.5/498 + 0.5/685 = 0.17 %) | L174, L184, L194 | `02_ph` |
| P2 | p_H at A, B+, C per stage | record only. "A, half load, pinion driving" ~ 498 MPa is the R-117 hypothesis; its tolerance depends on E (R-118) | L174, L184, L194 | `02_ph` |

## Scripts

PROVISIONAL. Run from this folder: `python 01_geometry.py [--smoke]`, `python 02_ph.py [--smoke]`.
`01_geometry` uses AxisForge only; `--smoke` changes nothing. `02_ph` runs through the gear adapter;
`--smoke`: point A only. No notebook yet (D-0022).

| Script | Content |
|---|---|
| `construction.py` | constants with their CSV lines; `build_pair`, `path_times`; nothing solved |
| `01_geometry` | AxisForge geometry of C40 and H501 against Tables 1 and 3; G1-G3 |
| `02_ph` | C40, K5/K7/K9: F_r and F_bn, contact inputs, bridge contacts, Hertz p_H; P0, P1, P2 |

## Provenance

PROVISIONAL. OPEN GL-01, R-146.

| | |
|---|---|
| `reference_values.csv` | copied unchanged from vault `80_Inbox/2026-10-16/reference_values/gears/fzg_geometry_ph/reference_values.csv`; sha256 (LF-normalised) `84d30d93e6b519017e8743015d0c6cef6c3c8605b8d5eab19bf48e8b0f1b83f1` |
| hash rule | sha256 of the bytes with CRLF replaced by LF (R-146: a checkout may turn the CSV to CRLF) |
| design-studies | set at first run |
| AxisForge | set at first run; checked at `73231b4` of branch `refactor/gears-mechanical-system` (the API used, `contact_inputs`, `path_points`, `mesh_forces`, is not on main, GL-01) |
| SlipPY | set at first run |

## Known issues

PROVISIONAL. OPEN R-117, R-118, R-120, R-121, R-123, R-124, R-125, R-126, GL-01, GL-09, GL-10, GL-11, GL-24, GL-25,
GL-30, GL-38.

- R-117, GL-30: the FEUP p_H point is not stated; at half load A gives about 497 MPa, B+ about 504.5.
- R-118: no E, nu for 20MnCr5 from a held source. Assumed for both gears: E = 206 000 N/mm^2, nu = 0.3, the
  generic steel of ISO 6336-2:2006 Table 1, not a 20MnCr5-specific value (`construction.GEAR_E_MPA`,
  `GEAR_NU`); the record shows `material1 = material2 = "given:steel_20MnCr5"`. p_H depends on E; the
  ratios of P1 do not.
- R-120: normal or transverse alpha, m not stated. R-121: C40 wheel d_a 115.54 vs 118.54.
- R-123: C40 Ra 0.7 (L15), 0.9 (L115) or 0.35/0.85 after run-in (L148-149); not used here.
- R-124, R-126, Q-09: wheel torque and speed, the driver assumed gear 1. R-125: F_bn is passed by hand
  from `mesh_forces`.
- GL-10: eps_alpha AxisForge 1.4624 vs printed 1.44 (C40); H501 1.474 vs 1.45, d_a 80.436/116.627 vs 80.37/116.57.
- GL-24: the path speed v_n is not exposed by AxisForge; `path_times` duplicates `contact.py:188`.
- GL-01: the gear API is only on `refactor/gears-mechanical-system`.
- GL-38: the H501 hand of helix is not printed; gear 1 +1, gear 2 -1 assumed (only the sign of F_a depends on it).

## Result

Not run: scaffold.
