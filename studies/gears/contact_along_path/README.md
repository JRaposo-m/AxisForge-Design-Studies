# contact_along_path

Study (demonstration): the dry Hertz contact and the EHL film of the FZG C40 spur pair at the named points
of its path of contact, AxisForge `contact_inputs` -> `slippy_bridge` gear adapter -> Hertz and
`ehl.run`. V-GEAR-05.

**Status: SCAFFOLD (2026-10-17).** Structure agreed in `debate_gears-lubrication-scaffold_2026-10-17`;
not validated, not run. A study, not a validation: no film along a gear path is published (R-145).

## Question

PROVISIONAL. How do the Hertz pressure, the half-width, the mean entrainment speed, the slide-to-roll ratio
and the central film change from A to E along the path of a C40 pair at one FZG load stage and 80 C, and
does each point's SlipPY film agree with the Moes line formula at the same M, L? OPEN R-145, GL-39.

## Sources

PROVISIONAL. Rows in `reference_values.csv` (oil data only; header = line 1). Geometry and loads are those of
`validation/gears/fzg_geometry_ph`, repeated in `construction.py`. OPEN R-295.

| Key | Item | PDF page | CSV lines |
|---|---|---|---|
| Martins2015 | Table 1: five oils, rho15, nu40/80/100, alpha at 80 C, VI, LP | 2 | L2-41 |
| Fernandes2015a | Table 1: four oils, rho15, alpha_t, nu40/70/100, D341 m, n, beta, VI | 2 | L42-89 |
| Fernandes2015a | Tables 3-4: LP, alpha_FTM and alpha_Gold at 50/80/100 C | 4-5 | L90-125 |
| Fernandes2015a | Tables 2, 5 and text: ball-on-disc test data (not used for the gears) | 3-6 | L126-173 |
| Fernandes2015a | Eqs (1)-(10): SRR, Gold alpha, film formula Eqs (3)-(6), thermal correction | 3-7 | L174-183 |

## Conventions

PROVISIONAL. OPEN GL-14, R-091.

- Speed: the bridge records the MEAN entrainment speed `u_e = (u1 + u2)/2`; Moes groups use the sum,
  converted inside `references.contact.ehl` only. FEUP Eq. (4) uses the SUM `U_S` (L177) in a
  Hamrock-Dowson-form formula: read with the mean or the sum, h changes by `2^0.67 = 1.59`. It is shown as
  that bracket, never pass/fail (GL-14).
- v_g = u1 - u2 [m/s], signed, referenced to gear 1 (AxisForge `surface_speeds`); zero at C.
- Quasi-steady: each point is a steady line contact; sliding is ignored (isothermal, Newtonian).
- Spur only (R-027). Oil data converted to SI once in `construction.py`; alpha printed as "x 10^-8", read
  as 1/Pa (R-295). Units: AxisForge for the pair, SI for the oils and the contacts. The CSV symbol column is
  not printed (GL-23).

## Reference

PROVISIONAL. None published (demonstration). The check is `references.contact.ehl.moes(line=True)` at the
point's own M, L. OPEN R-093, R-145.

## Case

PROVISIONAL. FZG C40 (z 16/24, mn 4.5 mm, FEUP x), gear 1 driving (Q-09). Operating point: wheel torque and
speed not chosen (OPEN GL-39); the smoke run borrows K5 (104.97 N m) at 200 rpm from
`fzg_geometry_ph`. Oil at 80 C: the two FEUP sources are kept side by side (`OIL_MARTINS2015`,
`OIL_FERNANDES2015A`). eta0, the Roelands z and rho(80 C) are not printed (`None`); which oil and which alpha
is OPEN R-295. Steel E, nu (R-118 open): E = 206 000 N/mm^2, nu = 0.3, the generic steel of ISO 6336-2:2006
Table 1, assumed, not 20MnCr5-specific (`construction.GEAR_E_MPA`, `GEAR_NU`). Assumptions: rigid load sharing, local equivalent cylinders,
smooth surfaces, steady, isothermal, Newtonian, pure rolling for the film. OPEN R-295, R-118, GL-39.

## Criteria

PROVISIONAL. Fixed before running; **[proposed]**; a demonstration check, not a validation. OPEN R-145.

| ID | Quantity | Criterion | Script |
|---|---|---|---|
| - | p_H, b, R_x, u_e, v_g, share at A, B-, B+, C, D-, D+, E | record (no criterion) | `01_hertz_path` |
| S1 | h_c per point | within +-10 % of `moes(line=True)` at the same M, L [proposed] | `02_film_path` |
| - | FEUP Eqs (3)-(4) | mean/sum bracket x 1.59, record only, never pass/fail (GL-14) | `02_film_path` |

## Scripts

PROVISIONAL. Run from this folder: `python 01_hertz_path.py [--smoke]`, `python 02_film_path.py [--smoke]`.
Without `--smoke` both raise `NotImplementedError` until the operating point is chosen. With `--smoke`,
point A only, at the smoke operating point; `01_hertz_path` runs, `02_film_path` raises `NotImplementedError`
in `ehl.run`. No notebook yet (D-0022).

| Script | Content |
|---|---|
| `construction.py` | this study's own `build_pair` and `path_times` (copied from `fzg_geometry_ph`, studies import no case); the oils in SI; nothing solved |
| `01_hertz_path` | contacts and Hertz along the path |
| `02_film_path` | one steady `ehl.run` per point; S1 and the FEUP bracket |

## Provenance

PROVISIONAL. OPEN GL-01, R-146.

| | |
|---|---|
| `reference_values.csv` | copied unchanged from vault `80_Inbox/2026-10-16/reference_values/gears/fzg_film_path/reference_values.csv`; sha256 (LF-normalised) `c69645443a3bf1704b84f2785e94f070a7cbff8ffa444ba8320cc9dd597f9ebe` |
| hash rule | sha256 of the bytes with CRLF replaced by LF (R-146: a checkout may turn the CSV to CRLF) |
| design-studies | set at first run |
| AxisForge | set at first run; checked at `73231b4` of branch `refactor/gears-mechanical-system` (the API used, `contact_inputs`, `path_points`, `mesh_forces`, is not on main, GL-01) |
| SlipPY | set at first run |

## Known issues

PROVISIONAL. OPEN R-295, R-091, R-024, R-027, GL-14, GL-35, GL-39.

- R-295: FEUP oil data conflict (VI ESTR 162 vs 140, PAOR 149 vs 150); Martins2015 alpha at 80 C matches
  neither alpha_FTM nor alpha_Gold; no unit for alpha.
- R-091: which alpha (initial slope or Blok's alpha*) feeds the Moes L.
- R-024: the sign of rho2 for a ring gear is not confirmed; C40 is external (rho2 > 0).
- R-027: spur only; a helical pair is a mid-face slice and EHL on it is refused.
- GL-14: FEUP Eq. (4) with the sum speed; C_0 of Eq. (3) and E* are not printed.
- GL-39: the operating point (FZG load stage, wheel speed) is not chosen; `None` until José picks it.
- GL-35: no EHL check yet with two finite radii or a concave r2 (every `validation/ehl` case has a flat body 2).

## Result

Not run: scaffold.
