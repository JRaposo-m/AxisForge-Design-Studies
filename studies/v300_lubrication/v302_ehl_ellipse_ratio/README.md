# ellipse_ratio

Validation case: SlipPY steady EHL of elliptical contacts, through `slippy_bridge.lubrication.ehl.run`,
against the film versus ellipticity D of Canzi2010 (Tables 4-6). V-EHL-08; the Hamrock-Dowson (1977) part,
V-EHL-10, is blocked.

**Status: SCAFFOLD (2026-10-17).** Structure agreed in `debate_gears-lubrication-scaffold_2026-10-17`;
not validated, not run.

## Question

PROVISIONAL. With R_x along rolling and R_y = R_x / D across it, does SlipPY follow Canzi's calculated
H_c and H_m as D goes from 1 to 0.01 at fixed alpha_bar and L? OPEN R-073.

## Sources

PROVISIONAL. Rows in `reference_values.csv` (one merged vault file; header = line 1). OPEN R-092.

| Key | Item | PDF page | CSV lines |
|---|---|---|---|
| Canzi2010 | D sweep 1 ... 0.01; M2c and Eq. (23) | 4-5 | L2-8 |
| Canzi2010 | Table 4: alpha_bar 20, M2c 500: H_c, H_m, H_c/H_m | 5 | L9-28 |
| Canzi2010 | Table 5: alpha_bar 10, M2c 100: H_c, H_m, H_c/H_m | 5 | L29-48 |
| Canzi2010 | Table 6: H_Ccalc and CELINE (Eq. 24), five (alpha_bar, M2c) blocks | 6 | L49-113 |
| Canzi2010 | Eqs (2)-(24): H, L, M2, alpha_bar, E_c, kappa, equivalent line, CELINE | 2-7 | L114-137 |
| HamrockDowson1977 | Tables 5-6: 34 cases, H_min and H_c (V-EHL-10) | 10-11 | L138-636 (no script) |

## Conventions

PROVISIONAL. OPEN R-073, GL-13.

- Speed: Canzi's u_Sigma is the SUM speed (Moes groups); the bridge records the mean speed only.
- Film: Moes H (Eq. 5) (`film_scalings`, `H_moes`).
- D = R_x / R_y (Canzi's convention, the inverse of the classical ellipticity). R_x is along the flow,
  array axis 0; for D < 1 the contact ellipse is longer across rolling (R-073: orientation).
- M2 per D from M2c by Eq. (23) with kappa and E_c; E_c's printed upper limit 2 pi is probably a
  misprint (pi/2), to verify before use (GL-13).
- Units SI throughout; the CSV symbol column is not printed (non-ASCII, GL-23).

## Reference

PROVISIONAL. Tables 4-5 calculated H_c, H_m and Table 6 H_Ccalc (equal to Tables 4-5 where they
overlap). Canzi claims < 1 % numerical error (p. 3); CELINE is a formula prediction (< 10 % from
calculated, p. 6), recorded only. OPEN R-114, GL-13.

## Case

PROVISIONAL. Smooth steel ellipsoid `(R_x, R_x / D)` on a steel flat (E 210 GPa, nu 0.3),
compressible (D-H). SlipPY D-H (5.9e8 Pa, LM:216) and Roelands (9.67, LM:103-106) are dimensional: the SI point must reproduce the printed p_h (alpha_bar = alpha p_h) as well as the Moes point (GL-34). The rheology, the domain and the SI point are not printed and not decided (`None`
in `construction.py`, R-114). The alpha_bar = 0 block of Table 6 is isoviscous and is recorded only:
`ehl.run` refuses z = 0 and alpha < 1e-9 (GL-17). Assumptions: smooth, steady, isothermal, Newtonian,
elastic half-spaces, pure rolling. OPEN R-114, GL-17.

## Criteria

PROVISIONAL. Fixed before running; all **[proposed]**. OPEN R-114.

| ID | Quantity | Criterion | CSV lines | Script |
|---|---|---|---|---|
| R1 | H_c vs H_Ccalc, 4 blocks x 6 D | within 3 % [proposed] | L50-55, L63-68, L76-81, L89-94 | `01_canzi_tables` |
| R2 | H_m vs Tables 4-5, 2 blocks x 6 D | within 5 % [proposed] (side lobes for D >= 0.2) | L17-22, L37-42 | `01_canzi_tables` |
| R3 | H_c/H_m (Tables 4-5) and CELINE (Table 6) | record only | L23-28, L43-48; L56-61, L69-74, L82-87, L95-100 | `01_canzi_tables` |
| - | Table 6, alpha_bar = 0 (isoviscous) | record only (GL-17) | L102-113 | - |
| - | Hamrock-Dowson 1977 (V-EHL-10) | blocked: no script until R-092 | L138-636 | - |

## Scripts

PROVISIONAL. `python 01_canzi_tables.py [--smoke]` from this folder. `--smoke`: Table 5 at D = 1 only.
Today the script raises `NotImplementedError` at `canzi_moes`. No Hamrock-Dowson script (R-092). No
notebook yet (D-0022).

| Script | Content |
|---|---|
| `construction.py` | named SI constants and the CSV line maps; nothing solved |
| `01_canzi_tables` | Tables 4-6, four (alpha_bar, M2c) blocks x six D; R1-R3 |

## Provenance

PROVISIONAL. OPEN R-146.

| | |
|---|---|
| `reference_values.csv` | copied unchanged from vault `80_Inbox/2026-10-16/reference_values/ehl/ellipse_ratio/reference_values.csv`; sha256 (LF-normalised) `0d920c24747c25e8911fd96351fcbf499de340cae1b032ad150b3ab3c2fee1a6` |
| hash rule | sha256 of the bytes with CRLF replaced by LF (R-146: a checkout may turn the CSV to CRLF) |
| design-studies | set at first run |
| AxisForge | set at first run (not imported by this case) |
| SlipPY | set at first run |

## Known issues

PROVISIONAL. OPEN R-092, R-073, GL-13, GL-17.

- R-092: Hamrock-Dowson's U may be mean or sum speed (defined in H-D 1976 Part I, not held); no script.
  Those rows also need the Blok alpha* (G = E'/p_iv,as), not the initial slope of Roelands.
- R-073: the ellipse orientation against the entrainment direction is to be tested.
- GL-13: the (alpha_bar, M2c, D) -> (M, L) map needs Eqs (19), (23) with kappa and E_c.
- GL-17: the isoviscous block (alpha_bar = 0) cannot pass the viscosity guards of `ehl.run`.
- Text p. 5 says "Tables 3 and 4" for Tables 4 and 5 (a misprint of the source).

## Result

Not run: scaffold.
