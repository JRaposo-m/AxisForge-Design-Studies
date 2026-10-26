# line_contact

Validation case: SlipPY steady EHL of a line contact, through `slippy_bridge.lubrication.ehl.run`,
against Venner & Bos (1994, Tables 1-2) and Canzi2010 (Table 2). V-EHL-03, V-EHL-04, V-EHL-06.

**Status: SCAFFOLD (2026-10-17).** Structure agreed in `debate_gears-lubrication-scaffold_2026-10-17`;
not validated, not run. Runs after `venner_bos_circular` (R-115).

## Question

PROVISIONAL. Does SlipPY, with the line contact periodic along the roller axis, reproduce the central
and minimum film of Venner & Bos at M = 20 and M = 200 (L = 10) and Canzi's calculated H_c at
alpha_bar = 20, M1 = 17.97? OPEN R-084, GL-16.

## Sources

PROVISIONAL. Rows in `reference_values.csv` (one merged vault file; header = line 1). OPEN R-108.

| Key | Item | PDF page | CSV lines |
|---|---|---|---|
| VennerBos1994 | Table 1: M = 20, levels 5-12 (n = 225 ... 28673) | 6 | L2-49 |
| VennerBos1994 | Parameters M = 20: W, G, U, p_h, domain, levels | 5-6 | L50-58 |
| VennerBos1994 | Table 2: M = 200, levels 5-13 (n = 225 ... 57345) | 8 | L79-132 |
| VennerBos1994 | Parameters M = 200 | 7-8 | L133-141 |
| VennerBos1994 | Roelands alpha, z; D-H Eq. (4) | 2 | L71-75, L151-155 |
| Canzi2010 | Table 2: H_c calculated and Moes formula, alpha_bar x M1, compressible | 3 | L159-176 |
| Canzi2010 | Eqs (1), (3), (4), (6)-(10): M1, alpha_bar, Moes line formula | 1-2 | L186-190 |

## Conventions

PROVISIONAL. OPEN R-084, R-091, GL-18.

- Speed: Venner's U is based on the MEAN speed; Moes M = W (2U)^(-1/2), L = G (2U)^(1/4) use the SUM
  speed; Canzi's u_Sigma is the sum speed. The bridge records the mean speed only.
- Film: Venner `H = h R / b^2` with b the Hertz half-width (bridge `s_x`); Canzi uses Moes H (Eq. 5).
- Line contact: periodic on array axis 1 in the step and the solver, the total load `w n_y h` (R-084).
- Domain in units of b: M = 20 on -4 <= X <= 1.5; M = 200 on -2.5 <= X <= 1.5; inlet on -X (OPEN R-073, R-048).
- J-V rows recorded only (no SlipPY model). Units SI; the CSV symbol column is not printed (GL-23).

## Reference

PROVISIONAL. Venner's finest levels (M = 20, level 12, L44-47; M = 200, level 13, L127-130), converged
to 3-4 digits over the last 3 levels (no error estimate printed). Canzi Table 2 "Hc calculated" 4.91
(L165); the "Moes formula" 4.77 (L166) judges `references.contact.ehl.moes` in the tests. OPEN R-093, GL-13.

## Case

PROVISIONAL. Steel cylinder on a steel flat (E 210 GPa, nu 0.3), Roelands z = 0.69, density incompressible or D-H. `R_X_M`, `U_MEAN_M_PER_S`, `ROELANDS_P0_PA`,
`RHO0_KG_PER_M3`, the SlipPY node count `N_X_RUN` and Canzi's domain are not decided. Assumptions:
smooth, steady, isothermal, Newtonian, elastic half-spaces, pure rolling, infinite cylinder. SlipPY D-H (5.9e8 Pa, LM:216) and Roelands (9.67, LM:103-106) are dimensional: the SI point must reproduce the printed p_h
(1.05 and 3.3 GPa) as well as M, L, z; the printed G/alpha is not the E' of steel (GL-36). OPEN GL-16,
R-115, GL-34, GL-36, GL-37.

## Criteria

PROVISIONAL. Fixed before running; all **[proposed]**. OPEN GL-16.

| ID | Quantity | Criterion | CSV lines | Script |
|---|---|---|---|---|
| N1 | M = 20, incompressible and D-H, H_c and H_m | H_c within 2 %, H_m within 3 % [proposed] | L44-47 | `01_film` |
| N2 | M = 200, incompressible and D-H, H_c and H_m | H_c within 2 %, H_m within 5 % [proposed] (sharp exit constriction) | L127-130 | `01_film` |
| N3 | Canzi Table 2, alpha_bar 20, M1 17.97, H_c calculated | within 3 % [proposed] | L165 | `01_film` |
| - | Canzi "Moes formula" 4.77 | judges `moes(line=True)` in the tests, not SlipPY | L166 | tests |

## Scripts

PROVISIONAL. `python 01_film.py [--smoke]` from this folder. `--smoke`: M = 20 incompressible only.
Today the script raises `NotImplementedError` at `moes_inverse`. No notebook yet (D-0022).

| Script | Content |
|---|---|
| `construction.py` | named SI constants with their CSV lines; nothing solved |
| `01_film` | Venner M = 20 and M = 200, incompressible + D-H; Canzi Table 2 point; N1-N3 |

## Provenance

PROVISIONAL. OPEN R-146.

| | |
|---|---|
| `reference_values.csv` | copied unchanged from vault `80_Inbox/2026-10-16/reference_values/ehl/line_contact/reference_values.csv`; sha256 (LF-normalised) `396f1c74180636c12fcbc58317fa998455d05a5ea49ad9b7836f76783f55a58c` |
| hash rule | sha256 of the bytes with CRLF replaced by LF (R-146: a checkout may turn the CSV to CRLF) |
| design-studies | set at first run |
| AxisForge | set at first run (not imported by this case) |
| SlipPY | set at first run |

## Known issues

PROVISIONAL. OPEN R-084, R-108, GL-16.

- R-084: a line contact must be periodic on axis 1 in three places; `ehl.run` refuses a contradicting kwarg.
- R-108, GL-16: SlipPY's line contact is a finite periodic strip, not plane strain; the phase-1 dry
  half-width was -1.6 % from it, and the EHL deflection inherits the same kernel.
- Table 1 level 6 D-H H_m (L10, 7.928e-02) is an apparent misprint; not used.
- Venner's text ratios "minimum to central" (L61-63, L144-146) are H_c/H_m; recorded only.

## Result

Not run: scaffold.
