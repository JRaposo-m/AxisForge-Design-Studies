# canzi2010

Validation case: SlipPY steady EHL, through `slippy_bridge.lubrication.ehl.run`, against the multigrid
films published in Canzi2010: the grid study of an elliptical contact (Table 1) and the circular
contacts (Table 3). V-EHL-05, V-EHL-07.

**Status: SCAFFOLD (2026-10-17).** Structure agreed in `debate_gears-lubrication-scaffold_2026-10-17`;
not validated, not run.

## Question

PROVISIONAL. How large is SlipPY's grid error on an elliptical contact (D = 0.2) against Canzi's
converged 2048x2048 film, and does SlipPY reproduce Canzi's calculated circular H_c over
alpha_bar in {20, 10, 5} x M2 in {100, 500}? OPEN R-114.

## Sources

PROVISIONAL. Two vault folders, two CSVs (GL-22); line numbers: header = line 1. OPEN GL-22.

| Key | Item | PDF page | File, CSV lines |
|---|---|---|---|
| Canzi2010 | Table 1: H_c, H_m vs grid 128^2 ... 2048^2; alpha_bar 10, M2 2027, D 0.2, compressible | 3 | `reference_values.csv` L2-21 |
| Canzi2010 | Eqs (5), (2), (12), (14), (11): H, L, M2, circular alpha_bar, compressibility | 2 | `reference_values.csv` L22-35 |
| Canzi2010 | Table 3: H_c calculated and Moes formula; alpha_bar x M2, D = 1, compressible | 4 | `reference_values_circular.csv` L2-20 |
| Canzi2010 | Eqs (15)-(18): Moes circular formula | 2 | `reference_values_circular.csv` L32-33 |

## Conventions

PROVISIONAL. OPEN GL-12, GL-13.

- Speed: Canzi's `u_Sigma = u1 + u2` is the SUM speed; Moes H, L, M1, M2 use it. The bridge's mean
  speed is doubled inside `references.contact.ehl` only.
- Film: Moes `H = (h/R_x) (eta0 u_Sigma/(E' R_x))^(-1/2)` (Eq. 5) (`film_scalings`, `H_moes`).
- (alpha_bar, M2) -> (M, L) by `canzi_moes` (Eq. 14 circular); the elliptical Table 1 point needs
  Eq. (19) with D and kappa (OPEN GL-13).
- D = R_x / R_y (Canzi's convention, the inverse of the classical ellipticity); R_x along rolling, axis 0.
- Rheology: Roelands + Dowson-Higginson; z, alpha, p0 and the D-H constants are NOT printed (R-114).
- Units SI throughout; the CSV symbol column is not printed (non-ASCII, GL-23).

## Reference

PROVISIONAL. Table 1 at 2048x2048: H_c 1.886, H_m 1.556 (L20-21); Table 3 "Hc calculated" (L9, 11, 13,
15, 17, 19). Canzi claims < 1 % numerical error (p. 3). The "Moes formula" column (L10, 12, ...) is a
formula prediction, recorded only (V-EHL-11). OPEN R-114, GL-12.

## Case

PROVISIONAL. Smooth steel body on a steel flat (E 210 GPa, nu 0.3), compressible (D-H). SlipPY D-H (5.9e8 Pa, LM:216) and Roelands (9.67, LM:103-106) are dimensional: the SI point must reproduce the printed p_h
(alpha_bar = alpha p_h) as well as the Moes point (GL-34).
The assumed z, p0, the D-H constants and the domain are written in `construction.py` BEFORE the run
(R-114); they are `None` today. Assumptions: smooth, steady, isothermal, Newtonian, elastic
half-spaces, pure rolling. OPEN R-114.

## Criteria

PROVISIONAL. Fixed before running; all **[proposed]**. OPEN R-114.

| ID | Quantity | Criterion | File, CSV lines | Script |
|---|---|---|---|---|
| C1 | H_c, H_m at SlipPY's finest grid vs Table 1 at 2048^2 | within 3 % [proposed] (1 % reference + SlipPY grid error) | `reference_values.csv` L20-21 | `01_grid` |
| C2 | SlipPY grid sweep vs Table 1 grids | record only (sets the grid error budget) | `reference_values.csv` L12-21 | `01_grid` |
| C3 | circular H_c calculated, 6 points | within 3 % [proposed] | `reference_values_circular.csv` L9, 11, 13, 15, 17, 19 | `02_circular` |
| - | "Moes formula" column | record only | `reference_values_circular.csv` L10, 12, 14, 16, 18, 20 | `02_circular` |

## Scripts

PROVISIONAL. Run from this folder: `python 01_grid.py [--smoke]`, `python 02_circular.py [--smoke]`.
`--smoke`: the coarsest grid / the first point only. Today both raise `NotImplementedError` at the
first scaffold body (`canzi_moes`). No notebook yet (D-0022).

| Script | Content |
|---|---|
| `construction.py` | named SI constants with their CSV lines; nothing solved |
| `01_grid` | Table 1 point, SlipPY grid sweep; C1, C2 |
| `02_circular` | Table 3, six circular points; C3 |

## Provenance

PROVISIONAL. Two byte-identical copies of two vault files, each with its own hash (GL-22). OPEN R-146.

| | |
|---|---|
| `reference_values.csv` | copied unchanged from vault `80_Inbox/2026-10-16/reference_values/ehl/canzi_grid/reference_values.csv`; sha256 (LF-normalised) `a8ee771fcc71a47ad66cd47e1b38c7545ebb7cc030ae9e719b30525e141c6ff9` |
| `reference_values_circular.csv` | copied unchanged from vault `80_Inbox/2026-10-16/reference_values/ehl/canzi_circular_extra/reference_values.csv`; sha256 (LF-normalised) `30130176aa0f701ae0c07eb020cb742bb0fb54f54277b36bd63a30e831b69f56` |
| hash rule | sha256 of the bytes with CRLF replaced by LF (R-146: a checkout may turn the CSV to CRLF) |
| design-studies | set at first run |
| AxisForge | set at first run (not imported by this case) |
| SlipPY | set at first run |

## Known issues

PROVISIONAL. OPEN R-114, GL-12, GL-13, GL-22.

- R-114: z, p0, D-H constants and the domain are not printed; the assumed values go in `construction.py` before the run.
- GL-12: the outer exponent of Eq. (15) as recorded (t/s) does not reproduce Table 3; only 1/s does. The formula column waits.
- GL-13: the Moes columns need a dimensional p_h (Eq. 11) and the alpha_bar -> L maps (Eq. 3, 14, 19).
- Eq. (21) / E_c: the printed 2 pi upper limit of E_c is probably a misprint (pi/2); verify before use.
- GL-22: two vault folders merged here; two CSVs.
- GL-37: `Grid.around` rounds h down to a power of two, so the sweep does not have 128 ... 2048 nodes per side; the script records `n_x`, `n_y`.

## Result

Not run: scaffold.
