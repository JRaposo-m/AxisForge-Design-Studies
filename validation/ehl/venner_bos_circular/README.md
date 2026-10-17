# venner_bos_circular

Validation case: SlipPY steady EHL, through `slippy_bridge.lubrication.ehl.run`, against the circular
contacts of Venner & Bos (1994). V-EHL-01, V-EHL-02, V-EHL-13.

**Status: SCAFFOLD (2026-10-17).** Structure agreed in `debate_gears-lubrication-scaffold_2026-10-17`;
not validated, not run. This case gates every other EHL result (R-115).

## Question

PROVISIONAL. Does SlipPY's `IterSemiSystem` + `UnifiedReynoldsSolver` reproduce the central and
minimum film of a smooth, steady, isothermal, Newtonian circular contact at M = 50 and M = 1000
(L = 10), incompressible and Dowson-Higginson? OPEN R-115.

## Sources

PROVISIONAL. Rows in `reference_values.csv` (line numbers: header = line 1). OPEN GL-26.

| Key | Item | PDF page | CSV lines |
|---|---|---|---|
| VennerBos1994 | Table 3: M = 50, levels 3-7 (33x33 ... 513x513) | 10 | L2-31 |
| VennerBos1994 | Table 4: M = 1000, levels 4-7 (65x65 ... 513x513) | 12 | L51-74 |
| VennerBos1994 | Parameters M, L, W, G, U, p_h, domain (M = 50 / 1000) | 9, 11 | L32-39, L75-83 |
| VennerBos1994 | Roelands alpha, z; Dowson-Higginson Eq. (4) | 2 | L43-47 |
| VennerBos1994 | V-EHL-13: H_c/H_m per law, Eq. (24) predictions | 9, 11 | L95-112 |

## Conventions

PROVISIONAL. OPEN R-091, GL-18.

- Speed: VennerBos `U = eta0 u_s/(2 E' R_x)` is based on the MEAN speed, the bridge's
  `u_mean_m_per_s`; Moes M and L use the SUM speed, `M = W (2U)^(-3/4)`, `L = G (2U)^(1/4)` (pp. 14-15).
- Film: `H = h R_x / a^2` with `a` the Hertz radius, the bridge's `s_x` (`references.contact.ehl.film_scalings`, `H_venner`).
- `alpha` 1.7e-8 is printed without unit (1/Pa assumed) and is the Roelands initial slope that
  `moes_inverse` derives from (eta0, z, p0) (R-091). p0 is not printed (GL-18).
- Domain in units of `a`: M = 50 on -4.5 <= X <= 1.5, -3 <= Y <= 3; M = 1000 on -2.5 <= X <= 1.5,
  -2 <= Y <= 2; inlet on the -X side (OPEN R-073), an `(inlet, outlet)` span (OPEN R-048).
- J-V density rows are recorded only: SlipPY has no Jacobson-Vinet model.
- Units SI throughout; the CSV symbol column is not printed (non-ASCII, GL-23).

## Reference

PROVISIONAL. The 513x513 rows of Tables 3 and 4 (level 7, the source's converged values; numerical error
< 0.1 % for M = 50, about 1 % for M = 1000). The dimensionless point (M, L) is turned into SI inputs by
`references.contact.ehl.moes_inverse` and the film back by `film_scalings`. OPEN R-091, R-093.

## Case

PROVISIONAL. Steel ball on a steel flat (E 210 GPa, nu 0.3), Roelands z = 0.69, density incompressible or D-H. `R_X_M`, `U_MEAN_M_PER_S`, `ROELANDS_P0_PA` and
`RHO0_KG_PER_M3` are not decided (`None` in `construction.py`). SlipPY D-H (5.9e8 Pa, LM:216) and Roelands (9.67, LM:103-106) are dimensional: the SI point must reproduce the printed p_h (0.79 and 2.14 GPa) as well as
M, L, z. The printed G/alpha = 2.78e11 Pa is not the E' of steel (2.31e11 Pa), so the G, U, p_h
cross-check cannot all pass (GL-36). Assumptions: smooth, steady, isothermal, Newtonian, elastic
half-spaces, pure rolling. OPEN GL-34, GL-36, GL-18, R-089, R-090.

## Criteria

PROVISIONAL. Fixed before running; all **[proposed]**. OPEN R-115.

| ID | Quantity | Criterion | CSV lines | Script |
|---|---|---|---|---|
| E1 | M = 50, incompressible and D-H, H_c and H_m at 513x513 | H_c within 2 %, H_m within 3 % [proposed] | L26-29 | `01_film` |
| E2 | M = 1000, incompressible and D-H, H_c and H_m at 513x513 | H_c within 2 %, H_m within 5 % [proposed] | L69-72 | `01_film` |
| E3 | D-H / incompressible H_c and H_m ratio (V-EHL-13) | record only | L26-29, L69-72; L97-102 | `01_film` |
| - | J-V rows | record only (no SlipPY model) | L30-31, L73-74 | - |

## Scripts

PROVISIONAL. `python 01_film.py [--smoke]` from this folder (imports `construction` as a sibling).
`--smoke`: M = 50 incompressible on the 65x65 level only. Today the script raises
`NotImplementedError` at the first scaffold body (`moes_inverse`). No notebook yet (D-0022).

| Script | Content |
|---|---|
| `construction.py` | named SI constants with their CSV lines; nothing solved |
| `01_film` | M = 50 and M = 1000, incompressible + D-H; E1-E3 |

## Provenance

PROVISIONAL. OPEN R-146.

| | |
|---|---|
| `reference_values.csv` | copied unchanged from vault `80_Inbox/2026-10-16/reference_values/ehl/venner_bos_circular/reference_values.csv`; sha256 (LF-normalised) `94654e62e0c64dc8bfe45090d413c891272f1c1f01b009c1a371e41d499a53ce` |
| hash rule | sha256 of the bytes with CRLF replaced by LF (R-146: a checkout may turn the CSV to CRLF) |
| design-studies | set at first run |
| AxisForge | set at first run (not imported by this case) |
| SlipPY | set at first run |

## Known issues

PROVISIONAL. OPEN R-115, R-048, R-091, GL-18, GL-26, GL-34, GL-36, GL-37.

- R-115: SlipPY EHL is only feasibility-tested; its default 65x65 over 4a is too short an inlet.
- R-048: the asymmetric domain needs `Grid.around(span_x=(inlet, outlet))`, a scaffold today.
- R-091, GL-18: which alpha, and the unprinted Roelands p0.
- GL-34, GL-36: the SI point per case is undecided; G/alpha is not the E' of steel.
- GL-37: `Grid.around` rounds h down to a power of two, so a run does not have 513 or 65 nodes per side; the script records `n_x`, `n_y`.
- GL-26: the printed ratio H_c/H_m = 1.86 (L97) is not reproduced by Table 3 (3.004/1.645 = 1.83); record only.
- Table 3 level 3 incompressible H_c (L3) is an apparent misprint (2.762e-02); not used.

## Result

Not run: scaffold.
