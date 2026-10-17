# style

`axisforge.mplstyle` is the house paper style of the repository (accepted 2026-10-17): serif text with
STIX maths, closed box, inward ticks on four sides with minor ticks, no grid, black first in the colour
cycle, PDF at 300 dpi for LaTeX. It is the only place where figure appearance is set.

## Use
- Studies and examples: `plt.style.use("<repo>/style/axisforge.mplstyle")`, once, in the entry script.
- Validation cases: `validation._support.style.use_style()` at the top of the script.

Figure functions take data and return a `matplotlib.figure.Figure`; they set no colours, fonts or sizes.

## Semantics
| what | drawn as |
|---|---|
| numerical result (SlipPY, AxisForge FEM) | filled markers, no connecting line unless asked |
| formula / theory / reference | lines: own formula solid, others dashed or dotted |
| published or experimental value | open markers (`mfc="none"`) |

- Colour never carries identity alone: every series also has its own marker or line style.
- Direct labels next to the curves before a legend; a legend, if any, is frameless and off the data.
- No titles (captions live in LaTeX); per-point information goes to one `fig.text` line.
- Panels tagged `(a)`, `(b)` at the top left inside the axes.

## Labels
- Dimensional: symbol, then the unit in parentheses: `$h$ ($\mu$m)`, `$p$ (MPa)`, `$x$ (mm)`.
- Dimensionless: bare ratios: `$x/s_x$`, `$p/p_0$`, `$z/s_\mathrm{min}$`.
- Never square brackets, never the ISO solidus form `$h$ / mm`.

## Symbols (contact)
`s_x`, `s_y`: semi-axes of the contact along rolling (`x`) and across it (`y`), as in `slippy_bridge`;
depth is `z/s_min` with `s_min = min(s_x, s_y)`. Johnson and Harris write `a` for the major and `b` for
the minor semi-axis: for a line contact and a circle `s_min` is Johnson's `a`; on a ball-raceway contact
`s_x` (along rolling) is the minor semi-axis `b`.
