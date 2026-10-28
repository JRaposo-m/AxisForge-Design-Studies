# studies/outputs

Plot and report models of the studies (D-0036 amendment items 3 and 10): no case draws or formats by
itself; it builds records and calls a function listed here. Tables are printed and written with
`axisforge.outputs._format` (`format_table`, `write_records_csv`) directly; a report model is added here
only when a case needs more than that.

```text
outputs/
  common/style/      paper.mplstyle (house paper style, ported from figures/paper-style 19cc689), PAPER
  l100_dry_contact/  plots/contact.py
  tests/             the index test below
```

- Every figure is drawn inside `matplotlib.pyplot.style.context(PAPER)` and written as
  `<case>_<file>_<function>[_<variant>].png` into the calling case's `results/` (git-ignored, D-0020).
- Semantics of the style: numerical result = filled markers; formula / theory = lines (own solid,
  others dashed); published value = open markers. Colour never carries identity alone. No titles.
- A test (`tests/test_index.py`) fails when a public function of a model module is missing from the
  index below.

## Index
| function | file | what it shows | record fields | called by |
|---|---|---|---|---|
| `relative_errors` | `l100_dry_contact/plots/contact.py` | `abs(rel_err)` [%] of BEM quantities vs a swept parameter, one panel per series, tolerances dashed | `<x>`, `quantity`, `series`, `rel_err`, opt. `tol`, `verdict` | v101, v102, v103, v104 |
| `pressure_cut` | `l100_dry_contact/plots/contact.py` | BEM `p/p0_H` along cuts through the centre vs the Hertz profile `sqrt(1 - xi^2)` | `axis`, `xi`, `p_over_p0_ref` | v101, v102, v103 |
