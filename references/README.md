# references

Closed-form solutions used as **independent judges** of AxisForge (and, for
contact, of SlipPY).

```
references/
├── beams/      uniform_beam (Macaulay), stepped_beam (piecewise exact), shear_coefficients (κ)
└── contact/    Hertz — empty until the reference implementation is chosen
```

## Rules

- Never call an AxisForge or SlipPY **solver** from here: a reference must not
  reuse the code it checks (κ is deliberately re-implemented, not taken from
  AxisForge's `shear_factor.py`).
- Reading AxisForge **input** objects is allowed (the loads and supports of a
  constructed `ShaftSystem`, `LoadPlane`), so the reference sees exactly the
  loads the solver saw.
- Imports nothing else from this repository. Never a validation case or a study
  (examples in docstrings included).
- Docstrings follow ADR-003: units, conventions, assumptions, source (book,
  standard clause).
