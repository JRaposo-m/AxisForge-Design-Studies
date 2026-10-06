# validation

AxisForge checked against an **independent reference** — a closed form, a
standard's worked example, a textbook example — with the acceptance criteria
written in the case README **before** running. A case answers "does AxisForge
get the known answer?"; an open engineering question belongs in `studies/`.

```
validation/
├── _support/          shared by the cases below only: reports, figures, convergence adapters
├── shafts/
│   ├── fem_uniform_radial_load/
│   └── fem_stepped_distributed_gear/
├── bearings/          (planned) iso281_catalogue_life, iso16281_harris_examples
└── gears/             (planned) iso6336_tr30_examples
```

## Rules

- **One folder per case**, named by what it checks (no numbers). Inside: a
  README, an optional `construction.py`, and the scripts/notebooks numbered in
  the order they are meant to be read (`01_…`, `02_…`).
- **Construction is written against the AxisForge API**, inside the case. No
  construction shared between cases: a case is a proof of that API, and the
  reader must see the whole of it. Repetition between cases is intended.
- `construction.py` returns the system **not solved**; the scripts choose the
  solver, beam model and mesh.
- **Imports:** a case may import `references`, `validation._support` and its own
  sibling `construction`. Never another case.
- **The README of each case records:** objective, AxisForge branch + commit of
  the last full run, reference, criteria (fixed before running), result.
- **Verification vs validation.** Mesh convergence (GCI, observed order) shows
  the discrete model converges — verification. Agreement with the reference is
  validation. A case can do both; the README says which result is which.
- Regression values (for `pytest`) live in `tests/`, not in the cases.
- Generated files go to `artifacts/validation/<domain>/<case>/` (git-ignored).

## Planned cases (have a known answer → validation, not studies)

| Case | Reference |
|---|---|
| `bearings/iso281_catalogue_life` | L10 and modified life (a_ISO) vs published catalogue values |
| `bearings/iso16281_harris_examples` | Internal load distribution vs the worked examples in Harris & Kotzalas, *Rolling Bearing Analysis* |
| `gears/iso6336_tr30_examples` | Safety factors vs ISO/TR 6336-30 calculation examples |
