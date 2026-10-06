# studies

**Engineering questions** answered with AxisForge (and, for contact, SlipPY).
A study has no known answer to check against — if it has one, it is a
validation case (`validation/`).

```
studies/
├── shafts/
│   ├── slenderness_eb_vs_timoshenko/
│   └── stepped_fatigue_fillet/
├── bearings/
│   └── clearance_load_distribution/
├── gears/
│   └── khbeta_shaft_deflection/
├── systems/
│   └── hyperstatic_reactions_vs_bearing_stiffness/
└── contact/
    └── ball_on_raceway/          the AxisForge → SlipPY spike
```

All are planned; each README holds the question and why it matters.

## Rules

- **One folder per study**, named by the question (no numbers). Inside:
  README, optional `construction.py`, scripts numbered by stage (`01_…`, `02_…`).
- **Construction against the AxisForge API, inside the study.** `construction.py`
  returns the system **not solved**. When the variable is the system itself
  (clearance, L/d, bearing choice), `construction.py` takes it as a parameter and
  the study sweeps it. Keep construction in the study script when it is trivial
  or used once — separate it when that makes the study easier to read.
- **Stages:** build → solve → analyse. Frozen data between stages is a file in
  `artifacts/` stamped with the AxisForge commit that produced it — e.g. the
  ADR-002 fields (`Q_j`, `alpha_j`, `phi_j`, `r1`, `r2`) written by the AxisForge
  stage and read by the SlipPY stage.
- **Imports:** only `references` and the study's own files. Never another
  study, never `validation._support` (copy what you need).
- **README of each study:** question, AxisForge commit, criteria if any (fixed
  before running), what each file does, result and conclusion.
- Generated files go to `artifacts/studies/<domain>/<study>/` (git-ignored).
