# studies/bearings/load_distribution

How do the parameters of a rolling bearing change its internal load distribution (ISO/TS 16281):
number of loaded elements, largest element load, deflection, stiffness and reference life?

The bearing geometries are **didactic**: they are chosen to show the influence of each parameter,
not to represent catalogue bearings. Designations are labels only.

## Layout: question → bearing type

```
load_distribution/
├── README.md                 this file
├── _common/                  shared code, no data
│   ├── bearings/             one specification per bearing type
│   │   ├── base.py             BearingSpec
│   │   ├── deep_groove.py      DeepGrooveSpec          (locating)
│   │   ├── angular_contact.py  AngularContactSpec      (locating)
│   │   └── cylindrical_roller.py  CylindricalRollerSpec (non-locating)
│   ├── system.py             SystemSpec + build_system: the two-shaft gear system
│   ├── solve.py              solve_fem, solve_bearing
│   ├── rows.py               result tables (one schema for every type), derived columns
│   ├── study.py              LoadDistributionStudy: the sweep loop
│   ├── reports.py            report.txt and the CSV tables (system, FEM, ISO/TS 16281)
│   ├── store.py              writes and reads results/
│   ├── plots.py              figure style and generic figures
│   └── compare.py            reads the results/ of the types of one question
├── clearance_and_alpha/      QUESTION
│   ├── README.md               question, axes, assumptions, result
│   ├── study.py                the axes and the hooks of the question
│   ├── deep_groove/            TYPE under study (sweeps s)
│   │   ├── run.py                chooses the BearingSpec and runs the question
│   │   ├── results/              report.txt + CSV tables of this type (versioned)
│   │   └── plots/                figures of this type
│   ├── cylindrical_roller/     (same; sweeps s)
│   ├── angular_contact/        (same; sweeps alpha_0)
│   └── compare/
│       ├── compare.py            only reads ../<type>/results, never recomputes
│       └── plots/
├── misalignment/             (same form: deep_groove, cylindrical_roller, compare)
└── tests/
    ├── test_common.py        plumbing, with a fake solver (no AxisForge needed)
    └── test_regression.py    every (question, type) against its stored results/
```

**Who decides what.** The question's `study.py` declares the swept axes (names and values, in sweep
order; either the same for every type, `axes`, or one set per type, `axes_by_kind`), the helix
angle and the hooks `configure` (how a point changes the bearing), `power_W` (the transmitted
power of the gear stage) and `psi`. The type's `run.py` only chooses the bearing under study.
Everything else (system, FEM, loop, rows, derived columns, files, figures) lives in `_common`.

## Construction rule

Each shaft has two bearings: **1 locating + 1 non-locating**.

| Slot | Arrangement | Allowed types | Why |
|---|---|---|---|
| bearing 1 (x = 10 mm) | `locating` | `DeepGrooveSpec`, `AngularContactSpec` | it carries the axial load of the gear mesh |
| bearing 2 (x = 190 mm) | `non-locating` | `CylindricalRollerSpec` | axially free; no axial capacity (Fa = 0) |

`SystemSpec` refuses any other pair. The type folder is the bearing **under study**; the other slot
holds the partner in its base configuration (a roller for the locating types, a deep groove ball
bearing for the roller; `run.py` may pass another partner).

## Assumptions and limitations

- **Rigid supports.** The shaft FEM uses rigid supports, so each shaft is statically determinate:
  the reactions and slopes at the bearings depend only on positions, arrangement and gear loads,
  not on the bearing types. The FEM is solved once per study; only the bearing under study is
  solved, and the partner does not affect its results. This stops being true once the bearing
  stiffness is fed back into the shaft model (planned study
  `systems/hyperstatic_reactions_vs_bearing_stiffness`).
- **Load = transmitted power.** The load is a construction parameter: `SystemSpec.power_W`
  (nominal 10 471.9755 W = 100 N m at 1000 rpm). Each power is its own system and its own FEM
  solve; nothing is scaled after the solve. Every row records the power it was computed at.
- **Helix angle per question.** `clearance_and_alpha` and `misalignment` use spur gears (Fa = 0).
  A single-row angular contact bearing cannot reach axial equilibrium without Fa (see
  `clearance_and_alpha/README.md`); its meaningful case is a helical sub-case (planned). With
  helical gears Fa at the pitch radius adds a moment, so the two bearings of one shaft do not
  carry the same Fr even with the gear at mid-span, and the two shafts have Fa of opposite sign.
- **Life.** `L10r` is the ISO/TS 16281 basic reference rating life of the post-processing, not an
  ISO 281 life.
- **Ball contact stiffness.** Deflections and stiffnesses of ball bearings depend on the
  `PointContactStiffness.cp` correction (ISO/TS 16281, factor 2 on E*); record the AxisForge
  version that produced each data set (the RUN INFORMATION of report.txt does).

## Running

From the type folder:

```bash
python run.py            # writes results/ and plots/ of that type
```

From a question's `compare/` folder, after its types have been run:

```bash
python compare.py        # checks the data are comparable, writes compare/plots/
```

Tests, from this folder:

```bash
pytest tests             # regression is skipped for a type without results/ or without AxisForge
```

## Results and figures

Each type folder holds `results/` (versioned: the baseline of the regression and the only input of
`compare`) and `plots/`:

| File | Contents |
|---|---|
| `results/report.txt` | readable report: 1 run information, 2 system, 3 shaft FEM (bearing nodes), 4 ISO/TS 16281 table and element loads, 5 checks |
| `results/system.csv` | every system parameter: section, parameter, value, unit |
| `results/fem_bearing_nodes.csv` | bearing-node results of the FEM, one row per (power, shaft, node) |
| `results/iso16281_rows.csv` | one row per (point of the sweep, shaft) |
| `results/iso16281_elements.csv` | element loads Q_j |
| `results/iso16281_laminae.csv` | lamina loads of the most loaded roller (rollers only) |
| `plots/summary/<quantity>.png` | one figure per quantity against the first axis, one line per value of the second |
| `plots/load_distribution/<point>.png` | the bearing seen along its axis with the element loads as a line around it, one per point; same radial scale for the whole sweep |
| `plots/load_distribution_overview[__<value>].png` | those drawings side by side, same scale; one overview per value of the second axis, if any |
| `plots/roller_lamina_profile[__<value>].png` | lamina loads along the most loaded roller (rollers only); one figure per value of the second axis, if any |

The RUN INFORMATION block at the top of `report.txt` records the question, the axes, both bearing
specifications, the helix angle, the power, the psi source (FEM or prescribed), the FEM slope at
the bearing, the AxisForge version, a hash of the parameters and the date. It is read by `compare`
(which refuses data produced on different AxisForge versions, psi sources, helix angles or powers)
and by the regression tests: do not edit it.

## Dependencies

A study of this family may import `_common` of this family and its own question's `study.py`;
never another question, never `_common` of another family. `compare/` reads only the `results/` of the
sibling type folders of the same question.
