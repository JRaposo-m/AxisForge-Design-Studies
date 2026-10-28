# studies

The cases of this repository: **r** analytical references (independent judges), **v** validations
(AxisForge or SlipPY against a known answer) and **s** studies (open engineering questions). Layout and rules
of D-0036 with its amendment of 2026-10-27; the reasons are in the vault (`01_Decisions/_register.md` D-0036,
`80_Inbox/2026-10-27/D-0036_amendment_notes_2026-10-27.md`). The old v and s cases were swept on
2026-10-27; their code is in the tag `studies-pre-sweep` until each is ported.

```text
studies/
  r00_analytical_references/      r cases sit directly below the level folder
    r01_beams/  r02_hertz_contact/  r04_involute/  r05_ehl_film_and_rheology/
  v000_axisforge/  v100_dry_contact/  ...   v cases, by level (0 AxisForge, 1 dry contact, 2 rough, 3 EHL, ...)
  s000_axisforge/  s100_dry_contact/  ...   s cases, by level
  outputs/                        plot and report models (to come at run O1)
```
A level folder exists only when it holds a case.

## A case is a small repository
```text
xNNN_<name>/
  README.md        1 question or reference, 2 sources, 3 scope, 4 public API, 5 data, 6 criteria,
                   7 depends on, 8 how to run (9 result, after a run)
  __init__.py      public API: explicit imports and __all__
  construction.py  the system, NOT solved (default; a construction/ folder only when it grows)
  check.py (v)     check(system, ...) -> comparison records; takes any system, from the literature or a study
  literature/ (v)  the published systems the v knows, with published/ tables of the papers
  model.py (s)     the model of the study
  study.py         one-step case; a study/ folder with __main__.py (orchestrator) for several steps
  tests/           fast tests of the case
  results/         generated, git-ignored (D-0020), read by no other case
```
- **r:** one module of formulas with sources (a second past about 400 lines), `published/`, tests against the
  printed tables; no `study.py`, no `results/`, no notebook.
- **What an r imports (rule 13 of the D-0036 amendment):** nothing from `studies/` (no other case, no `outputs/`); NumPy, SciPy and the
  standard library; from the libraries (AxisForge, SlipPY, `slippy_bridge`) only a function that is itself
  analytical formulas José has verified, listed with the date in `R_VERIFIED` of `tests/test_imports.py`
  (today r02 -> SlipPY `hertz_full`). Otherwise the r writes the formulas itself.
- **v:** `check(system)` plus `literature/`; one Jupyter notebook (jupytext pair) that compares with the
  literature; its `check` calls the matching r.
- **s:** validates its own system through `v.check(its_system)` at its own conditions; never calls an r directly.
  A model shared by later studies lives in the first study and the later ones import its API; it moves to the
  library when three cases use it.
- **One case = one validation or one study:** no case holds several questions or several validations.
- **No plot or report-formatting code in a case:** styles, plot and report models live in `outputs/`.
- Relative imports inside a case; a case is imported as `studies.<level>.<case>` and run as
  `python -m studies.<level>.<case>.study`; records are `list[dict]` with units in the keys, written with
  `axisforge.outputs._format`.
