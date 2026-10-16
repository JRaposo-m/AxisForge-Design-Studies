# AxisForge Design Studies

Proof that AxisForge works, and the place where AxisForge and SlipPY answer engineering questions.
Shared rules for all repositories: `../CLAUDE.md`. Layout and rules: `README.md`.

## Branch
Work on `main` (secondary was fast-forwarded into it on 2026-10-10). One branch per task.

## Where things go
- `references/` closed-form judges (imports nothing from this repo).
- `slippy_bridge/` the ONLY place that joins AxisForge to SlipPY (D-0019): `contact/` (base, elements/
  bearings|gears, reference, smooth, rough, static_step, outputs), `lubrication/` later. Studies and
  validation cases import SlipPY things from here and only give inputs. Rolling = array axis 0;
  AxisForge radii are converted here (outer raceway sign), AxisForge itself is not changed.
- `validation/<case>/` known answer, criteria fixed before running; may use `references`,
  `validation._support`, its own files.
- `studies/<family>/<study>/` open questions; may use `references`, its own files and the family's
  `_common/` (D-0006, proposed). **No study imports another study.**
- `examples/` how a library is driven, no acceptance criterion.
- Inside a case or study: `README.md` (question, AxisForge commit, criteria, result), optional
  `construction.py` returning the system **unsolved**, scripts numbered by stage (`01_…`).
- Generated output: `results/` next to each study or validation case (git-ignored, D-0020).

## Reuse before writing (D-0018)
This repository builds on two libraries; most helpers already exist there.
- Records, tables, CSV: `axisforge.outputs._format` (`write_records_csv`, `read_records_csv`,
  `format_table`, `format_key_values`, `union_fields`) and `axisforge.outputs.construction|solvers`.
- Contact geometry, surfaces, static step, Hertz on bridge axes: `slippy_bridge.contact`.
- Hertz reference: `slippy.contact.hertz.hertz_full` (local fixes in SlipPY `local-fixes`), through
  `slippy_bridge.contact.reference`.
- Figure appearance: `style/axisforge.mplstyle` via `plt.style.use(...)`; no hand-written styling
  functions.
- Solving: call AxisForge solvers directly; a study does not wrap them in its own solver layer.
Known duplicates to remove when you touch these files (do not copy the pattern):
`load_distribution/_common/reports.py::union_fields` (pass-through of `_format.union_fields`);
`load_distribution/_common/plots.py::style_axes/style_legend` (duplicate the style sheet).

## Commands
```bat
pip install -e ..\AxisForge-Shaft-Bearing-Gear-System-Analysis-Platform
pip install -e . && pip install -e ..\slippy
python -m pytest                                              :: tests/ : imports + validation regression
python -m pytest studies\bearings\load_distribution\tests     :: study tests (not in the default run)
```
Notebooks are jupytext pairs (`py:percent`); review and edit the `.py`.

## Plan in force (2026-10-10)
Dry contact first (two confirming studies), then roughness and adhesion/wear, lubrication last
(`Tese/80_Inbox/plan_contact_first_2026-10-10.md`). Code loop: coder -> minimal-reviewer +
physics-reviewer -> José.

## Known drift (2026-10-10)
- 48 result files (`load_distribution/clearance_and_alpha`, `iso16281_ball_contact/results`) are
  tracked, against the git-ignored `results/` rule (D-0020); the `hertz_vs_bem` ones were untracked
  on 2026-10-16. Do not add more; untracking the rest is José's call.
- README sections "Dependency rules" (no `_common`), "Vault" (old vault path) predate D-0006 and the
  `Tese` vault.
