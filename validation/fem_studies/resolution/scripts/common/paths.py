"""
validation/abaqus_comparison/scripts/common/paths.py

Path helpers shared by every case_*.py and compare_case_*.py.

Root discovery: the suite root (abaqus_comparison/, whatever your repo
actually calls it -- e.g. validation/fem_studies/resolution/) is found
by walking UP from the calling script until an ancestor directory
contains BOTH a "scripts" subdirectory AND a "results" subdirectory --
NOT by requiring the calling script itself to be located inside
"scripts/". This matters because a compare_case_*.py does not have to
live under scripts/ next to its case_*.py -- you may prefer to keep it
right next to the comparison/ output it produces instead (e.g. inside
comparison/.../case_x/). Either location works; root discovery finds
the suite root the same way regardless of where in the tree the calling
file sits, as long as it's somewhere inside the suite (under scripts/,
results/, abaqus_results/, comparison/, or the suite root itself).

Two ways to say which case a script's results/abaqus_results/comparison
paths belong to:

1. Default (every case_*.py so far, and any compare_case_*.py placed
   under scripts/ next to its case_*.py): the case subtree is derived
   from the CALLING SCRIPT'S OWN POSITION relative to scripts/ -- e.g.
   scripts/uniform_shafts/point_load/timoshenko/case_x.py implies
   subtree uniform_shafts/point_load/timoshenko/case_x. A
   compare_case_x.py placed the same way needs `case_stem="case_x"` to
   override just the LAST component (its own filename differs from the
   case's), see case_subtree()'s own docstring.

2. Explicit override (`case_subtree_override=`): give the full subtree
   string yourself, e.g. "uniform_shafts/point_load/timoshenko/case_x"
   -- the calling script's own location is not used to derive it at
   all. Use this for a compare_case_*.py you deliberately placed
   somewhere OTHER than scripts/ (e.g. next to the comparison/ output
   it targets) -- root discovery (see above) still finds the suite
   root from wherever the file is, but the case subtree itself can no
   longer be inferred from "position relative to scripts/" once the
   file isn't under scripts/, so you say it directly instead.

Caller-controlled sub-path: every case script decides its OWN target
sub-path under results/ / abaqus_results/ / comparison/, by passing
extra path segments (`*extra`) -- instead of this module silently
assuming "one script = one result folder". This matters once a script
sweeps several values per shaft (a common shape here: several loads,
several shafts, one script) -- each value/shaft combination gets its
own segment, chosen by the script that knows what it's sweeping, not
guessed by this module from the filename alone.

Usage
-----
    import sys
    from pathlib import Path
    _p = Path(__file__).resolve()
    _root = None
    for ancestor in _p.parents:
        if (ancestor / "scripts").is_dir() and (ancestor / "results").is_dir():
            _root = ancestor
            break
    if _root is None:
        raise RuntimeError(f"{__file__}: not inside the abaqus_comparison suite tree")
    sys.path.insert(0, str(_root / "scripts"))
    from common.paths import results_dir, abaqus_results_dir, case_name

    # case_*.py under scripts/ -- subtree auto-derived from position, no
    # extra segments -- one result location for the whole case, as every
    # case_*.py in this suite has done so far:
    RESULTS_DIR = results_dir(__file__)          # .../results/<case_subtree>/
    out_dir = RESULTS_DIR / shear_theory

    # OR, when a script sweeps values per shaft/case and needs each
    # sweep point in its own folder -- pass whatever segments make sense
    # for THIS script, in whatever order makes sense for THIS script:
    out_dir = results_dir(__file__, shaft_name, f"F={load_n:.0f}N", shear_theory)
    # -> .../results/<case_subtree>/<shaft_name>/F=500N/<shear_theory>/

    # OR, a compare_case_x.py NOT placed under scripts/ -- say the
    # subtree directly, its own location is irrelevant to path building:
    RESULTS_DIR = results_dir(__file__, case_subtree_override="uniform_shafts/point_load/timoshenko/case_x")

Nothing here creates directories -- every case_*.py still calls
`.mkdir(parents=True, exist_ok=True)` itself before writing into one.
"""
from __future__ import annotations

from pathlib import Path


def _find_suite_root(script_file: str) -> Path:
    """
    Walk up from `script_file` until an ancestor directory contains
    BOTH a "scripts" subdirectory AND a "results" subdirectory -- that
    ancestor IS the suite root (abaqus_comparison/ / .../resolution/),
    regardless of where under the suite tree `script_file` itself
    lives (scripts/, results/, abaqus_results/, comparison/, or the
    root itself all work).

    Raises ValueError if no such ancestor exists -- every case_*.py /
    compare_case_*.py in this suite must live somewhere inside the
    suite tree.
    """
    p = Path(script_file).resolve()
    for ancestor in p.parents:
        if (ancestor / "scripts").is_dir() and (ancestor / "results").is_dir():
            return ancestor
    raise ValueError(
        f"{script_file}: no ancestor directory containing both 'scripts/' "
        f"and 'results/' found -- this file must live somewhere inside "
        f"the abaqus_comparison suite tree."
    )


def _find_scripts_root(script_file: str) -> Path:
    """scripts/ itself, found via _find_suite_root() -- kept as a
    separate name since some callers only care about this one folder
    (e.g. the sys.path bootstrap in this module's own docstring)."""
    return _find_suite_root(script_file) / "scripts"


def _abaqus_comparison_root(script_file: str) -> Path:
    return _find_suite_root(script_file)


def case_name(script_file: str) -> str:
    """The case's own name -- the script's filename without .py."""
    return Path(script_file).resolve().stem


def case_subtree(
    script_file: str,
    case_stem: str | None = None,
    case_subtree_override: str | Path | None = None,
) -> Path:
    """
    The case subtree used to build results_dir()/abaqus_results_dir()/
    comparison_dir() -- e.g. uniform_shafts/point_load/timoshenko/case_x.

    Two ways to get it -- see this module's own top docstring for the
    full explanation of when to use which:

    - `case_subtree_override` given: returned AS-IS (as a Path), the
      calling script's own location is not consulted at all. Use this
      for a compare_case_*.py you placed somewhere other than
      scripts/ (its own position can't imply a subtree if it isn't
      under scripts/ to begin with).

    - `case_subtree_override` omitted (default): derived from
      `script_file`'s OWN path relative to scripts/ -- e.g.
      scripts/uniform_shafts/point_load/timoshenko/case_x.py implies
      uniform_shafts/point_load/timoshenko/case_x. Requires
      `script_file` to actually be located under scripts/ (raises via
      _find_suite_root() otherwise). `case_stem` overrides just the
      LAST component (normally the script's own filename stem) --
      needed by a compare_case_x.py placed under scripts/ next to its
      case_x.py: without it, results_dir()/etc. would mirror
      "compare_case_x" (a folder case_x.py never wrote to) instead of
      "case_x" (the folder that actually holds its results).
    """
    if case_subtree_override is not None:
        return Path(case_subtree_override)

    p = Path(script_file).resolve()
    scripts_root = _find_scripts_root(script_file)
    rel = p.relative_to(scripts_root)  # <...>/<case>.py
    return rel.parent / (case_stem if case_stem is not None else rel.stem)


def _mirror_dir(
    script_file: str,
    top: str,
    *extra: str,
    case_stem: str | None = None,
    case_subtree_override: str | Path | None = None,
) -> Path:
    root = _abaqus_comparison_root(script_file)
    subtree = case_subtree(
        script_file, case_stem=case_stem, case_subtree_override=case_subtree_override,
    )
    return root.joinpath(top, subtree, *extra)


def results_dir(
    script_file: str, *extra: str,
    case_stem: str | None = None,
    case_subtree_override: str | Path | None = None,
) -> Path:
    """.../results/<case_subtree>/<extra...>/ -- AxisForge's own output.
    Pass e.g. (shaft_name, value_label, shear_theory) when this script
    sweeps more than one result location; pass nothing for a single-
    location case, same as every case_*.py has done so far. Pass
    `case_stem=` or `case_subtree_override=` from a compare_case_*.py --
    see case_subtree()'s own docstring for which one and why."""
    return _mirror_dir(
        script_file, "results", *extra,
        case_stem=case_stem, case_subtree_override=case_subtree_override,
    )


def abaqus_results_dir(
    script_file: str, *extra: str,
    case_stem: str | None = None,
    case_subtree_override: str | Path | None = None,
) -> Path:
    """.../abaqus_results/<case_subtree>/<extra...>/ -- where the
    manually-exported Abaqus report/CSV for this (sub-)case is expected
    to live. Not created automatically -- this is a read location, filled
    by hand. Pass the SAME extra segments you used for results_dir() so
    the two trees line up for a diff. Pass `case_stem=` or
    `case_subtree_override=` from a compare_case_*.py -- see
    case_subtree()'s own docstring for which one and why."""
    return _mirror_dir(
        script_file, "abaqus_results", *extra,
        case_stem=case_stem, case_subtree_override=case_subtree_override,
    )


def comparison_dir(
    script_file: str, *extra: str,
    case_stem: str | None = None,
    case_subtree_override: str | Path | None = None,
) -> Path:
    """.../comparison/<case_subtree>/<extra...>/ -- compare.py's own
    output (diffs, overlay plots). Pass `case_stem=` or
    `case_subtree_override=` from a compare_case_*.py -- see
    case_subtree()'s own docstring for which one and why."""
    return _mirror_dir(
        script_file, "comparison", *extra,
        case_stem=case_stem, case_subtree_override=case_subtree_override,
    )