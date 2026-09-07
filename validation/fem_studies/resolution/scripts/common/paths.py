"""
validation/abaqus_comparison/scripts/common/paths.py

Path helpers shared by every case_*.py.

Two problems this version fixes vs. the previous one (which hardcoded
_LEVELS_UNDER_SCRIPTS = 4):

1. Depth-agnostic base path. The root is found by walking UP from the
   script file until an ancestor directory is literally named "scripts"
   -- its parent is abaqus_comparison/. This means the tree under
   scripts/ can grow or shrink a level (e.g. you add a level for abaqus
   element type, a sweep dimension, whatever) without editing this file
   -- there is no depth constant to keep "in step" any more.

2. Caller-controlled sub-path. Every case script decides its OWN target
   sub-path under results/ / abaqus_results/ / comparison/, by passing
   extra path segments -- instead of this module silently assuming "one
   script = one result folder". This matters once a script sweeps
   several values per shaft (a common shape here: several loads,
   several shafts, one script) -- each value/shaft combination gets its
   own segment, chosen by the script that knows what it's sweeping, not
   guessed by this module from the filename alone.

The base location (mirrored 1:1 across results/ / abaqus_results/ /
comparison/) is still the script's own position under scripts/, WITHOUT
the .py suffix -- that part stays automatic, so a case's script location
and its default result location can't drift apart by a typo. What you
control per call is what comes AFTER that base.

Usage
-----
    import sys
    from pathlib import Path
    sys.path.insert(0, str(_find_scripts_root(__file__).parent))
    from common.paths import results_dir, abaqus_results_dir, case_name

    # same as before (no extra segments) -- one result location for the
    # whole case, as every case_*.py in this suite has done so far:
    RESULTS_DIR = results_dir(__file__)          # .../results/<case_subtree>/
    out_dir = RESULTS_DIR / shear_theory

    # OR, when a script sweeps values per shaft/case and needs each
    # sweep point in its own folder -- pass whatever segments make sense
    # for THIS script, in whatever order makes sense for THIS script:
    out_dir = results_dir(__file__, shaft_name, f"F={load_n:.0f}N", shear_theory)
    # -> .../results/<case_subtree>/<shaft_name>/F=500N/<shear_theory>/

Nothing here creates directories -- every case_*.py still calls
`.mkdir(parents=True, exist_ok=True)` itself before writing into one.
"""
from __future__ import annotations

from pathlib import Path


def _find_scripts_root(script_file: str) -> Path:
    """
    Walk up from `script_file` until an ancestor is literally named
    "scripts" -- that ancestor IS scripts/, regardless of how many
    levels separate it from `script_file`.

    Raises ValueError if no such ancestor exists -- every case_*.py in
    this suite must live somewhere under abaqus_comparison/scripts/.
    """
    p = Path(script_file).resolve()
    for ancestor in p.parents:
        if ancestor.name == "scripts":
            return ancestor
    raise ValueError(
        f"{script_file}: no ancestor directory named 'scripts' found -- "
        f"every case_*.py must live under abaqus_comparison/scripts/."
    )


def _abaqus_comparison_root(script_file: str) -> Path:
    return _find_scripts_root(script_file).parent


def case_name(script_file: str) -> str:
    """The case's own name -- the script's filename without .py."""
    return Path(script_file).resolve().stem


def case_subtree(script_file: str) -> Path:
    """
    The script's own path relative to scripts/, without the .py suffix
    -- e.g. uniform_shafts/point_load/timoshenko/case_x for a script at
    scripts/uniform_shafts/point_load/timoshenko/case_x.py. This is the
    piece that is automatic; everything passed as `*extra` to the
    functions below is appended AFTER it, caller's choice.
    """
    p = Path(script_file).resolve()
    scripts_root = _find_scripts_root(script_file)
    rel = p.relative_to(scripts_root)  # <...>/<case>.py
    return rel.parent / rel.stem


def _mirror_dir(script_file: str, top: str, *extra: str) -> Path:
    root = _abaqus_comparison_root(script_file)
    return root.joinpath(top, case_subtree(script_file), *extra)


def results_dir(script_file: str, *extra: str) -> Path:
    """.../results/<case_subtree>/<extra...>/ -- AxisForge's own output.
    Pass e.g. (shaft_name, value_label, shear_theory) when this script
    sweeps more than one result location; pass nothing for a single-
    location case, same as every case_*.py has done so far."""
    return _mirror_dir(script_file, "results", *extra)


def abaqus_results_dir(script_file: str, *extra: str) -> Path:
    """.../abaqus_results/<case_subtree>/<extra...>/ -- where the
    manually-exported Abaqus report/CSV for this (sub-)case is expected
    to live. Not created automatically -- this is a read location, filled
    by hand. Pass the SAME extra segments you used for results_dir() so
    the two trees line up for a diff."""
    return _mirror_dir(script_file, "abaqus_results", *extra)


def comparison_dir(script_file: str, *extra: str) -> Path:
    """.../comparison/<case_subtree>/<extra...>/ -- reserved for a future
    compare.py's own output (diffs, overlay plots). Not written by any
    case_*.py today."""
    return _mirror_dir(script_file, "comparison", *extra)