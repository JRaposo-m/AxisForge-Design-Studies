"""
validation/abaqus_comparison/scripts/common/paths.py

Path helpers shared by every case_*.py and compare_case_*.py.

Layout (post-reorg): each case is a self-contained folder --

    <shaft_type>/<load_type>/<theory>/case_x/
        case_x.py
        results/
        abaqus_results/
        comparison/
            compare_case_x.py
            <shear_theory>/...

-- so a script's own directory tells you the case root, but NOT always
directly: case_x.py sits straight inside case_x/ (its own parent IS the
case root), while compare_case_x.py sits one level deeper, inside
case_x/comparison/ (its parent is named "comparison", so the case root
is one level up from there instead). case_dir() below handles both.

No more mirroring across three parallel top-level trees, and no more
case_subtree/case_stem/case_subtree_override -- case_dir() resolves the
case root directly from wherever the calling script actually lives.

Root discovery: the suite root (e.g. validation/fem_studies/resolution/)
is found by walking UP from the calling script until an ancestor
directory contains a "scripts/common" subdirectory -- that's the one
thing every case script needs to find (bearings.py, compare.py, this
module itself), regardless of how deep the calling script sits under
<shaft_type>/<load_type>/<theory>/case_x/(comparison/).

Usage
-----
    import sys
    from pathlib import Path
    _p = Path(__file__).resolve()
    _root = None
    for ancestor in _p.parents:
        if (ancestor / "scripts" / "common").is_dir():
            _root = ancestor
            break
    if _root is None:
        raise RuntimeError(f"{__file__}: not inside the suite tree")
    sys.path.insert(0, str(_root / "scripts"))
    from common.paths import results_dir, abaqus_results_dir, comparison_dir

    RESULTS_DIR = results_dir(__file__)              # .../case_x/results/
    out_dir = RESULTS_DIR / shear_theory

    # A script that sweeps values per shaft/case and needs each sweep
    # point in its own folder -- pass whatever segments make sense:
    out_dir = results_dir(__file__, shaft_name, f"F={load_n:.0f}N", shear_theory)
    # -> .../case_x/results/<shaft_name>/F=500N/<shear_theory>/

Nothing here creates directories -- every case_*.py still calls
`.mkdir(parents=True, exist_ok=True)` itself before writing into one.
"""
from __future__ import annotations

from pathlib import Path


# Directory names that mean "this is an output folder, not the case
# root itself" -- a script found directly inside one of these needs to
# go up one more level to reach the case root. Only "comparison" is
# actually used this way today (compare_case_x.py lives at
# case_x/comparison/compare_case_x.py), but "results" and
# "abaqus_results" are included too in case a helper script is ever
# placed inside one of them the same way.
_OUTPUT_DIR_NAMES = {"results", "abaqus_results", "comparison"}


def _find_suite_root(script_file: str) -> Path:
    """
    Walk up from `script_file` until an ancestor directory contains a
    "scripts/common" subdirectory -- that ancestor IS the suite root,
    regardless of how deep under the suite tree `script_file` lives.

    Raises ValueError if no such ancestor exists.
    """
    p = Path(script_file).resolve()
    for ancestor in p.parents:
        if (ancestor / "scripts" / "common").is_dir():
            return ancestor
    raise ValueError(
        f"{script_file}: no ancestor directory containing 'scripts/common/' "
        f"found -- this file must live somewhere inside the suite tree."
    )


def case_dir(script_file: str) -> Path:
    """The case's own root folder.

    - case_x.py sits directly inside case_x/ -- its own parent IS the
      case root.
    - compare_case_x.py sits inside case_x/comparison/ -- its parent's
      NAME is "comparison" (see _OUTPUT_DIR_NAMES), so the case root is
      one level further up instead.
    """
    d = Path(script_file).resolve().parent
    if d.name in _OUTPUT_DIR_NAMES:
        return d.parent
    return d


def case_name(script_file: str) -> str:
    """The case's own name -- its folder's name (case_x)."""
    return case_dir(script_file).name


def results_dir(script_file: str, *extra: str) -> Path:
    """.../case_x/results/<extra...>/ -- AxisForge's own output."""
    return case_dir(script_file).joinpath("results", *extra)


def abaqus_results_dir(script_file: str, *extra: str) -> Path:
    """.../case_x/abaqus_results/<extra...>/ -- where the manually-
    exported Abaqus report/CSV for this (sub-)case is expected to live.
    Not created automatically -- this is a read location, filled by
    hand. Pass the SAME extra segments you used for results_dir() so
    the two trees line up for a diff."""
    return case_dir(script_file).joinpath("abaqus_results", *extra)


def comparison_dir(script_file: str, *extra: str) -> Path:
    """.../case_x/comparison/<extra...>/ -- compare.py's own output
    (diffs, overlay plots)."""
    return case_dir(script_file).joinpath("comparison", *extra)
