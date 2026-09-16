"""
validation/fem_studies/resolution/common/paths.py

Path helpers shared by every case_*.py, compare_case_*.py and
analytical_case_*.py.

Layout (post-reorg) actually on disk today:

    <shaft_type>/<load_type>/case_x/
        case_x.py
        euler_bernoulli/
            results/
            abaqus_results/
            comparison/
                compare_case_x.py
        timoshenko/
            results/<shear_theory>/
            abaqus_results/
            comparison/
                compare_case_x.py
        analytical/
            analytical_case_x.py
            results/euler_bernoulli/, results/timoshenko/<shear_theory>/
            comparison/euler_bernoulli/, comparison/timoshenko/<shear_theory>/
                compare_analytical_case_x.py

CHANGED (this pass): analytical/ used to be a special, theory-agnostic
"analytical_solution/" folder with its own analytical_dir() helper,
living directly under the case root with NO results/comparison split at
that level. It is now a plain THIRD SIBLING of euler_bernoulli/ and
timoshenko/, shaped exactly like them (its own results/ and comparison/,
with the per-theory/shear_theory split happening ONE level deeper,
inside results/ and comparison/, via ordinary results_dir()/
comparison_dir() extra segments -- no analytical_dir() any more, no
special-casing in case_dir() either). The one real difference from
euler_bernoulli/timoshenko: analytical/ has no abaqus_results/ (nothing
Abaqus-shaped to diff against there) and its comparison/ diffs the
closed-form curve against each theory's OWN AxisForge resolution csv,
not against Abaqus.

CORRECTED (an earlier pass): this docstring used to describe the layout
as <shaft_type>/<load_type>/<theory>/case_x/ -- theory ABOVE case_x.
That was never what's on disk (case_x is above <theory>, not the
reverse). case_dir() itself was never wrong; only that comment was
stale, presumably left over from an even earlier layout.

case_dir()'s real contract, precisely: it returns the nearest
"self-contained root" the calling script belongs to -- NOT always the
true top of the case. Two different meanings depending on where the
script sits, and both are intentional:

  - case_x.py / analytical_case_x.py sit directly inside their own root
    (case_x/ itself, or case_x/analytical/ -- "analytical" is NOT in
    _OUTPUT_DIR_NAMES, so case_dir() does not pop past it, exactly like
    "euler_bernoulli"/"timoshenko" are not in that set for case_x.py).
    results_dir(__file__) etc. from either script need the theory
    passed explicitly as an `extra` segment, since each of these
    scripts writes into MORE than one theory's results/ (analytical_case_x.py
    the same way case_x.py itself does).

  - compare_case_x.py / compare_analytical_case_x.py sit inside
    case_x/<theory>/comparison/ (Abaqus side) or
    case_x/analytical/comparison/ (analytical side, theory NOT baked
    into the folder name the same way) -- popping ONE level (past
    "comparison", per _OUTPUT_DIR_NAMES) lands on case_x/<theory>/ or
    case_x/analytical/ respectively, NOT case_x/ itself. This is
    deliberate, not a bug: a comparison script only ever compares
    within its OWN sibling's results/(abaqus_results/), so treating
    that sibling as its "case_dir" means results_dir(__file__),
    abaqus_results_dir(__file__) and comparison_dir(__file__) resolve
    directly with no extra segments needed at the call site for the
    Abaqus side; the analytical side still needs the theory as an
    extra segment on comparison_dir(), same reason as
    analytical_case_x.py itself (its comparison/ isn't split into
    per-theory folders the way euler_bernoulli//timoshenko/ are).

Root discovery (`_find_suite_root`): walk UP from the calling script
until an ancestor directory contains a "common" subdirectory -- that's
the one thing every case/compare/analytical script needs to find
(bearings.py, compare.py, this module itself), regardless of how deep
the calling script sits under the tree above. (Previously this looked
for "scripts/common" -- the suite tree was flattened one level,
scripts/ removed, common/ moved straight under the suite root; see
_find_suite_root()'s own docstring.)

Usage
-----
    import sys
    from pathlib import Path
    _p = Path(__file__).resolve()
    _root = None
    for ancestor in _p.parents:
        if (ancestor / "common").is_dir():
            _root = ancestor
            break
    if _root is None:
        raise RuntimeError(f"{__file__}: not inside the suite tree")
    sys.path.insert(0, str(_root))
    from common.paths import results_dir, abaqus_results_dir, comparison_dir

    # from case_x.py (case root) -- theory is an explicit extra segment,
    # theory ABOVE results/ on disk, so build it off case_dir() directly
    # rather than through results_dir() (see case_radial_single_position.py
    # for the worked example -- results_dir()'s own extra segments join
    # AFTER the fixed folder name, which is the WRONG order for this):
    from common.paths import case_dir
    eb_dir = case_dir(__file__) / "euler_bernoulli" / "results"
    ts_dir = case_dir(__file__) / "timoshenko" / "results" / shear_theory

    # from analytical_case_x.py (case_x/analytical/) -- theory (and
    # shear_theory) as extra segments on results_dir() ITSELF this time,
    # because analytical/ has only ONE results/ folder shared by both
    # theories, split internally -- extra-after-fixed-name is exactly
    # right here, unlike the euler_bernoulli/timoshenko case above:
    out_dir = results_dir(__file__, "euler_bernoulli")            # .../analytical/results/euler_bernoulli/
    out_dir = results_dir(__file__, "timoshenko", shear_theory)   # .../analytical/results/timoshenko/<shear_theory>/

    # from compare_case_x.py (case_x/<theory>/comparison/) -- no extra
    # segment needed, case_dir() already sits inside <theory>/:
    csv_dir = results_dir(__file__) / "csv"              # .../case_x/<theory>/results/csv/
    abq_dir = abaqus_results_dir(__file__)                # .../case_x/<theory>/abaqus_results/

    # from compare_analytical_case_x.py (case_x/analytical/comparison/) --
    # case_dir() sits inside analytical/, same as analytical_case_x.py's
    # own case_dir() -- theory as an extra segment on comparison_dir():
    out_dir = comparison_dir(__file__, "timoshenko", shear_theory)   # .../analytical/comparison/timoshenko/<shear_theory>/

Nothing here creates directories -- every case_*.py / compare_case_*.py
/ analytical_case_*.py still calls `.mkdir(parents=True, exist_ok=True)`
itself before writing into one.
"""
from __future__ import annotations

from pathlib import Path


# Directory names that mean "this is an output folder, not the root the
# calling script treats as its own case_dir() -- a script found
# directly inside one of these needs to go up one more level to reach
# ITS OWN self-contained root (see this module's own docstring for why
# that root differs between case_x.py, compare_case_x.py and
# analytical_case_x.py / compare_analytical_case_x.py). Only
# "comparison" is actually used this way today ("results"/
# "abaqus_results" are included too in case a helper script is ever
# placed inside one of them the same way). "analytical" is deliberately
# NOT in this set -- it is a plain sibling of "euler_bernoulli"/
# "timoshenko", not an output folder a script needs to pop past.
_OUTPUT_DIR_NAMES = {"results", "abaqus_results", "comparison"}


def _find_suite_root(script_file: str) -> Path:
    """
    Walk up from `script_file` until an ancestor directory contains a
    "common" subdirectory -- that ancestor IS the suite root, regardless
    of how deep under the suite tree `script_file` lives.

    Previously this suite had common/ living inside a "scripts/" wrapper
    (scripts/common/) -- the tree was flattened one level (scripts/
    removed, common/ moved straight under the suite root), so every
    anchor check in this suite now looks for "common" directly, not
    "scripts/common". If this ever moves again, this is the one place
    -- plus every case_*.py / compare_case_*.py / analytical_case_*.py's
    own copy of this same walk-up block -- that needs updating together.

    Raises ValueError if no such ancestor exists.
    """
    p = Path(script_file).resolve()
    for ancestor in p.parents:
        if (ancestor / "common").is_dir():
            return ancestor
    raise ValueError(
        f"{script_file}: no ancestor directory containing 'common/' "
        f"found -- this file must live somewhere inside the suite tree."
    )


def case_dir(script_file: str) -> Path:
    """The calling script's own self-contained root -- see this
    module's own docstring for the precise (and NOT always "the true
    top of the case") meaning.

    - case_x.py / analytical_case_x.py sit directly inside their own
      root (case_x/ itself, or case_x/analytical/ -- "analytical" is
      NOT in _OUTPUT_DIR_NAMES, exactly like "euler_bernoulli"/
      "timoshenko" aren't, so no popping happens for any of them).
    - compare_case_x.py / compare_analytical_case_x.py sit inside
      case_x/<theory-or-analytical>/comparison/ -- its parent's NAME is
      "comparison" (see _OUTPUT_DIR_NAMES), so popping one level lands
      on case_x/<theory>/ or case_x/analytical/, not case_x/ itself --
      see module docstring for why that is the intended behaviour.
    """
    d = Path(script_file).resolve().parent
    if d.name in _OUTPUT_DIR_NAMES:
        return d.parent
    return d


def case_name(script_file: str) -> str:
    """The case_dir()'s own name -- see case_dir()'s docstring for which
    folder that actually is, depending on the caller."""
    return case_dir(script_file).name


def results_dir(script_file: str, *extra: str) -> Path:
    """<case_dir()>/results/<extra...>/ -- AxisForge's own output, or
    (from analytical_case_x.py) the closed-form output.

    - From compare_case_x.py: no extra segment needed (case_dir() there
      already sits inside <theory>/).
    - From analytical_case_x.py: pass the theory (and shear_theory,
      where relevant) as `extra` -- analytical/ has one results/ folder
      shared by both theories, split internally, so extra-after-
      "results" is exactly right here.
    - From case_x.py itself (euler_bernoulli/timoshenko FEM output):
      do NOT use this helper -- <theory>/ sits ABOVE results/ there,
      the opposite nesting. Build the path off case_dir() directly
      instead (see this module's own docstring, "Usage")."""
    return case_dir(script_file).joinpath("results", *extra)


def abaqus_results_dir(script_file: str, *extra: str) -> Path:
    """<case_dir()>/abaqus_results/<extra...>/ -- where the manually-
    exported Abaqus report/CSV for this (sub-)case is expected to live.
    Not created automatically -- this is a read location, filled by
    hand. Pass the SAME extra segments you used for results_dir() so
    the two trees line up for a diff. Only meaningful for
    euler_bernoulli/timoshenko's own compare_case_x.py -- analytical/
    has no Abaqus side to diff against."""
    return case_dir(script_file).joinpath("abaqus_results", *extra)


def comparison_dir(script_file: str, *extra: str) -> Path:
    """<case_dir()>/comparison/<extra...>/ -- compare.py's own output
    (diffs, overlay plots).

    - From compare_case_x.py (Abaqus side): pass the shear_theory as
      `extra` for the timoshenko variant, nothing for euler_bernoulli --
      matches results_dir()'s own per-script convention there.
    - From compare_analytical_case_x.py: pass the theory (and
      shear_theory) as `extra`, same reason as results_dir() from
      analytical_case_x.py itself -- analytical/comparison/ is one
      folder shared by both theories, split internally."""
    return case_dir(script_file).joinpath("comparison", *extra)
