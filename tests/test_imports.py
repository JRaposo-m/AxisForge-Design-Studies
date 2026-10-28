"""
Every case package imports, and the layer law holds where an import test can see it (D-0036).

- A case package is ``studies/<level>/<case>/`` with an ``__init__.py``; it is imported through
  its public API. Case folders without ``__init__.py`` (not migrated yet) are not imported.
- An r module (rule 13 of the D-0036 amendment, AST scan) imports nothing from ``studies`` (no other case, no
  outputs) and only NumPy, SciPy and the standard library, plus the library functions of ``R_VERIFIED``:
  analytical formulas José verified, one entry per function with the date. Relative imports only inside the
  case. The case tests may import the case package itself.
- ``slippy_bridge`` imports nothing from ``studies``.
"""
from __future__ import annotations

import ast
import importlib
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
STUDIES = ROOT / "studies"
R_LEVEL = STUDIES / "r00_analytical_references"
ALLOWED_IN_R = {"numpy", "scipy", "__future__"} | set(sys.stdlib_module_names)
LIBRARIES = {"axisforge", "slippy", "slippy_bridge"}
R_VERIFIED = {   # r case -> {library function it may import: date José verified it} (rule 13)
    "r02_hertz_contact": {"slippy.contact.hertz.hertz_full": "2026-10-27"},   # against Johnson / Hamrock-Brewe
}

# cases sit in the r / v / s level folders; studies/outputs holds plot and report models, not cases
CASES = sorted(p.parent for p in STUDIES.glob("*/*/__init__.py") if p.parents[1].name != "outputs")


def _imports(path: pathlib.Path):
    """(top-level module name or None, relative level, line, dotted names) of every import in a file; the
    dotted names are ``pkg.mod`` for ``import``, ``pkg.mod.name`` for ``from``."""
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0], 0, node.lineno, (alias.name,)
        elif isinstance(node, ast.ImportFrom):
            dotted = tuple(f"{node.module}.{alias.name}" for alias in node.names)
            yield (node.module or "").split(".")[0] or None, node.level, node.lineno, dotted


def _r_modules():
    """Every module of an r case, its tests excluded."""
    return sorted(p for p in R_LEVEL.glob("r*/**/*.py") if "tests" not in p.relative_to(R_LEVEL).parts)


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_case_package_imports(case):
    module = importlib.import_module(".".join(case.relative_to(ROOT).parts))
    assert module.__all__, f"{case.name}: the public API is declared in __all__"


@pytest.mark.parametrize("path", _r_modules(), ids=lambda p: p.relative_to(R_LEVEL).as_posix())
def test_r_module_imports_no_study(path):
    bad = [f"line {line}: {name or '.' * level}" for name, level, line, _ in _imports(path)
           if name == "studies" or level > 1]
    assert not bad, f"{path.name} imports from studies/ or outside its case: {bad}"


@pytest.mark.parametrize("path", _r_modules(), ids=lambda p: p.relative_to(R_LEVEL).as_posix())
def test_r_module_imports_only_numpy_scipy_stdlib_and_verified_functions(path):
    verified = R_VERIFIED.get(path.relative_to(R_LEVEL).parts[0], {})
    bad = [f"line {line}: {', '.join(dotted)}" for name, level, line, dotted in _imports(path)
           if level == 0 and name not in ALLOWED_IN_R and name != "studies"
           and not (name in LIBRARIES and all(d in verified for d in dotted))]
    assert not bad, (f"{path.name} imports {bad}: an r case writes the analytical formula itself, or José "
                     f"verifies the library function and it is added to R_VERIFIED with the date")


def test_the_bridge_imports_nothing_from_studies():
    bad = [f"{p.relative_to(ROOT).as_posix()}:{line}" for p in sorted((ROOT / "slippy_bridge").rglob("*.py"))
           for name, _, line, _ in _imports(p) if name == "studies"]
    assert not bad

