"""
Every case package imports, and the layer law holds where an import test can see it (D-0036).

- A case package is ``studies/<level>/<case>/`` with an ``__init__.py``; it is imported through
  its public API. Case folders without ``__init__.py`` (not migrated yet) are not imported.
- An r module imports only NumPy, SciPy and the standard library (AST scan; relative imports only
  inside the case): no AxisForge, SlipPY, bridge or other case. The case tests may import the case package itself.
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

CASES = sorted(p.parent for p in STUDIES.glob("*/*/__init__.py"))


def _imports(path: pathlib.Path):
    """(top-level module name or None, relative level, line) of every import in a file."""
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0], 0, node.lineno
        elif isinstance(node, ast.ImportFrom):
            yield (node.module or "").split(".")[0] or None, node.level, node.lineno


def _r_modules():
    """Every module of an r case, its tests excluded."""
    return sorted(p for p in R_LEVEL.glob("r*/**/*.py") if "tests" not in p.relative_to(R_LEVEL).parts)


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_case_package_imports(case):
    module = importlib.import_module(".".join(case.relative_to(ROOT).parts))
    assert module.__all__, f"{case.name}: the public API is declared in __all__"


@pytest.mark.parametrize("path", _r_modules(), ids=lambda p: p.relative_to(R_LEVEL).as_posix())
def test_r_module_imports_only_numpy_scipy_stdlib(path):
    bad = [f"line {line}: {name or '.' * level}" for name, level, line in _imports(path)
           if (level == 0 and name not in ALLOWED_IN_R) or level > 1]
    assert not bad, f"{path.name} imports outside its case: {bad}"


def test_the_bridge_imports_nothing_from_studies():
    bad = [f"{p.relative_to(ROOT).as_posix()}:{line}" for p in sorted((ROOT / "slippy_bridge").rglob("*.py"))
           for name, _, line in _imports(p) if name == "studies"]
    assert not bad

