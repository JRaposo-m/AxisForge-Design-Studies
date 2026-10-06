"""
Shared test helpers.

Validation case folders are not packages, and every case has a module called
``construction``. ``load_case`` imports a case's ``construction.py`` from its
path under a unique module name, so two cases never collide in
``sys.modules``.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

CASE_DIRS = {
    "fem_uniform_radial_load": ROOT / "validation" / "shafts" / "fem_uniform_radial_load",
    "fem_stepped_distributed_gear": ROOT / "validation" / "shafts" / "fem_stepped_distributed_gear",
}


def _load(case: str):
    name = f"_case_{case}_construction"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, CASE_DIRS[case] / "construction.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def load_case():
    """Return a function ``case name -> construction module``."""
    return _load
