"""
Every shared module imports cleanly, and every case/study ``construction.py``
loads.

Catches broken paths after a move. Notebook scripts (``0*_*.py``) are not
imported: they run the case at import time.
"""
from __future__ import annotations

import importlib
import pathlib
import pkgutil

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGES = ["references", "validation._support"]


def _modules():
    for pkg in PACKAGES:
        mod = importlib.import_module(pkg)
        yield pkg
        for info in pkgutil.walk_packages(mod.__path__, prefix=f"{pkg}."):
            yield info.name


@pytest.mark.parametrize("name", sorted(set(_modules())))
def test_import(name):
    importlib.import_module(name)


@pytest.mark.parametrize("case", ["fem_uniform_radial_load", "fem_stepped_distributed_gear"])
def test_case_construction_loads(case, load_case):
    assert callable(load_case(case).build_system)
