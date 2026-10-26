"""
Regression of every (question, type): re-run it and compare with its stored results/iso16281_rows.csv.

Skipped when AxisForge is not installed or when a type has no stored data yet. The first run of
a type (``python run.py``) writes the baseline; after that, any change in the numbers is a
failure until the baseline is deliberately re-generated.
"""
from __future__ import annotations

import importlib
import math
from pathlib import Path

import pytest

FAMILY_DIR = Path(__file__).resolve().parents[1]

# (question folder, type folder, study class, spec factory)
CASES = [
    ("clearance_and_alpha", "deep_groove", "ClearanceAndAlphaStudy", "DeepGrooveSpec", {}),
    ("clearance_and_alpha", "cylindrical_roller", "ClearanceAndAlphaStudy",
     "CylindricalRollerSpec", {}),
    ("clearance_and_alpha", "angular_contact", "ClearanceAndAlphaStudy", "AngularContactSpec", {}),
    ("misalignment", "deep_groove", "MisalignmentStudy", "DeepGrooveSpec", {}),
    ("misalignment", "cylindrical_roller", "MisalignmentStudy", "CylindricalRollerSpec", {}),
]
RTOL = 1e-9


@pytest.mark.parametrize("question, type_name, study_name, spec_name, spec_kwargs", CASES,
                         ids=[f"{q}/{t}" for q, t, *_ in CASES])
def test_matches_the_stored_data(question, type_name, study_name, spec_name, spec_kwargs):
    pytest.importorskip("axisforge")
    from _common import bearings
    from _common.store import read_results

    data_dir = FAMILY_DIR / question / type_name / "results"
    if not (data_dir / "iso16281_rows.csv").exists():
        pytest.skip(f"no baseline yet: run {question}/{type_name}/run.py")
    stored = read_results(data_dir)

    study_cls = getattr(importlib.import_module(f"{question}.study"), study_name)
    spec = getattr(bearings, spec_name)(**spec_kwargs)
    fresh = study_cls(spec).run()

    assert fresh.meta["parameter_hash"] == stored.meta["parameter_hash"], (
        "the study definition changed: re-generate the baseline on purpose")
    assert len(fresh.rows) == len(stored.rows)
    for new, old in zip(fresh.rows, stored.rows):
        for key, value in old.items():
            current = new[key]
            if isinstance(value, float):
                if math.isnan(value):
                    assert math.isnan(current), (key, new["label"])
                else:
                    assert current == pytest.approx(value, rel=RTOL, abs=1e-12), (key, new["label"])
            else:
                assert current == value, (key, new["label"])
