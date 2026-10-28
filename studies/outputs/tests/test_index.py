"""Every public plot or report function of studies/outputs is in the README index (amendment item 10)."""
import importlib
import pkgutil
from pathlib import Path

import studies.outputs as outputs

README = Path(outputs.__file__).with_name("README.md").read_text(encoding="utf-8")


def _model_modules():
    for info in pkgutil.walk_packages(outputs.__path__, prefix="studies.outputs."):
        if ".plots." in info.name or ".reports." in info.name:
            yield importlib.import_module(info.name)


def test_every_public_function_is_in_the_index():
    modules = list(_model_modules())
    assert modules, "no plot or report module found"
    for module in modules:
        file = module.__name__.removeprefix("studies.outputs.").replace(".", "/") + ".py"
        for name in module.__all__:
            assert f"| `{name}` | `{file}` |" in README, f"{file}::{name} missing from studies/outputs/README.md"


def test_the_style_sheet_exists():
    from studies.outputs.common.style import PAPER
    assert PAPER.is_file()
