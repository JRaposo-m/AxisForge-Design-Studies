"""The house style sheet and the figure functions (smoke tests from tiny synthetic inputs).

A smoke test checks the form, not the physics: a ``Figure`` comes back (or the files are written),
no axes carries a title and no grid is drawn. Each test closes what it opened.
"""
import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import to_hex

from validation._support.style import STYLE_PATH


@pytest.fixture(autouse=True)
def house_style():
    """Every test draws in the house style and leaves no figure open."""
    with plt.style.context(STYLE_PATH):
        yield
    plt.close("all")


def test_style_sheet_sets_the_house_style():
    """The sheet loads and sets the paper look: no grid, inward ticks, black first and for bare lines."""
    rc = plt.rcParams
    assert rc["axes.grid"] is False
    assert rc["xtick.direction"] == "in" and rc["ytick.direction"] == "in"
    assert to_hex(rc["axes.prop_cycle"].by_key()["color"][0]) == "#000000"
    assert to_hex(rc["lines.color"]) == "#000000"
