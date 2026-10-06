"""
validation/_support/style.py -- the report style of the validation cases.

The style sheet itself is ``style/axisforge.mplstyle`` at the repository
root, so that studies and examples can use the same file directly
(``plt.style.use(<path>)``) without importing anything from
``validation/``. This module only locates it and holds the two colour
constants the validation figure functions need.

Conventions (enforced by the style file and by the figure modules that use
it): ISO 80000-1 axis labels (quantity symbol in italics, unit after a
solidus, e.g. ``$v$ / mm``); series identity never carried by colour alone.

Notes
-----
``STYLE_PATH`` is resolved from this file's location, so it is valid for a
source checkout or an editable install (``pip install -e .``), which is how
this repository is meant to be used.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

STYLE_PATH: Path = Path(__file__).resolve().parents[2] / "style" / "axisforge.mplstyle"
"""Absolute path of the repository's matplotlib style sheet."""

INK = "#1f1f1e"
"""Main line / text colour (near black)."""

MUTED = "#8a8a84"
"""Secondary colour for supports, reference lines and annotations."""


def use_style() -> None:
    """Activate the repository report style for the current session.

    Examples
    --------
    >>> STYLE_PATH.is_file()
    True
    >>> use_style()
    """
    plt.style.use(STYLE_PATH)
