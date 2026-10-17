"""
validation/_support/style.py -- the report style of the validation cases.

The style sheet itself is ``style/axisforge.mplstyle`` at the repository
root, so that studies and examples can use the same file directly
(``plt.style.use(<path>)``) without importing anything from
``validation/``. This module only locates it and activates it.

Conventions (``style/README.md``): numerical results as filled markers,
formulas as lines, published values as open markers; labels "symbol (unit)"
(``$h$ ($\\mu$m)``) or a bare ratio (``$p/p_0$``); series identity never
carried by colour alone; no titles. A case script calls ``use_style()`` once,
at the top, before it draws.

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


def use_style() -> None:
    """Activate the repository report style for the current session.

    Examples
    --------
    >>> STYLE_PATH.is_file()
    True
    >>> use_style()
    """
    plt.style.use(STYLE_PATH)
