"""
validation._support.figures -- Pure figure functions: data in, ``matplotlib.figure.Figure`` out.

The layout helpers shared by the figure modules live here (no new module for a few functions):
``_words`` writes code identifiers as words, ``_note`` the one information line under a figure,
``_tag`` the panel tag above a frame, ``_supports`` named dotted verticals (bearings, gears, path
points), ``_stacked`` panels stacked on a shared abscissa.
"""
from __future__ import annotations

import re
from typing import Sequence

import matplotlib.pyplot as plt

PANEL_HEIGHT = 0.4      # height of one stacked panel, as a fraction of the sheet's figure height [-]


def _words(text: str) -> str:
    """Code identifiers as words: underscores outside ``$...$`` mathtext become spaces.

    Inside mathtext an underscore is a subscript and is kept (``$Q_\\mathrm{max}$``); outside it,
    ``radial_heavy`` reads "radial heavy", ``6210_A`` "6210 A".
    """
    parts = text.split("$")
    return "$".join(part.replace("_", " ") if k % 2 == 0 else part for k, part in enumerate(parts))


def _wrap(text: str, width: int) -> str:
    """Break ``text`` into lines of about ``width`` visible characters, at spaces outside ``$...$``.

    Matplotlib's own ``wrap=True`` breaks inside mathtext too, and a broken ``$...$`` group prints
    as raw LaTeX; here a math group is never split. Its visible length is estimated without the
    ``$``, backslash commands, braces and sub/superscript marks.
    """
    lines = []
    for paragraph in text.split("\n"):
        words, in_math, current = [], False, ""
        for ch in paragraph:                               # split at spaces outside math only
            if ch == "$":
                in_math = not in_math
            if ch == " " and not in_math:
                words.append(current)
                current = ""
            else:
                current += ch
        words.append(current)
        line, size = [], 0
        for word in words:
            visible = len(re.sub(r"\\[a-zA-Z]+|[${}^_]", "", word))
            if line and size + 1 + visible > width:
                lines.append(" ".join(line))
                line, size = [], 0
            line.append(word)
            size += visible + (1 if size else 0)
        lines.append(" ".join(line))
    return "\n".join(lines)


def _note(fig, text: str | None) -> None:
    """The figure's information line (what a title used to carry), centred under the axes.

    Placed below the figure (``y = 0``, top-aligned); the sheet saves with ``bbox_inches="tight"``,
    which takes it in. Code identifiers in it are written as words (``_words``); long notes are
    broken at spaces outside mathtext, to the figure width (``_wrap``).
    """
    if text:
        # about one character per half font size: the visible width of a serif text line
        width = int(fig.get_figwidth() * 72.0 / (0.5 * plt.rcParams["font.size"]))
        fig.text(0.5, 0.0, _wrap(_words(text), width), ha="center", va="top")


def _tag(ax, tag: str) -> None:
    """Panel tag just above the top-left corner of the frame, off the data (identifiers as words)."""
    ax.text(0.0, 1.02, _words(tag), transform=ax.transAxes, ha="left", va="bottom")


def _supports(ax, positions: Sequence[float], names: Sequence[str]) -> None:
    """Dotted verticals at the given positions; names just above the frame, off the data."""
    for x0, name in zip(positions, names):
        ax.axvline(x0, linestyle=":", linewidth=plt.rcParams["axes.linewidth"], zorder=0)
        ax.annotate(name, xy=(x0, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, 2), textcoords="offset points", ha="center", va="bottom")


def _stacked(n: int):
    """``n`` panels stacked on a shared abscissa, tagged (a), (b), ... above each frame when more than one.

    Sheet width; ``PANEL_HEIGHT`` sheet heights per panel, never less than one sheet height. Tick
    labels never carry an offset ("x10^n" or "+1e3" above the axis): values are read as printed.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray of Axes
        Top to bottom.
    """
    width, height = plt.rcParams["figure.figsize"]
    fig, axes = plt.subplots(n, 1, sharex=True, squeeze=False,
                             figsize=(width, max(height, n * PANEL_HEIGHT * height)))
    axes = axes[:, 0]
    for k, ax in enumerate(axes):
        ax.ticklabel_format(useOffset=False, style="plain")   # neither "+1e3" nor "x10^n"
        if n > 1:
            _tag(ax, f"({'abcdefgh'[k]})")
    return fig, axes
