"""
validation._support.figures -- Pure figure functions: data in, ``matplotlib.figure.Figure`` out.

The two layout helpers shared by the figure modules live here (no new module for two functions):
``_note`` writes the one information line under a figure, ``_stacked`` makes panels stacked on a
shared abscissa.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

PANEL_HEIGHT = 0.4      # height of one stacked panel, as a fraction of the sheet's figure height [-]


def _note(fig, text: str | None) -> None:
    """The figure's information line (what a title used to carry), centred under the axes.

    Placed below the figure (``y = 0``, top-aligned); the sheet saves with ``bbox_inches="tight"``,
    which takes it in.
    """
    if text:
        fig.text(0.5, 0.0, text, ha="center", va="top", wrap=True)


def _stacked(n: int):
    """``n`` panels stacked on a shared abscissa, tagged (a), (b), ... when more than one.

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
        ax.ticklabel_format(useOffset=False)
        if n > 1:
            ax.text(0.02, 0.92, f"({'abcdefgh'[k]})", transform=ax.transAxes, va="top")
    return fig, axes
