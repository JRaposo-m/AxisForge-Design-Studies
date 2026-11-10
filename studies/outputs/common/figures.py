"""File naming and saving shared by every plot module (D-0036 amendment item 10).

A figure is written as ``<case>_<module>_<function>[_<variant>].png`` into the calling case's ``results/``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

__all__ = ["MARKERS", "figure_path", "save"]

MARKERS = "osD^v<>"     # marker cycle of the plot modules (colour never carries identity alone)


def figure_path(results_dir, case: str, module: str, function: str, variant: str = "") -> Path:
    """``results_dir/<case>_<module>_<function>[_<variant>].png``, the folder created."""
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{case}_{module}_{function}{'_' + variant if variant else ''}.png"


def save(fig, path: Path, caption: str = "") -> Path:
    """Write ``fig`` as PNG to ``path`` (an optional caption line under the axes), close it, return ``path``."""
    if caption:
        fig.supxlabel(caption, fontsize=7)
    fig.savefig(path, format="png")
    plt.close(fig)
    return path
