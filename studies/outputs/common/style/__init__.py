"""The paper style of every figure of the studies (``paper.mplstyle``, accepted by Jose 2026-10-17)."""
from pathlib import Path

__all__ = ["PAPER"]

PAPER = Path(__file__).with_name("paper.mplstyle")      # matplotlib.pyplot.style.context(PAPER)
