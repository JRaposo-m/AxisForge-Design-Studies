"""Convergence views of the numerical solvers: the Newton residual history of many solves on one plot.

Every function takes records (``list[dict]``) as ``numerical_solvers`` writes them, draws with the paper style and
writes one PNG to ``results_dir`` named ``<case>_convergence_<function>[_<variant>].png``; it returns that path.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from ..figures import MARKERS, figure_path, save
from ..style import PAPER

__all__ = ["newton_history"]


def newton_history(records: list[dict], results_dir, case: str, *, tol: float | None = None, variant: str = "",
                   caption: str = "", x_label: str = r"Newton iteration $k$ [-]",
                   y_label: str = r"$\|F / f_\mathrm{scale}\|$ [-]") -> Path:
    """Scaled residual norm against the Newton iteration, one line per solve, with the stopping tolerance.

    Shows: ``residual_norm`` (``||F / f_scale||``) [-] on a log axis against the iteration ``k`` [-], one colour
    and marker per ``label`` (a solve), filled markers joined by thin lines; the stopping tolerance ``tol`` as a
    dashed line. Quadratic convergence shows as a curve whose log value doubles per step at the end; a stall or a
    line-search cut shows as a flat run.

    Parameters
    ----------
    records : list of dict
        ``NewtonResult.records()`` of the solves, concatenated: fields ``label`` (str), ``k`` [-],
        ``residual_norm`` [-].
    results_dir, case : path, str
        Folder of the figure and the case id.
    tol : float, optional
        ``NewtonOptions.residual_abs`` of the solves [-].
    variant, caption : str, optional
        Last part of the file name; caption line under the axes.
    x_label, y_label : str, optional
        Axis labels; give them when the history is not a Newton residual (s004: fixed-point passes, max |delta
        psi| [rad]).

    Returns
    -------
    Path
        ``<case>_convergence_newton_history[_<variant>].png``. Called by s102, v005, s004, s003.
    """
    labels = list(dict.fromkeys(r["label"] for r in records))
    with plt.style.context(PAPER):
        fig, ax = plt.subplots(figsize=(5.0, 3.4))
        colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for i, lab in enumerate(labels):
            rows = sorted((r for r in records if r["label"] == lab), key=lambda r: r["k"])
            ax.plot([r["k"] for r in rows], [max(r["residual_norm"], 1e-18) for r in rows],
                    marker=MARKERS[i % len(MARKERS)], markersize=3, linewidth=0.5,
                    color=colours[i % len(colours)], label=lab)
        if tol is not None:
            ax.axhline(tol, color="k", linestyle="--", linewidth=0.6, label=f"tolerance {tol:g}")
        ax.set_yscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend(fontsize=5, ncol=1, loc="lower left")
        return save(fig, figure_path(results_dir, case, "convergence", "newton_history", variant), caption)
