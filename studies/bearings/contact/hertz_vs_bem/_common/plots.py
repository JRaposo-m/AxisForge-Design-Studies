"""Diagnostic figure of one run (one build, one resolution) of ``hertz_vs_bem``.

Three panels: the surface SlipPy built (its own ``show``), the numerical pressure field with
the Hertz contact (ellipse, or the two edges of the strip for a line contact), and the pressure
cuts through the apex against the Hertz profile. The array may be rectangular.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                      # write files only, no window
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_run"]


def plot_run(out_png, surface, p: np.ndarray, h: float, ref: dict, title: str = "") -> None:
    """Save the diagnostic figure of one run.

    Parameters
    ----------
    out_png : str or pathlib.Path
        Output file (folder created if missing).
    surface : slippy surface
        The discrete surface of the build; its ``show('profile', 'surface', ax)`` is used if present.
    p : numpy.ndarray
        Numerical pressure [Pa] on the grid (rows x columns).
    h : float
        Grid spacing [m].
    ref : dict
        ``hertz_full`` results: ``contact_radii`` [m] (one may be infinite for a line contact)
        and ``max_pressure`` [Pa].
    title : str, optional
        Figure title.

    Notes
    -----
    The Hertz profile along a direction is ``p0 sqrt(1 - (s/c)^2)``. The semi-axis ``c`` of each
    direction is taken from the measured half-extent (the direction where the loaded nodes extend
    further gets the larger Hertz semi-axis; a direction that the load spans completely gets an
    infinite one), because the orientation of x and y in the arrays is not assumed. The apex is the
    node ``n/2`` of each axis (even n).
    """
    ny, nx = p.shape
    s_r = (np.arange(ny) - ny / 2.0) * h              # coordinate along the rows
    s_c = (np.arange(nx) - nx / 2.0) * h              # coordinate along the columns
    radii = sorted((float(r) for r in ref["contact_radii"]), reverse=True)
    finite = [r for r in radii if math.isfinite(r)]
    p0 = float(ref["max_pressure"])

    mask = p > 1e-6 * p.max()
    rows, cols = np.where(mask)
    half_r, half_c = (rows.max() - rows.min() + 1) * h / 2, (cols.max() - cols.min() + 1) * h / 2
    span_r, span_c = len(np.unique(rows)) == ny, len(np.unique(cols)) == nx
    if len(finite) == 1:                              # line contact: the strip axis has no edge
        c_col = math.inf if span_c else finite[0]       # loaded along every column: the line axis
        c_row = math.inf if span_r else finite[0]
    else:
        c_row, c_col = (finite[0], finite[1]) if half_r >= half_c else (finite[1], finite[0])
    a_plot = finite[0]

    fig = plt.figure(figsize=(15, 4.5))
    ax0 = fig.add_subplot(131, projection="3d")
    if hasattr(surface, "show"):
        surface.show("profile", "surface", ax0)
    ax0.set_title("Surface (SlipPy)")

    ax1 = fig.add_subplot(132)
    ext = [s_c[0] * 1e3, s_c[-1] * 1e3, s_r[0] * 1e3, s_r[-1] * 1e3]
    im = ax1.imshow(p * 1e-6, extent=ext, origin="lower", aspect="auto" if ny != nx else "equal")
    fig.colorbar(im, ax=ax1, label="pressure [MPa]")
    if math.isfinite(c_row) and math.isfinite(c_col):
        t = np.linspace(0, 2 * math.pi, 200)
        ax1.plot(c_col * np.cos(t) * 1e3, c_row * np.sin(t) * 1e3, "w--", lw=1, label="Hertz ellipse")
    else:
        edge = c_col if math.isfinite(c_col) else c_row
        for sgn in (-1, 1):
            if math.isfinite(c_col):
                ax1.axvline(sgn * edge * 1e3, color="w", ls="--", lw=1, label="Hertz edge" if sgn < 0 else None)
            else:
                ax1.axhline(sgn * edge * 1e3, color="w", ls="--", lw=1, label="Hertz edge" if sgn < 0 else None)
    ax1.set_xlabel("columns [mm]"); ax1.set_ylabel("rows [mm]"); ax1.legend(loc="upper right")

    ax2 = fig.add_subplot(133)
    for cut, s_axis, c_h, name in ((p[ny // 2, :], s_c, c_col, "along columns"),
                                   (p[:, nx // 2], s_r, c_row, "along rows")):
        ax2.plot(s_axis * 1e3, cut * 1e-6, label=f"numerical, {name}")
        if math.isfinite(c_h):
            x = np.linspace(-c_h, c_h, 400)
            ax2.plot(x * 1e3, p0 * np.sqrt(1 - (x / c_h) ** 2) * 1e-6, "k--", lw=1)
    ax2.plot([], [], "k--", label="Hertz")
    ax2.set_xlim(-1.5 * a_plot * 1e3, 1.5 * a_plot * 1e3)
    ax2.set_xlabel("distance from apex [mm]"); ax2.set_ylabel("pressure [MPa]"); ax2.legend()

    fig.suptitle(title)
    fig.tight_layout()
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
