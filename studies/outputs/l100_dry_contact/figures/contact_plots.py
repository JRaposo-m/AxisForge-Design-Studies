"""Figures of the ``validation/contact`` cases: convergence of a sweep and pressure cuts.

Appearance comes from ``style/axisforge.mplstyle``, activated through ``validation._support.style``.

The functions only pass data: no colours, fonts or sizes are set here. A series is identified by
its line style and marker as well as by its colour.
"""
from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")                      # write files only, no window
import matplotlib.pyplot as plt
import numpy as np

from slippy_bridge.contact import hertz
from validation._support.style import use_style

__all__ = ["convergence", "pressure_cuts"]

use_style()

_LINES = ("-", "--")                       # one per build
_MARKERS = ("o", "s", "^", "D")            # one per quantity


def convergence(records: list[dict], out_png) -> None:
    """Relative error against Hertz along the mesh sweep and along the domain sweep.

    Parameters
    ----------
    records : list of dict
        From ``validation._support.convergence.sweep``: ``build``, ``sweep``, ``span_x``, ``h_m`` and the errors
        ``err_p0``, ``err_s_x``, ``err_s_y``, ``err_delta`` (fractions).
    out_png : path
        Output file; its folder must exist.

    Notes
    -----
    Left: ``|error|`` of the maximum pressure, the two semi-axes and the approach against the grid
    spacing (logarithmic axes), one line style per build. A quantity that has no finite Hertz value
    (line contact: ``s_y``, approach) is left out. Right: signed error of the maximum pressure
    against the half-length of the domain along rolling.
    """
    fig, (ax_h, ax_d) = plt.subplots(1, 2)
    quantities = (("err_p0", "$p_0$"), ("err_s_x", "$s_x$"), ("err_s_y", "$s_y$"), ("err_delta", r"$\delta$"))
    for build, ls in zip(dict.fromkeys(r["build"] for r in records), _LINES):
        mesh = [r for r in records if r["build"] == build and r["sweep"] == "mesh"]
        domain = [r for r in records if r["build"] == build and r["sweep"] == "domain"]
        for (key, symbol), marker, colour in zip(quantities, _MARKERS, plt.rcParams["axes.prop_cycle"]):
            if all(math.isfinite(r[key]) for r in mesh):
                ax_h.loglog([r["h_m"] * 1e6 for r in mesh], [abs(r[key]) * 100 for r in mesh],
                            ls=ls, marker=marker, label=f"{symbol}, {build}", **colour)
        ax_d.plot([r["span_x"] for r in domain], [r["err_p0"] * 100 for r in domain],
                  ls=ls, marker=_MARKERS[0], label=build)
    ax_h.set_xlabel(r"$h$ / $\mu$m")
    ax_h.set_ylabel(r"$|\varepsilon|$ / %")
    ax_h.legend()
    ax_d.set_xlabel("domain half-length / $s_x$")
    ax_d.set_ylabel(r"$\varepsilon_{p_0}$ / %")
    ax_d.legend()
    fig.savefig(out_png)
    plt.close(fig)


def pressure_cuts(p: np.ndarray, grid, case, out_png) -> None:
    """Pressure through the contact centre, along and across rolling, against Hertz.

    Parameters
    ----------
    p : ndarray
        Pressure [Pa] on ``grid`` (rolling along axis 0), from ``static_step.run``.
    grid : slippy_bridge.contact.Grid
        The grid of the solution; its centre node ``(n_x//2, n_y//2)`` is the contact centre.
    case : ContactCase
        The contact, for the Hertz pressure ``p0 sqrt(1 - (x/s_x)^2 - (y/s_y)^2)``.
    out_png : path
        Output file; its folder must exist.
    """
    i0, j0 = grid.X.shape[0] // 2, grid.X.shape[1] // 2
    reference = hertz(case)["pressure"]
    fig, (ax_x, ax_y) = plt.subplots(1, 2, sharey=True)
    for ax, coord, cut, symbol in ((ax_x, grid.X[:, j0], p[:, j0], "x"), (ax_y, grid.Y[i0, :], p[i0, :], "y")):
        fine = np.linspace(coord[0], coord[-1], 801)
        zero = np.zeros_like(fine)             # the other coordinate: the cut passes through the centre
        hertz_cut = reference(fine, zero) if symbol == "x" else reference(zero, fine)
        ax.plot(fine * 1e3, hertz_cut * 1e-6, ls="-", label="Hertz")
        ax.plot(coord * 1e3, cut * 1e-6, ls="none", marker="o", label="SlipPY")
        ax.set_xlabel(f"${symbol}$ / mm")
    ax_x.set_ylabel("$p$ / MPa")
    ax_x.legend()
    fig.savefig(out_png)
    plt.close(fig)
