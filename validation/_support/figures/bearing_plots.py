"""Figures of the rolling bearing cases: element loads, contact ellipses around the bearing, A2 sign check.

Data in, ``matplotlib.figure.Figure`` out; appearance only from ``style/axisforge.mplstyle``, activated
by the caller (``validation._support.style.use_style()``). AxisForge results are numerical: filled
markers, no connecting line.

Frames and symbols (AxisForge ISO/TS 16281 solvers,
``axisforge/outputs/solvers/bearings/load_distribution/load_distribution_results.py:156-190``)
-------------------------------------------------------------------------------------------------
* The outer ring is fixed; the inner ring is displaced by ``delta_r`` toward the element at
  ``phi_j = 0`` (``U_j = A cos a0 + delta_r cos phi_j``,
  ``axisforge/solvers/machine_elements/bearings/load_distribution/iso_16281/contact_solver.py:655``).
* ``phi_j`` is the LOCAL azimuth of element ``j``, measured from the load line (0 = the element
  under the load), wrapped to [-pi, pi). The global azimuth ``phi_j_global`` (D-0010: from +Z toward
  +Y) adds the load direction ``phi_Fr``; where the global ``phi_j = 0`` points is under review
  (A2, R-003: a 180 deg offset).
* ``psi`` is the inner-ring tilt and ``psi_j`` the roller tilt per element (Eq. (39)). The azimuth
  is never written ``psi_j`` here, although ISO/TS 16281 and Harris use psi for it.
* ``delta_j`` is the total elastic approach of the element (both contacts); for balls it is
  floored at 0, for rollers it is the raw approach and may be negative (clearance gap).
* ``alpha_j``: operating contact angle for balls; for rollers ``alpha_0`` repeated, not a result.
* Contact semi-axes use the bridge names: ``s_x`` along rolling, ``s_y`` across it (along the
  groove). On a ball-raceway contact ``s_x`` is the minor semi-axis (Harris ``b``), ``s_y`` the
  major (Harris ``a``).
"""
from __future__ import annotations

import math
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from axisforge.outputs.solvers.bearings.load_distribution.load_distribution_results import wrap_angle
from matplotlib.figure import Figure

from validation._support.figures import _note, _stacked, _words

__all__ = ["A2_NOTE", "SIGN_CHECK_NOTE", "contact_ellipses", "element_loads", "sign_check"]

A2_NOTE = r"A2: global direction of $\varphi_j = 0$ under review (R-003); $\varphi_j$ local, 0 on the load line"
"""Stamp of every figure with an azimuth axis, until José rules on A2 (one place to change; the
load distribution study plots carry a copy, ``studies/bearings/load_distribution/_common/plots.py``)."""

SIGN_CHECK_NOTE = (r"$\theta$: direction of the radial load on the shaft (AxisForge RadialLoad, from +Y toward +Z); "
                   r"$\varphi_{F_r}$: load direction used by the solver; $\varphi_\mathrm{exp}$: direction of the "
                   r"shaft load on the inner ring; azimuth from +Z toward +Y (D-0010). One ball always sits on "
                   r"$\varphi_{F_r}$ by construction: the evidence (R-003) is the offset between the two directions "
                   r"and the residual $|r|/|F_r|$ of the inner-ring equilibrium.")
"""Stamp of the sign check figure: it is the evidence, not a figure under the A2 caveat."""

_ALPHA_CONSTANT_DEG = 0.01   # alpha_j spread below which it is reported as one value: the solver's
                             # round-off is ~1e-10 deg, a physical drift (axial load) is degrees


def sign_check(phi_global_rad: Sequence, Q_N: Sequence, *, phi_Fr_rad: Sequence[float],
               phi_expected_rad: Sequence[float], series: Sequence[str], labels: Sequence[str] = (),
               note: str | None = SIGN_CHECK_NOTE) -> Figure:
    """Element loads against the GLOBAL azimuth, with the load direction used and the expected one.

    The evidence for the A2 ruling (R-003): if the shaft -> bearing path is right, the loaded zone
    is centred on the direction in which the shaft pushes the inner ring (``phi_expected``); the
    FAIL signature is a zone centred 180 deg away, with an equilibrium residual of twice the load.

    Parameters
    ----------
    phi_global_rad : sequence of sequence of array_like
        Per panel (load direction), per series (bearing): global azimuth of each element [rad],
        D-0010 frame (from +Z toward +Y).
    Q_N : sequence of sequence of array_like
        Element loads [N], aligned with ``phi_global_rad``.
    phi_Fr_rad : sequence of float
        Per panel, the load direction the solver used, ``atan2(Fr_xy, Fr_xz)`` [rad].
    phi_expected_rad : sequence of float
        Per panel, the direction of the shaft load on the inner ring, same frame [rad].
    series : sequence of str
        Name of each series (bearing), drawn as direct labels; markers 'o' (filled) then 'x'.
    labels : sequence of str, optional
        Per panel, a short text after the panel tag (load direction, residual).
    note : str, optional
        Line under the figure; default ``SIGN_CHECK_NOTE``.

    Returns
    -------
    matplotlib.figure.Figure
        One panel per load direction, stacked on a shared azimuth 0..360 deg. The caller closes it.
    """
    deg = math.degrees(1.0)                               # [rad] -> [deg], plot unit of the azimuth
    fig, axes = _stacked(len(phi_global_rad))
    for i, ax in enumerate(axes):
        top = 0.0
        for k, (phi_rad, q) in enumerate(zip(phi_global_rad[i], Q_N[i])):
            phi = (np.asarray(phi_rad, dtype=float) % (2.0 * math.pi)) * deg
            Q = np.asarray(q, dtype=float)
            ax.plot(phi, Q, linestyle="none", marker="ox"[k % 2])
            # name at the least loaded element of the zone (low in the panel, clear of the direction
            # labels in the top band): first series to its left, second to its right
            j = int(np.argmin(np.where(Q > 0.0, Q, np.inf))) if (Q > 0.0).any() else int(np.argmax(Q))
            ax.annotate(_words(series[k]), (phi[j], Q[j]), xytext=(-6 if k % 2 == 0 else 6, 0), textcoords="offset points",
                        ha="right" if k % 2 == 0 else "left", va="center")
            top = max(top, Q.max())
        ax.set_ylim(top=1.3 * top)                         # free band at the top for the labels
        # the two directions: used by the solver (dashed) and expected (dotted), labelled in the free band
        for angle, linestyle, text in ((phi_Fr_rad[i], "--", r"$\varphi_{F_r}$"),
                                       (phi_expected_rad[i], ":", r"$\varphi_\mathrm{exp}$")):
            x = (angle % (2.0 * math.pi)) * deg
            ax.axvline(x, linestyle=linestyle, linewidth=plt.rcParams["axes.linewidth"])
            ax.annotate(text, (x, 0.82), xycoords=("data", "axes fraction"), xytext=(4, 0),
                        textcoords="offset points", ha="left", va="center")
        ax.set_ylabel(r"$Q_j$ (N)")
        if i < len(labels):
            # beside the panel tag, above the frame: clear of the direction lines and their labels
            ax.text(0.04, 1.02, _words(labels[i]), transform=ax.transAxes, ha="left", va="bottom")
    # half a tick step beyond 0 and 360 deg, so that a direction at 0 deg is not hidden by the frame
    axes[-1].set_xlim(-22.5, 382.5)
    axes[-1].set_xticks(np.arange(0.0, 361.0, 45.0))
    axes[-1].set_xlabel(r"global azimuth $\varphi_j$ (deg)")
    _note(fig, note)
    return fig


def element_loads(phi_local_rad, Q_N, *, delta_mm=None, alpha_rad=None, note: str | None = A2_NOTE) -> Figure:
    """Element load, approach and contact angle around the bearing (stacked panels on ``phi_j``).

    Shows the shape and extent of the loaded zone, the drift of the contact angle across it and,
    for rollers, the clearance gap as a negative approach.

    Parameters
    ----------
    phi_local_rad : array_like
        Local azimuth of each element [rad], 0 on the load line (``element_records`` ``phi_rad``).
        Not the global azimuth (R-003).
    Q_N : array_like
        Element load [N].
    delta_mm : array_like, optional
        Total elastic approach of the element [mm]; drawn in um for every element. Balls: floored
        at 0 where unloaded (the model; said in the note). Rollers: the raw approach, may be
        negative. Omitted panel if None (the study tables drop it).
    alpha_rad : array_like, optional
        Operating contact angle [rad], drawn for the loaded elements only (an unloaded ball has
        no contact, its angle is geometric). If it is constant over the loaded zone (spread below
        0.01 deg) the panel is dropped and the value is written in the note. Pass None for rollers
        (``alpha_0`` repeated, not a result).
    note : str, optional
        Line under the figure, after the zone count; default ``A2_NOTE``.

    Returns
    -------
    matplotlib.figure.Figure
        The caller closes it.

    Notes
    -----
    The loaded zone is ``Q_j > 0`` (not ``delta_j > 0``: for rollers the two differ). Its first and
    last loaded element are marked by thin dotted lines in every panel, and the count of loaded
    elements is written in the note line. R-101 concerns ``psi`` or ``Fa`` taken from the shaft
    path; loads given directly to the solver are not affected.
    """
    deg = math.degrees(1.0)                               # [rad] -> [deg]
    phi = wrap_angle(phi_local_rad) * deg                 # [-180, 180), as element_records
    Q = np.asarray(Q_N, dtype=float)
    loaded = Q > 0.0
    texts = [f"loaded {int(loaded.sum())} of {len(Q)} elements (dotted: ends of the loaded zone)"]
    panels = [(phi, Q, r"$Q_j$ (N)")]
    if delta_mm is not None:
        delta = np.asarray(delta_mm, dtype=float)
        panels.append((phi, delta * 1e3, r"$\delta_j$ ($\mu$m)"))   # [mm] -> [um]
        if (~loaded).any() and np.all(delta[~loaded] == 0.0):
            texts.append(r"$\delta_j$ floored at 0 where unloaded (model)")
    if alpha_rad is not None and loaded.any():
        alpha = np.asarray(alpha_rad, dtype=float)[loaded] * deg
        if np.ptp(alpha) < _ALPHA_CONSTANT_DEG:
            texts.append(rf"$\alpha_j$ = {alpha.mean():.3f} deg, constant over the loaded zone")
        else:
            panels.append((phi[loaded], alpha, r"$\alpha_j$ (deg), loaded elements"))
    fig, axes = _stacked(len(panels))
    for ax, (x, y, label) in zip(axes, panels):
        ax.plot(x, y, linestyle="none", marker="o")
        ax.set_ylabel(label)
        if loaded.any():                                  # ends of the loaded zone
            for end in (phi[loaded].min(), phi[loaded].max()):
                ax.axvline(end, linestyle=":", linewidth=plt.rcParams["axes.linewidth"])
    if delta_mm is not None:
        axes[1].axhline(0.0, linewidth=plt.rcParams["axes.linewidth"])   # roller: below = clearance gap
    axes[-1].set_xlim(-180.0, 180.0)
    axes[-1].set_xticks(np.arange(-180.0, 181.0, 45.0))
    axes[-1].set_xlabel(r"$\varphi_j$ (deg)")
    _note(fig, "; ".join(texts + ([note] if note else [])))
    return fig


def contact_ellipses(phi_local_rad, race, p0_Pa, s_x_m, s_y_m, *, note: str | None = A2_NOTE) -> Figure:
    """Hertz pressure and contact semi-axes of every loaded contact, per raceway, around the bearing.

    Shows which raceway is critical (the inner one, with the smaller conformity along rolling) and
    how the ellipse changes across the loaded zone.

    Parameters
    ----------
    phi_local_rad : array_like
        Local azimuth of the element of each contact [rad], 0 on the load line.
    race : array_like of str
        Raceway of each contact (``"inner"``, ``"outer"``).
    p0_Pa : array_like
        Maximum Hertz pressure [Pa]; drawn in MPa.
    s_x_m, s_y_m : array_like
        Semi-axes along rolling and across it [m]; drawn in um.
    note : str, optional
        Line under the figure, after the symbol definitions; default ``A2_NOTE``.

    Returns
    -------
    matplotlib.figure.Figure
        Panels p0, s_x, s_y stacked on ``phi_j``; raceways by marker (and the cycle's colour),
        labelled directly at the edge of the loaded zone in panel (a). The caller closes it.

    Notes
    -----
    The values are those of ``references.contact.hertz.hertz_point`` on the contacts of
    ``slippy_bridge.contact.contacts``, whose raceway radii are taken at the operating angle
    ``alpha_j``; AxisForge evaluates them at ``alpha_0`` (R-019, finding F1). The note says so.
    """
    deg = math.degrees(1.0)                               # [rad] -> [deg]
    phi = wrap_angle(phi_local_rad) * deg                 # the bridge's phi_local_rad is not wrapped
    race = np.asarray(race)
    panels = [(np.asarray(p0_Pa, dtype=float) * 1e-6, r"$p_0$ (MPa)"),           # [Pa] -> [MPa]
              (np.asarray(s_x_m, dtype=float) * 1e6, r"$s_x$ ($\mu$m)"),         # [m] -> [um]
              (np.asarray(s_y_m, dtype=float) * 1e6, r"$s_y$ ($\mu$m)")]
    fig, axes = _stacked(len(panels))
    for k, name in enumerate(dict.fromkeys(race.tolist())):
        on = race == name
        for ax, (y, _) in zip(axes, panels):
            ax.plot(phi[on], y[on], linestyle="none", marker="os^D"[k % 4])
        # name at the zone edge (most positive phi): first raceway above its point, second below
        j = np.flatnonzero(on)[np.argmax(phi[on])]
        axes[0].annotate(name, (phi[j], panels[0][0][j]), xytext=(6, 4 if k % 2 == 0 else -4),
                         textcoords="offset points", va="bottom" if k % 2 == 0 else "top")
    for ax, (y, label) in zip(axes, panels):
        ax.set_ylabel(label)
        ax.set_ylim(0.0, 1.15 * y.max())                  # from 0 (all positive), room above the highest point
    axes[-1].set_xlim(-180.0, 180.0)
    axes[-1].set_xticks(np.arange(-180.0, 181.0, 45.0))
    axes[-1].set_xlabel(r"$\varphi_j$ (deg)")
    symbols = (r"$s_x$ along rolling, $s_y$ across (Harris $b$, $a$); raceway radii at $\alpha_j$ (bridge), "
               r"not at $\alpha_0$ as in AxisForge (R-019)")
    _note(fig, "; ".join([symbols] + ([note] if note else [])))
    return fig
