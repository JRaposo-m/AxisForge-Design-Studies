"""Figures of the cylindrical gear cases: quantities along the path of contact, the line of action, FZG stages.

Data in, ``matplotlib.figure.Figure`` out; appearance only from ``style/axisforge.mplstyle``, activated
by the caller (``validation._support.style.use_style()``). AxisForge results are numerical: filled
markers, no connecting line; published values: open markers.

Frames and symbols (AxisForge ``CylindricalFamily``, ISO 21771:2007 5.4 and 5.6)
--------------------------------------------------------------------------------
* The path of contact is the part of the line of action (common tangent T1-T2 of the two base circles,
  transverse plane) on which one tooth pair touches: from A (start of meshing, tip circle of the driven
  gear) to E (end, tip circle of the driving gear); C is the pitch point, B and D bound the zone of
  single tooth contact of a spur pair (``CylindricalFamily.path_points``).
* Abscissa ``g_y`` [mm]: distance from C along the line of action, positive toward the tip of gear 1.
  It does not depend on the driver (``s`` from A does): with gear 1 driving the contact travels from A
  (g_y < 0) to E, with gear 2 driving from A (g_y > 0) to E.
* Distances along the line of action from T1 (``T1A``..``T1E``, ``T1C``, ``T1T2``) are signed with the
  sign of z2: ``T1T2 = s2 a_w sin(alpha_wt) < 0`` for an internal pair, whose ``rb2`` and ``a_w`` are
  stored positive (``axisforge/core/machine_elements/gears/cylindrical.py:743``).
* Radii of curvature are signed, positive for a convex flank: the concave flank of a ring has
  ``rho2 < 0`` (ISO 21771 Eq. (87) NOTE, R-024); they are drawn signed, never as magnitudes.
* ``u_e`` is the MEAN of the two surface speeds (R-075); ``v_g = u1 - u2`` is signed and crosses zero at C.
"""
from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator

from validation._support.figures import _note, _stacked, _supports, _tag

__all__ = ["line_of_action", "load_stages", "path_of_contact"]

PATH_KEYS = ("g_y_mm", "rho1_n_mm", "rho2_n_mm", "rho_red_n_mm", "u1_m_per_s", "u2_m_per_s", "ue_m_per_s",
             "vg_m_per_s", "share", "w_N_per_mm")
"""Keys of AxisForge ``contact_inputs`` (``axisforge/solvers/machine_elements/gears/contact.py:206-211``)
read by :func:`path_of_contact`."""

PATH_NOTE = r"$g_y$ from the pitch point C, positive toward the tip of gear 1; contact travels from A to E"
"""First part of the information line of the path figures: the abscissa and the direction of travel."""

SYMBOLS_KINEMATICS = (r"$\rho_n$: normal radii of curvature of the flanks, positive = convex (triangles: reduced); "
                      r"$u_1$, $u_2$: surface speeds, $u_e$ their mean")
SYMBOLS_LOAD = (r"$F/F_{bn}$: share of the normal load on one tooth pair; $w$: load per unit length of contact "
                r"line; rigid load sharing")
"""Second part of the information line: the symbols of each path figure in words."""

_MAX_MARKERS = 40      # a dense path record shows about this many markers per series, not one per step
_NAMED_TOL_MM = 1e-3   # a step this close to a named point is an event sample: >> the one-sided offset of
                       # B-/B+ (1e-6 mm, construction.DELTA_S_MM), << a dense grid step (~0.1 mm)
_TICK_G_Y_MM = 5.0     # g_y tick step: FZG paths are 15-25 mm long, so 4-6 integer ticks


def _named(points_g_y_mm: dict) -> dict[str, float]:
    """Named points by drawn name: "B-" and "B+" (one-sided limits, 1e-6 mm apart) collapse into one "B"."""
    out = {}
    for key, g in points_g_y_mm.items():
        out.setdefault(key.rstrip("+-"), float(g))
    return out


def _label_ends(ax, x, ys, names) -> None:
    """Direct labels right of each series' right-most step (inside the margin the caller left), spread so that
    close values do not overlap."""
    j = int(np.argmax(x))
    lo, hi = ax.get_ylim()
    height_pt = ax.bbox.height * 72.0 / ax.figure.dpi       # panel height in points (before the final layout)
    previous = -np.inf
    for k in np.argsort([y[j] for y in ys]):                 # bottom to top
        at = (ys[k][j] - lo) / (hi - lo) * height_pt
        placed = max(at, previous + 9.0)                     # 9 pt apart: one line of 9 pt text
        ax.annotate(names[k], (x[j], ys[k][j]), xytext=(5, placed - at), textcoords="offset points", va="center")
        previous = placed


def path_of_contact(inputs: dict, points_g_y_mm: dict, *, p0_Pa=None, note: str | None = None,
                    kinematics_note: str | None = None, load_note: str | None = None) -> tuple[Figure, Figure]:
    """Radii, speeds and load of one tooth pair along the path of contact (two figures of stacked panels).

    Shows what a table at the five named points hides: the steps of the load share and of the line load
    at B and D (single tooth contact between them, spur pair), the sliding speed crossing zero at the
    pitch point C, and how the reduced radius and the entrainment speed change from A to E.

    Parameters
    ----------
    inputs : dict
        The record of AxisForge ``contact_inputs`` (one entry per step, arrays of equal length). Keys read
        (``PATH_KEYS``): ``g_y_mm`` position from C [mm]; ``rho1_n_mm``, ``rho2_n_mm``, ``rho_red_n_mm``
        normal radii of curvature, signed [mm]; ``u1_m_per_s``, ``u2_m_per_s`` surface speeds,
        ``ue_m_per_s`` entrainment speed (mean), ``vg_m_per_s`` sliding speed, signed [m/s]; ``share``
        load share of the tooth pair [-]; ``w_N_per_mm`` load per unit length of contact line [N/mm].
        To show the steps at B and D as steps, the steps must include the one-sided times B-, B+, D-, D+
        (``construction.path_times``, GL-09): the function never interpolates.
    points_g_y_mm : dict
        Named points -> position g_y [mm] (``CylindricalFamily.path_points(pair, driver)``: "A".."E";
        "B-"/"B+", "D-"/"D+" are drawn as one "B", "D"). Thin dotted verticals in every panel, names above
        the top frame. For a helical pair pass A, C, E only: B and D bound no single-contact zone (GL-25).
    p0_Pa : array_like, optional
        Maximum Hertz pressure at each step [Pa], drawn in MPa as a third panel of the load figure.
        Omitted panel if None.
    note : str, optional
        Appended to the information line of both figures (pair, stage, speed, force).
    kinematics_note, load_note : str, optional
        Appended to the information line of one figure only, e.g. for a helical pair "radii and speeds at
        mid face width (R-027)" (kinematics) and "share and w over the whole contact line (Q36)" (load).

    Returns
    -------
    fig_kinematics : matplotlib.figure.Figure
        (a) rho1_n, rho2_n, rho_red_n with a zero line; (b) u1, u2, u_e; (c) v_g with a zero line.
    fig_load : matplotlib.figure.Figure
        (a) share; (b) w; (c) p0 if given. The caller closes both.

    Raises
    ------
    KeyError
        If a key of ``PATH_KEYS`` is missing from ``inputs`` (all missing keys named).
    ValueError
        If the record is empty, or ``p0_Pa`` has another length.

    Notes
    -----
    Markers: about ``_MAX_MARKERS`` per series. In the kinematics figure (continuous quantities) each named
    point is drawn once; in the load figure both one-sided steps at a named point are drawn and joined by a
    thin vertical segment, so that a step of the share reads as a step, not as two separate levels.

    Assumptions of the record (``contact_inputs``): rigid unmodified involute flanks, rigid load sharing
    (one load per unit length on all contact lines, Q30), steady speed and constant normal force. For a
    helical pair the radii and speeds are those of the transverse section at mid face width, while the
    share and w describe the whole contact line (R-027, Q36): the caller says so in the notes.
    """
    missing = [k for k in PATH_KEYS if k not in inputs]
    if missing:
        raise KeyError(f"contact_inputs record lacks {missing}")
    g = np.asarray(inputs["g_y_mm"], dtype=float)
    if g.size == 0:
        raise ValueError("empty contact_inputs record: nothing to draw")
    col = {k: np.asarray(inputs[k], dtype=float) for k in PATH_KEYS}
    named = _named(points_g_y_mm)
    # markers: a regular subsample of the dense steps, without the samples crowding a named point, plus the
    # event samples at the named points (kinematics: one; load: every one-sided step)
    order = np.argsort(g, kind="stable")
    stride = max(1, g.size // _MAX_MARKERS)
    gap = 0.5 * stride * np.ptp(g) / max(1, g.size - 1)     # half a marker spacing [mm]
    to_named = np.min(np.abs(g[:, None] - np.array(list(named.values()) or [np.inf])[None, :]), axis=1)
    regular = [i for i in order[::stride] if to_named[i] >= gap]
    once = [int(np.argmin(np.abs(g - gp))) for gp in named.values()]
    every = {"kinematics": sorted(set(regular) | set(once)),
             "load": sorted(set(regular) | set(np.flatnonzero(to_named <= _NAMED_TOL_MM).tolist()))}
    kinematics = [  # (series, direct labels, y label, zero line)
        ([col["rho1_n_mm"], col["rho2_n_mm"], col["rho_red_n_mm"]],
         [r"$\rho_{1n}$", r"$\rho_{2n}$", r"$\rho_{\mathrm{red},n}$"], r"$\rho_n$ (mm)", True),   # signed (R-024)
        ([col["u1_m_per_s"], col["u2_m_per_s"], col["ue_m_per_s"]],
         [r"$u_1$", r"$u_2$", r"$u_e$"], r"$u$ (m/s)", False),                                       # u_e: mean (R-075)
        ([col["vg_m_per_s"]], [None], r"$v_g = u_1 - u_2$ (m/s)", True),
    ]
    # OPEN GL-05: slide/roll ratio not drawn; v_g is defined (cylindrical.py:1179), the ratio is not ruled
    load = [([col["share"]], [None], r"$F/F_{bn}$", False),           # share of the normal force on this pair
            ([col["w_N_per_mm"]], [None], r"$w$ (N/mm)", False)]      # AxisForge units, N/m only after the adapter (R-023)
    if p0_Pa is not None:
        # OPEN R-022, R-118: p_H along the path is blocked (adapter raises; E, nu unknown); panel tested synthetic only
        p0 = np.asarray(p0_Pa, dtype=float)
        if p0.shape != g.shape:
            raise ValueError(f"p0_Pa has {p0.size} values, the record {g.size}")
        load.append(([p0 * 1e-6], [None], r"$p_0$ (MPa)", False))   # [Pa] -> [MPa]
    # consecutive steps closer than the event tolerance: the two sides of a step of the rigid load sharing
    jumps = [(order[k], order[k + 1]) for k in range(g.size - 1) if g[order[k + 1]] - g[order[k]] <= _NAMED_TOL_MM]
    figures = []
    for kind, panels, symbols, own in (("kinematics", kinematics, SYMBOLS_KINEMATICS, kinematics_note),
                                       ("load", load, SYMBOLS_LOAD, load_note)):
        fig, axes = _stacked(len(panels))
        labelled = any(names[0] is not None for _, names, _, _ in panels)
        # room right of the path for the direct labels, inside the frame (the figure keeps the sheet width), and
        # left of A for the panel tag above the frame, which would otherwise touch the name "A"
        axes[-1].set_xlim(g.min() - 0.06 * np.ptp(g), g.max() + (0.10 if labelled else 0.03) * np.ptp(g))
        for k, (ax, (ys, names, label, zero)) in enumerate(zip(axes, panels)):
            for y, marker in zip(ys, "os^"):
                line, = ax.plot(g, y, linestyle="none", marker=marker, markevery=every[kind])
                if kind == "load":
                    for i, j in jumps:                       # thin connector between the one-sided levels
                        ax.plot(g[[i, j]], y[[i, j]], color=line.get_color(), linestyle="-",
                                linewidth=plt.rcParams["axes.linewidth"])
            if zero:
                ax.axhline(0.0, linewidth=plt.rcParams["axes.linewidth"])
            if names[0] is not None:
                _label_ends(ax, g, ys, names)
            _supports(ax, list(named.values()), list(named) if k == 0 else [""] * len(named))
            ax.set_ylabel(label)
        if kind == "load":
            for ax in axes:
                ax.set_ylim(0.0, 1.15 * ax.get_ylim()[1])   # share, w, p0 >= 0: from zero, room above
        axes[-1].xaxis.set_major_locator(MultipleLocator(_TICK_G_Y_MM))
        axes[-1].set_xlabel(r"$g_y$ (mm)")
        _note(fig, "; ".join(t for t in (PATH_NOTE, symbols, note, own) if t))
        figures.append(fig)
    return figures[0], figures[1]


def _segment_distance(q, a, b) -> float:
    """Distance from point ``q`` to the segment ``a``-``b`` (drawing units)."""
    ab = b - a
    t = np.clip(np.dot(q - a, ab) / max(np.dot(ab, ab), 1e-300), 0.0, 1.0)
    return float(np.linalg.norm(q - (a + t * ab)))


def line_of_action(T1C_mm: float, T1T2_mm: float, rb1_mm: float, rb2_mm: float, ra1_mm: float, ra2_mm: float,
                   alpha_wt_rad: float, points_g_y_mm: dict, *, limits_mm=None, note: str | None = None) -> Figure:
    """Base and tip circles, line of action T1-T2 and the named points of the path, in the transverse plane.

    A review check, not a thesis figure: it shows where A..E sit for the chosen driver (A on the tip
    circle of the driven gear, E on that of the driver) and whether an internal pair is drawn as one
    (ring around the pinion).

    Parameters
    ----------
    T1C_mm : float
        Distance from T1 (tangent point on base circle 1) to the pitch point C, ``rb1 tan(alpha_wt)`` [mm].
    T1T2_mm : float
        Signed length T1-T2, ``s2 a_w sin(alpha_wt)`` [mm]; negative for an internal pair, whose
        ``rb2`` and ``a_w`` are positive (``cylindrical.py:743``). The only carrier of the pair kind.
    rb1_mm, rb2_mm : float
        Base radii of gear 1 and gear 2 [mm], positive.
    ra1_mm, ra2_mm : float
        Tip radii of gear 1 and gear 2 [mm], positive (``gear.da / 2``; for a ring ``da < d``: its tip circle
        lies inside its reference circle). Dashed.
    alpha_wt_rad : float
        Transverse working pressure angle [rad] (``pair.alphatw``).
    points_g_y_mm : dict
        Named points -> g_y from C [mm] (``CylindricalFamily.path_points(pair, driver)``), drawn at
        ``T1C + g_y`` from T1; "B-"/"B+" are drawn as one "B". Only the points given are defined in the note.
    limits_mm : tuple of float, optional
        ``(xmin, xmax, ymin, ymax)`` of panel (a) [mm], square, in the drawing frame (O1 at the origin, O2 on
        the y axis): pass the same limits to compare pairs. Default: the four circles, 5 % margin.
    note : str, optional
        Appended to the information line (which states the pair kind and defines the symbols).

    Returns
    -------
    matplotlib.figure.Figure
        (a) the whole pair; (b) zoom on T1, T2 and the path (a tenth of that length beyond each end), named
        points labelled, arrow of increasing g_y. Both panels square with equal aspect, line of centres
        vertical. The caller closes it.

    Notes
    -----
    Construction in the frame (n, e) of the line of action, n from O1 to T1, e along the line from T1
    toward increasing g_y: T1 = rb1 n, a point at distance s from T1 is T1 + s e, T2 = T1 + T1T2 e and
    O2 = T2 + sign(T1T2) rb2 n (other side of the line for an external pair, same side for an internal
    one). Then ``|O1 O2| = a_w`` in both cases (external: (rb1 + rb2)^2 + T1T2^2 = a_w^2; internal: rb2 - rb1
    = a_w cos(alpha_wt)) and C lies on the line of centres. The frame is turned by ``pi/2 - alpha_wt`` so
    that the line of centres points along +y. ISO 21771:2007, 5.4.3, Eq. (81)-(87).

    Labels are placed, among a few candidate directions, where they are farthest from the circles, the
    lines and the points (a review figure must not hide what it checks).
    """
    # (n, e) -> drawing frame: the line of centres, at alpha_wt from n toward e, is drawn along +y
    n = np.array([np.sin(alpha_wt_rad), np.cos(alpha_wt_rad)])
    e = np.array([-np.cos(alpha_wt_rad), np.sin(alpha_wt_rad)])
    internal = T1T2_mm < 0.0
    O1 = np.zeros(2)
    T1 = rb1_mm * n
    T2 = T1 + T1T2_mm * e
    O2 = T2 + np.sign(T1T2_mm) * rb2_mm * n                  # internal: the ring centre on the side of O1
    named = _named(points_g_y_mm)
    s = {name: T1C_mm + g for name, g in named.items()}      # distance from T1 along the line [mm]
    at = {name: T1 + d * e for name, d in s.items()}
    ink = plt.rcParams["lines.color"]                        # geometry: lines in the sheet's line colour, identity by name
    circles = [(O1, rb1_mm, "-"), (O2, rb2_mm, "-"), (O1, ra1_mm, "--"), (O2, ra2_mm, "--")]   # base, tip
    ends = [0.0, T1T2_mm, *s.values()]                       # the tangent line spans T1, T2 and the whole path
    line = (T1 + min(ends) * e, T1 + max(ends) * e)          # (internal pair: the path lies beyond T1)
    span = max(ends) - min(ends)
    # positive g_y (toward the tip of gear 1, the abscissa of the path figures): an arrow parallel to the line,
    # 6 % of the span off it (~2 mm for FZG), centred on C, 30 % of the path long, on the side clear of circles
    C = T1 + T1C_mm * e
    half_arrow = 0.15 * (max(s.values()) - min(s.values()) if len(s) > 1 else span)

    def clearance(q, segments=(), points=()):
        """Distance from q to the nearest circle, segment or point [mm]."""
        d = [abs(np.linalg.norm(q - c) - r) for c, r, _ in circles]
        d += [_segment_distance(q, a, b) for a, b in segments]
        d += [float(np.linalg.norm(q - p)) for p in points]
        return min(d)

    side = max((1.0, -1.0), key=lambda k: min(clearance(C + k * 0.06 * span * n + t * e) for t in
                                              (-half_arrow, 0.0, half_arrow)))
    arrow = (C + side * 0.06 * span * n - half_arrow * e, C + side * 0.06 * span * n + half_arrow * e)

    def place(ax, text, p, directions, d, segments, points):
        """Write ``text`` at distance d from p, in the candidate direction with the most clearance."""
        u = max(directions, key=lambda v: clearance(p + d * v, segments, points))
        ax.text(*(p + d * u), text, ha="center", va="center")
        return p + d * u

    compass = [np.array([np.cos(a), np.sin(a)]) for a in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)]
    fig, axes = plt.subplots(1, 2)
    angle = np.linspace(0.0, 2.0 * np.pi, 721)
    if limits_mm is None:                                    # the four circles, square, 5 % margin
        lo = np.min([c - r for c, r, _ in circles], axis=0)
        hi = np.max([c + r for c, r, _ in circles], axis=0)
        mid, half = 0.5 * (lo + hi), 0.525 * np.max(hi - lo)
        limits_mm = (mid[0] - half, mid[0] + half, mid[1] - half, mid[1] + half)
    zoom_mid, zoom_half = T1 + 0.5 * (max(ends) + min(ends)) * e, 0.6 * span   # a tenth of the span beyond each end
    for k, ax in enumerate(axes):
        for c, r, linestyle in circles:
            ax.plot(c[0] + r * np.cos(angle), c[1] + r * np.sin(angle), color=ink, linestyle=linestyle)
        ax.plot(*np.transpose(line), color=ink, linestyle="-")
        ax.plot(*np.transpose([O1, O2]), color=ink, linestyle=":", linewidth=plt.rcParams["axes.linewidth"])
        for name, xy in (("O1", O1), ("O2", O2)):
            ax.plot(*xy, linestyle="none", marker="+", color=ink)
            ax.annotate(name, xy, xytext=(4, -4), textcoords="offset points", ha="left", va="top")
        if k == 0:
            ax.set_xlim(*limits_mm[:2])
            ax.set_ylim(*limits_mm[2:])
            d = 0.05 * (limits_mm[1] - limits_mm[0])         # label distance in (a)
        else:
            ax.set_xlim(zoom_mid[0] - zoom_half, zoom_mid[0] + zoom_half)
            ax.set_ylim(zoom_mid[1] - zoom_half, zoom_mid[1] + zoom_half)
            d = 0.07 * span                                  # label distance in (b)
        for name, xy in (("T1", T1), ("T2", T2)):
            ax.plot(*xy, linestyle="none", marker="+", color=ink)
            place(ax, name, xy, compass, d, [line, (O1, O2)], [T1, T2])
        ax.set_aspect("equal", adjustable="box")             # both square: same size, tags at one height
        ax.set_xlabel(r"$x$ (mm)")
        ax.set_ylabel(r"$y$ (mm)")
        _tag(ax, f"({'ab'[k]})")
    # (b) only: the named points (computed positions, filled markers), their names, the g_y arrow
    taken = list(at.values())
    for name in sorted(s, key=s.get):
        axes[1].plot(*at[name], linestyle="none", marker="o", color=ink)
        # off the line: across it, or across and along it (where a circle crowds the straight position)
        off = [n, -n, *(v / np.linalg.norm(v) for v in (n + e, n - e, -n + e, -n - e))]
        taken.append(place(axes[1], name, at[name], off, d, [line, arrow], taken))
    axes[1].annotate("", xy=arrow[1], xytext=arrow[0], arrowprops=dict(arrowstyle="->", color=ink))
    place(axes[1], r"$g_y$", arrow[1], compass, 0.05 * span, [line, arrow], taken)
    kind = (r"internal pair ($T_1T_2$ < 0): ring centre O2 on the side of O1" if internal
            else r"external pair ($T_1T_2$ > 0)")
    meaning = {"A": "A: start of meshing", "B": "B, D: limits of single tooth contact", "C": "C: pitch point",
               "E": "E: end of meshing"}
    words = ["T1, T2: tangent points of the line of action on the base circles (solid); dashed: tip circles",
             *(meaning[p] for p in "ABCE" if p in named),
             r"$g_y$ positive toward the tip of gear 1, as on the path figures"]
    _note(fig, "; ".join(t for t in (f"transverse plane, line of centres along +y; {kind}", *words, note) if t))
    return fig


def load_stages(published: dict, computed: dict, *, quantity: str, unit: str) -> Figure:
    """Computed against published values per FZG load stage and gear, on a two-level categorical axis.

    Stages (K1..K9, inner level) are grouped by gear (outer level), in the manner of Martins 2015
    Fig. 15 (the model figure is not yet checked against the paper, r3 gear-expert sec. 5).

    Parameters
    ----------
    published, computed : dict
        ``{(stage, gear): value}`` in ``unit``. The categories are the union of the keys; a value missing
        on one side is simply not drawn (the printed helical column is "H501/951", the cases build
        "H501"; the caller maps the keys if they are to share a group).
    quantity : str
        Symbol of the quantity, mathtext allowed (``r"$F_\\mathrm{r}$"``).
    unit : str
        Unit of the values, as printed in the label ``"<quantity> (<unit>)"``.

    Returns
    -------
    matplotlib.figure.Figure
        (a) values: computed as filled circles (first colour of the cycle), published as larger open
        circles around them, frameless legend; (b) relative deviation (computed - published) / published
        in %, for the keys present on both sides (panel omitted if there is none). The caller closes it.

    Raises
    ------
    ValueError
        If both mappings are empty (no empty axes).
    """
    # OPEN R-022, R-118: a computed p_H is blocked (gear adapter raises, E and nu unknown); F_r only today
    keys = list(dict.fromkeys([*published, *computed]))
    if not keys:
        raise ValueError("no published or computed value: nothing to draw")

    def stage_order(stage):                                  # K2 before K10
        digits = re.sub(r"\D", "", str(stage))
        return (int(digits) if digits else 0, str(stage))

    both = [k for k in keys if k in published and k in computed]
    fig, axes = _stacked(2 if both else 1)
    gears = list(dict.fromkeys(gear for _, gear in keys))
    x, ticks, labels, x0 = {}, [], [], 0.0
    for gear in gears:
        stages = sorted({stage for stage, gg in keys if gg == gear}, key=stage_order)
        for j, stage in enumerate(stages):
            x[(stage, gear)] = x0 + j
        ticks += [x0 + j for j in range(len(stages))]
        labels += stages
        # outer level: the gear name centred under its stages
        axes[-1].annotate(gear, (x0 + 0.5 * (len(stages) - 1), 0.0), xycoords=("data", "axes fraction"),
                          xytext=(0, -16), textcoords="offset points", ha="center", va="top")
        if x0 > 0.0:                                         # thin separator between gear groups
            for ax in axes:
                ax.axvline(x0 - 1.0, linestyle=":", linewidth=plt.rcParams["axes.linewidth"])
        x0 += len(stages) + 1.0                              # one empty slot between groups
    ax = axes[0]
    if computed:                                             # first: the cycle's first colour
        ax.plot([x[k] for k in computed], list(computed.values()), linestyle="none", marker="o", label="computed")
    if published:                                            # open circle around the computed marker, same x
        ax.plot([x[k] for k in published], list(published.values()), linestyle="none", marker="o", mfc="none",
                markersize=2.0 * plt.rcParams["lines.markersize"], label="published")
    ax.set_ylim(bottom=0.0)
    ax.set_ylabel(f"{quantity} ({unit})")
    ax.legend()
    if both:
        deviation = [100.0 * (computed[k] - published[k]) / published[k] for k in both]
        axes[1].plot([x[k] for k in both], deviation, linestyle="none", marker="o")
        axes[1].axhline(0.0, linewidth=plt.rcParams["axes.linewidth"])
        axes[1].set_ylabel(r"$\Delta$ (%)")
        _note(fig, r"$\Delta$ = (computed - published) / published")
    axes[-1].set_xticks(ticks, labels)
    for ax in axes:                                          # categories: no minor ticks between them
        ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    axes[-1].set_xlim(-0.6, x0 - 1.4)
    return fig
