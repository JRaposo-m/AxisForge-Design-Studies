"""SlipPY static (dry, normal) contact of a contact on its grid, measured and checked against Hertz.

:func:`solve` builds the two surfaces (:func:`surfaces.surface_pair`, with rough heights when
given), the SlipPY ``ContactModel`` and one ``StaticStep``, solves, measures the pressure field and
fills a :class:`StaticResult` with the Hertz reference, the relative errors, the checks and the
commits of the three repositories.
"""
from __future__ import annotations

import dataclasses
import functools
import itertools
import math
import subprocess
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import slippy
import slippy.contact as c
from scipy.ndimage import map_coordinates

from ..mesh.grid import SPAN_X_0
from ..roughness.macro.measures import THRESHOLD_REL
from ..results.hertz import HertzResult
from ..results.static import StaticResult
from .hertz import hertz
from .surfaces import surface_pair

if TYPE_CHECKING:
    from ..core.contact import Contact
    from ..mesh.grid import Grid
    from ..roughness.macro.surfaces import RoughHeights

__all__ = ["solve"]

_ids = itertools.count()       # SlipPY model and material names are process-wide
_EDGE = THRESHOLD_REL          # [-] loaded: p > _EDGE p0; below it the solver's numerical zero (measure and edge_clear)


# OPEN SB-09: span_x_min, min_nodes, rtol_reference kept because the checks need them
# OPEN SB-18: solve takes the HertzResult the grid was built with, so Hertz runs once per contact
def solve(contact: Contact, grid: Grid, output_dir, *, hertz_result: HertzResult | None = None,
          use_frequency_domain: bool = True, method: str = "pk", tolerance: float = 1e-9,
          span_x_min: float = SPAN_X_0, min_nodes: float = 8.0, rtol_reference: float = 0.03,
          heights: tuple[RoughHeights | None, RoughHeights | None] | None = None, **step_kwargs) -> StaticResult:
    """Solve the dry normal contact of ``contact`` on ``grid`` with the SlipPY BEM and measure it.

    Both bodies are elastic half-spaces with their exact profiles (plus their rough ``heights``
    when given). The pressure field is measured and compared with Hertz. Load control
    (``Grid.step_args``), one of three:

    * ``"normal_load"`` (nothing given): the load is ``contact.load`` (a line: load per unit
      length x period on a strip periodic along y); the Hertz reference is ``hertz_result``
      (or :func:`hertz` of ``contact``);
    * ``"interference"`` [m] (the rigid approach of the bodies, solved by SlipPY's BCCG at that
      approach; ``method`` is then not used) or ``"mean_gap"`` [m] (``method="rey"`` only, which no
      bridge contact can use: ``Grid.step_args``, OPEN SB-61) in ``step_kwargs``: the load is an output, SlipPY
      ``total_normal_load`` of the step (``HeightOptimisationFunction.results``,
      ``slippy/contact/_step_utils.py:226-241``). The Hertz reference and every ``err_*`` are
      then computed at that load (a line: the found load / period [N/m]), and ``hertz_result``
      is not used for the comparison (only the grid was built with it). Hertz at the found load
      hides the kernel's approach error in the load (it shows in ``err_delta`` only, OPEN SB-57).
      The approach of a line contact is not defined locally (Johnson 1985, 4.2(c)): the
      interference given to a line is a number of its grid and kernel (the zero-frequency term
      is summed over the padded domain), meaningful only on the grid it came from.

    Offsets ``off_set_x``, ``off_set_y`` [m] are passed to SlipPY (``off_set_x`` shifts body 2
    along array axis 0, grid x; ``_model_utils.get_gap_from_model``). The measurements assume the
    apex on the grid centre node. With an offset ``s`` the contact moves to the minimum of the
    shifted gap, ``x* = (K1 + K2)^-1 K2 s`` for paraboloids (not ``s`` itself), shown by
    ``apex_offset_nodes``; ``p0``, ``a``, ``b`` (cuts through the peak) and ``delta`` do not
    depend on it, but ``margin_ok`` counts the spans from the grid centre and overstates the
    margin by up to ``|x*|`` (OPEN SB-63). With ``periodic_geometry=False`` (a point contact by
    default) SlipPY keeps only the overlap of the two surfaces (``_model_utils.py:137-150``), so
    a pressure field smaller than the grid comes back and is refused below;
    ``periodic_geometry=True`` keeps the grid (and wraps body 2 along both axes).

    With rough ``heights`` (``build = "rough"``, also for heights all zero) the Hertz-shape
    measures, which assume one smooth patch (:func:`_measure`), are not applied and Hertz is a
    reference value only (field by field: :class:`StaticResult`). ``delta_mean_planes = delta +
    m``, the approach of the mean planes (Johnson 1985, 13.5; probe C2), with ``m`` SlipPY's shift of
    the gap to first touch read back from its own gap (``-(z1 + z2) - gap0``, a constant when SlipPY
    solved these surfaces; otherwise nan with a warning). The interpolation mode must be
    ``"nearest"`` (SlipPY's default; the others swap the axes of body 2, R-051). Real area, patches
    and mean real pressure: ``roughness.macro.rough_measures`` on ``StaticResult.p``.

    Parameters
    ----------
    contact : Contact
        Its ``label`` is used in SlipPY file names: keep it file-name safe.
    grid : Grid
        From ``Grid.around`` (origin on node ``(n_x//2, n_y//2)``).
    output_dir : path
        Where SlipPY writes its .sar/.sdb/.log files; created if missing. SlipPY's global
        ``OUTPUT_DIR`` is restored after the solve.
    hertz_result : HertzResult, optional
        The Hertz solution the grid was built with (``Grid.around``), used as is; computed with
        :func:`hertz` when None.
    use_frequency_domain : bool, optional
        Influence matrix of both bodies (:func:`surface_pair`).
    method, tolerance, **step_kwargs
        Passed to ``slippy.contact.StaticStep`` unchecked (with the load and periodic flags of
        ``Grid.step_args``); SlipPY raises its own errors (see ``Grid.step_args`` for its limits
        and their source lines).
    span_x_min : float
        Limit of ``margin_ok`` [-]: the domain must reach ``span_x_min`` half-extents of the Hertz
        ellipse on each side along grid x (rolling when ``grid.angle = 0``; the minor axis on a
        principal grid), because the frequency-domain kernel's image loads change p0 even where the
        pressure is zero (see ``mesh.grid.SPAN_X_0``). Grid x only: the image error along grid y is
        not bounded by a margin (it is not monotonic in the domain size, R-509). Conservative with
        the spatial kernel, which has no images.
    min_nodes : float
        Limit of ``resolution_ok`` [-]: nodes across the minor semi-axis ``b``; fewer cannot
        resolve the Hertz profile near its edge.
    rtol_reference : float
        Limit of ``within_reference`` [-]: relative error of p0, a, b and the approach against
        Hertz (meaningful for smooth surfaces only: not applied with ``heights``).
    heights : (RoughHeights or None, RoughHeights or None), optional
        Rough heights of body 1 and body 2 on ``grid`` (``roughness.macro.rough_heights``), passed
        to :func:`surface_pair`. None (default): both bodies smooth, and the call and every number
        are those of a solve without this argument.

    Returns
    -------
    StaticResult
        Fields, measured values, Hertz reference, errors, checks, provenance, the load control
        and its inputs. The measured load is the bridge's own ``sum p h^2``, independent of
        SlipPY's bookkeeping (it equals SlipPY ``total_normal_load``, recorded as
        ``load_found``). The checks:

        * ``converged`` - SlipPY's own flag;
        * ``load_ok`` - the pressure integrates to the applied load within 1e-3: a consistency
          check of the solve (what tolerance a study needs is open, R-057). Under
          ``interference`` or ``mean_gap`` no load is applied (``load_applied`` and ``err_load``
          nan) and ``load_ok`` is True: not a check there (OPEN SB-57);
        * ``edge_clear`` - the pressure on the domain boundary is at most 1e-6 p0 (the threshold
          of the measured edge); a line: only the two ends along grid x (y is periodic).
          Necessary, not sufficient;
        * ``margin_ok``, ``resolution_ok``, ``within_reference`` - see the parameters (with
          ``heights``, ``margin_ok`` and ``resolution_ok`` are those of the nominal Hertz
          contact; the resolution of the roughness is ``RoughHeights.grid_rule_ok``);
        * ``hertz_valid`` - Hertz validity of the reference (``HertzResult.valid``).

    Raises
    ------
    ValueError
        From ``Grid.step_args`` (``normal_load`` given, a non-periodic or turned line),
        :func:`surface_pair` (also heights not on ``grid``) and SlipPY (its options; with heights, an
        interpolation mode other than ``"nearest"``); if SlipPY returns a pressure field not of
        the grid's shape (an offset on a non-periodic geometry); from :class:`Contact` if the
        load found under ``interference`` is not positive (no Hertz reference).
    TypeError
        From SlipPY, when ``interference`` leaves no node in contact (``interference <= 0``):
        its results are never set (``_step_utils.py:190``; OPEN SB-62).
    """
    name = f"{contact.label or type(contact).__name__}_{next(_ids)}"
    s1, s2 = surface_pair(contact, grid, name=name, use_frequency_domain=use_frequency_domain, heights=heights)
    rough = heights is not None and any(rh is not None for rh in heights)
    if rough and step_kwargs.get("profile_interpolation_mode", "nearest") != "nearest":
        # SlipPY Surface.interpolate swaps the axes outside 'nearest' (Surface_class.py:1213-1224, R-051): body 2
        # would be solved transposed, which a smooth axisymmetric profile hides and rough heights do not
        raise ValueError("solve with heights needs profile_interpolation_mode='nearest' (R-051: SlipPY's other "
                         "modes interpolate body 2 on swapped axes)")
    args = grid.step_args(contact, **step_kwargs)
    control = next((k for k in ("interference", "mean_gap") if args.get(k) is not None), "normal_load")
    load_applied = args.get("normal_load", math.nan)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = slippy.OUTPUT_DIR              # ContactModel(output_dir=) sets SlipPY's global (R-088, R-511)
    try:
        model = c.ContactModel(name, s1, s2, output_dir=str(out_dir))
        model.add_step(c.StaticStep("static", method=method, tolerance=tolerance, **args))
        t0 = time.perf_counter()
        sol = model.solve()
        solve_time = time.perf_counter() - t0
    finally:
        slippy.OUTPUT_DIR = saved

    p = np.asarray(sol["loads_z"], dtype=float)                  # a pressure [Pa]
    if p.shape != grid.X.shape:
        raise ValueError(f"SlipPY returned a pressure field of shape {p.shape}, the grid is {grid.X.shape}")
    load_found = float(sol["total_normal_load"])                 # [N], the step's total (_step_utils.py:226)
    if control == "normal_load":
        hz = hertz(contact) if hertz_result is None else hertz_result
    else:
        # OPEN SB-56: the Hertz reference at the found load; the contact keeps its nominal load
        found = load_found / grid.length_y_m if contact.line else load_found      # [N/m] (line) or [N]
        hz = hertz(dataclasses.replace(contact, load=found))
    h = grid.h
    # psi is from bridge x; the cuts are on the grid, at psi - gamma from grid x (a, b stay intrinsic)
    if rough:                                                    # one smooth patch is not assumed
        p0, b, a, apex_offset = float(p.max()), math.nan, math.nan, math.nan
    else:
        p0, b, a, apex_offset = _measure(p, h, contact.psi - grid.angle, contact.line)
    load = float(p.sum() * h**2)
    # SlipPY interference: the approach of the bodies; not defined locally for a line (Johnson 4.2(c))
    delta = math.nan if contact.line else float(sol["interference"])
    z1, z2 = np.asarray(s1.profile), np.asarray(s2.profile)
    gap0 = np.asarray(sol["just_touching_gap"], dtype=float)
    delta_mean = math.nan
    # rough: SlipPY shifted its gap -(z1 + z2) to first touch by a constant (_model_utils.py:173-177), read back
    # from its own gap; with an offset body 2 is moved, so the smooth apex is not at 0 on these nodes (OPEN R4-17)
    if rough and not contact.line and not any(args.get(k) for k in ("off_set_x", "off_set_y")):
        shift = -(z1 + z2) - gap0
        if np.ptp(shift) <= 1e-12 * max(float(np.abs(z1 + z2).max()), 1e-300):
            delta_mean = float(delta + shift.mean())
        else:
            warnings.warn(f"solve: SlipPY's gap is not -(z1 + z2) shifted by a constant (spread {np.ptp(shift):.3g} "
                          "m): it solved other surfaces than z1, z2; delta_mean_planes set to nan", stacklevel=2)
    displacements = {k: np.asarray(sol[k], dtype=float)
                     for k in ("total_displacement_z", "surface_1_displacement_z", "surface_2_displacement_z")}

    err = {"err_p0": _rel_err(p0, hz.p0), "err_a": _rel_err(a, hz.a), "err_b": _rel_err(b, hz.b),
           "err_delta": _rel_err(delta, hz.delta), "err_load": _rel_err(load, load_applied)}
    if rough:                                                    # Hertz is a reference value only
        err.update(err_p0=math.nan, err_a=math.nan, err_b=math.nan, err_delta=math.nan)
    refs = {"err_p0": hz.p0, "err_a": hz.a, "err_b": hz.b, "err_delta": hz.delta}
    edge = np.concatenate([p[0, :], p[-1, :]] + ([] if contact.line else [p[:, 0], p[:, -1]]))
    span_x, _ = grid.spans(hz)
    checks = {
        # OPEN SB-57: no applied load under interference / mean_gap control: nothing to balance
        "load_ok": control != "normal_load" or abs(err["err_load"]) <= 1e-3,
        "edge_clear": bool(edge.max(initial=0.0) <= _EDGE * p0),
        "margin_ok": bool(span_x >= span_x_min),
        "resolution_ok": bool(hz.b / h >= min_nodes),
        # a missing measurement fails a finite reference; an infinite or nan reference is skipped
        "within_reference": None if rough else all(math.isfinite(err[k]) and abs(err[k]) <= rtol_reference
                                                   for k, ref in refs.items() if math.isfinite(ref)),
    }
    rough_fields = {"build": "rough", "heights": tuple(heights), "delta_mean_planes": delta_mean} if rough else {}
    return StaticResult(contact=contact, grid=grid, z1=z1, z2=z2, **rough_fields,
                        gap0=gap0,
                        load_applied=load_applied, period=grid.length_y_m if contact.line else math.nan,
                        load_control=control, load_found=load_found,
                        **{k: math.nan if args.get(k) is None else float(args[k])     # nan: not given
                           for k in ("interference", "mean_gap", "off_set_x", "off_set_y")},
                        use_frequency_domain=use_frequency_domain, method=method, tolerance=tolerance,
                        converged=bool(sol.get("converged", False)), solve_time=solve_time,
                        p=p, delta=delta, **displacements, p0=p0, a=a, b=b, load=load,
                        apex_offset_nodes=apex_offset, hertz=hz, **err, **checks, commits=dict(_commits()))


def _measure(p: np.ndarray, h: float, psi: float, line: bool) -> tuple[float, float, float, float]:
    """Peak pressure [Pa], semi-axes ``b`` along ``psi`` and ``a`` across it [m] (``a = inf`` for a
    line), and the distance of the peak from the grid centre node [nodes].

    ``psi`` is the direction of ``b`` from the grid x axis (array axis 0) [rad]: ``contact.psi -
    grid.angle``, which is ``contact.psi`` on a grid on the bridge axes and 0 on a principal grid.
    Each semi-axis is measured on the cut through the peak: for ``psi = 0`` the grid lines through
    it (b along axis 0, a along axis 1); otherwise the line through the peak sampled every ``h``
    by bilinear interpolation of ``p**2`` (``scipy.ndimage.map_coordinates``, order 1), keeping
    only samples whose four surrounding nodes are all loaded. A sample interpolated between a
    loaded and an unloaded node would push the edge outward by up to one node (+6.5 % in b at
    b/h = 13); ``p**2``, nearly linear near a Hertz edge, interpolates better than ``p``.
    """
    i0, j0 = np.unravel_index(int(np.argmax(p)), p.shape)
    p0 = float(p[i0, j0])
    thr = _EDGE * p0
    if psi == 0.0:                                    # principal axes on the grid lines
        b = _semi_axis(p[:, j0], h, i0, thr)
        a = math.inf if line else _semi_axis(p[i0, :], h, j0, thr)
    else:
        n = p.shape[0] + p.shape[1]                   # samples per side: longer than any cut through the grid
        t = np.arange(-n, n + 1, dtype=float)

        loaded = (p > thr).astype(float)

        def cut(direction: float) -> np.ndarray:
            # node coordinates (axis 0 = grid x, axis 1 = grid y) of the samples, one node apart
            coords = [i0 + t * math.cos(direction), j0 + t * math.sin(direction)]
            inside = map_coordinates(loaded, coords, order=1, mode="constant", cval=0.0) >= 1.0 - 1e-12
            p2 = map_coordinates(p**2, coords, order=1, mode="constant", cval=0.0)
            return np.where(inside, np.sqrt(np.clip(p2, 0.0, None)), 0.0)

        # the last fully supported sample can lie up to about sqrt(2) nodes inside the edge
        b = _semi_axis(cut(psi), h, n, thr, reach=2.0)
        a = math.inf if line else _semi_axis(cut(psi + math.pi / 2), h, n, thr, reach=2.0)
    c0, c1 = p.shape[0] // 2, p.shape[1] // 2         # grid centre node (Grid.around)
    offset = abs(i0 - c0) if line else math.hypot(i0 - c0, j0 - c1)
    return p0, float(b), float(a), float(offset)


def _semi_axis(p_cut: np.ndarray, h: float, i0: int, thr: float, reach: float = 1.0) -> float:
    """Semi-axis of the loaded region on one cut ``p_cut`` through the maximum ``i0`` [m].

    It is half the length of the connected loaded segment through the maximum: the Hertz semi-axis
    of a smooth contact, but only one contact patch of a rough one.

    Each edge is placed where ``p**2``, linear in position near a Hertz-like edge, reaches zero,
    extrapolated from the last two loaded nodes; this resolves the edge below one node, unlike a
    count of loaded nodes. ``reach`` bounds the extrapolation beyond the last loaded sample [samples]:
    1 on a grid line (the next node is unloaded), 2 on an interpolated cut.
    """
    edges = []
    for step in (1, -1):
        k = i0
        while 0 <= k + step < p_cut.size and p_cut[k + step] > thr:
            k += step
        p2_k, p2_in = p_cut[k] ** 2, p_cut[k - step] ** 2 if k != i0 else p_cut[k] ** 2
        frac = p2_k / (p2_in - p2_k) if p2_in > p2_k else 0.5
        edges.append((k - i0) * h + step * min(frac, reach) * h)
    return (edges[0] - edges[1]) / 2.0


def _rel_err(num: float, ref: float) -> float:
    """``(num - ref) / ref``; ``nan`` when the reference is missing, infinite or zero."""
    return math.nan if ref is None or not math.isfinite(ref) or ref == 0 else (num - ref) / ref


@functools.cache
def _commits() -> dict:
    """HEAD commit of the axisforge, slippy and design-studies repositories, with ``"+dirty"``
    when tracked files other than ``.pyc`` differ from it (R-099); ``"unknown"`` when not
    readable (not a git checkout, git missing). Read once per process: a study run starts from a
    committed state; in a long interactive kernel the value may be stale."""
    import axisforge

    import slippy_bridge

    out = {}
    for key, module in (("axisforge", axisforge), ("slippy", slippy), ("design_studies", slippy_bridge)):
        try:
            git = ["git", "-C", str(Path(module.__file__).resolve().parent)]     # TypeError: no __file__
            head = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10,
                                  check=True).stdout.strip()
            status = subprocess.run([*git, "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                                    text=True, timeout=10, check=True).stdout.splitlines()
            # tracked .pyc files (AxisForge tracks some) change on every import: not a change of the code
            dirty = [line for line in status if not line.rstrip().endswith(".pyc")]
            out[key] = (head + ("+dirty" if dirty else "")) if head else "unknown"
        except (OSError, subprocess.SubprocessError, TypeError):
            out[key] = "unknown"
    return out
