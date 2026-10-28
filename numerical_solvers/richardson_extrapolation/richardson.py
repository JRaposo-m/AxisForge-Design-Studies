"""Richardson extrapolation and Grid Convergence Index (GCI) over a sequence of refinement levels.

Knows nothing about the model: the caller passes ``evaluate(level)``, which runs its model at one
refinement level (a BEM grid spacing, a number of elements, a time step, any scalar parameter) and returns
one or several scalar metrics. ``richardson`` evaluates the levels coarse to fine, judges every metric on
each new triple of consecutive levels and stops when all judged metrics are converged, when the levels
run out, or when the callback asks. Every choice is a field of ``RichardsonOptions``; the result is a
``RichardsonResult`` (``numerical_solvers.richardson_extrapolation``).

Formulas of one triple (``richardson_triple``), levels coarse ``f_c``, medium ``f_m``, fine ``f_f`` and
refinement ratio ``r`` (> 1, equal for both intervals):

- observed order ``p = ln((f_c - f_m) / (f_m - f_f)) / ln r``, clamped to ``[min_p, max_p]``;
- Richardson extrapolation ``f_h0 = f_f + (f_f - f_m) / (r^p - 1)``;
- relative differences ``e_mc = (f_m - f_c) / f_c``, ``e_fm = (f_f - f_m) / f_m`` (each divided by the
  coarser value of its pair);
- ``GCI_mc = Fs |e_mc| / (r^p - 1)``, ``GCI_fm = Fs |e_fm| / (r^p - 1)``; converged if both are below
  ``threshold``.

These are the formulas, checks and order of checks of AxisForge ``RichardsonGCI``
(``axisforge/solvers/mesh/convergence_solver.py``), so that the same inputs give the same numbers. The
change: where AxisForge raises (and ``_compute_gci`` returns a ``_DummyGCI`` without a reason), this
module returns a ``MetricConvergence`` with a status (``"oscillatory"``, ``"non-convergent"``,
``"invalid"``), NaN numbers, ``converged=False`` and a reason.

References
----------
Richardson (1911), the deferred approach to the limit; Roache (1998), Verification and Validation in
Computational Science and Engineering, Hermosa (GCI, safety factor 1.25 for three grids); Celik et al.
(2008), J. Fluids Eng. 130, 078001 (procedure for unequal ratios, not implemented). None is in the library
(source to find); the formulas are those of AxisForge ``RichardsonGCI``.

Examples
--------
A model with ``f(h) = 2 + 0.5 h^2`` on ``h = 0.2, 0.1, 0.05``:

>>> from numerical_solvers import richardson
>>> r = richardson(lambda h: 2.0 + 0.5 * h**2, [0.2, 0.1, 0.05], name="demo")
>>> r.converged, round(r.final["value"].p, 12), round(r.final["value"].f_extrapolated, 12)
(True, 2.0, 2.0)
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

from .richardson_results import LevelRecord, MetricConvergence, RichardsonResult

__all__ = ["RichardsonError", "RichardsonOptions", "richardson", "richardson_triple"]

_DEGENERATE = 1e-12                 # |r^p - 1| below this: no extrapolation, as AxisForge
_SCALAR = "value"                   # metric name when evaluate returns a float (OPEN R3-09)


@dataclass(frozen=True, kw_only=True)
class RichardsonOptions:
    """Options of ``richardson`` and ``richardson_triple``. Frozen; stored on the ``RichardsonResult``.

    Parameters
    ----------
    kind : {"size", "count"}
        What a level is: a size that decreases with refinement (``h``, a time step; ratio
        ``r = level[k-1] / level[k]``) or a count that increases (elements or intervals along one
        direction, ``r = level[k] / level[k-1]``; not grid points, not a total count in 2-D or 3-D:
        pass sizes then). Default ``"size"``.
    safety_factor : float
        Safety factor ``Fs`` of the GCI [-]. Positive. Default 1.25 (Roache, three grids).
    threshold : float
        Relative GCI below which an interval is converged (strict ``<``) [-]. Positive. Default 0.01.
    min_p, max_p : float or None
        Clamp of the observed order ``p`` before the extrapolation; they change ``f_h0`` and both GCI
        [-]. None: no clamp. Default None. Without ``max_p`` a plateau converges falsely: values
        ``10, 20, 20 + 1e-10`` give ``p = 36.5`` and GCI about 0 despite a 100 % coarse step; set
        ``max_p`` near the formal order of the method (``# OPEN R3-17``).
    flat_atol, flat_rtol : float
        A triple is flat when both level differences are at most
        ``max(flat_atol, flat_rtol * max(|f_c|, |f_m|, |f_f|))`` [units of the metric, -]. Defaults
        1e-8 and 1e-8 give the AxisForge tolerance ``max(1e-9, 1e-8 max(|f|, 1))``; ``flat_atol`` is in
        the units of the metric, so set it to 0 (or to a value in those units) for a metric much
        smaller than 1, e.g. an approach of 1e-5 m (``# OPEN R3-10``).
    ratio_rtol : float
        Two consecutive ratios are equal if ``|r_fm - r_mc| <= ratio_rtol * r_mc`` [-]. Default 1e-9
        (AxisForge compares them exactly, ``ratio_rtol = 0``).
    min_levels : int
        Levels evaluated before convergence may stop the run; at least 3 [-]. Default 3.
    max_levels : int or None
        Most levels evaluated; required when ``levels`` is a callable [-]. None: the length of the
        sequence. Default None.
    stop_when_converged : bool
        Stop as soon as every judged metric is converged on the last triple. False: evaluate every
        level and judge the last triple. Default True.
    metrics : sequence of str or None
        Names of the metrics judged; the others are recorded only. None: all. Default None.
    on_failure : {"return", "raise"}
        Return the result with ``converged=False`` (also after a callback stop), or raise
        ``RichardsonError`` carrying it. Default ``"return"``.

    Raises
    ------
    ValueError
        At construction, if an option is out of range (see ``validate``).
    """

    kind: str = "size"
    safety_factor: float = 1.25
    threshold: float = 0.01
    min_p: float | None = None
    max_p: float | None = None
    flat_atol: float = 1e-8
    flat_rtol: float = 1e-8
    ratio_rtol: float = 1e-9                     # OPEN R3-09: AxisForge compares exactly
    min_levels: int = 3
    max_levels: int | None = None
    stop_when_converged: bool = True             # OPEN R3-09: default True (stop), as newton
    metrics: Sequence[str] | None = None
    on_failure: str = "return"

    def __post_init__(self) -> None:
        """Store ``metrics`` as a tuple and fail fast: see ``validate``."""
        if self.metrics is not None and not isinstance(self.metrics, str):
            object.__setattr__(self, "metrics", tuple(self.metrics))
        self.validate_or_raise()

    def validate(self) -> list[str]:
        """Return the list of error messages; empty if the options are consistent."""
        e: list[str] = []
        if self.kind not in ("size", "count"):
            e.append(f"RichardsonOptions: kind must be 'size' or 'count', got {self.kind!r}")
        if not self.safety_factor > 0:
            e.append(f"RichardsonOptions: safety_factor must be > 0, got {self.safety_factor}")
        if not self.threshold > 0:
            e.append(f"RichardsonOptions: threshold must be > 0, got {self.threshold}")
        for nm in ("min_p", "max_p"):
            v = getattr(self, nm)
            if v is not None and not v > 0:
                e.append(f"RichardsonOptions: {nm} must be > 0 or None, got {v}")
        if self.min_p is not None and self.max_p is not None and not self.min_p <= self.max_p:
            e.append("RichardsonOptions: min_p must be <= max_p.")
        if not (self.flat_atol >= 0 and self.flat_rtol >= 0):
            e.append("RichardsonOptions: flat_atol and flat_rtol must be >= 0.")
        if not self.ratio_rtol >= 0:
            e.append(f"RichardsonOptions: ratio_rtol must be >= 0, got {self.ratio_rtol}")
        if self.min_levels < 3:
            e.append(f"RichardsonOptions: min_levels must be >= 3, got {self.min_levels}")
        if self.max_levels is not None and self.max_levels < max(self.min_levels, 3):
            e.append("RichardsonOptions: max_levels must be >= min_levels or None.")
        if self.metrics is not None and (isinstance(self.metrics, str) or len(self.metrics) == 0
                                         or not all(isinstance(m, str) for m in self.metrics)):
            e.append("RichardsonOptions: metrics must be a non-empty sequence of names or None.")
        if self.on_failure not in ("return", "raise"):
            e.append(f"RichardsonOptions: on_failure must be 'return' or 'raise', "
                     f"got {self.on_failure!r}")
        return e

    def validate_or_raise(self) -> None:
        """Raise ValueError with all messages of ``validate()`` if there are any."""
        errors = self.validate()
        if errors:
            raise ValueError("\n".join(errors))


class RichardsonError(RuntimeError):
    """Raised by ``richardson`` with ``on_failure="raise"`` when the run does not converge.

    Parameters
    ----------
    result : RichardsonResult
        The unconverged result, available as ``error.result``.
    """

    def __init__(self, result: RichardsonResult):
        """Store the result; the message names the analysis, the reason and the metric statuses."""
        statuses = ", ".join(f"{m} {c.status}" for m, c in result.final.items())
        super().__init__(f"richardson {result.name!r} {result.label!r} did not converge: "
                         f"{result.reason} ({statuses})")
        self.result = result


def richardson_triple(f_c: float, f_m: float, f_f: float, r: float | tuple[float, float], *,
                      options: RichardsonOptions | None = None) -> MetricConvergence:
    """Judge one metric over three consecutive levels: observed order, extrapolation and GCI.

    Parameters
    ----------
    f_c, f_m, f_f : float
        Metric values at the coarse, medium and fine levels [units of the metric].
    r : float or (float, float)
        Refinement ratio, or the pair ``(r_medium_coarse, r_fine_medium)`` [-].
    options : RichardsonOptions or None
        ``safety_factor``, ``threshold``, ``min_p``, ``max_p`` and ``ratio_rtol`` are used. None:
        ``RichardsonOptions()``.

    Returns
    -------
    MetricConvergence
        Status ``"monotonic"`` or ``"flat"`` with the numbers, or ``"oscillatory"``,
        ``"non-convergent"`` or ``"invalid"`` with NaN numbers, ``converged=False`` and a reason.
        ``metric`` is ``""`` and ``fine_index`` -1.

    Notes
    -----
    Checks in the order of AxisForge ``RichardsonGCI``: values finite (added here); both ratios above
    1 and equal within ``ratio_rtol``; flat field (both differences at most
    ``max(flat_atol, flat_rtol max(|f_c|, |f_m|, |f_f|))``: ``p`` NaN, GCI 0, ``f_h0 = f_f``, converged);
    ``f_m != f_f``; convergence ratio ``(f_c - f_m) / (f_m - f_f) > 0`` (else oscillatory); ``p > 0``
    before the clamp; ``|r^p - 1| >= 1e-12`` after it; ``f_c != 0`` and ``f_m != 0``. ``p`` and
    ``f_h0`` use ``r_fine_medium``; each GCI uses the ratio of its own interval.
    """
    opt = RichardsonOptions() if options is None else options
    r_mc, r_fm = (float(r), float(r)) if not isinstance(r, (tuple, list)) else map(float, r)
    f_c, f_m, f_f = float(f_c), float(f_m), float(f_f)
    base = dict(f_coarse=f_c, f_medium=f_m, f_fine=f_f, r_medium_coarse=r_mc, r_fine_medium=r_fm)

    def fail(status: str, reason: str) -> MetricConvergence:
        return MetricConvergence(**base, status=status, reason=reason)

    if not all(math.isfinite(v) for v in (f_c, f_m, f_f, r_mc, r_fm)):
        return fail("invalid", "a metric value or a ratio is not finite")
    if not (r_mc > 1.0 and r_fm > 1.0):
        return fail("invalid", f"refinement ratios must be > 1, got r_mc={r_mc:.6g}, r_fm={r_fm:.6g}")
    if abs(r_fm - r_mc) > opt.ratio_rtol * r_mc:         # OPEN R3-12: Celik 2008 iteration not implemented
        return fail("invalid", f"unequal refinement ratios r_mc={r_mc:.6g}, r_fm={r_fm:.6g} "
                               f"(ratio_rtol {opt.ratio_rtol:g}; unequal ratios not implemented)")
    flat_tol = max(opt.flat_atol, opt.flat_rtol * max(abs(f_c), abs(f_m), abs(f_f)))
    if abs(f_m - f_c) <= flat_tol and abs(f_f - f_m) <= flat_tol:
        return MetricConvergence(**base, f_extrapolated=f_f, e_medium_coarse=0.0, e_fine_medium=0.0,
                                 gci_medium_coarse=0.0, gci_fine_medium=0.0, converged=True,
                                 status="flat")
    if f_m == f_f:
        return fail("invalid", "f_medium == f_fine with f_coarse different: convergence ratio "
                               "undefined")
    ratio = (f_c - f_m) / (f_m - f_f)
    if ratio <= 0.0:
        return fail("oscillatory", f"level differences change sign (convergence ratio {ratio:.6g} <= 0)")
    p = math.log(ratio) / math.log(r_fm)
    if p <= 0.0:
        return fail("non-convergent", f"observed order p={p:.6g} <= 0 (convergence ratio "
                                      f"{ratio:.6g} <= 1: differences do not decrease)")
    if opt.min_p is not None:
        p = max(p, opt.min_p)
    if opt.max_p is not None:
        p = min(p, opt.max_p)
    if abs(r_fm**p - 1.0) < _DEGENERATE:
        return fail("non-convergent", f"r^p - 1 ~ 0 (p={p:.6g}): no extrapolation")
    if f_c == 0.0 or f_m == 0.0:
        return fail("invalid", "zero coarser value: relative difference undefined "
                               f"(f_coarse={f_c:.6g}, f_medium={f_m:.6g})")
    e_mc = (f_m - f_c) / f_c
    e_fm = (f_f - f_m) / f_m
    gci_mc = opt.safety_factor * abs(e_mc) / (r_mc**p - 1)
    gci_fm = opt.safety_factor * abs(e_fm) / (r_fm**p - 1)
    return MetricConvergence(**base, p=p, f_extrapolated=f_f + (f_f - f_m) / (r_fm**p - 1),
                             e_medium_coarse=e_mc, e_fine_medium=e_fm, gci_medium_coarse=gci_mc,
                             gci_fine_medium=gci_fm,
                             converged=gci_mc < opt.threshold and gci_fm < opt.threshold,
                             status="monotonic")


def richardson(evaluate: Callable[[float], float | Mapping[str, float]],
               levels: Sequence[float] | Callable[[int], float], *,
               options: RichardsonOptions | None = None,
               callback: Callable[[LevelRecord], bool | None] | None = None,
               name: str = "", label: str = "") -> RichardsonResult:
    """Evaluate a model on refinement levels coarse to fine and judge its convergence by Richardson / GCI.

    Parameters
    ----------
    evaluate : callable
        ``evaluate(level) -> float or Mapping[str, float]``: runs the caller's model at one level and
        returns one metric (recorded as ``"value"``) or several by name [units of each metric]. Called
        once per level, in order; every call must return the same names.
    levels : sequence of float or callable
        The refinement parameters, coarse to fine (at least 3), or ``refine(k) -> level`` for
        ``k = 0, 1, ...`` (adaptive; ``options.max_levels`` required) [units of the caller]. What a
        level is: ``options.kind``.
    options : RichardsonOptions or None
        None: ``RichardsonOptions()``.
    callback : callable or None
        ``callback(record)`` after each level with its ``LevelRecord``; returning True stops the run
        with reason ``"callback"`` (unless it stopped converged), which counts as not converged.
    name, label : str
        Identification of the analysis, stored on the result and copied into every row of
        ``RichardsonResult.records()``.

    Returns
    -------
    RichardsonResult
        Levels, history, per-metric convergence of every triple, final status and reason.

    Raises
    ------
    ValueError
        If ``levels`` has fewer than ``options.min_levels`` entries, a callable ``levels`` comes without
        ``options.max_levels``, ``evaluate`` returns other names than at the first level or a
        non-scalar value, or a judged metric is missing.
    RichardsonError
        If the run does not converge (callback stop included) and ``options.on_failure == "raise"``.

    Notes
    -----
    From the third level on, every judged metric is passed to ``richardson_triple`` with the last
    three values and the ratios of the last two intervals. The run is converged when every judged
    metric is converged on the last triple; a failed status (oscillatory, non-convergent, invalid) on
    one triple does not stop the run, a later triple may converge.
    """
    opt = RichardsonOptions() if options is None else options
    n_seq = None                         # length of a sequence of levels; None if adaptive
    if callable(levels):
        if opt.max_levels is None:
            raise ValueError(f"richardson {name!r}: a callable levels needs options.max_levels.")
        n_max, level_at = opt.max_levels, levels
    else:
        seq = [float(v) for v in levels]
        n_seq = len(seq)
        if len(seq) < opt.min_levels:
            raise ValueError(f"richardson {name!r}: at least min_levels = {opt.min_levels} levels are "
                             f"needed, got {len(seq)}.")
        n_max = len(seq) if opt.max_levels is None else min(len(seq), opt.max_levels)
        level_at = seq.__getitem__

    history: list[LevelRecord] = []
    convergence: dict[str, list[MetricConvergence]] = {}
    judged: tuple[str, ...] = ()
    converged, reason = False, None
    while reason is None:
        k = len(history)
        level = float(level_at(k))
        if not (level > 0 and math.isfinite(level)):
            raise ValueError(f"richardson {name!r}: level {k} is {level}; levels must be positive.")
        t0 = time.perf_counter()
        out = evaluate(level)
        wall = time.perf_counter() - t0
        values = {_SCALAR: float(out)} if not isinstance(out, Mapping) else \
            {str(m): float(v) for m, v in out.items()}
        if k == 0:
            judged = tuple(values) if opt.metrics is None else tuple(opt.metrics)
            missing = [m for m in judged if m not in values]
            if missing:
                raise ValueError(f"richardson {name!r}: judged metrics {missing} not returned by "
                                 f"evaluate (names: {list(values)}).")
            convergence = {m: [] for m in judged}
        elif values.keys() != history[0].metrics.keys():
            raise ValueError(f"richardson {name!r}: evaluate returned {list(values)} at level {k}, "
                             f"{list(history[0].metrics)} at level 0.")
        ratio = float("nan") if k == 0 else \
            (history[-1].level / level if opt.kind == "size" else level / history[-1].level)
        if not (k == 0 or ratio > 1.0):
            raise ValueError(f"richardson {name!r}: ratio {ratio:.6g} at level {k} is not > 1: levels "
                             f"must refine (decreasing for kind 'size', increasing for 'count').")
        rec = LevelRecord(index=k, level=level, ratio=ratio, metrics=values, wall_time_s=wall)
        history.append(rec)
        if k >= 2:
            pair = (history[-2].ratio, ratio)
            for m in judged:
                f_c, f_m, f_f = (h.metrics[m] for h in history[-3:])
                convergence[m].append(replace(richardson_triple(f_c, f_m, f_f, pair, options=opt),
                                              metric=m, fine_index=k))
            converged = all(convergence[m][-1].converged for m in judged)
        if converged and opt.stop_when_converged and k + 1 >= opt.min_levels:
            reason = "converged"
        elif callback is not None and callback(rec):     # OPEN R3-16: a stop counts as not converged
            converged, reason = False, "callback"
        elif k + 1 >= n_max:
            reason = "converged" if converged else \
                ("levels_exhausted" if n_max == n_seq else "max_levels")

    final = {m: c[-1] for m, c in convergence.items() if c}
    result = RichardsonResult(name=name, label=label, options=opt,
                              levels=tuple(h.level for h in history), history=tuple(history),
                              convergence={m: tuple(c) for m, c in convergence.items()},
                              final=final, converged=converged, reason=reason,
                              n_evaluations=len(history))
    if not converged and opt.on_failure == "raise":
        raise RichardsonError(result)
    return result
