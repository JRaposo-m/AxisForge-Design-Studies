"""Result containers of the Richardson extrapolation / GCI solver: ``LevelRecord`` (one refinement
level), ``MetricConvergence`` (one metric over one triple of levels) and ``RichardsonResult`` (one run,
with its history). Data only: the computation is in ``numerical_solvers.richardson_extrapolation.richardson``,
referenced here by type only.

Metric values and extrapolated values are in the units of the caller's metric; relative differences,
observed orders and GCI are dimensionless.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .richardson import RichardsonOptions

__all__ = ["LevelRecord", "MetricConvergence", "RichardsonResult"]

_NAN = float("nan")


@dataclass(frozen=True, eq=False, kw_only=True)
class LevelRecord:
    """One evaluated refinement level.

    Parameters
    ----------
    index : int
        Position of the level in the run, 0 for the coarsest [-].
    level : float
        Refinement parameter given by the caller (a size ``h`` or a count ``n``, see
        ``RichardsonOptions.kind``) [units of the caller].
    ratio : float
        Refinement ratio to the previous level, ``r = level[k-1] / level[k]`` for a size and
        ``level[k] / level[k-1]`` for a count; NaN for the first level [-].
    metrics : dict of str to float
        Value of each metric returned by ``evaluate(level)`` [units of the metric].
    wall_time_s : float
        Wall time of the ``evaluate`` call [s].
    """

    index: int
    level: float
    ratio: float
    metrics: dict[str, float]
    wall_time_s: float


@dataclass(frozen=True, eq=False, kw_only=True)
class MetricConvergence:
    """Richardson extrapolation and GCI of one metric over one triple of levels (coarse, medium, fine).

    Every status other than ``"monotonic"`` and ``"flat"`` carries NaN numbers, ``converged=False``
    and a ``reason``.

    Parameters
    ----------
    metric : str
        Name of the metric; ``""`` from ``richardson_triple`` called directly.
    fine_index : int
        ``LevelRecord.index`` of the fine level of the triple; -1 from ``richardson_triple`` [-].
    f_coarse, f_medium, f_fine : float
        Metric values at the three levels [units of the metric].
    r_medium_coarse, r_fine_medium : float
        Refinement ratios of the two intervals [-].
    p : float
        Observed order ``ln((f_c - f_m) / (f_m - f_f)) / ln r`` after the ``min_p`` / ``max_p``
        clamp; NaN if ``"flat"`` or not computed [-].
    f_extrapolated : float
        Richardson extrapolation ``f_h0 = f_f + (f_f - f_m) / (r^p - 1)``; ``f_f`` if ``"flat"``
        [units of the metric].
    e_medium_coarse, e_fine_medium : float
        Relative differences ``(f_m - f_c) / f_c`` and ``(f_f - f_m) / f_m``, each divided by the
        coarser value of its pair [-].
    gci_medium_coarse, gci_fine_medium : float
        ``Fs |e| / (r^p - 1)`` of each interval [-].
    converged : bool
        True if both GCI are below ``RichardsonOptions.threshold`` (always True if ``"flat"``).
    status : str
        ``"monotonic"``, ``"flat"``, ``"oscillatory"``, ``"non-convergent"`` or ``"invalid"``.
    reason : str
        Why the triple is not monotonic or flat; ``""`` otherwise.
    """

    metric: str = ""
    fine_index: int = -1
    f_coarse: float
    f_medium: float
    f_fine: float
    r_medium_coarse: float
    r_fine_medium: float
    p: float = _NAN
    f_extrapolated: float = _NAN
    e_medium_coarse: float = _NAN
    e_fine_medium: float = _NAN
    gci_medium_coarse: float = _NAN
    gci_fine_medium: float = _NAN
    converged: bool = False
    status: str
    reason: str = ""


@dataclass(frozen=True, eq=False, kw_only=True)
class RichardsonResult:
    """Outcome of one ``richardson`` run.

    A run that did not converge is still returned (``on_failure="return"``): check ``converged`` and
    ``reason``, and ``final`` for the status of each metric.

    Parameters
    ----------
    name, label : str
        Identification of the analysis given by the caller (for example name ``"bem_grid"``, label
        ``"ball 1, P 1 kN"``); copied into ``record()`` and every row of ``records()``.
    options : RichardsonOptions
        Options of the run.
    levels : tuple of float
        Refinement parameters evaluated, coarse to fine [units of the caller].
    history : tuple of LevelRecord
        One record per evaluated level.
    convergence : dict of str to tuple of MetricConvergence
        Per judged metric, one entry per triple, in order (the first triple ends at level index 2).
    final : dict of str to MetricConvergence
        Per judged metric, the entry of the last triple (empty if fewer than three levels ran).
    converged : bool
        True if every judged metric is converged on the last triple.
    reason : str
        Why the run stopped: ``"converged"``, ``"levels_exhausted"`` (the sequence ended),
        ``"max_levels"`` or ``"callback"``.
    n_evaluations : int
        Calls of ``evaluate`` [-].
    """

    name: str
    label: str
    options: "RichardsonOptions"
    levels: tuple[float, ...]
    history: tuple[LevelRecord, ...]
    convergence: dict[str, tuple[MetricConvergence, ...]]
    final: dict[str, MetricConvergence]
    converged: bool
    reason: str
    n_evaluations: int

    def record(self) -> dict:
        """Return the summary of the run as a flat dict.

        Returns
        -------
        dict
            ``name``, ``label``, ``converged``, ``reason``, ``n_evaluations``, ``n_levels``, and per
            judged metric ``<metric>:f_extrapolated``, ``<metric>:p``, ``<metric>:gci_fine_medium``
            and ``<metric>:status`` of the last triple.
        """
        out = {"name": self.name, "label": self.label, "converged": self.converged,
               "reason": self.reason, "n_evaluations": self.n_evaluations,
               "n_levels": len(self.history)}
        for m, c in self.final.items():
            out |= {f"{m}:f_extrapolated": c.f_extrapolated, f"{m}:p": c.p,
                    f"{m}:gci_fine_medium": c.gci_fine_medium, f"{m}:status": c.status}
        return out

    def records(self) -> list[dict]:
        """Return the history, one flat dict per level and metric (long format).

        Returns
        -------
        list of dict
            ``name``, ``label``, ``index``, ``level``, ``ratio``, ``wall_time_s``, ``metric``,
            ``value`` and, for a judged metric from the third level on, the fields of the
            ``MetricConvergence`` of the triple ending at that level (NaN / ``""`` / False before)
            except those the row already has (``metric``, ``fine_index`` = ``index``, ``f_fine`` =
            ``value``, ``r_fine_medium`` = ``ratio``); the rows for
            ``axisforge.outputs._format.write_records_csv`` and convergence plots.
        """
        # OPEN R3-11: long format (one row per level and metric), fixed columns
        skip = ("metric", "fine_index", "f_fine", "r_fine_medium")
        empty = {f.name: "" if f.type == "str" else False if f.type == "bool" else _NAN
                 for f in fields(MetricConvergence) if f.name not in skip}
        by_level = {(c.metric, c.fine_index): c for cs in self.convergence.values() for c in cs}
        rows = []
        for h in self.history:
            for m, v in h.metrics.items():
                c = by_level.get((m, h.index))
                conv = empty if c is None else {k: x for k, x in asdict(c).items() if k not in skip}
                rows.append({"name": self.name, "label": self.label, "index": h.index,
                             "level": h.level, "ratio": h.ratio, "wall_time_s": h.wall_time_s,
                             "metric": m, "value": v, **conv})
        return rows
