"""Result containers of the generic Newton-Raphson solver: ``IterationRecord`` (one iteration)
and ``NewtonResult`` (one solve, with its history). Data only: the computation is in
``numerical_solvers.newton_raphson.newton``, referenced here by type only. Ported from AxisForge
``axisforge/results/numerics/newton_raphson/newton_results.py`` at d34ec7f.

Norms stored here are norms of the scaled vectors ``F / f_scale`` and ``dx / x_scale`` of the
solve, so they are dimensionless when the scales carry the units of the problem.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:  # pragma: no cover
    from .newton import NewtonOptions

__all__ = ["IterationRecord", "NewtonResult"]

DC = dict(frozen=True, eq=False, kw_only=True)   # as axisforge.results._base.DC


@dataclass(**DC)
class IterationRecord:
    """State of the Newton iteration after iteration ``k`` (``k = 0`` is the initial point).

    Parameters
    ----------
    k : int
        Iteration number; 0 for the initial point, before any step.
    residual_norm : float
        Norm of the scaled residual ``F(x_k) / f_scale`` [-].
    relative_residual : float
        ``residual_norm`` divided by that of the initial point; NaN if the initial norm is 0 [-].
    step_norm : float
        Norm of the scaled step taken, ``(x_k - x_{k-1}) / x_scale``; NaN at ``k = 0`` [-].
    step_length : float
        Fraction of the Newton step taken: ``alpha * t``, with ``alpha`` 1, the damping factor
        or the line-search value, and ``t`` the ``max_step`` cap factor (1 if not capped); NaN
        at ``k = 0`` [-].
    condition : float
        2-norm condition number of the scaled Jacobian used for the step; NaN when it was not
        estimated (``k = 0``, sparse or large Jacobian, estimate switched off) [-].
    jacobian_evaluated : bool
        True if the Jacobian was evaluated (analytic call or finite differences) for this step.
    n_residual_evaluations : int
        Residual evaluations so far, finite-difference evaluations included [-].
    """

    k: int
    residual_norm: float
    relative_residual: float
    step_norm: float = float("nan")
    step_length: float = float("nan")
    condition: float = float("nan")
    jacobian_evaluated: bool = False
    n_residual_evaluations: int = 0


@dataclass(**DC)
class NewtonResult:
    """Outcome of one Newton-Raphson solve of ``F(x) = 0``.

    A solve that did not converge is still returned (``on_failure="return"``): check
    ``converged`` and ``reason``.

    Parameters
    ----------
    name, label : str
        Identification of the analysis given by the caller (for example name
        ``"shaft_bearing_coupled"``, label ``"case 3, Fr 5 kN"``); copied into the summary
        ``record()`` and every row of ``records()``.
    x : numpy.ndarray, shape (n,)
        Last iterate, in the units of the caller.
    F : numpy.ndarray, shape (n,)
        Residual at ``x``, in the units of the caller.
    converged : bool
        True if the stopping test was met (by Newton, or by the fallback when one was used).
    reason : str
        Why the solve stopped. Converged: the criteria met, joined by ``"+"``
        (``"residual_abs"``, ``"residual_rel"``, ``"step_abs"``, ``"step_rel"``). Not
        converged: ``"max_iter"``, ``"max_residual_evaluations"``, ``"singular_jacobian"``,
        ``"line_search_failed"``, ``"non_finite_residual"`` or ``"callback"``. After a
        fallback: ``"fallback_<method>"`` or ``"fallback_<method>_failed"``.
    newton_reason : str
        Reason of the Newton iteration itself; equal to ``reason`` when no fallback ran.
    fallback : str or None
        SciPy ``root`` method run after Newton failed (``"hybr"`` or ``"lm"``), None if none ran.
    residual_norm : float
        Norm of the scaled residual ``F / f_scale`` at the returned ``x`` (after the fallback,
        if one ran; ``history[-1].residual_norm`` is that of the last Newton iterate) [-].
    n_iter : int
        Newton iterations done (steps taken) [-].
    n_residual_evaluations : int
        Residual evaluations, finite-difference and fallback evaluations included [-].
    n_jacobian_evaluations : int
        Jacobian evaluations by Newton (analytic calls or finite-difference builds) [-].
    history : tuple of IterationRecord
        One record per iteration, ``history[0]`` for the initial point.
    options : NewtonOptions
        Options of the solve.
    """

    name: str
    label: str
    x: np.ndarray
    F: np.ndarray
    converged: bool
    reason: str
    newton_reason: str
    fallback: str | None
    residual_norm: float
    n_iter: int
    n_residual_evaluations: int
    n_jacobian_evaluations: int
    history: tuple[IterationRecord, ...]
    options: "NewtonOptions"

    def record(self) -> dict:
        """Return the summary of the solve as a flat dict with units-free keys.

        Returns
        -------
        dict
            ``name``, ``label``, ``converged``, ``reason``, ``newton_reason``, ``fallback``,
            ``n_iter``, ``n_residual_evaluations``, ``n_jacobian_evaluations`` and
            ``residual_norm``.
        """
        return {
            "name": self.name, "label": self.label,
            "converged": self.converged, "reason": self.reason,
            "newton_reason": self.newton_reason, "fallback": self.fallback,
            "n_iter": self.n_iter,
            "n_residual_evaluations": self.n_residual_evaluations,
            "n_jacobian_evaluations": self.n_jacobian_evaluations,
            "residual_norm": self.residual_norm,
        }

    def records(self) -> list[dict]:
        """Return the convergence history, one flat dict per iteration.

        Returns
        -------
        list of dict
            ``{"name", "label", **dataclasses.asdict(record)}`` for each entry of ``history``;
            the rows for ``axisforge.outputs`` tables and convergence plots.
        """
        return [{"name": self.name, "label": self.label, **asdict(r)} for r in self.history]
