"""
Shaft FEM and the ISO/TS 16281 solve of ONE bearing.

Method
------
The load is never scaled after the fact: every load level is a system built with its own
transmitted power (``SystemSpec.power_W``) and its own FEM solve. With rigid supports the FEM
does not depend on the bearings, so a study solves it once per distinct power and re-uses it
for every bearing parameter at that power. The contact solver reads only the bearing nodes.
"""
from __future__ import annotations

from .system import beam_model


def solve_fem(system, theory_key: str = "timoshenko/cowper") -> dict:
    """Solve the rigid-support shaft FEM of every shaft once.

    Parameters
    ----------
    system: SpurHelicalGearSystem
        From ``system.build_system``.
    theory_key: str
        Key of ``system.THEORY_ARGS``.

    Returns
    -------
    fem: dict
        {shaft name: ShaftResults}; the bearing nodes hold Fr, Fa and the slopes psi.
    """
    from axisforge.solvers.machine_elements.shaft.fem_solvers.global_solver.rigid_support import (
        RigidSupportFEMSolver,
    )
    settings = beam_model(theory_key)
    return {ss.name: RigidSupportFEMSolver(settings).solve(ss) for ss in system.shafts}


def solve_bearing(shaft_system, bearing, shaft_results, psi: float | None = None):
    """Run the ISO/TS 16281 contact solver on ONE bearing.

    The post-processing (stiffness, life) can fail on its own, for example a bearing with no
    loaded element has no finite life; in that case the solve is repeated without it so the
    load distribution is kept.

    Parameters
    ----------
    shaft_system: ShaftSystem
        The shaft that carries the bearing.
    bearing: Bearing
        Its ``label`` must match a bearing node of ``shaft_results``.
    shaft_results: ShaftResults
        FEM results of the system at the power of this point.
    psi: float, optional (None)
        Inner-ring tilt [rad] to prescribe; None keeps the FEM slope.

    Returns
    -------
    analysis: BearingAnalysisResult or None
        None if the solver failed.
    error: str
        "" if all went well, otherwise what failed.
    postprocessed: bool
        False if ``analysis`` has no stiffness / life.
    """
    # importing contact_solver registers the ball and roller solvers with the dispatcher
    from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281 import (  # noqa: F401
        contact_solver,
    )
    from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281.dispatch import (
        resolve_solver_cls,
    )

    label = bearing.label
    solver_cls = resolve_solver_cls(bearing, label=label)
    psi_override = None if psi is None else {label: psi}
    try:
        results = solver_cls(psi_input=psi is not None, postprocess=True).solve(
            shaft_system, {label: bearing}, shaft_results, psi_override=psi_override)
        return results[label], "", True
    except Exception as exc_full:                                   # noqa: BLE001
        try:
            results = solver_cls(psi_input=psi is not None, postprocess=False).solve(
                shaft_system, {label: bearing}, shaft_results, psi_override=psi_override)
            return (results[label],
                    f"postprocess failed: {type(exc_full).__name__}: {exc_full}", False)
        except Exception as exc:                                    # noqa: BLE001
            return None, f"{type(exc).__name__}: {exc}", False