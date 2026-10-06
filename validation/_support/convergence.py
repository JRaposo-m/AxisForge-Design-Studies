"""
validation/_support/convergence.py -- adapters for AxisForge's mesh
convergence machinery (``MeshConvergenceStudy``, ``SubmodelSolver``).

Only what the validation cases repeated verbatim lives here. The grade loop
itself stays in each notebook on purpose: it is the documented procedure of
that study (which quantities, which thresholds, when to stop), not plumbing.
Promote a loop here when two validation cases need the same one.

Extracted on 2026-10-06 from the two ``02_verification_*`` notebooks.
"""
from __future__ import annotations

import numpy as np

__all__ = ["ResultantView"]


class ResultantView:
    """Read-only view of a ``SubmodelResult`` adding the resultant fields.

    ``SubmodelResult`` (AxisForge) stores the two bending planes separately;
    ``MeshConvergenceStudy`` evaluates named fields. This view exposes

    * ``v_res = hypot(v_xz, v_xy)`` -- resultant deflection [mm];
    * ``M_res = hypot(M_xz, M_xy)`` -- resultant bending moment [N.mm];

    and forwards every other attribute to the wrapped result. Nothing in
    AxisForge is modified.

    Parameters
    ----------
    result : SubmodelResult
        As returned by ``SubmodelSolver.solve``.

    Notes
    -----
    The resultant is computed node by node; its maximum is not the maximum
    of either plane, and it is non-smooth where one plane changes sign.
    """

    def __init__(self, result):
        self._r = result
        self.x_nodes = result.x_nodes
        self.v_res = np.hypot(result.v_xz, result.v_xy)
        self.M_res = np.hypot(result.M_xz, result.M_xy)

    def __getattr__(self, name):
        return getattr(self._r, name)
