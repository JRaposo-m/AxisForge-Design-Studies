"""
Regression of the shaft FEM validation cases against
``tests/fem_regression_values.py``.

Fast (minimal mesh only, ~1 s). The full convergence studies stay in the
case notebooks.
"""
from __future__ import annotations

import numpy as np
import pytest

from fem_regression_values import EXACT, FEM_MINIMAL_MESH, RTOL


@pytest.mark.parametrize("case", sorted(FEM_MINIMAL_MESH))
def test_fem_minimal_mesh(case, load_case):
    cd = load_case(case)
    system = cd.build_system()
    for theory, shafts in FEM_MINIMAL_MESH[case].items():
        res = cd.solve(system, cd.THEORIES[theory])
        for shaft, (v_max, M_max, n_nodes) in shafts.items():
            r = res[shaft]
            assert len(r.x_nodes) == n_nodes, (case, theory, shaft)
            assert r.v_max == pytest.approx(v_max, rel=RTOL), (case, theory, shaft)
            assert r.M_max == pytest.approx(M_max, rel=RTOL), (case, theory, shaft)


def test_exact_uniform(load_case):
    from references.beams import uniform_beam as an
    cd = load_case("fem_uniform_radial_load")
    p = cd.section_properties()
    xd = np.linspace(0.0, cd.SHAFT_LENGTH_MM, 2001)
    for ss in cd.build_system().shafts:
        sol = an.solve_shaft(ss, p.EI, an.kappa_cowper(p.nu) * p.GA)
        v_max, M_max = EXACT["fem_uniform_radial_load"][ss.name]
        assert float(np.max(sol.v(xd))) == pytest.approx(v_max, rel=RTOL)
        assert float(np.max(sol.M(xd))) == pytest.approx(M_max, rel=RTOL)


def test_exact_uniform_midspan(load_case):
    """Was the doctest of ``references.beams.uniform_beam.solve_shaft``."""
    from references.beams import uniform_beam as an
    cd = load_case("fem_uniform_radial_load")
    p = cd.section_properties()
    sol = an.solve_shaft(cd.build_system().shafts[0], p.EI, an.kappa_cowper(p.nu) * p.GA)
    assert round(float(sol.v(100.0)), 4) == 0.3763


def test_exact_stepped(load_case):
    from references.beams import stepped_beam as an
    cd = load_case("fem_stepped_distributed_gear")
    xd = np.linspace(0.0, cd.SHAFT_LENGTH_MM, 2001)
    for ss in cd.build_system().shafts:
        sol = an.solve_shaft(ss, cd.segments(), an.kappa_cowper(cd.POISSON), cd.SHAFT_LENGTH_MM)
        v_max, M_max = EXACT["fem_stepped_distributed_gear"][ss.name]
        assert float(np.max(sol.v(xd))) == pytest.approx(v_max, rel=RTOL)
        assert float(np.max(sol.M(xd))) == pytest.approx(M_max, rel=RTOL)
