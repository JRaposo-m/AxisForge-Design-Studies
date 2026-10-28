"""Fast tests of v001: the system, the comparator on a coarse mesh, the reduced-integration identity, the literature."""
import math

import numpy as np
import pytest
from axisforge.core.loads import LoadPlane

from studies.v000_axisforge.v001_shaft_uniform import (THEORIES, check, kga_reduced_integration, literature, section,
                                                       uniform_shaft)
from studies.v000_axisforge.v001_shaft_uniform.study import _node, solve


@pytest.fixture(scope="module")
def system():
    return uniform_shaft()


def test_the_system_is_returned_unsolved_with_one_force_per_plane(system):
    assert [b.position for b in system.bearings] == [25.0, 175.0]
    assert [b.arrangement for b in system.bearings] == ["locating", "non-locating"]
    (F1, F2) = system.radial_loads
    assert F1.component(LoadPlane.XZ) == pytest.approx(-500.0) and F2.component(LoadPlane.XY) == pytest.approx(200.0)


@pytest.mark.parametrize("theory", ["euler_bernoulli", "timoshenko/cowper/single_point"])
def test_check_passes_on_the_coarsest_mesh(system, theory):
    settings = THEORIES[theory]
    rows = check(system, solve(system, settings, 8), settings)
    judged = [r for r in rows if r["verdict"] != "RECORD"]
    expected = {"C1", "C2", "C3"} if theory == "euler_bernoulli" else {"C3", "C4"}
    assert {r["id"] for r in judged} == expected and {r["plane"] for r in judged} == {"XZ", "XY"}
    assert all(r["verdict"] == "PASS" for r in judged), judged


def test_one_point_timoshenko_error_is_second_order(system):
    """T1 (RECORD) halves the element and divides the nodal error by 4 (kga_reduced_integration)."""
    settings = THEORIES["timoshenko/cowper/single_point"]
    err = [max(r["rel_err"] for r in check(system, solve(system, settings, n), settings)
               if r["id"] == "T1" and r["quantity"] == "v") for n in (8, 16)]
    assert err[0] / err[1] == pytest.approx(4.0, rel=0.02)


def test_kga_reduced_integration_gives_onate_eq_2_41():
    """One element, cantilever: h/kGA_eff + h^3/(3 EI) = h/kGA + h^3/(4 EI) (Onate Eq. (2.41), PDF p. 84)."""
    EI, kGA, h = 1.6e9, 2.2e7, 50.0
    k = kga_reduced_integration(kGA, EI, h)
    assert h / k + h**3 / (3 * EI) == pytest.approx(h / kGA + h**3 / (4 * EI), rel=1e-14)


def test_section_uses_the_override_for_timoshenko_only(system):
    EI, kGA = section(system, THEORIES["timoshenko/cowper/single_point"], kGA_override=1.0)
    assert EI == pytest.approx(210000.0 * math.pi * 20.0**4 / 64.0) and kGA == 1.0
    assert section(system, THEORIES["euler_bernoulli"], kGA_override=1.0)[1] is None


def test_onate_table_2_1_two_elements():
    """r_w = 0.9375 for 2 elements (printed 0.938, PDF p. 84), lambda -> inf."""
    case = literature.cases()[0]
    s = case.build()
    EI, _ = section(s, THEORIES["euler_bernoulli"])
    L, P = literature.HALF_LENGTH_MM, literature.TIP_FORCE_N
    r = solve(s, THEORIES["timoshenko/cowper/single_point"], 4, kGA_override=literature.SLENDER_KGA_FACTOR * EI / L**2)
    assert _node(r, L) / (-P * L**3 / (3 * EI)) == pytest.approx(0.9375, abs=1e-5)


def test_onate_fig_1_17_central_deflection_in_mm():
    """|P| L^3 / (48 EI) = 131.25 m with Onate's printed I (Fig. 1.17, PDF p. 65), on 2 elements."""
    s = literature.cases()[1].build("point")
    r = solve(s, THEORIES["euler_bernoulli"], 2)
    assert _node(r, 2000.0) / literature.M_TO_MM == pytest.approx(-4.0**3 / (48 * 47619.05 * 2.133e-7), rel=1e-9)


def test_literature_rows_carry_their_pages():
    cases = literature.cases()
    assert [c.name for c in cases] == ["onate2013_table_2_1", "onate2013_table_1_1", "onate2013_table_2_3"]
    assert [r["value"] for r in cases[0].published] == [0.75, 0.938, 0.984, 0.996, 0.999]
    assert all(r["pdf_page"] and r["source_key"] == "Onate2013_vol2" for c in cases for r in c.published)


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        literature.onate_fig_1_17("couple")
    with pytest.raises(ValueError):
        uniform_shaft(supports_mm=(25.0, 250.0))
    assert np.isfinite(kga_reduced_integration(1e7, 1e9, 1.0))


def test_c4_is_record_with_a_distributed_load():
    """kGA_eff holds for point loads only (R5-25): with a uniform load C4 is RECORD, not a false MISS."""
    s = literature.onate_table_2_3(0.1)
    settings = THEORIES["timoshenko/cowper/single_point"]
    rows = check(s, solve(s, settings, 8, kGA_override=literature.kga_square(0.1)), settings,
                 kGA_override=literature.kga_square(0.1))
    assert {r["verdict"] for r in rows if r["id"] == "C4"} == {"RECORD"}
