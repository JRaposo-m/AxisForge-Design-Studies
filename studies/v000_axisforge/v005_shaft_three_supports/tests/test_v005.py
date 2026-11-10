"""Fast tests of v005: three supports through AxisForge vs r01 extended and Harris, fed-back sign, Hertz bearings."""
import math

import numpy as np
import pytest

from studies.v000_axisforge.v005_shaft_three_supports import (MODELS, SETTLEMENTS_MM, SPRINGS, check, literature,
                                                              three_support_shaft)
from studies.v000_axisforge.v005_shaft_three_supports.construction import SUPPORTS_MM
from studies.v000_axisforge.v005_shaft_three_supports.study import (first_check, hertz_flexible, literature_series,
                                                                    solve, solve_hertz)

EB, TIM = MODELS["euler_bernoulli"], MODELS["timoshenko/cowper/single_point"]


def _judged(rows):
    return [r for r in rows if r["verdict"] != "RECORD"]


def test_the_system_has_three_supports_and_two_loaded_planes():
    ss = three_support_shaft()
    assert sorted(b.position for b in ss.bearings) == list(SUPPORTS_MM)
    assert [b.arrangement for b in ss.bearings].count("locating") == 1


def test_fed_back_bearing_force_is_the_reported_reaction_r003():
    rows = first_check()
    assert all(r["verdict"] == "PASS" for r in _judged(rows)), _judged(rows)
    (d,) = [r for r in rows if r["id"] == "F0d"]
    assert d["value"] == pytest.approx(180.0, abs=1e-9)      # phi_Fr points away from the push on the bearing


def test_three_rigid_pins_give_harris_eqs_9_42_to_9_44():
    assert all(r["verdict"] == "PASS" for r in literature_series())


@pytest.mark.parametrize("settings, ids", [(EB, {"C1", "C2", "C3", "C3c"}), (TIM, {"C4", "C3", "C3c"})])
def test_springs_settlements_and_rotational_spring_match_r01_on_the_coarse_mesh(settings, ids):
    ss = three_support_shaft()
    r = solve(ss, settings, 27, springs=SPRINGS, settlements_mm=SETTLEMENTS_MM)
    rows = _judged(check(ss, r, settings, springs=SPRINGS, settlements_mm=SETTLEMENTS_MM))
    assert {r["id"] for r in rows} == ids and {r["plane"] for r in rows} == {"XY", "XZ"}
    assert all(r["verdict"] == "PASS" for r in rows), rows


def test_a_very_soft_middle_spring_leaves_two_supports():
    ss = three_support_shaft()
    r = solve(ss, EB, 27, springs={110.0: (1e-6, 0.0)})
    order = np.argsort([b.position for b in ss.bearings])
    assert np.asarray(r.R_xy)[order][1] == pytest.approx(0.0, abs=1e-6)


def test_harris_eq_9_48_is_equilibrium_and_rigid_compatibility():
    sp = literature.spans()
    F = literature.harris_eq_9_48(**sp)
    assert sum(F) == pytest.approx(sp["P"], rel=1e-13)
    assert F[1] * sp["l1"] + F[2] * (sp["l1"] + sp["l2"]) == pytest.approx(sp["P"] * sp["a"], rel=1e-13)
    d = [math.copysign(abs(f / 1e6) ** (1 / 1.5), f) for f in F]
    assert (d[0] - d[1]) / sp["l1"] == pytest.approx((d[1] - d[2]) / sp["l2"], rel=1e-10)


def test_hertz_newton_converges_and_is_self_consistent():
    rows, flex = hertz_flexible(n_elements=27)
    assert flex["newton"].converged
    assert all(r["verdict"] == "PASS" for r in _judged(rows)), _judged(rows)


def test_a_stiff_shaft_approaches_harris_eq_9_48():
    from studies.v000_axisforge.v005_shaft_three_supports.construction import LOAD_P, stiff_material
    from axisforge.core.loads import RadialLoad
    ss = three_support_shaft(loads=[RadialLoad(LOAD_P[0], LOAD_P[1], theta_deg=LOAD_P[2])],
                             material_id=stiff_material(1e4), label="t_stiff")
    _, res, prob = solve_hertz(ss, EB, 27)
    F = prob["force"](res.x)[np.argsort(prob["positions"])]
    assert res.converged
    assert F == pytest.approx(literature.harris_eq_9_48(**literature.spans()), abs=1e-3 * LOAD_P[1])


def test_a_spring_off_the_bearing_positions_is_refused():
    with pytest.raises(ValueError, match="not bearing positions"):
        solve(three_support_shaft(), EB, 27, springs={111.0: (1e4, 0.0)})
