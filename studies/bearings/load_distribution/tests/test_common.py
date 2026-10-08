"""
Unit tests of ``_common`` that do not need AxisForge.

The solver is replaced by a fake that returns a fixed load distribution, so these tests check
the plumbing (construction rule, sweep, rows, derived columns, files, compare validation),
not the mechanics. The mechanics are checked by test_regression.py against the stored data.
"""
from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace

import pytest

import _common.study as study_module
from _common.bearings import AngularContactSpec, CylindricalRollerSpec, DeepGrooveSpec
from _common.rows import add_derived, make_row, row_fields
from _common.store import check_compatible, read_results, write_results
from _common.study import LoadDistributionStudy
from _common.system import NOMINAL_POWER_W, SystemSpec, bearing_label

# =============================================================================
# Construction rule
# =============================================================================


def test_system_spec_accepts_the_allowed_pairs():
    SystemSpec(DeepGrooveSpec(), CylindricalRollerSpec())
    SystemSpec(AngularContactSpec(), CylindricalRollerSpec(), helix_angle_deg=20.0)


@pytest.mark.parametrize("locating, non_locating", [
    (CylindricalRollerSpec(), CylindricalRollerSpec()),
    (DeepGrooveSpec(), DeepGrooveSpec()),
    (CylindricalRollerSpec(), DeepGrooveSpec()),
])
def test_system_spec_rejects_a_bearing_in_the_wrong_slot(locating, non_locating):
    with pytest.raises(TypeError):
        SystemSpec(locating, non_locating)


def test_angular_contact_with_spur_gears_warns():
    with pytest.warns(UserWarning, match="floats"):
        SystemSpec(AngularContactSpec(), CylindricalRollerSpec(), helix_angle_deg=0.0)


def test_labels_do_not_depend_on_the_parameters():
    assert bearing_label("shaft_1", DeepGrooveSpec(s=0.0).kind, 1) == "shaft_1_ball_bearing_1"
    assert bearing_label("shaft_2", CylindricalRollerSpec().kind, 2) == "shaft_2_roller_bearing_2"


# =============================================================================
# Fake solver
# =============================================================================


def fake_bearing(spec, shaft_name):
    n = 1 if spec.arrangement == "locating" else 2
    return SimpleNamespace(label=bearing_label(shaft_name, spec.kind, n), position=10.0 * n,
                           arrangement=spec.arrangement, Z=spec.Z, C=5000.0,
                           s=getattr(spec, "s", math.nan), alpha_0=0.0, spec=spec)


def fake_analysis(Fr, s=0.0, line_contact=False):
    """A load distribution where delta_r = s/2 + Fr/1e5 and L10r = (1e4/Fr)^3."""
    Q = [Fr / 2.0, Fr / 4.0, 0.0, 0.0, Fr / 4.0]
    ring = SimpleNamespace(Q_j=Q, phi_j=[0.0, 1.2, 2.5, -2.5, -1.2], alpha_j=[0.0] * 5,
                           n_loaded=3, q_jk=[[1.0, 2.0, 1.0]] * 5, x_k=[-1.0, 0.0, 1.0])
    delta_r = s / 2.0 + Fr / 1e5
    stiffness = SimpleNamespace(Kr_xz=Fr / delta_r, Kr_xy=math.inf, delta_r_xz=delta_r,
                                delta_r_xy=0.0)
    ld = SimpleNamespace(row=ring, Fr=Fr, Fa=0.0, psi=1e-3, delta_r=delta_r, delta_a=0.0, Mz=0.0,
                         equilibrium_error=(0.0, 0.0), ok=True, residual=1e-12, n_iter=5,
                         stiffness=stiffness, is_line_contact=line_contact)
    return SimpleNamespace(load_distribution=ld, basic_life=SimpleNamespace(L10r=(1e4 / Fr) ** 3))


@dataclasses.dataclass
class FakeNode:
    """Stand-in for an AxisForge bearing node (a dataclass of scalars)."""
    label: str
    x: float
    Fr: float
    Fa: float
    psi_xz: float


@dataclasses.dataclass
class FakeShaftResults:
    factor: float                       # power / nominal power: Fr is proportional to it
    bearing_nodes: list


@pytest.fixture
def fake_axisforge(monkeypatch):
    """Replace every AxisForge call of the study loop by the fakes above."""
    shafts = [SimpleNamespace(name="shaft_1", speed_rpm=1000.0),
              SimpleNamespace(name="shaft_2", speed_rpm=500.0)]
    built = []

    def build(spec):
        built.append(spec.power_W)
        return SimpleNamespace(shafts=shafts, power_W=spec.power_W)

    monkeypatch.setattr(study_module, "build_system", build)
    def fem(system, key):
        factor = system.power_W / NOMINAL_POWER_W
        nodes = [FakeNode("b1", 10.0, 1000.0 * factor, 0.0, 2e-3),
                 FakeNode("b2", 190.0, 1000.0 * factor, 0.0, -2e-3)]
        return {ss.name: FakeShaftResults(factor, nodes) for ss in system.shafts}

    monkeypatch.setattr(study_module, "solve_fem", fem)
    monkeypatch.setattr(study_module, "assemble_bearing", fake_bearing)

    def solve(ss, bearing, fem, psi=None):
        line = bearing.spec.kind == "roller"
        return fake_analysis(1000.0 * fem.factor, getattr(bearing.spec, "s", 0.0), line), "", True

    monkeypatch.setattr(study_module, "solve_bearing", solve)
    monkeypatch.setattr(study_module, "axisforge_version", lambda: "test", raising=False)
    return built


class TwoAxisStudy(LoadDistributionStudy):
    question = "two_axis"
    axes = {"s_mm": [0.0, 0.02], "power_W": [0.5 * NOMINAL_POWER_W, NOMINAL_POWER_W,
                                             2.0 * NOMINAL_POWER_W]}

    def configure(self, spec, *, s_mm, **_):
        import dataclasses
        return dataclasses.replace(spec, s=s_mm)


# =============================================================================
# Sweep and rows
# =============================================================================


def test_sweep_is_the_cartesian_product_and_solves_only_the_bearing_under_study(fake_axisforge):
    data = TwoAxisStudy(DeepGrooveSpec()).run()
    assert len(data.rows) == 2 * 3 * 2                          # s x power x shafts
    assert {r["kind"] for r in data.rows} == {"ball"}
    assert [r["s_mm"] for r in data.rows[:6:2]] == [0.0, 0.0, 0.0]
    assert data.meta["axis_order"] == ["s_mm", "power_W"]
    assert data.meta["psi_source"] == "fem"
    assert data.lam_rows == []


def test_roller_under_study_gets_a_deep_groove_partner_and_lamina_rows(fake_axisforge):
    study = TwoAxisStudy(CylindricalRollerSpec())
    assert isinstance(study.partner, DeepGrooveSpec)
    data = study.run()
    assert {r["kind"] for r in data.rows} == {"roller"}
    assert data.lam_rows and not math.isnan(data.rows[0]["lamina_peak_over_mean"])


def test_partner_in_the_same_slot_is_rejected():
    with pytest.raises(ValueError):
        TwoAxisStudy(DeepGrooveSpec(), partner=AngularContactSpec())


def test_derived_columns_follow_the_fake_law(fake_axisforge):
    data = TwoAxisStudy(DeepGrooveSpec()).run()
    rows = [r for r in data.rows if r["shaft"] == "shaft_1" and r["s_mm"] == 0.0]
    for r in rows:
        assert r["Kr_tangent_N_per_mm"] == pytest.approx(1e5)    # dFr/d delta_r of the fake
        assert r["m_defl"] == pytest.approx(1.0)                 # delta_r proportional to Fr
        assert r["p_life"] == pytest.approx(3.0)                 # L10r ~ Fr^-3
    # the plane with ~0 displacement (inf) is not taken
    assert all(math.isfinite(r["Kr_N_per_mm"]) for r in data.rows)


def test_prescribed_psi_is_reported_as_prescribed():
    point = {"psi_mrad": 2.5}
    spec = DeepGrooveSpec()
    row, _, _ = make_row(point, "shaft_1", 1000.0, fake_bearing(spec, "shaft_1"), "ball",
                         fake_analysis(1000.0), "", True)
    assert row["psi_mrad"] == 2.5                               # the fake solver says 1.0
    assert list(row)[0] == "psi_mrad" and list(row) == row_fields(["psi_mrad"])


def test_one_system_and_fem_per_distinct_power(fake_axisforge):
    data = TwoAxisStudy(DeepGrooveSpec()).run()
    # nominal (psi record) + the three powers of the axis, the nominal one re-used
    assert sorted(fake_axisforge) == sorted(TwoAxisStudy.axes["power_W"])
    assert all(r["power_W"] in TwoAxisStudy.axes["power_W"] for r in data.rows)


def test_power_is_recorded_when_it_is_not_an_axis(fake_axisforge):
    class OneAxis(LoadDistributionStudy):
        question = "one_axis"
        axes = {"psi_mrad": [0.0, 1.0]}

    data = OneAxis(DeepGrooveSpec()).run()
    assert {r["power_W"] for r in data.rows} == {NOMINAL_POWER_W}
    assert data.meta["system"]["power_W"] == NOMINAL_POWER_W
    assert data.meta["psi_source"] == "imposed"


def test_system_spec_rejects_a_non_positive_power():
    with pytest.raises(ValueError, match="power_W"):
        SystemSpec(power_W=0.0)


def test_add_derived_without_load_axis_does_nothing():
    rows = [dict(shaft="a", label="b", s_mm=0.0, ok=True, Fr_N=1.0, delta_r_mm=1.0,
                 L10r_Mrev=1.0, Kr_tangent_N_per_mm=math.nan)]
    add_derived(rows, ["s_mm"])
    assert math.isnan(rows[0]["Kr_tangent_N_per_mm"])


# =============================================================================
# Files and compare
# =============================================================================


def _write(tmp_path, name, study_cls, spec):
    data = study_cls(spec).run()
    names = list(study_cls.axes)
    from _common.rows import lamina_fields, q_fields
    write_results(tmp_path / name / "results", data, row_fields=row_fields(names),
                  q_fields=q_fields(names), lamina_fields=lamina_fields(names), type_name=name)
    return data


def test_store_round_trip(fake_axisforge, tmp_path):
    data = _write(tmp_path, "deep_groove", TwoAxisStudy, DeepGrooveSpec())
    back = read_results(tmp_path / "deep_groove" / "results")
    assert len(back.rows) == len(data.rows)
    for a, b in zip(data.rows, back.rows):
        for key, value in a.items():
            if isinstance(value, float) and math.isnan(value):
                assert math.isnan(b[key]), key
            else:
                assert b[key] == value, key
    assert back.meta["parameter_hash"] == data.meta["parameter_hash"]


def test_compare_accepts_the_same_axes(fake_axisforge, tmp_path):
    _write(tmp_path, "deep_groove", TwoAxisStudy, DeepGrooveSpec())
    _write(tmp_path, "cylindrical_roller", TwoAxisStudy, CylindricalRollerSpec())
    check_compatible({n: read_results(tmp_path / n / "results")
                      for n in ("deep_groove", "cylindrical_roller")})


def test_compare_refuses_different_axes(fake_axisforge, tmp_path):
    class OtherAxes(TwoAxisStudy):
        axes = {"s_mm": [0.0, 0.03], "power_W": TwoAxisStudy.axes["power_W"]}

    _write(tmp_path, "deep_groove", TwoAxisStudy, DeepGrooveSpec())
    _write(tmp_path, "cylindrical_roller", OtherAxes, CylindricalRollerSpec())
    with pytest.raises(ValueError, match="axes"):
        check_compatible({n: read_results(tmp_path / n / "results")
                          for n in ("deep_groove", "cylindrical_roller")})


def test_figures_are_written(fake_axisforge, tmp_path):
    study = TwoAxisStudy(CylindricalRollerSpec())
    study.run_and_save(tmp_path / "cylindrical_roller")
    plots_dir = tmp_path / "cylindrical_roller" / "plots"
    assert {p.name for p in plots_dir.iterdir()} == {
        "summary", "load_distribution",
        "load_distribution_overview__power_W_5235.99.png",
        "load_distribution_overview__power_W_10472.png",
        "load_distribution_overview__power_W_20944.png",
        "roller_lamina_profile__power_W_5235.99.png",
        "roller_lamina_profile__power_W_10472.png",
        "roller_lamina_profile__power_W_20944.png"}
    assert len(list((plots_dir / "load_distribution").glob("*.png"))) == 2 * 3   # one per point
    assert (plots_dir / "summary" / "Q_max_over_mean.png").exists()
    results = tmp_path / "cylindrical_roller" / "results"
    assert {p.name for p in results.iterdir()} == {
        "report.txt", "system.csv", "fem_bearing_nodes.csv", "iso16281_rows.csv",
        "iso16281_elements.csv", "iso16281_laminae.csv"}


def test_report_sections_and_fem_table(fake_axisforge, tmp_path):
    TwoAxisStudy(DeepGrooveSpec()).run_and_save(tmp_path / "deep_groove")
    results = tmp_path / "deep_groove" / "results"
    text = (results / "report.txt").read_text(encoding="utf-8")
    for header in ("1  RUN INFORMATION", "2  SYSTEM", "3  SHAFT FEM", "4  ISO/TS 16281",
                   "5  CHECKS"):
        assert header in text
    assert "bearing 1 (locating, under study)" in text
    assert "bearing 2 (non-locating, partner)" in text
    fem = read_results(results).fem_rows
    assert len(fem) == 3 * 2 * 2                      # powers x shafts x bearing nodes
    assert {r["label"] for r in fem} == {"b1", "b2"}


# =============================================================================
# Axes per type
# =============================================================================


class PerTypeAxes(LoadDistributionStudy):
    question = "per_type"
    axes_by_kind = {"ball": {"s_mm": [0.0, 0.02]}, "roller": {"s_mm": [0.0, 0.01, 0.02]}}

    def configure(self, spec, **point):
        import dataclasses
        return dataclasses.replace(spec, s=point["s_mm"])


def test_axes_by_kind_selects_the_axes_of_the_type(fake_axisforge):
    assert PerTypeAxes(DeepGrooveSpec()).axes == {"s_mm": [0.0, 0.02]}
    assert len(PerTypeAxes(CylindricalRollerSpec()).run().rows) == 3 * 2


def test_a_type_missing_from_axes_by_kind_is_refused():
    # refused before the system is built, so the spur-gear warning is never reached
    with pytest.raises(ValueError, match="does not answer"):
        PerTypeAxes(AngularContactSpec())


def test_compare_on_a_result_column_accepts_different_axes(fake_axisforge, tmp_path):
    from _common.compare import compare_types
    for name, spec in (("deep_groove", DeepGrooveSpec()), ("cylindrical_roller",
                                                          CylindricalRollerSpec())):
        PerTypeAxes(spec).run_and_save(tmp_path / name)
    with pytest.raises(ValueError, match="axes"):       # same axes required by default
        compare_types(tmp_path, ["deep_groove", "cylindrical_roller"])
    compare_types(tmp_path, ["deep_groove", "cylindrical_roller"], x_columns=["s_mm"])
    assert (tmp_path / "compare" / "plots" / "vs_s_mm" / "n_loaded.png").exists()


def test_clearance_and_alpha_question_configures_each_type():
    from clearance_and_alpha.study import ClearanceAndAlphaStudy
    ball = ClearanceAndAlphaStudy(DeepGrooveSpec())
    assert list(ball.axes) == ["s_mm"]
    assert ball.configure(ball.spec, s_mm=0.03).s == 0.03
    with pytest.warns(UserWarning, match="floats"):
        angular = ClearanceAndAlphaStudy(AngularContactSpec())
    assert list(angular.axes) == ["alpha0_deg"]
    assert angular.configure(angular.spec, alpha0_deg=35.0).alpha_0_deg == 35.0


def test_compare_with_two_axes_draws_one_folder_per_value_of_the_second(fake_axisforge, tmp_path):
    from _common.compare import compare_types
    for name, spec in (("deep_groove", DeepGrooveSpec()), ("cylindrical_roller",
                                                          CylindricalRollerSpec())):
        TwoAxisStudy(spec).run_and_save(tmp_path / name)
    compare_types(tmp_path, ["deep_groove", "cylindrical_roller"])
    folders = {p.name for p in (tmp_path / "compare" / "plots" / "s_mm").iterdir()}
    assert folders == {"power_W_5235.99", "power_W_10472", "power_W_20944"}
    assert (tmp_path / "compare" / "plots" / "s_mm" / "power_W_10472" / "Q_max.png").exists()
