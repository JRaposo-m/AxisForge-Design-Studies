"""
validation/fem_studies/resolution/uniform_shafts/point_load/case_radial_single_position/case_radial_single_position.py

Case group: uniform shaft, ONE extra point RadialLoad, position varied,
solved under BOTH beam theories (Euler-Bernoulli and Timoshenko). See
the suite's top-level README.md for the euler_bernoulli/ vs
timoshenko/ vs analytical_solution/ split, and for the "vehicle, not
mechanism" discipline shared by every case script.

REORG NOTE (this pass): previously this file only solved Timoshenko
(shaft_fem=("shaft_fem.timoshenko_rigid",)) and wrote into
results_dir(__file__)/<shear_theory>/ with NO theory segment in the
path -- which never actually matched the on-disk layout
(case_radial_single_position/{euler_bernoulli,timoshenko}/results/...,
theory as a real folder). Whatever previously populated
euler_bernoulli/results/ was NOT this script (there was no code path
here that ever called shaft_fem.euler_bernoulli_rigid), and the
timoshenko/results/ output on disk predates this rewrite -- both are
now regenerated correctly by the loop in main() below, each branch
building its own case_dir(__file__) / "<theory>" / "results" path
directly (NOT results_dir(__file__, theory) -- that helper joins its
extra segments AFTER "results", by design, matching how
compare_case_x.py already relies on it; here <theory>/ sits ABOVE
results/ instead, the opposite nesting -- see common/paths.py's own
docstring, and the inline comment at the first case_dir() call below).

A single 1-stage gear chain (2 shafts) is built purely to get two
independent, individually-resolved ShaftSystem objects out of one script
run; the gear mesh itself is a SPUR pair (zero axial thrust, so it never
pollutes the axial DOF that the axial/moment case group cares about)
with small, FIXED, point Fr/Ft -- present identically in every case in
this file, not what is being compared.

What actually varies between the two shafts is the POSITION of one
explicit RadialLoad (core.loads.RadialLoad, source="user"):

    shaft1  extra RadialLoad @ x= 60.0 mm  (near the locating/ball end)
    shaft2  extra RadialLoad @ x=140.0 mm  (near the non-locating/roller end)

Same magnitude/direction on both (500 N, theta=270 deg) -- only the axial
position changes, so a diff against the two equivalent Abaqus models
isolates the effect of load position on v(x)/M(x)/bearing reactions,
with everything else (BC, section, gear-mesh background load) held fixed.

BC: fixed for this whole suite -- ball (locating) @10mm, roller
(non-locating) @190mm -- see common/bearings.py. Both bearings
constrain v=0 only (u=0 additionally at the locating one); NEITHER ever
constrains theta -- see solvers/.../constraints/boundary_conditions.py.
That makes both supports true pins with no moment reaction, which is
exactly what analytical_solution/'s closed-form beam formulas assume --
see that script's own docstring.

integration_method="single_point" (selective/reduced integration) for
the Timoshenko branch -- this is a resolution study, not the
locking-demonstration integration_method="exact" used by the mesh
convergence group. Euler-Bernoulli has no shear term, so
integration_method does not apply to that branch at all.

Outputs, per theory:
  euler_bernoulli/results/               (report .txt, plots/*.png, csv/*.csv)
                                          -- no <shear_theory>/ subfolder:
                                          Euler-Bernoulli has no shear
                                          correction to sweep, exactly
                                          one AxisForge solve per shaft.
  timoshenko/results/<shear_theory>/     (report .txt, plots/*.png, csv/*.csv)
                                          -- one subfolder per shear_theory
                                          swept (cowper, hutchinson).
Both relative to this case's own folder -- see common/paths.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

_p = Path(__file__).resolve()
_root = None
for _ancestor in _p.parents:
    if (_ancestor / "common").is_dir():
        _root = _ancestor
        break
if _root is None:
    raise RuntimeError(
        f"{__file__}: no ancestor directory containing 'common/' "
        f"found -- this file must live somewhere inside the suite tree."
    )
sys.path.insert(0, str(_root))
from common.bearings import build_case_bearings  # noqa: E402
from common.paths import case_dir  # noqa: E402

from axisforge.core.loads import RadialLoad, TorqueLoad  # noqa: E402
from axisforge.fixtures.construction.construction_capabilities import ConstructionCapabilities  # noqa: E402
from axisforge.fixtures.construction.outputs.text_report import write_construction_report  # noqa: E402
from axisforge.fixtures.studies.study_capabilities import StudyCapabilities  # noqa: E402
from axisforge.fixtures.studies.outputs.text_report import write_studies_report  # noqa: E402
from axisforge.fixtures.studies.shafts.fem_studies.outputs.plots import write_resolution_plots  # noqa: E402
from axisforge.fixtures.studies.shafts.fem_studies.outputs.resolution_csv import resolution_csv  # noqa: E402


SHAFT_LENGTH_MM = 200.0
SHAFT_DIAMETER_MM = 20.0

LOAD_N = 500.0
LOAD_THETA_DEG = 270.0
LOAD_X_SHAFT1_MM = 60.0
LOAD_X_SHAFT2_MM = 140.0

# Timoshenko-branch integration method -- "exact" is the other option,
# but this is a resolution study, not the locking-demonstration group.
# Not used at all by the Euler-Bernoulli branch (no shear term).
TIMOSHENKO_INTEGRATION_METHOD = "single_point"

# Shear theories swept for the Timoshenko branch only.
SHEAR_THEORIES = ("cowper", "hutchinson")


def build_system() -> tuple["ConstructionCapabilities", "SpurHelicalGearSystem"]:
    """
    Builds and RESOLVES the ShaftSystem for this case -- geometry,
    bearings, gear mesh, and every load (user + gear_mesh-sourced),
    independent of which beam theory will later solve it (bending
    theory is a Studies-stage choice, made in main() below, not here).

    Exposed (not just called from main()) specifically so that
    analytical_solution/analytical_case_radial_single_position.py can
    import and call this SAME function to get the identical, already-
    resolved ShaftSystem -- same geometry, same bearing positions, same
    gear-mesh Ft/Fr (a core-level statics computation, not FEM) -- for
    its own closed-form (non-FEM) solve, rather than re-deriving any of
    those numbers by hand and risking a second, independently-wrong
    source of truth.
    """
    construction = ConstructionCapabilities(
        shaft=("shafts.generic",),
        bearings=("bearings.deep_groove_ball", "bearings.cylindrical_roller"),
        gears=("gears.spur",),
        system=("systems.parallel_axis_linear",),
    )
    objs = construction.resolve()

    make_shaft = objs["factory"]
    SectionSpec = objs["SectionSpec"]
    make_deep_groove_ball_bearing = objs["make_deep_groove_ball_bearing"]
    make_cylindrical_roller_bearing = objs["make_cylindrical_roller_bearing"]
    make_spur_gear = objs["make_spur_gear"]
    ShaftSpec = objs["ShaftSpec"]
    StageSpec = objs["StageSpec"]
    build_linear_system = objs["build_linear_system"]

    def make_shaft_geometry(name: str):
        return make_shaft(
            sections=[SectionSpec(length=SHAFT_LENGTH_MM, diameter=SHAFT_DIAMETER_MM, label=name)],
            transitions={},
            name=name,
        ).shaft

    shaft1 = make_shaft_geometry("shaft1")
    shaft2 = make_shaft_geometry("shaft2")
    bearings1 = build_case_bearings(make_deep_groove_ball_bearing, make_cylindrical_roller_bearing, "shaft1")
    bearings2 = build_case_bearings(make_deep_groove_ball_bearing, make_cylindrical_roller_bearing, "shaft2")

    g_driver = make_spur_gear(mn=2.0, z=20, b=15.0, position=100.0, label="g_driver")
    g_driven = make_spur_gear(mn=2.0, z=40, b=15.0, position=100.0, label="g_driven")

    load1 = RadialLoad(LOAD_X_SHAFT1_MM, LOAD_N, theta_deg=LOAD_THETA_DEG, label="radial_test_load")
    load2 = RadialLoad(LOAD_X_SHAFT2_MM, LOAD_N, theta_deg=LOAD_THETA_DEG, label="radial_test_load")

    shaft_specs = [
        ShaftSpec(shaft=shaft1, bearings=bearings1, speed_rpm=1000,
                  loads=(load1,), name="shaft1"),
        ShaftSpec(shaft=shaft2, bearings=bearings2, speed_rpm=500,
                  loads=(load2,), name="shaft2"),
    ]
    stage_specs = [
        StageSpec(gear_driver=g_driver, gear_driven=g_driven, phi_deg=0.0, label="stage1"),
    ]

    system = build_linear_system(
        shaft_specs, stage_specs, P=10471.9755, rotation_dir_source=1,
        label="uniform_point_load_radial_single_position",
    )

    # shaft2 (driven shaft) only receives the gear-mesh reaction torque
    # (source="gear_mesh", at x=100mm) -- nothing balances that torque
    # on this shaft. validate_torsion_equilibrium() would flag this,
    # and an Abaqus model built from the construction report ends up
    # with DOF 4 (torsion) unconstrained -- zero pivot in the solver.
    #
    # Driven-machine torque sink, placed on the far side of the gear
    # (x=200mm > 100mm, so AxisForge's own T(x) closes back to zero past
    # the application point -- see solve_torsion). Magnitude is derived
    # from the net already present rather than hand-copied.
    shaft2_sys = system.shafts[-1]
    net_before = sum(ld.magnitude for ld in shaft2_sys.torque_loads)
    shaft2_sys.add_load(TorqueLoad(
        position=SHAFT_LENGTH_MM,
        magnitude=-net_before,
        label="output_coupling",
        source="user",
    ))

    return construction, system


def _write_theory_outputs(system, out_dir: Path, library, title: str) -> None:
    """Report + plots + csv for one already-solved library, shared by
    both theory branches in main() below (the only thing that differs
    between them is which capability solved `library` and where
    `out_dir` points)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    write_studies_report(
        system, out_dir / "report_resolution.txt",
        title=title, shaft_fem_library=library,
    )
    write_resolution_plots(library, system, out_dir)

    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    for ss in system.shafts:
        r = library.get_or_none(ss.name)
        if r is None:
            continue
        (csv_dir / f"{ss.name}_resolution.csv").write_text(
            resolution_csv(r, decimals=10), encoding="utf-8",
        )
    print(f"[OK] report + plots + csv written under {out_dir}")


def main() -> None:
    construction, system = build_system()

    print("uniform_shafts/point_load/case_radial_single_position")
    print(f"  shaft1: RadialLoad {LOAD_N:.0f} N @ x={LOAD_X_SHAFT1_MM:.1f} mm")
    print(f"  shaft2: RadialLoad {LOAD_N:.0f} N @ x={LOAD_X_SHAFT2_MM:.1f} mm")

    # ------------------------------------------------------------------
    # Euler-Bernoulli branch -- one solve per shaft, no shear sweep.
    # ------------------------------------------------------------------
    # NOTE: NOT results_dir(__file__, "euler_bernoulli") -- that helper
    # joins its extra segments AFTER "results" (case_dir()/results/<extra>),
    # by design, matching how compare_case_x.py already uses it
    # (comparison_dir(__file__, shear_theory) -> comparison/<shear_theory>).
    # On disk here, <theory>/ sits ABOVE results/ (a sibling of
    # abaqus_results/ and comparison/ under euler_bernoulli/ or
    # timoshenko/), the opposite nesting -- so the theory folder is
    # built directly off case_dir() instead.
    eb_dir = case_dir(__file__) / "euler_bernoulli" / "results"
    eb_dir.mkdir(parents=True, exist_ok=True)
    write_construction_report(
        system, eb_dir / "construction_report.txt",
        title="uniform_shafts/point_load/case_radial_single_position -- Construction (Euler-Bernoulli)",
    )

    eb_study = StudyCapabilities(construction=construction, shaft_fem=("shaft_fem.euler_bernoulli_rigid",))
    eb_solve_system = eb_study.resolve()["solve_system"]
    # solve_system() has ONE uniform signature across both theories --
    # shear_theory/integration_method are required keyword-only args
    # even on the Euler-Bernoulli branch (confirmed by running this:
    # TypeError without them). CORRECTED (this was wrong on the first
    # attempt): they are NOT silently ignored/inert for euler_bernoulli
    # the way kGA_override is -- BeamModelSettings.__post_init__()
    # actively VALIDATES and REJECTS a non-None shear_theory when
    # beam_theory="euler_bernoulli" ("does not use shear_theory ...
    # pass None"), confirmed by running this with shear_theory="cowper"
    # here. Both are therefore explicitly None for this branch -- not
    # an arbitrary placeholder value, the ONLY value BeamModelSettings
    # accepts here.
    eb_library = eb_solve_system(
        system, construction,
        shear_theory=None,
        integration_method=None,
    )

    for ss in system.shafts:
        r = eb_library.get_or_none(ss.name)
        if r is None:
            continue
        print(f"    [euler_bernoulli] {ss.name:8s} "
              f"v_max={r.v_max:7.4f} mm @ x={r.x_v_max:6.1f} mm  "
              f"sigma_b_max={r.sigma_b_max:8.2f} MPa @ x={r.x_sigma_b_max:6.1f} mm")

    _write_theory_outputs(
        system, eb_dir, eb_library,
        title="uniform_shafts/point_load/case_radial_single_position -- radial load, position sweep (euler_bernoulli)",
    )

    # ------------------------------------------------------------------
    # Timoshenko branch -- swept over shear_theory, as before.
    # ------------------------------------------------------------------
    ts_root = case_dir(__file__) / "timoshenko" / "results"
    ts_root.mkdir(parents=True, exist_ok=True)
    write_construction_report(
        system, ts_root / "construction_report.txt",
        title="uniform_shafts/point_load/case_radial_single_position -- Construction (timoshenko)",
    )

    ts_study = StudyCapabilities(construction=construction, shaft_fem=("shaft_fem.timoshenko_rigid",))
    ts_solve_system = ts_study.resolve()["solve_system"]

    for shear_theory in SHEAR_THEORIES:
        ts_library = ts_solve_system(
            system, construction,
            shear_theory=shear_theory,
            integration_method=TIMOSHENKO_INTEGRATION_METHOD,
        )

        for ss in system.shafts:
            r = ts_library.get_or_none(ss.name)
            if r is None:
                continue
            print(f"    [timoshenko/{shear_theory}] {ss.name:8s} "
                  f"v_max={r.v_max:7.4f} mm @ x={r.x_v_max:6.1f} mm  "
                  f"sigma_b_max={r.sigma_b_max:8.2f} MPa @ x={r.x_sigma_b_max:6.1f} mm")

        _write_theory_outputs(
            system, ts_root / shear_theory, ts_library,
            title=f"uniform_shafts/point_load/case_radial_single_position -- radial load, position sweep (timoshenko/{shear_theory})",
        )


if __name__ == "__main__":
    main()
