"""
validation/abaqus_comparison/uniform_shafts/point_load/timoshenko/case_axial_and_moment/case_axial_and_moment.py

Case group: uniform shaft, Timoshenko beam theory, AxialLoad and
ExternalMoment (both always point loads -- core.loads.AxialLoad /
ExternalMoment have no distributed variant, unlike RadialLoad). Same
1-stage spur-gear vehicle and fixed BC as case_radial_single_position.py
-- see that case's own header and the suite's top-level README.md for
the shared discipline.

Two shafts, two different load TYPES/combinations (not just positions,
since that's already covered by case_radial_single_position.py):

    shaft1  AxialLoad only        : Fa = +2000 N (tension) @ x=190.0 mm
                                     (applied at the non-locating/roller
                                     end on purpose -- with only ONE
                                     locating bearing reacting axial load
                                     regardless of where Fa is applied,
                                     this isolates whether AxialLoad's
                                     node position matters to u(x) at all
                                     for a straight uniform shaft, a
                                     useful sanity check against Abaqus).
    shaft2  AxialLoad + ExternalMoment combined @ x=100.0 mm :
                                     Fa = +2000 N, M = 15000 N*mm,
                                     theta=90 deg -- a genuine combination
                                     case (two load types superposed at
                                     the same station), not just two
                                     loads at different positions.

BC: fixed for this whole suite -- ball (locating) @10mm, roller
(non-locating) @190mm -- see scripts/common/bearings.py.

integration_method="exact" fixed for this whole case group, same
choice as case_radial_single_position.py -- full Gauss integration, no
reduced/selective integration, so shear-locking mitigation is never a
variable when diffing against Abaqus. Applies uniformly to every pass
below, including the abaqus_kGA one (integration method and shear
stiffness override are independent knobs -- see ABAQUS_KGA_N's own
note below).

Third pass -- ABAQUS_KGA_N (kGA_override):
    Besides the two "real" shear-correction-theory passes (cowper,
    hutchinson), this script also runs a THIRD pass with
    kGA_override=ABAQUS_KGA_N -- Abaqus's own reported transverse
    shear stiffness (from its *Preprint, model=YES section-properties
    printout: K*G(23)*A = K*G(13)*A = 2.22606E+07, same value for both
    the XZ and XY planes on this uniform circular section). This is
    NOT a third shear_theory -- it bypasses shear_theory entirely, see
    TimoshenkoBeam._shear_correction_factor()'s own docstring -- it is
    a way of testing whether AxisForge's Timoshenko element reproduces
    Abaqus's deflection/rotation field when fed the *exact same*
    transverse shear stiffness Abaqus itself used internally
    (including whatever slenderness-compensation factor Abaqus bakes
    into that number), independent of which shear-correction-factor
    theory (cowper/hutchinson) AxisForge would otherwise pick. shaft.
    shear_theory="cowper" is still passed to solve_system() for this
    pass -- it has no numerical effect once kGA_override is given (see
    _shear_correction_factor()), it only has to match the value the
    elements were themselves built with, which is "cowper" either way
    (BeamModelSettings still requires a valid, non-None shear_theory
    string whenever beam_theory="timoshenko", override or not).

    Written to its own RESULTS_DIR / "abaqus_kGA" folder, parallel to
    cowper/ and hutchinson/, so it never overwrites or gets confused
    with either real shear-theory pass.

    Console cross-check -- by THEORY/SHEAR-FACTOR, not by displacement:
    main() prints, for every (pass, shaft) pair, the actual
    elem.shear_theory string AND the effective shear_factor/K*G*A that
    TimoshenkoBeam._shear_correction_factor() computed for it. This is
    a DIRECT read of what each pass actually used, not an inference
    from how much v_max moved -- v_max could differ for reasons
    unrelated to shear (or fail to differ even with a real change,
    depending on load case), so it is not proof either way on its own.
    This diagnostic solves each shaft a SECOND time with a bare
    RigidSupportFEMSolver, built directly in this script (bypassing
    the study.resolve()["solve_system"] indirection used for the real
    report/csv passes below), purely so elements[0] and its computed
    shear_factor/kGA are inspectable here -- that indirection's own
    forwarding of kGA_override into fem_simple.solve_system() has
    never actually been confirmed, see this suite's own validation
    conversation, and this diagnostic does not depend on it being
    correct: it always shows the true shear_theory/shear_factor for
    the exact (shear_theory, kGA_override) pair THIS script passed in
    for that pass, independent of whatever the indirection does with
    the same pair.
"""
from __future__ import annotations

import sys
from pathlib import Path

_p = Path(__file__).resolve()
_root = None
for _ancestor in _p.parents:
    if (_ancestor / "scripts" / "common").is_dir():
        _root = _ancestor
        break
if _root is None:
    raise RuntimeError(
        f"{__file__}: no ancestor directory containing 'scripts/common/' "
        f"found -- this file must live somewhere inside the suite tree."
    )
sys.path.insert(0, str(_root / "scripts"))
from common.bearings import build_case_bearings  # noqa: E402
from common.paths import results_dir  # noqa: E402

from axisforge.core.loads import AxialLoad, ExternalMoment, TorqueLoad  # noqa: E402
from axisforge.fixtures.construction.construction_capabilities import ConstructionCapabilities  # noqa: E402
from axisforge.fixtures.construction.outputs.text_report import write_construction_report  # noqa: E402
from axisforge.fixtures.studies.study_capabilities import StudyCapabilities  # noqa: E402
from axisforge.fixtures.studies.outputs.text_report import write_studies_report  # noqa: E402
from axisforge.fixtures.studies.shafts.fem_studies.outputs.plots import write_resolution_plots  # noqa: E402
from axisforge.fixtures.studies.shafts.fem_studies.outputs.resolution_csv import resolution_csv  # noqa: E402

# Only used by the console theory/shear-factor diagnostic below --
# NOT part of the real solve_system()-driven report/csv passes, see
# _print_shear_diagnostics()'s own docstring.
from axisforge.mesh.shaft.beam_model_settings import BeamModelSettings  # noqa: E402
from axisforge.solvers.machine_elements.shaft.fem_solvers.rigid_support import (  # noqa: E402
    RigidSupportFEMSolver,
)
from axisforge.mesh.shaft.element_type.timoshenko.two_noded import TimoshenkoBeam  # noqa: E402
from axisforge.mesh.shaft.element_type.timoshenko.shear_factor import ShearFactor  # noqa: E402


RESULTS_DIR = results_dir(__file__)

SHAFT_LENGTH_MM = 200.0
SHAFT_DIAMETER_MM = 20.0

AXIAL_N = 2000.0
AXIAL_X_SHAFT1_MM = 190.0
COMBINED_X_SHAFT2_MM = 100.0
MOMENT_NMM = 15_000.0
MOMENT_THETA_DEG = 90.0

# Same choice as case_radial_single_position.py -- see this module's
# own top docstring.
INTEGRATION_METHOD = "single_point"

# Abaqus *Preprint, model=YES section-properties printout for this
# uniform circular section -- K*G(23)*A = K*G(13)*A, same value for
# both planes. Fed into TimoshenkoBeam via kGA_override for the third
# ("abaqus_kGA") pass below -- see this module's own top docstring.
ABAQUS_KGA_N = 2.22606e7


def build_system() -> tuple["ConstructionCapabilities", "SpurHelicalGearSystem"]:
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

    axial_only = AxialLoad(AXIAL_X_SHAFT1_MM, AXIAL_N, label="axial_test_load")
    axial_combined = AxialLoad(COMBINED_X_SHAFT2_MM, AXIAL_N, label="axial_test_load")
    moment_combined = ExternalMoment(
        COMBINED_X_SHAFT2_MM, MOMENT_NMM, theta_deg=MOMENT_THETA_DEG, label="moment_test_load",
    )

    shaft_specs = [
        ShaftSpec(shaft=shaft1, bearings=bearings1, speed_rpm=1450.0,
                  loads=(axial_only,), name="shaft1"),
        ShaftSpec(shaft=shaft2, bearings=bearings2, speed_rpm=725.0,
                  loads=(axial_combined, moment_combined), name="shaft2"),
    ]
    stage_specs = [
        StageSpec(gear_driver=g_driver, gear_driven=g_driven, phi_deg=0.0, label="stage1"),
    ]

    system = build_linear_system(
        shaft_specs, stage_specs, P=10000.0, rotation_dir_source=1,
        label="uniform_point_load_axial_and_moment",
    )

    shaft2_sys = system.shafts[-1]
    net_before = sum(ld.magnitude for ld in shaft2_sys.torque_loads)
    shaft2_sys.add_load(TorqueLoad(
        position=SHAFT_LENGTH_MM,
        magnitude=-net_before,
        label="output_coupling",
        source="user",
    ))
    return construction, system


def _print_shear_diagnostics(
    label: str,
    ss,
    *,
    shear_theory: str,
    kGA_override: float | None,
) -> None:
    """
    Direct, by-theory/by-shear-factor console check -- NOT inferred
    from displacement outputs (see this module's own top docstring).

    Solves `ss` a second time with a bare RigidSupportFEMSolver built
    directly here (bypassing study.resolve()["solve_system"]), then
    reads elements[0] (every element in this uniform-section case
    shares the same E/A/v, so one is representative) and prints:

      - elem.shear_theory   -- the theory string the element was
                                built with (always "cowper" here for
                                BOTH the "cowper" and "abaqus_kGA"
                                passes, since abaqus_kGA still passes
                                shear_theory="cowper" through -- see
                                this module's own top docstring for
                                why that's expected and not a bug)
      - shear_factor        -- TimoshenkoBeam.shear_factor()'s actual
                                return value for this pass (itself a
                                thin wrapper over
                                ShearFactor.shear_correction_factor(),
                                which does the real cowper/hutchinson/
                                override computation from elem.v/
                                elem.radius_ratio/elem.E/elem.A). This
                                is the number that actually varies:
                                cowper and hutchinson each compute it
                                from elem.v/radius_ratio, while
                                kGA_override REPLACES it with
                                kGA_override/(G*A) regardless of theory.
      - effective K*G*A      -- shear_factor_value * G * A, recomputed
                                here independently of solve_system() --
                                for the abaqus_kGA pass this MUST equal
                                kGA_override (2.22606e7 N) to 6+ sig
                                figs, since that is exactly how
                                ShearFactor.shear_correction_factor()
                                derives it in that branch.

    This never touches study.resolve()["solve_system"], so it is
    unaffected by whether that indirection actually forwards
    kGA_override -- it always reflects the true value for the
    (shear_theory, kGA_override) pair passed in right here.
    """
    settings = BeamModelSettings(
        beam_theory="timoshenko",
        shear_theory=shear_theory,
        integration_method=INTEGRATION_METHOD,
    )
    solver = RigidSupportFEMSolver(settings, kGA_override=kGA_override)
    solver.solve(ss)
    elem = solver.elements[0]

    beam = TimoshenkoBeam()
    shear_factor_value = beam.shear_factor(
        elem, ShearFactor(), elem.shear_theory, kGA_override=kGA_override,
    )
    G = elem.E / (2 * (1 + elem.v))
    kGA_effective = shear_factor_value * G * elem.A

    print(f"    [{label}] {ss.name:8s} elem.shear_theory={elem.shear_theory!r:12s}  "
          f"kGA_override={kGA_override!r:14}  shear_factor={shear_factor_value:.6f}  "
          f"effective K*G*A={kGA_effective:.6e} N")


def _run_pass(
    label: str,
    solve_system,
    system,
    construction,
    *,
    shear_theory: str,
    kGA_override: float | None = None,
) -> dict[str, tuple[float, float, float]]:
    """
    One resolved pass (report + plots + csv), shared by the two "real"
    shear_theory passes and the ABAQUS_KGA_N override pass below --
    `label` is just the output subfolder name (e.g. "cowper",
    "hutchinson", "abaqus_kGA"), it does not have to equal
    shear_theory (the override pass uses label="abaqus_kGA" but still
    passes shear_theory="cowper" through, see this module's own top
    docstring for why that's inert once kGA_override is given).

    integration_method is the module-level INTEGRATION_METHOD constant
    for every pass -- not something that varies here, same as
    case_radial_single_position.py.

    Returns {shaft_name: (v_max, x_v_max, sigma_b_max)} -- one entry
    per shaft -- so main() can cross-check the three passes actually
    produced DIFFERENT numbers (see main()'s own comment on why this
    is necessary: if solve_system silently drops kGA_override, e.g.
    because the StudyCapabilities/study.resolve() indirection doesn't
    forward it, this pass would run with plain shear_theory="cowper"
    and its numbers would come out identical to the real "cowper"
    pass, with NO exception raised anywhere).
    """
    library = solve_system(
        system, construction,
        shear_theory=shear_theory,
        integration_method=INTEGRATION_METHOD,
        kGA_override=kGA_override,
    )

    summary: dict[str, tuple[float, float, float]] = {}
    for ss in system.shafts:
        r = library.get_or_none(ss.name)
        if r is None:
            continue
        summary[ss.name] = (r.v_max, r.x_v_max, r.sigma_b_max)
        print(f"    [{label}] {ss.name:8s} "
              f"v_max={r.v_max:7.4f} mm @ x={r.x_v_max:6.1f} mm  "
              f"sigma_b_max={r.sigma_b_max:8.2f} MPa @ x={r.x_sigma_b_max:6.1f} mm")

    out_dir = RESULTS_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)
    write_studies_report(
        system, out_dir / f"report_resolution_{label}.txt",
        title=f"uniform_shafts/point_load/timoshenko -- axial + moment ({label})",
        shaft_fem_library=library,
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
    print(f"[OK] [{label}] report + plots + csv written under {out_dir}")
    return summary


def main() -> None:
    construction, system = build_system()

    # Construction report FIRST, once per case -- see
    # case_radial_single_position.py's own main() for why this is
    # unconditional and written outside the shear_theory loop.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_construction_report(
        system, RESULTS_DIR / "construction_report.txt",
        title="uniform_shafts/point_load/timoshenko/case_axial_and_moment -- Construction",
    )
    print(f"[OK] construction_report.txt written under {RESULTS_DIR} "
          f"-- use this to build the equivalent Abaqus model")

    study = StudyCapabilities(construction=construction, shaft_fem=("shaft_fem.timoshenko_rigid",))
    solve_system = study.resolve()["solve_system"]

    print("uniform_shafts/point_load/timoshenko/case_axial_and_moment")
    print(f"  shaft1: AxialLoad {AXIAL_N:.0f} N @ x={AXIAL_X_SHAFT1_MM:.1f} mm (alone)")
    print(f"  shaft2: AxialLoad {AXIAL_N:.0f} N + ExternalMoment {MOMENT_NMM:.0f} N*mm "
          f"@ x={COMBINED_X_SHAFT2_MM:.1f} mm (combined)")

    # Console cross-check FIRST, by theory/shear-factor value -- not by
    # displacement output. See _print_shear_diagnostics()'s own
    # docstring: this solves each shaft independently (bypassing
    # study.resolve()["solve_system"]) purely to print the true
    # elem.shear_theory / shear_factor / effective K*G*A for each pass,
    # so you can see directly whether the three passes are actually
    # using different shear stiffness -- not infer it from how much
    # v_max moved.
    print()
    print("shear diagnostics (elem.shear_theory / shear_factor / effective K*G*A per pass):")
    PASSES = (
        ("cowper",     "cowper",     None),
        ("hutchinson", "hutchinson", None),
        ("abaqus_kGA", "cowper",     ABAQUS_KGA_N),
    )
    for ss in system.shafts:
        for label, shear_theory_arg, kga in PASSES:
            _print_shear_diagnostics(label, ss, shear_theory=shear_theory_arg, kGA_override=kga)
        print()

    for shear_theory in ("cowper", "hutchinson"):
        _run_pass(
            shear_theory, solve_system, system, construction, shear_theory=shear_theory,
        )

    # Third pass: Abaqus's own reported K*G*A, bypassing shear_theory
    # entirely -- see this module's own top docstring.
    print(f"    (abaqus_kGA pass -- kGA_override={ABAQUS_KGA_N:.5e} N, "
          f"from Abaqus's *Preprint section-properties printout)")
    _run_pass(
        "abaqus_kGA", solve_system, system, construction,
        shear_theory="cowper", kGA_override=ABAQUS_KGA_N,
    )


if __name__ == "__main__":
    main()