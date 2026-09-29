"""
validation/fem_studies/resolution/uniform_shafts/point_load/case_radial_single_position/case_radial_single_position.py

Case group: uniform shaft, ONE extra point RadialLoad, position varied,
solved under BOTH beam theories (Euler-Bernoulli and Timoshenko). See
the suite's top-level README.md for the euler_bernoulli/ vs
timoshenko/ vs analytical/ split, and for the "vehicle, not mechanism"
discipline shared by every case script.

REWRITE (this pass) -- no more fixtures / capabilities layer:
  - The system is now built DIRECTLY from the AxisForge core
    (Shaft/ShaftSection, Bearing.assemble via common/bearings.py,
    SpurHelicalGear + SpurHelicalGearMeshing, ShaftSystem/GearElement,
    SpurHelicalMeshLink, SpurHelicalGearSystem.resolve()). The old
    ConstructionCapabilities / StudyCapabilities / build_linear_system /
    solve_system calls are gone -- the modules behind them no longer
    exist.
  - The FEM solve is RigidSupportFEMSolver(BeamModelSettings).solve(ss),
    which now returns a finished ShaftResults directly (no
    ShaftResultsReader, no results library).
  - Reports come from the bridge: construction_report.txt from
    outputs.construction.text_report.write_construction_report, and
    report_resolution.txt from outputs.solver_results.fem_report.
    write_fem_report (same layout as the previous report_resolution.txt).
  - Plots come from outputs.solver_results.plots.fem_plots.write_fem_plots
    (same 9 PNGs per shaft, same folder layout as before). CSV: a
    TEMPORARY local writer below reproduces the columns
    compare_case_*.py reads; replace it once the bridge grows a real
    outputs module for it.

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
exactly what analytical/'s closed-form beam formulas assume.

integration_method="single_point" (selective/reduced integration) for
the Timoshenko branch -- this is a resolution study, not the
locking-demonstration integration_method="exact" used by the mesh
convergence group. Euler-Bernoulli has no shear term, so
BeamModelSettings requires shear_theory=None and integration_method=None
for that branch (it raises otherwise).

Outputs, per theory (relative to this case's own folder -- see
common/paths.py):
  euler_bernoulli/results/              construction_report.txt,
                                        report_resolution.txt, csv/*.csv
  timoshenko/results/<shear_theory>/    report_resolution.txt, csv/*.csv
  timoshenko/results/                   construction_report.txt
"""
from __future__ import annotations

import sys
from pathlib import Path

# --- path bootstrap ---------------------------------------------------------
# 1) suite root: nearest ancestor containing common/  -> `import common.*`
# 2) package root: the folder that CONTAINS the axisforge_bridge/ directory
#    -> `import axisforge_bridge.*` (no install needed for the bridge).
# `axisforge` itself (the base package) must be importable on its own:
# pip install -e <AxisForge repo>, or PYTHONPATH pointing at its root.
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
_bridge_parent = None
for _ancestor in _p.parents:
    if _ancestor.name == "axisforge_bridge":
        _bridge_parent = _ancestor.parent
        break
if _bridge_parent is None:
    raise RuntimeError(
        f"{__file__}: no ancestor directory named 'axisforge_bridge' found."
    )
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_bridge_parent))

import numpy as np  # noqa: E402

from common.bearings import build_case_bearings  # noqa: E402
from common.materials import ensure_case_materials  # noqa: E402
from common.paths import case_dir  # noqa: E402

from axisforge.core.loads import RadialLoad, TorqueLoad  # noqa: E402
from axisforge.core.machine_elements.shaft.shaft import Shaft, ShaftSection  # noqa: E402
from axisforge.core.machine_elements.gears.parallel_axis.gear_properties.spur_helical_gear import (  # noqa: E402
    SpurHelicalGear,
)
from axisforge.core.machine_elements.gears.parallel_axis.gear_meshing.spurhelical_meshing import (  # noqa: E402
    SpurHelicalGearMeshing,
)
from axisforge.core.mechanical_system.parallel_axis.spur_helical.shaft_system import (  # noqa: E402
    GearElement,
    ShaftSystem,
)
from axisforge.core.mechanical_system.parallel_axis.spur_helical.gear_system import (  # noqa: E402
    SpurHelicalGearSystem,
    SpurHelicalMeshLink,
)
from axisforge.mesh.shaft.beam_model_settings import BeamModelSettings  # noqa: E402
from axisforge.solvers.machine_elements.shaft.fem_solvers.global_solver.rigid_support import (  # noqa: E402
    RigidSupportFEMSolver,
)
from axisforge.results.fem_results.shaft_results import ShaftResults  # noqa: E402

from axisforge_bridge.outputs.construction.text_report import write_construction_report  # noqa: E402
from axisforge_bridge.outputs.solver_results.fem_report import write_fem_report  # noqa: E402
from axisforge_bridge.outputs.solver_results.plots.fem_plots import write_fem_plots  # noqa: E402


SHAFT_LENGTH_MM = 200.0
SHAFT_DIAMETER_MM = 20.0

LOAD_N = 500.0
LOAD_THETA_DEG = 270.0
LOAD_X_SHAFT1_MM = 60.0
LOAD_X_SHAFT2_MM = 140.0

# Source-shaft power / speed. P = T * omega with T = 100 N*m at 1000 rpm.
POWER_W = 10471.9755
SHAFT1_RPM = 1000.0
SHAFT2_RPM = 500.0            # z1/z2 = 20/40 -> informational, see ShaftSystem.speed_rpm
ROTATION_DIR_SOURCE = 1

# Timoshenko-branch integration method -- "exact" is the other option,
# but this is a resolution study, not the locking-demonstration group.
TIMOSHENKO_INTEGRATION_METHOD = "single_point"

# Shear theories swept for the Timoshenko branch only.
SHEAR_THEORIES = ("cowper", "hutchinson")


def build_system() -> SpurHelicalGearSystem:
    """
    Builds and RESOLVES the SpurHelicalGearSystem for this case --
    geometry, bearings, gear mesh, and every load (user + gear_mesh-
    sourced), independent of which beam theory will later solve it.

    Exposed (not just called from main()) specifically so that
    analytical/analytical_case_radial_single_position.py can import and
    call this SAME function to get the identical, already-resolved
    system -- same geometry, same bearing positions, same gear-mesh
    Ft/Fr -- for its own closed-form (non-FEM) solve, rather than
    re-deriving any of those numbers by hand.

    NOTE for that caller: this used to return (construction, system);
    the capabilities layer is gone, so it now returns `system` only.
    """
    ensure_case_materials()   # core material registry is empty until registered

    def make_shaft_geometry(name: str) -> Shaft:
        shaft = Shaft(label=name)
        shaft.add_section(ShaftSection(
            length=SHAFT_LENGTH_MM, diameter=SHAFT_DIAMETER_MM, label=name,
        ))
        return shaft

    # --- shaft containers ------------------------------------------------
    ss1 = ShaftSystem(make_shaft_geometry("shaft1"), name="shaft1", speed_rpm=SHAFT1_RPM)
    ss2 = ShaftSystem(make_shaft_geometry("shaft2"), name="shaft2", speed_rpm=SHAFT2_RPM)

    for ss in (ss1, ss2):
        for brg in build_case_bearings(ss.name):
            ss.add_bearing(brg)

    # --- gears + mesh ----------------------------------------------------
    g_driver = SpurHelicalGear(mn=2.0, z=20, b=15.0, position=100.0, label="g_driver")
    g_driven = SpurHelicalGear(mn=2.0, z=40, b=15.0, position=100.0, label="g_driven")
    meshing = SpurHelicalGearMeshing(g_driver, g_driven, label="stage1")

    ge_driver = GearElement(g_driver, role="driver", label="g_driver")
    ge_driven = GearElement(g_driven, role="driven", label="g_driven")
    ss1.add_gear(ge_driver)
    ss2.add_gear(ge_driven)

    # --- the one thing that varies between shafts --------------------------
    ss1.add_load(RadialLoad(LOAD_X_SHAFT1_MM, LOAD_N, theta_deg=LOAD_THETA_DEG, label="radial_test_load"))
    ss2.add_load(RadialLoad(LOAD_X_SHAFT2_MM, LOAD_N, theta_deg=LOAD_THETA_DEG, label="radial_test_load"))

    link = SpurHelicalMeshLink(
        ss1, ge_driver, ss2, ge_driven, meshing,
        phi_deg=0.0, label="stage1",
    )
    system = SpurHelicalGearSystem(
        [ss1, ss2], [link],
        label="uniform_point_load_radial_single_position",
    )
    system.validate_or_raise()
    system.resolve(P=POWER_W, rpm=SHAFT1_RPM, rotation_dir_source=ROTATION_DIR_SOURCE)

    # shaft2 (driven shaft) only receives the gear-mesh reaction torque
    # (source="gear_mesh", at x=100mm) -- nothing balances that torque
    # on this shaft. TorsionSolver.validate_equilibrium() would flag
    # this, and an Abaqus model built from the construction report ends
    # up with DOF 4 (torsion) unconstrained -- zero pivot in the solver.
    #
    # Driven-machine torque sink, placed on the far side of the gear
    # (x=200mm > 100mm, so T(x) closes back to zero past the
    # application point). Magnitude is derived from the net already
    # present rather than hand-copied.
    shaft2_sys = system.shafts[-1]
    net_before = sum(ld.magnitude for ld in shaft2_sys.torque_loads)
    shaft2_sys.add_load(TorqueLoad(
        position=SHAFT_LENGTH_MM,
        magnitude=-net_before,
        label="output_coupling",
        source="user",
    ))

    return system


def solve_system(system: SpurHelicalGearSystem, settings: BeamModelSettings) -> dict[str, ShaftResults]:
    """
    Solve every ShaftSystem in `system` with the same BeamModelSettings.
    One fresh RigidSupportFEMSolver per shaft (cheap, and keeps the old
    "no solver instance reused across shafts" rule trivially true).
    Returns {ShaftSystem.name: ShaftResults}.
    """
    results: dict[str, ShaftResults] = {}
    for ss in system.shafts:
        results[ss.name] = RigidSupportFEMSolver(settings).solve(ss)
    return results


# --- TEMPORARY csv writer --------------------------------------------------
# Reproduces the columns compare_case_*.py reads (x_mm, u_mm, v_xy_mm,
# v_xz_mm, theta_xy_rad, theta_xz_rad -- see common/compare.py's
# DEFAULT_/EXTENDED_COLUMN_MAP). The remaining columns (M, V, T, stresses)
# are named by analogy and NOT verified against the old resolution_csv or
# against compare_analytical_case_*.py -- check those two before relying
# on them. Replace with a proper bridge outputs module.
_CSV_COLUMNS = (
    ("x_mm",         "x"),
    ("u_mm",         "u"),
    ("v_xz_mm",      "v_xz"),
    ("v_xy_mm",      "v_xy"),
    ("v_mm",         "v"),
    ("theta_xz_rad", "theta_xz"),
    ("theta_xy_rad", "theta_xy"),
    ("M_xz_Nmm",     "M_xz"),
    ("M_xy_Nmm",     "M_xy"),
    ("M_Nmm",        "M"),
    ("V_xz_N",       "V_xz"),
    ("V_xy_N",       "V_xy"),
    ("V_N",          "V"),
    ("T_Nm",         "T"),
    ("phi_rad",      "phi"),
    ("d_mm",         "d"),
    ("W_mm3",        "W"),
    ("Wt_mm3",       "Wt"),
    ("sigma_b_MPa",  "sigma_b"),
    ("tau_MPa",      "tau"),
)


def _write_resolution_csv(result: ShaftResults, path: Path, decimals: int = 10) -> None:
    cols = [np.asarray(getattr(result, attr), dtype=float) for _, attr in _CSV_COLUMNS]
    n = len(cols[0])
    lines = [",".join(name for name, _ in _CSV_COLUMNS)]
    for i in range(n):
        lines.append(",".join(f"{c[i]:.{decimals}g}" for c in cols))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_theory_outputs(system, results: dict[str, ShaftResults], out_dir: Path, title: str) -> None:
    """Report + plots + csv for one already-solved theory branch, shared by both
    branches in main()."""
    out_dir.mkdir(parents=True, exist_ok=True)
    write_fem_report(system, results, out_dir / "report_resolution.txt", title=title)
    write_fem_plots(system, results, out_dir)          # -> out_dir/plots/<shaft>/*.png

    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    for ss in system.shafts:
        r = results.get(ss.name)
        if r is None:
            continue
        _write_resolution_csv(r, csv_dir / f"{ss.name}_resolution.csv")
    print(f"[OK] report + plots + csv written under {out_dir}")


def _print_envelope(tag: str, system, results: dict[str, ShaftResults]) -> None:
    for ss in system.shafts:
        r = results.get(ss.name)
        if r is None:
            continue
        print(f"    [{tag}] {ss.name:8s} "
              f"v_max={r.v_max:7.4f} mm @ x={r.x_v_max:6.1f} mm  "
              f"sigma_b_max={r.sigma_b_max:8.2f} MPa @ x={r.x_sigma_b_max:6.1f} mm")


def main() -> None:
    system = build_system()

    print("uniform_shafts/point_load/case_radial_single_position")
    print(f"  shaft1: RadialLoad {LOAD_N:.0f} N @ x={LOAD_X_SHAFT1_MM:.1f} mm")
    print(f"  shaft2: RadialLoad {LOAD_N:.0f} N @ x={LOAD_X_SHAFT2_MM:.1f} mm")

    # ------------------------------------------------------------------
    # Euler-Bernoulli branch -- one solve per shaft, no shear sweep.
    # <theory>/ sits ABOVE results/ on disk, so the path is built off
    # case_dir() directly, NOT through results_dir() (see common/paths.py).
    # BeamModelSettings itself rejects a non-None shear_theory /
    # integration_method for euler_bernoulli.
    # ------------------------------------------------------------------
    eb_dir = case_dir(__file__) / "euler_bernoulli" / "results"
    eb_dir.mkdir(parents=True, exist_ok=True)
    write_construction_report(
        system, eb_dir / "construction_report.txt",
        title="uniform_shafts/point_load/case_radial_single_position -- Construction (Euler-Bernoulli)",
    )

    eb_results = solve_system(
        system, BeamModelSettings(beam_theory="euler_bernoulli",
                                  shear_theory=None, integration_method=None),
    )
    _print_envelope("euler_bernoulli", system, eb_results)
    _write_theory_outputs(
        system, eb_results, eb_dir,
        title="uniform_shafts/point_load/case_radial_single_position -- radial load, position sweep (euler_bernoulli)",
    )

    # ------------------------------------------------------------------
    # Timoshenko branch -- swept over shear_theory.
    # ------------------------------------------------------------------
    ts_root = case_dir(__file__) / "timoshenko" / "results"
    ts_root.mkdir(parents=True, exist_ok=True)
    write_construction_report(
        system, ts_root / "construction_report.txt",
        title="uniform_shafts/point_load/case_radial_single_position -- Construction (timoshenko)",
    )

    for shear_theory in SHEAR_THEORIES:
        ts_results = solve_system(
            system, BeamModelSettings(beam_theory="timoshenko",
                                      shear_theory=shear_theory,
                                      integration_method=TIMOSHENKO_INTEGRATION_METHOD),
        )
        _print_envelope(f"timoshenko/{shear_theory}", system, ts_results)
        _write_theory_outputs(
            system, ts_results, ts_root / shear_theory,
            title=f"uniform_shafts/point_load/case_radial_single_position -- radial load, position sweep (timoshenko/{shear_theory})",
        )


if __name__ == "__main__":
    main()