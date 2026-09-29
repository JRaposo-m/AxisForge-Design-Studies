from pathlib import Path

from axisforge.core.materials import register, get_material
from axisforge.core.materials.base import Material, IsotropicElastic, StrengthProperties

from axisforge.core.machine_elements.shaft.shaft import Shaft, ShaftSection
from axisforge.core.machine_elements.bearings.base import BearingCatalog
from axisforge.core.machine_elements.bearings.bearing import Bearing
from axisforge.core.machine_elements.bearings.families.family import (
    DeepGrooveBallFamily, AngularContactFamily, CylindricalRollerFamily,
)
from axisforge.core.machine_elements.bearings.contact_models.analysis import ContactAnalysis
from axisforge.core.machine_elements.gears.parallel_axis.gear_properties.spur_helical_gear import SpurHelicalGear
from axisforge.core.machine_elements.gears.parallel_axis.gear_meshing.spurhelical_meshing import SpurHelicalGearMeshing
from axisforge.core.mechanical_system.parallel_axis.spur_helical.shaft_system import ShaftSystem, GearElement
from axisforge.core.mechanical_system.parallel_axis.spur_helical.gear_system import SpurHelicalMeshLink, SpurHelicalGearSystem

from axisforge.mesh.shaft.beam_model_settings import BeamModelSettings
from axisforge.solvers.machine_elements.shaft.fem_solvers.global_solver.rigid_support import RigidSupportFEMSolver
from axisforge.solvers.machine_elements.bearings.load_distribution.iso_16281.contact_solver import (
    ISO16281BallSolver, ISO16281RollerSolver,
)

from axisforge_bridge.outputs.construction.text_report import write_construction_report

# --- material: S355 (EN 10025-2), registado manualmente para os veios ---
try:
    get_material("S355")
except KeyError:
    register(Material(
        material_id="S355",
        density=7850.0,
        elastic=IsotropicElastic(E=206000.0, poisson_ratio=0.3),
        strength=StrengthProperties(Sut=470.0, Sy=355.0),
        description="S355 structural steel (EN 10025-2), typical values",
    ))

# --- aço de rolamento (100Cr6/AISI 52100), elemento rolante == pista ---
E_BEARING_STEEL, NU_BEARING_STEEL = 208000.0, 0.3

# --- shaft 1: pinion, deep groove ball (locating) + cylindrical roller (floating) ---
shaft1 = Shaft(label="shaft_1")
shaft1.add_section(ShaftSection(length=200.0, diameter=40.0, label="s1"))
sys1 = ShaftSystem(shaft1, name="shaft_1_system", speed_rpm=1500.0, label="input")

cat_dgb = BearingCatalog(d=20.0, D=47.0, b=14.0, designation="6204", label="B1_dgb", position=20.0, arrangement="locating")
brg_dgb = Bearing.assemble(DeepGrooveBallFamily(), cat_dgb, dict(
    Dw=7.94, Dpw=33.5, Z=8, s=0.0,
    contact=ContactAnalysis.ISO16281,
    e1=E_BEARING_STEEL, e2=E_BEARING_STEEL, nu1=NU_BEARING_STEEL, nu2=NU_BEARING_STEEL,
))
sys1.add_bearing(brg_dgb)

cat_roll1 = BearingCatalog(d=40.0, D=80.0, b=18.0, designation="NU208", label="B2_roll", position=180.0, arrangement="floating")
brg_roll1 = Bearing.assemble(CylindricalRollerFamily(), cat_roll1, dict(
    Dwe=12.0, Lwe=12.0, Dpw=60.0, Z=13, s=0.02, n_s=30,
    contact=ContactAnalysis.ISO16281,
    e1=E_BEARING_STEEL, e2=E_BEARING_STEEL, nu1=NU_BEARING_STEEL, nu2=NU_BEARING_STEEL,
))
sys1.add_bearing(brg_roll1)

pinion = SpurHelicalGear(mn=3.0, z=20, b=25.0, position=100.0, label="pinion")
ge_pinion = GearElement(pinion, role="driver", rotation_dir=1, label="pinion")
sys1.add_gear(ge_pinion)

# --- shaft 2: wheel, angular contact (locating) + cylindrical roller (floating) ---
shaft2 = Shaft(label="shaft_2")
shaft2.add_section(ShaftSection(length=250.0, diameter=50.0, label="s1"))
sys2 = ShaftSystem(shaft2, name="shaft_2_system", speed_rpm=500.0, label="output")

cat_ang = BearingCatalog(d=25.0, D=52.0, b=15.0, designation="7205", label="B3_ang", position=30.0, arrangement="locating")
brg_ang = Bearing.assemble(AngularContactFamily(), cat_ang, dict(
    Dw=7.5, Dpw=38.5, Z=10, alpha_0_deg=25.0,
    contact=ContactAnalysis.ISO16281,
    e1=E_BEARING_STEEL, e2=E_BEARING_STEEL, nu1=NU_BEARING_STEEL, nu2=NU_BEARING_STEEL,
))
sys2.add_bearing(brg_ang)

cat_roll2 = BearingCatalog(d=50.0, D=90.0, b=20.0, designation="NU210", label="B4_roll", position=220.0, arrangement="floating")
brg_roll2 = Bearing.assemble(CylindricalRollerFamily(), cat_roll2, dict(
    Dwe=13.0, Lwe=13.0, Dpw=70.0, Z=13, s=0.02, n_s=30,
    contact=ContactAnalysis.ISO16281,
    e1=E_BEARING_STEEL, e2=E_BEARING_STEEL, nu1=NU_BEARING_STEEL, nu2=NU_BEARING_STEEL,
))
sys2.add_bearing(brg_roll2)

wheel = SpurHelicalGear(mn=3.0, z=60, b=25.0, position=100.0, label="wheel")
ge_wheel = GearElement(wheel, role="driven", label="wheel")
sys2.add_gear(ge_wheel)

# --- stage / link ---
meshing = SpurHelicalGearMeshing(pinion, wheel, label="stage_1")
link = SpurHelicalMeshLink(sys1, ge_pinion, sys2, ge_wheel, meshing, phi_deg=0.0, label="stage_1")
system = SpurHelicalGearSystem(shafts=[sys1, sys2], links=[link], label="single_stage_system")
system.resolve(P=5000.0, rpm=1500.0, rotation_dir_source=1)

sys1.validate_or_raise()
sys2.validate_or_raise()

print(sys1.summary())
print(sys2.summary())

# --- FEM analysis (Euler-Bernoulli), um solve por veio ---
settings = BeamModelSettings(beam_theory="euler_bernoulli", shear_theory=None, integration_method=None)
fem_solver = RigidSupportFEMSolver(settings)

fem_results = {
    sys1.name: fem_solver.solve(sys1),
    sys2.name: fem_solver.solve(sys2),
}

# --- ISO/TS 16281 -- distribuição de carga interna por rolamento ---
ball_solver   = ISO16281BallSolver(postprocess=True)
roller_solver = ISO16281RollerSolver(postprocess=True)

iso_results = {}
iso_results.update(ball_solver.solve(sys1, {"B1_dgb": brg_dgb}, fem_results[sys1.name]))
iso_results.update(roller_solver.solve(sys1, {"B2_roll": brg_roll1}, fem_results[sys1.name]))
iso_results.update(ball_solver.solve(sys2, {"B3_ang": brg_ang}, fem_results[sys2.name]))
iso_results.update(roller_solver.solve(sys2, {"B4_roll": brg_roll2}, fem_results[sys2.name]))

for label, res in iso_results.items():
    ld = res.load_distribution
    print(f"\n[{label}] Fr={ld.Fr:.1f} N  delta_r={ld.delta_r:.5f} mm  "
          f"n_loaded={ld.row.n_loaded}/{ld.row.Z}  ok={ld.ok}")
    if res.basic_life is not None:
        print(f"         L10r={res.basic_life.L10r:.2f} x1e6 rev  "
              f"Pref_r={res.basic_life.Pref_r}")

# --- construction report (com FEM) ---
report_path = Path(__file__).resolve().parent / "construction_report.txt"
write_construction_report(system, report_path, title="Single Stage System", fem_results=fem_results)
print(f"\nreport written to: {report_path}")