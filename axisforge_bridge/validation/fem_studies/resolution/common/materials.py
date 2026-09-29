"""
validation/fem_studies/resolution/common/materials.py

Registers the materials this suite needs in the core material registry
(axisforge.core.materials.base). That registry is empty until something
calls register(), and Elem.from_mesh()/TorsionSolver look every section's
material_id up in it (KeyError "Material 'S355' not found. Available: ").

Values are the old core/materials.py S355 (Shigley-based library):
E = 210 GPa, nu = 0.3 (that library's default), rho = 7850 kg/m^3,
Sut = 590 MPa, Sy = 355 MPa. The FEM solve only reads E and nu.

Idempotent: safe to call from every case script / from build_system().
"""
from __future__ import annotations

from axisforge.core.materials.base import (
    IsotropicElastic,
    Material,
    StrengthProperties,
    available_materials,
    register,
)


def ensure_case_materials() -> None:
    if "S355" in available_materials():
        return
    register(Material(
        material_id="S355",
        density=7850.0,
        elastic=IsotropicElastic(E=210_000.0, poisson_ratio=0.3),
        strength=StrengthProperties(Sut=590.0, Sy=355.0),
        description="EN 10025-2 S355 -- structural steel",
    ))