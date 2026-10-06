"""
validation/fem_studies/resolution/common/materials.py
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