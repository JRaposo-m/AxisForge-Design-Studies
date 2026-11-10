"""v005 shaft on three supports: AxisForge shaft FEM on three rigid, elastic or settled supports against r01 extended,
on three Hertz bearings against Harris Eq. 9.48 and against itself (README)."""
from . import literature
from .check import along_shaft, check, reference, supports
from .construction import MODELS, SETTLEMENTS_MM, SPRINGS, stiff_material, three_support_shaft

__all__ = ["check", "reference", "supports", "along_shaft", "literature", "three_support_shaft", "stiff_material",
           "MODELS", "SPRINGS", "SETTLEMENTS_MM"]
