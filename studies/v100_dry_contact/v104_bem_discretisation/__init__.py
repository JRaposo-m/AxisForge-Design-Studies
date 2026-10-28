"""v104 BEM discretisation: h, domain and kernel on smooth Hertz contacts; the grid rule (README)."""
from . import literature
from .check import check, judge
from .construction import ellipsoid_on_flat, systems

__all__ = ["check", "judge", "literature", "systems", "ellipsoid_on_flat"]
