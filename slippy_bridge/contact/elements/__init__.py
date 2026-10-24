"""Machine elements of the bridge: ``bearings`` and ``gears``, one module each.

AxisForge gives lengths in mm and moduli in MPa; the bridge works in m and Pa. Both element
modules convert with these two factors, at the boundary.
"""

_MM = 1e-3      # [m/mm]
_MPA = 1e6      # [Pa/MPa]
