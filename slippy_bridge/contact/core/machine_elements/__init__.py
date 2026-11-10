"""Machine elements -> contacts: the only modules that read AxisForge models.

AxisForge works in mm, N and MPa; the bridge in SI. The conversion happens here, once, with the
two factors below, when a contact is built from an AxisForge model.
"""
_MM = 1e-3  # [m/mm]
_MPA = 1e6  # [Pa/MPa]
