"""AxisForge bearings of the ``hertz_vs_bem`` study, assembled and NOT solved.

Scripts 02 and 03 pass these bearings to ``slippy_bridge.contact.contacts`` and check that the
contact the bridge extracts is the study case of ``_common/case.py`` (written there by hand, in
SI). The values are repeated here on purpose, in AxisForge units, so that the check compares two
independent descriptions of the same bearing.

Units: AxisForge (mm, N, MPa). The conversion to SI is made only inside the bridge.

The boundary dimensions (d, D, B) are those of a 6210 / NU 210 and the number of rolling elements
is a typical value; AxisForge needs them to assemble a bearing, the contact of one element with
the inner raceway does not depend on them.
"""
import axisforge.core as af_c

__all__ = ["ball_bearing", "roller_bearing"]

E_STEEL = 210000.0      # Young's modulus [MPa]
NU_STEEL = 0.3          # Poisson's ratio [-]
DPW = 70.0              # pitch diameter [mm]
Z = 9                   # rolling elements [-]

DW = 12.7               # ball diameter [mm]
DWE = 10.0              # roller diameter [mm]
LWE = 10.0              # effective roller length [mm]
N_S = 50                # laminae along the roller [-] (ISO/TS 16281; not used by the contact case)


def _assemble(family, arrangement: str, designation: str, **geometry):
    catalog = af_c.BearingCatalog(d=50.0, D=90.0, b=20.0, designation=designation, position=0.0,
                                  arrangement=arrangement, label=designation)
    return af_c.Bearing.assemble(family=family, catalog=catalog,
                                 geometry={**geometry, "contact": af_c.ContactAnalysis.ISO16281,
                                           "e1": E_STEEL, "e2": E_STEEL, "nu1": NU_STEEL, "nu2": NU_STEEL})


def ball_bearing():
    """Deep groove ball bearing, ``Dw`` 12.7 mm on ``Dpw`` 70 mm.

    Radial clearance ``s = 0``, so the free contact angle is zero and the rolling radius of the
    inner raceway is ``(Dpw - Dw) / 2``. The groove radius is the family default, ``0.52 Dw``.
    """
    return _assemble(af_c.DeepGrooveBallFamily(), "locating", "6210", Dw=DW, Dpw=DPW, Z=Z, s=0.0)


def roller_bearing():
    """Cylindrical roller bearing, ``Dwe`` 10 mm and ``Lwe`` 10 mm on ``Dpw`` 70 mm, ``s = 0``."""
    return _assemble(af_c.CylindricalRollerFamily(), "non-locating", "NU210",
                     Dwe=DWE, Lwe=LWE, Dpw=DPW, Z=Z, n_s=N_S, s=0.0)
