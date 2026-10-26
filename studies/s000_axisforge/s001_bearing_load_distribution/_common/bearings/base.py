"""
Base class of the bearing specifications of the load distribution studies.

A specification is a small frozen record of the parameters of ONE bearing type (envelope and
internal geometry). It knows its AxisForge family and how to assemble an ``af_c.Bearing``; it
does not know where it sits on the shaft (that is decided by ``system.py`` from its
arrangement).

The geometries are DIDACTIC: they are chosen to show the influence of each parameter, not to
represent a catalogue bearing. ``designation`` is only a label.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import ClassVar


@dataclass(frozen=True)
class BearingSpec(ABC):
    """Parameters of one bearing type.

    Class attributes
    ----------------
    kind: str
        Short name used in the bearing labels and in the result tables
        ("ball" | "angular" | "roller").
    arrangement: str
        "locating" (carries the axial load) or "non-locating" (axially free).

    Attributes
    ----------
    d_mm, D_mm, b_mm: float
        Bore, outside diameter and width of the envelope [mm].
    designation: str
        Label only; no catalogue data is read from it.
    """

    kind: ClassVar[str]
    arrangement: ClassVar[str]

    d_mm: float = 20.0
    D_mm: float = 42.0
    b_mm: float = 12.0
    designation: str = "didactic"

    @abstractmethod
    def family(self):
        """AxisForge bearing family of this type.

        Returns
        -------
        family: BearingFamily
            A new family instance (for example ``af_c.DeepGrooveBallFamily()``).
        """

    @abstractmethod
    def geometry(self) -> dict:
        """Internal geometry handed to ``af_c.Bearing.assemble``.

        Returns
        -------
        geometry: dict
            Family-specific keyword arguments (Dw, Dpw, Z, s, ...), without contact data.
        """

    def assemble(self, label: str, position: float, contact: dict):
        """Assemble the AxisForge bearing.

        Parameters
        ----------
        label: str
            Bearing label; it must match the bearing node of the shaft FEM.
        position: float
            Axial position of the bearing on the shaft [mm].
        contact: dict
            ISO/TS 16281 contact data (contact model, e1, e2, nu1, nu2).

        Returns
        -------
        bearing: af_c.Bearing
            The assembled bearing.
        """
        import axisforge.core as af_c

        catalog = af_c.BearingCatalog(d=self.d_mm, D=self.D_mm, b=self.b_mm,
                                      designation=self.designation, position=position,
                                      arrangement=self.arrangement, label=label)
        return af_c.Bearing.assemble(family=self.family(), catalog=catalog,
                                     geometry={**self.geometry(), **contact})

    def describe(self) -> dict:
        """Plain description for the run information of report.txt.

        Returns
        -------
        description: dict
            Class name, kind, arrangement and every parameter.
        """
        return dict(spec=type(self).__name__, kind=self.kind, arrangement=self.arrangement,
                    **asdict(self))
