"""Inputs of the contact part of the bridge, and the registry of machine elements.

* :class:`ContactCase` - the Hertz inputs of one contact (radii, materials, load), SI units,
  ``x`` = rolling direction. Pure data; no AxisForge and no SlipPY import.
* :class:`Contact` - one contact of one machine element: the case plus *which* element, subtype
  and location it is, and (optionally) its kinematics. This is what the elements hand to the
  surface, reference and step modules, and what the outputs record.
* :class:`ElementAdapter` - the translator of one machine element (bearing, gear) into a list of
  :class:`Contact`; one subclass per element, stateless.
* Registry - every machine element subtype that the bridge can translate is registered here by
  its AxisForge class (bearing family, gear mesh), as AxisForge registers its own families.
  :func:`contacts` dispatches to the registered adapter; an unregistered subtype raises.
"""
from __future__ import annotations

import hashlib
import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

__all__ = ["ContactCase", "Contact", "ElementAdapter", "register", "contacts", "registered"]

# AxisForge -> SI factors, used only by the element adapters at the boundary (R-001)
_MM = 1e-3      # [m/mm]
_MPA = 1e6      # [Pa/MPa]


def _pair(value, name: str) -> tuple[float, float]:
    """Return ``value`` as a 2-tuple of floats (a scalar is repeated), as ``hertz_full`` does."""
    if isinstance(value, (int, float)):
        value = (value, value)
    value = tuple(float(v) for v in value)
    if len(value) == 1:
        value = (value[0], value[0])
    if len(value) != 2:
        raise ValueError(f"{name} must be a number or a 2-element sequence, got {value}")
    return value


@dataclass(frozen=True)
class ContactCase:
    """Inputs of one Hertzian contact, in the form ``hertz_full`` expects.

    Parameters
    ----------
    label : str
        Short identifier used in file names and reports (no spaces).
    r1, r2 : float or tuple of float
        Principal radii of curvature ``(r_x, r_y)`` of body 1 (rolling element) and body 2
        (raceway) at the contact point [m]. ``x`` is the rolling direction. A scalar means
        ``(r, r)``. Sign convention: convex > 0, concave < 0, flat = ``math.inf``. These are
        radii of curvature, **not** ellipsoid semi-axes (compare ``slippy.surface.RoundSurface``).
    E : float or tuple of float
        Young's moduli ``(E1, E2)`` [Pa]; a scalar applies to both bodies (``moduli`` in SlipPY's
        ``hertz_full``).
    nu : float or tuple of float
        Poisson's ratios ``(nu1, nu2)`` [-]; a scalar applies to both bodies (``v`` in SlipPY's
        ``hertz_full``).
    load : float
        Normal load [N] for a point contact; load per unit length [N/m] for a line contact.
    angle : float, optional
        Angle between the x axes of the two bodies [rad]. Default 0.
    line : bool, optional
        True for a line contact (parallel cylinders), as the ``line`` flag of ``hertz_full``.

    Raises
    ------
    ValueError
        If a radius is zero or NaN, a modulus or the load is not positive, or a Poisson's ratio
        is outside (0, 0.5).

    References
    ----------
    Johnson, K. L. (1985). *Contact Mechanics*. Cambridge University Press, ch. 4.

    Examples
    --------
    A 12.7 mm steel ball on a flat, 500 N:

    >>> c = ContactCase("ball_flat", r1=6.35e-3, r2=math.inf, E=208e9, nu=0.3, load=500.0)
    >>> c.hertz_args()["r1"]
    (0.00635, 0.00635)
    """

    label: str
    r1: tuple[float, float]
    r2: tuple[float, float]
    E: tuple[float, float]
    nu: tuple[float, float]
    load: float
    angle: float = 0.0
    line: bool = False

    def __post_init__(self) -> None:
        for name in ("r1", "r2", "E", "nu"):
            object.__setattr__(self, name, _pair(getattr(self, name), name))
        for name in ("r1", "r2"):
            if any(r == 0 or math.isnan(r) for r in getattr(self, name)):
                raise ValueError(f"{name} contains a zero or NaN radius (use math.inf for flat)")
        if any(e <= 0 for e in self.E):
            raise ValueError(f"E must be positive, got {self.E}")
        if any(not 0.0 < n < 0.5 for n in self.nu):
            raise ValueError(f"nu must be in (0, 0.5), got {self.nu}")
        if not self.load > 0:
            raise ValueError(f"load must be positive, got {self.load}")

    def hertz_args(self) -> dict:
        """Return the keyword arguments for ``slippy.contact.hertz_full(**case.hertz_args())``."""
        return {"r1": self.r1, "r2": self.r2, "moduli": self.E, "v": self.nu,
                "load": self.load, "angle": self.angle, "line": self.line}

    def to_dict(self) -> dict:
        """Return the case as a JSON-serialisable dict (``inf`` written as the string ``"inf"``).

        :attr:`E` and :attr:`nu` are written under the keys ``moduli`` and ``v``. This dict is
        what :meth:`case_hash` hashes, so the keys are kept and the hash of a saved result stays
        the same.
        """

        def enc(x):
            if isinstance(x, tuple):
                return [enc(i) for i in x]
            if isinstance(x, float) and math.isinf(x):
                return "inf" if x > 0 else "-inf"
            return x

        return {"label": self.label, "r1": enc(self.r1), "r2": enc(self.r2),
                "moduli": list(self.E), "v": list(self.nu), "load": self.load,
                "angle": self.angle, "line": self.line}

    def case_hash(self) -> str:
        """Return a short SHA-256 hash of the physical content of the case (label excluded)."""
        d = self.to_dict()
        d.pop("label")
        return hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


# =============================================================================
# One contact of one machine element
# =============================================================================

@dataclass(frozen=True)
class Contact:
    """One contact of one machine element, ready for the surface, reference and step modules.

    Parameters
    ----------
    element : str
        Machine element: ``"bearing"`` or ``"gear"``.
    subtype : str
        AxisForge class of the element subtype (e.g. ``"DeepGrooveBallFamily"``).
    location : dict
        Where the contact is, element-specific, units in the keys. Bearings: ``race`` and
        ``alpha_rad``, the contact angle used (the given one or ``alpha_0`` with a load, ``alpha_j``
        with a result); with a result also ``element_j``, ``row``, ``phi_local_rad``, and
        ``phi_global_rad`` when the result is a ``BearingResult`` (single or multi row; not a bare
        row; convention: see AxisForge ``BearingResult.phi_j_global``). For a roller lamina also
        ``lamina_k`` and ``x_k_mm``, the position of the lamina along the roller axis (AxisForge
        ``x_k`` [mm]; the roller axis is the bridge ``y``, across rolling). Gears: the point on the
        path of contact (``point``, ``s_mm``, ``g_y_mm``, ``t_s``) and its load sharing (``share``,
        ``n_pairs``, ``l_sum_mm``); see ``SpurHelicalContacts.contacts``.
    case : ContactCase
        The Hertz inputs.
    kinematics : dict, optional
        ``u_e_m_per_s``: MEAN entrainment speed (u1+u2)/2 [m/s], never the sum u1+u2 (R-075, R-028);
        ``slide_roll`` (sliding/entrainment ratio) [-], when the caller gave the speeds; empty
        otherwise. Needed by lubrication, not by the static step.
    """

    element: str
    subtype: str
    location: dict
    case: ContactCase
    kinematics: dict = field(default_factory=dict)

    def record(self) -> dict:
        """Flat record (identification + case hash + kinematics), for the outputs."""
        head = {"element": self.element, "subtype": self.subtype}
        case = {"label": self.case.label, "case_hash": self.case.case_hash(), "line": self.case.line}
        parts = (head, self.location, case, self.kinematics)
        keys = [k for part in parts for k in part]
        if len(keys) != len(set(keys)):
            raise ValueError(f"record keys overlap: {sorted({k for k in keys if keys.count(k) > 1})}")
        return {k: v for part in parts for k, v in part.items()}


# =============================================================================
# Registry: AxisForge subtype class -> adapter
# =============================================================================

class ElementAdapter(ABC):
    """Translator of one AxisForge machine element into the contacts it contains.

    One subclass per machine element (bearing, gear mesh). An adapter holds no state and is never
    instantiated: it is the named place of the conversion AxisForge (mm, N, MPa, its own sign of
    the outer raceway) -> :class:`Contact` (SI, convex > 0, ``x`` = rolling direction), registered
    with :func:`register` for the AxisForge subtype classes it serves.

    Attributes
    ----------
    element : str
        Machine element the adapter serves (``"bearing"``, ``"gear"``); written to
        ``Contact.element`` and used as the key of :func:`registered`.
    """

    element: ClassVar[str]

    @classmethod
    @abstractmethod
    def contacts(cls, obj: Any, *args, **kwargs) -> list[Contact]:
        """Return the requested contacts of the AxisForge object ``obj`` (SI units)."""


_ADAPTERS: dict[type, type[ElementAdapter]] = {}


def register(*subtypes: type):
    """Class-level registration of an adapter for one or more AxisForge subtype classes.

    Used as a decorator on an :class:`ElementAdapter` subclass; the machine element is read from
    its ``element`` attribute.

    Raises
    ------
    TypeError
        If the decorated object is not a subclass of :class:`ElementAdapter`.
    ValueError
        If one of the subtypes already has an adapter.
    """
    def wrap(adapter: type[ElementAdapter]) -> type[ElementAdapter]:
        if not (isinstance(adapter, type) and issubclass(adapter, ElementAdapter)):
            raise TypeError(f"register decorates an ElementAdapter subclass, got {adapter!r}")
        for cls in subtypes:
            if cls in _ADAPTERS:
                raise ValueError(f"{cls.__name__} is already registered")
            _ADAPTERS[cls] = adapter
        return adapter
    return wrap


def _subtype(obj: Any) -> type:
    """The registry key of an AxisForge object: a bearing's family class, else the object's class."""
    family = getattr(obj, "family", None)
    return type(family) if family is not None else type(obj)


def contacts(obj: Any, *args, **kwargs) -> list[Contact]:
    """All requested contacts of an AxisForge machine element, through its registered adapter.

    Raises
    ------
    NotImplementedError
        If the element subtype has no adapter (the message lists the registered ones).
    """
    from . import elements  # noqa: F401  (registers the adapters if contact/__init__ did not)
    key = _subtype(obj)
    if key not in _ADAPTERS:
        raise NotImplementedError(f"no SlipPY adapter for {key.__name__}; registered: "
                                  f"{', '.join(sorted(c.__name__ for c in _ADAPTERS)) or 'none'}")
    return _ADAPTERS[key].contacts(obj, *args, **kwargs)


def registered() -> dict[str, list[str]]:
    """Registered subtypes per machine element: ``{"bearing": [...], "gear": [...]}``."""
    from . import elements  # noqa: F401
    out: dict[str, list[str]] = {}
    for cls, adapter in _ADAPTERS.items():
        out.setdefault(adapter.element, []).append(cls.__name__)
    return {k: sorted(v) for k, v in out.items()}
