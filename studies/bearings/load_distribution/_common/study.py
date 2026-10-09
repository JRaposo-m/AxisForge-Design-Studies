"""
LoadDistributionStudy: the sweep loop shared by every question of the family.

Who decides what
----------------
<question>/study.py      the QUESTION: the swept axes (names and values), the helix angle and
                         how a point of the sweep changes the bearing, the transmitted
                         power or psi (the hooks ``configure``, ``power_W``, ``psi``)
<question>/<type>/run.py the BEARING UNDER STUDY: which BearingSpec (and, optionally, the
                         partner); it runs the question and writes results/ and plots/
this module              everything else: system, FEM, loop, rows, derived columns, files

The sweep is the cartesian product of the axes, in the order they are declared (the first
axis is the outer loop and the x axis of the figures). Only the bearing under study is solved:
with rigid supports the partner does not change its results (see ``system.py``); the partner
is there so that the shaft has its two supports.

The load is a construction parameter: each distinct transmitted power is its own system and its
own FEM solve (cached by power, since with rigid supports the FEM does not depend on the
bearings). Nothing is scaled after the solve.
"""
from __future__ import annotations

import itertools
from abc import ABC
from pathlib import Path
from typing import ClassVar, Sequence

from axisforge.outputs.solvers.bearings import bearing_analysis_result as af_analysis

from . import plots, reports
from .bearings import BearingSpec, CylindricalRollerSpec, DeepGrooveSpec
from .rows import add_derived, lamina_fields, make_row, q_fields, row_fields
from .solve import solve_bearing, solve_fem
from .store import StudyData, make_meta, write_results
from .system import NOMINAL_POWER_W, SystemSpec, assemble_bearing, build_system

# keyed by the slot the partner fills (the one the bearing under study leaves free)
DEFAULT_PARTNER = {
    "non-locating": CylindricalRollerSpec,  # a locating bearing under study: roller partner
    "locating": DeepGrooveSpec,             # a roller under study: deep groove partner
}


class LoadDistributionStudy(ABC):
    """One question of the load distribution family, for one bearing under study.

    Class attributes (set by the question)
    ---------------------------------------
    question: str
        Name of the question; also its folder name.
    axes: dict
        {axis name: values}; the sweep is their cartesian product, in this order. Used when
        the question has the same axes for every type.
    axes_by_kind: dict, optional
        {BearingSpec.kind: axes} when each type sweeps its own parameter (for example s for the
        deep groove and the roller, alpha_0 for the angular contact bearing). A type missing
        from it does not answer the question. Takes precedence over ``axes``.
    helix_angle_deg: float
        Gear helix angle of the question [deg]; 0 = spur gears (Fa = 0).
    theory_key: str
        Beam model of the shaft FEM (key of ``system.THEORY_ARGS``).

    Parameters
    ----------
    spec: BearingSpec
        The bearing under study; its arrangement decides its slot.
    partner: BearingSpec, optional (None)
        The other bearing; None takes DEFAULT_PARTNER of the slot left free.

    Raises
    ------
    TypeError
        If ``spec`` is not a BearingSpec, or the pair breaks the construction rule.
    ValueError
        If the partner sits in the same slot as the bearing under study, the type does not
        answer the question, or an axis is empty.
    """

    question: ClassVar[str]
    axes: dict[str, Sequence[float]]
    axes_by_kind: ClassVar[dict[str, dict[str, Sequence[float]]] | None] = None
    helix_angle_deg: ClassVar[float] = 0.0
    theory_key: ClassVar[str] = "timoshenko/cowper"

    def __init__(self, spec: BearingSpec, partner: BearingSpec | None = None):
        if not isinstance(spec, BearingSpec):
            raise TypeError(f"spec must be a BearingSpec, got {type(spec).__name__}")
        free_slot = "non-locating" if spec.arrangement == "locating" else "locating"
        partner = DEFAULT_PARTNER[free_slot]() if partner is None else partner
        if partner.arrangement != free_slot:
            raise ValueError(f"the partner must be {free_slot}, got {type(partner).__name__} "
                             f"({partner.arrangement})")
        if self.axes_by_kind is not None:
            if spec.kind not in self.axes_by_kind:
                raise ValueError(f"{type(spec).__name__} does not answer the {self.question!r} "
                                 f"question (types: {sorted(self.axes_by_kind)})")
            self.axes = dict(self.axes_by_kind[spec.kind])
        for name, values in self.axes.items():
            if len(values) == 0:
                raise ValueError(f"axis {name!r} has no values")
        self.spec = spec
        self.partner = partner
        # raises TypeError if the pair is not allowed
        self.system_spec()

    # --- hooks of the question -------------------------------------------------------------

    def configure(self, spec: BearingSpec, **point) -> BearingSpec:
        """The bearing under study at one point of the sweep (default: unchanged).

        Parameters
        ----------
        spec: BearingSpec
            The bearing given to the study.
        **point: float
            {axis name: value} of this point.

        Returns
        -------
        spec: BearingSpec
            The bearing to solve at this point.
        """
        return spec

    def power_W(self, **point) -> float:
        """Transmitted power at one point (default: the ``power_W`` axis, else the nominal power).

        Parameters
        ----------
        **point: float
            {axis name: value} of this point.

        Returns
        -------
        power_W: float
            Power of the gear stage [W] at the speed of the driving shaft; the nominal value
            (system.NOMINAL_POWER_W) gives 100 N m at 1000 rpm.
        """
        return float(point.get("power_W", NOMINAL_POWER_W))

    def psi(self, **point) -> float | None:
        """Prescribed inner-ring tilt at one point (default: the ``psi_mrad`` axis, else None).

        Parameters
        ----------
        **point: float
            {axis name: value} of this point.

        Returns
        -------
        psi_mrad: float or None
            Prescribed tilt [mrad]; None keeps the FEM slope.
        """
        value = point.get("psi_mrad")
        return None if value is None else float(value)

    # --- the shared part -------------------------------------------------------------------

    @property
    def axis_names(self) -> list[str]:
        """Names of the swept axes, in sweep order."""
        return list(self.axes)

    def points(self) -> list[dict]:
        """Every point of the sweep.

        Returns
        -------
        points: list of dict
            {axis name: value}, cartesian product of the axes (first axis outermost).
        """
        names = self.axis_names
        return [dict(zip(names, values)) for values in itertools.product(*self.axes.values())]

    def system_spec(self, power_W: float = NOMINAL_POWER_W) -> SystemSpec:
        """The system: bearing under study in its slot, partner in the other.

        Parameters
        ----------
        power_W: float
            Transmitted power [W].

        Returns
        -------
        spec: SystemSpec
            Bearings, helix angle and power of this study.
        """
        if self.spec.arrangement == "locating":
            locating, non_locating = self.spec, self.partner
        else:
            locating, non_locating = self.partner, self.spec
        return SystemSpec(locating=locating, non_locating=non_locating,
                          helix_angle_deg=self.helix_angle_deg, power_W=power_W,
                          label=self.question)

    def run(self) -> StudyData:
        """Build and solve the system at each power, and sweep the bearing under study.

        Returns
        -------
        data: StudyData
            Rows (one per point and shaft, derived columns filled), element and lamina loads,
            the run information, the system description and the FEM bearing-node results.
        """
        kind = self.spec.kind
        solved: dict[float, tuple] = {}            # power [W] -> (system, fem)

        def system_at(power):
            if power not in solved:
                system = build_system(self.system_spec(power))
                solved[power] = (system, solve_fem(system, self.theory_key))
            return solved[power]

        # FEM slope at the bearing under study at the nominal power, for the record and figures
        system, fem = system_at(NOMINAL_POWER_W)
        psi_fem_mrad = {}
        for ss in system.shafts:
            analysis, _, _ = solve_bearing(ss, assemble_bearing(self.spec, ss.name), fem[ss.name])
            if analysis is not None:
                psi_fem_mrad[ss.name] = abs(1e3 * analysis.load_distribution.psi)

        points = self.points()
        rows, q_rows, lam_rows = [], [], []
        analysis_text = ""
        for point in points:
            spec = self.configure(self.spec, **point)
            power = self.power_W(**point)
            system, fem = system_at(power)
            psi_mrad = self.psi(**point)
            psi_rad = None if psi_mrad is None else 1e-3 * psi_mrad
            for ss in system.shafts:
                bearing = assemble_bearing(spec, ss.name)
                analysis, error, postprocessed = solve_bearing(ss, bearing, fem[ss.name], psi_rad)
                if analysis is not None and not analysis_text:       # first solved point
                    analysis_text = af_analysis.analysis_text(analysis)
                row, q, lam_k = make_row(point, ss.name, ss.speed_rpm, bearing, kind, analysis,
                                         error, postprocessed)
                row["power_W"] = power
                rows.append(row)
                q_rows.extend(q)
                lam_rows.extend(lam_k)
        add_derived(rows, self.axis_names)

        psi_source = "fem" if self.psi(**points[0]) is None else "imposed"
        meta = make_meta(question=self.question, axes={k: list(v) for k, v in self.axes.items()},
                         system=self.system_spec().describe(), study_kind=self.spec.arrangement,
                         psi_source=psi_source, psi_fem_mrad=psi_fem_mrad)
        nominal_system, _ = solved[NOMINAL_POWER_W]
        system_rows, system_text = reports.describe_system(
            self.system_spec(), nominal_system, study_slot=self.spec.arrangement)
        fem_rows = [r for power in sorted(solved)
                    for r in reports.fem_node_rows(power, solved[power][1])]
        fem_summary_rows = [r for power in sorted(solved)
                            for r in reports.fem_summary_rows(power, solved[power][1])]
        system_summary = reports.summarize_system(self.system_spec(), nominal_system,
                                                  study_slot=self.spec.arrangement)
        return StudyData(rows=rows, q_rows=q_rows, lam_rows=lam_rows, meta=meta,
                         system_rows=system_rows, system_text=system_text, fem_rows=fem_rows,
                         analysis_text=analysis_text, system_summary=system_summary,
                         fem_summary_rows=fem_summary_rows)

    def figures(self, data: StudyData, plots_dir: Path) -> None:
        """Figures of one bearing type (override to add or change figures).

        Parameters
        ----------
        data: StudyData
            From ``run``.
        plots_dir: Path
            ``<question>/<type>/plots``.
        """
        names = self.axis_names
        x_axis = names[0]
        group_axis = names[1] if len(names) > 1 else None
        title = f"{self.question}: {plots.KIND_NAME.get(self.spec.kind, self.spec.kind)}"
        plots.plot_summary(data.rows, plots_dir / "summary", x_axis=x_axis,
                           group_axis=group_axis, title=title)
        plots.plot_polar_distributions(data.rows, data.q_rows, plots_dir, axis_names=names,
                                       title=title)
        if data.lam_rows:
            plots.plot_lamina_profile(data.lam_rows, plots_dir,
                                      x_axis=x_axis, group_axis=group_axis, title=title)

    def run_and_save(self, type_dir: Path) -> StudyData:
        """Run, check, write ``results/`` and draw ``plots/`` of one bearing type.

        Parameters
        ----------
        type_dir: Path
            ``<question>/<type>`` (the folder of the run.py).

        Returns
        -------
        data: StudyData
            The results that were written.
        """
        data = self.run()
        print(f"== {self.question} / {type_dir.name}  ({len(data.rows)} rows)")
        for line in reports.check_lines(data.rows, self.axis_names,
                                        spur=self.helix_angle_deg == 0.0):
            print(line)
        names = self.axis_names
        write_results(type_dir / "results", data, row_fields=row_fields(names),
                      q_fields=q_fields(names), lamina_fields=lamina_fields(names),
                      type_name=type_dir.name)
        self.figures(data, type_dir / "plots")
        print(f"results: {type_dir / 'results'}\nplots:   {type_dir / 'plots'}")
        return data
