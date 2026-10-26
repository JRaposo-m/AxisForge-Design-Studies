"""
Comparison of the bearing types of ONE question.

It only READS ``<question>/<type>/results`` of the sibling type folders; it never recomputes and
never imports a run.py. Before comparing it checks that every data set was produced on the same
axes, AxisForge version, psi source and helix angle (``store.check_compatible``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from . import plots
from .store import StudyData, check_compatible, read_results


def load_types(question_dir: Path, type_names: Sequence[str],
               same_axes: bool = True) -> dict[str, StudyData]:
    """Read and validate the data of several bearing types of one question.

    Parameters
    ----------
    question_dir: Path
        ``<question>``.
    type_names: sequence of str
        Type folders to compare (for example ["deep_groove", "cylindrical_roller"]).
    same_axes: bool
        See ``store.check_compatible``.

    Returns
    -------
    datasets: dict
        {type folder name: StudyData}.

    Raises
    ------
    FileNotFoundError
        If a type was never run.
    ValueError
        If the data sets are not comparable.
    """
    datasets = {name: read_results(question_dir / name / "results") for name in type_names}
    check_compatible(datasets, same_axes=same_axes)
    return datasets


def compare_types(question_dir: Path, type_names: Sequence[str], *,
                  x_columns: Sequence[str] | None = None,
                  figure_types: dict[str, Sequence[str]] | None = None) -> dict[str, StudyData]:
    """Draw the comparison figures of one question into ``<question>/compare/plots``.

    Parameters
    ----------
    question_dir: Path
        ``<question>``.
    type_names: sequence of str
        Type folders to compare.
    x_columns: sequence of str, optional (None)
        None: every type has the same axes; one figure per quantity against the first axis,
        one line per type, and one folder per value of the second axis. Otherwise each type sweeps its own
        parameter and one figure is drawn against each of these RESULT columns (for example
        "s_mm", which every type reports), with no grouping.
    figure_types: dict, optional (None)
        {x column: type folders drawn in that figure}; default all of ``type_names``. Used when
        a column has no meaning for some type (alpha0_deg of a roller).

    Returns
    -------
    datasets: dict
        The data that were compared.
    """
    same_axes = x_columns is None
    datasets = load_types(question_dir, type_names, same_axes=same_axes)
    question = next(iter(datasets.values())).meta["question"]
    out_dir = question_dir / "compare" / "plots"
    if same_axes:
        axis_order = next(iter(datasets.values())).meta["axis_order"]
        x_axis = axis_order[0]
        rows = {name: d.rows for name, d in datasets.items()}
        title = f"{question}: " + " vs ".join(type_names)
        if len(axis_order) == 1:
            plots.plot_compare(rows, out_dir / x_axis, x_axis=x_axis, title=title)
            return datasets
        group_axis = axis_order[1]          # one folder per value of the second axis
        for value in next(iter(datasets.values())).meta["axes"][group_axis]:
            point = {group_axis: value}
            plots.plot_compare(rows, out_dir / x_axis / plots.point_name(point, [group_axis]),
                               x_axis=x_axis, group_axis=group_axis, group_value=value,
                               title=f"{title}, {plots.point_title(point, [group_axis])}")
        return datasets
    figure_types = figure_types or {}
    for column in x_columns:
        names = list(figure_types.get(column, type_names))
        plots.plot_compare({name: datasets[name].rows for name in names},
                           out_dir / f"vs_{column}", x_axis=column,
                           title=f"{question}: " + " vs ".join(names))
    return datasets
