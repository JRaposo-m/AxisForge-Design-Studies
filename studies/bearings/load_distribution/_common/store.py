"""
Writing and reading the results of one bearing type: ``<question>/<type>/results``.

The files are described in ``reports.py`` (report.txt and the CSV tables). The results folder
is the ONLY source of the results of that type: ``compare`` reads it and never recomputes.
What produced the results (question, axes, bearing specifications, helix angle, power, psi
source, AxisForge version, parameter hash, date) is the RUN INFORMATION block of report.txt;
``check_compatible`` reads it and refuses to compare data produced under different conditions.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from axisforge.outputs import _format as af_format

from . import reports
from .rows import BOOL_FIELDS, TEXT_FIELDS

REPORT_FILE = "report.txt"
SYSTEM_FILE = "system.csv"
FEM_FILE = "fem_bearing_nodes.csv"
FEM_SUMMARY_FILE = "fem_shaft_summary.csv"
SYSTEM_TEXT_FILE = "system.txt"
ANALYSIS_FILE = "analysis_reference.txt"
ROWS_FILE = "iso16281_rows.csv"
Q_FILE = "iso16281_elements.csv"
LAMINA_FILE = "iso16281_laminae.csv"
META_VERSION = 2

# run-information keys that must be identical for two data sets to be compared
COMPATIBILITY_KEYS = ("axisforge_version", "psi_source", "helix_angle_deg", "power_W")


@dataclass
class StudyData:
    """The results of one bearing type.

    Attributes
    ----------
    rows, q_rows, lam_rows: list of dict
        ISO/TS 16281 results, see ``rows``.
    meta: dict
        Run information (written at the top of report.txt).
    system_rows: list of dict
        {section, parameter, value, unit} of the system.
    system_text: str
        The complete description of the system as AxisForge prints it (system.txt).
    system_summary: str
        The short description for report.txt.
    fem_summary_rows: list of dict
        Maxima of the shaft FEM result, one per (power, shaft).
    fem_rows: list of dict
        Bearing-node results of the shaft FEM, one per (power, shaft, node).
    analysis_text: str
        ``analysis_text`` of AxisForge for the first point of the sweep and the first shaft.
    """

    rows: list[dict]
    q_rows: list[dict]
    lam_rows: list[dict]
    meta: dict
    system_rows: list[dict] = field(default_factory=list)
    system_text: str = ""
    system_summary: str = ""
    fem_summary_rows: list[dict] = field(default_factory=list)
    fem_rows: list[dict] = field(default_factory=list)
    analysis_text: str = ""


def axisforge_version() -> str:
    """Installed AxisForge version, or "unknown".

    Returns
    -------
    version: str
        Distribution version of the ``axisforge`` package.
    """
    try:
        from importlib.metadata import version
        return version("axisforge")
    except Exception:                                               # noqa: BLE001
        return "unknown"


def parameter_hash(payload: dict) -> str:
    """Short hash of everything that defines the results.

    Parameters
    ----------
    payload: dict
        JSON-serialisable description (axes, specifications, helix angle, psi source).

    Returns
    -------
    digest: str
        First 16 hexadecimal digits of the SHA-256 of the canonical JSON.
    """
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def make_meta(*, question: str, axes: dict, system: dict, study_kind: str,
              psi_source: str, psi_fem_mrad: dict) -> dict:
    """Run information of one run.

    Parameters
    ----------
    question: str
        Name of the question (folder name).
    axes: dict
        {axis name: list of values}, in sweep order.
    system: dict
        ``SystemSpec.describe()`` at the nominal power.
    study_kind: str
        Which slot holds the bearing under study ("locating" | "non-locating").
    psi_source: str
        "fem" (FEM slope) or "imposed" (prescribed by the question).
    psi_fem_mrad: dict
        {shaft name: |psi| [mrad]} of the FEM for the bearing under study, nominal power.

    Returns
    -------
    meta: dict
        The RUN INFORMATION block of report.txt.
    """
    axes = {name: [float(v) for v in values] for name, values in axes.items()}
    defining = dict(axes=axes, system=system, study_kind=study_kind, psi_source=psi_source)
    return dict(meta_version=META_VERSION, question=question, axis_order=list(axes), axes=axes,
                study_kind=study_kind, helix_angle_deg=system["helix_angle_deg"],
                power_W=system["power_W"], psi_source=psi_source, psi_fem_mrad=psi_fem_mrad,
                system=system, axisforge_version=axisforge_version(),
                parameter_hash=parameter_hash(defining), created_utc=reports.utc_now())


def write_results(results_dir: Path, data: StudyData, *, row_fields, q_fields, lamina_fields,
                  type_name: str) -> None:
    """Write report.txt and the CSV tables of one bearing type.

    Parameters
    ----------
    results_dir: Path
        ``<question>/<type>/results``; created if missing.
    data: StudyData
        The results.
    row_fields, q_fields, lamina_fields: list of str
        Column order of the ISO/TS 16281 tables.
    type_name: str
        Name of the type folder (report title).
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    axis_names = data.meta["axis_order"]
    with open(results_dir / REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(reports.build_report(data, axis_names, type_name))
    af_format.write_records_csv(results_dir / SYSTEM_FILE, data.system_rows,
                                ["section", "parameter", "value", "unit"])
    af_format.write_records_csv(results_dir / FEM_FILE, data.fem_rows)
    af_format.write_records_csv(results_dir / FEM_SUMMARY_FILE, data.fem_summary_rows)
    (results_dir / SYSTEM_TEXT_FILE).write_text(data.system_text + "\n", encoding="utf-8")
    (results_dir / ANALYSIS_FILE).write_text(
        "AxisForge result of the first point of the sweep, first shaft:\n\n"
        + data.analysis_text + "\n", encoding="utf-8")
    af_format.write_records_csv(results_dir / ROWS_FILE, data.rows, row_fields)
    af_format.write_records_csv(results_dir / Q_FILE, data.q_rows, q_fields)
    lamina_path = results_dir / LAMINA_FILE
    if data.lam_rows:
        af_format.write_records_csv(lamina_path, data.lam_rows, lamina_fields)
    elif lamina_path.exists():
        lamina_path.unlink()                  # stale file of an earlier run


def _read_csv(path: Path) -> list[dict]:
    """Records of one results CSV; empty if the file does not exist."""
    if not path.exists():
        return []
    return af_format.read_records_csv(path, text_fields=TEXT_FIELDS, bool_fields=BOOL_FIELDS)


def read_results(results_dir: Path) -> StudyData:
    """Read the results of one bearing type.

    Parameters
    ----------
    results_dir: Path
        ``<question>/<type>/results``.

    Returns
    -------
    data: StudyData
        ISO/TS 16281 tables (numbers as float, ok / postprocessed as bool, empty cells as NaN),
        FEM table and run information.

    Raises
    ------
    FileNotFoundError
        If report.txt or iso16281_rows.csv is missing (the type was never run).
    """
    for name in (REPORT_FILE, ROWS_FILE):
        if not (results_dir / name).exists():
            raise FileNotFoundError(f"{results_dir / name} is missing: run that type first")
    meta = reports.read_run_information((results_dir / REPORT_FILE).read_text(encoding="utf-8"))
    return StudyData(rows=_read_csv(results_dir / ROWS_FILE), q_rows=_read_csv(results_dir / Q_FILE),
                     lam_rows=_read_csv(results_dir / LAMINA_FILE), meta=meta,
                     system_rows=_read_csv(results_dir / SYSTEM_FILE),
                     fem_rows=_read_csv(results_dir / FEM_FILE),
                     fem_summary_rows=_read_csv(results_dir / FEM_SUMMARY_FILE))


def check_compatible(datasets: dict[str, StudyData], same_axes: bool = True) -> None:
    """Refuse to compare data sets that were not produced under the same conditions.

    Parameters
    ----------
    datasets: dict
        {type folder name: StudyData}, all of the same question.
    same_axes: bool
        True: the axes (names, order and values) must be identical. False: each type sweeps
        its own parameter and the comparison uses a result column common to all of them.

    Raises
    ------
    ValueError
        If the question, the axes (when ``same_axes``), the AxisForge version, the psi source,
        the helix angle or the nominal power differ; the message lists every difference.
    """
    if len(datasets) < 2:
        raise ValueError("compare needs the data of at least two bearing types")
    names = list(datasets)
    reference = datasets[names[0]].meta
    problems = []
    for name in names[1:]:
        meta = datasets[name].meta
        keys = ["question", *COMPATIBILITY_KEYS] + (["axis_order", "axes"] if same_axes else [])
        for key in keys:
            if meta.get(key) != reference.get(key):
                problems.append(f"{name}: {key} = {meta.get(key)!r} "
                                f"!= {reference.get(key)!r} ({names[0]})")
    if problems:
        raise ValueError("data sets are not comparable (re-run the types):\n  "
                         + "\n  ".join(problems))
