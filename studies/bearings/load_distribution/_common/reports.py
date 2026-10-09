"""
Text reports and tables of one bearing type: ``<question>/<type>/results``.

Files
-----
report.txt              the report to read, in five short sections:
                          1 RUN INFORMATION   what produced the results (also read by compare
                                              and by the regression tests: do not edit)
                          2 CONSTRUCTION      what was built: gear stage, shafts, bearings, loads
                          3 SOLVER: SHAFT FEM what the shaft FEM returned: bearing nodes, maxima
                                              and the values at every mesh node
                          4 SOLVER: ISO/TS 16281  one line per point and shaft, then the
                                              element loads of the reference shaft
                          5 CHECKS            convergence, equilibrium, symmetry, errors
analysis_reference.txt  AxisForge's own result text for the first point of the sweep
system.csv              section, parameter, value, unit
fem_bearing_nodes.csv   one row per (power, shaft, bearing node), every numeric node field
fem_shaft_summary.csv   one row per (power, shaft): maxima of the shaft FEM result
fem_mesh_nodes.csv      one row per (power, shaft, mesh node): the FEM along the shaft
iso16281_rows.csv       one row per (point, shaft): see rows.py
iso16281_elements.csv   rolling-element loads Q_j
iso16281_laminae.csv    lamina loads of the most loaded roller (line contact only)

The CSV files are for Excel or any other tool; the report is for reading and for writing up.

The description of the system and the bearing-node results of the FEM come from
``axisforge.outputs`` (records and text of AxisForge's own objects); the tables of the ISO/TS
16281 results, the checks and the RUN INFORMATION block belong to this study family.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone

from axisforge.outputs import _format as af_format
from axisforge.outputs.construction import system as af_system
from axisforge.outputs.solvers.fem_results import shaft_results as af_shaft_results

from . import system as sysmod

RUN_HEADER = "1  RUN INFORMATION"
RUN_NOTE = "(read by compare/ and tests/: do not edit)"
WIDTH = 100


# =============================================================================
# Collecting the data
# =============================================================================


def describe_system(spec, system, study_slot: str = "") -> tuple[list[dict], str]:
    """Parameters of the built system as a table, and the same description as text.

    Parameters
    ----------
    spec: SystemSpec
        The system specification (nominal power).
    system: SpurHelicalGearSystem
        The built system, after ``resolve``. Everything is read from it (``axisforge.outputs``),
        so the description always matches the model that was solved.
    study_slot: str
        Arrangement of the bearing under study, to label "study" / "partner".

    Returns
    -------
    rows: list of dict
        {section, parameter, value, unit}, for system.csv: the records of AxisForge for the
        gear system, every shaft, gear, bearing and load, followed by the role of each bearing
        in this study and the materials passed to the build.
    text: str
        The role of each bearing followed by the short description of AxisForge
        (``system_text``), for report.txt. The materials passed to the build are in ``rows``
        (system.csv) only.
    """
    rows = af_system.system_records(system, power_W=spec.power_W)

    study = []
    first = system.shafts[0]
    for bearing in first.bearings:
        role = "under study" if bearing.arrangement == study_slot else "partner"
        study.append(af_format.kv_record("study", f"bearing {bearing.label} ({bearing.arrangement})",
                                         role))
    for role, data in (("shaft", sysmod.SHAFT_MATERIAL), ("bearing", sysmod.BEARING_MATERIAL),
                       ("gear", sysmod.GEAR_MATERIAL)):
        section = "materials passed to the build"
        study.append(af_format.kv_record(section, f"{role}", data["material_id"]))
        study.append(af_format.kv_record(section, f"{role} E", data["E"], "MPa"))
        study.append(af_format.kv_record(section, f"{role} poisson ratio", data["poisson"]))

    roles = [r for r in study if r["section"] == "study"]
    text = (af_format.format_key_values(roles) + "\n\n"
            + af_system.system_text(system, power_W=spec.power_W))
    return rows + study, text


def fem_node_rows(power_W: float, fem: dict) -> list[dict]:
    """Bearing-node results of the shaft FEM at one power.

    Every field of the AxisForge bearing node is exported (``axisforge.outputs``), with the unit
    in the key: N, mm, N mm, rad.

    Parameters
    ----------
    power_W: float
        Power of this FEM solve [W].
    fem: dict
        {shaft name: ShaftResults}.

    Returns
    -------
    rows: list of dict
        One per (shaft, bearing node): ``power_W`` followed by the node record.
    """
    rows = []
    for shaft_name, results in fem.items():
        for record in af_shaft_results.bearing_node_records(results, shaft_name):
            rows.append(dict(power_W=power_W, **record))
    return rows


def fem_summary_rows(power_W: float, fem: dict) -> list[dict]:
    """Maxima of the shaft FEM result at one power.

    Parameters
    ----------
    power_W: float
        Power of this FEM solve [W].
    fem: dict
        {shaft name: ShaftResults}.

    Returns
    -------
    rows: list of dict
        One per shaft: ``power_W`` followed by ``axisforge.outputs`` ``shaft_summary_record``
        (maximum moment, deflection, stresses and torsion angle with their positions).
    """
    return [dict(power_W=power_W, **af_shaft_results.shaft_summary_record(results, name))
            for name, results in fem.items()]


def fem_mesh_rows(power_W: float, fem: dict) -> list[dict]:
    """Values of the shaft FEM at every mesh node.

    Parameters
    ----------
    power_W: float
        Power of this FEM solve [W].
    fem: dict
        {shaft name: ShaftResults}.

    Returns
    -------
    rows: list of dict
        One per (shaft, mesh node): ``power_W`` followed by ``axisforge.outputs``
        ``mesh_node_records`` (position, diameter, moment, shear force, torque, deflection,
        rotations, twist and stresses).
    """
    return [dict(power_W=power_W, **record)
            for name, results in fem.items()
            for record in af_shaft_results.mesh_node_records(results, name)]


def union_fields(rows: list[dict]) -> list[str]:
    """Every key of a list of dicts, in order of first appearance."""
    return af_format.union_fields(rows)


# =============================================================================
# Checks
# =============================================================================


def check_lines(rows: list[dict], axis_names, spur: bool, equilibrium_rtol: float = 1e-6,
                symmetry_rtol: float = 1e-6) -> list[str]:
    """Convergence, equilibrium, symmetry between shafts and errors, as report lines.

    Parameters
    ----------
    rows: list of dict
        Rows of one bearing type.
    axis_names: sequence of str
        Names of the swept axes.
    spur: bool
        True with spur gears: both shafts must then give the same Q_max (with helical gears
        Fa changes sign between the shafts and the check does not apply).
    equilibrium_rtol, symmetry_rtol: float
        Tolerances.

    Returns
    -------
    lines: list of str
        Ready to print or to write.
    """
    bad = [r for r in rows
           if not r["ok"] or not r["equil_err_rel"] <= equilibrium_rtol
           or (r["error"] and not r["postprocessed"])]
    lines = [f"convergence / equilibrium (|Fr - sum Fr_row| / Fr <= {equilibrium_rtol:g}) / "
             f"errors: {len(rows) - len(bad)}/{len(rows)} clean"]
    for r in bad:
        point = ", ".join(f"{a} = {r[a]:g}" for a in axis_names)
        lines.append(f"    CHECK {r['label']} ({point}): ok = {r['ok']}, "
                     f"equilibrium error = {r['equil_err_rel']:.2e}  {r['error']}")
    if spur:
        groups: dict[tuple, list[float]] = {}
        for r in rows:
            if r["ok"]:
                groups.setdefault(tuple(r[a] for a in axis_names), []).append(r["Q_max_N"])
        worst = 0.0
        for q in groups.values():
            if len(q) > 1 and max(q) > 0.0:
                worst = max(worst, (max(q) - min(q)) / max(q))
        flag = "OK" if worst <= symmetry_rtol else "CHECK"
        lines.append(f"largest Q_max spread between the two shafts (spur gears): {worst:.2e}  "
                     f"[{flag}]")
    else:
        lines.append("symmetry between shafts not checked: helical gears give Fa of opposite "
                     "sign on the two shafts")
    failed_post = [r for r in rows if r["error"] and r["postprocessed"] is False and r["ok"]]
    if failed_post:
        lines.append(f"{len(failed_post)} row(s) without stiffness / life (post-processing "
                     "failed, load distribution kept)")
    return lines


# =============================================================================
# The report
# =============================================================================


def _rule(char="=") -> str:
    return char * WIDTH


def _title(text: str) -> list[str]:
    return ["", _rule(), text, _rule()]


ISO_COLUMNS = [
    ("shaft", "shaft", ""), ("power_W", "P [W]", ".1f"), ("Fr_N", "Fr [N]", ".2f"),
    ("Fa_N", "Fa [N]", ".2f"), ("psi_mrad", "psi [mrad]", ".4f"), ("s_mm", "s [mm]", ".4f"),
    ("alpha0_deg", "alpha0 [deg]", ".2f"), ("n_loaded", "n_loaded", ".0f"),
    ("zone_half_angle_deg", "zone [deg]", ".1f"), ("Q_max_N", "Q_max [N]", ".2f"),
    ("Q_max_over_Fr_per_Z", "Q_max/(Fr/Z)", ".4f"), ("alpha_max_deg", "alpha_max [deg]", ".2f"),
    # mm -> um and N/mm -> N/um for reading (the optional fourth item is a scale factor)
    ("delta_r_mm", "delta_r [um]", ".3f", 1e3), ("delta_a_mm", "delta_a [um]", ".3f", 1e3),
    ("Kr_N_per_mm", "Kr [N/um]", ".2f", 1e-3), ("lamina_peak_over_mean", "q_peak/q_mean", ".3f"),
    ("L10r_Mrev", "L10r [1e6 rev]", ".4g"), ("ok", "ok", ""),
]


def _has_values(rows, key):
    return any(not (isinstance(r.get(key), float) and math.isnan(r[key])) for r in rows)


def build_report(data, axis_names, type_name: str) -> str:
    """The text of report.txt.

    Parameters
    ----------
    data: StudyData
        Results of one bearing type.
    axis_names: sequence of str
        Names of the swept axes.
    type_name: str
        Name of the type folder.

    Returns
    -------
    text: str
        The whole report.
    """
    meta = data.meta
    lines = [f"AxisForge design studies - bearings / load_distribution / {meta['question']} / "
             f"{type_name}",
             f"Generated {meta.get('created_utc', '')}"]

    # 1 run information (machine-readable block)
    lines += ["", _rule(), RUN_HEADER, RUN_NOTE, _rule()]
    for key, value in meta.items():
        lines.append(f"  {key} = {json.dumps(value, sort_keys=True)}")
    lines.append("")

    # 2 construction
    lines += _title("2  CONSTRUCTION - what was built")
    lines.append(data.system_text if data.system_text
                 else af_format.format_key_values(data.system_rows))

    # 3 solver: shaft FEM
    lines += _title("3  SOLVER - SHAFT FEM (rigid supports)")
    lines += ["Bearing nodes: reactions and slopes. One solve per power.", ""]
    lines.append(af_shaft_results.bearing_nodes_table_text(data.fem_rows)
                 if data.fem_rows else "(no bearing-node data)")
    if data.fem_summary_rows:
        lines += ["", "Maxima along each shaft (units assumed N, mm, MPa):", "",
                  af_shaft_results.shaft_summary_table_text(data.fem_summary_rows)]
    if data.fem_mesh_rows:
        nominal = meta.get("power_W", data.fem_mesh_rows[0]["power_W"])
        at_nominal = [r for r in data.fem_mesh_rows if abs(r["power_W"] - nominal) <= 1e-9 * max(
            1.0, abs(nominal))] or data.fem_mesh_rows
        for shaft in dict.fromkeys(r["shaft"] for r in at_nominal):
            rows = [r for r in at_nominal if r["shaft"] == shaft]
            lines += ["", f"Along {shaft} at P = {rows[0]['power_W']:.1f} W, one line per mesh "
                          "node (all powers in fem_mesh_nodes.csv):", "",
                      af_shaft_results.mesh_nodes_table_text(rows)]

    # 4 ISO/TS 16281
    lines += _title("4  SOLVER - ISO/TS 16281 LOAD DISTRIBUTION (bearing under study)")
    lines.append("Displacements in um and stiffness in N/um in this table (mm and N/mm in the CSV).")
    columns = [(a, a, ".6g") for a in axis_names]
    columns += [c for c in ISO_COLUMNS if c[0] not in axis_names and _has_values(data.rows, c[0])]
    lines += ["", af_format.format_table(columns, data.rows)]

    if data.q_rows:
        shaft = data.rows[0]["shaft"]
        lines += ["", f"Element loads Q_j [N] of {shaft} (phi_j = 0 on the load line), "
                      "one column per point of the sweep:"]
        points = [r for r in data.rows if r["shaft"] == shaft]
        keyed = {}
        for q in data.q_rows:
            if q["shaft"] == shaft:
                keyed.setdefault(tuple(q[a] for a in axis_names), []).append(q)
        first = keyed.get(tuple(points[0][a] for a in axis_names), [])
        table_rows = []
        for i, q0 in enumerate(first):
            row = dict(j=q0["j"], phi=q0["phi_deg"])
            for p in points:
                elements = keyed.get(tuple(p[a] for a in axis_names), [])
                row[_point_label(p, axis_names)] = elements[i]["Q_N"] if i < len(elements) else math.nan
            table_rows.append(row)
        cols = [("j", "j", ".0f"), ("phi", "phi [deg]", ".1f")]
        cols += [(_point_label(p, axis_names), _point_label(p, axis_names), ".2f") for p in points]
        lines.append(af_format.format_table(cols, table_rows))

    # 5 checks
    lines += _title("5  CHECKS")
    lines += check_lines(data.rows, axis_names, spur=meta.get("helix_angle_deg", 0.0) == 0.0)
    lines.append("")
    return "\n".join(lines)


def _point_label(point: dict, axis_names) -> str:
    return " ".join(f"{a}={point[a]:g}" for a in axis_names)


def read_run_information(text: str) -> dict:
    """The RUN INFORMATION block of a report.

    Parameters
    ----------
    text: str
        Contents of report.txt.

    Returns
    -------
    meta: dict
        {key: value} as written by ``build_report``.

    Raises
    ------
    ValueError
        If the block is missing.
    """
    lines = text.splitlines()
    try:
        start = lines.index(RUN_HEADER)
    except ValueError:
        raise ValueError("report.txt has no RUN INFORMATION block") from None
    meta = {}
    for line in lines[start + 3:]:
        if not line.strip():
            break
        key, _, value = line.strip().partition(" = ")
        meta[key] = json.loads(value)
    return meta


def utc_now() -> str:
    """Current UTC time, ISO 8601 to the second."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")