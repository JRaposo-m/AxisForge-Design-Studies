"""
Text reports and tables of one bearing type: ``<question>/<type>/results``.

Files
-----
report.txt              the readable report, in five sections:
                          1 RUN INFORMATION   what produced the results (also read by compare
                                              and by the regression tests: do not edit)
                          2 SYSTEM            gear stage, shafts, materials, bearings, loads
                          3 SHAFT FEM         bearing-node results of the rigid-support FEM
                          4 ISO/TS 16281      one line per point and shaft, then the element
                                              loads of the reference shaft
                          5 CHECKS            convergence, equilibrium, symmetry, errors
system.csv              section, parameter, value, unit
fem_bearing_nodes.csv   one row per (power, shaft, bearing node), every numeric node field
iso16281_rows.csv       one row per (point, shaft): see rows.py
iso16281_elements.csv   rolling-element loads Q_j
iso16281_laminae.csv    lamina loads of the most loaded roller (line contact only)

The CSV files are for Excel or any other tool; the report is for reading and for writing up.
"""
from __future__ import annotations

import dataclasses
import json
import math
from datetime import datetime, timezone

from . import system as sysmod

RUN_HEADER = "1  RUN INFORMATION"
RUN_NOTE = "(read by compare/ and tests/: do not edit)"
WIDTH = 100


# =============================================================================
# Collecting the data
# =============================================================================


def describe_system(spec, system=None, study_slot: str = "") -> tuple[list[dict], list[str]]:
    """Parameters of the system as a table, and the AxisForge view of it as text.

    Parameters
    ----------
    spec: SystemSpec
        The system specification (nominal power).
    system: SpurHelicalGearSystem, optional (None)
        The built system; when given, the shaft loads after ``resolve`` and the bearing
        summaries are added to the text.
    study_slot: str
        Arrangement of the bearing under study, to label "study" / "partner".

    Returns
    -------
    rows: list of dict
        {section, parameter, value, unit}, for system.csv and the report.
    extra_lines: list of str
        Shaft loads and bearing summaries as printed by AxisForge.
    """
    rows: list[dict] = []

    def add(section, parameter, value, unit=""):
        rows.append(dict(section=section, parameter=parameter, value=value, unit=unit))

    rpm1 = sysmod.SHAFT_RPM[sysmod.SHAFT_NAMES[0]]
    torque = spec.power_W / (2.0 * math.pi * rpm1 / 60.0)
    add("gear stage", "power", spec.power_W, "W")
    add("gear stage", "driving shaft torque", torque, "N m")
    for name, rpm in sysmod.SHAFT_RPM.items():
        add("gear stage", f"speed {name}", rpm, "rpm")
    add("gear stage", "normal module", sysmod.GEAR_MODULE_MM, "mm")
    add("gear stage", "teeth driver / driven", f"{sysmod.GEAR_Z_DRIVER} / {sysmod.GEAR_Z_DRIVEN}")
    add("gear stage", "face width", sysmod.GEAR_FACE_WIDTH_MM, "mm")
    add("gear stage", "helix angle", spec.helix_angle_deg, "deg")
    add("gear stage", "gear position", sysmod.GEAR_POSITION_MM, "mm")
    add("gear stage", "line of centres angle", sysmod.MESH_PHI_DEG, "deg")

    for i, (length, diameter, label) in enumerate(sysmod.SECTIONS, start=1):
        add("shafts (both identical)", f"section {i} {label}", f"L = {length:g}, d = {diameter:g}",
            "mm")
    add("shafts (both identical)", "shoulder fillet radius", sysmod.SHOULDER_FILLET_MM, "mm")
    add("shafts (both identical)", "total length", sum(s[0] for s in sysmod.SECTIONS), "mm")

    for role, data in (("shaft", sysmod.SHAFT_MATERIAL), ("bearing", sysmod.BEARING_MATERIAL),
                       ("gear", sysmod.GEAR_MATERIAL)):
        add("materials", f"{role}", data["material_id"])
        add("materials", f"{role} E", data["E"], "MPa")
        add("materials", f"{role} poisson ratio", data["poisson"])

    for slot, bearing in (("locating", spec.locating), ("non-locating", spec.non_locating)):
        n, position = sysmod.BEARING_SLOTS[slot]
        role = "under study" if slot == study_slot else "partner"
        section = f"bearing {n} ({slot}, {role})"
        add(section, "type", type(bearing).__name__)
        add(section, "position", position, "mm")
        for f in dataclasses.fields(bearing):
            add(section, f.name, getattr(bearing, f.name), _unit(f.name))

    extra: list[str] = []
    if system is not None:
        for ss in getattr(system, "shafts", []):
            extra.append(f"Loads on {ss.name} after system.resolve:")
            for load in getattr(ss, "loads", []):
                extra.append(f"    {load}")
        first = getattr(system, "shafts", [None])[0]
        for bearing in getattr(first, "bearings", []) if first is not None else []:
            try:
                extra.append(f"Bearing {bearing.label} (AxisForge summary):")
                extra.extend(f"    {line}" for line in str(bearing.summary()).splitlines())
            except Exception as exc:                                   # noqa: BLE001
                extra.append(f"    summary not available: {type(exc).__name__}: {exc}")
    return rows, extra


def _unit(name: str) -> str:
    if name.endswith("_mm") or name in ("s", "Dw", "Dwe", "Lwe", "Dpw"):
        return "mm"
    if name.endswith("_deg"):
        return "deg"
    return ""


def fem_node_rows(power_W: float, fem: dict) -> list[dict]:
    """Bearing-node results of the shaft FEM at one power.

    Every scalar field of the AxisForge bearing node is exported as it is (AxisForge units:
    N, mm, N mm, rad).

    Parameters
    ----------
    power_W: float
        Power of this FEM solve [W].
    fem: dict
        {shaft name: ShaftResults}.

    Returns
    -------
    rows: list of dict
        One per (shaft, bearing node).
    """
    rows = []
    for shaft_name, results in fem.items():
        for node in getattr(results, "bearing_nodes", []) or []:
            row = dict(power_W=power_W, shaft=shaft_name)
            names = ([f.name for f in dataclasses.fields(node)] if dataclasses.is_dataclass(node)
                     else list(vars(node)))
            for name in names:
                value = getattr(node, name, None)
                if isinstance(value, (bool, int, float, str)):
                    row[name] = value
            rows.append(row)
    return rows


def union_fields(rows: list[dict]) -> list[str]:
    """Every key of a list of dicts, in order of first appearance."""
    fields: dict[str, None] = {}
    for r in rows:
        fields.update(dict.fromkeys(r))
    return list(fields)


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


def _fmt(value, spec=".6g") -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return "-" if math.isnan(value) else format(value, spec)
    return str(value)


def _table(columns: list[tuple[str, str, str]], rows: list[dict]) -> list[str]:
    """Fixed-width table: columns are (key, header, format)."""
    cells = [[_fmt(r.get(key, math.nan), fmt) for key, _, fmt in columns] for r in rows]
    widths = [max([len(h)] + [len(c[i]) for c in cells]) for i, (_, h, _) in enumerate(columns)]
    out = ["  ".join(h.rjust(w) for (_, h, _), w in zip(columns, widths)),
           "  ".join("-" * w for w in widths)]
    out += ["  ".join(c.rjust(w) for c, w in zip(row, widths)) for row in cells]
    return out


ISO_COLUMNS = [
    ("shaft", "shaft", ""), ("power_W", "P [W]", ".1f"), ("Fr_N", "Fr [N]", ".2f"),
    ("Fa_N", "Fa [N]", ".2f"), ("psi_mrad", "psi [mrad]", ".4f"), ("s_mm", "s [mm]", ".4f"),
    ("alpha0_deg", "alpha0 [deg]", ".2f"), ("n_loaded", "n_loaded", ".0f"),
    ("zone_half_angle_deg", "zone [deg]", ".1f"), ("Q_max_N", "Q_max [N]", ".2f"),
    ("Q_max_over_Fr_per_Z", "Q_max/(Fr/Z)", ".4f"), ("alpha_max_deg", "alpha_max [deg]", ".2f"),
    ("delta_r_mm", "delta_r [um]", ""), ("delta_a_mm", "delta_a [um]", ""),
    ("Kr_N_per_mm", "Kr [N/um]", ""), ("lamina_peak_over_mean", "q_peak/q_mean", ".3f"),
    ("L10r_Mrev", "L10r [1e6 rev]", ".4g"), ("ok", "ok", ""),
]


def _iso_table_rows(rows):
    """Copies of the rows with displacements in um and stiffness in N/um for the report."""
    out = []
    for r in rows:
        r = dict(r)
        for key in ("delta_r_mm", "delta_a_mm"):
            r[key] = _fmt(1e3 * r[key], ".3f") if isinstance(r[key], float) else r[key]
        r["Kr_N_per_mm"] = (_fmt(1e-3 * r["Kr_N_per_mm"], ".2f")
                            if isinstance(r["Kr_N_per_mm"], float) else r["Kr_N_per_mm"])
        out.append(r)
    return out


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

    # 2 system
    lines += _title("2  SYSTEM")
    section = None
    for r in data.system_rows:
        if r["section"] != section:
            section = r["section"]
            lines += ["", f"{section}:"]
        lines.append(f"    {r['parameter']:<34} {_fmt(r['value']):>18}  {r['unit']}")
    if data.system_text:
        lines += [""] + data.system_text

    # 3 FEM
    lines += _title("3  SHAFT FEM (rigid supports) - bearing nodes")
    lines.append("Units as returned by AxisForge: N, mm, N mm, rad. One solve per power.")
    if not data.fem_rows:
        lines.append("(no bearing-node data)")
    for r in data.fem_rows:
        head = f"P = {r['power_W']:.4f} W | {r['shaft']}"
        label = r.get("label") or r.get("bearing_label") or ""
        lines += ["", f"{head} | node {label}".rstrip()]
        for key, value in r.items():
            if key in ("power_W", "shaft"):
                continue
            lines.append(f"    {key:<24} {_fmt(value, '.8g')}")

    # 4 ISO/TS 16281
    lines += _title("4  ISO/TS 16281 LOAD DISTRIBUTION - bearing under study")
    lines.append("Displacements in um and stiffness in N/um in this table (mm and N/mm in the CSV).")
    columns = [(a, a, ".6g") for a in axis_names]
    columns += [c for c in ISO_COLUMNS if c[0] not in axis_names and _has_values(data.rows, c[0])]
    lines += [""] + _table(columns, _iso_table_rows(data.rows))

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
        lines += _table(cols, table_rows)

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
