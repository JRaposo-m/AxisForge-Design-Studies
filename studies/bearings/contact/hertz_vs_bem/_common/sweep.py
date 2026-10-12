"""The mesh and domain sweep shared by the three scripts, and the criteria on its records.

Nothing is built, measured or compared with Hertz here: the grid, the surfaces, the static step,
the measurements, the Hertz reference and the boolean checks all come from
``slippy_bridge.contact``.
"""
from __future__ import annotations

import math

from slippy_bridge.contact import Grid, pair, static_step

from .case import SPAN_X_0

__all__ = ["sweep", "criteria", "RECORD_COLUMNS", "CRITERIA_COLUMNS"]

# Console tables (``axisforge.outputs._format.format_table``): key, header, format, scale.
RECORD_COLUMNS = (("build", "build", "s"), ("sweep", "sweep", "s"), ("span_x", "span_x [s_x]", "g"),
                  ("h_m", "h [um]", ".3f", 1e6), ("n_x", "n_x", ".0f"), ("n_y", "n_y", ".0f"),
                  ("p0_Pa", "p0 [MPa]", ".2f", 1e-6), ("err_p0", "err p0 [%]", "+.3f", 100.0),
                  ("err_s_x", "err s_x [%]", "+.3f", 100.0), ("err_s_y", "err s_y [%]", "+.3f", 100.0),
                  ("err_delta", "err delta [%]", "+.3f", 100.0), ("solve_time_s", "t [s]", ".2f"))
CRITERIA_COLUMNS = (("criterion", "criterion", "s"), ("build", "build", "s"), ("value", "value", "s"),
                    ("limit", "limit", "s"), ("result", "result", "s"))


def sweep(contact, builds, h_levels, span_x_levels, output_dir) -> list[dict]:
    """Solve one smooth static contact on a series of grids, for each build.

    Two sweeps around the default grid of the bridge (spacing ``h0`` = largest ``2**k`` not above
    ``s_x / 10``, half-length ``4 s_x`` along rolling):

    * ``"mesh"``: spacing ``f * h0`` for each ``f`` in ``h_levels``, half-length ``4 s_x``. Shows
      the discretisation error.
    * ``"domain"``: half-length ``span_x * s_x`` for each ``span_x`` in ``span_x_levels``, spacing ``h0``.
      Shows the dependence on the domain. A half-space BEM should have none once the contact is
      inside; SlipPY's default frequency-domain kernel repeats the load with period twice the
      domain, and this sweep measures that image error.

    Parameters
    ----------
    contact : Contact or ContactCase
        The contact (SI units, ``x`` = rolling direction).
    builds : sequence of {"composite", "two_profiles"}
        Surface builds of ``slippy_bridge.contact.pair``.
    h_levels : sequence of float
        Grid spacings in units of ``h0`` [-]; powers of two keep the spacing dyadic.
    span_x_levels : sequence of float
        Half-lengths of the domain along rolling in units of the Hertz semi-axis ``s_x`` [-].
    output_dir : path
        Folder for SlipPY's .sar/.sdb/.log files.

    Returns
    -------
    list of dict
        One record of ``static_step.run`` per solve (measurements, Hertz reference, relative
        errors as fractions, checks), plus ``build``, ``sweep`` (``"mesh"`` or ``"domain"``) and
        ``span_x`` [-]. The default grid appears once in each sweep.
    """
    case = getattr(contact, "case", contact)
    h0 = Grid.around(case).h
    grids = [("mesh", f * h0, SPAN_X_0) for f in h_levels] + [("domain", h0, span_x) for span_x in span_x_levels]
    records = []
    for build in builds:
        for i, (kind, h, span_x) in enumerate(grids):
            grid = Grid.around(case, h=h, span_x=span_x)
            record, _ = static_step.run(contact, pair(case, grid, build=build), grid, output_dir,
                                        name=f"{case.label}_{build}_{kind}_{i}")
            records.append({"build": build, "sweep": kind, "span_x": span_x, **record})
    return records


def criteria(records: list[dict]) -> list[dict]:
    """Evaluate the acceptance criteria of the study on the records of :func:`sweep`.

    The limits are those of the README, fixed before running. Per build, on the default grid
    (``h0``, half-length ``4 s_x``) unless said otherwise:

    1. the six checks of ``static_step.run`` are true;
    2. ``|err p0| <= 1 %``, ``|err s_x|`` and ``|err s_y| <= 3 %``, ``|err delta| <= 2 %``. A
       quantity without a finite Hertz value is skipped (line contact: ``s_y`` is infinite and the
       approach of two half-spaces in plane strain is not defined);
    3. the builds agree in ``p0`` within 1 %;
    4. domain sweep: ``|err p0|`` does not grow when the domain is extended;
    5. mesh sweep: ``|err p0| <= 0.5 %`` on the finest grid.

    Parameters
    ----------
    records : list of dict
        From :func:`sweep` (errors are fractions).

    Returns
    -------
    list of dict
        One record per criterion and build: ``criterion``, ``build``, ``value`` (text), ``limit``
        (text) and ``result`` (``"PASS"`` or ``"FAIL"``).
    """
    out = []

    def add(criterion, build, value, limit, ok):
        out.append({"criterion": criterion, "build": build, "value": value, "limit": limit,
                    "result": "PASS" if ok else "FAIL"})

    def pct(x):
        return f"{100.0 * x:+.3f} %"

    builds = list(dict.fromkeys(r["build"] for r in records))
    default = {}
    for build in builds:
        mesh = sorted((r for r in records if r["build"] == build and r["sweep"] == "mesh"),
                      key=lambda r: r["h_m"])
        domain = sorted((r for r in records if r["build"] == build and r["sweep"] == "domain"),
                        key=lambda r: r["span_x"])
        # the default grid is the mesh level whose spacing is the one of the domain sweep (h0)
        d = default[build] = next(r for r in mesh if r["h_m"] == domain[0]["h_m"])

        checks = ("converged", "load_ok", "edge_clear", "margin_ok", "resolution_ok", "within_reference")
        failed = [k for k in checks if not d[k]]
        add("1 six checks", build, "all true" if not failed else "false: " + " ".join(failed),
            "all true", not failed)
        for key, ref, limit in (("err_p0", "p0_ref_Pa", 0.01), ("err_s_x", "s_x_ref_m", 0.03),
                                ("err_s_y", "s_y_ref_m", 0.03), ("err_delta", "delta_ref_m", 0.02)):
            if math.isfinite(d[ref]):
                add(f"2 |{key}|", build, pct(d[key]), f"<= {100 * limit:g} %", abs(d[key]) <= limit)
        errs = [abs(r["err_p0"]) for r in domain]
        add("4 domain: |err_p0| not growing", build,
            "  ".join(f"{r['span_x']:g}: {pct(r['err_p0'])}" for r in domain), "non-increasing",
            all(b <= a for a, b in zip(errs, errs[1:])))
        add("5 finest mesh |err_p0|", build, pct(mesh[0]["err_p0"]), "<= 0.5 %",
            abs(mesh[0]["err_p0"]) <= 0.005)

    if len(builds) > 1:
        p0 = [default[b]["p0_Pa"] for b in builds]
        spread = (max(p0) - min(p0)) / min(p0)
        add("3 builds agree in p0", " vs ".join(builds), pct(spread), "<= 1 %", spread <= 0.01)
    return sorted(out, key=lambda r: r["criterion"])
