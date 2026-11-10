"""Comparator of v104: discretisation measures of one solved smooth point contact (README section 6).

:func:`check` takes a solved point contact with its ellipse on the grid lines (``contact.psi ==
grid.angle``) and returns ONE record of discretisation errors against r02 Hertz at the load the BEM
carries (through v101's ``reference`` and ``bem_load``): the bridge's measured errors (approach,
p0, a, b) and the measures of the two published grid studies, Polonsky and Keer (1999) ``eps_r``,
``eps_p`` and Muser (2014) contact area from a fit of the pressure profile.
"""
from __future__ import annotations

import math

import numpy as np

from studies.v100_dry_contact.v101_hertz_point import bem_load, pressure_l1_error, reference

__all__ = ["check", "judge", "TOL_DOMAIN"]

TOL_DOMAIN = 5e-4     # D1: change of delta and p0 against the largest domain, spatial kernel [-]


def check(result, *, case: str = "v104") -> dict:
    """Discretisation record of one solved point contact against r02 Hertz.

    Fields (errors relative, ``(BEM - Hertz)/Hertz``):

    * grid: ``kernel``, ``load_control``, ``h_m``, ``b_over_h``, ``a_over_h``, ``M = 2 b_H/h``,
      ``span_x``, ``span_y`` (shorter side in Hertz semi-axes), ``n_x``, ``n_y``, ``solve_time_s``;
    * ``err_delta``, ``err_p0``, ``err_a``, ``err_b``: the bridge's measured values (approach: the
      given one under interference control) against Hertz at the BEM load;
    * ``eps_r``: Polonsky and Keer's contact-size error, ``|sqrt(A/pi) - sqrt(a_H b_H)| / sqrt(a_H
      b_H)`` with ``A = N_c h^2`` the count of loaded nodes (their circle, read on an ellipse through
      the equivalent radius); ``eps_p``: their L1 pressure error over the loaded nodes,
      ``sum |p - p_H| / sum p_H`` (PDF p. 10; v101 ``pressure_l1_error``);
    * ``err_area_fit``: Muser's contact area, ``pi a b`` from a least-squares fit of
      ``p^2 = p0^2 (1 - x^2/b^2 - y^2/a^2)`` over the loaded nodes (his "fit of the radial pressure
      profile", PDF p. 7, written for the ellipse), against ``pi a_H b_H``.
    """
    c, g = result.contact, result.grid
    if c.line or not math.isclose(c.psi, g.angle, abs_tol=1e-12):
        raise ValueError("v104.check takes a point contact with its ellipse axes on the grid lines")
    ref = reference(c, bem_load(result))
    h, p = g.h, np.asarray(result.p)
    # OPEN R2-05: the bridge's numerical zero (solvers.static._EDGE, private) repeated here and in v101
    loaded = p > 1e-6 * result.p0
    X, Y = g.X[loaded], g.Y[loaded]                  # b along grid x, a along grid y
    r_H = math.sqrt(ref.a_m * ref.b_m)
    A, B, C = np.linalg.lstsq(np.column_stack([np.ones_like(X), X**2, Y**2]), p[loaded] ** 2, rcond=None)[0]
    # a fit that does not close an ellipse (B or C >= 0, only on a grid far too coarse) gives no area
    area_fit = math.pi * math.sqrt(A / -B) * math.sqrt(A / -C) if A > 0 and B < 0 and C < 0 else math.nan
    delta = result.delta if result.load_control == "normal_load" else result.interference
    span_x, span_y = g.spans(result.hertz)
    return {"case": case, "label": c.label, "kernel": "frequency" if result.use_frequency_domain else "spatial",
            "load_control": result.load_control, "h_m": h, "b_over_h": ref.b_m / h, "a_over_h": ref.a_m / h,
            "M": 2.0 * ref.b_m / h, "span_x": span_x, "span_y": span_y, "n_x": g.X.shape[0], "n_y": g.X.shape[1],
            "converged": result.converged, "solve_time_s": result.solve_time,
            "err_delta": delta / ref.delta_m - 1.0, "err_p0": result.p0 / ref.p0_Pa - 1.0,
            "err_a": result.a / ref.a_m - 1.0, "err_b": result.b / ref.b_m - 1.0,
            "eps_r": abs(math.sqrt(loaded.sum() * h**2 / math.pi) - r_H) / r_H,
            "eps_p": pressure_l1_error(result, ref),
            "err_area_fit": area_fit / (math.pi * ref.a_m * ref.b_m) - 1.0,
            "delta_m": delta, "delta_H_m": ref.delta_m, "p0_Pa": result.p0, "p0_H_Pa": ref.p0_Pa,
            "a_H_m": ref.a_m, "b_H_m": ref.b_m}


def judge(records: list[dict], *, case: str = "v104") -> list[dict]:
    """Criteria of the sweeps (README section 6), one schema: ``id, label, kernel, quantity, value,
    limit, verdict, M, span_x``.

    * D1 change of ``delta`` and ``p0`` against the largest domain of the span sweep, each system and
      kernel; judged on the spatial kernel (``TOL_DOMAIN``), RECORD on the frequency kernel;
    * L1 (Polonsky-Keer) ``eps_r < 1/M`` on every h-sweep grid of the circle, judged where ``M >= 64``
      (their range), RECORD below;
    * L2 (Polonsky-Keer) observed order of ``eps_p`` between the two finest grids, RECORD (they print
      "on the order of M^-3/2", with no constant);
    * L3 (Muser) at ``b/h >= 32`` asked: ``|err_area_fit| < 0.1 %``; judged on the frequency kernel at
      half-length ``2 a`` (period ``8 a``, his ``a_c/L = 1/8``), RECORD on the spatial kernel (no cell).
    """
    out = []

    def row(cid, label, kernel, quantity, value, limit, verdict, r):
        out.append({"case": case, "id": cid, "label": label, "kernel": kernel, "quantity": quantity, "value": value,
                    "limit": limit, "verdict": verdict, "M": r["M"], "span_x": r["span_x"]})

    for label in dict.fromkeys(r["label"] for r in records):
        for kernel in dict.fromkeys(r["kernel"] for r in records):
            mine = [r for r in records if r["label"] == label and r["kernel"] == kernel]
            spans = sorted((r for r in mine if r["sweep"] == "span"), key=lambda r: r["span_x"])
            for r in spans[:-1]:
                last = spans[-1]
                change = max(abs(r["delta_m"] / last["delta_m"] - 1.0), abs(r["p0_Pa"] / last["p0_Pa"] - 1.0))
                row("D1", label, kernel, f"delta, p0 change against span {last['span_x']:.2f}", change, TOL_DOMAIN,
                    ("PASS" if change < TOL_DOMAIN else "MISS") if kernel == "spatial" else "RECORD", r)
            if label != "ball_on_flat":
                continue
            sweep = sorted((r for r in mine if r["sweep"] == "h"), key=lambda r: r["M"])
            for r in sweep:
                row("L1", label, kernel, "eps_r", r["eps_r"], 1.0 / r["M"],
                    ("PASS" if r["eps_r"] < 1.0 / r["M"] else "MISS") if r["M"] >= 64.0 else "RECORD", r)
            f, g = sweep[-2], sweep[-1]
            row("L2", label, kernel, "order of eps_p (two finest grids)",
                math.log(f["eps_p"] / g["eps_p"]) / math.log(g["M"] / f["M"]), 1.5, "RECORD", g)
            for r in (r for r in mine if r["sweep"] == "muser"):
                verdict = "PASS" if abs(r["err_area_fit"]) < 0.001 else "MISS"
                row("L3", label, kernel, "err_area_fit", r["err_area_fit"], 0.001,
                    verdict if kernel == "frequency" else "RECORD", r)
    return out
