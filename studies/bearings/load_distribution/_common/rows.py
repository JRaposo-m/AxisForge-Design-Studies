"""
The result tables of a load distribution study.

One schema for every bearing type and every question: the swept axes come first (named after
the axis), followed by RESULT_FIELDS. A column that does not apply to a type is NaN (for
example the lamina columns of a ball bearing), so the data of different types can be read
and compared by the same code.

Tables
------
rows      one row per (point of the sweep, shaft): the bearing under study
q_rows    one row per rolling element: angle phi_j from the load line, Q_j, alpha_j
lam_rows  line-contact bearings only: lamina loads q_k of the most loaded roller

Conventions
-----------
power_W      power transmitted by the gear stage [W] (100 N m at 1000 rpm = 10 471.9755 W);
             the load is set through the construction, never scaled after the solve
s_mm         total radial internal clearance of the bearing [mm]; for the angular contact
             bearing it is the clearance implied by alpha_0
alpha0_deg   free contact angle of the bearing [deg]
Fr / C       radial load over the dynamic rating bearing.C
psi          inner-ring tilt on the Fr plane [mrad]; either the FEM slope or prescribed
Q_j          rolling-element contact force [N] (ball, or sum of the roller laminae);
             phi_j = 0 is the element on the load line
alpha_max    operating contact angle of the most loaded ball [deg]
delta_r/_a   radial / axial approach of the inner ring [mm], from the concentric position
             (it includes the s/2 needed to close the clearance)
Kr           secant radial stiffness Fr / delta_r [N/mm] of the post-processing
Kr_tangent   dFr / d delta_r along the load sweep [N/mm]   (only with a power_W axis)
m_defl       d ln delta_r / d ln Fr                         (only with a power_W axis)
p_life       - d ln L10r / d ln Fr                          (only with a power_W axis)
L10r         ISO/TS 16281 basic reference rating life [10^6 rev], NOT an ISO 281 life;
             L10h uses the shaft speed (inner ring rotating)
"""
from __future__ import annotations

import math

import numpy as np

NAN = math.nan

RESULT_FIELDS = [
    # identification
    "shaft", "label", "kind", "position_mm", "arrangement", "Z", "C_N", "s_mm", "alpha0_deg",
    # operating point of the gear stage
    "power_W",
    # loads and displacements
    "Fr_N", "Fa_N", "Fa_over_Fr", "Fr_over_C", "psi_mrad", "delta_r_mm", "delta_a_mm", "Mz_Nmm",
    # stiffness and exponents
    "Kr_N_per_mm", "Kr_xz_N_per_mm", "Kr_xy_N_per_mm", "Kr_tangent_N_per_mm", "m_defl", "p_life",
    # load distribution
    "n_loaded", "zone_half_angle_deg", "alpha_max_deg", "Q_max_N", "Q_max_over_Fr_per_Z",
    # roller laminae (line contact only)
    "q_lamina_max_N", "lamina_loaded_frac", "lamina_peak_over_mean",
    # solver and life
    "equil_err_rel", "ok", "residual", "n_iter", "L10r_Mrev", "L10h_h", "postprocessed", "error",
]
TEXT_FIELDS = {"shaft", "label", "kind", "arrangement", "error"}
BOOL_FIELDS = {"ok", "postprocessed"}
Q_RESULT_FIELDS = ["shaft", "label", "kind", "j", "phi_deg", "Q_N", "alpha_j_deg"]
LAMINA_RESULT_FIELDS = ["shaft", "label", "kind", "roller_j", "k", "x_mm", "q_N"]


def _with_axes(axis_names, fields):
    return list(axis_names) + [f for f in fields if f not in axis_names]


def row_fields(axis_names) -> list[str]:
    """Columns of ``rows``.

    Parameters
    ----------
    axis_names: sequence of str
        Names of the swept axes, in sweep order.

    Returns
    -------
    fields: list of str
        Axis columns followed by RESULT_FIELDS (an axis that is also a result column, such as
        psi_mrad, appears once, in the axis position).
    """
    return _with_axes(axis_names, RESULT_FIELDS)


def q_fields(axis_names) -> list[str]:
    """Columns of ``q_rows`` (see ``row_fields``)."""
    return _with_axes(axis_names, Q_RESULT_FIELDS)


def lamina_fields(axis_names) -> list[str]:
    """Columns of ``lam_rows`` (see ``row_fields``)."""
    return _with_axes(axis_names, LAMINA_RESULT_FIELDS)


def _finite(x):
    return x if math.isfinite(x) else NAN


def make_row(point: dict, shaft_name: str, rpm: float, bearing, kind: str, analysis,
             error: str, postprocessed: bool):
    """Read one solver result into a row and the lists of its element and lamina loads.

    Parameters
    ----------
    point: dict
        {axis name: value} of this point of the sweep.
    shaft_name: str
        Shaft that carries the bearing.
    rpm: float
        Shaft speed [rpm], for the life in hours.
    bearing: Bearing
        The bearing that was solved.
    kind: str
        BearingSpec.kind of the bearing.
    analysis: BearingAnalysisResult or None
        From ``solve.solve_bearing``.
    error: str
        From ``solve.solve_bearing``.
    postprocessed: bool
        From ``solve.solve_bearing``.

    Returns
    -------
    row: dict
        Every column of ``row_fields``; identification only if ``analysis`` is None.
    q_rows: list of dict
        One per rolling element.
    lam_rows: list of dict
        Lamina loads of the most loaded roller; empty for point contact.
    """
    axis_names = list(point)
    row = {f: NAN for f in row_fields(axis_names)}
    row.update(shaft=shaft_name, label=bearing.label, kind=kind, position_mm=bearing.position,
               arrangement=bearing.arrangement, Z=bearing.Z, C_N=bearing.C,
               s_mm=float(getattr(bearing, "s", NAN)),
               alpha0_deg=math.degrees(getattr(bearing, "alpha_0", NAN)),
               ok=False, postprocessed=postprocessed, error=error)
    row.update(point)
    if analysis is None:
        return row, [], []

    ld = analysis.load_distribution
    ring = ld.row                                     # single-row bearing
    Q = np.asarray(ring.Q_j, dtype=float)
    phi = (np.asarray(ring.phi_j, dtype=float) + math.pi) % (2.0 * math.pi) - math.pi
    alpha_j = getattr(ring, "alpha_j", None)
    alpha_j = None if alpha_j is None else np.degrees(np.asarray(alpha_j, dtype=float))
    loaded = Q > 0.0
    Fr = ld.Fr
    dFr, _ = ld.equilibrium_error
    j_max = int(np.argmax(Q))

    row.update(
        Fr_N=Fr, Fa_N=ld.Fa,
        Fa_over_Fr=ld.Fa / Fr if Fr > 0.0 else NAN,
        Fr_over_C=Fr / bearing.C,
        psi_mrad=1e3 * ld.psi,
        delta_r_mm=ld.delta_r, delta_a_mm=ld.delta_a, Mz_Nmm=ld.Mz,
        n_loaded=ring.n_loaded,
        zone_half_angle_deg=math.degrees(np.max(np.abs(phi[loaded]))) if loaded.any() else NAN,
        alpha_max_deg=float(alpha_j[j_max]) if alpha_j is not None else NAN,
        Q_max_N=float(Q.max()),
        Q_max_over_Fr_per_Z=float(Q.max()) / (Fr / bearing.Z) if Fr > 0.0 else NAN,
        equil_err_rel=abs(dFr) / Fr if Fr > 0.0 else NAN,
        ok=bool(ld.ok), residual=ld.residual, n_iter=ld.n_iter,
    )

    st = ld.stiffness                                 # only with postprocess=True
    if st is not None:
        # Kr_xz = Fr_xz / delta_r_xz and Kr_xy = Fr_xy / delta_r_xy both equal Fr / delta_r; a
        # plane with ~0 displacement gives inf (known TODO of the post-processing), so Kr takes
        # the plane with the larger displacement.
        larger = st.Kr_xz if abs(st.delta_r_xz) >= abs(st.delta_r_xy) else st.Kr_xy
        row.update(Kr_N_per_mm=_finite(larger), Kr_xz_N_per_mm=_finite(st.Kr_xz),
                   Kr_xy_N_per_mm=_finite(st.Kr_xy))
    life = analysis.basic_life
    if life is not None:
        row.update(L10r_Mrev=life.L10r, L10h_h=life.L10r * 1e6 / (60.0 * rpm))

    ident = dict(point, shaft=shaft_name, label=bearing.label, kind=kind)
    q_rows = [dict(ident, j=j, phi_deg=math.degrees(phi[j]), Q_N=float(Q[j]),
                   alpha_j_deg=float(alpha_j[j]) if alpha_j is not None else NAN)
              for j in range(len(Q))]

    lam_rows = []
    if getattr(ld, "is_line_contact", False):         # lamina model: inside the most loaded roller
        q_k = np.asarray(ring.q_jk[j_max], dtype=float)
        positive = q_k > 0.0
        row.update(q_lamina_max_N=float(q_k.max()),
                   lamina_loaded_frac=float(positive.mean()),
                   lamina_peak_over_mean=(float(q_k.max() / q_k[positive].mean())
                                          if positive.any() else NAN))
        lam_rows = [dict(ident, roller_j=j_max, k=k, x_mm=float(ring.x_k[k]), q_N=float(q_k[k]))
                    for k in range(len(q_k))]
    # the axis values win over the values read back from the solver (for example a prescribed
    # psi_mrad is reported as prescribed, as in the original misalignment study)
    row.update(point)
    return row, q_rows, lam_rows


def add_derived(rows: list[dict], axis_names, load_axis: str = "power_W") -> None:
    """Fill Kr_tangent, m_defl and p_life by finite differences along the load axis.

    One curve per (shaft, label, values of the other axes); it needs >= 3 converged load
    points. The derivatives use np.gradient, so the end points are one-sided. Nothing is done
    if there is no load axis.

    Parameters
    ----------
    rows: list of dict
        From ``make_row``; modified in place.
    axis_names: sequence of str
        Names of the swept axes.
    load_axis: str
        Name of the load axis.
    """
    if load_axis not in axis_names:
        return
    others = [a for a in axis_names if a != load_axis]
    curves: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r["shaft"], r["label"], *(r[a] for a in others))
        curves.setdefault(key, []).append(r)
    for points in curves.values():
        points.sort(key=lambda r: r[load_axis])
        good = [r for r in points if r["ok"] and r["Fr_N"] > 0.0 and r["delta_r_mm"] > 0.0]
        if len(good) < 3:
            continue
        F = np.array([r["Fr_N"] for r in good])
        d = np.array([r["delta_r_mm"] for r in good])
        for r, kt, m in zip(good, np.gradient(F, d), np.gradient(np.log(d), np.log(F))):
            r["Kr_tangent_N_per_mm"], r["m_defl"] = float(kt), float(m)
        with_life = [r for r in good if r["L10r_Mrev"] > 0.0]          # NaN fails the test
        if len(with_life) >= 3:
            p = -np.gradient(np.log([r["L10r_Mrev"] for r in with_life]),
                             np.log([r["Fr_N"] for r in with_life]))
            for r, p_i in zip(with_life, p):
                r["p_life"] = float(p_i)