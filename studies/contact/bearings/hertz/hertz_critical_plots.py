"""
studies/contact/bearings/hertz/hertz_critical_plots.py

Plots the contact pressure distribution of only the MOST CRITICAL
(highest p0) element/race for each of the 4 bearings solved in
hertz_case_studies.py (same folder -- reuses its functions, doesn't
recompute the physics here).

Ball bearings (point contact): 2D elliptical pressure map, p(x,y),
via the 'pressure_f' closure hertz_full returns for an elliptical
contact.

Roller bearings (line contact):
  - 1D pressure profile p(x) across the contact half-width, via the
    'pressure_f' closure hertz_full returns for a line contact
    (COARSE case -- see hertz_case_studies.py docstring).
  - Subsurface stress profile on the z-axis below the pressure peak
    (x=0): sigma_x, sigma_z, tau_xz and the principal shear
    0.5*|sigma_z-sigma_x|, using a LOCAL, CORRECTED implementation of
    Johnson's closed-form line-contact stress field (Contact
    Mechanics, pg. 103, eq. 4.46/4.48) -- see _line_contact_stress()
    below for why this is not simply slippy's own 'stress_f'.

No displacement field is plotted for line contact: hertz_full does not
provide one for this contact shape (only for 'sphere'/'elliptical' --
the classical Hertz line-contact approach formula has a log-dependent,
not uniquely defined, total displacement without an external reference
length). Displacement is left for a future study on the elliptical
(ball-bearing) contacts, where it is well defined.

Output layout (organized by case study; point contact gets two PNGs per
bearing -- see plot_point_contact()'s docstring for the "_a" vs "_ab"
normalization trade-off):

    plots/
        point_contact/
            B1_dgb_a.png    B1_dgb_ab.png
            B3_ang_a.png    B3_ang_ab.png
        line_contact_coarse/
            pressure/
                B2_roll.png
                B4_roll.png
            stress/
                B2_roll.png
                B4_roll.png
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from hertz_case_studies import ball_case_study, roller_case_study, brg_dgb, brg_ang, brg_roll1, brg_roll2

# ---------------------------------------------------------------------------
# Report-style plot settings
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 10.5,
    "axes.edgecolor": "0.25",
    "axes.linewidth": 0.9,
    "grid.color": "0.85",
    "grid.linewidth": 0.6,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
})


# ---------------------------------------------------------------------------
# Local, corrected line-contact subsurface stress field.
#
# slippy.contact.hertz._stress_line_contact (the closure behind
# hertz_full's 'stress_f' key for line contact) has a confirmed bug: the
# m, n terms of Johnson's formula (Contact Mechanics, pg. 103, eq. 4.48a/b)
#
#     m^2 = 1/2 [ {(a^2-x^2+z^2)^2 + 4x^2z^2}^(1/2) + (a^2-x^2+z^2) ]
#     n^2 = 1/2 [ {(a^2-x^2+z^2)^2 + 4x^2z^2}^(1/2) - (a^2-x^2+z^2) ]
#
# are m^2/n^2 (note the ^2), but slippy assigns that expression directly
# to m/n without ever taking its square root. Verified two ways:
#   1) boundary condition sigma_z(x=0, z->0) must equal exactly -p0
#      (trivial -- it's the applied surface traction) -- slippy gives
#      -p0/25 instead;
#   2) matches Johnson eq. 4.46b (sigma_z on the z-axis, x=0) and
#      eq. 4.47 (max shear = 0.30*p0 at z=0.78a) to numerical precision
#      once the missing sqrt is added below; slippy's own values don't
#      match either reference and diverge with depth instead of decaying.
#
# SECOND bug, found after the first fix, off-axis only (x!=0): eq. 4.48a/b
# define TWO different quantities under the sqrt -- m^2 uses "+(a^2-x^2+z^2)"
# and n^2 uses "-(a^2-x^2+z^2)" (different signs, not the same number). The
# first pass here used the same m^2 expression for both m and n (i.e. wrote
# n = sign(x)*sqrt(m^2) instead of n = sign(x)*sqrt(n^2)). Both m and n^2 are
# always defined this way, but with the m^2 expression substituted for n^2,
# n does not go to zero continuously as x->0 at fixed z>0: sqrt(m^2) tends to
# a nonzero limit, so n (and tau_xz, and the Mohr's-circle magnitude derived
# from it) jumped discontinuously across x=0 instead of vanishing smoothly on
# the axis of symmetry, as it must by symmetry. That discontinuity is exactly
# what produced the earlier inconsistency (tau1 magnitude off-axis reading
# ~0.405*p0 immediately next to x=0, z=0.78a, against the correct on-axis
# value of 0.300*p0 -- eq. 4.47), and the "circle doesn't close" difference
# from Johnson's Fig. 4.5(b): the discontinuity broke the smoothness of the
# tau1 field near the symmetry axis. With n^2 computed from its own,
# correct expression (n^2 = 0.5*(d - s) below, not 0.5*(d + s)), tau_xz and
# tau1 are smooth and single-valued in x, and the field's global maximum is
# exactly 0.300*p0 at x=0, z=0.79a, matching eq. 4.47 with no near-surface
# artifact and no jump at the axis. Re-verified against both eq. 4.46b/4.47
# and continuity across x=0 after the fix.
#
# Implemented locally here (NOT patched into the installed slippy) so
# the case-study/report code stays untouched; only this plotting script
# uses it.
# ---------------------------------------------------------------------------
def _line_contact_stress(a, p0):
    """Closed-form subsurface stress field for a Hertzian line contact
    (Johnson, Contact Mechanics, pg. 103), corrected relative to
    slippy's '_stress_line_contact' (see module docstring above).
    """
    def stress(x, z):
        x, z = np.asarray(x, dtype=float), np.asarray(z, dtype=float)
        a2 = a ** 2
        s = a2 - x ** 2 + z ** 2
        d = np.sqrt(s ** 2 + 4 * x ** 2 * z ** 2)
        m2 = np.clip(0.5 * (d + s), 0.0, None)   # eq. 4.48a
        n2 = np.clip(0.5 * (d - s), 0.0, None)   # eq. 4.48b -- NOT the same as m2
        m = np.sign(z) * np.sqrt(m2)
        n = np.sign(x) * np.sqrt(n2)
        denom = m ** 2 + n ** 2
        sig_x = -1 * p0 / a * (m * (1 + (z ** 2 + n ** 2) / denom) - 2 * z)
        sig_z = -1 * p0 / a * m * (1 - (z ** 2 + n ** 2) / denom)
        tau_xz = -1 * p0 / a * n * ((m ** 2 - z ** 2) / denom)
        return {"sigma_x": sig_x, "sigma_z": sig_z, "tau_xz": tau_xz}

    return stress


def most_critical(records, label):
    if not records:
        raise RuntimeError(
            f"'{label}': ball_case_study/roller_case_study returned no records "
            f"(None or empty list). Most likely hertz_case_studies.py on disk is "
            f"out of date -- check that its functions end with 'return records' "
            f"and that the case-study calls are under 'if __name__ == \"__main__\":' "
            f"(otherwise they also fire a second time, unconditionally, at import)."
        )
    return max(records, key=lambda r: r["result"]["max_pressure"])


def plot_point_contact(rec, out_path, normalize="a"):
    """Dimensionless elliptical pressure map, p/p0, over the contact.

    normalize picks how the in-plane axes are non-dimensionalized -- this
    is a real trade-off, not just a style choice, so it's an explicit
    parameter rather than something buried in the plotting code:

      "a"  (default): BOTH x and y divided by the same length, the
           semi-major axis a. Preserves the TRUE aspect ratio b/a of the
           contact ellipse on the plot (e.g. the b/a~0.10 elements are
           visibly a thin sliver, not a circle) -- x/a in [-1,1] still
           reaches the ellipse boundary, but y/a only reaches +-(b/a).
           Use this when the shape of the contact patch itself matters
           (comparing how elongated different elements/bearings are).

      "ab": x divided by a, y divided by b (its OWN semi-axis). Always
           renders as a unit circle regardless of the true b/a, because
           each axis is separately rescaled to reach +-1 at its own
           ellipse boundary. Loses all information about how elongated
           the real contact is -- every element looks identical in shape.
           Use this only when comparing the pressure PROFILE shape
           (e.g. edge falloff) across elements, not the contact geometry.
    """
    res = rec["result"]
    a, b = res["contact_radii"]
    p0 = res["max_pressure"]
    x = np.linspace(-1.05 * a, 1.05 * a, 250)
    y = np.linspace(-1.05 * b, 1.05 * b, 250)
    X, Y = np.meshgrid(x, y)
    P = np.nan_to_num(res["pressure_f"](X, Y)) / p0

    if normalize == "a":
        Xn, Yn = X / a, Y / a
        y_label = "$y/a$, transverse direction"
    elif normalize == "ab":
        Xn, Yn = X / a, Y / b
        y_label = "$y/b$, transverse direction"
    else:
        raise ValueError(f"normalize must be 'a' or 'ab', got {normalize!r}")

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    cf = ax.contourf(Xn, Yn, P, levels=30, cmap="inferno")
    fig.colorbar(cf, ax=ax, label="$p/p_0$")
    ax.set_xlabel("$x/a$, rolling direction")
    ax.set_ylabel(y_label)
    ax.set_aspect("equal")
    # two lines, kept narrow so long titles never run into the colorbar
    ax.set_title(f"{rec['label']} — elem {rec['elem']} ({rec['race']})\n"
                 f"$Q_j$={rec['Qj']:.1f} N  $p_0$={p0:.1f} MPa  "
                 f"($a$={a:.3f} mm, $b$={b:.3f} mm)", fontsize=9.5, loc="center")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_line_contact(rec, out_path):
    res = rec["result"]
    a = res["contact_radii"][0]
    x = np.linspace(-a, a, 400)
    p = res["pressure_f"](x)

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.plot(x, p, color="firebrick", linewidth=1.5)
    ax.fill_between(x, p, alpha=0.15, color="firebrick")
    ax.set_xlabel("x, across contact width [mm]")
    ax.set_ylabel("Contact pressure, $p$ [MPa]")
    ax.set_ylim(bottom=0)
    ax.set_title(f"{rec['label']} — elem {rec['elem']} ({rec['race']})  "
                 f"$Q_j$={rec['Qj']:.1f} N  $p_0$={res['max_pressure']:.1f} MPa", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_line_contact_stress(rec, out_path):
    """Subsurface stresses in the style of Johnson, Contact Mechanics,
    Fig. 4.5 -- a single shared axis: sigma_x/p0, sigma_z/p0 and the
    SIGNED principal shear tau1/p0 (eq. 4.46/4.48) on the LEFT side
    (negative, x=0, the axis of symmetry), and contours of |tau1|/p0
    over the full (x/a, z/a) field on the RIGHT side (positive x/a) --
    both sharing the same horizontal scale (both are O(1) dimensionless
    quantities) and the same vertical z/a axis, exactly as Johnson
    overlays them on one set of axes. Uses the local corrected line-
    contact stress formula above (not slippy's 'stress_f').

    sigma_x/p0, sigma_z/p0, tau1/p0 (left) come from the closed-form
    z-axis (x=0) result, eq. 4.46; tau1 there is SIGNED
    (0.5*(sigma_z-sigma_x), matches eq. 4.46 exactly, verified
    numerically) not the unsigned Mohr's-circle magnitude, so it lands
    on the same (compressive) side as sigma_x/sigma_z. |tau1| peaks at
    0.30*p0 at z=0.78a (eq. 4.47).

    |tau1|/p0 contours (right) use the general, UNSIGNED Mohr's-circle
    magnitude 0.5*sqrt((sigma_x-sigma_z)^2+4*tau_xz^2) (matching the
    positive contour labels in Fig. 4.5(b); reduces to the signed left-
    side value at x=0, where tau_xz=0). Only x/a>=0 is computed (the
    field is symmetric in x), giving roughly double the line
    resolution for the same point budget.
    """
    res = rec["result"]
    a = res["contact_radii"][0]
    p0 = res["max_pressure"]
    stress_f = _line_contact_stress(a, p0)

    def tau1_magnitude(sx, sz, txz):
        return 0.5 * np.sqrt((sx - sz) ** 2 + 4 * txz ** 2)

    fig, ax = plt.subplots(figsize=(8.0, 6.4))

    # --- left side (x<0 on the shared axis): stresses at x=0, the axis
    # of symmetry -- plotted as sigma/p0 (negative, compressive) -------
    za = np.linspace(1e-4 * a, 2.2 * a, 400)
    s = stress_f(np.zeros_like(za), za)
    t1_signed = 0.5 * (s["sigma_z"] - s["sigma_x"])  # matches Johnson eq. 4.46 sign

    ax.plot(s["sigma_x"] / p0, za / a, label=r"$\sigma_x/p_0$", linewidth=1.4)
    ax.plot(s["sigma_z"] / p0, za / a, label=r"$\sigma_z/p_0$", linewidth=1.4)
    ax.plot(t1_signed / p0, za / a, label=r"$\tau_1/p_0$ (on-axis)", linewidth=1.6, color="black")

    # --- right side (x/a>=0 on the shared axis): |tau1|/p0 contours ---
    # z extends to 3.2a so the outer levels (e.g. 0.15) actually close
    # instead of being cut off at the bottom edge, matching Fig. 4.5(b).
    x_grid = np.linspace(0.0, 1.6 * a, 300)
    z_grid = np.linspace(1e-4 * a, 3.2 * a, 300)
    X, Z = np.meshgrid(x_grid, z_grid)
    s2 = stress_f(X, Z)
    T1 = tau1_magnitude(s2["sigma_x"], s2["sigma_z"], s2["tau_xz"]) / p0

    levels = [0.15, 0.2, 0.23, 0.25, 0.27, 0.29, 0.3]
    cf = ax.contour(X / a, Z / a, T1, levels=levels, colors="black", linewidths=0.9)

    # Near x/a=1, z/a~0 several close levels crowd into the same corner
    # and their inline labels overlap. Instead, place each label along
    # the x/a=0.18 column, where tau1 varies monotonically with z for
    # z>0.78a (past the max-shear depth) and each level is well
    # separated.
    x_label_col = 0.18  # in x/a; kept off the x=0 edge so labels aren't clipped
    x_col_idx = int(np.argmin(np.abs(x_grid - x_label_col * a)))
    z_col = z_grid / a
    t1_col = T1[:, x_col_idx]
    past_max = z_col > 0.78  # descending branch only, single-valued
    label_pos = []
    for lev in levels:
        zc, tc = z_col[past_max], t1_col[past_max]
        idx = np.argmin(np.abs(tc - lev))
        label_pos.append((x_label_col, zc[idx]))
    ax.clabel(cf, inline=False, fontsize=7.5, fmt="%.3f", manual=label_pos)

    # --- shared axis furniture ------------------------------------------
    ax.axvline(0, color="0.5", linewidth=0.8)
    ax.invert_yaxis()
    ax.set_xlim(-1.15, 1.65)
    z_view_max = za.max() / a  # crop to where panel (a)'s curves actually have data
    ax.set_ylim(z_view_max, 0)
    ax.set_ylabel(r"$z/a$")
    ax.set_aspect("equal")
    ax.legend(fontsize=9, frameon=False, loc="lower left")
    ax.grid(True, alpha=0.25)

    # numbered x-axis (ticks + label) at the TOP, as in Johnson's Fig. 4.5,
    # instead of matplotlib's default bottom placement
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", pad=6)

    # two-part x label, since the same axis carries sigma/p0 on the left
    # and x/a on the right (as in Fig. 4.5) -- in axes-fraction coordinates
    # (not data z/a) so it sits just above the numbered ticks regardless
    # of the data range, with the title further above via its own pad
    ax.text(0.232, 1.065, r"$\sigma/p_0$", ha="center", fontsize=10.5,
             transform=ax.transAxes, clip_on=False)
    ax.text(0.679, 1.065, r"$x/a$", ha="center", fontsize=10.5,
             transform=ax.transAxes, clip_on=False)

    # panel captions below each half, as in Johnson's Fig. 4.5 (a)/(b) --
    # positions computed from the actual xlim so they stay centered under
    # each half however the x-range is tuned later
    xlo, xhi = ax.get_xlim()
    frac_split = (0.0 - xlo) / (xhi - xlo)  # x/a=0, the boundary between halves
    cap_left = frac_split / 2
    cap_right = frac_split + (1 - frac_split) / 2
    ax.text(cap_left, -0.075, "(a) on-axis stresses ($x=0$)", ha="center", fontsize=9.5,
             transform=ax.transAxes, clip_on=False)
    ax.text(cap_right, -0.075, r"(b) contours of $\tau_1/p_0$ (Mohr's circle)", ha="center",
             fontsize=9.5, transform=ax.transAxes, clip_on=False)

    ax.set_title(f"{rec['label']} — elem {rec['elem']} ({rec['race']})  "
                 f"$Q_j$={rec['Qj']:.1f} N  $p_0$={p0:.1f} MPa", fontsize=10.5, pad=55)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


records = {
    "B1_dgb":  ball_case_study(brg_dgb, "B1_dgb"),
    "B3_ang":  ball_case_study(brg_ang, "B3_ang"),
    "B2_roll": roller_case_study(brg_roll1, "B2_roll"),
    "B4_roll": roller_case_study(brg_roll2, "B4_roll"),
}
critical = {label: most_critical(recs, label) for label, recs in records.items()}

plots_root = Path(__file__).resolve().parent / "plots"
point_dir = plots_root / "point_contact"
line_pressure_dir = plots_root / "line_contact_coarse" / "pressure"
line_stress_dir = plots_root / "line_contact_coarse" / "stress"
for d in (point_dir, line_pressure_dir, line_stress_dir):
    d.mkdir(parents=True, exist_ok=True)

print()
for label, rec in critical.items():
    if rec["contact"] == "point":
        # both normalizations (see plot_point_contact docstring): "_a" keeps
        # the true elongated ellipse shape, "_ab" is a unit circle that only
        # shows the pressure profile shape, not the contact geometry.
        out_path_a = point_dir / f"{label}_a.png"
        out_path_ab = point_dir / f"{label}_ab.png"
        plot_point_contact(rec, out_path_a, normalize="a")
        plot_point_contact(rec, out_path_ab, normalize="ab")
        print(f"plot written to: {out_path_a}")
        print(f"plot written to: {out_path_ab}")
    else:
        pressure_path = line_pressure_dir / f"{label}.png"
        stress_path = line_stress_dir / f"{label}.png"
        plot_line_contact(rec, pressure_path)
        plot_line_contact_stress(rec, stress_path)
        print(f"plot written to: {pressure_path}")
        print(f"plot written to: {stress_path}")
