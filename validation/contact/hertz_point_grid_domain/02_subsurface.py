# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Ball on flat: subsurface principal shear on the axis, SlipPY `SubsurfaceStress` against Johnson
#
# **Case:** `validation/contact/hertz_point_grid_domain` (V-DRY-01), criteria P4-P5 of the README.
#
# The ball-on-flat contact is solved once on the default grid of the bridge, with SlipPY's
# `SubsurfaceStress` sub-model on body 1 (non-periodic, depths $0 \le z \le 2a$, step $0.02a$). The
# model is built here with SlipPY directly (José, Q2: `slippy_bridge.contact.static_step` has no
# sub-models); the surfaces come from `slippy_bridge.contact.pair`.
#
# On the axis of a Hertz sphere $\sigma_r$, $\sigma_\theta$, $\sigma_z$ are principal, so
# $\tau_1 = (\sigma_1 - \sigma_3)/2$. The judge is Johnson (1985): the printed $0.31 p_0$ at
# $0.48a$ ($\nu = 0.3$, PDF 105; Eq. (3.46)'s $0.57a$ is a misprint) and Eq. (3.45) evaluated here
# directly. $a$ and $p_0$ are those of `references.contact.hertz.hertz_point`.
#
# **Solve protocol** (José, 2026-10-16): attempt 1 `model.solve()` as is; if it raises, the
# traceback and the SlipPY log are saved and the model is rebuilt with a new name; attempt 2
# `solve(skip_data_check=True)`; attempt 3 the same with SlipPY's own switch
# `slippy.ERROR_IF_MISSING_SUB_MODEL = False` (restored afterwards). Attempts 1-2 are José's
# protocol; attempt 3 is a diagnostic beyond it (SlipPY's own switch, named in its error message;
# no SlipPY change; José to confirm whether it counts), and a criterion graded on its field says
# so. Every outcome is recorded; a FAIL of the field carries "suspected SlipPY defect, not fixed".
# Nothing in SlipPY is worked around: a field that disagrees is recorded with value, expected and
# ratio as a suspected SlipPY defect. `--smoke` has no effect (one default grid already).

# %%
import math
import shutil
import time
import traceback

import matplotlib.pyplot as plt
import numpy as np
import slippy
import slippy.contact as c
from axisforge.outputs._format import format_table, write_records_csv
from slippy.contact.sub_models import SubsurfaceStress

from references.contact.hertz import hertz_point, hertz_point_axis_stress
from slippy_bridge.contact import Grid, pair
from validation._support.figures import contact_plots
from validation._support.style import use_style

from construction import CASE, RESULTS, Z_OVER_A

use_style()                                       # house paper style (style/README.md)

NAME = "02_subsurface"
SLIPPY_OUT = RESULTS / "slippy_subsurface"
SLIPPY_OUT.mkdir(parents=True, exist_ok=True)                       # git-ignored (D-0020)

JUDGE = hertz_point(CASE.r1, CASE.r2, CASE.E, CASE.nu, CASE.load)
A, P0 = JUDGE["s_x_m"], JUDGE["p0_Pa"]                              # contact radius [m], max pressure [Pa]
NU = CASE.nu[0]                                                     # both bodies steel, nu = 0.3
GRID = Grid.around(CASE)                                            # default grid of the bridge

# %% [markdown]
# ## 1. Model and solve protocol

# %%
def build_model(name):
    """ContactModel of the ball on the flat with the static step and SubsurfaceStress on body 1.

    The step arguments are those of ``slippy_bridge.contact.static_step.run`` (method "pk",
    tolerance 1e-9, the case load). Only the principal stresses are requested: SlipPY rejects tensor
    components in this sub-model (README, Known SlipPY issues).
    """
    s1, s2 = pair(CASE, GRID)
    model = c.ContactModel(name, s1, s2, output_dir=str(SLIPPY_OUT))
    step = c.StaticStep("static", normal_load=CASE.load, method="pk", tolerance=1e-9)
    model.add_step(step)
    step.add_sub_model(SubsurfaceStress(z=Z_OVER_A * A, surface=1, periodic_axes=(False, False),
                                        stress_components=("1", "3")))
    return model


# CSV labels (UTF-8 files; the console prints ASCII only)
SLIPPY_DEFECT = "suspected SlipPY defect, not fixed (José, 2026-10-16)"
ATTEMPT_3 = ("attempt 3: beyond the ruling's two attempts; SlipPY's own switch named in its error message; "
             "no SlipPY change; José to confirm whether it counts")
ATTEMPTS = ((1, "solve()", {}, False, "ruling (José)"),
            (2, "solve(skip_data_check=True)", {"skip_data_check": True}, False, "ruling (José)"),
            (3, "solve(skip_data_check=True), ERROR_IF_MISSING_SUB_MODEL=False", {"skip_data_check": True}, True,
             ATTEMPT_3))
outcomes, state, solved_by = [], None, None
for attempt, call, kwargs, switch_off, note in ATTEMPTS:
    model = build_model(f"{CASE.label}_subsurface_{attempt}_{time.time_ns()}")   # a new name every attempt
    saved = slippy.ERROR_IF_MISSING_SUB_MODEL
    t0 = time.perf_counter()
    try:
        if switch_off:
            slippy.ERROR_IF_MISSING_SUB_MODEL = False
        state = model.solve(**kwargs)
        solved_by = attempt
        outcomes.append({"attempt": attempt, "call": call, "outcome": "solved", "error": "", "note": note,
                         "wall_s": time.perf_counter() - t0})
        break
    except Exception as exc:                                        # recorded, never fixed here
        (RESULTS / f"{NAME}_traceback_{attempt}.txt").write_text(traceback.format_exc(), encoding="utf-8")
        log = SLIPPY_OUT / f"{model.name}.log"
        if log.is_file():
            shutil.copyfile(log, RESULTS / f"{NAME}_attempt_{attempt}.log")
        outcomes.append({"attempt": attempt, "call": call, "outcome": "raised", "note": note,
                         "error": f"{type(exc).__name__}: {str(exc).splitlines()[0]}",
                         "wall_s": time.perf_counter() - t0})
    finally:
        slippy.ERROR_IF_MISSING_SUB_MODEL = saved
print(format_table((("attempt", "attempt", ".0f"), ("call", "call", "s"), ("outcome", "outcome", "s"),
                    ("wall_s", "t [s]", ".2f"), ("error", "error", "s")), outcomes))

# %% [markdown]
# ## 2. Johnson Eq. (3.45): axis stresses of the Hertz sphere
#
# With $\zeta = z/a$:
# $\sigma_r/p_0 = \sigma_\theta/p_0 = -(1+\nu)\{1 - \zeta\tan^{-1}(1/\zeta)\} + \tfrac12(1+\zeta^2)^{-1}$,
# $\sigma_z/p_0 = -(1+\zeta^2)^{-1}$, $\tau_1 = \tfrac12|\sigma_z - \sigma_r|$ (Johnson 1985, Eq. (3.45),
# PDF 72), by `references.contact.hertz.hertz_point_axis_stress`. Evaluated on a fine depth grid
# for its maximum, and on `Z_OVER_A` for the comparison.

# %%
zeta_fine = np.linspace(0.0, 2.0, 200001)                           # step 1e-5 a: the depth to 5 digits
tau_fine = hertz_point_axis_stress(zeta_fine, NU)["tau1_over_p0"]
TAU_EQ, Z_EQ = float(tau_fine.max()), float(zeta_fine[tau_fine.argmax()])
print(f"Eq. (3.45), nu = {NU}: tau1 max {TAU_EQ:.5f} p0 at z = {Z_EQ:.5f} a")

# %% [markdown]
# ## 3. SlipPY field on the axis, and the criteria
#
# The axis is node $(n_x/2, n_y/2)$ (the grid centre is the contact centre). P5 reads the surface
# row (`Z_OVER_A[0]`, 1e-4 of the first depth step, SlipPY's own choice; its kernels are singular
# at $z = 0$): the shear $(\sigma_1-\sigma_3)/2$ at the origin, and at the first node with $r \ge a$
# along $+x$ the radial tension $\sigma_r = \sigma_1$ and the surface shear
# $(\sigma_r - \sigma_\theta)/2 = (\sigma_1-\sigma_3)/2$. Outside the contact $\sigma_z = 0$ and
# $\sigma_\theta = -\sigma_r$, so both equal $\sigma_r/p_0 = (1-2\nu)a^2/(3r^2)$ (Eq. (3.44), PDF 72),
# evaluated at that node's $r$ (0.13 $p_0$ at $r = a$). Tolerances: README P4, P5 (**[proposed]** for P5).

# %%
def row(cid, quantity, value, expected, limit, ok, note=""):
    """One criteria record; err is value - expected, ratio value/expected; ok None means recorded, not graded.

    A FAIL of a SlipPY field carries the defect label after ``note``.
    """
    return {"id": cid, "quantity": quantity, "value": value, "expected": expected, "err": value - expected,
            "ratio": value / expected if expected else math.nan, "limit": limit,
            "result": "recorded" if ok is None else ("PASS" if ok else "FAIL"),
            "note": "; ".join(s for s in (note, SLIPPY_DEFECT if ok is not None and not ok else "") if s)}


i0, j0 = GRID.X.shape[0] // 2, GRID.X.shape[1] // 2
criteria, profile = [], []
graded = f"graded on attempt {solved_by}" + (" (diagnostic)" if solved_by == 3 else "")
print(graded)
if state is None:
    for cid, q in (("P4", "tau1 max / p0"), ("P4", "depth of tau1 max / a"), ("P5", "surface stresses")):
        criteria.append(row(cid, q + " (no field: every attempt raised)", math.nan, math.nan, "-", None,
                            f"blocked: {SLIPPY_DEFECT}; tracebacks {NAME}_traceback_*.txt"))
else:
    s1, s3 = np.asarray(state["surface_1_1"]), np.asarray(state["surface_1_3"])   # (z, x, y) [Pa]
    tau_axis = 0.5 * (s1[:, i0, j0] - s3[:, i0, j0]) / P0
    k = int(np.nanargmax(tau_axis))
    tau_eq = hertz_point_axis_stress(Z_OVER_A, NU)["tau1_over_p0"]
    profile = [{"z_over_a": z, "tau1_over_p0": t, "tau1_eq_over_p0": te, "ratio": t / te if te else math.nan}
               for z, t, te in zip(Z_OVER_A, tau_axis, tau_eq)]
    k_eq = int(np.argmin(abs(Z_OVER_A - Z_EQ)))                     # node nearest Johnson's depth
    i_a = i0 + math.ceil(A / GRID.h)                                # first node with r >= a along +x
    r_a = (i_a - i0) * GRID.h                                       # its radius [m]
    edge = (1.0 - 2.0 * NU) * A**2 / (3.0 * r_a**2)                 # Eq. (3.44): sigma_r/p0 = -sigma_theta/p0
    tau_surf = 0.5 * (s1[0] - s3[0]) / P0
    criteria += [
        row("P4", "tau1 max / p0 (printed)", tau_axis[k], 0.31, "+-0.005",
            abs(tau_axis[k] - 0.31) <= 0.005, graded),
        row("P4", "depth of tau1 max / a (printed)", Z_OVER_A[k], 0.48, "+-0.01",
            abs(Z_OVER_A[k] - 0.48) <= 0.01, graded),
        row("P4", "tau1 max / p0 (Eq. 3.45)", tau_axis[k], TAU_EQ, "+-0.005",
            abs(tau_axis[k] - TAU_EQ) <= 0.005, graded),
        row("P4", "depth of tau1 max / a (Eq. 3.45)", Z_OVER_A[k], Z_EQ, "+-0.01",
            abs(Z_OVER_A[k] - Z_EQ) <= 0.01, graded),
        row("P4", f"tau1 / p0 at z = {Z_OVER_A[k_eq]:.2f} a (Eq. 3.45, field ratio)", tau_axis[k_eq],
            tau_eq[k_eq], "-", None, graded),
        row("P5", "shear at the origin / p0 [proposed]", tau_surf[i0, j0], 0.10, "+-0.005",
            abs(tau_surf[i0, j0] - 0.10) <= 0.005, graded),
        row("P5", f"surface shear at r = {r_a / A:.3f} a / p0 (Eq. 3.44) [proposed]", tau_surf[i_a, j0], edge,
            "+-0.005", abs(tau_surf[i_a, j0] - edge) <= 0.005, graded),
        row("P5", f"radial tension at r = {r_a / A:.3f} a / p0 (Eq. 3.44) [proposed]", s1[0, i_a, j0] / P0, edge,
            "+-0.005", abs(s1[0, i_a, j0] / P0 - edge) <= 0.005, graded),
    ]

print(format_table((("id", "ID", "s"), ("quantity", "quantity", "s"), ("value", "value", ".5g"),
                    ("expected", "expected", ".5g"), ("err", "err", "+.3e"), ("ratio", "ratio", ".3e"),
                    ("limit", "limit", "s"), ("result", "result", "s")), criteria))
write_records_csv(RESULTS / f"{NAME}.csv", outcomes + profile)
write_records_csv(RESULTS / f"{NAME}_criteria.csv", criteria)

# %% [markdown]
# ## 4. Figure: the closed-form stresses on the axis
#
# Johnson's Eq. (3.45) against depth, with the printed maximum (0.31 $p_0$ at 0.48 $a$) as an open
# marker. The SlipPY subsurface field is not drawn (OPEN R-268, R-269). Written to
# `results/02_subsurface_depth.png`.

# %%
z_plot = np.linspace(0.0, 3.0, 601)                                 # depth / a [-]
fig = contact_plots.subsurface_depth(z_plot, hertz_point_axis_stress(z_plot, NU),
                                     published=[(0.31, 0.48, "Johnson 1985 (printed)")],
                                     note=rf"circular contact, axis $r = 0$, $\nu$ = {NU:g}; "
                                          r"$\sigma_\theta = \sigma_r$ on the axis")
fig.savefig(RESULTS / f"{NAME}_depth.png")
plt.close(fig)                                    # shown once, by the line below
fig
