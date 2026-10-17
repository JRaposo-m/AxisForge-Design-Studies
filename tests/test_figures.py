"""The house style sheet and the figure functions (smoke tests from tiny synthetic inputs).

A smoke test checks the form, not the physics: a ``Figure`` comes back (or the files are written),
no axes carries a title and no grid is drawn. Each test closes what it opened.
"""
import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import to_hex

from validation._support.style import STYLE_PATH


@pytest.fixture(autouse=True)
def house_style():
    """Every test draws in the house style and leaves no figure open."""
    with plt.style.context(STYLE_PATH):
        yield
    plt.close("all")


def test_style_sheet_sets_the_house_style():
    """The sheet loads and sets the paper look: no grid, inward ticks, black first and for bare lines."""
    rc = plt.rcParams
    assert rc["axes.grid"] is False
    assert rc["xtick.direction"] == "in" and rc["ytick.direction"] == "in"
    assert to_hex(rc["axes.prop_cycle"].by_key()["color"][0]) == "#000000"
    assert to_hex(rc["lines.color"]) == "#000000"


# --- smoke tests of the figure functions (everything above runs with the style sheet alone) -----
import importlib.util  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from references.contact.hertz import hertz_line, hertz_point  # noqa: E402
from slippy_bridge.contact import ContactCase, Grid  # noqa: E402
from validation._support.figures import contact_plots  # noqa: E402
from validation._support.figures import shaft_studies as fg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def assert_house_form(fig):
    """A Figure with no axes title and no grid lines."""
    assert isinstance(fig, Figure)
    for ax in fig.axes:
        assert ax.get_title() == "" and ax.get_title(loc="left") == ""
        assert not any(line.get_visible() for line in ax.get_xgridlines() + ax.get_ygridlines())


# =============================================================================
# contact_plots
# =============================================================================


def _records(n_mesh, n_domain, scale_second=1.5):
    """Sweep records of two builds with errors that shrink with h and with the domain; the second
    build's mesh errors are ``scale_second`` times the first's (1.0: the same series)."""
    out = []
    for build, sign in (("composite", 1.0), ("two_profiles", -scale_second)):
        for k in range(n_mesh):
            h = 2e-6 * 2.0**-k                                # [m]
            out.append(dict(build=build, sweep="mesh", span_x=4.0, h_m=h, err_p0=sign * 1e3 * h,
                            err_s_x=2e3 * h, err_s_y=math.nan, err_delta=-5e2 * h))
        for k in range(n_domain):
            out.append(dict(build=build, sweep="domain", span_x=2.0 + k, h_m=2e-6,
                            err_p0=-0.02 / (1 + k), err_s_x=0.0, err_s_y=math.nan, err_delta=0.0))
    return out


def test_contact_convergence_draws_mesh_and_domain_panels(tmp_path):
    fig = contact_plots.convergence(_records(3, 3), tmp_path / "c.png", criterion=("E3 (p0)", 1.0), default_span=4.0)
    assert_house_form(fig)
    assert len(fig.axes) == 3                                 # one mesh panel per build + domain
    assert fig.axes[0].get_ylim() == fig.axes[1].get_ylim()   # one error scale for the mesh panels
    assert (tmp_path / "c.png").is_file()


def test_contact_convergence_draws_identical_builds_once():
    fig = contact_plots.convergence(_records(3, 3, scale_second=1.0))
    assert len(fig.axes) == 2 and "drawn once" in fig.texts[-1].get_text()


def test_contact_convergence_never_draws_an_empty_axes():
    """A smoke sweep (one grid, no domain sweep) has nothing to draw: no axes, only the note."""
    fig = contact_plots.convergence(_records(1, 0))
    assert_house_form(fig)
    assert fig.axes == [] and "< 3 points" in fig.texts[0].get_text()


@pytest.mark.parametrize("line", [False, True])
def test_pressure_cuts_point_and_line(line, tmp_path):
    """Hertz pressure on the grid as the 'solution': two panels for a point contact, one for a line."""
    if line:
        case = ContactCase("roller", r1=(5e-3, math.inf), r2=(30e-3, math.inf), E=210e9, nu=0.3,
                           load=5e4, line=True)                      # load per unit length [N/m]
        ref = hertz_line(case.r1, case.r2, case.E, case.nu, case.load)
    else:
        case = ContactCase("ball_flat", r1=6.35e-3, r2=math.inf, E=210e9, nu=0.3, load=500.0)
        ref = hertz_point(case.r1, case.r2, case.E, case.nu, case.load)
    grid = Grid.around(case)
    rho2 = (grid.X / ref["s_x_m"]) ** 2 + (0.0 if line else (grid.Y / ref["s_y_m"]) ** 2)
    p = ref["p0_Pa"] * np.sqrt(np.clip(1.0 - rho2, 0.0, None))         # [Pa]
    fig = contact_plots.pressure_cuts(p, grid, ref, tmp_path / "p.png")
    assert_house_form(fig)
    assert len(fig.axes) == (1 if line else 2)
    assert (tmp_path / "p.png").is_file()


# =============================================================================
# shaft_studies
# =============================================================================

X = np.linspace(0.0, 100.0, 11)                               # [mm]


def test_shaft_field_comparison():
    fig = fg.field_comparison([fg.Series("analytical", X, X**2, kind="theory"), fg.Series("FEM", X, X**2)],
                              [fg.Series("FEM", X, 0.01 * X)], ylabel="$v$ (mm)", delta_label="d (%)",
                              supports=(0.0, 100.0), support_names=("A", "B"), note="shaft1")
    assert_house_form(fig)


def test_shaft_convergence():
    n = np.array([4.0, 8.0, 16.0])
    fig = fg.convergence(n, [fg.Series("true error", n, n**-2.0), fg.Series("GCI", n, 2 * n**-2.0, kind="other")],
                         ylabel="relative error", note="span")
    assert_house_form(fig)


def test_shaft_layout():
    fig = fg.shaft_layout([0.0, 40.0, 100.0], [20.0, 30.0], [fg.Series("q", X, np.ones_like(X), kind="theory")],
                          supports=(10.0, 90.0), support_names=("A", "B"))
    assert_house_form(fig)


def test_series_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match="kind"):
        fg.Series("x", X, X, kind="measured")


# =============================================================================
# studies/bearings/load_distribution/_common/plots.py (a study package: loaded from its path)
# =============================================================================


@pytest.fixture(scope="module")
def ld_plots():
    path = ROOT / "studies" / "bearings" / "load_distribution" / "_common" / "plots.py"
    spec = importlib.util.spec_from_file_location("_ld_common_plots", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def saved(ld_plots, monkeypatch):
    """Keep the figures that ``_save`` would write and close: {file name: Figure}."""
    figures = {}
    monkeypatch.setattr(ld_plots, "_save", lambda fig, path: figures.__setitem__(path.name, fig))
    return figures


def _ld_rows(kind="roller"):
    """Rows of two clearances; element loads (Z = 5) of a loaded zone around phi = 0."""
    rows, q_rows, lam_rows = [], [], []
    for s in (0.0, 0.02, 0.04):
        rows.append(dict(shaft="shaft_1", kind=kind, Z=5, s_mm=s, ok=True, n_loaded=3, zone_half_angle_deg=80.0,
                         Fr_N=1000.0, Q_max_N=500.0 + 1e3 * s, Q_max_over_Fr_per_Z=(500.0 + 1e3 * s) / (1000.0 / 5), delta_r_mm=0.01 + s / 2,
                         Kr_N_per_mm=1e5, L10r_Mrev=100.0))
        for phi, Q in ((0.0, 500.0), (72.0, 250.0), (144.0, 0.0), (-144.0, 0.0), (-72.0, 250.0)):
            q_rows.append(dict(shaft="shaft_1", s_mm=s, phi_deg=phi, Q_N=Q))
        for x in (-1.0, 0.0, 1.0):
            lam_rows.append(dict(shaft="shaft_1", s_mm=s, x_mm=x, q_N=100.0 - 10 * x**2))
    return rows, q_rows, lam_rows


def test_ld_plot_summary(ld_plots, saved, tmp_path):
    rows, _, _ = _ld_rows()
    ld_plots.plot_summary(rows, tmp_path, x_axis="s_mm", note="clearance")
    assert set(saved) == {f"{m[3]}.png" for m in ld_plots.SUMMARY_METRICS}
    for fig in saved.values():
        assert_house_form(fig)


def test_ld_plot_polar_distributions(ld_plots, saved, tmp_path):
    rows, q_rows, _ = _ld_rows()
    ld_plots.plot_polar_distributions(rows, q_rows, tmp_path, axis_names=["s_mm"], note="clearance")
    assert len(saved) == 3 + 1                                # one per point + the overview
    for fig in saved.values():
        assert_house_form(fig)
        assert ld_plots.A2_NOTE in fig.texts[-1].get_text()     # the azimuth stamp (R-003)


def test_ld_plot_lamina_profile(ld_plots, saved, tmp_path):
    _, _, lam_rows = _ld_rows()
    ld_plots.plot_lamina_profile(lam_rows, tmp_path, x_axis="s_mm", note="clearance")
    assert list(saved) == ["roller_lamina_profile.png"]
    assert_house_form(saved["roller_lamina_profile.png"])


def test_ld_plot_compare(ld_plots, saved, tmp_path):
    datasets = {"deep_groove": _ld_rows("ball")[0], "cylindrical_roller": _ld_rows("roller")[0]}
    ld_plots.plot_compare(datasets, tmp_path, x_axis="s_mm", note="clearance")
    assert len(saved) == len(ld_plots.SUMMARY_METRICS)
    for fig in saved.values():
        assert_house_form(fig)


# =============================================================================
# bearing_plots and shaft_studies.shaft_fields
# =============================================================================

PHI = np.linspace(-math.pi, math.pi, 9, endpoint=False)       # 9 elements, local azimuth [rad]
Q = np.clip(1000.0 * np.cos(PHI), 0.0, None)                  # loaded zone around phi = 0 [N]


def test_sign_check_one_panel_per_load_direction():
    """Two load directions, two bearings each (drawn over each other): two panels."""
    from validation._support.figures import bearing_plots
    phi_g = PHI % (2 * math.pi)
    fig = bearing_plots.sign_check([[phi_g, phi_g]] * 2, [[Q, Q]] * 2, phi_Fr_rad=[0.0, 0.0],
                                   phi_expected_rad=[math.pi, math.pi], series=["A", "B"], labels=["case 1", "case 2"])
    assert_house_form(fig)
    assert len(fig.axes) == 2 and bearing_plots.SIGN_CHECK_NOTE in fig.texts[-1].get_text()


@pytest.mark.parametrize("kind, n_axes, text", [("ball, alpha drifts", 3, "floored at 0"),
                                                ("ball, alpha constant", 2, "constant over the loaded zone"),
                                                ("roller", 2, "loaded")])
def test_element_loads_alpha_panel(kind, n_axes, text):
    """The alpha panel only when alpha varies over the loaded zone; never for rollers."""
    from validation._support.figures import bearing_plots
    roller = kind == "roller"
    delta = np.clip(1e-5 * Q, 0.0, None) - (2e-3 if roller else 0.0)       # roller: negative approach = gap [mm]
    alpha = None if roller else 0.3 + (0.01 * np.cos(PHI) if "drifts" in kind else 0.0 * PHI)   # [rad]
    fig = bearing_plots.element_loads(PHI, Q, delta_mm=delta, alpha_rad=alpha)
    assert_house_form(fig)
    assert len(fig.axes) == n_axes
    note = fig.texts[-1].get_text()
    assert f"loaded {int((Q > 0).sum())} of 9" in note and bearing_plots.A2_NOTE in note and text in note


def test_contact_ellipses_two_races():
    from validation._support.figures import bearing_plots
    phi = np.concatenate([PHI[Q > 0]] * 2)
    race = ["inner"] * int((Q > 0).sum()) + ["outer"] * int((Q > 0).sum())
    p0 = np.concatenate([2e9 * np.ones((Q > 0).sum()), 1.6e9 * np.ones((Q > 0).sum())])     # [Pa]
    fig = bearing_plots.contact_ellipses(phi, race, p0, 3e-4 + 0 * p0, 2.5e-3 + 0 * p0)
    assert_house_form(fig)
    assert len(fig.axes) == 3


def test_shaft_fields():
    x = np.linspace(0.0, 200.0, 11)                            # [mm]
    fig = fg.shaft_fields(x, 1e-3 * np.sin(np.pi * x / 200), 1e-3 * np.cos(np.pi * x / 200), 0 * x,
                          1e5 * np.sin(np.pi * x / 200), 50 * np.sin(np.pi * x / 200),
                          bearings_mm={"A": 10.0, "B": 190.0}, gears_mm={"gear": 100.0}, R_N=[1000.0, 900.0])
    assert_house_form(fig)
    assert len(fig.axes) == 4 and "R_" in fig.texts[-1].get_text()


def test_words_keeps_mathtext_subscripts():
    """Code identifiers become words; underscores inside $...$ stay subscripts."""
    from validation._support.figures import _words
    assert _words(r"radial_heavy; $Q_\mathrm{max}$ = 1 N, 6210_A") == r"radial heavy; $Q_\mathrm{max}$ = 1 N, 6210 A"
