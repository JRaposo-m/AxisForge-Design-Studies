"""slippy_bridge.contact.roughness: rough surfaces on a grid (macro), their measures, and the rough heights
through solvers.surface_pair and solvers.solve (R4c)."""
import itertools
import math
import warnings

import numpy as np
import pytest

pytest.importorskip("slippy")

from slippy_bridge.contact.core.contact import Contact, SweptCircleBody  # noqa: E402
from slippy_bridge.contact.mesh.grid import Grid  # noqa: E402
from slippy_bridge.contact.roughness.macro import RoughnessSpec, band_limit, rough_heights, rough_measures  # noqa: E402
from slippy_bridge.contact.solvers.static import solve  # noqa: E402
from slippy_bridge.contact.solvers.surfaces import surface_pair  # noqa: E402

H = 2.0**-18                                     # 3.81 um [m]
UM = 1e-6
_ids = itertools.count()                         # SlipPY material names are process-wide


class _Point(Contact):
    line = False


def _grid(n0, n1, h=H):
    """Grid by hand, origin on node (n0//2, n1//2) as Grid.around."""
    X, Y = np.meshgrid((np.arange(n0) - n0 // 2) * h, (np.arange(n1) - n1 // 2) * h, indexing="ij")
    return Grid(X=X, Y=Y, h=h)


def _acf(rq=0.2 * UM, beta=5 * H, lambda_s=2 * H, seed=1, **kw):
    """Exponential-ACF spec; lambda_s = 2h keeps every wave vector up to the Nyquist along the axes."""
    return RoughnessSpec(kind="acf", rq=rq, beta_star=beta, lambda_s=lambda_s, seed=seed, **kw)


def _psd(rq=0.2 * UM, seed=1, **kw):
    kw = {"hurst": 0.8, "q_r": 2 * math.pi / (32 * H), "lambda_s": 4 * H, **kw}
    return RoughnessSpec(kind="psd", rq=rq, seed=seed, **kw)


def _quiet(spec, grid, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return rough_heights(spec, grid, **kw)


def _measured_acf(z):
    """Normalised periodic ACF of z by FFT."""
    z = z - z.mean()
    a = np.real(np.fft.ifft2(np.abs(np.fft.fft2(z)) ** 2)) / z.size
    return a / a[0, 0]


# ---- RoughnessSpec ------------------------------------------------------------------------------

def test_random_spec_needs_lambda_s_and_seed():
    errors = RoughnessSpec(kind="acf", rq=1e-7, beta_star=1e-5).validate()
    assert any("lambda_s" in e for e in errors) and any("seed" in e for e in errors)


def test_fir_filter_is_refused_with_its_reason():
    errors = _acf(acf_method="fir_filter").validate()
    assert len(errors) == 1 and "R2-17" in errors[0] and "Random.py:752-757" in errors[0]
    with pytest.raises(ValueError, match="fir_filter"):
        rough_heights(_acf(acf_method="fir_filter"), _grid(16, 16))


def test_psd_constant_gives_the_continuum_rq():
    spec = _psd(rq=None, psd_c0=1e-30)
    q_r, q_s, H_ = spec.q_r, 2 * math.pi / spec.lambda_s, spec.hurst
    q = np.linspace(0.0, q_s, 400001)
    c = np.where(q < q_r, spec.psd_c0, spec.psd_c0 * (np.maximum(q, q_r) / q_r) ** (-2 * (1 + H_)))
    assert spec.target_rq() == pytest.approx(math.sqrt(2 * math.pi * np.trapezoid(c * q, q)), rel=1e-6)


# ---- rough_heights ------------------------------------------------------------------------------

@pytest.mark.parametrize("spec", [_acf(), _acf(acf_shape="gaussian"), _psd()], ids=["exp", "gauss", "psd"])
def test_rq_is_exact_after_the_band_limit(spec):
    rh = _quiet(spec, _grid(128, 128))
    assert rh.rq_lattice == pytest.approx(spec.rq, rel=1e-12) and rh.rq_grid == pytest.approx(spec.rq, rel=1e-12)
    assert abs(rh.z.mean()) < 1e-12 * spec.rq


@pytest.mark.parametrize("shape", ["exponential", "gaussian"])
def test_acf_of_the_surface_matches_the_target(shape):
    """Mean of 3 realisations on 256^2 at beta*/h = 5, lags to 3 beta*: within 0.02 of the target
    (one realisation: 0.010-0.015 for the exponential ACF; the R4 scratch check)."""
    beta_n = 5.0           # lambda_s = h: the S-filter keeps the whole 2-D spectrum (|k| <= 1/(sqrt(2) h))
    a = np.mean([_measured_acf(_quiet(_acf(beta=beta_n * H, lambda_s=H, acf_shape=shape, seed=sd),
                                      _grid(256, 256)).z)
                 for sd in (1, 2, 3)], axis=0)
    r = np.arange(int(3 * beta_n) + 1)
    target = np.exp(-2.3 * r / beta_n) if shape == "exponential" else np.exp(-2.3 * (r / beta_n) ** 2)
    assert np.abs(a[r, 0] - target).max() < 0.02 and np.abs(a[0, r] - target).max() < 0.02


@pytest.mark.parametrize("spec", [_acf(lambda_s=8 * H), _psd(lambda_s=8 * H)], ids=["acf", "psd"])
def test_band_limit_removes_wavelengths_below_lambda_s(spec):
    g = _grid(128, 128)
    z = _quiet(spec, g).z
    k = np.hypot(*np.meshgrid(np.fft.fftfreq(128, H), np.fft.fftfreq(128, H), indexing="ij"))
    power = np.abs(np.fft.fft2(z)) ** 2
    assert power[k > 1.0 / spec.lambda_s].max() < 1e-20 * power.max()
    assert power[(k > 0) & (k <= 1.0 / spec.lambda_s)].min() > 0.0


def test_l_filter_removes_long_wavelengths():
    z = np.random.default_rng(0).normal(size=(64, 64))
    zf = band_limit(z, 1.0, 4.0, 16.0)
    k = np.hypot(*np.meshgrid(np.fft.fftfreq(64), np.fft.fftfreq(64), indexing="ij"))
    power = np.abs(np.fft.fft2(zf)) ** 2
    assert power[(k < 1 / 16.0) | (k > 1 / 4.0)].max() < 1e-20 * power.max()


def test_seed_is_reproducible_and_the_global_state_is_restored():
    g = _grid(64, 64)
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    z1, z2, z3 = (_quiet(_acf(seed=sd), g).z for sd in (7, 7, 8))
    assert np.array_equal(z1, z2) and not np.allclose(z1, z3)
    assert np.array_equal(np.random.get_state()[1], before)


def test_a_coarser_grid_samples_the_same_surface():
    fine, coarse = _grid(128, 128, H / 2), _grid(64, 64, H)
    spec = _acf(beta=10 * H / 2, lambda_s=4 * H)
    zf = _quiet(spec, fine).z
    zc = _quiet(spec, coarse, lattice=fine).z
    assert np.array_equal(zc, zf[0::2, 0::2][:64, :64])


def test_grid_rule_warns_with_the_numbers():
    with pytest.warns(UserWarning) as record:
        rh = rough_heights(_acf(beta=5 * H, lambda_s=2 * H), _grid(64, 64))
    messages = [str(w.message) for w in record]
    assert any("lambda_s/h = 2 < 8" in m for m in messages) and any("beta*/h_lattice = 5 < 10" in m for m in messages)
    assert not rh.grid_rule_ok and rh.lambda_s_over_h == pytest.approx(2.0) and rh.beta_over_h == pytest.approx(5.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ok = rough_heights(_acf(beta=10 * H, lambda_s=8 * H), _grid(64, 64))
    assert ok.grid_rule_ok and not ok.warnings


def test_measured_surface_is_levelled_and_resampled():
    h = H / 2
    n = 128
    x = (np.arange(n) - n // 2) * h
    X, Y = np.meshgrid(x, x, indexing="ij")
    wave = 0.1 * UM * np.cos(2 * math.pi * X / (n * h / 4))                   # 4 periods: periodic on the lattice
    spec = RoughnessSpec(kind="measured", heights=wave + 1e-3 * X + 2e-3 * Y + 1 * UM, spacing=h, lambda_s=4 * h)
    rh = _quiet(spec, _grid(32, 32, H))                                     # inside the measurement
    expected = 0.1 * UM * np.cos(2 * math.pi * _grid(32, 32, H).X / (n * h / 4))
    # the bilinear fit takes a little of the wave (x cos not orthogonal on the lattice) and the rest
    # leaks in the FFT of the non-periodic residual: within 2 % of the wave amplitude
    assert np.abs(rh.z - expected).max() < 2e-2 * 0.1 * UM


# ---- surface_pair with heights --------------------------------------------------------------------

def _ball_on_flat(load=500.0):
    return _Point(body1=SweptCircleBody(r=6.35e-3, E=210e9, nu=0.3),
                  body2=SweptCircleBody(r=math.inf, E=210e9, nu=0.3), load=load, label="ball")


def _heights(g, spec_1=None, spec_2=None, **kw):
    """(heights of body 1, heights of body 2) on g, None where no spec (grid-rule warnings silenced)."""
    return tuple(None if sp is None else _quiet(sp, g, **kw) for sp in (spec_1, spec_2))


@pytest.mark.parametrize("heights", [(None, None), "rq_0"], ids=["none", "rq_0"])
def test_zero_roughness_is_surface_pair(heights):
    contact, g = _ball_on_flat(), _grid(32, 32)
    heights = _heights(g, None, _acf(rq=0.0)) if heights == "rq_0" else heights
    a1, a2 = surface_pair(contact, g, name=f"r4_smooth_a{next(_ids)}")
    b1, b2 = surface_pair(contact, g, name=f"r4_smooth_b{next(_ids)}", heights=heights)
    for a, b in ((a1, b1), (a2, b2)):
        assert np.array_equal(np.asarray(a.profile), np.asarray(b.profile)) and a.grid_spacing == b.grid_spacing
        assert (a.material.E, a.material.v) == (b.material.E, b.material.v)


def test_rough_body_gets_profile_plus_heights():
    contact, g = _ball_on_flat(), _grid(32, 32)
    a1, a2 = surface_pair(contact, g, name=f"r4_rough_a{next(_ids)}")
    rh1, rh2 = _heights(g, None, _acf())
    b1, b2 = surface_pair(contact, g, name=f"r4_rough_b{next(_ids)}", heights=(rh1, rh2))
    assert np.array_equal(np.asarray(b1.profile), np.asarray(a1.profile)) and rh1 is None
    assert np.array_equal(np.asarray(b2.profile), np.asarray(a2.profile) + rh2.z)


def test_composite_spec_excludes_a_second_spec():
    g = _grid(16, 16)
    with pytest.raises(ValueError, match="composite"):
        surface_pair(_ball_on_flat(), g, name=f"r4_comp{next(_ids)}",
                     heights=_heights(g, _acf(composite=True), _acf(seed=2)))


@pytest.mark.parametrize("other", [(32, 32, 2 * H), (16, 16, H)], ids=["spacing", "shape"])
def test_heights_on_another_grid_are_refused(other):
    g = _grid(32, 32)
    with pytest.raises(ValueError, match="solve grid"):
        surface_pair(_ball_on_flat(), g, name=f"r4_other{next(_ids)}", heights=_heights(_grid(*other), None, _acf()))
    with pytest.raises(ValueError, match="heights must be"):
        surface_pair(_ball_on_flat(), g, name=f"r4_one{next(_ids)}", heights=_heights(g, None, _acf())[1:])


# ---- rough_measures -----------------------------------------------------------------------------

@pytest.mark.parametrize("connectivity, n_patches", [(4, 3), (8, 2)])
def test_measures_of_a_hand_made_field(connectivity, n_patches):
    g = _grid(20, 20)
    p = np.zeros((20, 20))
    p[2:5, 3:7] = 2e9                    # patch A: 12 nodes
    p[10:12, 10:12] = 1e9                # patch B: 4 nodes
    p[12, 12] = 4e9                      # touches B by a corner only
    m = rough_measures(p, g, area_reference=100 * H**2, connectivity=connectivity)
    assert m.area_real == pytest.approx(17 * H**2) and m.area_ratio == pytest.approx(0.17)
    assert m.n_patches == n_patches and m.p_max == 4e9
    assert m.p_mean_real == pytest.approx((12 * 2e9 + 4 * 1e9 + 4e9) / 17)
    assert m.threshold == pytest.approx(4e3) and m.record()["n_patches"] == n_patches


# ---- added after the R4a reviews ----------------------------------------------------------------

class _Line(Contact):
    line = True


def test_psd_spectrum_has_the_hurst_slope_and_the_plateau():
    """Radially averaged |FFT|^2 of a 'psd' surface: log-slope -2(1 + H) +- 0.2 between q_r and q_s,
    flat below q_r (q in rad/m, the convention of the spec)."""
    n, spec = 256, _psd(hurst=0.8, q_r=2 * math.pi / (32 * H), lambda_s=4 * H)
    z = _quiet(spec, _grid(n, n)).z
    q = 2 * math.pi * np.hypot(*np.meshgrid(np.fft.fftfreq(n, H), np.fft.fftfreq(n, H), indexing="ij")).ravel()
    power = (np.abs(np.fft.fft2(z)) ** 2).ravel()
    edges = np.geomspace(2 * math.pi / (n * H), 2 * math.pi / spec.lambda_s, 25)
    k = np.digitize(q, edges)
    qm = np.array([q[k == i].mean() for i in range(1, 25) if (k == i).sum() > 3])
    pm = np.array([power[k == i].mean() for i in range(1, 25) if (k == i).sum() > 3])
    law = (qm > 1.5 * spec.q_r) & (qm < 0.8 * 2 * math.pi / spec.lambda_s)
    assert np.polyfit(np.log(qm[law]), np.log(pm[law]), 1)[0] == pytest.approx(-2 * (1 + spec.hurst), abs=0.2)
    flat = qm < 0.9 * spec.q_r
    assert abs(np.polyfit(np.log(qm[flat]), np.log(pm[flat]), 1)[0]) < 0.2


def test_two_independent_rough_bodies_add_in_the_gap():
    contact, g = _ball_on_flat(), _grid(128, 128)
    a1, a2 = surface_pair(contact, g, name=f"r4_two_a{next(_ids)}")
    heights = _heights(g, _acf(beta=10 * H, lambda_s=8 * H, seed=1), _acf(beta=10 * H, lambda_s=8 * H, seed=2))
    b1, b2 = surface_pair(contact, g, name=f"r4_two_b{next(_ids)}", heights=heights)
    rough_gap = (np.asarray(a1.profile) + np.asarray(a2.profile)) - (np.asarray(b1.profile) + np.asarray(b2.profile))
    assert rough_gap.std() == pytest.approx(math.sqrt(2) * 0.2 * UM, rel=0.03)


def test_s_filter_lengthens_the_realised_correlation_length():
    """lambda_s = 0.8 beta* (probe C2): the realised 10 % length is about 1.2 beta* (physics review R4a);
    with no S-filter in effect it is beta* within 10 % (mean of 3 seeds, beta*/h 10, 256^2)."""
    beta = 10 * H
    filtered = np.mean([_quiet(_acf(beta=beta, lambda_s=0.8 * beta, seed=sd), _grid(256, 256)).beta_star_realised
                        for sd in (1, 2, 3)])
    free = np.mean([_quiet(_acf(beta=beta, lambda_s=H, seed=sd), _grid(256, 256)).beta_star_realised
                    for sd in (1, 2, 3)])
    assert free == pytest.approx(beta, rel=0.1) and 1.1 < filtered / beta < 1.3


def test_band_limit_refuses_an_empty_band():
    with pytest.raises(ValueError, match="no wave vector"):
        band_limit(np.ones((8, 8)), 1.0, 100.0)


def test_lattice_must_hold_the_grid_nodes():
    with pytest.raises(ValueError, match=r"h / 2\^k"):
        rough_heights(_acf(), _grid(16, 16, H), lattice=_grid(64, 64, 2 * H))
    with pytest.raises(ValueError, match="not nodes of the lattice"):
        rough_heights(_acf(), _grid(64, 64, H), lattice=_grid(32, 32, H / 2))


def test_measured_without_lambda_s_is_flagged():
    spec = RoughnessSpec(kind="measured", heights=np.random.default_rng(1).normal(size=(64, 64)) * UM, spacing=H)
    with pytest.warns(UserWarning, match="no lambda_s"):
        rh = rough_heights(spec, _grid(32, 32))
    assert not rh.grid_rule_ok


def test_non_square_psd_lattice_is_noted_and_refused_for_a_line():
    rh = _quiet(_psd(), _grid(64, 32))
    assert rh.z.shape == (64, 32) and any("not periodic" in w for w in rh.warnings)
    roller = _Line(body1=SweptCircleBody(r=(5e-3, math.inf), E=210e9, nu=0.3),
                   body2=SweptCircleBody(r=math.inf, E=210e9, nu=0.3), load=1e5, label="roller")
    g = _grid(64, 32)
    assert not rh.periodic_y and _quiet(_acf(), g).periodic_y
    with pytest.raises(ValueError, match="periodic along y"):
        surface_pair(roller, g, name=f"r4_line{next(_ids)}", heights=(None, rh))
    measured = RoughnessSpec(kind="measured", heights=np.zeros((80, 40)), spacing=H)
    with pytest.raises(ValueError, match="periodic along y"):
        surface_pair(roller, g, name=f"r4_line{next(_ids)}", heights=_heights(g, None, measured))
    acf_on_longer = _quiet(_acf(), g, lattice=_grid(64, 64))       # lattice period along y 64 h, the grid's 32 h
    with pytest.raises(ValueError, match="periodic along y"):
        surface_pair(roller, g, name=f"r4_line{next(_ids)}", heights=(None, acf_on_longer))


# ---- solve with heights (R4c) ------------------------------------------------------------------------

def _smoke():
    """50 N ball on a rough flat, 64^2 nodes at 7.63 um (domain about 1.9 a), spatial kernel."""
    g = _grid(64, 64, 2.0**-17)
    return _ball_on_flat(load=50.0), g, _heights(g, None, _acf(rq=0.05 * UM, beta=20e-6, lambda_s=16e-6))


def test_a_rough_pair_through_solve_equals_the_pair_driven_by_hand(tmp_path):
    """The same rough case through solve and through SlipPY ContactModel + StaticStep by hand: equal fields."""
    import slippy
    import slippy.contact as c
    contact, g, heights = _smoke()
    name = f"r4_hand{next(_ids)}"
    s1, s2 = surface_pair(contact, g, name=name, use_frequency_domain=False, heights=heights)
    saved = slippy.OUTPUT_DIR
    try:
        model = c.ContactModel(name, s1, s2, output_dir=str(tmp_path))
        model.add_step(c.StaticStep("static", method="pk", tolerance=1e-9, normal_load=contact.load))
        sol = model.solve()
    finally:
        slippy.OUTPUT_DIR = saved
    r = solve(contact, g, tmp_path, use_frequency_domain=False, heights=heights)
    assert sol["converged"] and r.converged
    assert np.array_equal(r.p, np.asarray(sol["loads_z"])) and r.delta == pytest.approx(float(sol["interference"]))
    assert np.array_equal(r.z2, np.asarray(s2.profile))
    m = rough_measures(r.p, r.grid, area_reference=1.0, surfaces=r.heights)
    assert m.load == pytest.approx(50.0, rel=1e-3) and m.n_patches >= 1 and m.p_max == r.p0


def test_rough_solve_does_not_apply_the_hertz_shape_measures(tmp_path):
    contact, g, heights = _smoke()
    r = solve(contact, g, tmp_path, use_frequency_domain=False, heights=heights)
    assert r.build == "rough" and r.heights == heights
    assert all(math.isnan(v) for v in (r.a, r.b, r.apex_offset_nodes, r.err_p0, r.err_a, r.err_b, r.err_delta))
    assert r.within_reference is None and r.load_ok and r.edge_clear and r.p0 == r.p.max()
    # SlipPY's own shift of the gap to first touch: -(z1 + z2) - gap0 is one constant m on every node, and the
    # mean-plane approach is delta + m (checks the _model_utils assumption, not the formula)
    shift = -(r.z1 + r.z2) - r.gap0
    assert np.ptp(shift) < 1e-15 and r.delta_mean_planes == pytest.approx(r.delta + shift.mean(), abs=1e-15)
    assert r.delta_mean_planes < r.delta                       # an asperity rises above the apex datum
    rec = r.record()
    assert rec["build"] == "rough" and rec["rough1_kind"] is None and rec["rough2_kind"] == "acf"
    assert rec["rough2_rq_grid_m"] == heights[1].rq_grid and rec["rough2_seed"] == 1
    assert {"rough2_grid_rule_ok", "rough2_warnings", "rough2_lambda_s_over_h", "delta_mean_planes_m"} <= rec.keys()
    assert rec["alpha_gt"] == pytest.approx(heights[1].rq_grid / r.hertz.delta)
    assert rec["heights_mean_m"] == pytest.approx(heights[1].z.mean())


def test_rough_solve_of_zero_heights_is_the_smooth_solve(tmp_path):
    """Limit Rq -> 0: heights all zero give the smooth pressure field, and the mean-plane approach is the smooth
    approach (apex on a node); the build is still "rough" (a spec was given)."""
    contact, g, _ = _smoke()
    r0 = solve(contact, g, tmp_path, use_frequency_domain=False)
    r = solve(contact, g, tmp_path, use_frequency_domain=False, heights=_heights(g, None, _acf(rq=0.0)))
    assert r.build == "rough" and np.array_equal(r.p, r0.p)
    assert r.delta_mean_planes == r0.delta and math.isnan(r.a) and r.within_reference is None


def test_smooth_solve_is_unchanged_by_the_heights_argument(tmp_path):
    """heights None (the default) and (None, None): the fields, the measures and the record of a solve without it."""
    contact, g, _ = _smoke()
    r0 = solve(contact, g, tmp_path, use_frequency_domain=False)
    for heights in (None, (None, None)):
        r = solve(contact, g, tmp_path, use_frequency_domain=False, heights=heights)
        assert r.build == "two_profiles" and r.heights == (None, None) and math.isnan(r.delta_mean_planes)
        for k in ("p", "z1", "z2", "gap0", "total_displacement_z"):
            assert np.array_equal(getattr(r, k), getattr(r0, k)), k
        a, b = r0.record(), r.record()
        a.pop("solve_time_s"), b.pop("solve_time_s")
        assert a.keys() == b.keys() and all(a[k] == b[k] or (a[k] != a[k] and b[k] != b[k]) for k in a)
        assert not any(k.startswith("rough") for k in b)


def test_rough_solve_under_interference_control(tmp_path):
    contact, g, heights = _smoke()
    r = solve(contact, g, tmp_path, use_frequency_domain=False, heights=heights, interference=2e-6)
    assert r.load_control == "interference" and r.build == "rough" and r.load_found > 0
    assert r.load == pytest.approx(r.load_found, rel=1e-6) and r.within_reference is None
    assert r.delta == pytest.approx(2e-6, rel=1e-6)


def test_solve_refuses_interpolation_other_than_nearest_with_heights(tmp_path):
    contact, g, heights = _smoke()
    with pytest.raises(ValueError, match="R-051"):
        solve(contact, g, tmp_path, use_frequency_domain=False, heights=heights, profile_interpolation_mode="linear")


def test_solve_refuses_heights_on_another_grid(tmp_path):
    contact, g, _ = _smoke()
    with pytest.raises(ValueError, match="solve grid"):
        solve(contact, g, tmp_path, use_frequency_domain=False, heights=_heights(_grid(32, 32, 2.0**-16), None, _acf()))
