"""slippy_bridge.contact.roughness.micro: the periodic rough patch (constructions only)."""
import itertools
import math
import warnings

import numpy as np
import pytest

pytest.importorskip("slippy")

from slippy_bridge.contact.mesh.grid import Grid  # noqa: E402
from slippy_bridge.contact.roughness.macro import RoughnessSpec, rough_heights  # noqa: E402
from slippy_bridge.contact.roughness.micro import PatchSpec, periodic_patch  # noqa: E402

H = 2.0**-20                                     # 0.95 um [m]
UM = 1e-6
_ids = itertools.count()                         # SlipPY material names are process-wide


def _patch(roughness, n=128, **kw):
    return PatchSpec(length=n * H, n=n, roughness=roughness, E_1=210e9, nu_1=0.3, label=f"r4_patch{next(_ids)}",
                     **kw)


_ACF = RoughnessSpec(kind="acf", rq=0.1 * UM, beta_star=10 * H, lambda_s=8 * H, seed=1)
_PSD = RoughnessSpec(kind="psd", rq=0.1 * UM, hurst=0.8, q_r=2 * math.pi / (16 * H), lambda_s=8 * H, seed=1)


def _quiet(spec):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return periodic_patch(spec)


def _wrap_inside_row_spread(z, axis):
    """True when the rms jump across the wrap lies within the min-max of the rms jumps between
    the other neighbouring rows (one row is a sample of the same statistic: a periodic surface)."""
    rows = np.sqrt(np.mean(np.diff(z, axis=axis) ** 2, axis=1 - axis))
    wrap = np.sqrt(np.mean((np.take(z, 0, axis) - np.take(z, -1, axis)) ** 2))
    return rows.min() <= wrap <= rows.max()


@pytest.mark.parametrize("rough", [_ACF, _PSD], ids=["acf", "psd"])
def test_heights_are_periodic_in_both_axes(rough):
    z = _quiet(_patch(rough))[2].z
    assert _wrap_inside_row_spread(z, 0) and _wrap_inside_row_spread(z, 1)


def test_periodicity_check_fails_on_a_cropped_surface():
    """Negative control: a 128^2 window of a 256^2 periodic surface is not periodic on the window."""
    x = (np.arange(256) - 128) * H
    X, Y = np.meshgrid(x, x, indexing="ij")
    w = (np.arange(128) - 64) * H
    Xw, Yw = np.meshgrid(w, w, indexing="ij")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        z = rough_heights(_ACF, Grid(X=Xw, Y=Yw, h=H), lattice=Grid(X=X, Y=Y, h=H)).z
    assert not (_wrap_inside_row_spread(z, 0) and _wrap_inside_row_spread(z, 1))


def test_rq_exact_and_flat_counter_face():
    s1, s2, rh = _quiet(_patch(_PSD))
    assert np.asarray(s1.profile).std() == pytest.approx(_PSD.rq, rel=1e-12) and rh.rq_grid == pytest.approx(_PSD.rq)
    assert not np.any(np.asarray(s2.profile)) and s1.grid_spacing == s2.grid_spacing == H


def test_both_materials_have_zero_frequency_value_zero_and_the_defaults():
    s1, s2, _ = _quiet(_patch(_ACF))
    for surface in (s1, s2):
        assert surface.material.zero_frequency_value == 0 and surface.material.E == pytest.approx(210e9)
    spec = _patch(_ACF, E_2=200e9, nu_2=0.29)
    assert _quiet(spec)[1].material.E == pytest.approx(200e9)
    assert spec.e_star == pytest.approx(1 / ((1 - 0.09) / 210e9 + (1 - 0.29**2) / 200e9))
    assert _patch(_ACF).e_star == pytest.approx(210e9 / (2 * (1 - 0.09)))


@pytest.mark.parametrize("rough, lam", [
    (_ACF, 2 * math.pi * 10 * H / 2.3),
    (_PSD, 16 * H),
    (RoughnessSpec(kind="acf", rq=0.1 * UM, beta_star=10 * H, lambda_s=8 * H, lambda_c=20 * H, seed=1), 20 * H),
    (RoughnessSpec(kind="psd", rq=0.1 * UM, hurst=0.8, q_r=0.0, lambda_s=8 * H, seed=1), 128 * H),
], ids=["acf_knee", "psd_roll_off", "lambda_c", "psd_no_roll_off"])
def test_largest_wavelength_of_each_kind(rough, lam):
    spec = _patch(rough)
    assert spec.lambda_largest == pytest.approx(lam) and spec.l_over_lambda == pytest.approx(128 * H / lam)


def test_rve_rule_warns_with_the_numbers():
    with pytest.warns(UserWarning, match=r"L / lambda_largest = 4.69 < 5"):
        periodic_patch(_patch(_ACF))                     # L / (2 pi beta* / 2.3) = 4.69
    with pytest.warns(UserWarning, match=r"L / lambda_largest = 1 < 5"):    # a pure power law always warns
        periodic_patch(_patch(RoughnessSpec(kind="psd", rq=0.1 * UM, hurst=0.8, q_r=0.0, lambda_s=8 * H, seed=1)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        periodic_patch(_patch(_PSD))                     # L / lambda 8, lambda_s / h 8


@pytest.mark.parametrize("kw, match", [
    ({"roughness": RoughnessSpec(kind="measured", heights=np.zeros((8, 8)), spacing=H)}, "not periodic"),
    ({"n": 7}, "even int"),
    ({"pressure_range": (2e6, 1e6)}, "pressure_range"),
    ({"nu_2": 0.5}, "nu_2"),
])
def test_invalid_specs_are_refused(kw, match):
    base = {"length": 128 * H, "n": 128, "roughness": _PSD, "E_1": 210e9, "nu_1": 0.3, "label": f"r4_bad{next(_ids)}"}
    with pytest.raises(ValueError, match=match):
        periodic_patch(PatchSpec(**{**base, **kw}))


def test_a_non_dyadic_spacing_is_refused_by_the_grid():
    with pytest.raises(ValueError, match="power of two"):
        periodic_patch(PatchSpec(length=1e-4, n=128, roughness=_PSD, E_1=210e9, nu_1=0.3, label=f"r4_h{next(_ids)}"))
