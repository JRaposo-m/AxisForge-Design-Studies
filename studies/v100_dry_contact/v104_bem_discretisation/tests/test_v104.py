"""Fast tests of v104: the systems, one discretisation record, the domain independence of the spatial kernel."""
import math

from slippy_bridge.contact import hertz

from studies.v100_dry_contact.v104_bem_discretisation import check, literature, systems
from studies.v100_dry_contact.v104_bem_discretisation.study import _solve


def test_the_systems_are_a_circle_and_an_ellipse_on_the_grid_lines():
    circle, ellipse = systems()
    assert math.isclose(circle.kappa_max, circle.kappa_min) and ellipse.kappa_max > 4.9 * ellipse.kappa_min
    assert circle.psi == ellipse.psi == 0.0


def test_one_record_and_the_spatial_kernel_does_not_see_the_domain(tmp_path):
    circle = systems()[0]
    small, large = (check(_solve(circle, hertz(circle), 8.0, span, False, tmp_path)) for span in (1.25, 4.0))
    assert small["kernel"] == "spatial" and 8.0 <= small["b_over_h"] < 16.0
    for key in ("err_delta", "err_p0", "eps_r", "eps_p", "err_area_fit"):
        assert math.isfinite(small[key])
    assert abs(small["err_delta"]) < 5e-3 and abs(small["err_p0"]) < 5e-3
    assert math.isclose(small["delta_m"], large["delta_m"], rel_tol=5e-4)
    assert math.isclose(small["p0_Pa"], large["p0_Pa"], rel_tol=5e-4)


def test_literature_rows_have_pages():
    pk, muser = literature.cases()
    assert pk.published and muser.published
    assert all(r["pdf_page"] for r in pk.published + muser.published)
