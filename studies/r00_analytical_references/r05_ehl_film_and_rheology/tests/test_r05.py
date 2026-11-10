"""r05_ehl_film_and_rheology (PROVISIONAL scaffold): the judge alone; SlipPY only in the Dowson-Higginson unit check.

Split from tests/test_contact_reference.py when its Hertz part moved to r02_hertz_contact (R1); moved here with
ehl.py in the sweep (SW3, closes MG-01). The content of the tests is unchanged; R3 extends it.
"""
import numpy as np
import pytest

from studies.r00_analytical_references.r05_ehl_film_and_rheology import canzi_moes, e_prime, groups, moes

# two materials (E [Pa], nu [-]): pairing nu of one body with E of the other would show
MIXED = dict(E=(210e9, 70e9), nu=(0.3, 0.33))


def _e_star(E, nu):
    """Contact modulus [Pa], ``1/E* = (1 - nu1^2)/E1 + (1 - nu2^2)/E2``."""
    (E1, E2), (nu1, nu2) = E, nu
    return 1.0 / ((1.0 - nu1**2) / E1 + (1.0 - nu2**2) / E2)


# ---- studies.r00_analytical_references.r05_ehl_film_and_rheology --------------------------------------------

def test_e_prime_definition():
    """E' = 2 / [(1 - nu1^2)/E1 + (1 - nu2^2)/E2] = 2 E*; for two equal steels E' = E / (1 - nu^2)."""
    out = e_prime(210e9, 0.3, 210e9, 0.3)
    assert out["E_prime_Pa"] == pytest.approx(210e9 / (1.0 - 0.3**2), rel=1e-15)
    assert out["E_prime_Pa"] == pytest.approx(2.0 * out["E_star_Pa"], rel=1e-15)
    assert e_prime(**{"E1_Pa": 210e9, "nu1": 0.3, "E2_Pa": 70e9, "nu2": 0.33})["E_star_Pa"] == pytest.approx(
        _e_star(**MIXED), rel=1e-15)                                         # E* is the Hertz modulus


@pytest.mark.parametrize("loads", [{}, {"load_N": 500.0, "w_N_per_m": 1e5}])
def test_groups_needs_one_load(loads):
    with pytest.raises(ValueError, match="exactly one"):
        groups(u_mean_m_per_s=2.0, eta0_Pa_s=0.1, E_prime_Pa=2.3e11, R_x_m=6.35e-3, alpha_per_Pa=2e-8, **loads)


@pytest.mark.parametrize("bad", [{"E1_Pa": 0.0}, {"nu2": 0.0}, {"nu1": 0.5}])
def test_e_prime_bad_input_raises(bad):
    with pytest.raises(ValueError):
        e_prime(**{"E1_Pa": 210e9, "nu1": 0.3, "E2_Pa": 210e9, "nu2": 0.3, **bad})


@pytest.mark.parametrize("key", ["load_N", "u_mean_m_per_s", "R_x_m", "eta0_Pa_s", "alpha_per_Pa"])
def test_groups_refuses_non_positive_input(key):
    kwargs = dict(u_mean_m_per_s=2.0, eta0_Pa_s=0.1, E_prime_Pa=2.3e11, R_x_m=6.35e-3, alpha_per_Pa=2e-8,
                  load_N=500.0)
    with pytest.raises(ValueError, match=key):
        groups(**{**kwargs, key: 0.0})


def test_moes_refuses_p_h_in_MPa():
    with pytest.raises(ValueError, match="in MPa"):
        moes(M=20.0, L=10.0, line=True, compressible=True, p_h_Pa=500.0)    # 500 MPa given as 500


@pytest.mark.xfail(strict=True, raises=NotImplementedError, reason="scaffold OPEN R-114 GL-13 GL-12")
@pytest.mark.parametrize("line, alpha_bar, M_canzi, H_c_printed", [
    (True, 20.0, 17.97, 4.77),      # Canzi2010 Table 2 "Hc Moes formula": line_contact CSV L166 (L160, L163)
    (False, 20.0, 100.0, 4.05),     # Canzi2010 Table 3 "Hc Moes formula": canzi_circular_extra CSV L10 (L3, L6)
], ids=["line", "circular"])
def test_moes_canzi_formula(line, alpha_bar, M_canzi, H_c_printed):
    """Moes' film formula, compressible-corrected (Canzi2010 Eq. 4 line, Eq. 15 circular, Eq. 11), as printed.

    The point is given in Canzi's (alpha_bar, M) and mapped to Moes (M, L) by Eq. 3 / Eq. 14. The
    compressibility correction needs a dimensional p_h that Canzi does not print (R-114, GL-13).
    """
    p_h_assumed_Pa = None           # OPEN R-114: the assumed p_h is written before the body is filled
    point = canzi_moes(alpha_bar=alpha_bar, M_canzi=M_canzi, line=line)
    H = moes(M=point["M_moes"], L=point["L_moes"], line=line, compressible=True, p_h_Pa=p_h_assumed_Pa)
    assert H["H_c"] == pytest.approx(H_c_printed, abs=0.005)


def test_dowson_higginson_asymptote_1p34():
    """SlipPY's Dowson-Higginson law, rho/rho0 = (5.9e8 + 1.34 p)/(5.9e8 + p) with p in Pa, tends to 1.34.

    The constant 5.9e8 is in Pa inside ``nd_dowson_higginson`` (lubricant_models.py:216), so the Hertz
    pressure that scales P = p/p_h must be given in Pa: a value in MPa leaves the oil almost
    incompressible (R-077; ruled note 2026-10-16).
    """
    lm = pytest.importorskip("slippy.contact.lubricant_models")
    rho = lm.nd_dowson_higginson(1.0e9)                                # p_h = 1 GPa, in Pa
    assert float(rho(np.array([0.0]))[0]) == pytest.approx(1.0, rel=1e-12)            # ambient pressure
    assert float(rho(np.array([1.0e6]))[0]) == pytest.approx(1.34, rel=1e-5)          # p = 1e15 Pa: the asymptote
    assert float(rho(np.array([1.0]))[0]) == pytest.approx((5.9e8 + 1.34e9) / (5.9e8 + 1.0e9), rel=1e-12)
    # the same 1 GPa written in MPa (p_h = 1000) gives rho/rho0 ~ 1 at P = 1: compressibility lost
    assert float(lm.nd_dowson_higginson(1.0e3)(np.array([1.0]))[0]) == pytest.approx(1.0, abs=1e-6)
