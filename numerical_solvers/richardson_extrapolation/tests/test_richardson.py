"""Tests of the Richardson extrapolation / GCI solver ``numerical_solvers.richardson_extrapolation``.

Analytic models ``f(h) = f0 + C h^p`` have the exact observed order ``p`` and the exact extrapolated
value ``f0`` on any triple with a constant ratio, so both are recovered to round-off (1e-10). The
statuses other than monotonic are each reached by a triple chosen for it. The cross-check against
AxisForge ``RichardsonGCI`` (skipped without AxisForge) runs the same values with node lists of
``n_c, r n_c, r^2 n_c`` elements. Richardson (1911), Roache (1998) and Celik et al. (2008) are not in
the library: no reference value is quoted from them.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import numerical_solvers
from numerical_solvers import (LevelRecord, MetricConvergence, RichardsonError, RichardsonOptions,
                               RichardsonResult, richardson, richardson_triple)

_ALL = dict(stop_when_converged=False)


def _model(f0: float, C: float, p: float):
    """``f(h) = f0 + C h^p`` [units of f]."""
    return lambda h: f0 + C * h**p


def _check_ids(r: RichardsonResult, name: str, label: str) -> None:
    """Name and label on the result, its summary and every row; one row per level and metric."""
    assert (r.name, r.label) == (name, label)
    assert (r.record()["name"], r.record()["label"]) == (name, label)
    rows = r.records()
    assert len(rows) == sum(len(h.metrics) for h in r.history)
    assert all(row["name"] == name and row["label"] == label for row in rows)
    assert len({tuple(row) for row in rows}) == 1                 # fixed columns
    assert r.n_evaluations == len(r.history) == len(r.levels)


# ---- analytic models: p and f0 recovered ----------------------------------------------------
@pytest.mark.parametrize("p", [1.0, 2.0, 4.0])
@pytest.mark.parametrize("ratio", [2.0, 1.5])
def test_analytic_order_and_limit_size(p, ratio):
    """Sizes h = 1, 1/r, 1/r^2, 1/r^3: every triple gives p and f0 to 1e-10."""
    levels = [ratio ** -k for k in range(4)]
    r = richardson(_model(3.0, 0.7, p), levels, options=RichardsonOptions(**_ALL),
                   name="analytic", label=f"p {p} r {ratio}")
    _check_ids(r, "analytic", f"p {p} r {ratio}")
    assert len(r.convergence["value"]) == 2
    for c in r.convergence["value"]:
        assert c.status == "monotonic" and c.reason == ""
        assert c.p == pytest.approx(p, abs=1e-10)
        assert c.f_extrapolated == pytest.approx(3.0, abs=1e-10)
        assert c.r_fine_medium == pytest.approx(ratio, rel=1e-12)
    assert [h.index for h in r.history] == [0, 1, 2, 3] and math.isnan(r.history[0].ratio)


def test_analytic_several_metrics_count():
    """Counts n = 8, 12, 18, 27 (r = 1.5): three metrics with orders 1, 2 and 4, each to 1e-10."""
    def evaluate(n):
        h = 1.0 / n
        return {"p0": 10.0 + 4.0 * h, "area": 2.0 - 30.0 * h**2, "delta": 0.5 + 900.0 * h**4}

    r = richardson(evaluate, [8, 12, 18, 27], options=RichardsonOptions(kind="count", **_ALL),
                   name="three", label="count")
    _check_ids(r, "three", "count")
    for m, (p, f0) in {"p0": (1.0, 10.0), "area": (2.0, 2.0), "delta": (4.0, 0.5)}.items():
        assert r.final[m].metric == m and r.final[m].fine_index == 3
        assert r.final[m].p == pytest.approx(p, abs=1e-10)
        assert r.final[m].f_extrapolated == pytest.approx(f0, abs=1e-10)
    assert [h.ratio for h in r.history[1:]] == [1.5, 1.5, 1.5]


def test_kind_size_and_count_give_the_same_triple():
    """h = 1/n as a size and n as a count give the same ratios and numbers."""
    f = _model(1.0, 2.0, 2.0)
    by_size = richardson(f, [1 / 4, 1 / 8, 1 / 16], options=RichardsonOptions(**_ALL))
    by_count = richardson(lambda n: f(1.0 / n), [4, 8, 16],
                          options=RichardsonOptions(kind="count", **_ALL))
    a, b = by_size.final["value"], by_count.final["value"]
    assert (a.r_fine_medium, a.p, a.f_extrapolated, a.gci_fine_medium) == \
        (b.r_fine_medium, b.p, b.f_extrapolated, b.gci_fine_medium)


def test_triple_formulas():
    """110, 102.5, 100.625 with r = 2 (f = 100 + 10 h^2): p 2, f_h0 100, the GCI of Roache."""
    c = richardson_triple(110.0, 102.5, 100.625, 2.0)
    e_mc, e_fm = (102.5 - 110.0) / 110.0, (100.625 - 102.5) / 102.5
    assert c.p == pytest.approx(2.0, abs=1e-12) and c.f_extrapolated == pytest.approx(100.0)
    assert (c.e_medium_coarse, c.e_fine_medium) == (e_mc, e_fm)
    assert c.gci_medium_coarse == pytest.approx(1.25 * abs(e_mc) / 3.0, rel=1e-12)
    assert c.gci_fine_medium == pytest.approx(1.25 * abs(e_fm) / 3.0, rel=1e-12)
    assert not c.converged                      # GCI_mc = 2.8 % > 1 %
    assert c.metric == "" and c.fine_index == -1


def test_clamp_of_p():
    """min_p / max_p clamp p before the extrapolation."""
    lo = richardson_triple(110.0, 102.5, 100.625, 2.0, options=RichardsonOptions(max_p=1.5))
    hi = richardson_triple(110.0, 102.5, 100.625, 2.0, options=RichardsonOptions(min_p=3.0))
    assert (lo.p, hi.p) == (1.5, 3.0)
    assert lo.f_extrapolated == pytest.approx(100.625 - 1.875 / (2.0**1.5 - 1.0))


# ---- adaptive refinement and stopping ------------------------------------------------------
def test_adaptive_stops_when_converged():
    """f = 1 + h^2, h = 2^-k, threshold 1 %: GCI_mc = 1.25 h_m^2 / (1 + 4 h_m^2) < 0.01 first for
    h_m = 1/16 (k = 4), so the run stops after the level k = 5: 6 evaluations."""
    calls: list[float] = []

    def evaluate(h):
        calls.append(h)
        return 1.0 + h**2

    r = richardson(evaluate, lambda k: 2.0 ** -k, options=RichardsonOptions(max_levels=20),
                   name="adaptive", label="1 + h^2")
    _check_ids(r, "adaptive", "1 + h^2")
    assert r.converged and r.reason == "converged" and r.n_evaluations == 6
    assert calls == [2.0 ** -k for k in range(6)]
    conv = r.convergence["value"]
    assert conv[-1].converged and not conv[-2].converged
    assert conv[-1].f_extrapolated == pytest.approx(1.0, abs=1e-12)


def test_adaptive_max_levels_and_sequence_exhausted():
    """Without convergence the run stops at max_levels (callable) or at the end of the sequence."""
    tight = RichardsonOptions(threshold=1e-12, max_levels=5)
    r = richardson(_model(1.0, 1.0, 1.0), lambda k: 2.0 ** -k, options=tight)
    assert not r.converged and r.reason == "max_levels" and r.n_evaluations == 5
    r = richardson(_model(1.0, 1.0, 1.0), [1.0, 0.5, 0.25, 0.125],
                   options=RichardsonOptions(threshold=1e-12))
    assert not r.converged and r.reason == "levels_exhausted" and r.n_evaluations == 4
    r = richardson(_model(1.0, 1.0, 1.0), [1.0, 0.5, 0.25, 0.125],
                   options=RichardsonOptions(threshold=1e-12, max_levels=3))
    assert r.reason == "max_levels" and r.n_evaluations == 3


def test_min_levels_and_judged_metrics():
    """A flat metric converges on the first triple; min_levels 4 still evaluates a fourth level.
    Only the judged metric decides; the other is recorded."""
    evaluate = lambda h: {"flat": 5.0, "slow": 1.0 + h}
    r = richardson(evaluate, lambda k: 2.0 ** -k,
                   options=RichardsonOptions(max_levels=8, min_levels=4, metrics=["flat"]))
    assert r.converged and r.n_evaluations == 4
    assert set(r.convergence) == {"flat"} and r.final["flat"].status == "flat"
    assert all("slow" in h.metrics for h in r.history)
    assert math.isnan(r.final["flat"].p) and r.final["flat"].f_extrapolated == 5.0


# ---- statuses other than monotonic ---------------------------------------------------------
@pytest.mark.parametrize("f, r, status, words", [
    ((100.0, 110.0, 105.0), 2.0, "oscillatory", "change sign"),
    ((110.0, 105.0, 100.0), 2.0, "non-convergent", "p=0"),            # equal differences
    ((110.0, 109.0, 105.0), 2.0, "non-convergent", "do not decrease"),
    ((110.0, 102.5, 100.625), 1.0, "invalid", "must be > 1"),
    ((110.0, 102.5, 100.625), (2.0, 2.5), "invalid", "unequal refinement ratios"),
    ((110.0, 100.0, 100.0), 2.0, "invalid", "f_medium == f_fine"),
    ((0.0, 7.5, 9.375), 2.0, "invalid", "zero coarser value"),
    ((110.0, float("nan"), 100.0), 2.0, "invalid", "not finite"),
], ids=["oscillatory", "p=0", "p<0", "r<=1", "unequal r", "f_m=f_f", "f_c=0", "nan"])
def test_failed_status_has_reason(f, r, status, words):
    """Each failed status: NaN numbers, not converged, a reason (AxisForge gives none)."""
    c = richardson_triple(*f, r)
    assert c.status == status and not c.converged and words in c.reason
    assert all(math.isnan(v) for v in (c.p, c.f_extrapolated, c.e_medium_coarse, c.e_fine_medium,
                                       c.gci_medium_coarse, c.gci_fine_medium))
    assert (c.f_coarse, c.r_medium_coarse) == (f[0], r if isinstance(r, float) else r[0])


def test_degenerate_r_power_p():
    """max_p clamps p so small that r^p - 1 ~ 0: non-convergent."""
    c = richardson_triple(110.0, 102.5, 100.625, 2.0, options=RichardsonOptions(max_p=1e-14))
    assert c.status == "non-convergent" and "r^p - 1" in c.reason


def test_flat_status():
    """Differences below max(1e-9, 1e-8 max(|f_c|, |f_m|, |f_f|, 1)): flat, converged, GCI 0."""
    c = richardson_triple(50.0, 50.0 + 1e-7, 50.0, 2.0)
    assert c.status == "flat" and c.converged and c.gci_fine_medium == 0.0 and c.reason == ""


def test_flat_tolerance_in_the_units_of_the_metric():
    """An approach of 1.2e-5 m with steps of 1.9e-9 and 7.6e-10 m is flat with the AxisForge default
    (|df| < 1e-8 in metres) and monotonic with flat_atol = 0, as the same values in mm are."""
    f = (1.2e-5 + 1.9e-9 + 7.6e-10, 1.2e-5 + 7.6e-10, 1.2e-5)
    assert richardson_triple(*f, 2.0).status == "flat"
    si = richardson_triple(*f, 2.0, options=RichardsonOptions(flat_atol=0.0))
    mm = richardson_triple(*(v * 1e3 for v in f), 2.0)
    assert si.status == mm.status == "monotonic" and si.p == pytest.approx(mm.p, rel=1e-6)


def test_failed_triple_does_not_stop_the_run():
    """An oscillating start is recorded; the run goes on and converges on a later triple."""
    values = iter([1.0, 1.2, 1.1, 1.000001, 1.0000005, 1.00000025])
    r = richardson(lambda h: next(values), lambda k: 2.0 ** -k, options=RichardsonOptions(max_levels=6))
    statuses = [c.status for c in r.convergence["value"]]
    assert statuses[0] == "oscillatory" and r.converged and r.final["value"].status == "monotonic"


# ---- on_failure, callback, errors ----------------------------------------------------------
def test_on_failure_return_and_raise():
    """return: converged False with the reason; raise: RichardsonError carrying the result."""
    seq = [1.0, 0.5, 0.25]
    oscillating = lambda h: {1.0: 100.0, 0.5: 110.0, 0.25: 105.0}[h]
    r = richardson(oscillating, seq, name="osc", label="return")
    assert not r.converged and r.reason == "levels_exhausted"
    assert r.final["value"].status == "oscillatory"
    with pytest.raises(RichardsonError, match="'osc' 'raise'.*value oscillatory") as err:
        richardson(oscillating, seq, options=RichardsonOptions(on_failure="raise"),
                   name="osc", label="raise")
    assert isinstance(err.value.result, RichardsonResult) and not err.value.result.converged


def test_callback_once_per_level_and_stop():
    """The callback sees every LevelRecord once, in order; True stops the run (not converged)."""
    seen: list[LevelRecord] = []
    r = richardson(_model(1.0, 1.0, 2.0), lambda k: 2.0 ** -k,
                   options=RichardsonOptions(max_levels=10, threshold=1e-12),
                   callback=lambda rec: seen.append(rec) or rec.index == 3)
    assert [s.index for s in seen] == [0, 1, 2, 3] and seen[-1] is r.history[-1]
    assert not r.converged and r.reason == "callback" and r.n_evaluations == 4
    with pytest.raises(RichardsonError, match="callback"):
        richardson(_model(1.0, 1.0, 2.0), lambda k: 2.0 ** -k,
                   options=RichardsonOptions(max_levels=10, threshold=1e-12, on_failure="raise"),
                   callback=lambda rec: rec.index == 3)


def test_input_errors():
    """Too few levels, adaptive without max_levels, changing names, missing judged metric, options."""
    with pytest.raises(ValueError, match="min_levels = 3"):
        richardson(lambda h: h, [1.0, 0.5])
    with pytest.raises(ValueError, match="min_levels = 5"):
        richardson(lambda h: h, [0.2, 0.1, 0.05], options=RichardsonOptions(min_levels=5))
    with pytest.raises(ValueError, match="max_levels"):
        richardson(lambda h: h, lambda k: 2.0 ** -k)
    names = iter([{"a": 1.0}, {"b": 1.0}])
    with pytest.raises(ValueError, match="evaluate returned"):
        richardson(lambda h: next(names), [1.0, 0.5, 0.25])
    with pytest.raises(ValueError, match="not returned"):
        richardson(lambda h: {"a": h}, [1.0, 0.5, 0.25], options=RichardsonOptions(metrics=["b"]))
    with pytest.raises(ValueError, match="not > 1"):
        richardson(lambda n: 1.0 / n, [8, 16, 32])                  # counts given as sizes
    with pytest.raises(ValueError, match="positive"):
        richardson(lambda h: h, [1.0, 0.5, 0.0])
    with pytest.raises(ValueError, match="kind"):
        RichardsonOptions(kind="ratio")
    with pytest.raises(ValueError, match="min_levels"):
        RichardsonOptions(min_levels=2)
    assert RichardsonOptions().validate() == []
    assert RichardsonOptions(metrics=["a"]).metrics == ("a",)


def test_records_and_summary():
    """records(): one row per level and metric, the triple fields from the third level on."""
    r = richardson(lambda h: {"a": 1.0 + h, "b": 2.0 + h**2}, [1.0, 0.5, 0.25, 0.125],
                   options=RichardsonOptions(**_ALL), name="rows", label="two")
    rows = r.records()
    assert len(rows) == 8 and [row["metric"] for row in rows[:2]] == ["a", "b"]
    assert all(math.isnan(row["p"]) and row["status"] == "" for row in rows[:4])
    assert [row["status"] for row in rows[4:]] == ["monotonic"] * 4
    assert rows[-1]["p"] == pytest.approx(2.0) and rows[-1]["index"] == 3
    s = r.record()
    assert s["n_levels"] == 4 and s["b:p"] == pytest.approx(2.0) and s["a:status"] == "monotonic"


def test_reexports():
    """The public names are reachable from the package rollups."""
    assert numerical_solvers.richardson is richardson
    assert numerical_solvers.richardson_extrapolation.richardson is richardson
    assert numerical_solvers.richardson_triple is richardson_triple
    assert isinstance(richardson_triple(3.0, 2.0, 1.5, 2.0), MetricConvergence)


# ---- cross-check against AxisForge RichardsonGCI ----------------------------------------
_NO_AF = "AxisForge (axisforge.solvers.mesh.convergence_solver) not installed"
_TRIPLES = [   # (f_c, f_m, f_f, r, n_c, min_p, max_p)
    (110.0, 102.5, 100.625, 2, 2, None, None),
    (10.0 + 4 / 8, 10.0 + 4 / 12, 10.0 + 4 / 18, 1.5, 8, None, None),
    (2.0 - 30 / 64, 2.0 - 30 / 144, 2.0 - 30 / 324, 1.5, 8, None, None),
    (1.0 + 2.0**-4, 1.0 + 2.0**-6, 1.0 + 2.0**-8, 2, 4, None, None),     # converged
    (110.0, 102.5, 100.625, 2, 2, None, 1.5),
    (110.0, 102.5, 100.625, 2, 2, 3.0, None),
    (50.0, 50.0, 50.0, 2, 2, None, None),                                 # flat
    (5.0, 4.0, 3.9, 3, 3, None, None),
]


@pytest.mark.parametrize("f_c, f_m, f_f, r, n_c, min_p, max_p", _TRIPLES)
def test_triple_matches_axisforge(f_c, f_m, f_f, r, n_c, min_p, max_p):
    """Same inputs, same p, f_h0, both GCI and converged as AxisForge RichardsonGCI (node lists of
    n_c, r n_c, r^2 n_c elements); skipped without AxisForge."""
    cs = pytest.importorskip("axisforge.solvers.mesh.convergence_solver", reason=_NO_AF)
    nodes = [list(range(int(round(n_c * r**i)) + 1)) for i in range(3)]
    af = cs.RichardsonGCI(f_c, f_m, f_f, *nodes, gci_threshold=0.01, safety_factor=1.25,
                          min_p=min_p, max_p=max_p)
    c = richardson_triple(f_c, f_m, f_f, (af.r_m_c, af.r_f_m),
                          options=RichardsonOptions(min_p=min_p, max_p=max_p))
    same = lambda a, b: a == b or (math.isnan(a) and math.isnan(b))
    assert same(c.p, af.p) and c.f_extrapolated == af.f_h0
    assert (c.e_medium_coarse, c.e_fine_medium) == (af.e_m_c, af.e_f_m)
    assert (c.gci_medium_coarse, c.gci_fine_medium) == (af.GCI_m_c, af.GCI_f_m)
    assert c.converged == af.converged


@pytest.mark.parametrize("f", [(100.0, 110.0, 105.0), (110.0, 100.0, 100.0), (110.0, 105.0, 100.0),
                               (0.0, 7.5, 9.375)])
def test_failed_triple_matches_axisforge_error(f):
    """Where AxisForge RichardsonGCI raises (and _compute_gci returns a _DummyGCI), the port returns
    NaN GCI, not converged, with a status and a reason."""
    cs = pytest.importorskip("axisforge.solvers.mesh.convergence_solver", reason=_NO_AF)
    nodes = [list(range(3)), list(range(5)), list(range(9))]
    with pytest.raises(ValueError):
        cs.RichardsonGCI(*f, *nodes)
    c = richardson_triple(*f, 2.0)
    assert not c.converged and math.isnan(c.gci_fine_medium) and math.isnan(c.gci_medium_coarse)
    assert c.status in ("oscillatory", "non-convergent", "invalid") and c.reason
