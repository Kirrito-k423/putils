import pytest

from putils.compare_perf.collector import measure_overhead


@pytest.mark.slow
def test_overhead_budget_under_3_percent(monkeypatch):
    measured_totals = iter([10_000_000, 10_200_000])
    monkeypatch.setattr(
        "putils.compare_perf.collector._run_n_times",
        lambda workload, iterations: next(measured_totals),
    )

    result = measure_overhead(
        baseline_workload=lambda: None,
        instrumented_workload=lambda: None,
        iterations=8,
    )

    assert result["baseline_total_ns"] == 10_000_000.0
    assert result["instrumented_total_ns"] == 10_200_000.0
    assert result["overhead_ns"] == 200_000.0
    assert result["overhead_pct"] == 2.0
