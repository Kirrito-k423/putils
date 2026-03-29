import re

import pytest

from putils.compare_perf.collector import TimingCollector, compare_perf


def test_single_wrapper_integration(monkeypatch):
    collector = TimingCollector(sync_mode="none")
    tick_values = iter(
        [
            1_000_000_000,
            1_120_000_000,
            2_000_000_000,
            2_250_000_000,
        ]
    )
    monkeypatch.setattr(
        "putils.compare_perf.collector.time.perf_counter_ns", lambda: next(tick_values)
    )

    with compare_perf("ctx_scope", collector=collector, threshold_seconds=0.1):
        pass

    @compare_perf("decorated_scope", collector=collector, threshold_seconds=0.1)
    def _workload(value: int) -> int:
        return value + 1

    assert _workload(41) == 42

    events_by_name = {event.name: event for event in collector.events}
    assert "ctx_scope" in events_by_name
    assert "decorated_scope" in events_by_name
    assert events_by_name["ctx_scope"].inclusive_ns == 120_000_000
    assert events_by_name["decorated_scope"].inclusive_ns == 250_000_000

    summary = collector.summary
    assert summary["ctx_scope"]["call_count"] == 1
    assert summary["decorated_scope"]["call_count"] == 1


def test_threshold_skips_detail_but_keeps_summary(monkeypatch):
    collector = TimingCollector(sync_mode="none")
    tick_values = iter(
        [
            1_000_000_000,
            1_050_000_000,
            2_000_000_000,
            2_030_000_000,
        ]
    )
    monkeypatch.setattr(
        "putils.compare_perf.collector.time.perf_counter_ns", lambda: next(tick_values)
    )

    with compare_perf("tiny_scope", collector=collector, threshold_seconds=0.1):
        pass
    with compare_perf("tiny_scope", collector=collector, threshold_seconds=0.1):
        pass

    assert collector.events == []
    summary = collector.summary["tiny_scope"]
    assert summary["call_count"] == 2
    assert summary["total_ns"] == 80_000_000
    assert summary["total_seconds"] == pytest.approx(0.08)


def test_compare_perf_rejects_empty_scope_name():
    expected = "scope_name must be a non-empty string"
    collector = TimingCollector(sync_mode="none")

    with pytest.raises(ValueError, match=re.escape(expected)):
        compare_perf("", collector=collector)

    with pytest.raises(ValueError, match=re.escape(expected)):
        compare_perf("   ", collector=collector)

    with pytest.raises(ValueError, match=re.escape(expected)):
        compare_perf(None, collector=collector)  # type: ignore[arg-type]
