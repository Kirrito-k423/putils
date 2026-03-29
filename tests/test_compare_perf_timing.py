import re

import pytest

from putils.compare_perf.collector import TimingCollector


def _cpu_work(units: int = 5_000) -> int:
    total = 0
    for i in range(units):
        total += (i % 7) * (i % 13)
    return total


def test_inclusive_exclusive_metrics():
    collector = TimingCollector(sync_mode="none")

    with collector.scope("outer"):
        _cpu_work(2_000)
        with collector.scope("inner"):
            _cpu_work(3_000)

    events = {event.name: event for event in collector.events}

    assert "outer" in events
    assert "inner" in events

    outer = events["outer"]
    inner = events["inner"]

    assert outer.inclusive_ns > 0
    assert inner.inclusive_ns > 0
    assert outer.exclusive_ns >= 0
    assert inner.exclusive_ns >= 0
    assert outer.inclusive_ns >= outer.exclusive_ns
    assert inner.inclusive_ns >= inner.exclusive_ns

    assert outer.inclusive_ns == outer.exclusive_ns + inner.inclusive_ns


def test_invalid_sync_mode_rejected():
    expected = "Invalid sync_mode: per_op_sync. Expected one of [none, boundary]"
    with pytest.raises(ValueError, match=re.escape(expected)):
        TimingCollector(sync_mode="per_op_sync")


def test_boundary_sync_calls_hook_at_scope_boundaries():
    called = {"count": 0}

    def _sync() -> None:
        called["count"] += 1

    collector = TimingCollector(sync_mode="boundary", synchronize=_sync)
    with collector.scope("single"):
        _cpu_work(500)

    assert called["count"] == 2


def test_dynamic_control_flow_nested_and_optional_scope(monkeypatch):
    collector = TimingCollector(sync_mode="none")
    tick_values = iter(
        [
            100,
            150,
            220,
            300,
            400,
            460,
        ]
    )
    monkeypatch.setattr(
        "putils.compare_perf.collector.time.perf_counter_ns", lambda: next(tick_values)
    )

    with collector.scope("step"):
        with collector.scope("branch"):
            pass

    with collector.scope("step"):
        pass

    assert [event.name for event in collector.events] == ["branch", "step", "step"]

    branch_event = collector.events[0]
    first_step_event = collector.events[1]
    second_step_event = collector.events[2]

    assert branch_event.inclusive_ns == 70
    assert branch_event.exclusive_ns == 70

    assert first_step_event.inclusive_ns == 200
    assert first_step_event.exclusive_ns == 130

    assert second_step_event.inclusive_ns == 60
    assert second_step_event.exclusive_ns == 60
