from __future__ import annotations

import time
from contextlib import ContextDecorator, contextmanager
from dataclasses import dataclass
from typing import Any, Callable

from putils.compare_perf.config import (
    DEFAULT_SYNC_MODE,
    DEFAULT_THRESHOLD_SECONDS,
    validate_sync_mode,
)


@dataclass(frozen=True)
class TimingEvent:
    name: str
    start_ns: int
    end_ns: int
    inclusive_ns: int
    exclusive_ns: int


class TimingCollector:
    def __init__(
        self,
        sync_mode: str = DEFAULT_SYNC_MODE,
        synchronize: Callable[[], None] | None = None,
    ):
        self.sync_mode = validate_sync_mode(sync_mode)
        self._synchronize = synchronize
        self._events: list[TimingEvent] = []
        self._stack: list[dict[str, int | str]] = []
        self._summary: dict[str, dict[str, Any]] = {}

    @property
    def events(self) -> list[TimingEvent]:
        return list(self._events)

    @property
    def summary(self) -> dict[str, dict[str, Any]]:
        return {
            scope_name: {
                "call_count": int(values["call_count"]),
                "total_ns": int(values["total_ns"]),
                "total_seconds": float(values["total_seconds"]),
            }
            for scope_name, values in self._summary.items()
        }

    def _sync_boundary(self) -> None:
        if self.sync_mode == "boundary" and self._synchronize is not None:
            self._synchronize()

    @contextmanager
    def scope(self, name: str):
        self._sync_boundary()
        start_ns = time.perf_counter_ns()
        frame = {
            "name": name,
            "start_ns": start_ns,
            "child_inclusive_ns": 0,
        }
        self._stack.append(frame)

        try:
            yield
        finally:
            end_ns = time.perf_counter_ns()
            self._sync_boundary()

            current_frame = self._stack.pop()
            inclusive_ns = int(end_ns - int(current_frame["start_ns"]))
            child_inclusive_ns = int(current_frame["child_inclusive_ns"])
            exclusive_ns = max(0, inclusive_ns - child_inclusive_ns)

            event = TimingEvent(
                name=str(current_frame["name"]),
                start_ns=int(current_frame["start_ns"]),
                end_ns=end_ns,
                inclusive_ns=inclusive_ns,
                exclusive_ns=exclusive_ns,
            )
            self._events.append(event)

            if self._stack:
                parent = self._stack[-1]
                parent["child_inclusive_ns"] = int(parent["child_inclusive_ns"]) + inclusive_ns

    @contextmanager
    def compare_scope(self, scope_name: str, threshold_seconds: float = DEFAULT_THRESHOLD_SECONDS):
        _validate_scope_name(scope_name)
        if threshold_seconds <= 0:
            raise ValueError("threshold_seconds must be > 0")

        before_events = len(self._events)
        with self.scope(scope_name):
            yield

        if len(self._events) <= before_events:
            return

        event = self._events[-1]
        if event.name != scope_name:
            return

        self._record_summary(scope_name=scope_name, inclusive_ns=event.inclusive_ns)
        if event.inclusive_ns < int(threshold_seconds * 1_000_000_000):
            self._events.pop()

    def _record_summary(self, scope_name: str, inclusive_ns: int) -> None:
        current = self._summary.get(scope_name)
        if current is None:
            current = {
                "call_count": 0,
                "total_ns": 0,
                "total_seconds": 0.0,
            }
            self._summary[scope_name] = current

        total_ns = int(current["total_ns"]) + int(inclusive_ns)
        call_count = int(current["call_count"]) + 1
        current["call_count"] = call_count
        current["total_ns"] = total_ns
        current["total_seconds"] = float(total_ns) / 1_000_000_000.0


def _validate_scope_name(scope_name: str) -> str:
    if not isinstance(scope_name, str) or not scope_name.strip():
        raise ValueError("scope_name must be a non-empty string")
    return scope_name


class _ComparePerfContext(ContextDecorator):
    def __init__(self, collector: TimingCollector, scope_name: str, threshold_seconds: float):
        self.collector = collector
        self.scope_name = _validate_scope_name(scope_name)
        self.threshold_seconds = threshold_seconds
        self._active_context = None

    def __enter__(self):
        self._active_context = self.collector.compare_scope(
            scope_name=self.scope_name,
            threshold_seconds=self.threshold_seconds,
        )
        return self._active_context.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        if self._active_context is None:
            return False
        return self._active_context.__exit__(exc_type, exc_value, traceback)


def compare_perf(
    scope_name: str,
    *,
    collector: TimingCollector | None = None,
    threshold_seconds: float = DEFAULT_THRESHOLD_SECONDS,
    sync_mode: str = DEFAULT_SYNC_MODE,
    synchronize: Callable[[], None] | None = None,
) -> _ComparePerfContext:
    _validate_scope_name(scope_name)

    target_collector = collector
    if target_collector is None:
        target_collector = TimingCollector(sync_mode=sync_mode, synchronize=synchronize)

    return _ComparePerfContext(
        collector=target_collector,
        scope_name=scope_name,
        threshold_seconds=threshold_seconds,
    )


def _run_n_times(workload: Callable[[], None], iterations: int) -> int:
    started_ns = time.perf_counter_ns()
    for _ in range(iterations):
        workload()
    ended_ns = time.perf_counter_ns()
    return ended_ns - started_ns


def measure_overhead(
    baseline_workload: Callable[[], None],
    instrumented_workload: Callable[[], None],
    iterations: int = 5,
) -> dict[str, float]:
    if iterations <= 0:
        raise ValueError("iterations must be > 0")

    baseline_total_ns = _run_n_times(baseline_workload, iterations)
    instrumented_total_ns = _run_n_times(instrumented_workload, iterations)
    overhead_ns = instrumented_total_ns - baseline_total_ns

    if baseline_total_ns <= 0:
        overhead_pct = 0.0
    else:
        overhead_pct = (overhead_ns / baseline_total_ns) * 100.0

    return {
        "baseline_total_ns": float(baseline_total_ns),
        "instrumented_total_ns": float(instrumented_total_ns),
        "overhead_ns": float(overhead_ns),
        "overhead_pct": float(overhead_pct),
    }
