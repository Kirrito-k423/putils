from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from putils.compare_perf.collector import TimingCollector
from putils.compare_perf.export import build_chrome_trace
from putils.compare_perf.schema import build_schema


def dump_compare_perf_snapshot(
    *,
    collector: TimingCollector,
    output_dir: str | Path,
    step: int,
    tag: str,
    filename_template: str = "compare_perf_step_{step}_{tag}.json",
    chrome_trace_filename_template: str | None = None,
    run_metadata_extra: dict[str, Any] | None = None,
) -> Path:
    if not isinstance(collector, TimingCollector):
        raise TypeError("collector must be a TimingCollector")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    events_payload = [_timing_event_to_dict(event) for event in collector.events]
    run_metadata: dict[str, Any] = dict(run_metadata_extra or {})
    run_metadata["tag"] = tag
    run_metadata["step"] = int(step)

    payload = build_schema(
        events=events_payload,
        run_metadata=run_metadata,
        alignment=_empty_alignment(),
        summary=collector.summary,
    )

    file_name = filename_template.format(step=step, tag=tag)
    snapshot_path = output_path / file_name
    snapshot_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if chrome_trace_filename_template is not None:
        trace_payload = build_chrome_trace(_collector_events_to_timeline_events(collector=collector, tag=tag))
        trace_file_name = chrome_trace_filename_template.format(step=step, tag=tag)
        chrome_trace_path = output_path / trace_file_name
        chrome_trace_path.write_text(
            json.dumps(trace_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return snapshot_path


def _timing_event_to_dict(event: Any) -> dict[str, Any]:
    return {
        "name": str(event.name),
        "start_ns": int(event.start_ns),
        "end_ns": int(event.end_ns),
        "inclusive_ns": int(event.inclusive_ns),
        "exclusive_ns": int(event.exclusive_ns),
    }


def _empty_alignment() -> dict[str, Any]:
    return {
        "matched": [],
        "ambiguous": [],
        "unmatched": {"left": [], "right": []},
        "counts": {"matched": 0, "ambiguous": 0, "unmatched": 0},
    }


def _collector_events_to_timeline_events(*, collector: TimingCollector, tag: str) -> list[dict[str, Any]]:
    timeline_events: list[dict[str, Any]] = []
    for event in collector.events:
        timeline_events.append(
            {
                "name": str(event.name),
                "start_ns": int(event.start_ns),
                "end_ns": int(event.end_ns),
                "pid": 1,
                "tid": 0,
                "thread_name": "main",
                "process_name": str(tag),
            }
        )
    return timeline_events
