import json
from typing import cast

import pytest

from putils.compare_perf import TimingCollector, compare_perf, dump_compare_perf_snapshot


def test_dump_compare_perf_snapshot_writes_schema_payload(monkeypatch, tmp_path):
    collector = TimingCollector(sync_mode="none")
    tick_values = iter([1_000_000_000, 1_250_000_000])
    monkeypatch.setattr(
        "putils.compare_perf.collector.time.perf_counter_ns", lambda: next(tick_values)
    )

    with compare_perf("train.step", collector=collector, threshold_seconds=1e-9):
        pass

    snapshot_path = dump_compare_perf_snapshot(
        collector=collector,
        output_dir=tmp_path,
        step=10,
        tag="baseline",
    )

    assert snapshot_path.exists()
    assert snapshot_path.name == "compare_perf_step_10_baseline.json"

    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert set(payload.keys()) == {
        "schema_version",
        "events",
        "run_metadata",
        "alignment",
        "summary",
    }
    assert payload["run_metadata"] == {"tag": "baseline", "step": 10}
    assert payload["summary"]["train.step"]["call_count"] == 1
    assert payload["summary"]["train.step"]["total_ns"] == 250_000_000
    assert payload["alignment"]["counts"] == {"matched": 0, "ambiguous": 0, "unmatched": 0}

    event = payload["events"][0]
    assert event["name"] == "train.step"
    assert event["start_ns"] == 1_000_000_000
    assert event["end_ns"] == 1_250_000_000
    assert event["inclusive_ns"] == 250_000_000
    assert event["exclusive_ns"] == 250_000_000
    assert "duration_ns" not in event


def test_dump_compare_perf_snapshot_supports_custom_filename_and_empty_events(monkeypatch, tmp_path):
    collector = TimingCollector(sync_mode="none")
    tick_values = iter([2_000_000_000, 2_020_000_000])
    monkeypatch.setattr(
        "putils.compare_perf.collector.time.perf_counter_ns", lambda: next(tick_values)
    )

    with compare_perf("tiny.step", collector=collector, threshold_seconds=0.1):
        pass

    snapshot_path = dump_compare_perf_snapshot(
        collector=collector,
        output_dir=tmp_path,
        step=3,
        tag="target",
        filename_template="{tag}-s{step}.json",
    )

    assert snapshot_path.name == "target-s3.json"
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert payload["events"] == []
    assert payload["summary"]["tiny.step"]["call_count"] == 1
    assert payload["summary"]["tiny.step"]["total_ns"] == 20_000_000


def test_dump_compare_perf_snapshot_rejects_invalid_collector(tmp_path):
    with pytest.raises(TypeError, match="collector must be a TimingCollector"):
        dump_compare_perf_snapshot(
            collector=cast(TimingCollector, object()),
            output_dir=tmp_path,
            step=1,
            tag="bad",
        )
