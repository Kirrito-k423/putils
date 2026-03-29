import re

import pytest

from putils.compare_perf.export import (
    build_aligned_chrome_trace,
    build_aligned_stack_chrome_trace,
    build_chrome_trace,
    export_compare_result,
)


def test_trace_json_is_valid(tmp_path):
    timeline_events = [
        {
            "name": "encoder.layer.1.mlp",
            "start_ns": 2_000_000,
            "end_ns": 4_000_000,
            "pid": 11,
            "tid": 2,
            "thread_name": "rank-0-main",
            "process_name": "target",
            "args": {"scope": "mlp"},
        },
        {
            "name": "encoder.layer.0.attn",
            "start_ns": 1_000_000,
            "end_ns": 1_500_000,
            "pid": 10,
            "tid": 1,
            "thread_name": "rank-0-main",
            "process_name": "baseline",
            "args": {"scope": "attn"},
        },
        {
            "name": "encoder.layer.1.mlp.fc1",
            "start_ns": 2_200_000,
            "end_ns": 2_600_000,
            "pid": 10,
            "tid": 1,
            "thread_name": "rank-0-main",
            "process_name": "baseline",
            "args": {"scope": "mlp_fc1"},
        },
        {
            "name": "enc.block.1.mlp.fc1",
            "start_ns": 2_250_000,
            "end_ns": 2_700_000,
            "pid": 11,
            "tid": 2,
            "thread_name": "rank-0-main",
            "process_name": "target",
            "args": {"scope": "mlp_fc1"},
        },
    ]

    out_trace = tmp_path / "trace.json"
    out_trace_aligned = tmp_path / "trace_aligned.json"
    out_trace_aligned_stack = tmp_path / "trace_aligned_stack.json"
    out_summary_json = tmp_path / "summary.json"
    out_summary_md = tmp_path / "summary.md"

    exported = export_compare_result(
        timeline_events=timeline_events,
        alignment={
            "matched": [
                {
                    "left": "encoder.layer.0.attn",
                    "right": "enc.block.0.attention",
                    "confidence": 0.96,
                    "status": "auto_matched",
                    "source": "rule",
                    "score_components": {
                        "token": 0.8,
                        "path": 0.9,
                        "type": 1.0,
                        "layer": 1.0,
                        "order": 1.0,
                        "weights": {
                            "token": 0.35,
                            "path": 0.25,
                            "type": 0.2,
                            "layer": 0.12,
                            "order": 0.08,
                        },
                        "left_module_type": "attention",
                        "right_module_type": "attention",
                    },
                },
                {
                    "left": "encoder.layer.1.mlp",
                    "right": "enc.block.1.mlp",
                    "confidence": 0.98,
                    "status": "auto_matched",
                    "source": "rule",
                    "score_components": {
                        "token": 1.0,
                        "path": 0.9,
                        "type": 1.0,
                        "layer": 1.0,
                        "order": 1.0,
                        "weights": {
                            "token": 0.35,
                            "path": 0.25,
                            "type": 0.2,
                            "layer": 0.12,
                            "order": 0.08,
                        },
                        "left_module_type": "feedforward",
                        "right_module_type": "feedforward",
                    },
                },
            ],
            "counts": {"matched": 2, "ambiguous": 0, "unmatched": 0},
            "unmatched": {"left": [], "right": []},
        },
        diff_result={
            "modules": [
                {
                    "baseline_module": "encoder.layer.0.attn",
                    "target_module": "enc.block.0.attention",
                    "baseline_ms": 0.5,
                    "target_ms": 0.8,
                    "delta_ms": 0.3,
                    "delta_pct": 60.0,
                },
                {
                    "baseline_module": "encoder.layer.1.mlp",
                    "target_module": "enc.block.1.mlp",
                    "baseline_ms": 2.0,
                    "target_ms": 2.5,
                    "delta_ms": 0.5,
                    "delta_pct": 25.0,
                },
            ],
            "counts": {"matched_pairs": 2, "compared_modules": 2, "excluded_pairs": 0},
            "rank_aggregation": {"enabled": False, "percentiles": [], "modules": [], "excluded": []},
        },
        config_echo={"threshold_seconds": 0.1},
        trace_json_path=str(out_trace),
        aligned_trace_json_path=str(out_trace_aligned),
        aligned_stack_trace_json_path=str(out_trace_aligned_stack),
        summary_json_path=str(out_summary_json),
        summary_md_path=str(out_summary_md),
    )

    payload = build_chrome_trace(timeline_events)
    assert payload["displayTimeUnit"] == "us"
    assert isinstance(payload["traceEvents"], list)
    assert len(payload["traceEvents"]) == 12

    phases = [item["ph"] for item in payload["traceEvents"]]
    assert phases.count("M") == 4
    assert phases.count("B") == 4
    assert phases.count("E") == 4

    runtime_events = [item for item in payload["traceEvents"] if item["ph"] in {"B", "E"}]
    assert runtime_events[0]["name"] == "encoder.layer.0.attn"
    assert runtime_events[0]["ts"] == 1000
    assert runtime_events[1]["name"] == "encoder.layer.0.attn"
    assert runtime_events[1]["ts"] == 1500

    assert out_trace.exists()
    assert out_trace_aligned.exists()
    assert out_trace_aligned_stack.exists()
    assert out_summary_json.exists()
    assert out_summary_md.exists()
    assert exported["trace"]["displayTimeUnit"] == "us"
    assert exported["trace_aligned"]["displayTimeUnit"] == "us"
    assert exported["trace_aligned_stack"]["displayTimeUnit"] == "us"

    aligned_runtime_events = [
        item for item in exported["trace_aligned"]["traceEvents"] if item.get("ph") in {"B", "E"}
    ]
    assert len(aligned_runtime_events) == 8
    first_pair_begin = [
        item for item in aligned_runtime_events if item.get("ph") == "B" and item.get("args", {}).get("pair_index") == 0
    ]
    assert len(first_pair_begin) == 2
    assert first_pair_begin[0]["ts"] == first_pair_begin[1]["ts"]

    pair0_module_names = {item["name"] for item in first_pair_begin}
    pair0_end_ts = max(
        item["ts"] for item in aligned_runtime_events if item.get("ph") == "E" and item.get("name") in pair0_module_names
    )
    pair1_begin_ts = {
        item["ts"] for item in aligned_runtime_events if item.get("ph") == "B" and item.get("args", {}).get("pair_index") == 1
    }
    assert len(pair1_begin_ts) == 1
    assert pair1_begin_ts.pop() == pair0_end_ts

    stack_runtime_events = [
        item for item in exported["trace_aligned_stack"]["traceEvents"] if item.get("ph") in {"B", "E"}
    ]
    stack_pair0_parent_begin = [
        item
        for item in stack_runtime_events
        if item.get("ph") == "B"
        and item.get("args", {}).get("pair_index") == 0
        and item.get("args", {}).get("child_match_status") == "parent"
    ]
    assert len(stack_pair0_parent_begin) == 2
    assert stack_pair0_parent_begin[0]["ts"] == stack_pair0_parent_begin[1]["ts"]

    stack_pair0_parent_names = {item["name"] for item in stack_pair0_parent_begin}
    stack_pair0_parent_end_ts = max(
        item["ts"]
        for item in stack_runtime_events
        if item.get("ph") == "E" and item.get("name") in stack_pair0_parent_names
    )
    stack_pair1_parent_begin_ts = {
        item["ts"]
        for item in stack_runtime_events
        if item.get("ph") == "B"
        and item.get("args", {}).get("pair_index") == 1
        and item.get("args", {}).get("child_match_status") == "parent"
    }
    assert len(stack_pair1_parent_begin_ts) == 1
    assert stack_pair1_parent_begin_ts.pop() == stack_pair0_parent_end_ts

    assert any(
        item.get("ph") == "B"
        and str(item.get("name", "")).startswith("encoder.layer.1.mlp.")
        and item.get("args", {}).get("view") == "aligned_stack"
        for item in exported["trace_aligned_stack"]["traceEvents"]
    )


def test_build_aligned_trace_keeps_optional_non_default_gap():
    alignment = {
        "matched": [
            {
                "left": "encoder.layer.0.attn",
                "right": "enc.block.0.attention",
                "confidence": 0.96,
                "status": "auto_matched",
                "source": "rule",
                "score_components": {},
            },
            {
                "left": "encoder.layer.1.mlp",
                "right": "enc.block.1.mlp",
                "confidence": 0.98,
                "status": "auto_matched",
                "source": "rule",
                "score_components": {},
            },
        ]
    }
    diff_result = {
        "modules": [
            {
                "baseline_module": "encoder.layer.0.attn",
                "target_module": "enc.block.0.attention",
                "baseline_ms": 0.5,
                "target_ms": 0.8,
                "delta_ms": 0.3,
                "delta_pct": 60.0,
            },
            {
                "baseline_module": "encoder.layer.1.mlp",
                "target_module": "enc.block.1.mlp",
                "baseline_ms": 2.0,
                "target_ms": 2.5,
                "delta_ms": 0.5,
                "delta_pct": 25.0,
            },
        ]
    }

    payload = build_aligned_chrome_trace(alignment=alignment, diff_result=diff_result, slot_gap_us=1000)
    runtime_events = [item for item in payload["traceEvents"] if item.get("ph") in {"B", "E"}]

    pair0_begin = [
        item for item in runtime_events if item.get("ph") == "B" and item.get("args", {}).get("pair_index") == 0
    ]
    pair0_module_names = {item["name"] for item in pair0_begin}
    pair0_end_ts = max(
        item["ts"] for item in runtime_events if item.get("ph") == "E" and item.get("name") in pair0_module_names
    )
    pair1_begin_ts = {
        item["ts"] for item in runtime_events if item.get("ph") == "B" and item.get("args", {}).get("pair_index") == 1
    }

    assert len(pair1_begin_ts) == 1
    assert pair1_begin_ts.pop() == pair0_end_ts + 1000


def test_build_aligned_stack_trace_keeps_optional_non_default_gap():
    alignment = {
        "matched": [
            {
                "left": "encoder.layer.1.mlp",
                "right": "enc.block.1.mlp",
                "confidence": 0.98,
                "status": "auto_matched",
                "source": "rule",
                "score_components": {},
            }
        ]
    }
    diff_result = {
        "modules": [
            {
                "baseline_module": "encoder.layer.1.mlp",
                "target_module": "enc.block.1.mlp",
                "baseline_ms": 2.0,
                "target_ms": 2.5,
                "delta_ms": 0.5,
                "delta_pct": 25.0,
            }
        ]
    }
    timeline_events = [
        {
            "name": "encoder.layer.1.mlp.fc1",
            "start_ns": 1_000,
            "end_ns": 2_000,
        },
        {
            "name": "enc.block.1.mlp.fc1",
            "start_ns": 1_000,
            "end_ns": 2_000,
        },
    ]

    payload = build_aligned_stack_chrome_trace(
        timeline_events=timeline_events,
        alignment=alignment,
        diff_result=diff_result,
        slot_gap_us=1000,
    )
    runtime_events = [item for item in payload["traceEvents"] if item.get("ph") in {"B", "E"}]
    parent_begin_ts = {
        item["ts"]
        for item in runtime_events
        if item.get("ph") == "B" and item.get("args", {}).get("child_match_status") == "parent"
    }
    assert parent_begin_ts == {0}


def test_build_aligned_stack_trace_uses_side_specific_child_durations_with_identical_names():
    alignment = {
        "matched": [
            {
                "left": "encoder.layer.1.mlp",
                "right": "encoder.layer.1.mlp",
                "confidence": 1.0,
                "status": "confirmed",
                "source": "manual",
                "score_components": {},
            }
        ]
    }
    diff_result = {
        "modules": [
            {
                "baseline_module": "encoder.layer.1.mlp",
                "target_module": "encoder.layer.1.mlp",
                "baseline_ms": 10.0,
                "target_ms": 10.0,
                "delta_ms": 0.0,
                "delta_pct": 0.0,
            }
        ]
    }
    timeline_events = [
        {
            "name": "encoder.layer.1.mlp.fc1",
            "start_ns": 1_000,
            "end_ns": 1_001_000,
            "pid": 10,
            "process_name": "baseline",
        },
        {
            "name": "encoder.layer.1.mlp.fc1",
            "start_ns": 2_000,
            "end_ns": 3_002_000,
            "pid": 11,
            "process_name": "target",
        },
    ]

    payload = build_aligned_stack_chrome_trace(
        timeline_events=timeline_events,
        alignment=alignment,
        diff_result=diff_result,
    )
    runtime_events = [item for item in payload["traceEvents"] if item.get("ph") in {"B", "E"}]
    child_begin_events = [
        item
        for item in runtime_events
        if item.get("name") == "encoder.layer.1.mlp.fc1"
        and item.get("args", {}).get("child_match_status") in {"matched", "matched_unaligned"}
        and item.get("ph") == "B"
    ]

    assert len(child_begin_events) == 2

    begin_ts_by_pid = {(item["pid"], item["name"]): item["ts"] for item in child_begin_events}
    end_ts_by_pid = {}
    for item in runtime_events:
        if item.get("ph") != "E":
            continue
        if item.get("name") != "encoder.layer.1.mlp.fc1":
            continue
        key = (item["pid"], item["name"])
        begin_ts = begin_ts_by_pid.get(key)
        if begin_ts is None:
            continue
        if item["ts"] >= begin_ts:
            end_ts_by_pid[key] = item["ts"]

    baseline_duration_us = (
        end_ts_by_pid[(1, "encoder.layer.1.mlp.fc1")] - begin_ts_by_pid[(1, "encoder.layer.1.mlp.fc1")]
    )
    target_duration_us = (
        end_ts_by_pid[(2, "encoder.layer.1.mlp.fc1")] - begin_ts_by_pid[(2, "encoder.layer.1.mlp.fc1")]
    )
    assert baseline_duration_us == 1000
    assert target_duration_us == 3000
    assert begin_ts_by_pid[(1, "encoder.layer.1.mlp.fc1")] == begin_ts_by_pid[(2, "encoder.layer.1.mlp.fc1")]


def test_build_aligned_stack_trace_expands_synthetic_parent_when_child_envelope_is_longer():
    alignment = {
        "matched": [
            {
                "left": "encoder.layer.1.mlp",
                "right": "enc.block.1.mlp",
                "confidence": 0.99,
                "status": "auto_matched",
                "source": "rule",
                "score_components": {},
            }
        ]
    }
    diff_result = {
        "modules": [
            {
                "baseline_module": "encoder.layer.1.mlp",
                "target_module": "enc.block.1.mlp",
                "baseline_ms": 1.0,
                "target_ms": 1.0,
                "delta_ms": 0.0,
                "delta_pct": 0.0,
            }
        ]
    }
    timeline_events = [
        {
            "name": "encoder.layer.1.mlp.fc1",
            "start_ns": 0,
            "end_ns": 3_000_000,
            "pid": 1,
            "process_name": "baseline",
        },
        {
            "name": "enc.block.1.mlp.fc1",
            "start_ns": 0,
            "end_ns": 4_000_000,
            "pid": 2,
            "process_name": "target",
        },
    ]

    payload = build_aligned_stack_chrome_trace(
        timeline_events=timeline_events,
        alignment=alignment,
        diff_result=diff_result,
    )
    runtime_events = [item for item in payload["traceEvents"] if item.get("ph") in {"B", "E"}]

    parent_begin_events = [
        item
        for item in runtime_events
        if item.get("ph") == "B" and item.get("args", {}).get("child_match_status") == "parent"
    ]
    assert len(parent_begin_events) == 2

    begin_map = {(item["pid"], item["name"]): item for item in parent_begin_events}
    parent_end_ts = {}
    for item in runtime_events:
        if item.get("ph") != "E":
            continue
        key = (item["pid"], item["name"])
        begin_event = begin_map.get(key)
        if begin_event is None:
            continue
        if item["ts"] >= begin_event["ts"]:
            parent_end_ts[key] = item["ts"]

    baseline_key = (1, "encoder.layer.1.mlp")
    target_key = (2, "enc.block.1.mlp")
    baseline_parent_us = parent_end_ts[baseline_key] - begin_map[baseline_key]["ts"]
    target_parent_us = parent_end_ts[target_key] - begin_map[target_key]["ts"]

    assert baseline_parent_us > 1000
    assert target_parent_us > 1000

    assert begin_map[baseline_key]["args"]["parent_mode"] == "synthetic_aligned"
    assert begin_map[baseline_key]["args"]["real_parent_duration_ns"] == 1_000_000
    assert begin_map[baseline_key]["args"]["synthetic_parent_duration_ns"] == 4_040_000
    assert begin_map[target_key]["args"]["real_parent_duration_ns"] == 1_000_000
    assert begin_map[target_key]["args"]["synthetic_parent_duration_ns"] == 4_040_000


def test_build_chrome_trace_preserves_parent_child_nesting_order_for_same_start():
    timeline_events = [
        {
            "name": "encoder.layer.1.mlp",
            "start_ns": 1_000_000,
            "end_ns": 5_000_000,
            "pid": 1,
            "tid": 0,
            "thread_name": "main",
            "process_name": "baseline",
            "args": {},
        },
        {
            "name": "encoder.layer.1.mlp.1",
            "start_ns": 1_000_000,
            "end_ns": 2_000_000,
            "pid": 1,
            "tid": 0,
            "thread_name": "main",
            "process_name": "baseline",
            "args": {},
        },
    ]

    payload = build_chrome_trace(timeline_events)
    runtime_events = [item for item in payload["traceEvents"] if item.get("ph") in {"B", "E"}]
    assert len(runtime_events) == 4
    assert [(item["ph"], item["name"]) for item in runtime_events] == [
        ("B", "encoder.layer.1.mlp"),
        ("B", "encoder.layer.1.mlp.1"),
        ("E", "encoder.layer.1.mlp.1"),
        ("E", "encoder.layer.1.mlp"),
    ]


def test_build_aligned_stack_trace_skips_descendant_pairs_from_top_level_slots():
    alignment = {
        "matched": [
            {
                "left": "encoder.layer.1.mlp",
                "right": "enc.block.1.mlp",
                "confidence": 0.99,
                "status": "auto_matched",
                "source": "rule",
                "score_components": {},
            },
            {
                "left": "encoder.layer.1.mlp.fc1",
                "right": "enc.block.1.mlp.fc1",
                "confidence": 0.99,
                "status": "auto_matched",
                "source": "rule",
                "score_components": {},
            },
        ]
    }
    diff_result = {
        "modules": [
            {
                "baseline_module": "encoder.layer.1.mlp",
                "target_module": "enc.block.1.mlp",
                "baseline_ms": 4.0,
                "target_ms": 5.0,
                "delta_ms": 1.0,
                "delta_pct": 25.0,
            },
            {
                "baseline_module": "encoder.layer.1.mlp.fc1",
                "target_module": "enc.block.1.mlp.fc1",
                "baseline_ms": 1.0,
                "target_ms": 1.2,
                "delta_ms": 0.2,
                "delta_pct": 20.0,
            },
        ]
    }
    timeline_events = [
        {
            "name": "encoder.layer.1.mlp.fc1",
            "start_ns": 10_000,
            "end_ns": 1_010_000,
            "pid": 1,
            "process_name": "baseline",
        },
        {
            "name": "enc.block.1.mlp.fc1",
            "start_ns": 20_000,
            "end_ns": 1_220_000,
            "pid": 2,
            "process_name": "target",
        },
    ]

    payload = build_aligned_stack_chrome_trace(
        timeline_events=timeline_events,
        alignment=alignment,
        diff_result=diff_result,
    )
    runtime_events = [item for item in payload["traceEvents"] if item.get("ph") in {"B", "E"}]

    top_level_parent_begins = [
        item
        for item in runtime_events
        if item.get("ph") == "B" and item.get("args", {}).get("child_match_status") == "parent"
    ]
    assert len(top_level_parent_begins) == 2
    assert {item.get("args", {}).get("pair_index") for item in top_level_parent_begins} == {0}
    assert {item["name"] for item in top_level_parent_begins} == {"encoder.layer.1.mlp", "enc.block.1.mlp"}

    assert any(
        item.get("ph") == "B"
        and item.get("name") in {"encoder.layer.1.mlp.fc1", "enc.block.1.mlp.fc1"}
        and item.get("args", {}).get("child_match_status") in {"matched", "matched_unaligned"}
        for item in runtime_events
    )


def test_malformed_timeline_fails_with_explicit_error():
    malformed = [{"name": "encoder.layer.0.attn", "start_ns": 1_000_000}]

    expected = "malformed timeline event at index 0: missing required field 'end_ns'"
    with pytest.raises(ValueError, match=re.escape(expected)):
        build_chrome_trace(malformed)
