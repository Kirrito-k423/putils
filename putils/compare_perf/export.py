from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from putils.compare_perf.align import align_modules


def _require_mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return value


def _required_int(event: Mapping[str, Any], key: str, index: int) -> int:
    if key not in event:
        raise ValueError(f"malformed timeline event at index {index}: missing required field '{key}'")
    try:
        return int(event[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"malformed timeline event at index {index}: field '{key}' must be an integer"
        ) from exc


def _required_name(event: Mapping[str, Any], index: int) -> str:
    name = event.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"malformed timeline event at index {index}: field 'name' must be non-empty")
    return name


def _to_args(event: Mapping[str, Any], index: int) -> dict[str, Any]:
    raw_args = event.get("args", {})
    if raw_args is None:
        return {}
    if not isinstance(raw_args, Mapping):
        raise ValueError(f"malformed timeline event at index {index}: field 'args' must be a mapping")
    return dict(sorted(raw_args.items(), key=lambda item: str(item[0])))


def build_chrome_trace(timeline_events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(timeline_events):
        event = _require_mapping(raw, context=f"timeline event at index {index}")
        name = _required_name(event, index)
        start_ns = _required_int(event, "start_ns", index)
        end_ns = _required_int(event, "end_ns", index)
        if end_ns < start_ns:
            raise ValueError(
                f"malformed timeline event at index {index}: field 'end_ns' must be >= 'start_ns'"
            )

        pid = int(event.get("pid", 1))
        tid = int(event.get("tid", 0))
        thread_name = str(event.get("thread_name", f"thread-{tid}"))
        process_name = str(event.get("process_name", f"process-{pid}"))
        args = _to_args(event, index)

        normalized.append(
            {
                "name": name,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "pid": pid,
                "tid": tid,
                "thread_name": thread_name,
                "process_name": process_name,
                "args": args,
            }
        )

    normalized.sort(
        key=lambda item: (
            item["start_ns"],
            item["end_ns"],
            item["name"],
            item["pid"],
            item["tid"],
        )
    )

    trace_events: list[dict[str, Any]] = []

    process_entries = sorted({(item["pid"], item["process_name"]) for item in normalized})
    for pid, process_name in process_entries:
        trace_events.append(
            {
                "name": "process_name",
                "ph": "M",
                "ts": 0,
                "pid": pid,
                "args": {"name": process_name},
            }
        )

    thread_entries = sorted({(item["pid"], item["tid"], item["thread_name"]) for item in normalized})
    for pid, tid, thread_name in thread_entries:
        trace_events.append(
            {
                "name": "thread_name",
                "ph": "M",
                "ts": 0,
                "pid": pid,
                "tid": tid,
                "args": {"name": thread_name},
            }
        )

    boundaries: list[dict[str, Any]] = []
    for order_index, item in enumerate(normalized):
        start_us = int(item["start_ns"] // 1_000)
        end_us = int(item["end_ns"] // 1_000)
        duration_ns = int(item["end_ns"] - item["start_ns"])
        boundaries.append(
            {
                "name": item["name"],
                "ph": "B",
                "ts": start_us,
                "pid": item["pid"],
                "tid": item["tid"],
                "args": dict(item["args"]),
                "start_ns": int(item["start_ns"]),
                "end_ns": int(item["end_ns"]),
                "duration_ns": duration_ns,
                "order_index": order_index,
            }
        )
        boundaries.append(
            {
                "name": item["name"],
                "ph": "E",
                "ts": end_us,
                "pid": item["pid"],
                "tid": item["tid"],
                "args": {},
                "start_ns": int(item["start_ns"]),
                "end_ns": int(item["end_ns"]),
                "duration_ns": duration_ns,
                "order_index": order_index,
            }
        )

    boundaries.sort(
        key=lambda item: (
            item["pid"],
            item["tid"],
            item["ts"],
            0 if item["ph"] == "E" else 1,
            -item["duration_ns"] if item["ph"] == "B" else -item["start_ns"],
            item["name"],
            item["order_index"],
        )
    )

    for boundary in boundaries:
        trace_events.append(
            {
                "name": boundary["name"],
                "ph": boundary["ph"],
                "ts": boundary["ts"],
                "pid": boundary["pid"],
                "tid": boundary["tid"],
                "args": dict(boundary["args"]),
            }
        )

    return {
        "traceEvents": trace_events,
        "displayTimeUnit": "us",
    }


def build_summary(
    *,
    alignment: Mapping[str, Any],
    diff_result: Mapping[str, Any],
    config_echo: Mapping[str, Any] | None = None,
    top_n: int = 5,
) -> dict[str, Any]:
    if top_n <= 0:
        raise ValueError("top_n must be > 0")

    alignment_counts = dict(sorted(dict(alignment.get("counts", {})).items(), key=lambda item: item[0]))
    unmatched = alignment.get("unmatched", {})
    unmatched_left = [str(item) for item in unmatched.get("left", [])]
    unmatched_right = [str(item) for item in unmatched.get("right", [])]

    modules = [dict(item) for item in diff_result.get("modules", []) if isinstance(item, Mapping)]
    modules.sort(key=lambda item: (float(item.get("delta_ms", 0.0)), str(item.get("baseline_module", ""))))

    improvements = [
        item
        for item in modules
        if float(item.get("delta_ms", 0.0)) < 0.0
    ]
    regressions = [
        item
        for item in reversed(modules)
        if float(item.get("delta_ms", 0.0)) > 0.0
    ]

    top_improvements = [
        {
            "baseline_module": str(item.get("baseline_module", "")),
            "target_module": str(item.get("target_module", "")),
            "delta_ms": float(item.get("delta_ms", 0.0)),
            "delta_pct": item.get("delta_pct"),
        }
        for item in improvements[:top_n]
    ]
    top_regressions = [
        {
            "baseline_module": str(item.get("baseline_module", "")),
            "target_module": str(item.get("target_module", "")),
            "delta_ms": float(item.get("delta_ms", 0.0)),
            "delta_pct": item.get("delta_pct"),
        }
        for item in regressions[:top_n]
    ]

    diff_counts = dict(sorted(dict(diff_result.get("counts", {})).items(), key=lambda item: item[0]))
    rank_aggregation = _require_mapping(diff_result.get("rank_aggregation", {}), context="diff_result.rank_aggregation")

    return {
        "config": dict(sorted((config_echo or {}).items(), key=lambda item: item[0])),
        "alignment": {
            "counts": alignment_counts,
            "unmatched": {
                "left": sorted(unmatched_left),
                "right": sorted(unmatched_right),
            },
            "ambiguous_count": int(alignment_counts.get("ambiguous", 0)),
        },
        "diff": {
            "counts": diff_counts,
            "rank_aggregation": {
                "enabled": bool(rank_aggregation.get("enabled", False)),
                "percentiles": list(rank_aggregation.get("percentiles", [])),
            },
        },
        "top_regressions": top_regressions,
        "top_improvements": top_improvements,
    }


def build_aligned_chrome_trace(
    *,
    alignment: Mapping[str, Any],
    diff_result: Mapping[str, Any],
    slot_gap_us: int = 0,
) -> dict[str, Any]:
    if slot_gap_us < 0:
        raise ValueError("slot_gap_us must be >= 0")

    matched = [item for item in alignment.get("matched", []) if isinstance(item, Mapping)]
    diff_modules = [item for item in diff_result.get("modules", []) if isinstance(item, Mapping)]
    diff_lookup = {
        (str(item.get("baseline_module", "")), str(item.get("target_module", ""))): item
        for item in diff_modules
    }

    cursor_ns = 0
    gap_ns = int(slot_gap_us) * 1_000
    aligned_timeline: list[dict[str, Any]] = []

    for pair_index, pair in enumerate(matched):
        baseline_module = str(pair.get("left", "")).strip()
        target_module = str(pair.get("right", "")).strip()
        if not baseline_module or not target_module:
            continue

        diff_item = diff_lookup.get((baseline_module, target_module))
        if diff_item is None:
            continue

        baseline_ms = max(0.0, float(diff_item.get("baseline_ms", 0.0)))
        target_ms = max(0.0, float(diff_item.get("target_ms", 0.0)))
        baseline_duration_ns = max(1, int(round(baseline_ms * 1_000_000.0)))
        target_duration_ns = max(1, int(round(target_ms * 1_000_000.0)))

        shared_args = {
            "pair_index": pair_index,
            "baseline_module": baseline_module,
            "target_module": target_module,
            "confidence": float(pair.get("confidence", 0.0)),
            "match_status": str(pair.get("status", "")),
            "match_source": str(pair.get("source", "")),
            "delta_ms": float(diff_item.get("delta_ms", 0.0)),
            "delta_pct": diff_item.get("delta_pct"),
            "score_components": dict(pair.get("score_components", {})),
            "view": "aligned_pair",
        }

        aligned_timeline.append(
            {
                "name": baseline_module,
                "start_ns": cursor_ns,
                "end_ns": cursor_ns + baseline_duration_ns,
                "pid": 1,
                "tid": 0,
                "thread_name": "aligned",
                "process_name": "baseline_aligned",
                "args": dict(shared_args, role="baseline"),
            }
        )
        aligned_timeline.append(
            {
                "name": target_module,
                "start_ns": cursor_ns,
                "end_ns": cursor_ns + target_duration_ns,
                "pid": 2,
                "tid": 0,
                "thread_name": "aligned",
                "process_name": "target_aligned",
                "args": dict(shared_args, role="target"),
            }
        )

        cursor_ns += max(baseline_duration_ns, target_duration_ns) + gap_ns

    return build_chrome_trace(aligned_timeline)


def _aggregate_module_timing_by_side(
    timeline_events: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, int]]]:
    aggregated: dict[str, dict[str, dict[str, int]]] = {
        "baseline": {},
        "target": {},
        "unknown": {},
    }
    for index, raw in enumerate(timeline_events):
        event = _require_mapping(raw, context=f"timeline event at index {index}")
        module_name = _required_name(event, index)
        start_ns = _required_int(event, "start_ns", index)
        end_ns = _required_int(event, "end_ns", index)
        if end_ns < start_ns:
            raise ValueError(
                f"malformed timeline event at index {index}: field 'end_ns' must be >= 'start_ns'"
            )

        duration_ns = end_ns - start_ns
        process_name = str(event.get("process_name", "")).strip().lower()
        if "baseline" in process_name:
            side = "baseline"
        elif "target" in process_name:
            side = "target"
        else:
            pid_value = event.get("pid")
            try:
                pid = int(pid_value) if pid_value is not None else 0
            except (TypeError, ValueError):
                pid = 0
            if pid == 1:
                side = "baseline"
            elif pid == 2:
                side = "target"
            else:
                side = "unknown"

        side_bucket = aggregated[side]
        current = side_bucket.get(module_name)
        if current is None:
            side_bucket[module_name] = {
                "duration_ns": duration_ns,
                "first_start_ns": start_ns,
            }
            continue

        current["duration_ns"] = int(current["duration_ns"]) + duration_ns
        current["first_start_ns"] = min(int(current["first_start_ns"]), start_ns)
    return aggregated


def _get_module_timing_for_side(
    *,
    module_name: str,
    preferred: Mapping[str, Mapping[str, int]],
    fallback: Mapping[str, Mapping[str, int]],
) -> Mapping[str, int] | None:
    timing = preferred.get(module_name)
    if timing is not None:
        return timing
    return fallback.get(module_name)


def _prefixed_children(module_names: Sequence[str], *, parent_name: str) -> list[str]:
    prefix = f"{parent_name}."
    children = [name for name in module_names if name.startswith(prefix)]
    return children


def _relative_child_name(*, full_name: str, parent_name: str) -> str:
    prefix = f"{parent_name}."
    return full_name[len(prefix) :]


def _single_side_child_start(*, cursor_ns: int, duration_ns: int, parent_duration_ns: int) -> int:
    if duration_ns <= parent_duration_ns and cursor_ns + duration_ns > parent_duration_ns:
        return max(0, parent_duration_ns - duration_ns)
    return max(0, cursor_ns)


def _is_descendant_pair(
    *,
    candidate_left: str,
    candidate_right: str,
    parent_left: str,
    parent_right: str,
) -> bool:
    if candidate_left == parent_left or candidate_right == parent_right:
        return False
    return candidate_left.startswith(f"{parent_left}.") and candidate_right.startswith(f"{parent_right}.")


def _top_level_matched_pairs(matched: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    filtered: list[Mapping[str, Any]] = []
    for candidate in matched:
        candidate_left = str(candidate.get("left", "")).strip()
        candidate_right = str(candidate.get("right", "")).strip()
        if not candidate_left or not candidate_right:
            continue

        has_parent = False
        for possible_parent in matched:
            parent_left = str(possible_parent.get("left", "")).strip()
            parent_right = str(possible_parent.get("right", "")).strip()
            if not parent_left or not parent_right:
                continue
            if _is_descendant_pair(
                candidate_left=candidate_left,
                candidate_right=candidate_right,
                parent_left=parent_left,
                parent_right=parent_right,
            ):
                has_parent = True
                break

        if not has_parent:
            filtered.append(candidate)
    return filtered


def build_aligned_stack_chrome_trace(
    *,
    timeline_events: Sequence[Mapping[str, Any]],
    alignment: Mapping[str, Any],
    diff_result: Mapping[str, Any],
    slot_gap_us: int = 0,
) -> dict[str, Any]:
    if slot_gap_us < 0:
        raise ValueError("slot_gap_us must be >= 0")

    matched = [item for item in alignment.get("matched", []) if isinstance(item, Mapping)]
    top_level_matched = _top_level_matched_pairs(matched)
    diff_modules = [item for item in diff_result.get("modules", []) if isinstance(item, Mapping)]
    diff_lookup = {
        (str(item.get("baseline_module", "")), str(item.get("target_module", ""))): item
        for item in diff_modules
    }

    module_timing_by_side = _aggregate_module_timing_by_side(timeline_events)
    baseline_module_timing = dict(module_timing_by_side.get("baseline", {}))
    target_module_timing = dict(module_timing_by_side.get("target", {}))
    unknown_module_timing = dict(module_timing_by_side.get("unknown", {}))

    cursor_ns = 0
    gap_ns = int(slot_gap_us) * 1_000
    aligned_timeline: list[dict[str, Any]] = []

    for pair_index, pair in enumerate(top_level_matched):
        baseline_module = str(pair.get("left", "")).strip()
        target_module = str(pair.get("right", "")).strip()
        if not baseline_module or not target_module:
            continue

        diff_item = diff_lookup.get((baseline_module, target_module))
        if diff_item is None:
            continue

        baseline_ms = max(0.0, float(diff_item.get("baseline_ms", 0.0)))
        target_ms = max(0.0, float(diff_item.get("target_ms", 0.0)))
        real_baseline_duration_ns = max(1, int(round(baseline_ms * 1_000_000.0)))
        real_target_duration_ns = max(1, int(round(target_ms * 1_000_000.0)))

        shared_args = {
            "pair_index": pair_index,
            "baseline_module": baseline_module,
            "target_module": target_module,
            "confidence": float(pair.get("confidence", 0.0)),
            "match_status": str(pair.get("status", "")),
            "match_source": str(pair.get("source", "")),
            "delta_ms": float(diff_item.get("delta_ms", 0.0)),
            "delta_pct": diff_item.get("delta_pct"),
            "score_components": dict(pair.get("score_components", {})),
            "view": "aligned_stack",
        }

        left_children = _prefixed_children(list(baseline_module_timing.keys()), parent_name=baseline_module)
        right_children = _prefixed_children(list(target_module_timing.keys()), parent_name=target_module)

        if not left_children:
            left_children = _prefixed_children(list(unknown_module_timing.keys()), parent_name=baseline_module)
        if not right_children:
            right_children = _prefixed_children(list(unknown_module_timing.keys()), parent_name=target_module)

        left_children.sort(
            key=lambda name: (
                int(
                    (
                        _get_module_timing_for_side(
                            module_name=name,
                            preferred=baseline_module_timing,
                            fallback=unknown_module_timing,
                        )
                        or {}
                    ).get("first_start_ns", 0)
                ),
                name,
            )
        )
        right_children.sort(
            key=lambda name: (
                int(
                    (
                        _get_module_timing_for_side(
                            module_name=name,
                            preferred=target_module_timing,
                            fallback=unknown_module_timing,
                        )
                        or {}
                    ).get("first_start_ns", 0)
                ),
                name,
            )
        )

        left_relative = [_relative_child_name(full_name=name, parent_name=baseline_module) for name in left_children]
        right_relative = [_relative_child_name(full_name=name, parent_name=target_module) for name in right_children]

        child_alignment = align_modules(
            left_modules=left_relative,
            right_modules=right_relative,
        )

        left_lookup = {
            _relative_child_name(full_name=full_name, parent_name=baseline_module): full_name for full_name in left_children
        }
        right_lookup = {
            _relative_child_name(full_name=full_name, parent_name=target_module): full_name for full_name in right_children
        }

        shared_child_cursor_ns = 0
        left_child_cursor_ns = 0
        right_child_cursor_ns = 0
        child_envelope_ns = 0
        pair_child_events: list[dict[str, Any]] = []

        for child_pair_index, child_pair in enumerate(
            [item for item in child_alignment.get("matched", []) if isinstance(item, Mapping)]
        ):
            left_relative_name = str(child_pair.get("left", ""))
            right_relative_name = str(child_pair.get("right", ""))
            left_full_name = left_lookup.get(left_relative_name)
            right_full_name = right_lookup.get(right_relative_name)
            if not left_full_name or not right_full_name:
                continue

            left_timing = _get_module_timing_for_side(
                module_name=left_full_name,
                preferred=baseline_module_timing,
                fallback=unknown_module_timing,
            )
            right_timing = _get_module_timing_for_side(
                module_name=right_full_name,
                preferred=target_module_timing,
                fallback=unknown_module_timing,
            )
            if left_timing is None or right_timing is None:
                continue

            left_duration_ns = max(1, int(left_timing["duration_ns"]))
            right_duration_ns = max(1, int(right_timing["duration_ns"]))

            left_start_offset_ns = shared_child_cursor_ns
            right_start_offset_ns = shared_child_cursor_ns
            pair_child_events.append(
                {
                    "name": left_full_name,
                    "start_ns": cursor_ns + left_start_offset_ns,
                    "end_ns": cursor_ns + left_start_offset_ns + left_duration_ns,
                    "pid": 1,
                    "tid": 0,
                    "thread_name": "aligned_stack",
                    "process_name": "baseline_aligned_stack",
                    "args": dict(
                        shared_args,
                        role="baseline",
                        child_match_status="matched",
                        child_pair_index=child_pair_index,
                        child_relative_name=left_relative_name,
                        child_confidence=float(child_pair.get("confidence", 0.0)),
                    ),
                }
            )
            pair_child_events.append(
                {
                    "name": right_full_name,
                    "start_ns": cursor_ns + right_start_offset_ns,
                    "end_ns": cursor_ns + right_start_offset_ns + right_duration_ns,
                    "pid": 2,
                    "tid": 0,
                    "thread_name": "aligned_stack",
                    "process_name": "target_aligned_stack",
                    "args": dict(
                        shared_args,
                        role="target",
                        child_match_status="matched",
                        child_pair_index=child_pair_index,
                        child_relative_name=right_relative_name,
                        child_confidence=float(child_pair.get("confidence", 0.0)),
                    ),
                }
            )
            shared_child_cursor_ns += max(left_duration_ns, right_duration_ns)
            left_child_cursor_ns = max(left_child_cursor_ns, left_start_offset_ns + left_duration_ns)
            right_child_cursor_ns = max(right_child_cursor_ns, right_start_offset_ns + right_duration_ns)
            child_envelope_ns = max(child_envelope_ns, left_child_cursor_ns, right_child_cursor_ns)

        ambiguous_children = [item for item in child_alignment.get("ambiguous", []) if isinstance(item, Mapping)]
        for ambiguous_index, ambiguous_item in enumerate(ambiguous_children):
            left_relative_name = str(ambiguous_item.get("left", ""))
            left_full_name = left_lookup.get(left_relative_name)
            if not left_full_name:
                continue
            left_timing = _get_module_timing_for_side(
                module_name=left_full_name,
                preferred=baseline_module_timing,
                fallback=unknown_module_timing,
            )
            if left_timing is None:
                continue
            left_duration_ns = max(1, int(left_timing["duration_ns"]))
            left_start_offset_ns = left_child_cursor_ns
            pair_child_events.append(
                {
                    "name": left_full_name,
                    "start_ns": cursor_ns + left_start_offset_ns,
                    "end_ns": cursor_ns + left_start_offset_ns + left_duration_ns,
                    "pid": 1,
                    "tid": 0,
                    "thread_name": "aligned_stack",
                    "process_name": "baseline_aligned_stack",
                    "args": dict(
                        shared_args,
                        role="baseline",
                        child_match_status="ambiguous",
                        child_pair_index=None,
                        child_relative_name=left_relative_name,
                        child_ambiguous_index=ambiguous_index,
                    ),
                }
            )
            left_child_cursor_ns = max(left_child_cursor_ns, left_start_offset_ns + left_duration_ns)
            child_envelope_ns = max(child_envelope_ns, left_child_cursor_ns)

        unmatched_left_children = [str(item) for item in child_alignment.get("unmatched", {}).get("left", [])]
        unmatched_right_children = [str(item) for item in child_alignment.get("unmatched", {}).get("right", [])]

        for left_relative_name in unmatched_left_children:
            left_full_name = left_lookup.get(left_relative_name)
            if not left_full_name:
                continue
            left_timing = _get_module_timing_for_side(
                module_name=left_full_name,
                preferred=baseline_module_timing,
                fallback=unknown_module_timing,
            )
            if left_timing is None:
                continue
            left_duration_ns = max(1, int(left_timing["duration_ns"]))
            left_start_offset_ns = left_child_cursor_ns
            pair_child_events.append(
                {
                    "name": left_full_name,
                    "start_ns": cursor_ns + left_start_offset_ns,
                    "end_ns": cursor_ns + left_start_offset_ns + left_duration_ns,
                    "pid": 1,
                    "tid": 0,
                    "thread_name": "aligned_stack",
                    "process_name": "baseline_aligned_stack",
                    "args": dict(
                        shared_args,
                        role="baseline",
                        child_match_status="unmatched",
                        child_pair_index=None,
                        child_relative_name=left_relative_name,
                    ),
                }
            )
            left_child_cursor_ns = max(left_child_cursor_ns, left_start_offset_ns + left_duration_ns)
            child_envelope_ns = max(child_envelope_ns, left_child_cursor_ns)

        for right_relative_name in unmatched_right_children:
            right_full_name = right_lookup.get(right_relative_name)
            if not right_full_name:
                continue
            right_timing = _get_module_timing_for_side(
                module_name=right_full_name,
                preferred=target_module_timing,
                fallback=unknown_module_timing,
            )
            if right_timing is None:
                continue
            right_duration_ns = max(1, int(right_timing["duration_ns"]))
            right_start_offset_ns = right_child_cursor_ns
            pair_child_events.append(
                {
                    "name": right_full_name,
                    "start_ns": cursor_ns + right_start_offset_ns,
                    "end_ns": cursor_ns + right_start_offset_ns + right_duration_ns,
                    "pid": 2,
                    "tid": 0,
                    "thread_name": "aligned_stack",
                    "process_name": "target_aligned_stack",
                    "args": dict(
                        shared_args,
                        role="target",
                        child_match_status="unmatched",
                        child_pair_index=None,
                        child_relative_name=right_relative_name,
                    ),
                }
            )
            right_child_cursor_ns = max(right_child_cursor_ns, right_start_offset_ns + right_duration_ns)
            child_envelope_ns = max(child_envelope_ns, right_child_cursor_ns)

        synthetic_from_children_ns = 0
        if child_envelope_ns > 0:
            synthetic_from_children_ns = (child_envelope_ns * 101 + 99) // 100

        baseline_duration_ns = max(real_baseline_duration_ns, synthetic_from_children_ns)
        target_duration_ns = max(real_target_duration_ns, synthetic_from_children_ns)

        parent_common_args = dict(
            shared_args,
            child_match_status="parent",
            child_pair_index=None,
            parent_mode="synthetic_aligned",
            child_envelope_ns=int(child_envelope_ns),
        )
        aligned_timeline.append(
            {
                "name": baseline_module,
                "start_ns": cursor_ns,
                "end_ns": cursor_ns + baseline_duration_ns,
                "pid": 1,
                "tid": 0,
                "thread_name": "aligned_stack",
                "process_name": "baseline_aligned_stack",
                "args": dict(
                    parent_common_args,
                    role="baseline",
                    real_parent_duration_ns=int(real_baseline_duration_ns),
                    synthetic_parent_duration_ns=int(baseline_duration_ns),
                ),
            }
        )
        aligned_timeline.append(
            {
                "name": target_module,
                "start_ns": cursor_ns,
                "end_ns": cursor_ns + target_duration_ns,
                "pid": 2,
                "tid": 0,
                "thread_name": "aligned_stack",
                "process_name": "target_aligned_stack",
                "args": dict(
                    parent_common_args,
                    role="target",
                    real_parent_duration_ns=int(real_target_duration_ns),
                    synthetic_parent_duration_ns=int(target_duration_ns),
                ),
            }
        )

        aligned_timeline.extend(pair_child_events)

        cursor_ns += max(baseline_duration_ns, target_duration_ns) + gap_ns

    return build_chrome_trace(aligned_timeline)


def render_summary_markdown(summary: Mapping[str, Any]) -> str:
    alignment = _require_mapping(summary.get("alignment", {}), context="summary.alignment")
    alignment_counts = _require_mapping(alignment.get("counts", {}), context="summary.alignment.counts")
    diff = _require_mapping(summary.get("diff", {}), context="summary.diff")
    diff_counts = _require_mapping(diff.get("counts", {}), context="summary.diff.counts")
    rank_aggregation = _require_mapping(diff.get("rank_aggregation", {}), context="summary.diff.rank_aggregation")

    lines: list[str] = []
    lines.append("# Compare Performance Summary")
    lines.append("")

    lines.append("## Config Echo")
    config = _require_mapping(summary.get("config", {}), context="summary.config")
    if config:
        for key, value in sorted(config.items(), key=lambda item: str(item[0])):
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines.append("- (empty)")
    lines.append("")

    lines.append("## Alignment Statistics")
    for key in sorted(alignment_counts.keys()):
        lines.append(f"- {key}: {alignment_counts[key]}")

    unmatched = _require_mapping(alignment.get("unmatched", {}), context="summary.alignment.unmatched")
    left_unmatched = [str(item) for item in unmatched.get("left", [])]
    right_unmatched = [str(item) for item in unmatched.get("right", [])]
    lines.append(f"- unmatched_left: {len(left_unmatched)}")
    lines.append(f"- unmatched_right: {len(right_unmatched)}")
    lines.append("")

    lines.append("## Diff Statistics")
    for key in sorted(diff_counts.keys()):
        lines.append(f"- {key}: {diff_counts[key]}")
    lines.append(f"- rank_aggregation_enabled: {bool(rank_aggregation.get('enabled', False))}")
    lines.append("")

    lines.append("## Top Regressions")
    regressions = summary.get("top_regressions", [])
    if regressions:
        for item in regressions:
            lines.append(
                "- "
                f"{item.get('baseline_module', '')} -> {item.get('target_module', '')}: "
                f"delta_ms={item.get('delta_ms', 0.0)}, delta_pct={item.get('delta_pct', 'NA')}"
            )
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("## Top Improvements")
    improvements = summary.get("top_improvements", [])
    if improvements:
        for item in improvements:
            lines.append(
                "- "
                f"{item.get('baseline_module', '')} -> {item.get('target_module', '')}: "
                f"delta_ms={item.get('delta_ms', 0.0)}, delta_pct={item.get('delta_pct', 'NA')}"
            )
    else:
        lines.append("- (none)")
    lines.append("")

    return "\n".join(lines)


def export_compare_result(
    *,
    timeline_events: Sequence[Mapping[str, Any]],
    alignment: Mapping[str, Any],
    diff_result: Mapping[str, Any],
    config_echo: Mapping[str, Any] | None = None,
    trace_json_path: str | None = None,
    aligned_trace_json_path: str | None = None,
    aligned_stack_trace_json_path: str | None = None,
    summary_json_path: str | None = None,
    summary_md_path: str | None = None,
    top_n: int = 5,
) -> dict[str, Any]:
    trace_payload = build_chrome_trace(timeline_events)
    aligned_trace_payload = build_aligned_chrome_trace(
        alignment=alignment,
        diff_result=diff_result,
    )
    aligned_stack_trace_payload = build_aligned_stack_chrome_trace(
        timeline_events=timeline_events,
        alignment=alignment,
        diff_result=diff_result,
    )
    summary_payload = build_summary(
        alignment=alignment,
        diff_result=diff_result,
        config_echo=config_echo,
        top_n=top_n,
    )
    summary_markdown = render_summary_markdown(summary_payload)

    if trace_json_path:
        trace_path = Path(trace_json_path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(
            json.dumps(trace_payload, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )

    if aligned_trace_json_path:
        aligned_trace_path = Path(aligned_trace_json_path)
        aligned_trace_path.parent.mkdir(parents=True, exist_ok=True)
        aligned_trace_path.write_text(
            json.dumps(aligned_trace_payload, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )

    if aligned_stack_trace_json_path:
        aligned_stack_trace_path = Path(aligned_stack_trace_json_path)
        aligned_stack_trace_path.parent.mkdir(parents=True, exist_ok=True)
        aligned_stack_trace_path.write_text(
            json.dumps(aligned_stack_trace_payload, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )

    if summary_json_path:
        summary_json = Path(summary_json_path)
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(
            json.dumps(summary_payload, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )

    if summary_md_path:
        summary_md = Path(summary_md_path)
        summary_md.parent.mkdir(parents=True, exist_ok=True)
        summary_md.write_text(summary_markdown, encoding="utf-8")

    return {
        "trace": trace_payload,
        "trace_aligned": aligned_trace_payload,
        "trace_aligned_stack": aligned_stack_trace_payload,
        "summary_json": summary_payload,
        "summary_md": summary_markdown,
    }
