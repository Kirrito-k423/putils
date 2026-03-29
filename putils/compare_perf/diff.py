from __future__ import annotations

import math
from typing import Any, Mapping


ZERO_BASELINE_MARKER = "INF_OR_UNDEFINED"


def _to_ns(summary_item: Mapping[str, Any]) -> int:
    if "total_ns" in summary_item:
        return int(summary_item["total_ns"])
    if "total_seconds" in summary_item:
        return int(float(summary_item["total_seconds"]) * 1_000_000_000)
    return 0


def _to_call_count(summary_item: Mapping[str, Any]) -> int:
    return int(summary_item.get("call_count", 0))


def _delta_pct_or_marker(*, baseline_ns: int, target_ns: int) -> float | str:
    if baseline_ns == 0:
        return ZERO_BASELINE_MARKER
    return round(((target_ns - baseline_ns) / baseline_ns) * 100.0, 6)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if q < 0.0 or q > 1.0:
        raise ValueError("q must be in [0, 1]")

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]

    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _to_rank_ns_values(rank_values: list[Any]) -> list[int]:
    converted: list[int] = []
    for item in rank_values:
        if isinstance(item, (int, float)):
            converted.append(int(item))
            continue
        if isinstance(item, Mapping):
            converted.append(_to_ns(item))
    return converted


def _build_rank_stats(module_values_ns: list[int]) -> dict[str, float]:
    values_ms = [float(value) / 1_000_000.0 for value in module_values_ns]
    return {
        "p50_ms": round(_percentile(values_ms, 0.50), 6),
        "p95_ms": round(_percentile(values_ms, 0.95), 6),
    }


def compute_diff(
    *,
    alignment: Mapping[str, Any],
    baseline_summary: Mapping[str, Mapping[str, Any]],
    target_summary: Mapping[str, Mapping[str, Any]],
    baseline_rank_summary: Mapping[str, list[Any]] | None = None,
    target_rank_summary: Mapping[str, list[Any]] | None = None,
    enable_rank_aggregation: bool = False,
) -> dict[str, Any]:
    matched = alignment.get("matched", [])

    modules: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []

    for item in matched:
        if not isinstance(item, Mapping):
            continue

        left_module = str(item.get("left", ""))
        right_module = str(item.get("right", ""))
        if not left_module or not right_module:
            continue

        left_stats = baseline_summary.get(left_module)
        right_stats = target_summary.get(right_module)
        if left_stats is None or right_stats is None:
            excluded.append(
                {
                    "baseline_module": left_module,
                    "target_module": right_module,
                    "reason": "missing_summary",
                }
            )
            continue

        baseline_ns = _to_ns(left_stats)
        target_ns = _to_ns(right_stats)

        baseline_ms = round(float(baseline_ns) / 1_000_000.0, 6)
        target_ms = round(float(target_ns) / 1_000_000.0, 6)
        delta_ms = round(target_ms - baseline_ms, 6)

        baseline_call_count = _to_call_count(left_stats)
        target_call_count = _to_call_count(right_stats)

        modules.append(
            {
                "baseline_module": left_module,
                "target_module": right_module,
                "baseline_ms": baseline_ms,
                "target_ms": target_ms,
                "delta_ms": delta_ms,
                "delta_pct": _delta_pct_or_marker(baseline_ns=baseline_ns, target_ns=target_ns),
                "baseline_call_count": baseline_call_count,
                "target_call_count": target_call_count,
                "delta_call_count": target_call_count - baseline_call_count,
            }
        )

    rank_aggregation: dict[str, Any] = {
        "enabled": bool(enable_rank_aggregation),
        "percentiles": ["p50", "p95"] if enable_rank_aggregation else [],
        "modules": [],
        "excluded": [],
    }
    if enable_rank_aggregation:
        baseline_rank_summary = baseline_rank_summary or {}
        target_rank_summary = target_rank_summary or {}

        for module_item in modules:
            left_module = module_item["baseline_module"]
            right_module = module_item["target_module"]
            left_raw_values = baseline_rank_summary.get(left_module)
            right_raw_values = target_rank_summary.get(right_module)

            if left_raw_values is None or right_raw_values is None:
                rank_aggregation["excluded"].append(
                    {
                        "baseline_module": left_module,
                        "target_module": right_module,
                        "reason": "missing_rank_summary",
                    }
                )
                continue

            left_values = _to_rank_ns_values(left_raw_values)
            right_values = _to_rank_ns_values(right_raw_values)
            if not left_values or not right_values:
                rank_aggregation["excluded"].append(
                    {
                        "baseline_module": left_module,
                        "target_module": right_module,
                        "reason": "empty_rank_values",
                    }
                )
                continue

            baseline_percentiles = _build_rank_stats(left_values)
            target_percentiles = _build_rank_stats(right_values)

            p50_delta_ms = round(target_percentiles["p50_ms"] - baseline_percentiles["p50_ms"], 6)
            p95_delta_ms = round(target_percentiles["p95_ms"] - baseline_percentiles["p95_ms"], 6)

            rank_aggregation["modules"].append(
                {
                    "baseline_module": left_module,
                    "target_module": right_module,
                    "baseline": baseline_percentiles,
                    "target": target_percentiles,
                    "delta": {
                        "p50_ms": p50_delta_ms,
                        "p95_ms": p95_delta_ms,
                        "p50_pct": _delta_pct_or_marker(
                            baseline_ns=int(round(baseline_percentiles["p50_ms"] * 1_000_000.0)),
                            target_ns=int(round(target_percentiles["p50_ms"] * 1_000_000.0)),
                        ),
                        "p95_pct": _delta_pct_or_marker(
                            baseline_ns=int(round(baseline_percentiles["p95_ms"] * 1_000_000.0)),
                            target_ns=int(round(target_percentiles["p95_ms"] * 1_000_000.0)),
                        ),
                    },
                }
            )

    return {
        "modules": modules,
        "counts": {
            "matched_pairs": len(matched),
            "compared_modules": len(modules),
            "excluded_pairs": len(excluded),
        },
        "excluded": excluded,
        "rank_aggregation": rank_aggregation,
    }
