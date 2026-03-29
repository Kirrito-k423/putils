import pytest

from putils.compare_perf.diff import ZERO_BASELINE_MARKER, compute_diff


def test_delta_abs_pct_and_zero_division():
    alignment = {
        "matched": [
            {"left": "encoder.layer.0.attn", "right": "enc.block.0.attention"},
            {"left": "encoder.layer.1.mlp", "right": "enc.block.1.feedforward"},
        ],
        "unmatched": {
            "left": ["left_only.module"],
            "right": ["right_only.module"],
        },
    }

    baseline_summary = {
        "encoder.layer.0.attn": {
            "total_ns": 100_000_000,
            "call_count": 2,
        },
        "encoder.layer.1.mlp": {
            "total_ns": 0,
            "call_count": 1,
        },
        "left_only.module": {
            "total_ns": 500_000_000,
            "call_count": 100,
        },
    }
    target_summary = {
        "enc.block.0.attention": {
            "total_ns": 130_000_000,
            "call_count": 5,
        },
        "enc.block.1.feedforward": {
            "total_ns": 20_000_000,
            "call_count": 4,
        },
        "right_only.module": {
            "total_ns": 9_999_999,
            "call_count": 77,
        },
    }

    result = compute_diff(
        alignment=alignment,
        baseline_summary=baseline_summary,
        target_summary=target_summary,
    )

    assert result["counts"]["matched_pairs"] == 2
    assert result["counts"]["compared_modules"] == 2
    assert result["counts"]["excluded_pairs"] == 0

    by_left = {item["baseline_module"]: item for item in result["modules"]}
    assert "left_only.module" not in by_left
    assert "right_only.module" not in {item["target_module"] for item in result["modules"]}

    attn = by_left["encoder.layer.0.attn"]
    assert attn["delta_ms"] == 30.0
    assert attn["delta_pct"] == 30.0
    assert attn["delta_call_count"] == 3

    mlp = by_left["encoder.layer.1.mlp"]
    assert mlp["baseline_ms"] == 0.0
    assert mlp["target_ms"] == 20.0
    assert mlp["delta_ms"] == 20.0
    assert mlp["delta_pct"] == ZERO_BASELINE_MARKER
    assert mlp["delta_call_count"] == 3


def test_rank_aggregation_percentiles():
    alignment = {
        "matched": [
            {"left": "encoder.layer.0.attn", "right": "enc.block.0.attention"},
        ]
    }

    baseline_summary = {
        "encoder.layer.0.attn": {
            "total_ns": 250_000_000,
            "call_count": 4,
        },
    }
    target_summary = {
        "enc.block.0.attention": {
            "total_ns": 300_000_000,
            "call_count": 4,
        },
    }

    result = compute_diff(
        alignment=alignment,
        baseline_summary=baseline_summary,
        target_summary=target_summary,
        baseline_rank_summary={
            "encoder.layer.0.attn": [100_000_000, 200_000_000, 300_000_000, 400_000_000],
        },
        target_rank_summary={
            "enc.block.0.attention": [120_000_000, 220_000_000, 320_000_000, 520_000_000],
        },
        enable_rank_aggregation=True,
    )

    agg = result["rank_aggregation"]
    assert agg["enabled"] is True
    assert agg["percentiles"] == ["p50", "p95"]
    assert len(agg["modules"]) == 1
    assert agg["excluded"] == []

    module_agg = agg["modules"][0]
    assert module_agg["baseline"]["p50_ms"] == 250.0
    assert module_agg["target"]["p50_ms"] == 270.0
    assert module_agg["delta"]["p50_ms"] == 20.0
    assert module_agg["delta"]["p50_pct"] == 8.0

    assert module_agg["baseline"]["p95_ms"] == pytest.approx(385.0)
    assert module_agg["target"]["p95_ms"] == pytest.approx(490.0)
    assert module_agg["delta"]["p95_ms"] == pytest.approx(105.0)
    assert module_agg["delta"]["p95_pct"] == pytest.approx(27.272727)
