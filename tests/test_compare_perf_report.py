from putils.compare_perf.export import build_summary, render_summary_markdown


def test_summary_contains_required_sections():
    alignment = {
        "counts": {"matched": 3, "ambiguous": 1, "unmatched": 2},
        "unmatched": {
            "left": ["left.only"],
            "right": ["right.only"],
        },
    }
    diff_result = {
        "modules": [
            {
                "baseline_module": "encoder.layer.0.attn",
                "target_module": "enc.block.0.attention",
                "delta_ms": 12.0,
                "delta_pct": 10.0,
            },
            {
                "baseline_module": "encoder.layer.1.mlp",
                "target_module": "enc.block.1.feedforward",
                "delta_ms": -7.0,
                "delta_pct": -14.0,
            },
            {
                "baseline_module": "encoder.layer.2.norm",
                "target_module": "enc.block.2.norm",
                "delta_ms": 5.0,
                "delta_pct": 5.0,
            },
        ],
        "counts": {
            "matched_pairs": 3,
            "compared_modules": 3,
            "excluded_pairs": 0,
        },
        "rank_aggregation": {
            "enabled": True,
            "percentiles": ["p50", "p95"],
            "modules": [],
            "excluded": [],
        },
    }

    summary = build_summary(
        alignment=alignment,
        diff_result=diff_result,
        config_echo={
            "threshold_seconds": 0.1,
            "rank_strategy": "all",
            "top_n": 2,
        },
        top_n=2,
    )
    markdown = render_summary_markdown(summary)

    assert summary["top_regressions"][0]["baseline_module"] == "encoder.layer.0.attn"
    assert summary["top_improvements"][0]["baseline_module"] == "encoder.layer.1.mlp"

    assert "# Compare Performance Summary" in markdown
    assert "## Config Echo" in markdown
    assert "## Alignment Statistics" in markdown
    assert "## Top Regressions" in markdown
    assert "## Top Improvements" in markdown

    assert "matched: 3" in markdown
    assert "ambiguous: 1" in markdown
    assert "unmatched_left: 1" in markdown
    assert "rank_aggregation_enabled: True" in markdown
    assert "encoder.layer.0.attn -> enc.block.0.attention" in markdown
    assert "encoder.layer.1.mlp -> enc.block.1.feedforward" in markdown
