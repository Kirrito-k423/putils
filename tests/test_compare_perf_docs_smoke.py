from __future__ import annotations

import re
from pathlib import Path

from putils.compare_perf.cli import _build_parser


def _compare_perf_readme_text() -> str:
    project_root = Path(__file__).resolve().parents[1]
    return (project_root / "putils" / "compare_perf" / "README.md").read_text(encoding="utf-8")


def test_compare_perf_docs_include_minimal_integration_examples():
    readme = _compare_perf_readme_text()

    assert "## compare_perf 快速接入与结果解读" in readme
    assert "with compare_perf(\"forward_step\", collector=collector, threshold_seconds=0.05):" in readme
    assert "@compare_perf(\"post_process\", collector=collector, threshold_seconds=0.05)" in readme
    assert "collector.events" in readme
    assert "collector.summary" in readme
    assert (
        "from putils.compare_perf import TimingCollector, compare_perf, dump_compare_perf_snapshot"
        in readme
    )
    assert "默认只会把数据写到内存里的 `collector`，不会自动写文件" in readme
    assert "def dump_compare_perf_snapshot(" not in readme
    assert "for step in range(1, 51):" in readme
    assert "if step % snapshot_every == 0:" in readme
    assert "dump_compare_perf_snapshot(" in readme
    assert "output_dir=output_dir" in readme
    assert "step=step" in readme
    assert "tag=tag" in readme
    assert "`threshold_seconds` 可能让 `events` 在某些 step 为空" in readme
    assert "### 8) 真实 Torch E2E 示例（前向+反向+优化器）" in readme
    assert "python3 examples/compare_perf_torch_e2e.py" in readme
    assert "--output-dir /tmp/compare-perf-e2e" in readme
    assert "trace_aligned_stack.json" in readme
    assert "summary.json" in readme
    assert "top_regressions" in readme
    assert "encoder.layer.1.mlp" in readme


def test_compare_perf_docs_collect_diff_workflow_matches_cli():
    readme = _compare_perf_readme_text()

    required_doc_snippets = [
        "python3 -m putils.compare_perf.cli collect",
        "python3 -m putils.compare_perf.cli diff",
        "--event encoder.layer.0.attn:100",
        "--event enc.block.0.attention:120",
        "--baseline baseline.json",
        "--target target.json",
        "--trace-out compare_perf_trace.json",
        "--trace-aligned-out trace_aligned.json",
        "--trace-aligned-stack-out trace_aligned_stack.json",
        "--summary-json-out compare_perf_summary.json",
        "--summary-md-out compare_perf_summary.md",
        "--top-n 5",
        "--alignment-cache .cache/compare_perf_alignment.json",
        "--mapping encoder.layer.1.mlp=enc.block.1.feedforward",
        "--enable-rank-aggregation",
        "parent_mode=synthetic_aligned",
        "child_match_status",
        "matched",
        "ambiguous",
        "unmatched",
        "parent",
    ]
    for snippet in required_doc_snippets:
        assert snippet in readme

    parser = _build_parser()

    collect_args = parser.parse_args(
        [
            "collect",
            "--event",
            "encoder.layer.0.attn:100",
            "--event",
            "encoder.layer.1.mlp:40",
            "--output",
            "baseline.json",
            "--tag",
            "baseline",
        ]
    )
    assert collect_args.command == "collect"
    assert collect_args.output == "baseline.json"

    diff_args = parser.parse_args(
        [
            "diff",
            "--baseline",
            "baseline.json",
            "--target",
            "target.json",
            "--trace-out",
            "compare_perf_trace.json",
            "--trace-aligned-out",
            "trace_aligned.json",
            "--trace-aligned-stack-out",
            "trace_aligned_stack.json",
            "--summary-json-out",
            "compare_perf_summary.json",
            "--summary-md-out",
            "compare_perf_summary.md",
            "--top-n",
            "5",
            "--alignment-cache",
            ".cache/compare_perf_alignment.json",
            "--mapping",
            "encoder.layer.1.mlp=enc.block.1.feedforward",
            "--enable-rank-aggregation",
        ]
    )
    assert diff_args.command == "diff"
    assert diff_args.top_n == 5
    assert diff_args.enable_rank_aggregation is True
    assert diff_args.trace_aligned_out == "trace_aligned.json"
    assert diff_args.trace_aligned_stack_out == "trace_aligned_stack.json"


def test_troubleshooting_sections_present():
    readme = _compare_perf_readme_text()

    troubleshooting_match = re.search(
        r"### 7\) 常见错误排查\n(?P<body>[\s\S]+)$",
        readme,
    )
    assert troubleshooting_match is not None
    troubleshooting_body = troubleshooting_match.group("body")

    categories = re.findall(r"^#### [A-Z]\. .+$", troubleshooting_body, flags=re.MULTILINE)
    assert len(categories) >= 5

    required_categories = [
        "#### A. ERROR[2] invalid --event format",
        "#### B. ERROR[2] invalid --mapping format",
        "#### C. ERROR[10] missing input log",
        "#### D. ERROR[11] invalid json in input log",
        "#### E. ERROR[11] incompatible schema in input log",
        "#### F. Python API ValueError: threshold/scope/sync_mode",
    ]
    for category in required_categories:
        assert category in troubleshooting_body
