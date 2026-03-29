import json

from putils.compare_perf.cli import (
    EXIT_INVALID_ARGS,
    EXIT_MISSING_INPUT,
    EXIT_OK,
    EXIT_SCHEMA_ERROR,
    main,
)


def test_collect_diff_export_pipeline(tmp_path):
    baseline_log = tmp_path / "baseline.json"
    target_log = tmp_path / "target.json"
    trace_out = tmp_path / "trace.json"
    trace_aligned_out = tmp_path / "trace_aligned.json"
    trace_aligned_stack_out = tmp_path / "trace_aligned_stack.json"
    summary_json_out = tmp_path / "summary.json"
    summary_md_out = tmp_path / "summary.md"

    baseline_rc = main(
        [
            "collect",
            "--event",
            "encoder.layer.0.attn:100",
            "--event",
            "encoder.layer.1.mlp:40",
            "--output",
            str(baseline_log),
            "--tag",
            "baseline",
        ]
    )
    assert baseline_rc == EXIT_OK
    assert baseline_log.exists()

    target_rc = main(
        [
            "collect",
            "--event",
            "enc.block.0.attention:120",
            "--event",
            "enc.block.1.feedforward:20",
            "--output",
            str(target_log),
            "--tag",
            "target",
        ]
    )
    assert target_rc == EXIT_OK
    assert target_log.exists()

    diff_rc = main(
        [
            "diff",
            "--baseline",
            str(baseline_log),
            "--target",
            str(target_log),
            "--trace-out",
            str(trace_out),
            "--trace-aligned-out",
            str(trace_aligned_out),
            "--trace-aligned-stack-out",
            str(trace_aligned_stack_out),
            "--summary-json-out",
            str(summary_json_out),
            "--summary-md-out",
            str(summary_md_out),
            "--top-n",
            "2",
        ]
    )
    assert diff_rc == EXIT_OK

    assert trace_out.exists()
    assert trace_aligned_out.exists()
    assert trace_aligned_stack_out.exists()
    assert summary_json_out.exists()
    assert summary_md_out.exists()

    trace_payload = json.loads(trace_out.read_text(encoding="utf-8"))
    summary_payload = json.loads(summary_json_out.read_text(encoding="utf-8"))
    summary_markdown = summary_md_out.read_text(encoding="utf-8")

    assert trace_payload["displayTimeUnit"] == "us"
    assert summary_payload["alignment"]["counts"]["matched"] == 2
    assert summary_payload["diff"]["counts"]["compared_modules"] == 2
    assert "# Compare Performance Summary" in summary_markdown


def test_cli_error_codes_and_messages(tmp_path, capsys):
    missing_rc = main(
        [
            "diff",
            "--baseline",
            str(tmp_path / "missing-baseline.json"),
            "--target",
            str(tmp_path / "missing-target.json"),
        ]
    )
    missing_err = capsys.readouterr().err
    assert missing_rc == EXIT_MISSING_INPUT
    assert "missing input log" in missing_err

    valid_target = tmp_path / "target.json"
    collect_rc = main(
        [
            "collect",
            "--event",
            "enc.block.0.attention:20",
            "--output",
            str(valid_target),
        ]
    )
    assert collect_rc == EXIT_OK

    incompatible_baseline = tmp_path / "incompatible.json"
    incompatible_baseline.write_text(
        json.dumps(
            {
                "schema_version": 999,
                "events": [],
                "run_metadata": {},
                "alignment": {},
                "summary": {},
            }
        ),
        encoding="utf-8",
    )

    incompatible_rc = main(
        [
            "diff",
            "--baseline",
            str(incompatible_baseline),
            "--target",
            str(valid_target),
        ]
    )
    incompatible_err = capsys.readouterr().err
    assert incompatible_rc == EXIT_SCHEMA_ERROR
    assert "incompatible schema" in incompatible_err

    invalid_args_rc = main(
        [
            "collect",
            "--event",
            "bad_event_without_duration",
            "--output",
            str(tmp_path / "bad.json"),
        ]
    )
    invalid_args_err = capsys.readouterr().err
    assert invalid_args_rc == EXIT_INVALID_ARGS
    assert "expected <scope_name>:<duration_ms>" in invalid_args_err


def test_diff_rejects_corrupted_and_partial_logs(tmp_path, capsys):
    valid_target = tmp_path / "target.json"
    collect_rc = main(
        [
            "collect",
            "--event",
            "enc.block.0.attention:20",
            "--output",
            str(valid_target),
        ]
    )
    assert collect_rc == EXIT_OK

    corrupted_baseline = tmp_path / "corrupted.json"
    corrupted_baseline.write_text("{\"schema_version\":", encoding="utf-8")

    corrupted_rc = main(
        [
            "diff",
            "--baseline",
            str(corrupted_baseline),
            "--target",
            str(valid_target),
        ]
    )
    corrupted_err = capsys.readouterr().err
    assert corrupted_rc == EXIT_SCHEMA_ERROR
    assert "invalid json in input log" in corrupted_err

    partial_baseline = tmp_path / "partial.json"
    partial_baseline.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "events": [],
                "run_metadata": {},
                "alignment": {},
            }
        ),
        encoding="utf-8",
    )

    partial_rc = main(
        [
            "diff",
            "--baseline",
            str(partial_baseline),
            "--target",
            str(valid_target),
        ]
    )
    partial_err = capsys.readouterr().err
    assert partial_rc == EXIT_SCHEMA_ERROR
    assert "incompatible schema" in partial_err
    assert "Missing required field: summary" in partial_err


def test_diff_rejects_malformed_event_payload(tmp_path, capsys):
    baseline = tmp_path / "baseline.json"
    target = tmp_path / "target.json"

    baseline.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "events": [
                    {
                        "name": "encoder.layer.0.attn",
                        "start_ns": 100,
                    }
                ],
                "run_metadata": {},
                "alignment": {},
                "summary": {
                    "encoder.layer.0.attn": {
                        "call_count": 1,
                        "total_ns": 100,
                        "total_seconds": 0.0000001,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    target.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "events": [],
                "run_metadata": {},
                "alignment": {},
                "summary": {
                    "enc.block.0.attention": {
                        "call_count": 1,
                        "total_ns": 200,
                        "total_seconds": 0.0000002,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    rc = main(
        [
            "diff",
            "--baseline",
            str(baseline),
            "--target",
            str(target),
        ]
    )
    err = capsys.readouterr().err

    assert rc == EXIT_SCHEMA_ERROR
    assert "invalid event in baseline at index 0" in err
    assert "missing field 'end_ns'" in err


def test_diff_rejects_non_positive_top_n(tmp_path, capsys):
    baseline_log = tmp_path / "baseline.json"
    target_log = tmp_path / "target.json"

    baseline_rc = main(
        [
            "collect",
            "--event",
            "encoder.layer.0.attn:100",
            "--output",
            str(baseline_log),
        ]
    )
    assert baseline_rc == EXIT_OK

    target_rc = main(
        [
            "collect",
            "--event",
            "enc.block.0.attention:120",
            "--output",
            str(target_log),
        ]
    )
    assert target_rc == EXIT_OK

    diff_rc = main(
        [
            "diff",
            "--baseline",
            str(baseline_log),
            "--target",
            str(target_log),
            "--top-n",
            "0",
        ]
    )
    err = capsys.readouterr().err
    assert diff_rc == EXIT_INVALID_ARGS
    assert "argument --top-n: must be > 0" in err


def test_diff_preserves_summary_key_order_for_alignment(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline.json"
    target = tmp_path / "target.json"
    trace_out = tmp_path / "trace.json"
    summary_json_out = tmp_path / "summary.json"
    summary_md_out = tmp_path / "summary.md"

    baseline.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "events": [],
                "run_metadata": {},
                "alignment": {},
                "summary": {
                    "z.layer": {
                        "call_count": 1,
                        "total_ns": 10,
                        "total_seconds": 0.00000001,
                    },
                    "a.layer": {
                        "call_count": 1,
                        "total_ns": 20,
                        "total_seconds": 0.00000002,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    target.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "events": [],
                "run_metadata": {},
                "alignment": {},
                "summary": {
                    "y.layer": {
                        "call_count": 1,
                        "total_ns": 15,
                        "total_seconds": 0.000000015,
                    },
                    "b.layer": {
                        "call_count": 1,
                        "total_ns": 25,
                        "total_seconds": 0.000000025,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    captured: dict[str, list[str] | None] = {"left": None, "right": None}

    def _fake_align_modules(*, left_modules, right_modules, confirmed_mappings=None, cache=None):
        captured["left"] = list(left_modules)
        captured["right"] = list(right_modules)
        return {
            "matched": [],
            "ambiguous": [],
            "unmatched": {"left": list(left_modules), "right": list(right_modules)},
            "counts": {"matched": 0, "ambiguous": 0, "unmatched": len(left_modules) + len(right_modules)},
        }

    monkeypatch.setattr("putils.compare_perf.cli.align_modules", _fake_align_modules)

    rc = main(
        [
            "diff",
            "--baseline",
            str(baseline),
            "--target",
            str(target),
            "--trace-out",
            str(trace_out),
            "--summary-json-out",
            str(summary_json_out),
            "--summary-md-out",
            str(summary_md_out),
        ]
    )

    assert rc == EXIT_OK
    assert captured["left"] == ["z.layer", "a.layer"]
    assert captured["right"] == ["y.layer", "b.layer"]


def test_diff_default_output_dir_emits_aligned_stack_artifact(tmp_path):
    baseline_log = tmp_path / "baseline.json"
    target_log = tmp_path / "target.json"

    assert main([
        "collect",
        "--event",
        "encoder.layer.0.attn:100",
        "--event",
        "encoder.layer.1.mlp:40",
        "--output",
        str(baseline_log),
    ]) == EXIT_OK
    assert main([
        "collect",
        "--event",
        "enc.block.0.attention:120",
        "--event",
        "enc.block.1.feedforward:20",
        "--output",
        str(target_log),
    ]) == EXIT_OK

    output_dir = tmp_path / "artifacts"
    rc = main(
        [
            "diff",
            "--baseline",
            str(baseline_log),
            "--target",
            str(target_log),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert rc == EXIT_OK
    assert (output_dir / "compare_perf_trace.json").exists()
    assert (output_dir / "compare_perf_trace_aligned.json").exists()
    assert (output_dir / "compare_perf_trace_aligned_stack.json").exists()
