from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


pytest.importorskip("torch", reason="torch is required for torch e2e compare_perf example")


def test_torch_e2e_example_detects_sleep_regression(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "examples" / "compare_perf_torch_e2e.py"
    output_dir = tmp_path / "torch_e2e"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-dir",
            str(output_dir),
            "--steps",
            "4",
            "--sleep-seconds",
            "0.02",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    baseline_path = output_dir / "baseline.json"
    target_path = output_dir / "target.json"
    trace_path = output_dir / "trace.json"
    trace_aligned_path = output_dir / "trace_aligned.json"
    trace_aligned_stack_path = output_dir / "trace_aligned_stack.json"
    summary_json_path = output_dir / "summary.json"
    summary_md_path = output_dir / "summary.md"

    assert baseline_path.exists()
    assert target_path.exists()
    assert trace_path.exists()
    assert trace_aligned_path.exists()
    assert trace_aligned_stack_path.exists()
    assert summary_json_path.exists()
    assert summary_md_path.exists()

    baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    target_payload = json.loads(target_path.read_text(encoding="utf-8"))
    assert baseline_payload["run_metadata"]["tag"] == "baseline"
    assert target_payload["run_metadata"]["tag"] == "target"
    assert baseline_payload["run_metadata"]["source"] == "compare_perf_torch_e2e_example"
    assert target_payload["run_metadata"]["source"] == "compare_perf_torch_e2e_example"
    assert baseline_payload["run_metadata"]["step"] == 4
    assert target_payload["run_metadata"]["step"] == 4

    summary = json.loads(summary_json_path.read_text(encoding="utf-8"))
    top_regressions = summary.get("top_regressions", [])
    regression_modules = {item.get("baseline_module") for item in top_regressions}
    assert "encoder.layer.1.mlp" in regression_modules

    trace_payload = json.loads(trace_path.read_text(encoding="utf-8"))
    trace_events = trace_payload.get("traceEvents", [])
    assert any(
        isinstance(event, dict)
        and event.get("ph") == "B"
        and str(event.get("name", "")).startswith("encoder.layer.1.mlp.")
        for event in trace_events
    )

    assert "top_regressions" in completed.stdout
    assert "trace_aligned" in completed.stdout
    assert "trace_aligned_stack" in completed.stdout
    assert "encoder.layer.1.mlp" in completed.stdout
