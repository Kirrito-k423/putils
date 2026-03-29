# pyright: reportMissingImports=false

import os
import re
from datetime import datetime, timezone
from unittest.mock import patch

from putils.compare_perf.config import (
    ComparePerfTimingConfig,
    collect_runtime_metadata,
    make_timestamped_log_filename,
)


def test_default_and_override_threshold():
    default_cfg = ComparePerfTimingConfig()
    assert default_cfg.threshold_seconds == 0.1

    override_cfg = ComparePerfTimingConfig(threshold_seconds=0.25)
    assert override_cfg.threshold_seconds == 0.25


def test_timestamped_unique_log_filename(tmp_path):
    fixed_utc = datetime(2026, 3, 29, 8, 30, 1, 123456, tzinfo=timezone.utc)

    with patch("putils.compare_perf.config.time.time_ns", side_effect=[111, 222]):
        first = make_timestamped_log_filename(output_dir=str(tmp_path), now_utc=fixed_utc)
        second = make_timestamped_log_filename(output_dir=str(tmp_path), now_utc=fixed_utc)

    assert first != second

    first_name = os.path.basename(first)
    second_name = os.path.basename(second)

    expected_pattern = r"^compare_perf_\d{8}T\d{6}\.\d{6}Z_\d+_\d+\.json$"
    assert re.match(expected_pattern, first_name)
    assert re.match(expected_pattern, second_name)

    assert "20260329T083001.123456Z" in first_name
    assert "20260329T083001.123456Z" in second_name


def test_runtime_metadata_includes_required_fields():
    metadata = collect_runtime_metadata(now_utc=datetime(2026, 3, 29, 8, 30, 1, tzinfo=timezone.utc))

    assert metadata["backend"] is not None
    assert metadata["device"] in {"cpu", "cuda", "npu"}
    assert isinstance(metadata["rank"], int)
    assert isinstance(metadata["world_size"], int)
    assert isinstance(metadata["host"], str) and metadata["host"]
    assert isinstance(metadata["pid"], int)
    assert isinstance(metadata["commit"], str) and metadata["commit"]
    assert metadata["timestamp_utc"] == "20260329T083001.000000Z"
    assert isinstance(metadata["timestamp_unix_ns"], int)
