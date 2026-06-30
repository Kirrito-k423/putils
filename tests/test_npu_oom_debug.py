from putils import npu_oom_debug


class TestNpuOomDebug:
    def test_fmt_bytes(self):
        assert npu_oom_debug._fmt_bytes(None) == "None"
        assert npu_oom_debug._fmt_bytes(1024**3) == "1.000 GiB"

    def test_snapshot_summary(self):
        snapshot = {
            "segments": [
                {
                    "total_size": 10 * 1024**3,
                    "blocks": [
                        {"size": 2 * 1024**3, "state": "active_allocated"},
                        {"size": 3 * 1024**3, "state": "inactive"},
                        {"size": 5 * 1024**3, "state": "inactive"},
                    ],
                }
            ]
        }

        summary = npu_oom_debug._snapshot_summary(snapshot)

        assert summary["segment_count"] == 1
        assert summary["block_count"] == 3
        assert summary["active_total_bytes"] == 2 * 1024**3
        assert summary["inactive_total_bytes"] == 8 * 1024**3
        assert summary["largest_inactive_block_bytes"] == 5 * 1024**3
