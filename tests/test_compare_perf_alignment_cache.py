from putils.compare_perf.align import AlignmentMappingCache, align_modules


def test_alignment_cache_hit_on_second_run(tmp_path):
    left_modules = ["x.layer.3.unit"]
    right_modules = ["y.layer.3.unit_a", "y.layer.3.unit_b"]

    cache = AlignmentMappingCache(tmp_path / "alignment_cache.json")

    first = align_modules(
        left_modules=left_modules,
        right_modules=right_modules,
        confirmed_mappings={"x.layer.3.unit": "y.layer.3.unit_b"},
        cache=cache,
    )
    assert first["counts"]["matched"] == 1
    assert first["matched"][0]["status"] == "confirmed"
    assert first["matched"][0]["source"] == "manual"

    second = align_modules(
        left_modules=left_modules,
        right_modules=right_modules,
        cache=cache,
    )
    assert second["counts"]["matched"] == 1
    assert second["counts"]["ambiguous"] == 0
    assert second["matched"][0]["left"] == "x.layer.3.unit"
    assert second["matched"][0]["right"] == "y.layer.3.unit_b"
    assert second["matched"][0]["status"] == "confirmed"
    assert second["matched"][0]["source"] == "cache"
