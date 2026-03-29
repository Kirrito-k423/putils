from putils.compare_perf.align import align_modules


def test_alignment_classification_counts():
    left_modules = [
        "encoder.layer.0.attn",
        "encoder.layer.1.mlp",
        "x.layer.3.unit",
        "totally_unknown_module",
    ]
    right_modules = [
        "enc.block.0.attention",
        "enc.block.1.mlp",
        "y.layer.3.unit_a",
        "y.layer.3.unit_b",
        "other.layer.8.qq",
    ]

    result = align_modules(left_modules=left_modules, right_modules=right_modules)

    assert result["counts"]["matched"] == 2
    assert result["counts"]["ambiguous"] == 1
    assert result["counts"]["unmatched"] == 4

    matched_left = {item["left"] for item in result["matched"]}
    assert "encoder.layer.0.attn" in matched_left
    assert "encoder.layer.1.mlp" in matched_left

    matched_detail = next(item for item in result["matched"] if item["left"] == "encoder.layer.1.mlp")
    assert set(matched_detail["score_components"].keys()) >= {
        "token",
        "path",
        "type",
        "layer",
        "order",
        "weights",
        "left_module_type",
        "right_module_type",
    }
    assert matched_detail["score_components"]["left_module_type"] == "feedforward"
    assert matched_detail["score_components"]["right_module_type"] == "feedforward"

    assert result["ambiguous"][0]["left"] == "x.layer.3.unit"
    assert result["ambiguous"][0]["status"] == "ambiguous"
    assert len(result["ambiguous"][0]["candidates"]) == 2
    assert "score_components" in result["ambiguous"][0]["candidates"][0]

    assert "totally_unknown_module" in result["unmatched"]["left"]
    assert "other.layer.8.qq" in result["unmatched"]["right"]
