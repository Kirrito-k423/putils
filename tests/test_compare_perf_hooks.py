# pyright: reportMissingImports=false
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch is required for compare_perf hook tests")
from torch import nn

from putils.compare_perf.collector import TimingCollector
from putils.compare_perf.hooks import model_forward_timing, register_model_hooks


class _HookTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.ModuleDict(
            {
                "layer": nn.ModuleList(
                    [
                        nn.ModuleDict({"attn": nn.Linear(8, 8)}),
                        nn.ModuleDict(
                            {
                                "mlp": nn.Sequential(
                                    nn.Linear(8, 8),
                                    nn.GELU(),
                                    nn.Linear(8, 8),
                                )
                            }
                        ),
                    ]
                )
            }
        )
        self.head = nn.ModuleDict({"proj": nn.Linear(8, 4)})

    def forward(self, x):
        x = self.encoder["layer"][0]["attn"](x)
        x = self.encoder["layer"][1]["mlp"](x)
        return self.head["proj"](x)


def test_register_model_hooks_collects_named_module_paths():
    model = _HookTestModel()
    collector = TimingCollector(sync_mode="none")

    session = register_model_hooks(
        model,
        collector=collector,
        threshold_seconds=1e-9,
        leaf_only=False,
        module_filter=lambda name, _module: name in {
            "encoder.layer.0.attn",
            "encoder.layer.1.mlp",
            "head.proj",
        },
    )

    x = torch.randn(2, 8)
    model(x)
    session.remove()

    summary = collector.summary
    assert summary["encoder.layer.0.attn"]["call_count"] == 1
    assert summary["encoder.layer.1.mlp"]["call_count"] == 1
    assert summary["head.proj"]["call_count"] == 1


def test_register_model_hooks_threshold_keeps_summary_but_drops_detail():
    model = _HookTestModel()
    collector = TimingCollector(sync_mode="none")

    session = register_model_hooks(
        model,
        collector=collector,
        threshold_seconds=1.0,
        leaf_only=True,
        module_filter=lambda name, _module: name == "head.proj",
    )

    x = torch.randn(2, 8)
    model(x)
    session.remove()

    assert collector.events == []
    assert collector.summary["head.proj"]["call_count"] == 1


def test_model_forward_timing_context_removes_hooks_on_exit():
    model = _HookTestModel()
    collector = TimingCollector(sync_mode="none")
    x = torch.randn(2, 8)

    with model_forward_timing(
        model,
        collector=collector,
        threshold_seconds=1e-9,
        leaf_only=True,
        module_filter=lambda name, _module: name == "head.proj",
    ):
        model(x)

    call_count_after_context = collector.summary["head.proj"]["call_count"]
    model(x)

    assert call_count_after_context == 1
    assert collector.summary["head.proj"]["call_count"] == 1
