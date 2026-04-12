from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch is required for accuracy hook tests")
from torch import nn

from putils import accuracy


pytestmark = pytest.mark.requires_torch


class _AccuracyHookModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_tower = nn.ModuleDict(
            {
                "blocks": nn.ModuleList(
                    [
                        nn.Linear(8, 8),
                        nn.Sequential(nn.Linear(8, 8), nn.GELU()),
                    ]
                )
            }
        )
        self.language_model = nn.ModuleDict(
            {
                "layers": nn.ModuleList(
                    [
                        nn.Linear(8, 8),
                        nn.Sequential(nn.Linear(8, 8), nn.ReLU()),
                    ]
                )
            }
        )


def test_hook_for_model_registers_only_included_modules(monkeypatch):
    model = _AccuracyHookModel()
    captured_names = []

    monkeypatch.setattr(accuracy, "ifdebug", lambda: True)

    def _fake_hook_func(name, _module, _file_path):
        captured_names.append(name)

        def _hook(*_args, **_kwargs):
            return None

        return _hook

    monkeypatch.setattr(accuracy, "hook_func", _fake_hook_func)

    handles = accuracy.hook_for_model(
        model,
        include_list=["vision_tower.blocks.1", "language_model.layers.0", "missing.layer"],
    )

    assert len(handles) == 4
    assert captured_names == [
        "[forward]: vision_tower.blocks.1",
        "[backward]: vision_tower.blocks.1",
        "[forward]: language_model.layers.0",
        "[backward]: language_model.layers.0",
    ]

    for handle in handles:
        handle.remove()


def test_hook_for_model_skips_when_include_list_is_empty(monkeypatch):
    model = _AccuracyHookModel()

    monkeypatch.setattr(accuracy, "ifdebug", lambda: True)

    handles = accuracy.hook_for_model(model, include_list=[])

    assert handles == []
