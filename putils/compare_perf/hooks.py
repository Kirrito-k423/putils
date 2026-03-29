from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable

from putils.compare_perf.collector import TimingCollector
from putils.compare_perf.config import DEFAULT_THRESHOLD_SECONDS


@dataclass
class ModelHookSession:
    handles: list[Any]
    _module_contexts: dict[int, list[Any]]

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self._module_contexts.clear()


def register_model_hooks(
    model: Any,
    *,
    collector: TimingCollector,
    threshold_seconds: float = DEFAULT_THRESHOLD_SECONDS,
    scope_prefix: str | None = None,
    leaf_only: bool = True,
    module_filter: Callable[[str, Any], bool] | None = None,
) -> ModelHookSession:
    if collector is None:
        raise ValueError("collector must not be None")
    if threshold_seconds <= 0:
        raise ValueError("threshold_seconds must be > 0")
    if not hasattr(model, "named_modules"):
        raise ValueError("model must provide named_modules()")

    scope_prefix_value = "" if scope_prefix is None else str(scope_prefix).strip(".")

    module_contexts: dict[int, list[Any]] = {}
    handles: list[Any] = []

    for module_name, module in model.named_modules():
        if not module_name:
            continue

        if leaf_only and any(True for _ in module.children()):
            continue

        if module_filter is not None and not module_filter(module_name, module):
            continue

        scope_name = f"{scope_prefix_value}.{module_name}" if scope_prefix_value else module_name

        def _on_forward_pre(mod: Any, _args: Any, _kwargs: Any = None, *, _scope_name: str = scope_name) -> None:
            active_context = collector.compare_scope(
                scope_name=_scope_name,
                threshold_seconds=threshold_seconds,
            )
            active_context.__enter__()
            module_contexts.setdefault(id(mod), []).append(active_context)

        def _on_forward_post(
            mod: Any,
            _args: Any,
            _kwargs: Any = None,
            _output: Any = None,
            *,
            _scope_name: str = scope_name,
        ) -> None:
            stack = module_contexts.get(id(mod))
            if not stack:
                return
            active_context = stack.pop()
            active_context.__exit__(None, None, None)
            if not stack:
                module_contexts.pop(id(mod), None)

        pre_handle = module.register_forward_pre_hook(_on_forward_pre, with_kwargs=True)
        handles.append(pre_handle)

        try:
            post_handle = module.register_forward_hook(_on_forward_post, with_kwargs=True, always_call=True)
        except TypeError:
            post_handle = module.register_forward_hook(_on_forward_post, with_kwargs=True)
        handles.append(post_handle)

    return ModelHookSession(handles=handles, _module_contexts=module_contexts)


@contextmanager
def model_forward_timing(
    model: Any,
    *,
    collector: TimingCollector,
    threshold_seconds: float = DEFAULT_THRESHOLD_SECONDS,
    scope_prefix: str | None = None,
    leaf_only: bool = True,
    module_filter: Callable[[str, Any], bool] | None = None,
):
    session = register_model_hooks(
        model,
        collector=collector,
        threshold_seconds=threshold_seconds,
        scope_prefix=scope_prefix,
        leaf_only=leaf_only,
        module_filter=module_filter,
    )
    try:
        yield session
    finally:
        session.remove()
