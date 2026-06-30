"""Ascend NPU OOM diagnostics for FSDP/all-gather failures.

The main entrypoint is :func:`dump_npu_oom_debug`. Call it from an OOM
``except`` block to persist allocator stats, memory summaries, snapshots, and
optionally a post-OOM allocation probe in the same process.
"""

from __future__ import annotations

import gc
import json
import math
import os
import time
import traceback
from pathlib import Path
from typing import Any, Callable


MIB = 1024**2
GIB = 1024**3


def _rank() -> str:
    return os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "unknown"


def _safe_call(name: str, fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception as exc:
        return {"__error__": f"{name}: {type(exc).__name__}: {exc}"}


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _fmt_bytes(num_bytes: int | float | None) -> str:
    if num_bytes is None:
        return "None"
    return f"{num_bytes / GIB:.3f} GiB"


def _snapshot_summary(snapshot: Any) -> dict[str, Any]:
    segments = snapshot.get("segments") or [] if isinstance(snapshot, dict) else []
    largest_free = 0
    inactive_total = 0
    active_total = 0
    segment_total = 0
    block_count = 0
    free_blocks = 0
    active_blocks = 0
    inactive_blocks = 0
    states: dict[str, int] = {}

    for seg in segments:
        if not isinstance(seg, dict):
            continue
        segment_total += int(seg.get("total_size") or 0)
        for block in seg.get("blocks") or []:
            if not isinstance(block, dict):
                continue
            block_count += 1
            size = int(block.get("size") or 0)
            state = str(block.get("state") or "unknown")
            states[state] = states.get(state, 0) + 1
            if state.startswith("active"):
                active_total += size
                active_blocks += 1
            else:
                inactive_total += size
                inactive_blocks += 1
                free_blocks += 1
                largest_free = max(largest_free, size)

    return {
        "segment_count": len(segments),
        "segment_total_bytes": segment_total,
        "segment_total_gib": segment_total / GIB,
        "block_count": block_count,
        "active_blocks": active_blocks,
        "inactive_blocks": inactive_blocks,
        "free_blocks": free_blocks,
        "active_total_bytes": active_total,
        "inactive_total_bytes": inactive_total,
        "largest_inactive_block_bytes": largest_free,
        "largest_inactive_block_gib": largest_free / GIB,
        "block_states": states,
    }


def _get_npu_modules() -> tuple[Any, Any, Any]:
    import torch

    try:
        import torch_npu
    except Exception:
        torch_npu = None
    npu = getattr(torch, "npu", None)
    if npu is None and torch_npu is not None:
        npu = getattr(torch_npu, "npu", None)
    return torch, torch_npu, npu


def _mem_get_info(npu: Any, device: Any = None) -> Any:
    if npu is None or not hasattr(npu, "mem_get_info"):
        return None
    try:
        return npu.mem_get_info(device) if device is not None else npu.mem_get_info()
    except TypeError:
        return npu.mem_get_info()


def _current_device(npu: Any) -> Any:
    if npu is None or not hasattr(npu, "current_device"):
        return None
    return npu.current_device()


def _memory_value(npu: Any, attr: str, device: Any = None) -> Any:
    if npu is None or not hasattr(npu, attr):
        return None
    fn = getattr(npu, attr)
    try:
        return fn(device) if device is not None else fn()
    except TypeError:
        return fn()


def _take_snapshot(npu: Any) -> Any:
    if npu is None:
        return None
    memory_mod = getattr(npu, "memory", None)
    for owner in (memory_mod, npu):
        if owner is None:
            continue
        for name in ("_snapshot", "memory_snapshot"):
            if hasattr(owner, name):
                return getattr(owner, name)()
    return None


def _dump_snapshot_file(npu: Any, path: Path) -> Any:
    if npu is None:
        return None
    memory_mod = getattr(npu, "memory", None)
    for owner in (memory_mod, npu):
        if owner is None:
            continue
        for name in ("_dump_snapshot", "dump_snapshot"):
            if hasattr(owner, name):
                return getattr(owner, name)(str(path))
    return None


def _probe_max_alloc_bytes(torch: Any, npu: Any, low_gib: float, high_gib: float) -> dict[str, Any]:
    """Binary search the largest bf16 tensor allocation in the current process."""

    result: dict[str, Any] = {"enabled": True, "low_gib": low_gib, "high_gib": high_gib}
    if npu is None:
        result["error"] = "torch.npu is unavailable"
        return result
    dtype = getattr(torch, "bfloat16", torch.float16)
    element_size = torch.tensor([], dtype=dtype).element_size()

    def try_alloc(size_gib: float) -> tuple[bool, str | None]:
        numel = int(size_gib * GIB // element_size)
        try:
            tensor = torch.empty(numel, dtype=dtype, device="npu")
            if hasattr(npu, "synchronize"):
                npu.synchronize()
            del tensor
            gc.collect()
            if hasattr(npu, "empty_cache"):
                npu.empty_cache()
            return True, None
        except Exception as exc:
            gc.collect()
            if hasattr(npu, "empty_cache"):
                _safe_call("empty_cache_after_probe_fail", npu.empty_cache)
            return False, f"{type(exc).__name__}: {exc}"

    lo = low_gib
    hi = high_gib
    last_error = None
    for _ in range(8):
        mid = (lo + hi) / 2
        ok, err = try_alloc(mid)
        if ok:
            lo = mid
        else:
            hi = mid
            last_error = err
    result["max_ok_gib_approx"] = lo
    result["first_fail_gib_approx"] = hi
    result["last_error"] = last_error
    return result


def dump_npu_oom_debug(
    tag: str,
    requested_bytes: int | None = None,
    error: BaseException | None = None,
    dump_dir: str | os.PathLike[str] | None = None,
    run_probe: bool | None = None,
) -> None:
    """Print and persist NPU OOM diagnostics.

    Args:
        tag: Short label for the failing allocation site.
        requested_bytes: Size of the failed allocation if known.
        error: Original OOM exception, captured only for logging.
        dump_dir: Output directory. Defaults to ``$NPU_OOM_DEBUG_DIR`` or
            ``/tmp/npu_oom_debug``.
        run_probe: Whether to run a destructive post-OOM allocation probe.
            If ``None``, reads ``NPU_OOM_DEBUG_PROBE=1``.

    The probe is intended for a process that is about to fail anyway. It tries
    additional large allocations in the same process to distinguish allocator
    state problems from hard single-allocation limits.
    """

    torch, _torch_npu, npu = _get_npu_modules()
    rank = _rank()
    ts = int(time.time())
    out_dir = Path(dump_dir or os.environ.get("NPU_OOM_DEBUG_DIR", "/tmp/npu_oom_debug"))
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _safe_call("current_device", lambda: _current_device(npu))
    snapshot_path = out_dir / f"npu_oom_rank{rank}_{ts}.pickle"
    summary_path = out_dir / f"npu_oom_rank{rank}_{ts}.json"
    text_path = out_dir / f"npu_oom_rank{rank}_{ts}.txt"

    snapshot = _safe_call("snapshot", lambda: _take_snapshot(npu))
    snapshot_summary = _snapshot_summary(snapshot)
    _safe_call("dump_snapshot_file", lambda: _dump_snapshot_file(npu, snapshot_path))

    stats = _safe_call("memory_stats", lambda: _memory_value(npu, "memory_stats", device))
    summary = _safe_call("memory_summary", lambda: _memory_value(npu, "memory_summary", device))
    mem_info = _safe_call("mem_get_info", lambda: _mem_get_info(npu, device))
    if isinstance(mem_info, tuple) and len(mem_info) == 2:
        free_bytes, total_bytes = mem_info
    else:
        free_bytes, total_bytes = None, None

    payload: dict[str, Any] = {
        "tag": tag,
        "rank": rank,
        "local_rank": os.environ.get("LOCAL_RANK"),
        "device": device,
        "requested_bytes": requested_bytes,
        "requested_gib": None if requested_bytes is None else requested_bytes / GIB,
        "mem_get_info": {"free_bytes": free_bytes, "total_bytes": total_bytes},
        "memory_allocated": _safe_call("memory_allocated", lambda: _memory_value(npu, "memory_allocated", device)),
        "memory_reserved": _safe_call("memory_reserved", lambda: _memory_value(npu, "memory_reserved", device)),
        "max_memory_allocated": _safe_call(
            "max_memory_allocated", lambda: _memory_value(npu, "max_memory_allocated", device)
        ),
        "max_memory_reserved": _safe_call(
            "max_memory_reserved", lambda: _memory_value(npu, "max_memory_reserved", device)
        ),
        "snapshot_summary": snapshot_summary,
        "stats": stats,
        "error": None if error is None else f"{type(error).__name__}: {error}",
        "traceback": None if error is None else "".join(traceback.format_exception(error)),
        "env": {
            "PYTORCH_NPU_ALLOC_CONF": os.environ.get("PYTORCH_NPU_ALLOC_CONF"),
            "MULTI_STREAM_MEMORY_REUSE": os.environ.get("MULTI_STREAM_MEMORY_REUSE"),
            "HCCL_BUFFSIZE": os.environ.get("HCCL_BUFFSIZE"),
            "ASCEND_VISIBLE_DEVICES": os.environ.get("ASCEND_VISIBLE_DEVICES"),
        },
    }

    if run_probe is None:
        run_probe = os.environ.get("NPU_OOM_DEBUG_PROBE") == "1"
    if run_probe:
        high_gib = max(16.0, math.ceil((requested_bytes or 0) / GIB + 4))
        payload["probe_before_empty_cache"] = _probe_max_alloc_bytes(torch, npu, 1.0, high_gib)
        _safe_call("empty_cache_before_probe", lambda: npu.empty_cache())
        payload["probe_after_empty_cache"] = _probe_max_alloc_bytes(torch, npu, 1.0, high_gib)

    summary_path.write_text(json.dumps(_to_jsonable(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    text_path.write_text(str(summary), encoding="utf-8", errors="replace")

    print(
        f"[NPU_OOM_DEBUG][rank={rank}] tag={tag} requested={_fmt_bytes(requested_bytes)} "
        f"driver_free={_fmt_bytes(free_bytes)} driver_total={_fmt_bytes(total_bytes)} "
        f"reserved={_fmt_bytes(payload['memory_reserved'])} allocated={_fmt_bytes(payload['memory_allocated'])} "
        f"largest_inactive={snapshot_summary.get('largest_inactive_block_gib', 0):.3f} GiB "
        f"inactive_total={snapshot_summary.get('inactive_total_bytes', 0) / GIB:.3f} GiB "
        f"summary={summary_path} memory_summary={text_path} snapshot={snapshot_path}",
        flush=True,
    )
