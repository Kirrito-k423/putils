from __future__ import annotations

import os
import socket
import subprocess
import time
import importlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from putils.device import get_device_name


SUPPORTED_SYNC_MODES = ("none", "boundary")
DEFAULT_SYNC_MODE = "none"
SUPPORTED_RANK_STRATEGIES = ("all", "rank0")
DEFAULT_RANK_STRATEGY = "all"
DEFAULT_THRESHOLD_SECONDS = 0.1
DEFAULT_OUTPUT_DIR = "."
UTC_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S.%fZ"


def validate_sync_mode(sync_mode: str) -> str:
    if sync_mode not in SUPPORTED_SYNC_MODES:
        expected_modes = ", ".join(SUPPORTED_SYNC_MODES)
        raise ValueError(f"Invalid sync_mode: {sync_mode}. Expected one of [{expected_modes}]")
    return sync_mode


def validate_rank_strategy(rank_strategy: str) -> str:
    if rank_strategy not in SUPPORTED_RANK_STRATEGIES:
        expected_strategies = ", ".join(SUPPORTED_RANK_STRATEGIES)
        raise ValueError(
            f"Invalid rank_strategy: {rank_strategy}. Expected one of [{expected_strategies}]"
        )
    return rank_strategy


def should_emit_for_rank(rank: int, rank_strategy: str = DEFAULT_RANK_STRATEGY) -> bool:
    rank_strategy = validate_rank_strategy(rank_strategy)
    if rank_strategy == "all":
        return True
    return rank == 0


def format_utc_timestamp(now_utc: datetime | None = None) -> str:
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    elif now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    else:
        now_utc = now_utc.astimezone(timezone.utc)

    return now_utc.strftime(UTC_TIMESTAMP_FORMAT)


def make_timestamped_log_filename(
    output_dir: str,
    prefix: str = "compare_perf",
    extension: str = ".json",
    now_utc: datetime | None = None,
    unique_token: str | None = None,
) -> str:
    if not extension.startswith("."):
        raise ValueError(f"extension must start with '.', got: {extension}")

    utc_tag = format_utc_timestamp(now_utc)
    token = unique_token or f"{os.getpid()}_{time.time_ns()}"
    filename = f"{prefix}_{utc_tag}_{token}{extension}"
    return os.path.join(output_dir, filename)


def _safe_int_from_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def _distributed_runtime() -> tuple[int, int, str]:
    rank = _safe_int_from_env("RANK", 0)
    world_size = _safe_int_from_env("WORLD_SIZE", 1)
    backend = os.getenv("TORCH_DISTRIBUTED_BACKEND", "none")

    try:
        dist = importlib.import_module("torch.distributed")
    except Exception:
        return rank, world_size, backend

    try:
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            backend = dist.get_backend()
    except Exception:
        pass

    return int(rank), int(world_size), str(backend)


def _resolve_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        commit = result.stdout.strip()
        return commit if commit else "unknown"
    except Exception:
        return "unknown"


def collect_runtime_metadata(now_utc: datetime | None = None) -> dict[str, Any]:
    rank, world_size, backend = _distributed_runtime()
    return {
        "backend": backend,
        "device": get_device_name(),
        "rank": rank,
        "world_size": world_size,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "commit": _resolve_git_commit(),
        "timestamp_utc": format_utc_timestamp(now_utc),
        "timestamp_unix_ns": time.time_ns(),
    }


@dataclass(frozen=True)
class ComparePerfTimingConfig:
    threshold_seconds: float = DEFAULT_THRESHOLD_SECONDS
    output_dir: str = DEFAULT_OUTPUT_DIR
    rank_strategy: str = DEFAULT_RANK_STRATEGY
    sync_mode: str = DEFAULT_SYNC_MODE
    synchronize: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        validate_sync_mode(self.sync_mode)
        validate_rank_strategy(self.rank_strategy)
        if self.threshold_seconds <= 0:
            raise ValueError("threshold_seconds must be > 0")
