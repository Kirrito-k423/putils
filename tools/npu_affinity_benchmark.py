from __future__ import annotations

import argparse
import errno
import json
import os
import random
import socket
import time
from pathlib import Path
from typing import Any, Callable, Mapping

try:
    import torch  # type: ignore[reportMissingImports]
except ImportError:
    torch = None


SCHEMA_VERSION = 1
SUPPORTED_SCHEMA_VERSIONS = (SCHEMA_VERSION,)
REQUIRED_TOP_LEVEL_SECTIONS = (
    "run_metadata",
    "topology",
    "candidates",
    "results",
    "leaderboards",
    "failures",
    "remeasure_events",
)
PROFILE_CANDIDATE_LIMITS = {
    "smoke": 4,
    "medium": 12,
    "full": 64,
}
WORKLOAD_PROFILE_CONFIG = {
    "smoke": {"matrix_size": 64, "iterations": 4, "transfer_bytes": 4096},
    "medium": {"matrix_size": 128, "iterations": 8, "transfer_bytes": 16384},
    "full": {"matrix_size": 256, "iterations": 16, "transfer_bytes": 65536},
}
DEFAULT_WORKLOAD_SEED = 20260401
DEFAULT_BASELINE_DEVIATION_RATIO = 0.10
DEFAULT_MAX_REMEASURE_ATTEMPTS = 2

REMEASURE_DECISION_ACCEPTED = "accepted"
REMEASURE_DECISION_UNSTABLE = "unstable"

AFFINITY_BIND_STATUS_OK = "ok"
AFFINITY_BIND_STATUS_MISMATCH = "mismatch"
AFFINITY_BIND_STATUS_ERROR = "error"

AFFINITY_FAILURE_CODE_EPERM = "EPERM"
AFFINITY_FAILURE_CODE_EINVAL = "EINVAL"
AFFINITY_FAILURE_CODE_TIMEOUT = "TIMEOUT"
AFFINITY_FAILURE_CODE_OTHER = "OTHER"


class SchemaValidationError(ValueError):
    pass


def classify_affinity_failure(error: BaseException) -> dict[str, Any]:
    failure_code = AFFINITY_FAILURE_CODE_OTHER
    error_errno: int | None = None

    if isinstance(error, TimeoutError):
        failure_code = AFFINITY_FAILURE_CODE_TIMEOUT
    elif isinstance(error, OSError):
        error_errno = error.errno
        if error.errno == errno.EPERM:
            failure_code = AFFINITY_FAILURE_CODE_EPERM
        elif error.errno == errno.EINVAL:
            failure_code = AFFINITY_FAILURE_CODE_EINVAL

    return {
        "failure_code": failure_code,
        "errno": error_errno,
        "error_type": type(error).__name__,
        "message": str(error),
    }


def request_set_readback_affinity(
    *,
    pid: int,
    requested_affinity: list[int],
    set_affinity: Callable[..., None],
    get_affinity: Callable[[int], set[int] | list[int] | tuple[int, ...]],
    get_mems: Callable[[], list[int]] | None = None,
    timeout_ns: int | None = None,
) -> dict[str, Any]:
    requested = sorted({int(cpu) for cpu in requested_affinity})
    effective: list[int] = []
    effective_mems = get_mems() if get_mems is not None else None

    try:
        if timeout_ns is None:
            set_affinity(pid, requested)
        else:
            set_affinity(pid, requested, timeout_ns=timeout_ns)

        effective_raw = get_affinity(pid)
        effective = sorted({int(cpu) for cpu in effective_raw})
    except Exception as error:
        return {
            "bind_status": AFFINITY_BIND_STATUS_ERROR,
            "requested_affinity": requested,
            "effective_affinity": effective,
            "effective_mems": effective_mems,
            "mismatch": False,
            "ranking_valid": False,
            "failure": classify_affinity_failure(error),
        }

    is_mismatch = effective != requested
    return {
        "bind_status": AFFINITY_BIND_STATUS_MISMATCH if is_mismatch else AFFINITY_BIND_STATUS_OK,
        "requested_affinity": requested,
        "effective_affinity": effective,
        "effective_mems": effective_mems,
        "mismatch": is_mismatch,
        "ranking_valid": not is_mismatch,
        "failure": None,
    }


def mark_result_invalid_for_ranking_on_mismatch(result: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(result)
    samples = normalized.get("samples")
    bind_status = str(normalized.get("bind_status", "")).lower()
    mismatch = bool(normalized.get("mismatch", False))
    ranking_valid = bool(normalized.get("ranking_valid", True))

    should_invalidate = bind_status == AFFINITY_BIND_STATUS_MISMATCH or mismatch or not ranking_valid
    if not isinstance(samples, list) or not should_invalidate:
        return normalized

    invalid_samples: list[dict[str, Any]] = []
    for sample in samples:
        if isinstance(sample, Mapping):
            sample_with_flag = dict(sample)
            sample_with_flag["ranking_valid"] = False
            invalid_samples.append(sample_with_flag)
        else:
            invalid_samples.append({"ranking_valid": False})
    normalized["samples"] = invalid_samples
    normalized["ranking_valid"] = False
    return normalized


def _read_text_file(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return None


def _parse_cpu_list(raw_value: str | None) -> list[int]:
    if raw_value is None:
        return []

    values = set()
    text = raw_value.strip()
    if not text:
        return []

    for token in text.split(","):
        item = token.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", maxsplit=1)
            try:
                start = int(start_text)
                end = int(end_text)
            except ValueError:
                continue
            low = min(start, end)
            high = max(start, end)
            values.update(range(low, high + 1))
            continue
        try:
            values.add(int(item))
        except ValueError:
            continue

    return sorted(values)


def _read_first_available(paths: list[str], read_text: Callable[[str], str | None]) -> tuple[str | None, str | None]:
    for path in paths:
        content = read_text(path)
        if content is not None:
            return path, content
    return None, None


def _normalize_cgroup_path(cgroup_path: str) -> str:
    if not cgroup_path or cgroup_path == "/":
        return ""
    normalized = cgroup_path if cgroup_path.startswith("/") else f"/{cgroup_path}"
    return normalized.rstrip("/")


def _discover_process_cgroup_path(read_text: Callable[[str], str | None]) -> str:
    content = read_text("/proc/self/cgroup")
    if not content:
        return ""

    for line in content.splitlines():
        parts = line.split(":", maxsplit=2)
        if len(parts) != 3:
            continue
        _, controllers, path = parts
        controllers = controllers.strip()
        if controllers == "":
            return _normalize_cgroup_path(path.strip())
        if "cpuset" in controllers.split(","):
            return _normalize_cgroup_path(path.strip())
    return ""


def _discover_numa_nodes(
    read_text: Callable[[str], str | None],
) -> tuple[list[dict[str, Any]], list[int]]:
    node_ids = _parse_cpu_list(read_text("/sys/devices/system/node/online"))
    if not node_ids:
        return [], []

    nodes: list[dict[str, Any]] = []
    merged_cpus = set()
    for node_id in node_ids:
        cpus = _parse_cpu_list(read_text(f"/sys/devices/system/node/node{node_id}/cpulist"))
        merged_cpus.update(cpus)
        nodes.append({"node_id": node_id, "cpus": cpus})

    return nodes, sorted(merged_cpus)


def _collect_runtime_metadata(profile: str) -> dict[str, Any]:
    return {
        "profile": profile,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "timestamp_unix_ns": time.time_ns(),
    }


def _get_process_affinity(pid: int = 0) -> set[int]:
    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    if sched_getaffinity is not None:
        try:
            return set(sched_getaffinity(pid))
        except OSError:
            pass

    cpu_count = os.cpu_count() or 1
    return set(range(cpu_count))


def discover_topology(
    *,
    read_text: Callable[[str], str | None] = _read_text_file,
    get_affinity: Callable[[int], set[int]] = _get_process_affinity,
    pid: int = 0,
) -> dict[str, Any]:
    online_cpus = _parse_cpu_list(read_text("/sys/devices/system/cpu/online"))
    nodes, node_cpus = _discover_numa_nodes(read_text)
    if not online_cpus and node_cpus:
        online_cpus = node_cpus

    process_affinity = sorted(get_affinity(pid))
    process_cgroup_path = _discover_process_cgroup_path(read_text)

    cgroup_root = "/sys/fs/cgroup"
    cgroup_candidates = [
        f"{cgroup_root}{process_cgroup_path}/cpuset.cpus.effective",
        f"{cgroup_root}{process_cgroup_path}/cpuset.cpus",
        f"{cgroup_root}/cpuset.cpus.effective",
        f"{cgroup_root}/cpuset.cpus",
    ]
    mem_candidates = [
        f"{cgroup_root}{process_cgroup_path}/cpuset.mems.effective",
        f"{cgroup_root}{process_cgroup_path}/cpuset.mems",
        f"{cgroup_root}/cpuset.mems.effective",
        f"{cgroup_root}/cpuset.mems",
    ]

    cpuset_cpu_path, cpuset_cpu_text = _read_first_available(cgroup_candidates, read_text)
    cpuset_mem_path, cpuset_mem_text = _read_first_available(mem_candidates, read_text)
    cpuset_cpus = _parse_cpu_list(cpuset_cpu_text)
    cpuset_mems = _parse_cpu_list(cpuset_mem_text)

    effective_cpus_set = set(process_affinity)
    if online_cpus:
        effective_cpus_set &= set(online_cpus)
    if cpuset_cpu_path is not None:
        effective_cpus_set &= set(cpuset_cpus)
    effective_cpus = sorted(effective_cpus_set)

    if cpuset_mems:
        effective_mems = cpuset_mems
    else:
        node_ids = [node["node_id"] for node in nodes]
        effective_mems = node_ids

    error = None
    status = "ok"
    if not effective_cpus:
        status = "error"
        error = {
            "code": "EMPTY_EFFECTIVE_CPUS",
            "message": "effective CPU set is empty after applying runtime constraints",
        }

    return {
        "status": status,
        "error": error,
        "online_cpus": online_cpus,
        "nodes": nodes,
        "process_affinity": process_affinity,
        "process_cgroup_path": process_cgroup_path,
        "cpuset_effective_cpu_path": cpuset_cpu_path,
        "cpuset_effective_mem_path": cpuset_mem_path,
        "cpuset_effective_cpus": cpuset_cpus,
        "cpuset_effective_mems": cpuset_mems,
        "effective_cpus": effective_cpus,
        "effective_mems": effective_mems,
        "effective_domain": {
            "cpus": effective_cpus,
            "mems": effective_mems,
        },
    }


def _resolve_profile_candidate_limit(profile: str) -> int:
    if profile not in PROFILE_CANDIDATE_LIMITS:
        supported = ", ".join(sorted(PROFILE_CANDIDATE_LIMITS))
        raise ValueError(f"Unsupported profile '{profile}', expected one of: {supported}")
    return PROFILE_CANDIDATE_LIMITS[profile]


def _resolve_workload_profile_config(profile: str) -> dict[str, int]:
    if profile not in WORKLOAD_PROFILE_CONFIG:
        supported = ", ".join(sorted(WORKLOAD_PROFILE_CONFIG))
        raise ValueError(f"Unsupported profile '{profile}', expected one of: {supported}")
    config = WORKLOAD_PROFILE_CONFIG[profile]
    return {
        "matrix_size": int(config["matrix_size"]),
        "iterations": int(config["iterations"]),
        "transfer_bytes": int(config["transfer_bytes"]),
    }


def run_mixed_workload(
    *,
    profile: str,
    seed: int = DEFAULT_WORKLOAD_SEED,
    torch_module: Any | None = None,
    now_ns: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, Any]:
    config = _resolve_workload_profile_config(profile)
    resolved_torch = torch if torch_module is None else torch_module

    if resolved_torch is None:
        return {
            "status": "skipped",
            "reason": "TORCH_UNAVAILABLE",
            "profile": profile,
            "seed": seed,
            "config": config,
            "samples": [],
        }

    generator = resolved_torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    fallback_rng = random.Random(int(seed))

    matrix_size = config["matrix_size"]
    transfer_bytes = config["transfer_bytes"]
    transfer_elements = max(1, transfer_bytes // 4)
    samples: list[dict[str, Any]] = []

    for iteration in range(config["iterations"]):
        lhs = resolved_torch.rand((matrix_size, matrix_size), generator=generator, dtype=resolved_torch.float32)
        rhs = resolved_torch.rand((matrix_size, matrix_size), generator=generator, dtype=resolved_torch.float32)

        start_ns = now_ns()
        product = resolved_torch.matmul(lhs, rhs)
        transfer_sim = product.reshape(-1)[:transfer_elements].clone()
        latency_ns = max(0, now_ns() - start_ns)

        checksum = float(transfer_sim.sum().item())
        checksum = round(checksum + fallback_rng.random() * 1e-9, 9)
        samples.append(
            {
                "iteration": iteration,
                "latency_ns": latency_ns,
                "checksum": checksum,
                "transfer_elements": int(transfer_sim.numel()),
            }
        )

    return {
        "status": "ok",
        "reason": None,
        "profile": profile,
        "seed": seed,
        "config": config,
        "samples": samples,
    }


def _resolve_baseline_window(warmup_value: float, first_pass_value: float) -> dict[str, float]:
    low = min(float(warmup_value), float(first_pass_value))
    high = max(float(warmup_value), float(first_pass_value))
    return {
        "min": low,
        "max": high,
    }


def _resolve_threshold_window(
    *,
    baseline_window: Mapping[str, float],
    deviation_ratio: float,
) -> tuple[float, float]:
    low = float(baseline_window["min"])
    high = float(baseline_window["max"])
    return (
        low * (1.0 - deviation_ratio),
        high * (1.0 + deviation_ratio),
    )


def _evaluate_metric_drift(
    *,
    value: float,
    baseline_window: Mapping[str, float],
    deviation_ratio: float,
) -> dict[str, Any]:
    lower_bound, upper_bound = _resolve_threshold_window(
        baseline_window=baseline_window,
        deviation_ratio=deviation_ratio,
    )
    metric_value = float(value)
    within_threshold = lower_bound <= metric_value <= upper_bound

    deviation_ratio_abs = 0.0
    if metric_value < lower_bound:
        deviation_ratio_abs = (lower_bound - metric_value) / max(abs(lower_bound), 1e-12)
    elif metric_value > upper_bound:
        deviation_ratio_abs = (metric_value - upper_bound) / max(abs(upper_bound), 1e-12)

    return {
        "value": metric_value,
        "within_threshold": within_threshold,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "deviation_ratio": deviation_ratio_abs,
    }


def evaluate_baseline_remeasure(
    *,
    warmup_throughput: float,
    warmup_latency_p95_ms: float,
    first_pass_throughput: float,
    first_pass_latency_p95_ms: float,
    remeasure_samples: list[Mapping[str, Any]] | None = None,
    deviation_ratio: float = DEFAULT_BASELINE_DEVIATION_RATIO,
    max_remeasure_attempts: int = DEFAULT_MAX_REMEASURE_ATTEMPTS,
) -> dict[str, Any]:
    if deviation_ratio < 0:
        raise ValueError("deviation_ratio must be non-negative")
    if max_remeasure_attempts < 0:
        raise ValueError("max_remeasure_attempts must be non-negative")

    throughput_window = _resolve_baseline_window(warmup_throughput, first_pass_throughput)
    latency_window = _resolve_baseline_window(warmup_latency_p95_ms, first_pass_latency_p95_ms)
    decision_trace: list[dict[str, Any]] = []
    attempts_used = 0

    samples = list(remeasure_samples or [])
    capped_samples = samples[:max_remeasure_attempts]

    for attempt_index, sample in enumerate(capped_samples, start=1):
        attempts_used = attempt_index
        throughput_eval = _evaluate_metric_drift(
            value=float(sample["throughput"]),
            baseline_window=throughput_window,
            deviation_ratio=deviation_ratio,
        )
        latency_eval = _evaluate_metric_drift(
            value=float(sample["latency_p95_ms"]),
            baseline_window=latency_window,
            deviation_ratio=deviation_ratio,
        )

        trigger_metrics: list[str] = []
        if not throughput_eval["within_threshold"]:
            trigger_metrics.append("throughput")
        if not latency_eval["within_threshold"]:
            trigger_metrics.append("latency_p95_ms")

        accepted = not trigger_metrics
        decision_status = REMEASURE_DECISION_ACCEPTED if accepted else REMEASURE_DECISION_UNSTABLE
        decision_trace.append(
            {
                "attempt": attempt_index,
                "decision": decision_status,
                "trigger_metrics": trigger_metrics,
                "throughput": throughput_eval,
                "latency_p95_ms": latency_eval,
            }
        )

        if accepted:
            break

    final_status = REMEASURE_DECISION_ACCEPTED
    if decision_trace and decision_trace[-1]["decision"] == REMEASURE_DECISION_UNSTABLE:
        final_status = REMEASURE_DECISION_UNSTABLE

    return {
        "baseline": {
            "source": "same_run_warmup_first_pass",
            "deviation_ratio": float(deviation_ratio),
            "throughput": throughput_window,
            "latency_p95_ms": latency_window,
        },
        "max_remeasure_attempts": int(max_remeasure_attempts),
        "remeasure_count": attempts_used,
        "decision_status": final_status,
        "decisions": decision_trace,
    }


def _extract_effective_cpus(topology: Mapping[str, Any]) -> list[int]:
    effective_domain = topology.get("effective_domain")
    if isinstance(effective_domain, Mapping):
        effective_cpus = effective_domain.get("cpus")
        if isinstance(effective_cpus, list):
            return sorted({int(cpu) for cpu in effective_cpus})

    fallback = topology.get("effective_cpus")
    if isinstance(fallback, list):
        return sorted({int(cpu) for cpu in fallback})

    return []


def generate_candidate_cpu_lists(
    *,
    topology: Mapping[str, Any],
    profile: str,
) -> list[dict[str, Any]]:
    limit = _resolve_profile_candidate_limit(profile)
    effective_cpus = _extract_effective_cpus(topology)
    if not effective_cpus:
        return []

    effective_set = set(effective_cpus)
    nodes = topology.get("nodes")
    node_groups: list[tuple[int, tuple[int, ...]]] = []
    if isinstance(nodes, list):
        for node in nodes:
            if not isinstance(node, Mapping):
                continue
            node_id = node.get("node_id")
            cpus = node.get("cpus")
            if not isinstance(node_id, int) or not isinstance(cpus, list):
                continue
            filtered = tuple(sorted(set(int(cpu) for cpu in cpus if int(cpu) in effective_set)))
            if filtered:
                node_groups.append((node_id, filtered))

    node_groups.sort(key=lambda item: item[0])

    ordered_candidates: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()

    def add_candidate(cpus: tuple[int, ...]) -> None:
        if not cpus:
            return
        normalized = tuple(sorted(set(cpus)))
        if normalized in seen:
            return
        seen.add(normalized)
        ordered_candidates.append(normalized)

    for cpu in effective_cpus:
        add_candidate((cpu,))

    for _, cpus in node_groups:
        add_candidate(cpus)

    for left_index in range(len(node_groups)):
        _, left_cpus = node_groups[left_index]
        for right_index in range(left_index + 1, len(node_groups)):
            _, right_cpus = node_groups[right_index]
            add_candidate(left_cpus + right_cpus)

    add_candidate(tuple(effective_cpus))

    limited = ordered_candidates[:limit]
    return [
        {
            "candidate_id": f"c{index:03d}",
            "cpu_list": list(cpu_list),
        }
        for index, cpu_list in enumerate(limited)
    ]


def run_dry_run_topology(
    *,
    output_dir: str | Path,
    profile: str = "smoke",
    topology_discoverer: Callable[[], dict[str, Any]] = discover_topology,
    run_metadata_provider: Callable[[], dict[str, Any]] | None = None,
) -> str:
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    metadata_provider = run_metadata_provider
    if metadata_provider is None:
        metadata_provider = lambda: _collect_runtime_metadata(profile)

    topology = topology_discoverer()
    candidates = generate_candidate_cpu_lists(topology=topology, profile=profile)

    payload = build_schema(
        run_metadata=metadata_provider(),
        topology=topology,
        candidates=candidates,
        results=[],
        leaderboards={"throughput": [], "latency_p95_ms": []},
        failures=[],
        remeasure_events=[],
    )

    output_path = resolved_output_dir / "raw_results.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NPU affinity micro benchmark")
    parser.add_argument("--profile", default="smoke", help="Benchmark profile (smoke/medium/full)")
    parser.add_argument("--dry-run-topology", action="store_true", help="Only discover topology")
    parser.add_argument("--output-dir", default=".", help="Directory to write outputs")
    args = parser.parse_args(argv)

    if not args.dry_run_topology:
        raise SystemExit("Only --dry-run-topology is supported in current task scope")

    output_path = run_dry_run_topology(output_dir=args.output_dir, profile=args.profile)
    print(output_path)
    return 0


def build_schema(
    *,
    run_metadata: dict[str, Any],
    topology: dict[str, Any],
    candidates: list[dict[str, Any]],
    results: list[dict[str, Any]],
    leaderboards: dict[str, Any],
    failures: list[dict[str, Any]],
    remeasure_events: list[dict[str, Any]],
    schema_version: int = SCHEMA_VERSION,
) -> dict[str, Any]:
    payload = {
        "schema_version": schema_version,
        "run_metadata": run_metadata,
        "topology": topology,
        "candidates": candidates,
        "results": results,
        "leaderboards": leaderboards,
        "failures": failures,
        "remeasure_events": remeasure_events,
    }
    validate_schema(payload)
    return payload


def validate_schema(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise SchemaValidationError("Schema payload must be a mapping")

    if "schema_version" not in payload:
        raise SchemaValidationError("Missing required field: schema_version")

    schema_version = payload["schema_version"]
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        expected_versions = ", ".join(str(v) for v in SUPPORTED_SCHEMA_VERSIONS)
        raise SchemaValidationError(
            f"Incompatible schema_version: expected one of [{expected_versions}], got {schema_version}"
        )

    for section in REQUIRED_TOP_LEVEL_SECTIONS:
        if section not in payload:
            raise SchemaValidationError(f"Missing required field: {section}")

    _validate_results(payload["results"])
    return payload


def parse_schema(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return validate_schema(payload)


def _validate_results(results: Any) -> None:
    if not isinstance(results, list):
        raise SchemaValidationError("Field results must be a list")

    for index, result in enumerate(results):
        if not isinstance(result, Mapping):
            raise SchemaValidationError(f"Field results[{index}] must be a mapping")
        if "requested_affinity" not in result:
            raise SchemaValidationError(
                f"Missing required field: results[{index}].requested_affinity"
            )
        if "effective_affinity" not in result:
            raise SchemaValidationError(
                f"Missing required field: results[{index}].effective_affinity"
            )


if __name__ == "__main__":
    raise SystemExit(main())
