import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "npu_affinity_benchmark.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("npu_affinity_benchmark", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {MODULE_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _valid_payload(module):
    return {
        "schema_version": module.SCHEMA_VERSION,
        "run_metadata": {"profile": "smoke"},
        "topology": {"online_cpus": [0, 1], "nodes": []},
        "candidates": [{"candidate_id": "c0", "cpu_list": [0]}],
        "results": [
            {
                "npu_id": 0,
                "candidate_id": "c0",
                "requested_affinity": [0],
                "effective_affinity": [0],
                "samples": [],
            }
        ],
        "leaderboards": {
            "throughput": [{"npu_id": 0, "candidate_id": "c0", "value": 1.0}],
            "latency_p95_ms": [{"npu_id": 0, "candidate_id": "c0", "value": 10.0}],
        },
        "failures": [],
        "remeasure_events": [],
    }


@pytest.fixture
def fixture_deterministic_runtime_sources():
    base_files = {
        "/sys/devices/system/cpu/online": "0-5\n",
        "/sys/devices/system/node/online": "0-1\n",
        "/sys/devices/system/node/node0/cpulist": "0-2\n",
        "/sys/devices/system/node/node1/cpulist": "3-5\n",
        "/proc/self/cgroup": "0::/test.slice/mock\n",
        "/sys/fs/cgroup/test.slice/mock/cpuset.cpus.effective": "1,3-4\n",
        "/sys/fs/cgroup/test.slice/mock/cpuset.mems.effective": "1\n",
    }
    expected_npu_ids = [0, 2]
    expected_clock_ticks = [1_000_000_000, 1_100_000_000, 1_200_000_000]

    def _factory():
        file_map = dict(base_files)
        clock_index = {"value": 0}

        def _read_text(path: str):
            return file_map.get(path)

        def _get_affinity(_pid: int):
            return {1, 3, 4}

        def _list_npu_ids():
            return list(expected_npu_ids)

        def _now_ns():
            index = min(clock_index["value"], len(expected_clock_ticks) - 1)
            value = expected_clock_ticks[index]
            clock_index["value"] += 1
            return value

        return {
            "read_text": _read_text,
            "get_affinity": _get_affinity,
            "list_npu_ids": _list_npu_ids,
            "now_ns": _now_ns,
        }

    return {
        "factory": _factory,
        "expected_npu_ids": expected_npu_ids,
        "expected_clock_ticks": expected_clock_ticks,
    }


def test_build_schema_has_required_top_level_sections_schema():
    module = _load_module()
    payload = module.build_schema(
        run_metadata={"profile": "smoke"},
        topology={"online_cpus": [0], "nodes": []},
        candidates=[],
        results=[],
        leaderboards={"throughput": [], "latency_p95_ms": []},
        failures=[],
        remeasure_events=[],
    )

    assert payload["schema_version"] == module.SCHEMA_VERSION
    assert set(payload.keys()) == {
        "schema_version",
        "run_metadata",
        "topology",
        "candidates",
        "results",
        "leaderboards",
        "failures",
        "remeasure_events",
    }


def test_schema_missing_top_level_field_schema():
    module = _load_module()
    payload = _valid_payload(module)
    payload.pop("topology")

    with pytest.raises(module.SchemaValidationError, match="Missing required field: topology"):
        module.validate_schema(payload)


def test_schema_missing_requested_affinity_schema():
    module = _load_module()
    payload = _valid_payload(module)
    payload["results"][0].pop("requested_affinity")

    with pytest.raises(
        module.SchemaValidationError,
        match="Missing required field: results\\[0\\]\\.requested_affinity",
    ):
        module.validate_schema(payload)


def test_schema_missing_effective_affinity_schema():
    module = _load_module()
    payload = _valid_payload(module)
    payload["results"][0].pop("effective_affinity")

    with pytest.raises(
        module.SchemaValidationError,
        match="Missing required field: results\\[0\\]\\.effective_affinity",
    ):
        module.validate_schema(payload)


def test_discover_topology_collects_online_numa_and_effective_domain_topology():
    module = _load_module()

    files = {
        "/sys/devices/system/cpu/online": "0-1,4,7\n",
        "/sys/devices/system/node/online": "0-1\n",
        "/sys/devices/system/node/node0/cpulist": "0-1\n",
        "/sys/devices/system/node/node1/cpulist": "4,7\n",
        "/proc/self/cgroup": "0::/kubepods.slice/pod123\n",
        "/sys/fs/cgroup/kubepods.slice/pod123/cpuset.cpus.effective": "1,4-7\n",
        "/sys/fs/cgroup/kubepods.slice/pod123/cpuset.mems.effective": "1\n",
    }

    topology = module.discover_topology(
        read_text=files.get,
        get_affinity=lambda _pid: {1, 4, 7},
    )

    assert topology["status"] == "ok"
    assert topology["online_cpus"] == [0, 1, 4, 7]
    assert topology["nodes"] == [
        {"node_id": 0, "cpus": [0, 1]},
        {"node_id": 1, "cpus": [4, 7]},
    ]
    assert topology["process_affinity"] == [1, 4, 7]
    assert topology["effective_cpus"] == [1, 4, 7]
    assert topology["effective_mems"] == [1]
    assert topology["effective_domain"] == {
        "cpus": [1, 4, 7],
        "mems": [1],
    }


def test_discover_topology_empty_effective_cpus_is_classified_error_topology():
    module = _load_module()

    files = {
        "/sys/devices/system/cpu/online": "0-3\n",
        "/sys/devices/system/node/online": "0\n",
        "/sys/devices/system/node/node0/cpulist": "0-3\n",
        "/proc/self/cgroup": "0::/\n",
        "/sys/fs/cgroup/cpuset.cpus.effective": "\n",
        "/sys/fs/cgroup/cpuset.mems.effective": "0\n",
    }

    topology = module.discover_topology(
        read_text=files.get,
        get_affinity=lambda _pid: {0, 1},
    )

    assert topology["status"] == "error"
    assert topology["error"]["code"] == "EMPTY_EFFECTIVE_CPUS"
    assert topology["effective_cpus"] == []
    assert topology["effective_domain"]["cpus"] == []


def test_dry_run_topology_writes_machine_readable_payload_topology(tmp_path):
    module = _load_module()

    output = module.run_dry_run_topology(
        output_dir=tmp_path,
        topology_discoverer=lambda: {
            "status": "ok",
            "online_cpus": [0, 1],
            "nodes": [{"node_id": 0, "cpus": [0, 1]}],
            "process_affinity": [0, 1],
            "effective_cpus": [0, 1],
            "effective_mems": [0],
            "effective_domain": {"cpus": [0, 1], "mems": [0]},
        },
        run_metadata_provider=lambda: {"profile": "smoke"},
    )

    raw_results_path = Path(output)
    assert raw_results_path.name == "raw_results.json"
    assert raw_results_path.exists()

    payload = json.loads(raw_results_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == module.SCHEMA_VERSION
    assert payload["topology"]["effective_domain"]["cpus"] == [0, 1]


def test_generate_candidates_cross_numa_and_deduplicated_candidates():
    module = _load_module()

    topology = {
        "nodes": [
            {"node_id": 0, "cpus": [0, 1]},
            {"node_id": 1, "cpus": [4, 7]},
        ],
        "effective_domain": {"cpus": [1, 4, 7], "mems": [0, 1]},
    }

    candidates = module.generate_candidate_cpu_lists(topology=topology, profile="full")

    assert candidates
    cpu_lists = [tuple(item["cpu_list"]) for item in candidates]
    assert len(cpu_lists) == len(set(cpu_lists))

    effective_set = set(topology["effective_domain"]["cpus"])
    assert all(set(item["cpu_list"]).issubset(effective_set) for item in candidates)
    assert any(1 in item["cpu_list"] and 4 in item["cpu_list"] for item in candidates)


def test_generate_candidates_profile_limit_is_deterministic_candidates():
    module = _load_module()

    topology = {
        "nodes": [
            {"node_id": 0, "cpus": [0, 1, 2]},
            {"node_id": 1, "cpus": [3, 4, 5]},
            {"node_id": 2, "cpus": [6, 7, 8]},
            {"node_id": 3, "cpus": [9, 10, 11]},
        ],
        "effective_domain": {"cpus": list(range(12)), "mems": [0, 1, 2, 3]},
    }

    full_candidates = module.generate_candidate_cpu_lists(topology=topology, profile="full")
    medium_candidates = module.generate_candidate_cpu_lists(topology=topology, profile="medium")
    smoke_candidates = module.generate_candidate_cpu_lists(topology=topology, profile="smoke")
    medium_candidates_again = module.generate_candidate_cpu_lists(topology=topology, profile="medium")

    assert medium_candidates == medium_candidates_again
    assert len(smoke_candidates) <= len(medium_candidates) <= len(full_candidates)

    medium_limit = module.PROFILE_CANDIDATE_LIMITS["medium"]
    smoke_limit = module.PROFILE_CANDIDATE_LIMITS["smoke"]
    assert len(medium_candidates) <= medium_limit
    assert len(smoke_candidates) <= smoke_limit

    expected_medium = min(len(full_candidates), medium_limit)
    expected_smoke = min(len(full_candidates), smoke_limit)
    assert len(medium_candidates) == expected_medium
    assert len(smoke_candidates) == expected_smoke


def test_generate_candidates_single_numa_still_returns_valid_candidates_candidates():
    module = _load_module()

    topology = {
        "nodes": [{"node_id": 0, "cpus": [2, 3, 4]}],
        "effective_domain": {"cpus": [3, 4], "mems": [0]},
    }

    candidates = module.generate_candidate_cpu_lists(topology=topology, profile="full")

    assert candidates
    assert all(set(item["cpu_list"]).issubset({3, 4}) for item in candidates)
    assert not any(2 in item["cpu_list"] for item in candidates)


def test_dry_run_topology_populates_candidates_with_profile_cap_candidates(tmp_path):
    module = _load_module()

    topology = {
        "status": "ok",
        "online_cpus": list(range(12)),
        "nodes": [
            {"node_id": 0, "cpus": [0, 1, 2]},
            {"node_id": 1, "cpus": [3, 4, 5]},
            {"node_id": 2, "cpus": [6, 7, 8]},
            {"node_id": 3, "cpus": [9, 10, 11]},
        ],
        "process_affinity": list(range(12)),
        "effective_cpus": list(range(12)),
        "effective_mems": [0, 1, 2, 3],
        "effective_domain": {"cpus": list(range(12)), "mems": [0, 1, 2, 3]},
    }

    out1 = module.run_dry_run_topology(
        output_dir=tmp_path / "run1",
        profile="medium",
        topology_discoverer=lambda: topology,
        run_metadata_provider=lambda: {"profile": "medium"},
    )
    out2 = module.run_dry_run_topology(
        output_dir=tmp_path / "run2",
        profile="medium",
        topology_discoverer=lambda: topology,
        run_metadata_provider=lambda: {"profile": "medium"},
    )

    payload1 = json.loads(Path(out1).read_text(encoding="utf-8"))
    payload2 = json.loads(Path(out2).read_text(encoding="utf-8"))

    assert payload1["candidates"]
    assert payload1["candidates"] == payload2["candidates"]
    assert len(payload1["candidates"]) <= module.PROFILE_CANDIDATE_LIMITS["medium"]


def test_runtime_fixture_sources_are_deterministic_fixture(fixture_deterministic_runtime_sources):
    module = _load_module()
    fixture_meta = fixture_deterministic_runtime_sources

    source_a = fixture_meta["factory"]()
    source_b = fixture_meta["factory"]()

    topology_a = module.discover_topology(
        read_text=source_a["read_text"],
        get_affinity=source_a["get_affinity"],
    )
    topology_b = module.discover_topology(
        read_text=source_b["read_text"],
        get_affinity=source_b["get_affinity"],
    )

    assert topology_a == topology_b
    assert source_a["list_npu_ids"]() == fixture_meta["expected_npu_ids"]
    assert source_b["list_npu_ids"]() == fixture_meta["expected_npu_ids"]

    ticks_a = [source_a["now_ns"](), source_a["now_ns"](), source_a["now_ns"]()]
    ticks_b = [source_b["now_ns"](), source_b["now_ns"](), source_b["now_ns"]()]
    assert ticks_a == fixture_meta["expected_clock_ticks"]
    assert ticks_b == fixture_meta["expected_clock_ticks"]


def test_mixed_workload_is_reproducible_with_fixed_seed_workload():
    module = _load_module()

    class _FakeTensor:
        def __init__(self, values):
            self.values = [float(v) for v in values]

        def reshape(self, *_shape):
            return _FakeTensor(self.values)

        def __getitem__(self, item):
            if isinstance(item, slice):
                return _FakeTensor(self.values[item])
            return self.values[item]

        def clone(self):
            return _FakeTensor(self.values)

        def sum(self):
            return _FakeScalar(sum(self.values))

        def numel(self):
            return len(self.values)

    class _FakeScalar:
        def __init__(self, value):
            self._value = float(value)

        def item(self):
            return self._value

    class _FakeGenerator:
        def __init__(self):
            self.seed = 0

        def manual_seed(self, seed):
            self.seed = int(seed)
            return self

    class _FakeTorch:
        float32 = "float32"

        @staticmethod
        def Generator(device="cpu"):
            assert device == "cpu"
            return _FakeGenerator()

        @staticmethod
        def rand(shape, generator, dtype=None):
            count = int(shape[0]) * int(shape[1])
            base = generator.seed % 997
            values = [((base + idx) % 211) / 211.0 for idx in range(count)]
            generator.seed += count
            return _FakeTensor(values)

        @staticmethod
        def matmul(lhs, rhs):
            count = min(len(lhs.values), len(rhs.values))
            return _FakeTensor([lhs.values[idx] * rhs.values[idx] for idx in range(count)])

    ticks = [100, 160, 220, 280, 340, 400, 460, 520, 580]
    cursor_a = {"value": 0}
    cursor_b = {"value": 0}

    def _now_ns_factory(cursor):
        def _now_ns():
            idx = min(cursor["value"], len(ticks) - 1)
            cursor["value"] += 1
            return ticks[idx]

        return _now_ns

    out1 = module.run_mixed_workload(
        profile="smoke",
        seed=7,
        torch_module=_FakeTorch,
        now_ns=_now_ns_factory(cursor_a),
    )
    out2 = module.run_mixed_workload(
        profile="smoke",
        seed=7,
        torch_module=_FakeTorch,
        now_ns=_now_ns_factory(cursor_b),
    )

    assert out1["status"] == "ok"
    assert out2["status"] == "ok"
    assert out1["config"] == out2["config"]
    assert out1["samples"] == out2["samples"]
    assert len(out1["samples"]) == module.WORKLOAD_PROFILE_CONFIG["smoke"]["iterations"]


def test_mixed_workload_handles_torch_unavailable_gracefully_workload():
    module = _load_module()
    original_torch = getattr(module, "torch", None)
    setattr(module, "torch", None)

    try:
        result = module.run_mixed_workload(profile="smoke", seed=11)
    finally:
        setattr(module, "torch", original_torch)

    assert result["status"] == "skipped"
    assert result["reason"] == "TORCH_UNAVAILABLE"
    assert result["samples"] == []
    assert result["config"] == module.WORKLOAD_PROFILE_CONFIG["smoke"]


def test_baseline_source_and_audit_fields_are_machine_readable_baseline_protocol():
    module = _load_module()

    audit = module.evaluate_baseline_remeasure(
        warmup_throughput=1000.0,
        warmup_latency_p95_ms=10.0,
        first_pass_throughput=950.0,
        first_pass_latency_p95_ms=10.5,
        remeasure_samples=[{"throughput": 970.0, "latency_p95_ms": 10.2}],
        deviation_ratio=0.10,
        max_remeasure_attempts=2,
    )

    assert audit["baseline"]["source"] == "same_run_warmup_first_pass"
    assert audit["baseline"]["throughput"] == {"min": 950.0, "max": 1000.0}
    assert audit["baseline"]["latency_p95_ms"] == {"min": 10.0, "max": 10.5}
    assert audit["remeasure_count"] == 1
    assert audit["decision_status"] == module.REMEASURE_DECISION_ACCEPTED
    assert isinstance(audit["decisions"], list)


def test_baseline_remeasure_attempts_are_bounded_and_can_become_unstable_baseline_protocol():
    module = _load_module()

    audit = module.evaluate_baseline_remeasure(
        warmup_throughput=1000.0,
        warmup_latency_p95_ms=10.0,
        first_pass_throughput=1000.0,
        first_pass_latency_p95_ms=10.0,
        remeasure_samples=[
            {"throughput": 850.0, "latency_p95_ms": 12.0},
            {"throughput": 860.0, "latency_p95_ms": 12.2},
            {"throughput": 990.0, "latency_p95_ms": 10.2},
        ],
        deviation_ratio=0.10,
        max_remeasure_attempts=2,
    )

    assert audit["max_remeasure_attempts"] == 2
    assert audit["remeasure_count"] == 2
    assert len(audit["decisions"]) == 2
    assert audit["decision_status"] == module.REMEASURE_DECISION_UNSTABLE
