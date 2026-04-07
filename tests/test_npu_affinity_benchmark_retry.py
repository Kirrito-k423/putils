import errno
import importlib.util
from collections import deque
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


def _classify_injected_failure(error: Exception) -> str:
    if isinstance(error, TimeoutError):
        return "TIMEOUT"
    if isinstance(error, OSError):
        if error.errno == errno.EPERM:
            return "EPERM"
        if error.errno == errno.EINVAL:
            return "EINVAL"
    return "UNKNOWN"


@pytest.fixture
def fixture_controlled_clock():
    class ControlledClock:
        def __init__(self, start_ns: int = 0, step_ns: int = 1_000_000):
            self._current = start_ns
            self._step = step_ns

        def now_ns(self) -> int:
            value = self._current
            self._current += self._step
            return value

        def sleep_ns(self, duration_ns: int) -> None:
            self._current += max(0, duration_ns)

    return ControlledClock


@pytest.fixture
def fixture_mock_procfs_cgroup():
    mapping = {
        "/proc/self/cgroup": "0::/mock.slice/bench\n",
        "/sys/fs/cgroup/mock.slice/bench/cpuset.cpus.effective": "2-3,8\n",
        "/sys/fs/cgroup/mock.slice/bench/cpuset.mems.effective": "0\n",
    }

    def _read_text(path: str):
        return mapping.get(path)

    return {
        "read_text": _read_text,
        "paths": sorted(mapping.keys()),
    }


@pytest.fixture
def fixture_mock_npu_inventory():
    stable_ids = [0, 4, 7]

    def _list_npu_ids():
        return list(stable_ids)

    return _list_npu_ids


@pytest.fixture
def fixture_affinity_fault_injector():
    class Injector:
        def __init__(self, failure_plan=None, clock=None):
            self._plan = deque(failure_plan or [])
            self._clock = clock
            self._effective_affinity = [0]
            self.calls = []

        def set_affinity(self, pid: int, cpu_list: list[int], timeout_ns: int | None = None) -> None:
            self.calls.append({"pid": pid, "cpu_list": list(cpu_list), "timeout_ns": timeout_ns})
            action = self._plan.popleft() if self._plan else "ok"

            if action == "EPERM":
                raise OSError(errno.EPERM, "mock EPERM for set_affinity")
            if action == "EINVAL":
                raise OSError(errno.EINVAL, "mock EINVAL for set_affinity")
            if action == "TIMEOUT":
                if self._clock is not None:
                    if timeout_ns is None:
                        self._clock.sleep_ns(1)
                    else:
                        self._clock.sleep_ns(timeout_ns + 1)
                raise TimeoutError("mock timeout for set_affinity")

            self._effective_affinity = sorted(set(int(cpu) for cpu in cpu_list))

        def get_affinity(self, _pid: int) -> list[int]:
            return list(self._effective_affinity)

    return Injector


def test_fixture_mock_sources_are_deterministic_fixture(
    fixture_controlled_clock,
    fixture_mock_procfs_cgroup,
    fixture_mock_npu_inventory,
):
    clock_a = fixture_controlled_clock(start_ns=10, step_ns=5)
    clock_b = fixture_controlled_clock(start_ns=10, step_ns=5)

    ticks_a = [clock_a.now_ns(), clock_a.now_ns(), clock_a.now_ns()]
    ticks_b = [clock_b.now_ns(), clock_b.now_ns(), clock_b.now_ns()]
    assert ticks_a == [10, 15, 20]
    assert ticks_b == [10, 15, 20]

    read_text = fixture_mock_procfs_cgroup["read_text"]
    assert read_text("/proc/self/cgroup") == "0::/mock.slice/bench\n"
    assert fixture_mock_npu_inventory() == [0, 4, 7]
    assert fixture_mock_npu_inventory() == [0, 4, 7]


def test_fixture_injects_eperm_failure_fixture(fixture_affinity_fault_injector):
    injector = fixture_affinity_fault_injector(failure_plan=["EPERM"])

    with pytest.raises(OSError) as exc_info:
        injector.set_affinity(pid=1234, cpu_list=[2, 3])

    assert _classify_injected_failure(exc_info.value) == "EPERM"
    assert injector.calls == [{"pid": 1234, "cpu_list": [2, 3], "timeout_ns": None}]


def test_fixture_injects_einval_failure_fixture(fixture_affinity_fault_injector):
    injector = fixture_affinity_fault_injector(failure_plan=["EINVAL"])

    with pytest.raises(OSError) as exc_info:
        injector.set_affinity(pid=7, cpu_list=[9999])

    assert _classify_injected_failure(exc_info.value) == "EINVAL"
    assert injector.calls == [{"pid": 7, "cpu_list": [9999], "timeout_ns": None}]


def test_fixture_injects_timeout_style_failure_fixture(
    fixture_affinity_fault_injector,
    fixture_controlled_clock,
):
    clock = fixture_controlled_clock(start_ns=100, step_ns=10)
    injector = fixture_affinity_fault_injector(failure_plan=["TIMEOUT"], clock=clock)

    with pytest.raises(TimeoutError) as exc_info:
        injector.set_affinity(pid=9, cpu_list=[1], timeout_ns=50)

    assert _classify_injected_failure(exc_info.value) == "TIMEOUT"
    assert clock.now_ns() == 151
    assert injector.get_affinity(9) == [0]


def test_fixture_affinity_injector_success_path_updates_effective_affinity_fixture(
    fixture_affinity_fault_injector,
):
    injector = fixture_affinity_fault_injector(failure_plan=["ok"])
    injector.set_affinity(pid=11, cpu_list=[3, 2, 3])

    assert injector.get_affinity(11) == [2, 3]
    assert injector.calls == [{"pid": 11, "cpu_list": [3, 2, 3], "timeout_ns": None}]


def test_bind_verify_happy_path_records_requested_and_effective_affinity_bind_verify(
    fixture_affinity_fault_injector,
):
    module = _load_module()
    injector = fixture_affinity_fault_injector(failure_plan=["ok"])

    result = module.request_set_readback_affinity(
        pid=42,
        requested_affinity=[3, 2, 3],
        set_affinity=injector.set_affinity,
        get_affinity=injector.get_affinity,
        get_mems=lambda: [0],
    )

    assert result["bind_status"] == module.AFFINITY_BIND_STATUS_OK
    assert result["requested_affinity"] == [2, 3]
    assert result["effective_affinity"] == [2, 3]
    assert result["effective_mems"] == [0]
    assert result["ranking_valid"] is True
    assert result["failure"] is None


def test_bind_verify_mismatch_marks_sample_invalid_for_ranking_bind_verify():
    module = _load_module()

    def _set_affinity(_pid: int, _cpu_list: list[int], timeout_ns: int | None = None):
        _ = timeout_ns

    def _get_affinity(_pid: int):
        return [2]

    bind_result = module.request_set_readback_affinity(
        pid=42,
        requested_affinity=[2, 3],
        set_affinity=_set_affinity,
        get_affinity=_get_affinity,
    )
    assert bind_result["bind_status"] == module.AFFINITY_BIND_STATUS_MISMATCH
    assert bind_result["mismatch"] is True
    assert bind_result["ranking_valid"] is False

    result = module.mark_result_invalid_for_ranking_on_mismatch(
        {
            "bind_status": bind_result["bind_status"],
            "mismatch": bind_result["mismatch"],
            "ranking_valid": bind_result["ranking_valid"],
            "samples": [{"sample_id": "s1", "latency_ms": 1.0}],
        }
    )
    assert result["ranking_valid"] is False
    assert result["samples"][0]["ranking_valid"] is False


def test_bind_verify_classifies_eperm_einval_timeout_other_bind_verify(
    fixture_affinity_fault_injector,
    fixture_controlled_clock,
):
    module = _load_module()

    eperm_injector = fixture_affinity_fault_injector(failure_plan=["EPERM"])
    eperm_result = module.request_set_readback_affinity(
        pid=1,
        requested_affinity=[1],
        set_affinity=eperm_injector.set_affinity,
        get_affinity=eperm_injector.get_affinity,
    )
    assert eperm_result["bind_status"] == module.AFFINITY_BIND_STATUS_ERROR
    assert eperm_result["failure"]["failure_code"] == module.AFFINITY_FAILURE_CODE_EPERM

    einval_injector = fixture_affinity_fault_injector(failure_plan=["EINVAL"])
    einval_result = module.request_set_readback_affinity(
        pid=1,
        requested_affinity=[1],
        set_affinity=einval_injector.set_affinity,
        get_affinity=einval_injector.get_affinity,
    )
    assert einval_result["failure"]["failure_code"] == module.AFFINITY_FAILURE_CODE_EINVAL

    clock = fixture_controlled_clock(start_ns=0, step_ns=1)
    timeout_injector = fixture_affinity_fault_injector(failure_plan=["TIMEOUT"], clock=clock)
    timeout_result = module.request_set_readback_affinity(
        pid=1,
        requested_affinity=[1],
        set_affinity=timeout_injector.set_affinity,
        get_affinity=timeout_injector.get_affinity,
        timeout_ns=5,
    )
    assert timeout_result["failure"]["failure_code"] == module.AFFINITY_FAILURE_CODE_TIMEOUT

    def _raise_other(_pid: int, _cpu_list: list[int], timeout_ns: int | None = None):
        _ = timeout_ns
        raise RuntimeError("boom")

    other_result = module.request_set_readback_affinity(
        pid=1,
        requested_affinity=[1],
        set_affinity=_raise_other,
        get_affinity=lambda _pid: [1],
    )
    assert other_result["failure"]["failure_code"] == module.AFFINITY_FAILURE_CODE_OTHER


def test_remeasure_triggers_on_12pct_regression_retry_remeasure():
    module = _load_module()

    result = module.evaluate_baseline_remeasure(
        warmup_throughput=1000.0,
        warmup_latency_p95_ms=10.0,
        first_pass_throughput=1000.0,
        first_pass_latency_p95_ms=10.0,
        remeasure_samples=[
            {"throughput": 880.0, "latency_p95_ms": 11.2},
            {"throughput": 980.0, "latency_p95_ms": 10.8},
        ],
        deviation_ratio=0.10,
        max_remeasure_attempts=2,
    )

    assert result["remeasure_count"] == 2
    assert result["decision_status"] == module.REMEASURE_DECISION_ACCEPTED
    assert len(result["decisions"]) == 2
    assert result["decisions"][0]["decision"] == module.REMEASURE_DECISION_UNSTABLE
    assert result["decisions"][0]["trigger_metrics"] == ["throughput", "latency_p95_ms"]
    assert result["decisions"][1]["decision"] == module.REMEASURE_DECISION_ACCEPTED
    assert result["decisions"][1]["trigger_metrics"] == []


def test_remeasure_does_not_trigger_on_8pct_drift_retry_remeasure():
    module = _load_module()

    result = module.evaluate_baseline_remeasure(
        warmup_throughput=1000.0,
        warmup_latency_p95_ms=10.0,
        first_pass_throughput=1000.0,
        first_pass_latency_p95_ms=10.0,
        remeasure_samples=[{"throughput": 920.0, "latency_p95_ms": 10.8}],
        deviation_ratio=0.10,
        max_remeasure_attempts=2,
    )

    assert result["remeasure_count"] == 1
    assert result["decision_status"] == module.REMEASURE_DECISION_ACCEPTED
    assert result["decisions"]
    assert result["decisions"][0]["decision"] == module.REMEASURE_DECISION_ACCEPTED
    assert result["decisions"][0]["trigger_metrics"] == []
