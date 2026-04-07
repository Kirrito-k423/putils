import errno
import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "npu_affinity_benchmark.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("npu_affinity_benchmark", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {MODULE_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_failure_code_classifies_errno_failure_code():
    module = _load_module()

    eperm = module.classify_affinity_failure(OSError(errno.EPERM, "permission denied"))
    einval = module.classify_affinity_failure(OSError(errno.EINVAL, "invalid argument"))

    assert eperm["failure_code"] == module.AFFINITY_FAILURE_CODE_EPERM
    assert eperm["errno"] == errno.EPERM
    assert einval["failure_code"] == module.AFFINITY_FAILURE_CODE_EINVAL
    assert einval["errno"] == errno.EINVAL


def test_failure_code_classifies_timeout_and_other_failure_code():
    module = _load_module()

    timeout = module.classify_affinity_failure(TimeoutError("timed out"))
    other = module.classify_affinity_failure(RuntimeError("unexpected"))

    assert timeout["failure_code"] == module.AFFINITY_FAILURE_CODE_TIMEOUT
    assert timeout["errno"] is None
    assert other["failure_code"] == module.AFFINITY_FAILURE_CODE_OTHER
    assert other["errno"] is None


def test_failure_code_is_emitted_in_bind_result_failure_code():
    module = _load_module()

    def _raise_eperm(_pid: int, _cpus: list[int], timeout_ns: int | None = None):
        _ = timeout_ns
        raise OSError(errno.EPERM, "permission denied")

    bind_result = module.request_set_readback_affinity(
        pid=99,
        requested_affinity=[1, 2],
        set_affinity=_raise_eperm,
        get_affinity=lambda _pid: [1, 2],
        get_mems=lambda: [0],
    )

    assert bind_result["bind_status"] == module.AFFINITY_BIND_STATUS_ERROR
    assert bind_result["ranking_valid"] is False
    assert bind_result["failure"]["failure_code"] == module.AFFINITY_FAILURE_CODE_EPERM
