import csv
import functools
import os
import time
from contextlib import contextmanager
from datetime import datetime

import torch
import torch.distributed as dist

from .write2file import log2file

try:
    import torch_npu
except ImportError:  # pragma: no cover
    torch_npu = None


# Global switch for CPU timer output.
TIMER_VERBOSE = True


@contextmanager
def timer(name="", print_rank0_only=True):
    # Usage:
    #   with timer("sleep 1s"):
    #       time.sleep(1)
    if TIMER_VERBOSE:
        start = time.perf_counter()
        rank = dist.get_rank() if dist.is_initialized() else 0
        if not print_rank0_only or rank == 0:
            print(f"[Timer] '{name}' started...")
            prefix_str = f"{name} func started"
            log2file(f"{name} {mprint(prefix_str)}")
    try:
        yield
    finally:
        if TIMER_VERBOSE:
            end = time.perf_counter()
            duration = end - start
            rank = dist.get_rank() if dist.is_initialized() else 0
            if not print_rank0_only or rank == 0:
                print(f"[Timer] '{name}' finished in {duration:.6f} seconds.")
                prefix_str = f"{name} func end"
                log2file(f"{name} {mprint(prefix_str)} during {duration}")


def timer_decorator(name=None, print_rank0_only=True):
    # Usage:
    #   @timer_decorator("custom name")
    #   def my_func():
    #       time.sleep(0.5)
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with timer(name or func.__name__, print_rank0_only=print_rank0_only):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def to_filename_safe(timestamp_str):
    return (
        timestamp_str.replace("-", "")
        .replace(":", "")
        .replace(" ", "_")
        .replace(".", "_ms")
    )


def get_formatted_time():
    now = time.time()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
    milliseconds = int((now * 1000) % 1000)
    return f"{timestamp}.{milliseconds:03d}"


def get_time_str():
    return to_filename_safe(get_formatted_time())


def mprint(msg, level="INFO"):
    formatted_time = get_formatted_time()
    print(f"[{formatted_time}] [{level}] {msg}")
    return to_filename_safe(formatted_time)


def _check_disabled(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.disable:
            return None
        return func(self, *args, **kwargs)

    return wrapper


class CUDAEVENT_TIMER:
    """NPU event based timer for compute and communication streams.

    Events are grouped by label. Each label must be recorded in pairs:
    add_event("label") before the region and add_event("label") after it.
    flush() synchronizes the NPU, computes elapsed time for each pair, and
    writes one CSV row per rank.
    """

    def __init__(self):
        self.events = {}
        self.times = {}
        self.cpu_times = []
        self.shapes = {}
        self.flush_count = 0
        self.label_to_func = {}
        self.csv_output = True
        self.disable = True

    def _require_npu(self):
        if torch_npu is None:
            raise ImportError("torch_npu is required for CUDAEVENT_TIMER on NPU.")
        if not hasattr(torch, "npu"):
            raise RuntimeError("torch.npu is required for CUDAEVENT_TIMER.")

    @_check_disabled
    def add_cpu(self):
        self._require_npu()
        torch.npu.synchronize()
        self.cpu_times.append(time.time())

    @_check_disabled
    def add_tensor_shape(self, label, tensor):
        assert isinstance(tensor, torch.Tensor), f"tensor {tensor} is not a torch.Tensor"
        if label not in self.shapes:
            self.shapes[label] = tensor.shape

    def _get_stream_from_group(self, group):
        self._require_npu()
        if group is None:
            return None
        stream_id = group._get_backend(torch.device("npu"))._get_stream_id(False)
        if stream_id is None:
            return None
        if stream_id < 0:
            return torch.npu.current_stream()
        return torch_npu.npu.Stream(
            stream_id=stream_id,
            device_type=20,
            device_index=torch.distributed.get_rank() % 16,
        )

    @_check_disabled
    def add_event(self, label, group=None):
        """Record one NPU event on the current compute stream or a group's stream.

        Args:
            label: Events with the same label are consumed in start/end pairs.
            group: Optional distributed process group. When provided, the event
                is recorded on the communication stream associated with group.
        """
        self._require_npu()
        if label not in self.times:
            self.times[label] = []
            self.events[label] = []
        stream = self._get_stream_from_group(group)
        event = torch_npu.npu.Event(enable_timing=True)
        event.record(stream=stream)
        self.events[label].append(event)

    @_check_disabled
    def patch_fsdp_all_gather(self, label, patch, group=None):
        """Patch torch FSDP foreach_all_gather and record start/end events."""
        self._require_npu()
        from torch.distributed.fsdp._fully_shard import _fsdp_collectives
        from torch.distributed.fsdp._fully_shard import _fsdp_param_group

        if patch:
            _orig_foreach_all_gather = _fsdp_collectives.foreach_all_gather

            @functools.wraps(_orig_foreach_all_gather)
            @torch.no_grad()
            def _patched_foreach_all_gather(
                fsdp_params,
                process_group,
                async_op,
                all_gather_copy_in_stream,
                all_gather_stream,
                device,
            ):
                stream = self._get_stream_from_group(group) if group is not None else all_gather_stream
                ev_start = torch_npu.npu.Event(enable_timing=True)
                ev_end = torch_npu.npu.Event(enable_timing=True)
                ev_start.record(stream=stream)

                result = _orig_foreach_all_gather(
                    fsdp_params,
                    process_group,
                    async_op,
                    all_gather_copy_in_stream,
                    all_gather_stream,
                    device,
                )

                ev_end.record(stream=stream)
                if label not in self.times:
                    self.times[label] = []
                    self.events[label] = []
                self.events[label].append(ev_start)
                self.events[label].append(ev_end)
                return result

            _fsdp_collectives.foreach_all_gather = _patched_foreach_all_gather
            _fsdp_param_group.foreach_all_gather = _patched_foreach_all_gather
            self.label_to_func[label] = _orig_foreach_all_gather
            return

        if label not in self.label_to_func:
            raise ValueError(f"label {label} not patched")
        _fsdp_collectives.foreach_all_gather = self.label_to_func[label]
        _fsdp_param_group.foreach_all_gather = self.label_to_func[label]

    def _get_log_dir(self):
        return os.environ.get("LOG_DIR", "./logs")

    def _get_csv_path(self):
        log_dir = self._get_log_dir()
        os.makedirs(log_dir, exist_ok=True)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        return os.path.join(log_dir, f"event_timer_rank{rank}.csv")

    @_check_disabled
    def flush(self):
        self._require_npu()
        torch.npu.synchronize()
        self.flush_count += 1
        if not self.csv_output:
            self.cpu_times = []
            self.events = {}
            self.times = {}
            self.shapes = {}
            return

        row_data = {
            "timestamp": datetime.now().isoformat(),
            "flush_count": self.flush_count,
            "device": torch.npu.current_device(),
        }

        for label in sorted(self.times):
            events = self.events[label]
            if len(events) % 2 != 0:
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
                print(f"[rank {rank}] got {len(events)} {label} events, should be paired")
                row_data[f"{label}_status"] = "unpaired"
            else:
                times = []
                for i in range(0, len(events), 2):
                    times.append(events[i].elapsed_time(events[i + 1]))
                if label == "combine2_and_sharedEp1":
                    times = [-t for t in times]
                row_data[f"{label}_max_ms"] = max(times) if times else 0
                row_data[f"{label}_min_ms"] = min(times) if times else 0
                row_data[f"{label}_avg_ms"] = sum(times) / len(times) if times else 0
                row_data[f"{label}_times_ms"] = str(times)

        if len(self.cpu_times) == 2:
            cpu_time_ms = (self.cpu_times[1] - self.cpu_times[0]) * 1000
            row_data["cpu_time_ms"] = cpu_time_ms
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            print(f"\n [rank {rank}] cpu_time: {cpu_time_ms} ms")

        for label in self.shapes:
            row_data[f"{label}_shape"] = str(self.shapes[label])

        self._save_to_csv(row_data)
        self.cpu_times = []
        self.events = {}
        self.times = {}
        self.shapes = {}

    def _save_to_csv(self, row_data):
        csv_path = self._get_csv_path()
        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row_data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_data)


event_timer = CUDAEVENT_TIMER()
