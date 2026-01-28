import torch
import torch_npu
from contextlib import contextmanager
from .pprint import tprint, ifdebug
from .timer import mprint, get_time_str
from .device import get_torch_device
from torch_npu.contrib import transfer_to_npu

import os
import psutil
import time
import threading
import traceback
import tracemalloc
import sys
from datetime import datetime


# mem_allocated, mem_reserved, mem_used, mem_total = _get_current_mem_info()
def _get_current_mem_info(unit: str = "GB", precision: int = 2) -> tuple[str]:
    """Get current memory usage."""
    assert unit in ["GB", "MB", "KB"]
    divisor = 1024**3 if unit == "GB" else 1024**2 if unit == "MB" else 1024
    mem_allocated = get_torch_device().memory_allocated()
    mem_reserved = get_torch_device().memory_reserved()
    # use get_torch_device().mem_get_info to profile device memory
    # since vllm's sleep mode works below pytorch
    # see https://github.com/vllm-project/vllm/pull/11743#issuecomment-2754338119
    mem_free, mem_total = get_torch_device().mem_get_info()
    mem_used = mem_total - mem_free
    mem_allocated = f"{mem_allocated / divisor:.{precision}f}"
    mem_reserved = f"{mem_reserved / divisor:.{precision}f}"
    mem_used = f"{mem_used / divisor:.{precision}f}"
    mem_total = f"{mem_total / divisor:.{precision}f}"
    return mem_allocated, mem_reserved, mem_used, mem_total

def pmem_str():
    torch.npu.synchronize()
    mega_bytes = 1024.0 * 1024.0 * 1024.0
    string = ' | allocated: {}'.format(
        torch.npu.memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(
        torch.npu.max_memory_allocated() / mega_bytes)
    string += ' | reserved: {}'.format(
        torch.npu.memory_reserved() / mega_bytes)
    string += ' | max reserved: {}'.format(
        torch.npu.max_memory_reserved() / mega_bytes)
    return string


# get_torch_device().mem_get_info()（NPU 上对应 torch.npu.mem_get_info()）拿到的是设备/驱动层面看到的“当前设备还剩多少可用显存/总显存”。
# mem_used = total - free 表示设备上目前被占用的总量（往往是“全局/上下文视角”），会包含很多 PyTorch 追踪不到的部分。
# torch.npu.memory_allocated() 是 PyTorch 分配器统计的「本进程中仍被 Tensor 持有的显存」（active allocations）。它只统计 PyTorch 认为“正在被张量占用”的那部分。
# 因此一般会出现这种关系（同一时刻、同一单位下大概率满足）：
# memory_allocated <= memory_reserved <= mem_used <= mem_total

def report_memory(name, print_rank0_only=True):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    torch.npu.synchronize()
    mem_allocated, mem_reserved, mem_used, mem_total = _get_current_mem_info()
    string += f' | {mem_used}/{mem_total}'
    memory_allocated = torch.npu.memory_allocated() / mega_bytes
    string += ' | allocated: {}'.format(
        memory_allocated)
    string += ' | max allocated: {}'.format(
        torch.npu.max_memory_allocated() / mega_bytes)
    string += ' | reserved: {}'.format(
        torch.npu.memory_reserved() / mega_bytes)
    string += ' | max reserved: {}'.format(
        torch.npu.max_memory_reserved() / mega_bytes)
    

    rank = torch.distributed.get_rank()
    device = torch_npu.npu.current_device()
    if not print_rank0_only or rank == 0:
        record_time = mprint("[Rank {} Device {}] {}".format(rank, device, string))
    # if memory_allocated > 40000: # 40GB
    #     # torch_npu.npu.memory._record_memory_history()
    #     # xxx
    #     torch_npu.npu.memory._dump_snapshot(f"oom_snapshot_rank_{rank}_device_{device}_{memory_allocated}_{record_time}.pickle")
    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats(device)


_pmemory_used = False

# Usage:
# with pmemory("fun xxx"):
#     funx()
@contextmanager
def pmemory(tmp_str):
    global _pmemory_used
    tmp = f"TSJPMEM {tmp_str}"
    mem_str = pmem_str()
    enable = not _pmemory_used
    if True:
        _pmemory_used = True
        with torch.profiler.record_function(tmp+mem_str):
            rank = torch.distributed.get_rank()
            device = torch_npu.npu.current_device()
            torch_npu.npu.reset_peak_memory_stats()
            torch.npu.reset_accumulated_memory_stats()
            torch_npu.npu.memory._record_memory_history()
            report_memory(f"before {tmp}")

            try:
                yield
            except Exception as e:
                # 异常时也 dump 一份
                report_memory(f"exception in {tmp} {e}")
                peak_allocated = torch.npu.memory.max_memory_allocated()
                torch_npu.npu.memory._dump_snapshot(
                    f"{tmp_str}_oom_snapshot_rank_{rank}_device_{device}_{get_time_str()}_pick{peak_allocated}_EXC.pickle"
                )
                torch.npu.reset_peak_memory_stats()
                torch.npu.reset_accumulated_memory_stats()
                raise  # 继续抛出异常，不要吞掉
            else:
                # 正常执行完
                report_memory(f"after {tmp}")
                peak_allocated = torch.npu.memory.max_memory_allocated()
                torch_npu.npu.memory._dump_snapshot(
                    f"{tmp_str}_oom_snapshot_rank_{rank}_device_{device}_{get_time_str()}_pick{peak_allocated}.pickle"
                )
                torch.npu.reset_peak_memory_stats()
                torch.npu.reset_accumulated_memory_stats()
            finally:
                # 无论是否报错，都执行一次
                peak_allocated = torch.npu.memory.max_memory_allocated()
                torch_npu.npu.memory._dump_snapshot(
                    f"{tmp_str}_oom_snapshot_rank_{rank}_device_{device}_{get_time_str()}_pick{peak_allocated}_FINAL.pickle"
                )
                torch.npu.reset_peak_memory_stats()
                torch.npu.reset_accumulated_memory_stats()
    else:
        yield



# def get_tensor_size(param: torch.Tensor) -> int:
#     """计算张量的显存占用（字节）"""
#     dtype_size_map = {
#         torch.float32: 4,
#         torch.float: 4,
#         torch.float16: 2,
#         torch.half: 2,
#         torch.bfloat16: 2,
#         torch.int64: 8,
#         torch.long: 8,
#         torch.int32: 4,
#         torch.int: 4,
#         torch.int16: 2,
#         torch.int8: 1,
#         torch.uint8: 1,
#         torch.bool: 1,
#     }
#     element_size = dtype_size_map.get(param.dtype, 4)  # 默认按4字节处理
#     return param.numel() * element_size  # 总字节数 = 元素数量 × 单元素字节数

def get_tensor_size(tensor):
    """返回 tensor 的内存大小，单位为 GB"""
    return tensor.numel() * tensor.element_size() / (1024 ** 3)

# print model size
def pmsize(model):
    if torch.distributed.get_rank() == 0 and ifdebug():
        print(model)
        print(f"world size is {torch.distributed.get_world_size()}", flush=True)
        for name, module in model.named_modules():
            print(f"\nLayer detail" , flush=True)
            for param_name, param in module.named_parameters():
                tprint(param,param_name)
            break
        for name, module in model.named_modules():
            total_size = 0
            for param_name, param in module.named_parameters():
                total_size += get_tensor_size(param)
            print(f"layer {name} {total_size} GB", flush=True)

def get_main_thread_stack():
    main_thread = threading.main_thread()
    frames = sys._current_frames()
    main_thread_id = main_thread.ident

    if main_thread_id not in frames:
        return "[Main thread stack not found]"

    frame = frames[main_thread_id]
    return ''.join(traceback.format_stack(frame))

def start_host_mem_monitor(unique_str: str, print_interval: int = 5):
    """
    启动后台线程，周期性打印：
    - 当前进程RSS
    - 主线程堆栈
    - Python内存分配快照（tracemalloc）
    - 同时写入日志文件
    """
    pid = os.getpid()
    log_file = f"{unique_str}_{pid}_hostmem.log"
    process = psutil.Process(pid)

    tracemalloc.start()

    def monitor():
        with open(log_file, "a") as f:
            while True:
                now = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
                rss = process.memory_info().rss / 1024 ** 2  # MB

                # 主线程堆栈
                # stack_str = get_main_thread_stack()

                 # 主机内存使用情况
                virtual_mem = psutil.virtual_memory()
                total_mem = virtual_mem.total / 1024 ** 3  # GB
                used_mem = virtual_mem.used / 1024 ** 3    # GB
                available_mem = virtual_mem.available / 1024 ** 3  # GB
                shared_mem = virtual_mem.shared / 1024 ** 3    # GB
                mem_percent = virtual_mem.percent

                # tracemalloc统计
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')

                msg_lines = [
                    f"{now} [PID {pid}] RSS: {rss:.2f} MB",
                    f"Host Memory - Total: {total_mem:.2f} GB, Used: {used_mem:.2f} GB, Shared: {shared_mem:.2f} GB, Available: {available_mem:.2f} GB, Usage: {mem_percent:.1f}%",
                    # f"--- Main Thread Stack ---",
                    # stack_str.strip(),
                    f"--- Top 10 memory usage (tracemalloc) ---",
                ]
                for stat in top_stats[:10]:
                    msg_lines.append(str(stat))

                msg_lines.append("-" * 80)
                msg = "\n".join(msg_lines)

                # print(msg)
                f.write(msg + "\n")
                f.flush()

                time.sleep(print_interval)

    threading.Thread(target=monitor, daemon=True).start()
