import torch
import torch_npu
from contextlib import contextmanager
from .pprint import tprint, ifdebug
from .timer import mprint

def pmem_str():
    torch.npu.synchronize()
    mega_bytes = 1024.0 * 1024.0 * 1024.0
    string = ' | allocated: {}'.format(
        torch.cuda.memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(
        torch.cuda.max_memory_allocated() / mega_bytes)
    string += ' | reserved: {}'.format(
        torch.cuda.memory_reserved() / mega_bytes)
    string += ' | max reserved: {}'.format(
        torch.cuda.max_memory_reserved() / mega_bytes)
    return string


def report_memory(name):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    torch.npu.synchronize()
    memory_allocated = torch.cuda.memory_allocated() / mega_bytes
    string += ' | allocated: {}'.format(
        memory_allocated)
    string += ' | max allocated: {}'.format(
        torch.cuda.max_memory_allocated() / mega_bytes)
    string += ' | reserved: {}'.format(
        torch.cuda.memory_reserved() / mega_bytes)
    string += ' | max reserved: {}'.format(
        torch.cuda.max_memory_reserved() / mega_bytes)
    # if mpu.get_data_parallel_rank() == 0:
    rank = torch.distributed.get_rank()
    device = torch_npu.npu.current_device()
    record_time = mprint("[Rank {} Device {}] {}".format(rank, device, string))
    if memory_allocated > 40000: # 40GB
        # torch_npu.npu.memory._record_memory_history()
        # xxx
        torch_npu.npu.memory._dump_snapshot(f"oom_snapshot_rank_{rank}_device_{device}_{memory_allocated}_{record_time}.pickle")
    torch.npu.synchronize()
    
# Usage:
# with pmemory("fun xxx"):
#     funx()
@contextmanager
def pmemory(tmp_str):
    tmp = f"TSJPMEM {tmp_str}"
    mem_str = pmem_str()
    if ifdebug():
        with torch.profiler.record_function(tmp+mem_str):
            report_memory(f"before {tmp}")
            yield
            report_memory(f"after {tmp}")
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