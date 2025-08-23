import os
import torch.distributed as dist
import psutil
from contextlib import ContextDecorator

# 使用方式
#     dist.init_process_group("nccl")  # 或 gloo
#     with DistAffinityPriority(total_cpus=192, total_ranks=32, high_priority=-10):
#         # 在这里跑分布式代码
#         print(f"Rank {dist.get_rank()} running with affinity {psutil.Process().cpu_affinity()}")
class DistAffinityPriority(ContextDecorator):
    def __init__(self, total_cpus=192, total_ranks=32, high_priority=-10):
        self.total_cpus = total_cpus
        self.total_ranks = total_ranks
        self.high_priority = high_priority
        self.old_affinity = None
        self.old_priority = None

    def __enter__(self):
        pid = os.getpid()
        proc = psutil.Process(pid)

        # 获取当前 rank
        rank = dist.get_rank() if dist.is_initialized() else 0
        rank = rank - 16 if rank >= 16 else rank

        # 计算分配的 CPU 区间
        cpus_per_rank = self.total_cpus // self.total_ranks
        start = rank * cpus_per_rank
        end = start + cpus_per_rank
        cpu_set = list(range(start, end))

        # 保存旧设置
        self.old_affinity = proc.cpu_affinity()
        self.old_priority = proc.nice()

        # 设置新 affinity
        proc.cpu_affinity(cpu_set)

        # 设置高优先级（负数 = 更高优先级，需要 root 权限）
        try:
            proc.nice(self.high_priority)
        except PermissionError:
            print(f"[Rank {rank}] Warning: No permission to set priority {self.high_priority}")

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # pid = os.getpid()
        # proc = psutil.Process(pid)

        # # 恢复 affinity
        # if self.old_affinity is not None:
        #     proc.cpu_affinity(self.old_affinity)

        # # 恢复 nice
        # if self.old_priority is not None:
        #     try:
        #         proc.nice(self.old_priority)
        #     except PermissionError:
        #         print(f"Warning: No permission to restore priority {self.old_priority}")

        return False   # 不屏蔽异常



