import time
import os
import torch
import torch_npu
from contextlib import contextmanager

# Usage:
# export MM_ENABLE_PROFILING=1
# with profiling_this(True):
#     funx()


@contextmanager
def profiling_this(encoder_count):
    from datetime import datetime

    # 获取当前时间
    current_time = datetime.now()

    # 格式化为可读字符串（例如：2024-07-15 14:30:45）
    readable_time = current_time.strftime("%Y-%m-%d_%H_%M")
    profiling_path = os.environ.get("MM_ENABLE_PROFILING_PATH", f"/home/t00906153/profiling/{readable_time}")
    torch.npu.synchronize()
    start_time = time.time()
    if int(os.environ.get("MM_ENABLE_PROFILING", "0")) == 0:
        yield
    else:
        if encoder_count:
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                data_simplification=False,
                export_type=[
                    torch_npu.profiler.ExportType.Db
                ],
            )
            with torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,  # CPU侧python调用
                    torch_npu.profiler.ProfilerActivity.NPU],
                with_stack=False, # CPU侧python调用
                record_shapes=True,
                profile_memory=True,
                with_modules=False, # CPU侧python调用
                schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiling_path),
                experimental_config=experimental_config
            ):

                yield  # 在此处执行目标代码行
        else:
            yield
    torch.npu.synchronize()
    end_time = time.time()
    run_time = end_time - start_time
    print(f"time cost is: {run_time * 1000:.2f} ms")
