import time
import os
import torch
import torch_npu
from contextlib import contextmanager

# 函数装饰器，用于mstx采集关心时间范围
# tip1：如果嵌套函数两个函数都使用，只有内层会显示
# tip2: 内部的不需要prof, 直接使用mark, range_start api就行
def npu_profiler(range_name="update_actor start"):
    """
    NPU性能分析装饰器, 只采集关系函数
    
    Args:
        range_name (str): 性能分析范围的标记名称
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 创建实验配置
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                export_type=[
                    torch_npu.profiler.ExportType.Text,
                    torch_npu.profiler.ExportType.Db
                ],
                profiler_level=torch_npu.profiler.ProfilerLevel.Level_none, # 不需要其余信息
                aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
                l2_cache=False,
                op_attr=False,
                data_simplification=False,
                record_op_args=False,
                gc_detect_threshold=None,
                msprof_tx=True, # mstx开关
                mstx_domain_exclude=["communication"],
            )
            
            # 开始性能分析
            with torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.NPU],
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./mstx_result"),
                experimental_config=experimental_config):
                
                # 标记初始化点
                torch_npu.npu.mstx.mark(f"{range_name.split(' ')[0]} init")
                
                # 开始范围标记
                p_id1 = torch_npu.npu.mstx.range_start(range_name)
                
                try:
                    # 执行被装饰的函数
                    result = func(*args, **kwargs)
                    return result
                finally:
                    # 结束范围标记
                    torch_npu.npu.mstx.range_end(p_id1)
        
        return wrapper
    return decorator

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
