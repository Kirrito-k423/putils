# NPU OOM Debug Helper 使用说明

`putils.npu_oom_debug` 用于在 Ascend NPU 训练进程发生 OOM 时，把失败现场的
PyTorch allocator、driver HBM、memory snapshot 和可选的大块分配探测一起落盘。

它适合排查这类问题：

- 日志显示 `Tried to allocate 12.26 GiB`，但 `free` 仍然有几十 GiB。
- FSDP / FSDP2 forward prefetch 或 all-gather 阶段 OOM。
- 怀疑是 PyTorch 缓存池碎片、pending event、VMM 映射、页表/physical handle 压力，或者单次大块分配限制。

## 安装或接入

在训练环境中任选一种方式：

```bash
# 方式 1：从 GitHub 安装
pip install git+https://github.com/Kirrito-k423/putils.git

# 方式 2：开发目录直接加入 PYTHONPATH
export PYTHONPATH=/path/to/putils:${PYTHONPATH}
```

训练代码里导入：

```python
from putils.npu_oom_debug import dump_npu_oom_debug
```

## 推荐环境变量

```bash
# OOM debug 产物目录。建议放到共享盘或当前实验 work_dir。
export NPU_OOM_DEBUG_DIR=/mnt/huawei/hyf/npu_oom_debug

# 只有想在报错进程里额外二分探测“现场最大可申请块”时才打开。
# 这个 probe 会额外申请/释放大 tensor，适合进程即将失败的场景。
export NPU_OOM_DEBUG_PROBE=1
```

不想跑额外分配探测时不要设置 `NPU_OOM_DEBUG_PROBE`，或设为 `0`。

## 最小接入方式

在任何可能 OOM 的代码块外包一层：

```python
import torch
from putils.npu_oom_debug import dump_npu_oom_debug


try:
    # 原来的训练 / forward / all-gather 逻辑
    output = model(**batch)
except torch.OutOfMemoryError as exc:
    dump_npu_oom_debug(
        tag="train_forward",
        requested_bytes=None,
        error=exc,
    )
    raise
```

如果当前 torch 版本没有单独的 `torch.OutOfMemoryError`，可以兼容成：

```python
try:
    output = model(**batch)
except RuntimeError as exc:
    if "out of memory" in str(exc).lower():
        dump_npu_oom_debug(tag="train_forward", error=exc)
    raise
```

## FSDP all-gather 报错点接入模板

如果 OOM 栈落在：

```text
torch_npu/distributed/fsdp/_fsdp_collectives.py
all_gather_output = torch.empty(...)
```

建议在该 `torch.empty(...)` 外层加：

```python
from putils.npu_oom_debug import dump_npu_oom_debug


try:
    all_gather_output = torch.empty(
        # 保持原始参数不变
    )
except torch.OutOfMemoryError as exc:
    # 如果能拿到 shape/dtype，尽量精确计算 requested_bytes。
    # 不能精确拿到时可以先传 None，OOM 原始异常仍会写入 JSON。
    requested_bytes = None
    try:
        requested_bytes = all_gather_numel * dtype_byte_size
    except Exception:
        pass

    dump_npu_oom_debug(
        tag="fsdp_all_gather_copy_in_npu",
        requested_bytes=requested_bytes,
        error=exc,
        dump_dir="/mnt/huawei/hyf/npu_oom_debug",
    )
    raise
```

如果代码里 `torch.empty` 的 size 和 dtype 都是现成变量，可以这样算：

```python
requested_numel = 1
for dim in output_shape:
    requested_numel *= dim
requested_bytes = requested_numel * torch.empty((), dtype=output_dtype).element_size()
```

## 输出文件

每次调用会打印一行摘要，并生成三个文件：

```text
[NPU_OOM_DEBUG][rank=51] tag=fsdp_all_gather_copy_in_npu requested=12.260 GiB \
driver_free=48.500 GiB driver_total=61.280 GiB reserved=11.310 GiB \
allocated=1.260 GiB largest_inactive=... inactive_total=... \
summary=/path/npu_oom_rank51_1782792680.json \
memory_summary=/path/npu_oom_rank51_1782792680.txt \
snapshot=/path/npu_oom_rank51_1782792680.pickle
```

- `*.json`：结构化诊断结果，最适合机器解析和贴到 issue。
- `*.txt`：`torch.npu.memory_summary()` 文本结果。
- `*.pickle`：NPU memory snapshot，可进一步离线分析 allocator blocks。

## JSON 重点字段

```json
{
  "requested_gib": 12.26,
  "mem_get_info": {
    "free_bytes": 52000000000,
    "total_bytes": 65800000000
  },
  "memory_allocated": 1350000000,
  "memory_reserved": 12100000000,
  "snapshot_summary": {
    "inactive_total_bytes": 10000000000,
    "largest_inactive_block_bytes": 3000000000,
    "block_states": {
      "active_allocated": 10,
      "inactive": 23
    }
  },
  "probe_before_empty_cache": {
    "max_ok_gib_approx": 8.2,
    "first_fail_gib_approx": 8.3
  },
  "probe_after_empty_cache": {
    "max_ok_gib_approx": 12.8,
    "first_fail_gib_approx": 12.9
  }
}
```

字段含义：

- `mem_get_info.free_bytes`：driver/runtime 视角的当前设备可用 HBM。
- `memory_allocated`：PyTorch 仍被 tensor 持有的 active 显存。
- `memory_reserved`：PyTorch caching allocator 已经向设备保留的显存。
- `inactive_total_bytes`：snapshot 中非 active block 总量。
- `largest_inactive_block_bytes`：snapshot 中最大的非 active block。
- `probe_before_empty_cache`：不主动清 cache 的现场最大可申请块。
- `probe_after_empty_cache`：调用 `empty_cache()` 后的最大可申请块。

## 结果解读

### 1. 判断是否是 PyTorch 缓存池碎片

```text
inactive_total_bytes 很大
largest_inactive_block_bytes 明显小于 requested_bytes
```

说明 PyTorch 缓存池里有不少空闲/非 active block，但单块不够大，或不能被当前 stream 立即复用。
这时更像 allocator 碎片、split block、pending event 或 non-releasable block 问题。

### 2. 判断是否是 empty_cache 后可恢复

```text
probe_before_empty_cache < requested_gib
probe_after_empty_cache >= requested_gib
```

说明不是硬件单次分配限制，而是训练现场缓存池状态造成的。可以继续检查：

- FSDP prefetch 是否同时保留当前层和下一层 all-gather 权重。
- 是否存在多 stream event 延迟释放。
- 是否有临时 buffer 生命周期比预期更长。

### 3. 判断是否是 VMM / physical handle / 页表压力

```text
mem_get_info.free_bytes 很大
probe_after_empty_cache 仍明显小于 requested_gib
```

这时 HBM 总字节数看起来足够，但当前进程仍无法申请大块 tensor。可能原因包括：

- `expandable_segments=True` 下需要映射大量物理段。
- `segment_size_mb` 太小，12GiB 级别 tensor 需要数百个 physical handle。
- CANN/driver 的 VMM、页表、huge page 或 handle 资源到达边界。

可以 A/B 测试：

```bash
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True,segment_size_mb:128
# 或
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True,segment_size_mb:256
```

如果大 `segment_size_mb` 后同一 OOM 点消失，说明主要瓶颈不是 HBM 容量，而是小 segment
带来的 VMM 映射/handle 压力。

### 4. 判断是否是 FSDP 参数峰值过大

如果 `requested_gib` 已经接近单层权重 all-gather 大小，例如 EP1 下单层 MoE 权重达到
`12GiB+`，并且开启了 FSDP prefetch，则峰值至少要按下面思路估算：

```text
常驻 FSDP shard
+ 当前层 all-gather 输出
+ 预取层 all-gather 输出
+ all_gather_copy_in / staging buffer
+ 激活和 workspace
```

这类问题不是 debug helper 能“修复”的，需要降低单层 materialize 权重或减少重叠：

- 增大 EP，减少每层每 rank 需要 all-gather 的专家权重。
- 关闭或减弱 forward prefetch 做诊断。
- 拆更细的 FSDP wrap。
- 引入 TP/PP 或降低序列/微批。

## 注意事项

- `NPU_OOM_DEBUG_PROBE=1` 会在 OOM 后额外申请 tensor，只建议用于即将失败的 debug run。
- 多 rank 场景下每个 rank 都会写文件，请确认 `NPU_OOM_DEBUG_DIR` 空间足够。
- `snapshot` 和 `memory_summary` 依赖当前 torch/torch_npu 版本；如果接口不存在，工具会记录错误字段但不会吞掉原始异常。
- 工具只负责采集现场证据，不会自动释放训练中的活跃 tensor，也不会改变原始 OOM 行为。
