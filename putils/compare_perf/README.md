# compare_perf

## compare_perf 快速接入与结果解读

本节聚焦 `putils.compare_perf`，目标是低侵入记录关键路径耗时，并对 baseline/target 两次运行做可回放对比。

### 1) 最小接入, with 和 decorator 两种方式

```python
from putils.compare_perf import TimingCollector, compare_perf


collector = TimingCollector(sync_mode="none")


def run_step(x: int) -> int:
    with compare_perf("forward_step", collector=collector, threshold_seconds=0.05):
        return x + 1


@compare_perf("post_process", collector=collector, threshold_seconds=0.05)
def post_process(x: int) -> int:
    return x * 2


value = run_step(1)
value = post_process(value)

print([event.name for event in collector.events])
print(collector.summary)
```

说明:
- `threshold_seconds` 只影响是否保留 `events` 明细, 不影响 `summary` 聚合统计。
- `collector.events` 适合追踪明细时序, `collector.summary` 适合汇总对比。

#### 训练循环里每 N step 手动落盘一次（中间快照）

`with compare_perf(...)` 和 `@compare_perf(...)` 默认只会把数据写到内存里的 `collector`，不会自动写文件。

如果你想在训练中间查看结果，可以按 step 周期手动导出快照。下面示例每 10 step 导出一次，文件名包含 `step` 和 `tag`。

```python
import json
from pathlib import Path

from putils.compare_perf import TimingCollector, compare_perf
from putils.compare_perf.schema import build_schema


collector = TimingCollector(sync_mode="none")
snapshot_dir = Path("./compare_perf_snapshots")
snapshot_dir.mkdir(parents=True, exist_ok=True)

tag = "train-baseline"
snapshot_every = 10

for step in range(1, 51):
    with compare_perf("train.step", collector=collector, threshold_seconds=0.05):
        # your training step
        pass

    if step % snapshot_every == 0:
        events_payload = [
            {
                "name": event.name,
                "duration_ns": event.duration_ns,
                "start_ns": event.start_ns,
                "end_ns": event.end_ns,
            }
            for event in collector.events
        ]
        payload = build_schema(
            events=events_payload,
            run_metadata={"tag": tag, "step": step},
            alignment={
                "matched": [],
                "ambiguous": [],
                "unmatched": {"left": [], "right": []},
                "counts": {"matched": 0, "ambiguous": 0, "unmatched": 0},
            },
            summary=collector.summary,
        )
        snapshot_path = snapshot_dir / f"compare_perf_step_{step}_{tag}.json"
        with snapshot_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

```

提示:
- `threshold_seconds` 可能让 `events` 在某些 step 为空, 但 `summary` 仍会持续累积。
- 中间快照是手动导出, 适合训练中途审阅趋势和回归信号。

### 2) 参数与阈值解释

- `sync_mode`: `none` 或 `boundary`
  - `none`: 不执行额外同步，额外开销最低。
  - `boundary`: 在 scope 进入和退出时调用 `synchronize`，适合需要边界对齐的设备计时。
- `threshold_seconds`: 必须 `> 0`
  - 单次 scope 耗时 `< threshold_seconds` 时，不写入 `events`。
  - 但仍会累计到 `summary[scope_name]` 的 `call_count/total_ns/total_seconds`。

### 3) collect/diff CLI 工作流

推荐先生成两份 schema 日志，再做 diff 导出。

```bash
# 1) collect baseline
python3 -m putils.compare_perf.cli collect \
  --event encoder.layer.0.attn:100 \
  --event encoder.layer.1.mlp:40 \
  --output baseline.json \
  --tag baseline

# 2) collect target
python3 -m putils.compare_perf.cli collect \
  --event enc.block.0.attention:120 \
  --event enc.block.1.feedforward:20 \
  --output target.json \
  --tag target

# 3) diff + export
python3 -m putils.compare_perf.cli diff \
  --baseline baseline.json \
  --target target.json \
  --trace-out compare_perf_trace.json \
  --trace-aligned-out trace_aligned.json \
  --trace-aligned-stack-out trace_aligned_stack.json \
  --summary-json-out compare_perf_summary.json \
  --summary-md-out compare_perf_summary.md \
  --top-n 5
```

`diff` 命令会输出 5 行路径，依次是:
1. trace json
2. aligned trace json
3. aligned stack trace json
4. summary json
5. summary markdown

### 4) 映射机制, --mapping 和 --alignment-cache

默认会按 token/layer/order 做规则匹配，输出 `matched/ambiguous/unmatched`。

- 手工确认映射:

```bash
python3 -m putils.compare_perf.cli diff \
  --baseline baseline.json \
  --target target.json \
  --mapping encoder.layer.1.mlp=enc.block.1.feedforward
```

- 开启映射缓存，复用上次确认结果:

```bash
python3 -m putils.compare_perf.cli diff \
  --baseline baseline.json \
  --target target.json \
  --alignment-cache .cache/compare_perf_alignment.json \
  --mapping encoder.layer.1.mlp=enc.block.1.feedforward
```

补充:
- `--mapping` 可重复传入。
- `--enable-rank-aggregation` 可在日志包含 `run_metadata.rank_summary` 时生成 p50/p95 聚合对比。

### 5) 结果解读, 五个产物各看什么

- `compare_perf_trace.json`
  - Chrome Trace 格式，`displayTimeUnit=us`，可直接导入 `chrome://tracing`。
- `trace_aligned.json`
  - 匹配到的 baseline/target 模块按 pair 对齐展示，适合快速看同一 pair 的耗时差。
- `trace_aligned_stack.json`
  - 在 pair 对齐基础上保留 parent + child 层级，适合看父模块定位后的子模块对齐细节。
- `compare_perf_summary.json`
  - 结构化对比结论，重点关注:
    - `alignment.counts`
    - `diff.counts`
    - `top_regressions` / `top_improvements`
- `compare_perf_summary.md`
  - 便于审阅和贴到评审系统的文本摘要。

### 6) aligned stack 语义说明

`trace_aligned_stack.json` 主要服务于“父模块定位 + 子模块对齐观察”，可先找异常父模块，再看其子层级。

- `parent_mode=synthetic_aligned`
  - parent 是对齐视图中的 synthetic 父跨度，用于稳定承载子模块对齐结果。
- `child_match_status`
  - `matched`: 子模块在两侧成功匹配并可直接对齐。
  - `ambiguous`: 候选不唯一或置信度不足，需人工判断。
  - `unmatched`: 只在一侧出现，另一侧缺失。
  - `parent`: 当前事件是父层级标记，不是子匹配条目。
- child 对齐优先级
  - 在每个已匹配父模块内，先做 child alignment，再回填到 aligned stack 视图。

### 7) 常见错误排查

#### A. ERROR[2] invalid --event format

现象: `collect` 报错 `expected <scope_name>:<duration_ms>`。

排查:
- 确认 `--event` 形如 `name:123.4`。
- `duration_ms` 必须是数字且 `> 0`。

#### B. ERROR[2] invalid --mapping format

现象: `diff` 报错 `expected <baseline_module>=<target_module>`。

排查:
- 确认每个 `--mapping` 都包含 `=`。
- 左右模块名都不能为空。

#### C. ERROR[10] missing input log

现象: `diff` 报错 `missing input log`。

排查:
- 检查 `--baseline/--target` 路径是否存在。
- 确认路径指向文件，不是目录。

#### D. ERROR[11] invalid json in input log

现象: `diff` 报错 `invalid json in input log`。

排查:
- 输入文件可能截断或写坏。
- 先用 `collect` 重新生成一次日志再重试。

#### E. ERROR[11] incompatible schema in input log

现象: `diff` 报错 `incompatible schema` 或缺少 `summary/events` 等字段。

排查:
- 仅使用 `compare_perf` 产出的 schema 日志。
- 不要手工删改顶层字段: `schema_version/events/run_metadata/alignment/summary`。

#### F. Python API ValueError: threshold/scope/sync_mode

现象:
- `threshold_seconds must be > 0`
- `scope_name must be a non-empty string`
- `Invalid sync_mode: ... Expected one of [none, boundary]`

排查:
- 传入合法 `scope_name`。
- 设置 `threshold_seconds > 0`。
- `sync_mode` 仅用 `none` 或 `boundary`。
