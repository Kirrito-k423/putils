# NPU-CPU Affinity Micro Benchmark Plan (VM Topology Mismatch)

## TL;DR
> **Summary**: Build a standalone micro-benchmark script to discover per-NPU most affine CPU lists under virtualized topology uncertainty, with deterministic protocol, retry/remeasure guardrails, and one-run full artifact export.
> **Deliverables**:
> - `tools/npu_affinity_benchmark.py` (standalone CLI)
> - TDD test suite for protocol/retry/export/CLI contracts
> - One-run output artifacts: structured log + raw JSON + multi-sheet Excel
> **Effort**: Large
> **Parallel**: YES - 4 waves
> **Critical Path**: 1 → 2 → 4 → 5 → 8 → 10

## Context
### Original Request
在虚拟机环境中排查 CPU→NPU 下发任务性能问题，怀疑虚拟化 CPU/NPU ID 与真实拓扑不一致导致绑核到错误 NUMA。需要 micro-benchmark 自动找出每个 NPU 最亲和的 CPU list，并满足：一次运行采全量数据写 log+Excel、实验可重复可复现、极端值超出历史区间±10%重测、绑核失败重试并分析。

### Interview Summary
- 交付形态：`tools/` 下独立脚本（优先落地）。
- 搜索策略：全量跨 NUMA 扫描。
- 载荷策略：内置固定载荷，默认“混合载荷（计算+搬运）”。
- 评分策略：双榜单（吞吐榜 + 延迟榜），不强行合并为单值。
- 可信性：当次动态基线（warmup + 首轮基线），超出 ±10% 自动重测。
- 导出策略：单个 xlsx，多 sheet（Summary/Raw/Best/Retry-Failures）。
- 测试策略：TDD。
- 默认执行模型：v1 为 **每个 NPU 1 个 worker**（候选在该 NPU 下顺序测量）。
- 默认指标口径：吞吐=iterations/s（median）；时延榜=latency p95(ms)（越低越好）。
- 默认重试上限：绑核失败最多 3 次；越界重测最多 2 次。
- 依赖策略：`openpyxl` 作为 xlsx 导出依赖；JSON 导出始终可用。

### Metis Review (gaps addressed)
- 强化“请求绑定 vs 生效绑定”双重记录，防止错误亲和性结论。
- 增加失败分类与退出码契约（`EPERM`/`EINVAL`/空有效集/离线CPU）。
- 定义 profile 分档（`smoke`/`medium`/`full`）避免全量扫描失控。
- 明确 machine-readable 产物（JSON）与 Excel 并存，避免只依赖人工审阅。

## Work Objectives
### Core Objective
在 VM 环境中，为每个可见 NPU 自动评估 CPU 候选列表，输出“吞吐最优 CPU list”和“时延最优 CPU list”，并提供可复现证据链（配置、环境、重试、重测、失败分类）。

### Deliverables
- 新增脚本：`tools/npu_affinity_benchmark.py`
- 新增测试：
  - `tests/test_npu_affinity_benchmark_protocol.py`
  - `tests/test_npu_affinity_benchmark_retry.py`
  - `tests/test_npu_affinity_benchmark_export.py`
  - `tests/test_npu_affinity_benchmark_cli.py`
- 运行产物目录（参数可配）：
  - `benchmark.log`
  - `raw_results.json`
  - `report.xlsx`

### Definition of Done (verifiable conditions with commands)
- `pytest -q tests/test_npu_affinity_benchmark_protocol.py`
- `pytest -q tests/test_npu_affinity_benchmark_retry.py`
- `pytest -q tests/test_npu_affinity_benchmark_export.py`
- `pytest -q tests/test_npu_affinity_benchmark_cli.py`
- `python tools/npu_affinity_benchmark.py --profile medium --output-dir .sisyphus/evidence/bench-run`
- `python tools/npu_affinity_benchmark.py --profile smoke --output-dir .sisyphus/evidence/bench-smoke --dry-run-topology`

### Must Have
- 必须读取并记录 effective 约束（affinity/cpuset/mems）后再评估。
- 每个候选必须经历：绑定请求 → 生效回读验证 → warmup/measure → 异常重测判定。
- 绑核失败必须按类别记录并有限重试（非无限循环）。
- 输出双榜单与失败分析，并含完整实验配置与环境快照。
- 绑定失败重试策略固定：最多 3 次（线性 backoff 100/200/300ms）。
- 越界重测策略固定：最多 2 次，超过则标记 `unstable`。

### Must NOT Have (guardrails, AI slop patterns, scope boundaries)
- 不做驱动/内核/虚拟化平台调优。
- 不改全局系统参数（governor/sysctl/irq 绑定）
- 不实现服务化/UI/dashboard。
- 不依赖人工“打开 Excel 目检”作为验收。

## Verification Strategy
> ZERO HUMAN INTERVENTION — all verification is agent-executed.
- Test decision: **TDD** + `pytest`
- QA policy: Every task has agent-executed scenarios
- Evidence: `.sisyphus/evidence/task-{N}-{slug}.{ext}`

## Execution Strategy
### Parallel Execution Waves
> Target: 5-8 tasks per wave. <3 per wave (except final) = under-splitting.
> Extract shared dependencies as Wave-1 tasks for max parallelism.

Wave 1: contracts/fixtures/topology schema foundations  
Wave 2: benchmark engine + bind verification + retry taxonomy  
Wave 3: dual leaderboard + export + CLI integration  
Wave 4: end-to-end hardening + docs/tests stabilization

### Dependency Matrix (full, all tasks)
- 1,2,3,4 are foundation tasks.
- 5 depends on 2,3; 6 depends on 2,3; 7 depends on 4,5.
- 8 depends on 5,6,7; 9 depends on 8.
- 10 depends on 8,9.
- 11 depends on 10; 12 depends on all 1-11.

### Agent Dispatch Summary (wave → task count → categories)
- Wave 1 → 4 tasks → `quick`, `unspecified-low`
- Wave 2 → 3 tasks → `deep`, `unspecified-high`
- Wave 3 → 3 tasks → `unspecified-high`, `writing`
- Wave 4 → 2 tasks → `deep`, `unspecified-high`

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task MUST have: Agent Profile + Parallelization + QA Scenarios.

- [x] 1. 建立协议与结果数据契约（TDD 起点）

  **What to do**: 定义 benchmark 的内部数据结构与输出 schema（run metadata、topology、candidate、sample、remeasure_event、failure_event、leaderboards）；先写失败测试锁定字段与类型，再最小实现通过。  
  **Must NOT do**: 不引入与业务模型耦合字段；不把 schema 逻辑散落在多处。

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: 单文件契约与测试优先。
  - Skills: `[]` — 无额外技能依赖。
  - Omitted: `['playwright']` — 非浏览器任务。

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: [5,7,8,9] | Blocked By: []

  **References**:
  - Pattern: `putils/compare_perf/schema.py` — 结构化 schema 组织方式。
  - Pattern: `putils/compare_perf/collector.py` — timing record 的字段风格。
  - Test: `tests/test_compare_perf_snapshot.py` — 产物字段断言模式。

  **Acceptance Criteria**:
  - [ ] `pytest -q tests/test_npu_affinity_benchmark_protocol.py -k schema`
  - [ ] `python -m py_compile tools/npu_affinity_benchmark.py`

  **QA Scenarios**:
  ```
  Scenario: Schema happy path
    Tool: Bash
    Steps: 运行 pytest schema 用例，加载最小样例并序列化到 JSON
    Expected: 关键字段完整且类型匹配，测试通过
    Evidence: .sisyphus/evidence/task-1-schema.txt

  Scenario: Schema failure path
    Tool: Bash
    Steps: 构造缺失 requested_affinity/effective_affinity 字段的样例
    Expected: 校验失败并返回明确错误信息
    Evidence: .sisyphus/evidence/task-1-schema-error.txt
  ```

  **Commit**: YES | Message: `test(affinity-bench): define data contracts and schema` | Files: `tools/npu_affinity_benchmark.py`, `tests/test_npu_affinity_benchmark_protocol.py`

- [x] 2. 实现拓扑与 effective 约束发现

  **What to do**: 实现运行时拓扑采集：在线 CPU、NUMA 节点映射、当前进程 affinity、cpuset/cgroup effective cpu/mem；形成统一“可用域”对象。  
  **Must NOT do**: 不假设 vCPU id 连续；不直接信任请求 mask。

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: procfs/cgroup/sysfs 多源融合。
  - Skills: `[]`
  - Omitted: `['playwright']`

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: [3,5,10] | Blocked By: []

  **References**:
  - Pattern: `putils/device.py` — 环境能力探测风格。
  - Pattern: `putils/compare_perf/config.py` — run metadata 采集。
  - External: `https://man7.org/linux/man-pages/man2/sched_setaffinity.2.html` — affinity 语义。
  - External: `https://man7.org/linux/man-pages/man7/cpuset.7.html` — cpuset 约束。

  **Acceptance Criteria**:
  - [ ] `pytest -q tests/test_npu_affinity_benchmark_protocol.py -k topology`
  - [ ] `python tools/npu_affinity_benchmark.py --profile smoke --dry-run-topology --output-dir .sisyphus/evidence/task-2-topology`

  **QA Scenarios**:
  ```
  Scenario: Topology discovery happy path
    Tool: Bash
    Steps: 运行 dry-run-topology，读取生成的 raw_results.json
    Expected: topology 中包含 online_cpus/effective_cpus/effective_mems/nodes
    Evidence: .sisyphus/evidence/task-2-topology.json

  Scenario: Empty effective set edge case
    Tool: Bash
    Steps: 单测模拟 effective cpu 集为空
    Expected: 返回可分类失败码而非崩溃
    Evidence: .sisyphus/evidence/task-2-topology-error.txt
  ```

  **Commit**: YES | Message: `feat(affinity-bench): add topology and effective-set discovery` | Files: `tools/npu_affinity_benchmark.py`, `tests/test_npu_affinity_benchmark_protocol.py`

- [x] 3. 生成跨 NUMA 候选 CPU lists（含 profile 限流）

  **What to do**: 基于有效域生成候选 CPU list 集合，支持全量跨 NUMA 扫描；提供 `smoke/medium/full` 三档候选上限与采样策略。  
  **Must NOT do**: 不穷举不可控组合；不输出重复候选。

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: 组合生成与去重规则复杂。
  - Skills: `[]`
  - Omitted: `['playwright']`

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: [6,7] | Blocked By: [2]

  **References**:
  - Pattern: `putils/perf.py` — affinity 列表分配思路。
  - Test: `tests/test_compare_perf_integration.py` — 配置驱动参数化测试模式。

  **Acceptance Criteria**:
  - [ ] `pytest -q tests/test_npu_affinity_benchmark_protocol.py -k candidates`
  - [ ] `python tools/npu_affinity_benchmark.py --profile medium --dry-run-topology --output-dir .sisyphus/evidence/task-3-candidates`

  **QA Scenarios**:
  ```
  Scenario: Candidate generation happy path
    Tool: Bash
    Steps: 在 mocked 两节点拓扑下生成候选
    Expected: 候选非空、无重复、含跨 NUMA 条目
    Evidence: .sisyphus/evidence/task-3-candidates.txt

  Scenario: Single-node edge case
    Tool: Bash
    Steps: 在单 NUMA 拓扑下运行生成
    Expected: 仅产生单域候选且流程成功
    Evidence: .sisyphus/evidence/task-3-candidates-error.txt
  ```

  **Commit**: YES | Message: `feat(affinity-bench): add candidate generation profiles` | Files: `tools/npu_affinity_benchmark.py`, `tests/test_npu_affinity_benchmark_protocol.py`

- [x] 4. 测试夹具与故障注入基座

  **What to do**: 构建可复现测试夹具：mock affinity 设置/读取、mock procfs/cgroup、mock NPU id 列表、可控计时源，支持失败注入（`EPERM`/`EINVAL`/超时）。  
  **Must NOT do**: 不依赖真实 NPU/特权环境执行单元测试。

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: 测试基础设施搭建。
  - Skills: `[]`
  - Omitted: `['playwright']`

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: [5,6,7,8,9] | Blocked By: []

  **References**:
  - Pattern: `tests/conftest.py` — fixture 组织方式。
  - Pattern: `tests/test_compare_perf_timing.py` — monkeypatch 可控时间。
  - Pattern: `tests/test_device.py` — skipif/importorskip 用法。

  **Acceptance Criteria**:
  - [ ] `pytest -q tests/test_npu_affinity_benchmark_retry.py -k fixture`
  - [ ] `pytest -q tests/test_npu_affinity_benchmark_protocol.py -k fixture`

  **QA Scenarios**:
  ```
  Scenario: Fixture determinism
    Tool: Bash
    Steps: 同一随机种子重复运行同一测试两次
    Expected: 两次候选与样本序列一致
    Evidence: .sisyphus/evidence/task-4-fixture.txt

  Scenario: Injected EPERM
    Tool: Bash
    Steps: mock affinity setter 抛 EPERM
    Expected: 测试捕获并归类失败码
    Evidence: .sisyphus/evidence/task-4-fixture-error.txt
  ```

  **Commit**: YES | Message: `test(affinity-bench): add deterministic fixtures and fault injection` | Files: `tests/test_npu_affinity_benchmark_*.py`

- [x] 5. 实现绑核执行与生效验证链路

  **What to do**: 实现绑定流程：请求 affinity → 执行设置 → 回读 effective affinity/mems → 记录 request/effective 差异；失败时产出标准化失败记录。  
  **Must NOT do**: 不仅记录请求值；不忽略回读不一致。

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: 需要系统调用语义与失败路径完整性。
  - Skills: `[]`
  - Omitted: `['playwright']`

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: [7,8,10] | Blocked By: [1,2,4]

  **References**:
  - Pattern: `putils/perf.py` — `cpu_affinity` 调用方式。
  - Pattern: `putils/compare_perf/cli.py` — 错误分类与码。
  - External: `https://man7.org/linux/man-pages/man2/sched_setaffinity.2.html`

  **Acceptance Criteria**:
  - [ ] `pytest -q tests/test_npu_affinity_benchmark_retry.py -k bind_verify`
  - [ ] `pytest -q tests/test_npu_affinity_benchmark_cli.py -k failure_code`

  **QA Scenarios**:
  ```
  Scenario: Bind verify happy path
    Tool: Bash
    Steps: 使用 mock 可用 mask 运行 bind+verify
    Expected: requested_affinity 与 effective_affinity 按预期记录
    Evidence: .sisyphus/evidence/task-5-bind.txt

  Scenario: Effective mismatch
    Tool: Bash
    Steps: 模拟内核收窄 mask
    Expected: 标记 mismatch 且该样本不参与排名
    Evidence: .sisyphus/evidence/task-5-bind-error.txt
  ```

  **Commit**: YES | Message: `feat(affinity-bench): implement bind and effective verification` | Files: `tools/npu_affinity_benchmark.py`, `tests/test_npu_affinity_benchmark_retry.py`, `tests/test_npu_affinity_benchmark_cli.py`

- [x] 6. 实现可复现混合载荷执行器

  **What to do**: 实现内置混合载荷（计算+搬运），固定 seed 与参数；支持 profile 参数映射到输入规模与迭代次数；提供无 torch 保护。  
  **Must NOT do**: 不依赖外部业务命令；不在缺少 torch 时崩溃。

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: 载荷实现 + 兼容性处理。
  - Skills: `[]`
  - Omitted: `['playwright']`

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: [7,8] | Blocked By: [3,4]

  **References**:
  - Pattern: `putils/device.py` — 可选依赖判定。
  - Pattern: `tests/test_cache.py` — `skipif(not TORCH_AVAILABLE)` 样式。

  **Acceptance Criteria**:
  - [ ] `pytest -q tests/test_npu_affinity_benchmark_protocol.py -k workload`
  - [ ] `python tools/npu_affinity_benchmark.py --profile smoke --output-dir .sisyphus/evidence/task-6-workload --dry-run-topology`

  **QA Scenarios**:
  ```
  Scenario: Workload determinism
    Tool: Bash
    Steps: 同参数运行两次 workload 样本生成
    Expected: 迭代配置与预期时间窗口一致
    Evidence: .sisyphus/evidence/task-6-workload.txt

  Scenario: Torch unavailable
    Tool: Bash
    Steps: mock 无 torch 环境运行路径
    Expected: 返回可解释错误或降级路径，不崩溃
    Evidence: .sisyphus/evidence/task-6-workload-error.txt
  ```

  **Commit**: YES | Message: `feat(affinity-bench): add deterministic mixed workload` | Files: `tools/npu_affinity_benchmark.py`, `tests/test_npu_affinity_benchmark_protocol.py`

- [ ] 7. 实现基线区间与 ±10% 自动重测

  **What to do**: 实现 warmup + 首轮基线建立；若吞吐或时延任一指标越过基线±10%则触发重测；记录重测次数与结论（accepted/unstable）。  
  **Must NOT do**: 不做无限重测；不只看单指标。

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: 统计口径与状态机判断。
  - Skills: `[]`
  - Omitted: `['playwright']`

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: [8,9,10] | Blocked By: [1,4,5,6]

  **References**:
  - Pattern: `putils/compare_perf/collector.py` — 精细 timing 采集。
  - Test: `tests/test_compare_perf_overhead.py` — 阈值断言风格。

  **Acceptance Criteria**:
  - [ ] `pytest -q tests/test_npu_affinity_benchmark_retry.py -k remeasure`
  - [ ] `pytest -q tests/test_npu_affinity_benchmark_protocol.py -k baseline`

  **QA Scenarios**:
  ```
  Scenario: 12% regression triggers remeasure
    Tool: Bash
    Steps: 注入吞吐下降 12% 样本
    Expected: remeasure_count 增加且有 remeasure_audit 记录
    Evidence: .sisyphus/evidence/task-7-remeasure.txt

  Scenario: 8% drift no remeasure
    Tool: Bash
    Steps: 注入波动 8% 样本
    Expected: 不触发重测，样本正常入榜
    Evidence: .sisyphus/evidence/task-7-remeasure-error.txt
  ```

  **Commit**: YES | Message: `feat(affinity-bench): implement baseline window and auto remeasure` | Files: `tools/npu_affinity_benchmark.py`, `tests/test_npu_affinity_benchmark_retry.py`, `tests/test_npu_affinity_benchmark_protocol.py`

- [ ] 8. 生成双榜单与稳定性判定

  **What to do**: 输出 per-NPU 吞吐榜与时延榜（median 主排序），并附离散度指标（MAD/IQR）；当最优与次优无显著差距时标记不确定性。  
  **Must NOT do**: 不把无效样本（验证失败/异常未决）入榜。

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: 聚合规则与统计输出。
  - Skills: `[]`
  - Omitted: `['playwright']`

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: [9,10,11] | Blocked By: [1,5,6,7]

  **References**:
  - Pattern: `putils/compare_perf/export.py` — summary 报告组织。
  - Test: `tests/test_compare_perf_integration.py` — 聚合结果校验思路。

  **Acceptance Criteria**:
  - [ ] `pytest -q tests/test_npu_affinity_benchmark_protocol.py -k leaderboard`
  - [ ] `python tools/npu_affinity_benchmark.py --profile medium --output-dir .sisyphus/evidence/task-8-leaderboard --dry-run-topology`

  **QA Scenarios**:
  ```
  Scenario: Dual leaderboard happy path
    Tool: Bash
    Steps: 运行 mocked benchmark 样本聚合
    Expected: throughput/latency 两榜均有每 NPU 最优项
    Evidence: .sisyphus/evidence/task-8-leaderboard.json

  Scenario: Unstable tie edge
    Tool: Bash
    Steps: 构造最优与次优高度重叠离散度
    Expected: 标记 uncertain=true
    Evidence: .sisyphus/evidence/task-8-leaderboard-error.txt
  ```

  **Commit**: YES | Message: `feat(affinity-bench): add dual leaderboard and stability flags` | Files: `tools/npu_affinity_benchmark.py`, `tests/test_npu_affinity_benchmark_protocol.py`

- [ ] 9. 实现 JSON + Excel 多 sheet 导出

  **What to do**: 导出 `raw_results.json` 与 `report.xlsx`；xlsx 必含 `run_metadata/topology/candidates/results/leaderboard_throughput/leaderboard_latency/failures/remeasure_audit`。  
  **Must NOT do**: 不只写 Excel；不遗漏失败与重测审计表。

  **Recommended Agent Profile**:
  - Category: `unspecified-high` — Reason: 多格式导出一致性。
  - Skills: `[]`
  - Omitted: `['playwright']`

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: [11,12] | Blocked By: [1,7,8]

  **References**:
  - Pattern: `putils/compare_perf/export.py` — 结构化导出。
  - Pattern: `examples/compare_hooks/parse_hooks_to_excel.py` — openpyxl sheet 写入风格。
  - Pattern: `tools/python_stack_sniffer.py` — 原子文件写入。

  **Acceptance Criteria**:
  - [ ] `pytest -q tests/test_npu_affinity_benchmark_export.py`
  - [ ] `python tools/npu_affinity_benchmark.py --profile smoke --output-dir .sisyphus/evidence/task-9-export`

  **QA Scenarios**:
  ```
  Scenario: Export happy path
    Tool: Bash
    Steps: 运行 smoke profile 并检查输出目录
    Expected: report.xlsx 和 raw_results.json 同时存在且可解析
    Evidence: .sisyphus/evidence/task-9-export.txt

  Scenario: Partial write failure
    Tool: Bash
    Steps: mock 写入异常触发临时文件路径
    Expected: 无损坏目标文件，错误被记录分类
    Evidence: .sisyphus/evidence/task-9-export-error.txt
  ```

  **Commit**: YES | Message: `feat(affinity-bench): add json and xlsx exporters` | Files: `tools/npu_affinity_benchmark.py`, `tests/test_npu_affinity_benchmark_export.py`

- [ ] 10. 构建 CLI、日志与退出码契约

  **What to do**: 实现 argparse CLI（profile/output-dir/seed/retry-limit/dry-run-topology 等），输出结构化日志，定义退出码（成功/部分失败/致命失败）。  
  **Must NOT do**: 不返回模糊错误；不把配置隐式硬编码。

  **Recommended Agent Profile**:
  - Category: `quick` — Reason: CLI 契约与参数校验清晰。
  - Skills: `[]`
  - Omitted: `['playwright']`

  **Parallelization**: Can Parallel: YES | Wave 3 | Blocks: [11,12] | Blocked By: [2,5,7,8]

  **References**:
  - Pattern: `putils/compare_perf/cli.py` — `CliError` + exit code 风格。
  - Pattern: `tools/python_stack_sniffer.py` — argparse/logging/atomic save。
  - Test: `tests/test_compare_perf_cli.py` — CLI 错误路径断言。

  **Acceptance Criteria**:
  - [ ] `pytest -q tests/test_npu_affinity_benchmark_cli.py`
  - [ ] `python tools/npu_affinity_benchmark.py --help`

  **QA Scenarios**:
  ```
  Scenario: CLI happy path
    Tool: Bash
    Steps: 使用 smoke profile 启动并写到临时输出目录
    Expected: 退出码 0，日志含 run_id/profile/seed
    Evidence: .sisyphus/evidence/task-10-cli.txt

  Scenario: Invalid argument
    Tool: Bash
    Steps: 传入非法 profile 值
    Expected: 非零退出码且错误信息可读
    Evidence: .sisyphus/evidence/task-10-cli-error.txt
  ```

  **Commit**: YES | Message: `feat(affinity-bench): add cli contract logging and exit codes` | Files: `tools/npu_affinity_benchmark.py`, `tests/test_npu_affinity_benchmark_cli.py`

- [ ] 11. 端到端流程验证与失败分析覆盖

  **What to do**: 增加 e2e 测试覆盖完整流程（拓扑→候选→绑定→测量→重测→导出），并验证 failures sheet 字段：`errno/requested_affinity/effective_affinity/retry_count/final_status`。  
  **Must NOT do**: 不使用人工打开 Excel 验证；全部程序断言。

  **Recommended Agent Profile**:
  - Category: `deep` — Reason: 跨模块 e2e 验证。
  - Skills: `[]`
  - Omitted: `['playwright']`

  **Parallelization**: Can Parallel: YES | Wave 4 | Blocks: [12] | Blocked By: [8,9,10]

  **References**:
  - Test: `tests/test_compare_perf_torch_e2e_example.py` — e2e 框架模式。
  - Test: `tests/test_compare_perf_overhead.py` — 阈值型断言。

  **Acceptance Criteria**:
  - [ ] `pytest -q tests/test_npu_affinity_benchmark_* -k "e2e or failure"`
  - [ ] `python tools/npu_affinity_benchmark.py --profile medium --output-dir .sisyphus/evidence/task-11-e2e`

  **QA Scenarios**:
  ```
  Scenario: End-to-end happy path
    Tool: Bash
    Steps: 执行 medium profile 端到端
    Expected: 生成完整 artifacts，双榜单非空
    Evidence: .sisyphus/evidence/task-11-e2e.txt

  Scenario: Bind failure analysis
    Tool: Bash
    Steps: 注入 EPERM/EINVAL 失败样本
    Expected: failures 与 remeasure_audit 均含分类与最终状态
    Evidence: .sisyphus/evidence/task-11-e2e-error.txt
  ```

  **Commit**: YES | Message: `test(affinity-bench): add e2e and failure-analysis coverage` | Files: `tests/test_npu_affinity_benchmark_*.py`

- [ ] 12. 收敛与仓库规范对齐

  **What to do**: 对齐仓库测试 marker、命令和文档注释（脚本内 docstring + usage）；确保在无 NPU 环境下也能通过 dry-run/topology tests；补齐可重复执行说明。  
  **Must NOT do**: 不新增与目标无关的工具链改造。

  **Recommended Agent Profile**:
  - Category: `writing` — Reason: 规范化说明 + 轻量整合。
  - Skills: `[]`
  - Omitted: `['playwright']`

  **Parallelization**: Can Parallel: NO | Wave 4 | Blocks: [] | Blocked By: [9,10,11]

  **References**:
  - Pattern: `AGENTS.md` — 测试命令与 marker 约定。
  - Pattern: `README.md` — 文档风格。

  **Acceptance Criteria**:
  - [ ] `pytest -q tests/test_npu_affinity_benchmark_*`
  - [ ] `pytest -q -m "not slow" tests/test_npu_affinity_benchmark_*`
  - [ ] `python tools/npu_affinity_benchmark.py --help`

  **QA Scenarios**:
  ```
  Scenario: Repo-aligned test run
    Tool: Bash
    Steps: 执行完整与 not-slow 两套命令
    Expected: 均成功，且 slow 用例可被 marker 控制
    Evidence: .sisyphus/evidence/task-12-stabilize.txt

  Scenario: No-NPU fallback
    Tool: Bash
    Steps: 在无 NPU 条件下执行 dry-run-topology
    Expected: 输出可解释降级信息并成功退出
    Evidence: .sisyphus/evidence/task-12-stabilize-error.txt
  ```

  **Commit**: YES | Message: `chore(affinity-bench): align docs markers and fallback behavior` | Files: `tools/npu_affinity_benchmark.py`, `tests/test_npu_affinity_benchmark_*.py`, `README.md`

## Final Verification Wave (MANDATORY — after ALL implementation tasks)
> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
> **Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.**
> **Never mark F1-F4 as checked before getting user's okay.** Rejection or user feedback -> fix -> re-run -> present again -> wait for okay.
- [ ] F1. Plan Compliance Audit — oracle
- [ ] F2. Code Quality Review — unspecified-high
- [ ] F3. Real Manual QA — unspecified-high (+ playwright if UI)
- [ ] F4. Scope Fidelity Check — deep

## Commit Strategy
- Commit 1: `test(affinity-bench): define protocol and schema contracts (RED)`
- Commit 2: `feat(affinity-bench): add topology discovery and effective-mask verification`
- Commit 3: `feat(affinity-bench): implement benchmark engine and dual leaderboards`
- Commit 4: `feat(affinity-bench): add retry/remeasure and failure taxonomy`
- Commit 5: `feat(affinity-bench): add json/xlsx export and cli interface`
- Commit 6: `test(affinity-bench): finalize e2e and edge-case stabilization`

## Success Criteria
- 每个 NPU 均产出吞吐榜和时延榜最优 CPU list（或明确不可判定原因）。
- 结果可重放：同 profile + 同 seed + 同候选空间时，结果波动在定义离散度阈值内。
- 出现 ±10% 越界时触发重测并在审计 sheet 有可追踪记录。
- bind 失败有分类、重试轨迹与最终结论，且不会中断全任务。
