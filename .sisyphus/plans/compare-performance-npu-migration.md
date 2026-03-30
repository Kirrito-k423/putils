# Compare Performance Tool for GPU→NPU Migration

## TL;DR
> **Summary**: 设计并落地一个低侵入性能对比工具：`with/装饰器` 接入、结构化日志落盘、双日志对比、Chrome Trace + 摘要输出，并用半自动映射解决模块命名不一致。  
> **Deliverables**: 采集 API、schema、对齐引擎、差异引擎、可视化导出、TDD 测试、文档。  
> **Effort**: Large  
> **Parallel**: YES - 3 waves  
> **Critical Path**: 1 → 2 → 4 → 6 → 8

## Context
### Original Request
- 在 GPU→NPU 迁移中，先做 compare-performance 工具，要求低膨胀、低侵入、可日志化、可视化对比，且处理模块名不一致。

### Interview Summary
- 验收：仅性能（不输出精度判定）。
- 接入：一个 `with` 或一个 decorator。
- 详细采集阈值：固定阈值，默认 `0.1s` 可配置。
- 输出：Chrome Trace + Top 差异摘要。
- 对齐：半自动（规则推断 + 人工确认 + 缓存）。
- 测试：TDD。

### Metis Review (gaps addressed)
- 增加硬约束：开销预算（默认 <3%）、schema 版本、低置信映射不自动合并。
- 明确口径：统一计时源、同步语义、inclusive/exclusive、rank 聚合策略。
- 明确边界：不扩展到精度工具、不做 full profiler 平台。
- 增加边界场景：空/损坏日志、动态控制流、融合算子、schema 不兼容。

## Work Objectives
### Core Objective
构建“可比、可追溯、可自动验证”的性能差异分析工具，最小化迁移代码改动并控制采集开销。

### Deliverables
- compare-performance 核心模块（采集、对齐、差异、导出）。
- 带时间 tag 的结构化日志（含 schema_version 与运行元信息）。
- 输出 `trace.json` + `summary.json/md`。
- TDD 测试矩阵与文档。

### Definition of Done (verifiable conditions with commands)
- 单点接入可用（with/decorator）。  
  Command: `pytest -v tests/test_compare_perf_integration.py::test_single_wrapper_integration`
- 轻量采集开销预算达标。  
  Command: `pytest -v tests/test_compare_perf_overhead.py::test_overhead_budget_under_3_percent`
- 双日志对比可产出 trace+summary 且含对齐状态统计。  
  Command: `pytest -v tests/test_compare_perf_diff.py`
- 对齐缓存二次运行命中。  
  Command: `pytest -v tests/test_compare_perf_alignment_cache.py::test_alignment_cache_hit_on_second_run`

### Must Have
- 低侵入 API（with + decorator）。
- 固定阈值详细采集（默认 0.1s）。
- GPU/NPU 后端中立元信息。
- matched/unmatched/ambiguous 显式输出。
- Chrome Trace 可读 + 摘要可读。

### Must NOT Have (guardrails, AI slop patterns, scope boundaries)
- 不做精度误差结论。
- 不依赖 full stack profiler 才能工作。
- 不静默吞并低置信映射。
- 不依赖人工后处理日志才可完成主流程。

## Verification Strategy
> ZERO HUMAN INTERVENTION — all verification is agent-executed.
- Test decision: **TDD** + `pytest`
- QA policy: 每个任务必须有 happy + failure/edge 场景
- Evidence: `.sisyphus/evidence/task-{N}-{slug}.{ext}`

## Execution Strategy
### Parallel Execution Waves
Wave 1: 契约与基础（1-3）  
Wave 2: 核心能力（4-7）  
Wave 3: 端到端与稳定性（8-10）

### Dependency Matrix (full, all tasks)
- 1 → 2,3,4
- 2 → 4,6
- 3 → 5,8
- 4 → 6,7
- 5 → 6,8
- 6 → 7,8,9
- 7 → 8,10
- 8 → 9,10

### Agent Dispatch Summary (wave → task count → categories)
- Wave 1 → 3 tasks → `unspecified-high`, `deep`, `quick`
- Wave 2 → 4 tasks → `unspecified-high`, `deep`, `quick`
- Wave 3 → 3 tasks → `unspecified-high`, `writing`

## TODOs
> Implementation + Test = ONE task. Never separate.
> EVERY task MUST have: Agent Profile + Parallelization + QA Scenarios.

- [x] 1. 定义日志 schema 与版本策略
  **What to do**: 统一 `events/run_metadata/alignment/summary` 结构；加入 `schema_version` 与不兼容报错规则。  
  **Must NOT do**: 不加入精度判定字段。
  **Recommended Agent Profile**: Category `unspecified-high`; Skills `[]`; Omitted `git-master`。  
  **Parallelization**: NO | Wave 1 | Blocks: 2,3,4 | Blocked By: none  
  **References**: `tools/python_stack_sniffer.py`, `putils/trace_manager.py`, `putils/write2file.py`  
  **Acceptance Criteria**:
  - [ ] `pytest -v tests/test_compare_perf_schema.py::test_schema_version_required`
  - [ ] `pytest -v tests/test_compare_perf_schema.py::test_incompatible_schema_rejected`
  **QA Scenarios**:
  ```
  Scenario: Happy path schema validation
    Tool: Bash
    Steps: 生成日志并执行 schema 测试
    Expected: schema 关键字段完整
    Evidence: .sisyphus/evidence/task-1-schema-validation.txt

  Scenario: Failure incompatible schema
    Tool: Bash
    Steps: 输入旧版本 schema 日志
    Expected: 非0退出并给出 incompatible schema
    Evidence: .sisyphus/evidence/task-1-schema-error.txt
  ```
  **Commit**: YES | Message: `feat(compare-perf): define schema contract and versioning` | Files: compare-perf core + schema tests

- [x] 2. 实现低开销计时核心
  **What to do**: 统一 `perf_counter_ns` 计时；实现 `none|boundary` 同步策略；记录 inclusive/exclusive。  
  **Must NOT do**: 不做每层强制 sync。
  **Recommended Agent Profile**: Category `deep`; Skills `[]`; Omitted `playwright`。  
  **Parallelization**: PARTIAL | Wave 1 | Blocks: 4,6 | Blocked By: 1  
  **References**: `putils/timer.py`, `putils/device.py`, `putils/perf.py`  
  **Acceptance Criteria**:
  - [ ] `pytest -v tests/test_compare_perf_overhead.py::test_overhead_budget_under_3_percent`
  - [ ] `pytest -v tests/test_compare_perf_timing.py::test_inclusive_exclusive_metrics`
  **QA Scenarios**:
  ```
  Scenario: Happy path overhead budget
    Tool: Bash
    Steps: baseline 与 instrumented 各跑5次
    Expected: overhead_pct < 3.0
    Evidence: .sisyphus/evidence/task-2-overhead.txt

  Scenario: Failure invalid sync mode
    Tool: Bash
    Steps: 传入非法 sync_mode
    Expected: 参数校验失败
    Evidence: .sisyphus/evidence/task-2-invalid-sync.txt
  ```
  **Commit**: YES | Message: `feat(compare-perf): add low-overhead timing contracts` | Files: timing core + overhead tests

- [x] 3. 实现配置与运行元信息采集
  **What to do**: 支持阈值默认 0.1s 覆盖、输出目录、rank 策略；日志文件名带 UTC 时间 tag。  
  **Must NOT do**: 不硬编码 `/tmp`。
  **Recommended Agent Profile**: Category `quick`; Skills `[]`; Omitted `oracle`。  
  **Parallelization**: YES | Wave 1 | Blocks: 5,8 | Blocked By: 1  
  **References**: `putils/timer.py`, `putils/trace_manager.py`, `putils/__init__.py`  
  **Acceptance Criteria**:
  - [ ] `pytest -v tests/test_compare_perf_config.py::test_default_and_override_threshold`
  - [ ] `pytest -v tests/test_compare_perf_config.py::test_timestamped_unique_log_filename`
  **QA Scenarios**:
  ```
  Scenario: Happy path config override
    Tool: Bash
    Steps: 默认阈值运行，再用0.2覆盖运行
    Expected: 两份日志反映不同阈值
    Evidence: .sisyphus/evidence/task-3-config.txt

  Scenario: Failure unwritable output dir
    Tool: Bash
    Steps: 指向无权限目录
    Expected: 快速失败并提示不可写
    Evidence: .sisyphus/evidence/task-3-output-error.txt
  ```
  **Commit**: YES | Message: `feat(compare-perf): add config and run metadata capture` | Files: config core + config tests

- [x] 4. 实现 with/decorator 低侵入采集入口
  **What to do**: 提供 `with compare_perf(...)` 与 `@compare_perf(...)`；阈值下仅记录聚合统计。  
  **Must NOT do**: 不要求业务层大改。
  **Recommended Agent Profile**: Category `unspecified-high`; Skills `[]`; Omitted `playwright`。  
  **Parallelization**: NO | Wave 2 | Blocks: 6,7 | Blocked By: 1,2  
  **References**: `putils/timer.py`, `putils/pprint.py`, `tests/test_timer.py`  
  **Acceptance Criteria**:
  - [ ] `pytest -v tests/test_compare_perf_integration.py::test_single_wrapper_integration`
  - [ ] `pytest -v tests/test_compare_perf_integration.py::test_threshold_skips_detail_but_keeps_summary`
  **QA Scenarios**:
  ```
  Scenario: Happy path wrapper instrumentation
    Tool: Bash
    Steps: 同时测试with与decorator
    Expected: summary中均有scope记录
    Evidence: .sisyphus/evidence/task-4-wrapper.txt

  Scenario: Failure invalid scope name
    Tool: Bash
    Steps: scope_name 传空
    Expected: 参数校验错误
    Evidence: .sisyphus/evidence/task-4-scope-error.txt
  ```
  **Commit**: YES | Message: `feat(compare-perf): add with/decorator collector API` | Files: collector API + integration tests

- [x] 5. 实现半自动对齐与映射缓存
  **What to do**: 规则候选（层号/token/调用序）+ 人工确认 + 缓存；输出 matched/unmatched/ambiguous。  
  **Must NOT do**: 低置信映射不得自动进入 matched。
  **Recommended Agent Profile**: Category `deep`; Skills `[]`; Omitted `git-master`。  
  **Parallelization**: YES | Wave 2 | Blocks: 6,8 | Blocked By: 1,3  
  **References**: `tools/python_stack_sniffer.py`, `putils/cache.py`, `tests/test_cache.py`  
  **Acceptance Criteria**:
  - [ ] `pytest -v tests/test_compare_perf_alignment.py::test_alignment_classification_counts`
  - [ ] `pytest -v tests/test_compare_perf_alignment_cache.py::test_alignment_cache_hit_on_second_run`
  **QA Scenarios**:
  ```
  Scenario: Happy path semi-auto alignment
    Tool: Bash
    Steps: 输入 attn vs attention 示例日志
    Expected: 生成候选并确认后缓存命中
    Evidence: .sisyphus/evidence/task-5-alignment.txt

  Scenario: Failure low-confidence mapping
    Tool: Bash
    Steps: 输入命名差异极大日志
    Expected: 标记 ambiguous，不进 matched
    Evidence: .sisyphus/evidence/task-5-ambiguous.txt
  ```
  **Commit**: YES | Message: `feat(compare-perf): add semi-auto alignment and cache` | Files: alignment core + cache tests

- [x] 6. 实现差异计算引擎
  **What to do**: 输出 `delta_ms/delta_pct`、调用次数差、可选 rank 聚合 p50/p95。  
  **Must NOT do**: unmatched 不参与伪差异；分母0需显式标记。
  **Recommended Agent Profile**: Category `unspecified-high`; Skills `[]`; Omitted `oracle`。  
  **Parallelization**: NO | Wave 2 | Blocks: 7,8,9 | Blocked By: 2,4,5  
  **References**: `tools/python_stack_sniffer.py`, `putils/perf.py`  
  **Acceptance Criteria**:
  - [ ] `pytest -v tests/test_compare_perf_diff.py::test_delta_abs_pct_and_zero_division`
  - [ ] `pytest -v tests/test_compare_perf_diff.py::test_rank_aggregation_percentiles`
  **QA Scenarios**:
  ```
  Scenario: Happy path diff computation
    Tool: Bash
    Steps: compare --top-n 20
    Expected: 输出delta_ms与delta_pct
    Evidence: .sisyphus/evidence/task-6-diff.txt

  Scenario: Failure zero baseline
    Tool: Bash
    Steps: baseline耗时为0的日志输入
    Expected: pct 标记 INF_OR_UNDEFINED
    Evidence: .sisyphus/evidence/task-6-zero-baseline.txt
  ```
  **Commit**: YES | Message: `feat(compare-perf): implement diff engine and rank aggregation` | Files: diff core + diff tests

- [x] 7. 实现可视化导出（Trace + 摘要）
  **What to do**: 生成 Chrome Trace JSON 与 `summary.md/json`，摘要含 top regression/improvement 与对齐统计。  
  **Must NOT do**: 不仅输出 trace 不输出摘要。
  **Recommended Agent Profile**: Category `quick`; Skills `[]`; Omitted `playwright`。  
  **Parallelization**: YES | Wave 2 | Blocks: 8,10 | Blocked By: 4,6  
  **References**: `tools/python_stack_sniffer.py`, `README.md`  
  **Acceptance Criteria**:
  - [ ] `pytest -v tests/test_compare_perf_trace_export.py::test_trace_json_is_valid`
  - [ ] `pytest -v tests/test_compare_perf_report.py::test_summary_contains_required_sections`
  **QA Scenarios**:
  ```
  Scenario: Happy path trace+summary
    Tool: Bash
    Steps: 执行 diff 并指定 --trace-out --summary-out
    Expected: 两文件生成且字段完整
    Evidence: .sisyphus/evidence/task-7-artifacts.txt

  Scenario: Failure malformed timeline
    Tool: Bash
    Steps: 输入缺失end_ts日志
    Expected: 导出失败并提示 malformed timeline
    Evidence: .sisyphus/evidence/task-7-malformed.txt
  ```
  **Commit**: YES | Message: `feat(compare-perf): export chrome trace and summary report` | Files: export/report core + tests

- [x] 8. 提供端到端 CLI 流程
  **What to do**: 提供 `collect`/`diff` CLI，串联采集→对比→导出并定义错误码。  
  **Must NOT do**: 不要求手工改 JSON。
  **Recommended Agent Profile**: Category `unspecified-high`; Skills `[]`; Omitted `oracle`。  
  **Parallelization**: NO | Wave 3 | Blocks: 9,10 | Blocked By: 3,5,6,7  
  **References**: `tools/python_stack_sniffer.py`, `pytest.ini`  
  **Acceptance Criteria**:
  - [ ] `pytest -v tests/test_compare_perf_cli.py::test_collect_diff_export_pipeline`
  - [ ] `pytest -v tests/test_compare_perf_cli.py::test_cli_error_codes_and_messages`
  **QA Scenarios**:
  ```
  Scenario: Happy path CLI e2e
    Tool: Bash
    Steps: collect baseline -> collect target -> diff export
    Expected: 退出码0并产出 trace+summary
    Evidence: .sisyphus/evidence/task-8-cli-e2e.txt

  Scenario: Failure missing input
    Tool: Bash
    Steps: 指定不存在 baseline 日志
    Expected: 非0并报 missing input log
    Evidence: .sisyphus/evidence/task-8-cli-missing.txt
  ```
  **Commit**: YES | Message: `feat(compare-perf): add end-to-end CLI pipeline and errors` | Files: cli core + cli tests

- [x] 9. 完成 TDD 测试矩阵与回归
  **What to do**: 先写失败用例，再实现通过；覆盖 schema、阈值、开销、对齐、diff、CLI、损坏日志、动态控制流。  
  **Must NOT do**: 不只测 happy path。
  **Recommended Agent Profile**: Category `unspecified-high`; Skills `[]`; Omitted `playwright`。  
  **Parallelization**: YES | Wave 3 | Blocks: 10 | Blocked By: 6,8  
  **References**: `tests/test_timer.py`, `tests/test_cache.py`, `tests/conftest.py`  
  **Acceptance Criteria**:
  - [ ] `pytest -v tests/test_compare_perf_*.py`
  - [ ] `pytest -v -m "slow" tests/test_compare_perf_*.py`
  **QA Scenarios**:
  ```
  Scenario: Happy path full matrix
    Tool: Bash
    Steps: 运行 compare_perf 全量测试 + slow 子集
    Expected: 全通过
    Evidence: .sisyphus/evidence/task-9-tests.txt

  Scenario: Failure corrupted log
    Tool: Bash
    Steps: 注入缺字段日志
    Expected: 捕获预期异常
    Evidence: .sisyphus/evidence/task-9-corrupted-log.txt
  ```
  **Commit**: YES | Message: `test(compare-perf): add tdd matrix and edge-case coverage` | Files: compare-perf tests

- [x] 10. 文档化接入与结果解读
  **What to do**: 补充 README 与模块文档：最小接入、参数、阈值解释、映射机制、常见错误排查。  
  **Must NOT do**: 不写与实现不一致命令。
  **Recommended Agent Profile**: Category `writing`; Skills `[]`; Omitted `oracle`。  
  **Parallelization**: YES | Wave 3 | Blocks: none | Blocked By: 7,8,9  
  **References**: `README.md`, `putils/timer.py`, `putils/profiling.py`  
  **Acceptance Criteria**:
  - [ ] `pytest -v tests/test_compare_perf_docs_smoke.py`
  - [ ] `pytest -v tests/test_compare_perf_docs_smoke.py::test_troubleshooting_sections_present`
  **QA Scenarios**:
  ```
  Scenario: Happy path docs smoke
    Tool: Bash
    Steps: 按文档命令执行 collect/diff/export
    Expected: 命令可执行且输出路径一致
    Evidence: .sisyphus/evidence/task-10-doc-smoke.txt

  Scenario: Failure outdated command
    Tool: Bash
    Steps: 将示例命令改为错误参数并跑 smoke
    Expected: smoke 失败并指向错误段落
    Evidence: .sisyphus/evidence/task-10-doc-error.txt
  ```
  **Commit**: YES | Message: `docs(compare-perf): add integration and troubleshooting guide` | Files: docs + docs smoke tests

## Final Verification Wave (MANDATORY — after ALL implementation tasks)
> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.
> **Do NOT auto-proceed after verification. Wait for user's explicit approval before marking work complete.**
> **Never mark F1-F4 as checked before getting user's okay.** Rejection or user feedback -> fix -> re-run -> present again -> wait for okay.
- [x] F1. Plan Compliance Audit — oracle
- [x] F2. Code Quality Review — unspecified-high
- [x] F3. Real Manual QA — unspecified-high (+ playwright if UI)
- [x] F4. Scope Fidelity Check — deep

## Commit Strategy
- `feat(compare-perf): add schema and timing contracts`
- `feat(compare-perf): implement collector alignment and diff`
- `feat(compare-perf): add trace export summary and cli`
- `test(compare-perf): add tdd matrix and edge cases`
- `docs(compare-perf): add integration and troubleshooting guide`

## Success Criteria
- 接入改动 ≤ 2 行（with 或 decorator）。
- 默认配置可输出可比较差异，trace 可在 Chrome Trace Viewer 打开。
- unmatched/ambiguous 明确展示，不静默吞并。
- 开销预算、schema 兼容、对齐缓存命中均可自动化验证。
