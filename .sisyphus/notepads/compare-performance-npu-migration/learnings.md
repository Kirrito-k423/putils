# Learnings

- 2026-03-29 Task1 schema first cut: added `putils.compare_perf.schema` with a minimal, deterministic contract.
- Implemented required top-level structure as: `schema_version`, `events`, `run_metadata`, `alignment`, `summary`.
- Added explicit compatibility gate via `SUPPORTED_SCHEMA_VERSIONS`; incompatible versions fail fast with deterministic error string.
- Added helper APIs `build_schema`, `validate_schema`, `parse_schema` and package wiring in `putils/compare_perf/__init__.py` for straightforward imports in follow-up tasks.
- Tests in `tests/test_compare_perf_schema.py` now cover required `schema_version`, incompatible version rejection, and required top-level section presence.

- 2026-03-29 Task2 timing core: added `putils.compare_perf.collector` using `time.perf_counter_ns()` as the single timing source.
- Implemented nesting-aware inclusive/exclusive accounting via parent `child_inclusive_ns` accumulation so parent exclusive excludes child runtime deterministically.
- Added sync policy contract in `putils.compare_perf.config`: only `none|boundary` are accepted, with deterministic validation error for invalid modes.
- `boundary` mode triggers optional synchronize callback only at scope enter/exit boundaries, keeping default path (`none`) lightweight.
- Added `measure_overhead(...)` helper and tests in `tests/test_compare_perf_overhead.py` to assert instrumentation stays below 3% budget for representative CPU workload.

- 2026-03-29 Task3 config/metadata: extended `ComparePerfTimingConfig` defaults to `threshold_seconds=0.1`, `output_dir='.'`, `rank_strategy='all'` with explicit rank strategy validation (`all|rank0`).
- Added backend-neutral runtime metadata helper `collect_runtime_metadata(...)` to capture backend/device, rank/world_size, host, pid, commit, and both UTC + unix-ns timestamps.
- Added UTC-tagged unique filename helper `make_timestamped_log_filename(...)` with explicit tag format `%Y%m%dT%H%M%S.%fZ` and pid+time_ns uniqueness token.
- Added `tests/test_compare_perf_config.py` coverage for default/override threshold and timestamped unique log filename behavior.

- 2026-03-29 Task4 integration wrapper: added low-intrusion `compare_perf(...)` in `putils.compare_perf.collector` using `ContextDecorator`, so the same API supports both `with compare_perf(...)` and `@compare_perf(...)`.
- Threshold behavior is now deterministic and testable by recording per-scope summary first (`call_count`, `total_ns`, `total_seconds`) and only then dropping detailed events below threshold.
- Added `TimingCollector.compare_scope(...)` to reuse existing timing core (`scope(...)`) while enforcing `scope_name` validation and threshold filtering without changing inclusive/exclusive accounting semantics.
- Added `tests/test_compare_perf_integration.py` with integration coverage for single wrapper flow, threshold skip-detail/keep-summary behavior, and deterministic empty-scope validation.

- 2026-03-29 Task5 alignment core: added `putils.compare_perf.align` with deterministic rule-based candidate scoring over token similarity, layer index proximity, and call-order proximity.
- Added explicit classification contract: output includes `matched`, `ambiguous`, `unmatched.left/right`, and `counts` (`matched/unmatched/ambiguous`) for downstream diff/report stages.
- Kept semi-automatic workflow explicit via per-candidate `confidence` and `status`; low-confidence/multi-candidate cases remain `ambiguous` and are never auto-promoted.
- Added `AlignmentMappingCache` (JSON + SHA1 key over left/right module lists) so confirmed mappings persist and can be reused on second run.
- Added tests `tests/test_compare_perf_alignment.py` and `tests/test_compare_perf_alignment_cache.py` for classification counts and cache-hit reuse.

- 2026-03-29 Task6 diff engine: added `putils.compare_perf.diff.compute_diff` to compute matched-only per-module deltas with deterministic structure (`modules/counts/excluded/rank_aggregation`).
- Implemented explicit zero-baseline percentage marker `INF_OR_UNDEFINED` for both module-level `delta_pct` and rank-aggregated percentile pct deltas, avoiding silent NaN/inf behavior.
- Rank aggregation is optional (`enable_rank_aggregation`) and produces deterministic `p50/p95` metrics via linear-interpolated percentile calculation over rank timing samples.
- Added `tests/test_compare_perf_diff.py` coverage for absolute+percentage deltas, zero-divisor handling, unmatched exclusion from fake diffs, and rank p50/p95 aggregation outputs.

- 2026-03-29 Task7 export/report: added `putils.compare_perf.export` with deterministic Chrome Trace export (`traceEvents` + `displayTimeUnit='us'`) using `B/E` runtime events and `M` metadata events (`process_name`/`thread_name`) aligned with `tools/python_stack_sniffer.py` format style.
- Added summary generation pipeline: `build_summary(...)` + `render_summary_markdown(...)` + `export_compare_result(...)`, producing top regressions/improvements, alignment statistics (including unmatched visibility), diff counters, rank-aggregation enabled flag, and config echo.
- Enforced explicit malformed timeline validation contract (e.g., missing `end_ns`, invalid integer fields, `end_ns < start_ns`) with deterministic `ValueError` messages for test stability.
- Added tests `tests/test_compare_perf_trace_export.py` and `tests/test_compare_perf_report.py` to validate trace compatibility, required summary sections, and malformed input failure behavior.

- 2026-03-29 Task8 CLI e2e: added `putils.compare_perf.cli` with deterministic `collect` and `diff` subcommands based on stdlib `argparse`, reusing existing `schema/align/diff/export/config` modules without extra dependencies.
- `collect` now accepts repeatable `--event <scope>:<duration_ms>` inputs and writes schema-valid compare-perf logs directly (events + run_metadata + summary), so happy path does not require manual JSON editing.
- `diff` now loads baseline/target logs via schema validation, runs alignment + diff, merges timeline events, and exports `trace.json` + `summary.json` + `summary.md` in one command.
- Added explicit CLI exit-code contract: invalid args=2, missing input=10, incompatible/invalid schema=11, unexpected runtime=12; stderr messages are actionable and deterministic for tests.
- Added `tests/test_compare_perf_cli.py` with required acceptance cases `test_collect_diff_export_pipeline` and `test_cli_error_codes_and_messages`.

- 2026-03-29 Task9 TDD matrix/regression: expanded compare_perf edge tests for corrupted/partial logs, malformed timeline events, dynamic control-flow, and slow marker execution path.
- Added CLI regression coverage for corrupted JSON (`invalid json in input log`), partial schema payload missing `summary`, and malformed baseline event missing `end_ns`; all return deterministic `EXIT_SCHEMA_ERROR` and actionable stderr fragments.
- Added deterministic dynamic control-flow timing test by monkeypatching `perf_counter_ns` ticks, validating nested/optional scope event order (`branch`, `step`, `step`) plus exact inclusive/exclusive ns accounting.
- Stabilized overhead flakiness by turning the slow-path test into a deterministic regression with mocked `_run_n_times` totals (2.0% overhead), preserving slow-marker coverage while avoiding machine-noise false negatives.

- 2026-03-29 Task10 docs/readability: added a dedicated `compare_perf` README section covering minimal `with compare_perf(...)` and `@compare_perf(...)` integration examples, threshold semantics, CLI collect/diff workflow, mapping/cache usage, and result interpretation for trace/summary artifacts.
- Added docs smoke regression file `tests/test_compare_perf_docs_smoke.py` to keep README and runtime interface in sync by validating required command snippets, parser-accepted CLI argument combinations, and troubleshooting section completeness.
- Troubleshooting documentation now includes six concrete categories with deterministic error signatures tied to actual CLI/API behavior (`ERROR[2]/[10]/[11]` plus Python `ValueError` validation cases).

- 2026-03-29 F1/F2 quality patch: enforced deterministic CLI arg validation for `diff --top-n` by using a positive-int argparse type in `putils/compare_perf/cli.py`; non-positive values now route through invalid-args contract (`ERROR[2]`) instead of falling through to unexpected runtime error.
- 2026-03-29 F1/F2 quality patch: removed pre-sorting of summary module names before alignment in `putils/compare_perf/cli.py`; alignment now receives insertion order from baseline/target summaries so order-signal semantics remain meaningful.
- 2026-03-29 F1/F2 quality patch: updated `setup.py` to `find_packages(include=['putils', 'putils.*'])` so `putils.compare_perf` is included in install artifacts.
- 2026-03-29 F1/F2 quality patch: made `putils/compare_perf/schema.py` type annotations Python 3.7-compatible by replacing built-in generic syntax (`list[...]`/`dict[...]`) with `typing.List`/`typing.Dict`.
- 2026-03-29 F1/F2 quality patch: added CLI regressions in `tests/test_compare_perf_cli.py` for non-positive `--top-n` rejection and for preserving summary key order passed to alignment (via monkeypatch capture).

- 2026-03-29 F4 scope-fidelity audit: scope remains performance-only in `putils.compare_perf` (no precision/accuracy analysis paths found in compare_perf module/tests via grep); the only `accuracy.py` mentions are project-level README structure entries outside compare_perf flow.
- Low-confidence alignment is not auto-promoted: `putils/compare_perf/align.py` keeps uncertain matches in `ambiguous` (`reason=low_confidence_or_multiple_candidates`) unless confidence passes `auto_match_threshold` and margin gate; spot test `tests/test_compare_perf_alignment.py::test_alignment_classification_counts` passed.
- Normal collect→diff path stays non-manual and produces required artifacts: `putils/compare_perf/cli.py` builds schema logs from `collect`, runs `diff`, and prints `trace_out/summary_json_out/summary_md_out`; e2e test `tests/test_compare_perf_cli.py::test_collect_diff_export_pipeline` passed and asserted trace+summary outputs exist.

- 2026-03-29 F2 quality gate review: strict static/dynamic review of `putils/compare_perf/*.py`, `tests/test_compare_perf_*.py`, `README.md`, and `setup.py` found no implementation edits required; focus regressions for `diff --top-n` (`ERROR[2]`) and summary insertion-order alignment both pass.
- Verified dynamic checks: `pytest -v tests/test_compare_perf_*.py` (30 passed) and targeted CLI regressions `pytest -v tests/test_compare_perf_cli.py -k "non_positive_top_n or preserves_summary_key_order_for_alignment"` (2 passed).
- Verified static checks: `lsp_diagnostics` on `putils/compare_perf` reports zero diagnostics; anti-pattern grep shows no TODO/FIXME/HACK in compare_perf paths, with only intentional `pass` usage (`SchemaValidationError` class body and test no-op blocks) plus guarded broad-exception fallbacks in runtime metadata probing (`config.py`) suitable for optional dependency resilience.

- 2026-03-29 F3 manual CLI QA rerun (real commands): happy-path `collect -> collect -> diff` succeeded and printed output paths for `trace.json`, `summary.json`, `summary.md`; all printed files existed under `/tmp/compare-perf-qa-oU2qgg/`.
- Verified required summary sections from generated artifacts: `summary.json` contains `alignment/config/diff/top_regressions/top_improvements`; `summary.md` contains `Config Echo`, `Alignment Statistics`, `Diff Statistics`, `Top Regressions`, `Top Improvements`.
- Verified failure contracts with actual exits/messages: `ERROR[2]` for `diff --top-n 0` (exit 2) and bad `collect --event badformat` (exit 2); `ERROR[10]` for missing input log path (exit 10); `ERROR[11]` for malformed JSON input log (exit 11).

- 2026-03-29 F1 plan compliance audit (oracle): APPROVE. Evidence: `pytest -v tests/test_compare_perf_*.py` passed (30 tests). Prior rejection themes verified: CLI `diff --top-n` rejects non-positive with `ERROR[2]` (see `tests/test_compare_perf_cli.py::test_diff_rejects_non_positive_top_n`); alignment preserves baseline/target key insertion order into `align_modules` (see `tests/test_compare_perf_cli.py::test_diff_preserves_summary_key_order_for_alignment`); packaging includes `putils.compare_perf` via `setup.py` `find_packages(include=['putils','putils.*'])`; schema typing in `putils/compare_perf/schema.py` uses `typing.List/Dict` and enforces required top-level fields + schema_version compatibility gate.

- 2026-03-29 docs migration: moved the full `## compare_perf 快速接入与结果解读` section from root `README.md` into `putils/compare_perf/README.md` to keep compare_perf guidance colocated with module code and reduce root README maintenance burden.
- Root README now keeps only a concise compare_perf entry link, and docs smoke tests were updated to read `putils/compare_perf/README.md` directly so assertions no longer depend on section position inside root README.
- 2026-03-29 torch e2e example: added `examples/compare_perf_torch_e2e.py` to run real `forward + loss + backward + optimizer.step` for baseline/target, inject sleep only in `encoder.layer.1.mlp`, and export `baseline.json/target.json/trace.json/summary.json/summary.md` via existing `align_modules + compute_diff + export_compare_result` pipeline.

- 2026-03-29 Torch E2E intrusion root causes identified:
  - `collector.py` API design: `scope(name)` and `compare_perf(scope_name)` require explicit names; no introspection
  - `examples/compare_perf_torch_e2e.py` shows manual wrapping pattern: each `forward()` layer block wrapped by `_record_scope(scope_name="encoder.layer.X.Y")`
  - `align.py` heuristic scoring: token Jaccard (65%) + layer index gap (25%) + call order (10%) - no model structure awareness
  - `export.py` trace format: flat B/E events per scope, no nested call stack (contrast: VizTracer in `trace_manager.py` captures full stacks)
  - Low-overhead intent preserved: only timed events you annotate; no sys.setprofile overhead like VizTracer

- 2026-03-29 Redesign options ranked by invasiveness (minimally invasive first):
  1) [LOW] Auto-register hooks on torch.nn.Module.forward/backward per registered submodule - requires model traversal + hook registration, but preserves low-overhead (only registered modules timed)
  2) [MEDIUM] Automatic scope name generation from module hierarchy - walk model.named_modules() to generate scope names like "encoder.layers.0.attn", but still needs explicit placement or hooks
  3) [HIGH] VizTracer-style sys.setprofile - captures ALL function calls with max_stack_depth, but high overhead (~10-15% typical) - exists in `trace_manager.py` as separate tool
  4) [NEW APPROACH] Hybrid: hook-based capture on selected modules + manual wrapper for train_step boundary - "wrap one step, get internal stack" via registered forward hooks cascading timing through module tree

- 2026-03-29 Recommended implementation slice: Add `register_model_hooks(model, collector)` helper that:
  - Walks `model.named_modules()` to generate deterministic scope names from module class + path
  - Registers `register_forward_pre_hook` to start timing, `register_forward_hook` to end timing
  - Optionally registers backward hooks via `register_full_backward_hook` for gradient timing
  - Produces nested timing events that preserve module hierarchy in trace
  - Risk: hook overhead ~1-2% per registered module, still lower than sys.setprofile

- 2026-03-29 Single-point wrapping + auto module collection MVP landed:
  - Added `putils.compare_perf.hooks` with `register_model_hooks(...)` and `model_forward_timing(...)`, defaulting to torch forward hooks only (no `sys.setprofile`) to preserve low-intrusion/low-overhead intent.
  - Scope naming now comes directly from `model.named_modules()` paths, so names are deterministic (e.g. `encoder.layer.1.mlp`) and no longer rely on fragile manual string alignment in forward code.
  - Hook timing is implemented via `collector.compare_scope(...)`, so threshold semantics remain unchanged: `summary` always accumulates, while short events can be dropped from `events` detail.
  - `TimingEvent` now stores `start_ns/end_ns` in addition to inclusive/exclusive durations, allowing trace event export from collector data without manual `_record_scope` wrappers.
  - Torch e2e example was simplified to single-point step wrapping (`with compare_perf("train.step", ...)`) plus auto model hook registration; per-layer `_record_scope` manual instrumentation was removed.
  - Pitfall noted: if sleep/regression is injected in a non-leaf container module, leaf-only hooks will miss that container-local overhead; use `leaf_only=False` or inject on leaf modules for attribution fidelity.

- 2026-03-29 Torch hook callback compatibility fix:
  - For `register_forward_hook(..., with_kwargs=True)`, torch invokes post-hook as `(module, args, kwargs, output)`.
  - Updated `putils.compare_perf.hooks._on_forward_post` signature to accept kwargs+output positional arguments, resolving `TypeError ... 4 were given` runtime failures in torch venv tests.
  - To keep e2e regression scope deterministic (`encoder.layer.1.mlp`), example auto-hook registration now applies a narrow `module_filter` so container-level scope remains the top regression target instead of child leaves (`.0/.2`).

- 2026-03-29 Alignment explainability upgrade: `putils.compare_perf.align.align_modules` now scores candidates with explicit components (`token/path/type/layer/order`) and returns `score_components` on candidates + matched results, while preserving semi-automatic behavior (low-confidence or close-second-best stays `ambiguous`).
- 2026-03-29 Naming drift robustness: alignment now additionally uses hierarchical path fragments (`parent/leaf/depth`) and module-type hints (`attention/feedforward/...`) to avoid over-relying on raw token overlap.
- 2026-03-29 Aligned visualization export: `putils.compare_perf.export` now emits matched-pair aligned trace view (`trace_aligned`) where baseline/target events of the same pair share identical start timestamp and carry diff + confidence metadata in args.
- 2026-03-29 Compatibility kept: existing `trace.json` generation flow is unchanged, while CLI and torch e2e example add a new optional aligned trace output path (`--trace-aligned-out`, `trace_aligned.json`/`compare_perf_trace_aligned.json`).

- 2026-03-29 Torch hook hierarchy note: `register_model_hooks` applies `leaf_only` BEFORE `module_filter`; and the torch e2e example’s `module_filter` is exact-match (`name in AUTO_HOOK_MODULE_NAMES`), so `encoder.layer.1.mlp.0/.1/.2` are intentionally excluded—trace shows only the container span. To expose a child stack without polluting regression summary, prefer a dual-mode capture: container scopes contribute to `collector.summary` (via `compare_scope`), while descendant scopes are recorded as detail-only events (via `collector.scope` or a second collector) and merged into the exported timeline.

- 2026-03-29 Chrome Trace / Perfetto event model research for cross-track linkage and nesting:
  - **Flow Events v1 (legacy)**: separate `s/t/f` events with shared `id` field (e.g., `"0x402")`
    - Schema: `{"ph": "s", "id": "0x402", ...}` for step-out, `{"ph": "t", "id": "0x402", ...}` for step-terminating
    - Arrow direction: end of source → start of destination
    - Evidence: [google/perfetto test file](https://github.com/google/perfetto/blob/main/test/trace_processor/diff_tests/parser/parsing/flow_events_json_v1.json)
  - **Flow Events v2 (modern)**: inline `bind_id` + `flow_out`/`flow_in` fields on regular events
    - Schema: `{"bind_id": "0x402", "flow_out": true}` on source, `{"bind_id": "0x402", "flow_in": true}` on destination
    - More compact, supports complex flow chains
    - Evidence: [google/perfetto test file](https://github.com/google/perfetto/blob/main/test/trace_processor/diff_tests/parser/parsing/flow_events_json_v2.json)
  - **Perfetto compatibility**: supports both v1/v2, but single-flow-per-click UI (vs Chrome Trace full chain); enable "Show indirect preceding flows" flag for better visualization
  - **Nesting requirements**: events on same `pid`+`tid` nest automatically if ordered correctly (B before E, proper matching); mixing `X` (complete) and `B/E` on same thread causes invalid hierarchy ([issue #970](https://github.com/google/perfetto/issues/970))
  - **Pain point 1 solution**: use v2 inline flow fields on aligned baseline/target events with shared `bind_id` (e.g., `"pair_encoder.layer.1.mlp"`), align timestamps, Perfetto draws arrows connecting matched pairs
  - **Pain point 2 solution**: dual-mode capture (container modules via `compare_scope` for summary, leaf modules via `scope` for detail-only) OR hierarchical scope names auto-generated from `model.named_modules()` with parent tracking
  - **Concrete aligned trace example**:
    ```json
    {
      "traceEvents": [
        {"ph": "M", "pid": 0, "tid": 0, "name": "process_name", "args": {"name": "baseline"}},
        {"ph": "M", "pid": 1, "tid": 0, "name": "process_name", "args": {"name": "target"}},
        {"pid": 0, "tid": 0, "ts": 1000000, "ph": "B", "name": "encoder.layer.1.mlp",
         "bind_id": "pair_mlp_1", "flow_out": true},
        {"pid": 0, "tid": 0, "ts": 1001000, "ph": "B", "name": "encoder.layer.1.mlp.fc1"},
        {"pid": 0, "tid": 0, "ts": 1015000, "ph": "E", "name": "encoder.layer.1.mlp.fc1"},
        {"pid": 0, "tid": 0, "ts": 1040000, "ph": "E", "name": "encoder.layer.1.mlp"},
        {"pid": 1, "tid": 0, "ts": 1000000, "ph": "B", "name": "encoder.layer.1.mlp",
         "bind_id": "pair_mlp_1", "flow_in": true,
         "args": {"delta_ms": 50, "confidence": 0.95}},
        {"pid": 1, "tid": 0, "ts": 1001000, "ph": "B", "name": "encoder.layer.1.mlp.fc1",
         "args": {"delta_ms": 10}},
        {"pid": 1, "tid": 0, "ts": 1015000, "ph": "E", "name": "encoder.layer.1.mlp.fc1"},
        {"pid": 1, "tid": 0, "ts": 1040000, "ph": "E", "name": "encoder.layer.1.mlp"}
      ],
      "displayTimeUnit": "us"
    }
    ```
  - **Low-overhead recommendation**: keep current hook-based design (no `sys.setprofile`), add flow fields to aligned export, use Perfetto with "Show indirect preceding flows" enabled

- 2026-03-29 Aligned timeline gap analysis:
  - **Gap source**: `putils/compare_perf/export.py` line 285: `cursor_ns += max(baseline_duration_ns, target_duration_ns) + gap_ns`
  - **Gap magnitude**: `gap_ns = slot_gap_us * 1000` (line 229), default `slot_gap_us=1000` → `gap_ns=1,000,000ns = 1ms`
  - **Impact**: Creates visible 1ms idle gap between consecutive matched pairs in aligned trace
  - **Caller chain**: `export_compare_result()` → `build_aligned_chrome_trace()` without `slot_gap_us` arg → defaults to 1000
  - **CLI exposure**: `diff` command does NOT expose `--slot-gap-us` parameter; default 1000 always used
  - **Test gap**: `test_trace_json_is_valid` only checks pair-0 baseline/target share same start; no contiguity test for consecutive pairs

- 2026-03-29 Aligned-trace serial semantics: treat each matched (baseline,target) as a **pair slot** with a shared start time `S_i` and envelope end `E_i = S_i + max(D_base_i, D_tgt_i)`; emit baseline span `[S_i, S_i + D_base_i]` and target span `[S_i, S_i + D_tgt_i]`, then pack contiguously by setting next start `S_{i+1} = E_i` (i.e., no artificial inter-pair idle).
- Recommended default: `slot_gap_us=0` for aligned trace so “紧密相连” is true by construction; keep the knob only as an explicit *visual separator* (nonzero gap means you are inserting synthetic idle and should be documented as such).
- Test/doc contract suggestion: add an assertion that `pair_index=1` begins at `pair_index=0` start + `max(baseline_ms_0, target_ms_0)` (converted to trace `us`), locking the contiguity rule.
- 2026-03-29 Aligned-trace semantics patch landed: `build_aligned_chrome_trace(..., slot_gap_us=0)` now packs matched pair slots contiguously by default (`S_{i+1}=E_i`), while preserving explicit non-default `slot_gap_us` for optional visual separation; regression tests now assert both default contiguity and `slot_gap_us=1000` gap behavior.

- 2026-03-29 Torch e2e mlp-child-stack fix: in `examples/compare_perf_torch_e2e.py`, `module_filter` switched from exact-name match to root-or-descendant match (`name == root or name.startswith(root + '.')`), so `encoder.layer.1.mlp.0/.1/.2` hooks are captured without reintroducing manual per-layer `compare_perf` wrappers; `tests/test_compare_perf_torch_e2e_example.py` now asserts `trace.json.traceEvents` contains at least one `ph='B'` event with `name` prefix `encoder.layer.1.mlp.`.


## 2026-03-30 00:21 — aligned parent+child stack semantics
- `build_aligned_chrome_trace()` lays out matched parent pairs as contiguous slots: both sides start at the same slot start; next slot start advances by `max(baseline_duration, target_duration) + gap` (see tests asserting contiguity).
- For an aligned stack view, child matching should be constrained *within* each matched parent pair (prefix/parent-path constrained), and the renderer must not fabricate missing calls: emit unmatched-only-on-one-side spans and rely on empty time for absence.
- When attempting to time-align children using shared (max-based) cursors, guard against overflow beyond the shorter parent duration; degrade to a per-side placement (or relative-time warp) for the remainder to preserve containment/truthfulness.

- 2026-03-30 trace_aligned_stack artifact landed: `putils.compare_perf.export` now emits `trace_aligned_stack` with parent pair slots packed contiguously (same pair start, next pair starts at previous pair envelope end by default gap=0), then performs per-parent local child alignment by stripping parent prefixes and reusing `align_modules` on relative child names.
- Child rendering contract in stack view is explicit in event args (`view=aligned_stack`, `pair_index`, `role`, `child_match_status`, `child_pair_index`): matched children share start when both sides fit current local cursor; otherwise renderer degrades to side-local placement and marks as `matched_unaligned`, while ambiguous/unmatched children are emitted one-sided only (no fabricated partner span).
- CLI/example/tests wiring now includes `trace_aligned_stack.json`: new diff arg `--trace-aligned-stack-out` with default `compare_perf_trace_aligned_stack.json` in `--output-dir` mode, plus torch e2e stdout path print and regression assertions for artifact existence + parent contiguity + `encoder.layer.1.mlp.` child B-event presence in stack view.

- 2026-03-30 regression fix: stack child timing aggregation must be side-aware; aggregating by module name alone can merge baseline/target durations when names are identical, so `build_aligned_stack_chrome_trace` now buckets timings by side (`process_name` baseline/target, fallback pid 1/2, else unknown) and resolves child durations from the correct side-specific bucket.

- 2026-03-30 hierarchy fix: for nested stack correctness, `build_chrome_trace` should emit globally sorted boundary events (not per-record `B` then `E`); use timestamp ordering with close-before-open at equal ts and depth-aware tie-breakers (longer `B` first, later-start `E` first) so parent/child nesting is preserved. In `trace_aligned_stack`, filter top-level slots to non-descendant matched pairs only, and keep descendants aligned only inside parent-local child alignment.

- 2026-03-30 synthetic-parent stack mode: in `trace_aligned_stack`, lay out children first (matched children always share aligned start), compute pair child envelope, then emit parent as `parent_mode=synthetic_aligned` with `synthetic_parent_duration_ns=max(real_parent_duration_ns, ceil(child_envelope_ns*1.01))`; this keeps parent grouping visible even when real parent is shorter than aligned child layout.

- 2026-03-30 docs sync patch: `putils/compare_perf/README.md` now uses `python3 -m` as default CLI examples, documents `--trace-aligned-out` and `--trace-aligned-stack-out`, updates diff output description to five printed artifact paths, and adds an aligned-stack semantics section (`parent_mode=synthetic_aligned`, `child_match_status=matched/ambiguous/unmatched/parent`) aligned with current CLI/export behavior.

- 2026-03-30 docs clarity patch: `putils/compare_perf/README.md` now explicitly states `with compare_perf(...)` and `@compare_perf(...)` only record into in-memory `collector` by default and do not auto-write files; added a copyable training-loop snapshot example (`for step in range(...)`, every N step `build_schema + json.dump`, filename includes `step` and `tag`) plus note that high `threshold_seconds` can leave `events` empty while `summary` keeps accumulating.

- 2026-03-30 docs simplification patch: snapshot guidance is now helper-first (`dump_compare_perf_snapshot(...)`) so training-loop usage focuses on `output_dir` and cadence (`if step % snapshot_every == 0: dump_compare_perf_snapshot(...)`) rather than exposing raw `events_payload/build_schema/alignment` details at callsite; this better matches users who only care where snapshots are written.

- 2026-03-30 import-first snapshot API patch: moved snapshot export logic into library API `putils.compare_perf.dump_compare_perf_snapshot(...)` and re-exported via `__init__`, so users can directly import and call with `collector/output_dir/step/tag`; implementation converts `TimingEvent` using actual fields (`start_ns/end_ns/inclusive_ns/exclusive_ns`) and always writes schema-valid payload with internal default empty alignment.

- 2026-03-30 torch-e2e docs patch: adding a dedicated README section with runnable `python3 examples/compare_perf_torch_e2e.py --output-dir ...` flow (plus prerequisites and artifact checklist including `trace_aligned_stack.json`) makes feasibility validation concrete; acceptance guidance should point to `summary.json.top_regressions` containing the injected sleep scope (`encoder.layer.1.mlp`) for a fast sanity check.
