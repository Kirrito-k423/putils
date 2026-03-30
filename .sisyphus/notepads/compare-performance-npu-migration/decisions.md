# Decisions

- 2026-03-29 Aligned trace contiguity fix recommendation:
  - **Change 1**: Modify `build_aligned_chrome_trace` default `slot_gap_us` from 1000 → 0
  - **Change 2 (optional)**: Add `slot_gap_us` parameter to `export_compare_result()` and CLI `diff --slot-gap-us` for backward compatibility
  - **Change 3**: Add test assertion: `test_aligned_trace_pair_contiguity` to verify pair[N+1].start_ns == pair[N].max(baseline_end, target_end)
  - **Backward compatibility**: Users who want visual spacing can pass `slot_gap_us=1000` explicitly if CLI parameter added
  - **Minimal patch**: Only change default value (line 216) from `slot_gap_us: int = 1000` → `slot_gap_us: int = 0`


## 2026-03-30 00:21 — decision: `trace_aligned_stack.json` semantics
- Parent slots reuse `trace_aligned.json` slot semantics (contiguous by `max()` duration).
- Child alignment is local-per-parent: match descendants only under the same parent pair (use relative names) and record match confidence/status per child.
- Time transform must preserve: no fake spans, and child spans must not escape their parent's interval on that side; allow a 'degraded' mode when strict alignment would violate containment.
- Emit minimal args metadata for debuggability: `pair_index`, `role`, `parent_baseline_module`, `parent_target_module`, `parent_path`, `child_match_confidence`, `child_match_status`, `child_pair_index`, `depth`.
