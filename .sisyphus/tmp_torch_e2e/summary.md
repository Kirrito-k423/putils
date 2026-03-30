# Compare Performance Summary

## Config Echo
- `baseline`: `.sisyphus/tmp_torch_e2e/baseline.json`
- `scenario`: `torch_forward_backward_optimizer`
- `sleep_scope`: `encoder.layer.1.mlp`
- `sleep_seconds`: `0.02`
- `steps`: `4`
- `target`: `.sisyphus/tmp_torch_e2e/target.json`

## Alignment Statistics
- ambiguous: 1
- matched: 9
- unmatched: 1
- unmatched_left: 0
- unmatched_right: 1

## Diff Statistics
- compared_modules: 9
- excluded_pairs: 0
- matched_pairs: 9
- rank_aggregation_enabled: False

## Top Regressions
- encoder.layer.1.mlp -> encoder.layer.1.mlp: delta_ms=117.384876, delta_pct=16847.536843
- encoder.layer.1.mlp.0 -> encoder.layer.1.mlp.0: delta_ms=0.271209, delta_pct=615.796285
- encoder.layer.1.mlp.2 -> encoder.layer.1.mlp.2: delta_ms=0.023708, delta_pct=96.116111
- head.proj -> head.proj: delta_ms=0.016999, delta_pct=72.333092

## Top Improvements
- train.backward -> train.backward: delta_ms=-13.720458, delta_pct=-96.180992
- train.loss -> train.loss: delta_ms=-5.319626, delta_pct=-95.487829
- encoder.layer.0.attn -> encoder.layer.0.attn: delta_ms=-3.434001, delta_pct=-98.610891
- train.optimizer.step -> train.optimizer.step: delta_ms=-0.669582, delta_pct=-68.966931
- encoder.layer.1.mlp.1 -> encoder.layer.1.mlp.1: delta_ms=-0.480207, delta_pct=-89.926573
