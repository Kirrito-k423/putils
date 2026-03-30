# Compare Performance Summary

## Config Echo
- `baseline`: `.sisyphus/tmp_torch_e2e_verify/baseline.json`
- `scenario`: `torch_forward_backward_optimizer`
- `sleep_scope`: `encoder.layer.1.mlp`
- `sleep_seconds`: `0.02`
- `steps`: `4`
- `target`: `.sisyphus/tmp_torch_e2e_verify/target.json`

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
- encoder.layer.1.mlp -> encoder.layer.1.mlp: delta_ms=115.941832, delta_pct=73965.609151
- encoder.layer.1.mlp.0 -> encoder.layer.1.mlp.0: delta_ms=0.226376, delta_pct=814.536557
- train.optimizer.step -> train.optimizer.step: delta_ms=0.069917, delta_pct=36.359996
- encoder.layer.1.mlp.2 -> encoder.layer.1.mlp.2: delta_ms=0.029127, delta_pct=158.886101
- head.proj -> head.proj: delta_ms=0.023667, delta_pct=145.2587

## Top Improvements
- train.backward -> train.backward: delta_ms=-9.993749, delta_pct=-95.539876
- encoder.layer.0.attn -> encoder.layer.0.attn: delta_ms=-0.150207, delta_pct=-75.088107
