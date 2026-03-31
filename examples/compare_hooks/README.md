# compare_hooks 日志转 Excel 说明

## 功能

`parse_hooks_to_excel.py` 用于把两份 hooks 精度日志解析成一个 Excel 文件，便于后续可视化与差异分析。

## 命令行用法

```bash
python examples/compare_hooks/parse_hooks_to_excel.py \
  --base-log examples/compare_hooks/base.log \
  --target-log examples/compare_hooks/target.log \
  --output examples/compare_hooks/compare_result.xlsx
```

## 输出结构

输出 Excel 包含 3 个工作表：

1. `base_parsed`
   - base 日志解析结果
   - 列：`name, hash, dtype, shape, mean, sum, size, is_nan, tensor_preview`

2. `target_parsed`
   - target 日志解析结果
   - 列同 `base_parsed`

3. `comparison`
   - 同名元素比较结果（按 base 日志默认顺序）
   - 列：
     - `name`
     - `hash_match`
     - `shape_match`
     - `base_mean`
     - `target_mean`
     - `mean_diff_pct`
     - `base_sum`
     - `target_sum`
     - `sum_diff_pct`
     - `base_hash`
     - `target_hash`
     - `base_shape`
     - `target_shape`

## 差异百分比公式

- `mean_diff_pct = abs(base_mean - target_mean) / max(abs(base_mean), abs(target_mean), 1e-10) * 100`
- `sum_diff_pct = abs(base_sum - target_sum) / max(abs(base_sum), abs(target_sum), 1e-10) * 100`

## 解析规则

- 仅解析标准头行 + 下一行预览数组格式。
- 遇到大 tensor 多行打印（如 `tensor([...])`）会自动跳过。
- 仅保留预览数组前 10 个元素写入 `tensor_preview`。
- `None`、`dict` 等非标准条目会跳过。
