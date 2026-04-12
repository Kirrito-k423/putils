"""将两个 hooks 精度日志解析为 Excel 对比报告。

命令行：
    python parse_hooks_to_excel.py --base-log <base.log> --target-log <target.log> --output <result.xlsx>

输出包含 3 个工作表：
1) base_parsed：base 日志解析结果
2) target_parsed：target 日志解析结果
3) comparison：同名项对比（hash/shape/l1_norm/mean/sum）
"""

import argparse
import ast
import json
import re
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any

from openpyxl import Workbook


HEADER_RE = re.compile(
    r"^'(?P<name>[^']+)'\|\s*"
    r"(?:l1_norm\s+(?P<l1_norm>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*\|\s*)?"
    r"(?P<hash>[^|]+?)\s*\|\s*"
    r"(?P<dtype>[^|]+?)\s*\|\s*"
    r"(?P<shape>torch\.Size\([^)]*\))\s*\|\s*"
    r"continue:\s*(?:True|False)\s*\|\s*"
    r"mean\s+(?P<mean>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*\|\s*"
    r"sum\s+(?P<sum>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*\|\s*"
    r"Size\s+(?P<size>\d+)\s*\|\s*"
    r"Memory size:\s*[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*GB\s*\|\s*"
    r"isNan\s+(?P<is_nan>True|False)\s*$"
)


@dataclass
class HookEntry:
    name: str
    l1_norm: float | None
    hash_value: str
    dtype: str
    shape: str
    mean: float
    sum_value: float
    size: int
    is_nan: bool
    tensor_preview: list[Any]


def _parse_preview_line(line: str) -> list[Any]:
    stripped = line.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return []

    try:
        value = ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        return []

    if isinstance(value, list):
        return value[:10]
    return []


def parse_hook_log(log_path: Path) -> list[HookEntry]:
    lines = log_path.read_text(encoding="utf-8").splitlines()
    entries: list[HookEntry] = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        match = HEADER_RE.match(line)
        if not match:
            i += 1
            continue

        preview = []
        if i + 1 < len(lines):
            preview = _parse_preview_line(lines[i + 1])

        entries.append(
            HookEntry(
                name=match.group("name"),
                l1_norm=float(match.group("l1_norm")) if match.group("l1_norm") is not None else None,
                hash_value=match.group("hash").strip(),
                dtype=match.group("dtype").strip(),
                shape=match.group("shape").strip(),
                mean=float(match.group("mean")),
                sum_value=float(match.group("sum")),
                size=int(match.group("size")),
                is_nan=match.group("is_nan") == "True",
                tensor_preview=preview,
            )
        )
        i += 2

    return entries


def _diff_pct(base_val: float, target_val: float) -> float:
    denominator = max(abs(base_val), abs(target_val), 1e-10)
    return abs(base_val - target_val) / denominator * 100.0


def build_comparison_rows(
    base_entries: list[HookEntry], target_entries: list[HookEntry]
) -> list[list[Any]]:
    """按 base 日志默认顺序生成 comparison 数据。"""
    rows: list[list[Any]] = []

    processed_from_base: set[str] = set()

    base_seen: dict[str, list[HookEntry]] = {}
    for entry in base_entries:
        base_seen.setdefault(entry.name, []).append(entry)

    target_seen: dict[str, list[HookEntry]] = {}
    for entry in target_entries:
        target_seen.setdefault(entry.name, []).append(entry)

    for entry in base_entries:
        name = entry.name
        if name in processed_from_base:
            continue
        processed_from_base.add(name)

        base_list = base_seen.get(name, [])
        target_list = target_seen.get(name, [])

        for base_item, target_item in zip_longest(base_list, target_list):
            base_mean = base_item.mean if base_item else None
            target_mean = target_item.mean if target_item else None
            base_sum = base_item.sum_value if base_item else None
            target_sum = target_item.sum_value if target_item else None
            base_l1_norm = base_item.l1_norm if base_item else None
            target_l1_norm = target_item.l1_norm if target_item else None

            l1_norm_diff_pct = None
            mean_diff_pct = None
            sum_diff_pct = None
            if base_l1_norm is not None and target_l1_norm is not None:
                l1_norm_diff_pct = _diff_pct(base_l1_norm, target_l1_norm)
            if base_mean is not None and target_mean is not None:
                mean_diff_pct = _diff_pct(base_mean, target_mean)
            if base_sum is not None and target_sum is not None:
                sum_diff_pct = _diff_pct(base_sum, target_sum)

            rows.append(
                [
                    name,
                    (base_item.hash_value == target_item.hash_value)
                    if (base_item and target_item)
                    else False,
                    (base_item.shape == target_item.shape) if (base_item and target_item) else False,
                    base_l1_norm,
                    target_l1_norm,
                    l1_norm_diff_pct,
                    base_mean,
                    target_mean,
                    mean_diff_pct,
                    base_sum,
                    target_sum,
                    sum_diff_pct,
                    base_item.hash_value if base_item else "",
                    target_item.hash_value if target_item else "",
                    base_item.shape if base_item else "",
                    target_item.shape if target_item else "",
                ]
            )

    for entry in target_entries:
        name = entry.name
        if name in processed_from_base:
            continue
        processed_from_base.add(name)

        target_list = target_seen.get(name, [])

        for target_item in target_list:
            rows.append(
                [
                    name,
                    False,
                    False,
                    None,
                    target_item.l1_norm,
                    None,
                    None,
                    target_item.mean,
                    None,
                    None,
                    target_item.sum_value,
                    None,
                    "",
                    target_item.hash_value,
                    "",
                    target_item.shape,
                ]
            )

    return rows


def _write_parsed_sheet(workbook: Workbook, sheet_name: str, entries: list[HookEntry]) -> None:
    sheet = workbook.create_sheet(title=sheet_name)
    sheet.append(
        [
            "name",
            "hash",
            "dtype",
            "shape",
            "l1_norm",
            "mean",
            "sum",
            "size",
            "is_nan",
            "tensor_preview",
        ]
    )

    for entry in entries:
        sheet.append(
            [
                entry.name,
                entry.hash_value,
                entry.dtype,
                entry.shape,
                entry.l1_norm,
                entry.mean,
                entry.sum_value,
                entry.size,
                entry.is_nan,
                json.dumps(entry.tensor_preview, ensure_ascii=False),
            ]
        )


def _write_comparison_sheet(workbook: Workbook, rows: list[list[Any]]) -> None:
    sheet = workbook.create_sheet(title="comparison")
    sheet.append(
        [
            "name",
            "hash_match",
            "shape_match",
            "base_l1_norm",
            "target_l1_norm",
            "l1_norm_diff_pct",
            "base_mean",
            "target_mean",
            "mean_diff_pct",
            "base_sum",
            "target_sum",
            "sum_diff_pct",
            "base_hash",
            "target_hash",
            "base_shape",
            "target_shape",
        ]
    )

    for row in rows:
        sheet.append(row)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 hooks 日志解析为 Excel 对比文件")
    parser.add_argument("--base-log", required=True, help="基准日志文件路径")
    parser.add_argument("--target-log", required=True, help="目标日志文件路径")
    parser.add_argument("--output", required=True, help="输出 Excel 文件路径")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    base_log = Path(args.base_log).expanduser().resolve()
    target_log = Path(args.target_log).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not base_log.exists():
        raise FileNotFoundError(f"base log not found: {base_log}")
    if not target_log.exists():
        raise FileNotFoundError(f"target log not found: {target_log}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_entries = parse_hook_log(base_log)
    target_entries = parse_hook_log(target_log)
    comparison_rows = build_comparison_rows(base_entries, target_entries)

    workbook = Workbook()
    active_sheet = workbook.active
    if active_sheet is not None:
        workbook.remove(active_sheet)

    _write_parsed_sheet(workbook, "base_parsed", base_entries)
    _write_parsed_sheet(workbook, "target_parsed", target_entries)
    _write_comparison_sheet(workbook, comparison_rows)

    workbook.save(output_path)
    print(f"Wrote Excel comparison file: {output_path}")
    print(f"Parsed entries - base: {len(base_entries)}, target: {len(target_entries)}")


if __name__ == "__main__":
    main()
