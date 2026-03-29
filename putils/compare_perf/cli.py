from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence

from putils.compare_perf.align import AlignmentMappingCache, align_modules
from putils.compare_perf.config import collect_runtime_metadata, make_timestamped_log_filename
from putils.compare_perf.diff import compute_diff
from putils.compare_perf.export import export_compare_result
from putils.compare_perf.schema import SchemaValidationError, build_schema, parse_schema


EXIT_OK = 0
EXIT_INVALID_ARGS = 2
EXIT_MISSING_INPUT = 10
EXIT_SCHEMA_ERROR = 11
EXIT_RUNTIME_ERROR = 12


class CliError(RuntimeError):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = int(code)
        self.message = str(message)


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise CliError(EXIT_INVALID_ARGS, message)


def _positive_int(raw_value: str) -> int:
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc

    if value <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return value


def _build_parser() -> _ArgumentParser:
    parser = _ArgumentParser(prog="compare-perf")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument(
        "--event",
        action="append",
        required=True,
        help="Event spec format: <scope_name>:<duration_ms>. Repeatable.",
    )
    collect_parser.add_argument("--output", default=None, help="Output log path (.json)")
    collect_parser.add_argument("--output-dir", default=".", help="Output directory when --output is omitted")
    collect_parser.add_argument("--tag", default="collect", help="Run metadata tag")

    diff_parser = subparsers.add_parser("diff")
    diff_parser.add_argument("--baseline", required=True, help="Baseline log path")
    diff_parser.add_argument("--target", required=True, help="Target log path")
    diff_parser.add_argument("--trace-out", default=None, help="Output path for chrome trace json")
    diff_parser.add_argument(
        "--trace-aligned-out",
        default=None,
        help="Output path for matched-pair aligned chrome trace json",
    )
    diff_parser.add_argument(
        "--trace-aligned-stack-out",
        default=None,
        help="Output path for parent-aligned + child-stacked chrome trace json",
    )
    diff_parser.add_argument("--summary-json-out", default=None, help="Output path for summary json")
    diff_parser.add_argument("--summary-md-out", default=None, help="Output path for summary markdown")
    diff_parser.add_argument("--output-dir", default=".", help="Output directory for default diff artifacts")
    diff_parser.add_argument(
        "--top-n",
        type=_positive_int,
        default=5,
        help="Top N regressions/improvements in summary",
    )
    diff_parser.add_argument(
        "--enable-rank-aggregation",
        action="store_true",
        help="Enable p50/p95 rank aggregation when rank summaries exist",
    )
    diff_parser.add_argument(
        "--alignment-cache",
        default=None,
        help="Optional cache file for confirmed mappings",
    )
    diff_parser.add_argument(
        "--mapping",
        action="append",
        default=[],
        help="Manual mapping spec format: <baseline_module>=<target_module>. Repeatable.",
    )

    return parser


def _parse_event_spec(raw_spec: str) -> tuple[str, int]:
    if ":" not in raw_spec:
        raise CliError(
            EXIT_INVALID_ARGS,
            f"invalid --event '{raw_spec}': expected <scope_name>:<duration_ms>",
        )

    scope_name, duration_text = raw_spec.rsplit(":", 1)
    scope_name = scope_name.strip()
    duration_text = duration_text.strip()
    if not scope_name:
        raise CliError(
            EXIT_INVALID_ARGS,
            f"invalid --event '{raw_spec}': scope_name must be non-empty",
        )

    try:
        duration_ms = float(duration_text)
    except ValueError as exc:
        raise CliError(
            EXIT_INVALID_ARGS,
            f"invalid --event '{raw_spec}': duration_ms must be a number",
        ) from exc

    if duration_ms <= 0.0:
        raise CliError(
            EXIT_INVALID_ARGS,
            f"invalid --event '{raw_spec}': duration_ms must be > 0",
        )

    duration_ns = int(round(duration_ms * 1_000_000.0))
    return scope_name, duration_ns


def _parse_manual_mappings(raw_mappings: Sequence[str]) -> dict[str, str]:
    confirmed: dict[str, str] = {}
    for raw in raw_mappings:
        if "=" not in raw:
            raise CliError(
                EXIT_INVALID_ARGS,
                f"invalid --mapping '{raw}': expected <baseline_module>=<target_module>",
            )
        left, right = raw.split("=", 1)
        left_name = left.strip()
        right_name = right.strip()
        if not left_name or not right_name:
            raise CliError(
                EXIT_INVALID_ARGS,
                f"invalid --mapping '{raw}': both module names must be non-empty",
            )
        confirmed[left_name] = right_name
    return confirmed


def _build_collect_payload(event_specs: Sequence[str], tag: str) -> dict[str, Any]:
    runtime_metadata = collect_runtime_metadata()
    runtime_metadata["source"] = "compare_perf_cli_collect"
    runtime_metadata["tag"] = str(tag)

    timeline_events: list[dict[str, Any]] = []
    summary: dict[str, dict[str, Any]] = {}
    current_start_ns = 0

    for raw_spec in event_specs:
        scope_name, duration_ns = _parse_event_spec(raw_spec)
        start_ns = current_start_ns
        end_ns = start_ns + duration_ns
        current_start_ns = end_ns

        timeline_events.append(
            {
                "name": scope_name,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "pid": int(runtime_metadata.get("pid", 0)),
                "tid": 0,
                "thread_name": "main",
                "process_name": str(runtime_metadata.get("device", "unknown")),
                "args": {"source": "collect_cli"},
            }
        )

        current_summary = summary.get(scope_name)
        if current_summary is None:
            current_summary = {
                "call_count": 0,
                "total_ns": 0,
                "total_seconds": 0.0,
            }
            summary[scope_name] = current_summary

        total_ns = int(current_summary["total_ns"]) + duration_ns
        call_count = int(current_summary["call_count"]) + 1
        current_summary["call_count"] = call_count
        current_summary["total_ns"] = total_ns
        current_summary["total_seconds"] = float(total_ns) / 1_000_000_000.0

    alignment_placeholder = {
        "matched": [],
        "ambiguous": [],
        "unmatched": {"left": [], "right": []},
        "counts": {"matched": 0, "ambiguous": 0, "unmatched": 0},
    }

    payload = build_schema(
        events=timeline_events,
        run_metadata=runtime_metadata,
        alignment=alignment_placeholder,
        summary=summary,
    )
    return dict(payload)


def _load_schema_log(path: str) -> Mapping[str, Any]:
    if not os.path.exists(path):
        raise CliError(EXIT_MISSING_INPUT, f"missing input log: {path}")
    if not os.path.isfile(path):
        raise CliError(EXIT_MISSING_INPUT, f"missing input log: {path}")

    try:
        with open(path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
    except json.JSONDecodeError as exc:
        raise CliError(EXIT_SCHEMA_ERROR, f"invalid json in input log '{path}': {exc.msg}") from exc
    except OSError as exc:
        raise CliError(EXIT_RUNTIME_ERROR, f"failed to read input log '{path}': {exc}") from exc

    try:
        return parse_schema(payload)
    except SchemaValidationError as exc:
        raise CliError(EXIT_SCHEMA_ERROR, f"incompatible schema in input log '{path}': {exc}") from exc


def _extract_summary(payload: Mapping[str, Any], *, source: str) -> Mapping[str, Mapping[str, Any]]:
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        raise CliError(EXIT_SCHEMA_ERROR, f"invalid summary in {source}: expected mapping")
    return summary


def _extract_rank_summary(payload: Mapping[str, Any]) -> Mapping[str, list[Any]] | None:
    run_metadata = payload.get("run_metadata")
    if not isinstance(run_metadata, Mapping):
        return None
    rank_summary = run_metadata.get("rank_summary")
    if not isinstance(rank_summary, Mapping):
        return None
    return rank_summary


def _to_int_event_field(event: Mapping[str, Any], key: str, *, index: int, source: str) -> int:
    if key not in event:
        raise CliError(
            EXIT_SCHEMA_ERROR,
            f"invalid event in {source} at index {index}: missing field '{key}'",
        )
    try:
        return int(event[key])
    except (TypeError, ValueError) as exc:
        raise CliError(
            EXIT_SCHEMA_ERROR,
            f"invalid event in {source} at index {index}: field '{key}' must be an integer",
        ) from exc


def _normalize_events(payload: Mapping[str, Any], *, label: str, pid: int) -> list[dict[str, Any]]:
    raw_events = payload.get("events", [])
    if not isinstance(raw_events, Sequence):
        raise CliError(EXIT_SCHEMA_ERROR, f"invalid events in {label}: expected list")

    normalized: list[dict[str, Any]] = []
    for index, raw_event in enumerate(raw_events):
        if not isinstance(raw_event, Mapping):
            raise CliError(EXIT_SCHEMA_ERROR, f"invalid event in {label} at index {index}: expected mapping")

        name = str(raw_event.get("name", "")).strip()
        if not name:
            raise CliError(EXIT_SCHEMA_ERROR, f"invalid event in {label} at index {index}: empty name")

        start_ns = _to_int_event_field(raw_event, "start_ns", index=index, source=label)
        end_ns = _to_int_event_field(raw_event, "end_ns", index=index, source=label)
        if end_ns < start_ns:
            raise CliError(
                EXIT_SCHEMA_ERROR,
                f"invalid event in {label} at index {index}: end_ns must be >= start_ns",
            )

        normalized.append(
            {
                "name": name,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "pid": pid,
                "tid": int(raw_event.get("tid", 0)),
                "thread_name": str(raw_event.get("thread_name", "main")),
                "process_name": label,
                "args": {"origin": label},
            }
        )
    return normalized


def _write_json_file(path: str, payload: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def _collect_command(args: argparse.Namespace) -> int:
    output_path = args.output
    if output_path is None:
        output_path = make_timestamped_log_filename(
            output_dir=args.output_dir,
            prefix="compare_perf_log",
            extension=".json",
        )

    payload = _build_collect_payload(args.event, tag=args.tag)
    _write_json_file(output_path, payload)
    print(output_path)
    return EXIT_OK


def _diff_command(args: argparse.Namespace) -> int:
    baseline_payload = _load_schema_log(args.baseline)
    target_payload = _load_schema_log(args.target)

    baseline_summary = _extract_summary(baseline_payload, source="baseline")
    target_summary = _extract_summary(target_payload, source="target")

    left_modules = [str(module_name) for module_name in baseline_summary.keys()]
    right_modules = [str(module_name) for module_name in target_summary.keys()]

    cache = AlignmentMappingCache(args.alignment_cache) if args.alignment_cache else None
    confirmed_mappings = _parse_manual_mappings(args.mapping)
    alignment = align_modules(
        left_modules=left_modules,
        right_modules=right_modules,
        confirmed_mappings=confirmed_mappings,
        cache=cache,
    )

    diff_result = compute_diff(
        alignment=alignment,
        baseline_summary=baseline_summary,
        target_summary=target_summary,
        baseline_rank_summary=_extract_rank_summary(baseline_payload),
        target_rank_summary=_extract_rank_summary(target_payload),
        enable_rank_aggregation=bool(args.enable_rank_aggregation),
    )

    timeline_events = _normalize_events(baseline_payload, label="baseline", pid=1)
    timeline_events.extend(_normalize_events(target_payload, label="target", pid=2))

    trace_out = args.trace_out or os.path.join(args.output_dir, "compare_perf_trace.json")
    if args.trace_aligned_out:
        trace_aligned_out = args.trace_aligned_out
    elif args.trace_out:
        trace_aligned_out = str(Path(trace_out).with_name("trace_aligned.json"))
    else:
        trace_aligned_out = os.path.join(args.output_dir, "compare_perf_trace_aligned.json")
    if args.trace_aligned_stack_out:
        trace_aligned_stack_out = args.trace_aligned_stack_out
    elif args.trace_out:
        trace_aligned_stack_out = str(Path(trace_out).with_name("trace_aligned_stack.json"))
    else:
        trace_aligned_stack_out = os.path.join(args.output_dir, "compare_perf_trace_aligned_stack.json")
    summary_json_out = args.summary_json_out or os.path.join(args.output_dir, "compare_perf_summary.json")
    summary_md_out = args.summary_md_out or os.path.join(args.output_dir, "compare_perf_summary.md")

    export_compare_result(
        timeline_events=timeline_events,
        alignment=alignment,
        diff_result=diff_result,
        config_echo={
            "top_n": args.top_n,
            "enable_rank_aggregation": bool(args.enable_rank_aggregation),
            "baseline": str(args.baseline),
            "target": str(args.target),
        },
        trace_json_path=trace_out,
        aligned_trace_json_path=trace_aligned_out,
        aligned_stack_trace_json_path=trace_aligned_stack_out,
        summary_json_path=summary_json_out,
        summary_md_path=summary_md_out,
        top_n=int(args.top_n),
    )

    print(trace_out)
    print(trace_aligned_out)
    print(trace_aligned_stack_out)
    print(summary_json_out)
    print(summary_md_out)
    return EXIT_OK


def run_cli(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "collect":
        return _collect_command(args)
    if args.command == "diff":
        return _diff_command(args)

    raise CliError(EXIT_INVALID_ARGS, f"unsupported command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run_cli(argv)
    except CliError as exc:
        print(f"ERROR[{exc.code}] {exc.message}", file=sys.stderr)
        return exc.code
    except Exception as exc:
        print(f"ERROR[{EXIT_RUNTIME_ERROR}] unexpected failure: {exc}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
