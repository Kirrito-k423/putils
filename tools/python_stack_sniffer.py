#!/usr/bin/env python3
"""
Python Stack Sniffer Tool

This tool periodically captures stack traces from a Python process using py-spy
and converts them to Chrome Tracing JSON format for visualization.

Usage:
    python python_stack_sniffer.py -p <pid> -i <interval> -o <output.json>

中文使用说明：
    

使用场景示例：
    # 不传pid：自动从npu-smi info抓取所有进程pid
    python python_stack_sniffer.py -i 0.2 -d 10 -o trace.json
    
    # 仍然支持手工pid list
    python python_stack_sniffer.py -p 44002,44003 -i 0.2 -d 10 -o trace.json

    # 自动间隔保存
    python python_stack_sniffer.py -p 1667631  -i 0.1 -o stack_trace.json --autosave-interval 10
"""

import argparse
import subprocess
import json
import time
import logging
import re
import os
import tempfile
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _atomic_write_json(file_path: str, data: Any, indent: int = 2) -> None:
    dir_name = os.path.dirname(os.path.abspath(file_path)) or "."
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=dir_name, delete=False) as tf:
            tmp_path = tf.name
            json.dump(data, tf, indent=indent)
            tf.flush()
            os.fsync(tf.fileno())
        os.replace(tmp_path, file_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Python Stack Sniffer for Chrome Tracing')
    parser.add_argument(
        '-p',
        '--pid',
        dest='pids',
        action='append',
        required=False,
        help='Process ID list. Repeatable (-p 1 -p 2) or comma-separated (-p 1,2). If omitted, uses `npu-smi info` to discover PIDs.',
    )
    parser.add_argument('-i', '--interval', type=float, default=0.1, help='Sampling interval in seconds')
    parser.add_argument('-o', '--output', type=str, default='stack_trace.json', help='Output JSON file path')
    parser.add_argument('-d', '--duration', type=float, help='Duration to run in seconds (optional)')
    parser.add_argument('--autosave-interval', type=float, default=0.0, help='Auto-save interval in seconds (0 to disable)')
    parser.add_argument('--autosave-output', type=str, default=None, help='Auto-save JSON file path (default: same as --output)')
    parser.add_argument('--all-threads', action='store_true', help='Capture all threads (default: only MainThread)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    return parser.parse_args()


def _normalize_pids(pid_args: List[str]) -> List[int]:
    pids: List[int] = []
    seen = set()

    for item in pid_args:
        for part in item.split(','):
            part = part.strip()
            if not part:
                continue
            pid = int(part)
            if pid not in seen:
                pids.append(pid)
                seen.add(pid)

    if not pids:
        raise ValueError('empty pid list')

    return pids


def _filter_stack_data(stack_data: Dict[str, Any], main_thread_only: bool) -> Dict[str, Any]:
    if not main_thread_only:
        return stack_data

    threads = stack_data.get('threads') or []
    main_threads = [t for t in threads if (t.get('name') or '') == 'MainThread']
    if main_threads:
        return {"threads": main_threads}

    return stack_data


def _get_pids_from_npu_smi_info() -> List[int]:
    try:
        result = subprocess.run(
            ['npu-smi', 'info'],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        logger.error('npu-smi not found; please provide -p/--pid explicitly')
        return []
    except subprocess.CalledProcessError as e:
        logger.error(f"npu-smi info failed: {(e.stderr or '').strip()}")
        return []

    pids: List[int] = []
    seen = set()
    in_process_table = False

    for raw_line in (result.stdout or '').splitlines():
        line = raw_line.rstrip('\n')
        header = line.lower()
        if re.search(r'process\s*id', header) and re.search(r'process\s*name', header):
            in_process_table = True
            continue
        if not in_process_table:
            continue
        if line.startswith('+'):
            continue
        if not line.strip().startswith('|'):
            continue

        parts = [p.strip() for p in line.strip().strip('|').split('|')]
        if len(parts) < 3:
            continue

        pid_str: Optional[str] = None
        for token in parts[1:]:
            if token.isdigit():
                pid_str = token
                break
        if pid_str is None:
            continue

        pid = int(pid_str)
        if pid not in seen:
            pids.append(pid)
            seen.add(pid)

    return pids


def _run_pyspy_dump(cmd: List[str]) -> Tuple[Optional[str], Optional[str]]:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout, None
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        return None, stderr


def get_stack_traces(pid: int) -> str:
    """Capture stack traces using py-spy dump (text output)."""
    cmds = [
        ['py-spy', 'dump', '--pid', str(pid)],
        ['py-spy', 'dump', '-p', str(pid)],
    ]

    last_err = None
    for cmd in cmds:
        stdout, err = _run_pyspy_dump(cmd)
        if stdout is not None:
            return stdout
        last_err = err

    if last_err:
        logger.error(f"Failed to capture stack traces: {last_err}")
    return ""


_THREAD_HEADER_RE = re.compile(r'^Thread\s+(?P<rest>.+?)\s*$')
_FILE_FRAME_RE = re.compile(
    r'^\s*File\s+"(?P<file>[^"]+)",\s+line\s+(?P<line>\d+),\s+in\s+(?P<func>.+?)\s*$'
)
_SIMPLE_FRAME_RE = re.compile(r'^\s*(?P<func>.+?)\s+\((?P<file>.+?):(?P<line>\d+)\)\s*$')
_NATIVE_FRAME_RE = re.compile(r'^\s*\[native\]\s+(?P<func>.+?)\s*$')


def _parse_thread_header(rest: str) -> Tuple[int, str]:
    tid_token = rest.split()[0] if rest.split() else "0"
    if tid_token.startswith("0x"):
        try:
            tid = int(tid_token, 16)
        except ValueError:
            tid = 0
    else:
        try:
            tid = int(tid_token)
        except ValueError:
            tid = 0

    name = ""
    m = re.search(r'"([^"]+)"', rest)
    if m:
        name = m.group(1)
    elif ":" in rest:
        name = rest.split(":", 1)[1].strip().strip('"')

    if not name:
        name = f"Thread {tid_token}"

    return tid, name


def parse_pyspy_output(pyspy_output: str) -> Dict[str, Any]:
    """Parse py-spy dump text output into a normalized structure."""
    if not pyspy_output:
        return {"threads": []}

    threads: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for raw_line in pyspy_output.splitlines():
        line = raw_line.rstrip("\n")
        m_th = _THREAD_HEADER_RE.match(line)
        if m_th:
            if current is not None:
                threads.append(current)
            rest = m_th.group("rest")
            tid, name = _parse_thread_header(rest)
            current = {"id": tid, "name": name, "stack": []}
            continue

        if current is None:
            continue

        m_ff = _FILE_FRAME_RE.match(line)
        if m_ff:
            current["stack"].append(
                {
                    "name": m_ff.group("func"),
                    "filename": m_ff.group("file"),
                    "lineno": int(m_ff.group("line")),
                }
            )
            continue

        m_sf = _SIMPLE_FRAME_RE.match(line)
        if m_sf:
            current["stack"].append(
                {
                    "name": m_sf.group("func"),
                    "filename": m_sf.group("file"),
                    "lineno": int(m_sf.group("line")),
                }
            )
            continue

        m_nf = _NATIVE_FRAME_RE.match(line)
        if m_nf:
            current["stack"].append({"name": m_nf.group("func"), "filename": "[native]", "lineno": 0})
            continue

    if current is not None:
        threads.append(current)

    return {"threads": threads}


def convert_to_chrome_tracing(
    pid: int,
    stack_data: Dict[str, Any],
    ts_us: int,
    seen_thread_ids: set,
    open_stacks: Dict[int, List[str]],
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []

    current_tids = set()

    for thread in stack_data.get("threads", []):
        thread_id = int(thread.get("id", 0) or 0)
        thread_name = thread.get("name") or f"Thread {thread_id}"
        stack = thread.get("stack") or []

        current_tids.add(thread_id)

        if thread_id not in seen_thread_ids:
            events.append(
                {
                    "name": "thread_name",
                    "ph": "M",
                    "ts": ts_us,
                    "tid": thread_id,
                    "pid": pid,
                    "args": {"name": thread_name},
                }
            )
            seen_thread_ids.add(thread_id)

        frames = list(reversed(stack))

        names: List[str] = []
        for frame in frames:
            fn = frame.get("name", "unknown")
            file_name = frame.get("filename", "unknown")
            line_num = frame.get("lineno", 0)
            names.append(f"{fn} ({file_name}:{line_num})")

        prev_names = open_stacks.get(thread_id, [])
        common = 0
        max_common = min(len(prev_names), len(names))
        while common < max_common and prev_names[common] == names[common]:
            common += 1

        for i in range(len(prev_names) - 1, common - 1, -1):
            events.append(
                {
                    "name": prev_names[i],
                    "ph": "E",
                    "ts": ts_us,
                    "tid": thread_id,
                    "pid": pid,
                }
            )

        for i in range(common, len(names)):
            frame = frames[i]
            events.append(
                {
                    "name": names[i],
                    "ph": "B",
                    "ts": ts_us,
                    "tid": thread_id,
                    "pid": pid,
                    "args": {
                        "thread": thread_name,
                        "function": frame.get("name", "unknown"),
                        "file": frame.get("filename", "unknown"),
                        "line": frame.get("lineno", 0),
                    },
                }
            )

        open_stacks[thread_id] = names

    for thread_id in list(open_stacks.keys()):
        if thread_id in current_tids:
            continue
        prev_names = open_stacks.get(thread_id, [])
        for i in range(len(prev_names) - 1, -1, -1):
            events.append(
                {
                    "name": prev_names[i],
                    "ph": "E",
                    "ts": ts_us,
                    "tid": thread_id,
                    "pid": pid,
                }
            )
        open_stacks.pop(thread_id, None)

    return events


def close_open_stacks(pid: int, ts_us: int, open_stacks: Dict[int, List[str]]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for thread_id, names in list(open_stacks.items()):
        for i in range(len(names) - 1, -1, -1):
            events.append(
                {
                    "name": names[i],
                    "ph": "E",
                    "ts": ts_us,
                    "tid": thread_id,
                    "pid": pid,
                }
            )
    open_stacks.clear()
    return events


def main():
    args = parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if args.pids:
        pids = _normalize_pids(args.pids)
    else:
        pids = _get_pids_from_npu_smi_info()
        if not pids:
            raise RuntimeError('No PIDs found from `npu-smi info` and no -p/--pid provided')

    main_thread_only = not args.all_threads

    logger.info(f"Starting Python Stack Sniffer for PIDs: {','.join(map(str, pids))}")
    logger.info(f"Sampling interval: {args.interval} seconds")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Threads: {'MainThread only' if main_thread_only else 'all'}")

    autosave_interval_s = float(args.autosave_interval or 0.0)
    autosave_output = args.autosave_output or args.output
    if autosave_interval_s > 0:
        logger.info(f"Auto-save: every {autosave_interval_s} seconds -> {autosave_output}")

    chrome_tracing_data = {
        "traceEvents": [],
        "displayTimeUnit": "us",
    }

    start_time = time.monotonic()
    next_tick = start_time

    sample_count = 0

    states: Dict[int, Dict[str, Any]] = {}
    for pid in pids:
        states[pid] = {
            "seen_thread_ids": set(),
            "open_stacks": {},
            "last_ts_us": None,
        }

        chrome_tracing_data["traceEvents"].append(
            {
                "name": "process_name",
                "ph": "M",
                "ts": 0,
                "pid": pid,
                "args": {"name": f"python pid {pid}"},
            }
        )
    
    last_autosave_time = time.monotonic()

    try:
        while True:
            now = time.monotonic()
            if args.duration and (now - start_time) >= args.duration:
                break

            for pid in pids:
                ts_us = int((time.monotonic() - start_time) * 1000000)

                pyspy_output = get_stack_traces(pid)
                stack_data = parse_pyspy_output(pyspy_output) if pyspy_output else {"threads": []}
                stack_data = _filter_stack_data(stack_data, main_thread_only)

                st = states[pid]
                events = convert_to_chrome_tracing(
                    pid,
                    stack_data,
                    ts_us,
                    st["seen_thread_ids"],
                    st["open_stacks"],
                )
                chrome_tracing_data["traceEvents"].extend(events)
                sample_count += 1

                st["last_ts_us"] = ts_us

            if autosave_interval_s > 0 and (now - last_autosave_time) >= autosave_interval_s:
                _atomic_write_json(autosave_output, chrome_tracing_data, indent=2)
                last_autosave_time = now

            next_tick += args.interval
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            
    except KeyboardInterrupt:
        logger.info("Stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
    finally:
        end_ts_us = int((time.monotonic() - start_time) * 1000000)
        for pid in pids:
            st = states.get(pid)
            if not st:
                continue
            events = close_open_stacks(pid, end_ts_us, st["open_stacks"])
            chrome_tracing_data["traceEvents"].extend(events)

        _atomic_write_json(args.output, chrome_tracing_data, indent=2)
        
        logger.info(f"Capture completed")
        logger.info(f"Total samples: {sample_count}")
        logger.info(f"Chrome Tracing file saved to: {args.output}")
        logger.info(f"To view, open chrome://tracing in Chrome and load the file")


if __name__ == "__main__":
    main()
