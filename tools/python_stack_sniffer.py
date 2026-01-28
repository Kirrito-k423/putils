#!/usr/bin/env python3
"""
Python Stack Sniffer Tool

This tool periodically captures stack traces from a Python process using py-spy
and converts them to Chrome Tracing JSON format for visualization.

Usage:
    python python_stack_sniffer.py -p <pid> -i <interval> -o <output.json>
    python python_stack_sniffer.py -p 1549992  -i 0.2 -d 10 -o trace.json
"""

import argparse
import subprocess
import json
import time
import logging
import re
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Python Stack Sniffer for Chrome Tracing')
    parser.add_argument('-p', '--pid', type=int, required=True, help='Process ID to monitor')
    parser.add_argument('-i', '--interval', type=float, default=0.1, help='Sampling interval in seconds')
    parser.add_argument('-o', '--output', type=str, default='stack_trace.json', help='Output JSON file path')
    parser.add_argument('-d', '--duration', type=float, help='Duration to run in seconds (optional)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
    return parser.parse_args()


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
    interval_us: int,
    seen_thread_ids: set,
) -> List[Dict[str, Any]]:
    """Convert parsed stack traces to Chrome Tracing events."""
    events: List[Dict[str, Any]] = []

    interval_us = max(1, int(interval_us))

    for thread in stack_data.get("threads", []):
        thread_id = int(thread.get("id", 0) or 0)
        thread_name = thread.get("name") or f"Thread {thread_id}"
        stack = thread.get("stack") or []

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

        if not stack:
            continue

        frames = list(reversed(stack))
        n = len(frames)
        if n == 0:
            continue

        step = max(1, interval_us // max(1, n))
        taper_step = max(1, interval_us // 200)
        max_step = max(1, (interval_us - 1) // max(1, n - 1))
        taper_step = min(taper_step, max_step)

        end_offsets = [max(1, interval_us - i * taper_step) for i in range(n)]
        for i in range(1, n):
            if end_offsets[i] >= end_offsets[i - 1]:
                end_offsets[i] = max(1, end_offsets[i - 1] - 1)

        names: List[str] = []
        for frame in frames:
            fn = frame.get("name", "unknown")
            file_name = frame.get("filename", "unknown")
            line_num = frame.get("lineno", 0)
            names.append(f"{fn} ({file_name}:{line_num})")

        for frame_name, frame in zip(names, frames):
            events.append(
                {
                    "name": frame_name,
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

        for i in range(n - 1, -1, -1):
            events.append(
                {
                    "name": names[i],
                    "ph": "E",
                    "ts": ts_us + end_offsets[i],
                    "tid": thread_id,
                    "pid": pid,
                }
            )

    return events


def main():
    args = parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info(f"Starting Python Stack Sniffer for PID {args.pid}")
    logger.info(f"Sampling interval: {args.interval} seconds")
    logger.info(f"Output file: {args.output}")
    
    chrome_tracing_data = {
        "traceEvents": [],
        "displayTimeUnit": "us",
    }

    start_time = time.monotonic()
    next_tick = start_time

    sample_count = 0
    seen_thread_ids: set = set()

    last_ts_us: Optional[int] = None
    last_stack_data: Optional[Dict[str, Any]] = None
    last_interval_us = max(1, int(args.interval * 1000000))

    chrome_tracing_data["traceEvents"].append(
        {
            "name": "process_name",
            "ph": "M",
            "ts": 0,
            "pid": args.pid,
            "args": {"name": f"python pid {args.pid}"},
        }
    )
    
    try:
        while True:
            now = time.monotonic()
            if args.duration and (now - start_time) >= args.duration:
                break

            ts_us = int((now - start_time) * 1000000)

            pyspy_output = get_stack_traces(args.pid)
            stack_data = parse_pyspy_output(pyspy_output) if pyspy_output else {"threads": []}

            if last_ts_us is not None and last_stack_data is not None:
                interval_us = max(1, ts_us - last_ts_us)
                last_interval_us = interval_us
                events = convert_to_chrome_tracing(args.pid, last_stack_data, last_ts_us, interval_us, seen_thread_ids)
                chrome_tracing_data["traceEvents"].extend(events)
                sample_count += 1

            last_ts_us = ts_us
            last_stack_data = stack_data

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
        if last_ts_us is not None and last_stack_data is not None:
            events = convert_to_chrome_tracing(args.pid, last_stack_data, last_ts_us, last_interval_us, seen_thread_ids)
            chrome_tracing_data["traceEvents"].extend(events)
            sample_count += 1

        # Write the Chrome Tracing JSON file
        with open(args.output, 'w') as f:
            json.dump(chrome_tracing_data, f, indent=2)
        
        logger.info(f"Capture completed")
        logger.info(f"Total samples: {sample_count}")
        logger.info(f"Chrome Tracing file saved to: {args.output}")
        logger.info(f"To view, open chrome://tracing in Chrome and load the file")


if __name__ == "__main__":
    main()
