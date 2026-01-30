#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Python Stack Sniffer Tool

This tool periodically captures stack traces from a Python process using py-spy
and converts them to Chrome Tracing JSON format for visualization.

Usage:
    python python_stack_sniffer.py -p <pid> -i <interval> -o <output.json>

中文使用说明：
    
推荐:
    python python_stack_sniffer.py -i 1 -o stack_trace.json --autosave-interval 10 --npu-usage --cpu-mem-usage --all-thread

使用场景示例：
    # 不传pid：自动从npu-smi info抓取所有进程pid
    python python_stack_sniffer.py -i 0.2 -d 10 -o trace.json
    
    # 仍然支持手工pid list
    python python_stack_sniffer.py -p 44002,44003 -i 0.2 -d 10 -o trace.json

    # 自动间隔保存
    python python_stack_sniffer.py -p 1667631  -i 0.1 -o stack_trace.json --autosave-interval 10

    # 开启 NPU 监控（显存 + 利用率）（默认 refresh interval=1 秒，timeout=2 秒）：
    python python_stack_sniffer.py \
      -p 12345 -i 0.2 -d 10 -o trace.json \
      --npu-usage

    # 自定义 npu-smi 刷新与超时：
    python python_stack_sniffer.py \
      -p 12345 -i 0.2 -d 10 -o trace.json \
      --npu-aicore-usage \
      --npu-smi-refresh-interval 1 \
      --npu-smi-timeout 2.5

    # 开启CPU监控
    --cpu-mem-usage：开启 CPU 内存监控
    --cpu-mem-timeout：每次采样 free 的超时（默认 1s）

    # 长跑防止文件过大，及时保存
    --autosave-snapshot-interval：默认 7200 秒（2 小时），设为 0 可关闭
    --autosave-snapshot-output：快照输出基准路径（默认同 --output），实际写入时会自动生成

基础特性：

* 手工指定 pid list”场景生效）：
    * 当 py-spy 采集失败时，如果检测到对应 pid 已不存在（os.kill(pid, 0) -> ProcessLookupError），就把该 pid 从跟踪列表移除，并立刻关闭该 pid 的 open stacks，后续不再跟踪。
    * 如果手工指定的 pid list 全部都失效（运行中被移除到空列表，或启动前就都不存在），会保存最后的 JSON（启动前全失效则保存空 traceEvents），程序自动结束。
* 针对“不指定 pid（走 npu-smi info 自动发现）”场景：
    * npu-smi info PID 列表为空：每 5s 刷新一次；如果连续 180s 都为空，自动结束并在 finally 保存最后 JSON。
    * npu-smi info PID 列表非空：每 30s 刷新一次，把新出现的 PID 加入监控；同时把消失的 PID 从监控中移除并关闭其 open stacks，避免遗漏/脏数据。
    * 运行中如果某个 PID 退出导致采集失败，也会自动停止跟踪该 PID（不再仅限手工 pid list）。
* 记录项包括绝对时间
"""

import argparse
import subprocess
import json
import time
import logging
import re
import os
import tempfile
import select
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


def _format_abs_time(start_wall_time_s: float, ts_us: int) -> str:
    abs_s = float(start_wall_time_s) + (float(ts_us) / 1000000.0)
    return time.strftime('%Y-%m-%d %H:%M', time.localtime(abs_s))



def _make_time_tagged_path(file_path: str, ts: Optional[float] = None) -> str:
    if ts is None:
        ts = time.time()

    tag_base = time.strftime('%Y%m%d_%H%M%S', time.localtime(ts))
    ms = int((ts - int(ts)) * 1000)
    tag = f"{tag_base}_{ms:03d}"

    dir_name = os.path.dirname(file_path)
    base_name = os.path.basename(file_path)
    stem, ext = os.path.splitext(base_name)

    if ext:
        tagged = f"{stem}.{tag}{ext}"
    else:
        tagged = f"{base_name}.{tag}"

    return os.path.join(dir_name, tagged) if dir_name else tagged


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
    parser.add_argument(
        '--autosave-snapshot-interval',
        type=float,
        default=7200.0,
        help='Auto-save snapshot interval in seconds (0 to disable). Snapshot file names include a time tag.',
    )
    parser.add_argument(
        '--autosave-snapshot-output',
        type=str,
        default=None,
        help='Auto-save snapshot base JSON file path (default: same as --output)',
    )
    parser.add_argument('--all-threads', action='store_true', help='Capture all threads (default: only MainThread)')
    parser.add_argument(
        '--npu-usage',
        dest='npu_usage',
        action='store_true',
        help='Sample `npu-smi info -t usages` metrics (Aicore Usage Rate(%) / HBM Usage Rate(%)) and record into output JSON',
    )
    parser.add_argument(
        '--npu-aicore-usage',
        dest='npu_usage',
        action='store_true',
        help='Alias of --npu-usage',
    )
    parser.add_argument(
        '--npu-smi-refresh-interval',
        type=int,
        default=1,
        help='Refresh interval passed to `npu-smi info -t usages -i <N>` (seconds)',
    )
    parser.add_argument(
        '--npu-smi-timeout',
        type=float,
        default=2.0,
        help='Timeout in seconds for each npu-smi sampling attempt',
    )
    parser.add_argument(
        '--cpu-mem-usage',
        action='store_true',
        help='Sample `free -b` memory metrics and record into output JSON',
    )
    parser.add_argument(
        '--cpu-mem-timeout',
        type=float,
        default=1.0,
        help='Timeout in seconds for each free sampling attempt',
    )
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


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


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


def _sample_npu_usage_rates(refresh_interval_s: int, timeout_s: float) -> Optional[Dict[str, int]]:
    cmd = ['npu-smi', 'info', '-t', 'usages', '-i', str(int(refresh_interval_s))]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    deadline = time.monotonic() + max(float(timeout_s or 0.0), 0.1)
    found: Dict[str, int] = {}
    try:
        stdout = proc.stdout
        if stdout is None:
            return None

        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            rlist, _, _ = select.select([stdout], [], [], max(0.0, remaining))
            if not rlist:
                continue

            line = stdout.readline()
            if not line:
                break

            lower = line.lower()
            key: Optional[str] = None
            if 'aicore usage rate' in lower:
                key = 'aicore'
            elif 'hbm usage rate' in lower:
                key = 'hbm'

            if key is None or key in found:
                continue

            m = re.search(r':\s*(\d+)\b', line)
            if not m:
                continue

            found[key] = int(m.group(1))
            if 'aicore' in found and 'hbm' in found:
                break

        return found or None
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    pass


def _sample_cpu_mem_stats(timeout_s: float) -> Optional[Dict[str, int]]:
    timeout = max(float(timeout_s or 0.0), 0.1)
    result = subprocess.run(
        ['free', '-b'],
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout,
    )

    for raw_line in (result.stdout or '').splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if not (line.startswith('Mem:') or line.lower().startswith('mem:')):
            continue

        parts = line.split()
        if len(parts) < 7:
            return None

        total = int(parts[1])
        used = int(parts[2])
        free = int(parts[3])
        shared = int(parts[4])
        buff_cache = int(parts[5])
        available = int(parts[6])

        usage_rate = int(round((used * 100.0) / total)) if total > 0 else 0
        return {
            'total_bytes': total,
            'used_bytes': used,
            'free_bytes': free,
            'shared_bytes': shared,
            'buff_cache_bytes': buff_cache,
            'available_bytes': available,
            'usage_rate': usage_rate,
        }

    return None


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


def get_stack_traces(pid: int) -> Tuple[str, Optional[str]]:
    """Capture stack traces using py-spy dump (text output)."""
    cmds = [
        ['py-spy', 'dump', '--pid', str(pid), "--native"],
        ['py-spy', 'dump', '-p', str(pid)],
    ]

    last_err: Optional[str] = None
    for cmd in cmds:
        stdout, err = _run_pyspy_dump(cmd)
        if stdout is not None:
            return stdout, None
        last_err = err

    return "", last_err


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
    abs_time: str,
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
                    "args": {"name": thread_name, "abs_time": abs_time},
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
                    "args": {"abs_time": abs_time},
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
                        "abs_time": abs_time,
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
                    "args": {"abs_time": abs_time},
                }
            )
        open_stacks.pop(thread_id, None)

    return events


def close_open_stacks(pid: int, ts_us: int, abs_time: str, open_stacks: Dict[int, List[str]]) -> List[Dict[str, Any]]:
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
                    "args": {"abs_time": abs_time},
                }
            )
    open_stacks.clear()
    return events



def main():
    args = parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    manual_pid_list = bool(args.pids)

    if args.pids:
        pids = _normalize_pids(args.pids)
    else:
        pids = _get_pids_from_npu_smi_info()

    if manual_pid_list:
        alive: List[int] = []
        missing: List[int] = []
        for pid in pids:
            if _pid_exists(pid):
                alive.append(pid)
            else:
                missing.append(pid)

        if missing:
            logger.warning(f"Specified PIDs not running and will be skipped: {','.join(map(str, missing))}")

        pids = alive
        if not pids:
            chrome_tracing_data = {
                "traceEvents": [],
                "displayTimeUnit": "us",
            }
            _atomic_write_json(args.output, chrome_tracing_data, indent=2)
            logger.info("All specified PIDs are not running; saved empty trace and exiting")
            logger.info(f"Chrome Tracing file saved to: {args.output}")
            return

    main_thread_only = not args.all_threads

    if pids:
        logger.info(f"Starting Python Stack Sniffer for PIDs: {','.join(map(str, pids))}")
    else:
        logger.info("Starting Python Stack Sniffer with auto PID discovery from `npu-smi info` (no initial PIDs)")
        logger.info("PID discovery: refresh every 5s when empty; exit if empty for 180s; refresh every 30s when non-empty")

    logger.info(f"Sampling interval: {args.interval} seconds")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Threads: {'MainThread only' if main_thread_only else 'all'}")

    autosave_interval_s = float(args.autosave_interval or 0.0)
    autosave_output = args.autosave_output or args.output
    if autosave_interval_s > 0:
        logger.info(f"Auto-save: every {autosave_interval_s} seconds -> {autosave_output}")

    autosave_snapshot_interval_s = float(args.autosave_snapshot_interval or 0.0)
    autosave_snapshot_output = args.autosave_snapshot_output or args.output
    if autosave_snapshot_interval_s > 0:
        logger.info(
            f"Auto-save snapshot: every {autosave_snapshot_interval_s} seconds -> {autosave_snapshot_output} (+time tag)"
        )

    chrome_tracing_data = {
        "traceEvents": [],
        "displayTimeUnit": "us",
    }

    collect_npu_usage = bool(args.npu_usage)
    npu_smi_disabled = False
    npu_trace_pid = 0
    if collect_npu_usage:
        chrome_tracing_data["traceEvents"].append(
            {
                "name": "process_name",
                "ph": "M",
                "ts": 0,
                "pid": npu_trace_pid,
                "args": {"name": "npu"},
            }
        )

    collect_cpu_mem_usage = bool(args.cpu_mem_usage)
    cpu_mem_disabled = False
    cpu_trace_pid = -1
    if collect_cpu_mem_usage:
        chrome_tracing_data["traceEvents"].append(
            {
                "name": "process_name",
                "ph": "M",
                "ts": 0,
                "pid": cpu_trace_pid,
                "args": {"name": "cpu"},
            }
        )

    AUTO_PID_REFRESH_NO_PIDS_S = 5.0
    AUTO_PID_REFRESH_WITH_PIDS_S = 30.0
    AUTO_PID_EMPTY_EXIT_AFTER_S = 180.0

    start_time = time.monotonic()
    start_wall_time = time.time()
    abs_time0 = _format_abs_time(start_wall_time, 0)
    next_tick = start_time

    last_pid_refresh_time = time.monotonic()
    empty_since: Optional[float] = None
    if (not manual_pid_list) and (not pids):
        empty_since = start_time

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
    
    for ev in chrome_tracing_data.get("traceEvents", []):
        ev_args = ev.get("args")
        if not isinstance(ev_args, dict):
            ev_args = {}
            ev["args"] = ev_args
        ev_args.setdefault("abs_time", abs_time0)

    last_autosave_time = time.monotonic()
    last_autosave_snapshot_time = time.monotonic()

    try:
        while True:
            now = time.monotonic()
            if args.duration and (now - start_time) >= args.duration:
                break

            ts_us = int((time.monotonic() - start_time) * 1000000)
            abs_time = _format_abs_time(start_wall_time, ts_us)

            stop_capture = False
            if not manual_pid_list:
                refresh_interval_s = AUTO_PID_REFRESH_WITH_PIDS_S if pids else AUTO_PID_REFRESH_NO_PIDS_S
                if (now - last_pid_refresh_time) >= refresh_interval_s:
                    discovered = _get_pids_from_npu_smi_info()
                    last_pid_refresh_time = now

                    prev_set = set(pids)
                    new_set = set(discovered)

                    removed = sorted(prev_set - new_set)
                    added = sorted(new_set - prev_set)

                    for rpid in removed:
                        st = states.pop(rpid, None)
                        if st:
                            events = close_open_stacks(rpid, ts_us, abs_time, st["open_stacks"])
                            chrome_tracing_data["traceEvents"].extend(events)

                    if removed:
                        logger.info(f"Auto PID refresh: removed {','.join(map(str, removed))}")

                    for apid in added:
                        states[apid] = {
                            "seen_thread_ids": set(),
                            "open_stacks": {},
                            "last_ts_us": None,
                        }
                        chrome_tracing_data["traceEvents"].append(
                            {
                                "name": "process_name",
                                "ph": "M",
                                "ts": 0,
                                "pid": apid,
                                "args": {"name": f"python pid {apid}", "abs_time": abs_time},
                            }
                        )

                    if added:
                        logger.info(f"Auto PID refresh: added {','.join(map(str, added))}")

                    pids = sorted(new_set)

                    if not pids:
                        if empty_since is None:
                            empty_since = now
                        elif (now - empty_since) >= AUTO_PID_EMPTY_EXIT_AFTER_S:
                            logger.info(
                                f"No PIDs from `npu-smi info` for {int(AUTO_PID_EMPTY_EXIT_AFTER_S)}s; stopping capture"
                            )
                            stop_capture = True
                    else:
                        empty_since = None

            if stop_capture:
                break

            if collect_npu_usage and not npu_smi_disabled:
                try:
                    rates = _sample_npu_usage_rates(args.npu_smi_refresh_interval, args.npu_smi_timeout)
                except FileNotFoundError:
                    logger.error('npu-smi not found; disable --npu-usage')
                    npu_smi_disabled = True
                    rates = None

                if rates and ("aicore" in rates):
                    aicore = int(rates["aicore"])
                    chrome_tracing_data["traceEvents"].append(
                        {
                            "name": "npu_aicore_usage_rate",
                            "ph": "C",
                            "ts": ts_us,
                            "pid": npu_trace_pid,
                            "tid": 0,
                            "args": {"aicore": aicore, "abs_time": abs_time},
                        }
                    )
                    chrome_tracing_data.setdefault("npu", {}).setdefault("aicore_usage_rate", []).append(
                        {"ts_us": ts_us, "value": aicore}
                    )

                if rates and ("hbm" in rates):
                    hbm = int(rates["hbm"])
                    chrome_tracing_data["traceEvents"].append(
                        {
                            "name": "npu_hbm_usage_rate",
                            "ph": "C",
                            "ts": ts_us,
                            "pid": npu_trace_pid,
                            "tid": 0,
                            "args": {"hbm": hbm, "abs_time": abs_time},
                        }
                    )
                    chrome_tracing_data.setdefault("npu", {}).setdefault("hbm_usage_rate", []).append(
                        {"ts_us": ts_us, "value": hbm}
                    )

            if collect_cpu_mem_usage and not cpu_mem_disabled:
                try:
                    mem = _sample_cpu_mem_stats(args.cpu_mem_timeout)
                except FileNotFoundError:
                    logger.error('free not found; disable --cpu-mem-usage')
                    cpu_mem_disabled = True
                    mem = None
                except subprocess.TimeoutExpired:
                    mem = None
                except subprocess.CalledProcessError:
                    mem = None

                if mem:
                    chrome_tracing_data["traceEvents"].append(
                        {
                            "name": "cpu_mem",
                            "ph": "C",
                            "ts": ts_us,
                            "pid": cpu_trace_pid,
                            "tid": 0,
                            "args": {
                                "total_bytes": int(mem["total_bytes"]),
                                "used_bytes": int(mem["used_bytes"]),
                                "available_bytes": int(mem["available_bytes"]),
                                "usage_rate": int(mem["usage_rate"]),
                                "abs_time": abs_time,
                            },
                        }
                    )
                    chrome_tracing_data.setdefault("cpu", {}).setdefault("mem", []).append(
                        {"ts_us": ts_us, **mem}
                    )

            for pid in list(pids):
                pyspy_output, pyspy_err = get_stack_traces(pid)

                if not pyspy_output:
                    if pyspy_err:
                        logger.error(f"Failed to capture stack traces: {pyspy_err}")

                    if not _pid_exists(pid):
                        st = states.pop(pid, None)
                        if st:
                            events = close_open_stacks(pid, ts_us, abs_time, st["open_stacks"])
                            chrome_tracing_data["traceEvents"].extend(events)

                        try:
                            pids.remove(pid)
                        except ValueError:
                            pass

                        logger.info(f"PID {pid} not running; stop tracking")

                        if manual_pid_list:
                            if not pids:
                                logger.info("All specified PIDs have exited; stopping capture")
                                stop_capture = True
                                break
                        else:
                            if (not pids) and (empty_since is None):
                                empty_since = now

                    continue

                stack_data = parse_pyspy_output(pyspy_output)
                stack_data = _filter_stack_data(stack_data, main_thread_only)

                st = states[pid]
                events = convert_to_chrome_tracing(
                    pid,
                    stack_data,
                    ts_us,
                    abs_time,
                    st["seen_thread_ids"],
                    st["open_stacks"],
                )
                chrome_tracing_data["traceEvents"].extend(events)
                sample_count += 1

                st["last_ts_us"] = ts_us

            if stop_capture:
                break

            if autosave_interval_s > 0 and (now - last_autosave_time) >= autosave_interval_s:
                _atomic_write_json(autosave_output, chrome_tracing_data, indent=2)
                last_autosave_time = now

            if autosave_snapshot_interval_s > 0 and (now - last_autosave_snapshot_time) >= autosave_snapshot_interval_s:
                snapshot_path = _make_time_tagged_path(autosave_snapshot_output)
                _atomic_write_json(snapshot_path, chrome_tracing_data, indent=2)
                last_autosave_snapshot_time = now

            if (not manual_pid_list) and (not pids):
                next_tick = time.monotonic() + AUTO_PID_REFRESH_NO_PIDS_S
            else:
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
        end_abs_time = _format_abs_time(start_wall_time, end_ts_us)
        for pid in pids:
            st = states.get(pid)
            if not st:
                continue
            events = close_open_stacks(pid, end_ts_us, end_abs_time, st["open_stacks"])
            chrome_tracing_data["traceEvents"].extend(events)

        _atomic_write_json(args.output, chrome_tracing_data, indent=2)
        
        logger.info(f"Capture completed")
        logger.info(f"Total samples: {sample_count}")
        logger.info(f"Chrome Tracing file saved to: {args.output}")
        logger.info(f"To view, open chrome://tracing in Chrome and load the file")


if __name__ == "__main__":
    main()
