#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Python Stack Sniffer Tool

This tool periodically captures stack traces from a Python process using py-spy
and converts them to Chrome Tracing JSON format for visualization.

Usage:
    python python_stack_sniffer.py -p <pid> -i <interval> -o <output.json>

中文使用说明：
    
推荐:（全量PID采集，过于频繁会导致性能劣化3+倍）
    python python_stack_sniffer.py -i 60 -o stack_trace.json --autosave-interval 60 --npu-usage --cpu-mem-usage --all-thread

调试模式（抓取所有NPU进程，包括小内存进程如rayWorkerDict）：
    python python_stack_sniffer.py -i 2 -o stack_trace.json --autosave-interval 10 --npu-usage --cpu-mem-usage --all-thread --debug-pid-discovery

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
    * npu-smi info PID 列表 查询时，默认 --min-hbm-usage-mb (5000MB) 以下的任务为过滤项，不纳入监控。
    * 使用 --debug-pid-discovery 可禁用 HBM 过滤，抓取所有 NPU 进程（适合调试 rayWorkerDict 等小内存进程）。
    * npu-smi info PID 列表为空：每 5s 刷新一次；如果连续 180s 都为空，自动结束并在 finally 保存最后 JSON。
    * npu-smi info PID 列表非空：每 30s 刷新一次，把新出现的 PID 加入监控；同时把消失的 PID 从监控中移除并关闭其 open stacks，避免遗漏/脏数据。
    * 运行中如果某个 PID 退出导致采集失败，也会自动停止跟踪该 PID（不再仅限手工 pid list）。
* 记录项包括绝对时间
* 还会记录采样的各阶段和命令的耗时，最终汇总信息打印
"""

import argparse
import subprocess
import json
import time
import logging
import re
import os
import shutil
import tempfile
import select
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _ensure_output_log_file_handler(output_path: str) -> str:
    log_path = f"{output_path}.log"
    abs_log_path = os.path.abspath(log_path)

    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            try:
                if os.path.abspath(getattr(h, "baseFilename", "")) == abs_log_path:
                    return log_path
            except Exception:
                pass

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    return log_path


def _timing_add(stats: Dict[str, Dict[str, float]], name: str, dt_s: float) -> None:
    if dt_s < 0:
        return
    st = stats.get(name)
    if st is None:
        stats[name] = {
            "count": 1.0,
            "total_s": float(dt_s),
            "max_s": float(dt_s),
            "min_s": float(dt_s),
        }
        return

    st["count"] = float(st.get("count", 0.0)) + 1.0
    st["total_s"] = float(st.get("total_s", 0.0)) + float(dt_s)
    st["max_s"] = max(float(st.get("max_s", 0.0)), float(dt_s))
    st["min_s"] = min(float(st.get("min_s", dt_s)), float(dt_s))


def _format_timing_summary(stats: Dict[str, Dict[str, float]]) -> List[str]:
    lines: List[str] = []
    for name in sorted(stats.keys()):
        st = stats[name]
        count = int(st.get("count", 0.0))
        total_s = float(st.get("total_s", 0.0))
        if count <= 0:
            continue
        avg_ms = (total_s * 1000.0) / float(count)
        max_ms = float(st.get("max_s", 0.0)) * 1000.0
        min_ms = float(st.get("min_s", 0.0)) * 1000.0
        lines.append(
            f"{name}: count={count} avg={avg_ms:.2f}ms min={min_ms:.2f}ms max={max_ms:.2f}ms total={total_s:.3f}s"
        )
    return lines


def _atomic_write_json(file_path: str, data: Any, indent: int = 2) -> None:
    dir_name = os.path.dirname(os.path.abspath(file_path)) or "."
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=dir_name, delete=False
        ) as tf:
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
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(abs_s))


def _make_time_tagged_path(file_path: str, ts: Optional[float] = None) -> str:
    if ts is None:
        ts = time.time()

    tag_base = time.strftime("%Y%m%d_%H%M%S", time.localtime(ts))
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
    parser = argparse.ArgumentParser(
        description="Python Stack Sniffer for Chrome Tracing"
    )
    parser.add_argument(
        "-p",
        "--pid",
        dest="pids",
        action="append",
        required=False,
        help="Process ID list. Repeatable (-p 1 -p 2) or comma-separated (-p 1,2). If omitted, auto-discover via `npu-smi info` (NPU) or `nvidia-smi` (GPU).",
    )
    parser.add_argument(
        "-i", "--interval", type=float, default=0.1, help="Sampling interval in seconds"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="stack_trace.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "-d", "--duration", type=float, help="Duration to run in seconds (optional)"
    )
    parser.add_argument(
        "--autosave-interval",
        type=float,
        default=0.0,
        help="Auto-save interval in seconds (0 to disable)",
    )
    parser.add_argument(
        "--autosave-output",
        type=str,
        default=None,
        help="Auto-save JSON file path (default: same as --output)",
    )
    parser.add_argument(
        "--autosave-snapshot-interval",
        type=float,
        default=7200.0,
        help="Auto-save snapshot interval in seconds (0 to disable). Snapshot file names include a time tag.",
    )
    parser.add_argument(
        "--autosave-snapshot-output",
        type=str,
        default=None,
        help="Auto-save snapshot base JSON file path (default: same as --output)",
    )
    parser.add_argument(
        "--all-threads",
        action="store_true",
        help="Capture all threads (default: only MainThread)",
    )
    parser.add_argument(
        "--npu-usage",
        dest="npu_usage",
        action="store_true",
        help="Sample `npu-smi info -t usages` metrics (Aicore Usage Rate(%%) / HBM Usage Rate(%%)) and record into output JSON",
    )
    parser.add_argument(
        "--npu-aicore-usage",
        dest="npu_usage",
        action="store_true",
        help="Alias of --npu-usage",
    )
    parser.add_argument(
        "--npu-smi-refresh-interval",
        type=int,
        default=1,
        help="Refresh interval passed to `npu-smi info -t usages -i <N>` (seconds)",
    )
    parser.add_argument(
        "--npu-smi-timeout",
        type=float,
        default=2.0,
        help="Timeout in seconds for each npu-smi sampling attempt",
    )
    parser.add_argument(
        "--cpu-mem-usage",
        action="store_true",
        help="Sample `free -b` memory metrics and record into output JSON",
    )
    parser.add_argument(
        "--cpu-mem-timeout",
        type=float,
        default=1.0,
        help="Timeout in seconds for each free sampling attempt",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--debug-pid-discovery",
        action="store_true",
        help="Debug mode: include all PIDs from npu-smi info regardless of HBM usage (sets min HBM to 0)",
    )
    parser.add_argument(
        "--min-hbm-usage-mb",
        type=int,
        default=5000,
        help="Minimum HBM usage (MB) to include PID from npu-smi info. Default: 5000. Set to 0 to include all.",
    )
    parser.add_argument(
        "--gpu-usage",
        action="store_true",
        help="Sample `nvidia-smi` metrics (utilization.gpu / utilization.memory) and record into output JSON",
    )
    parser.add_argument(
        "--gpu-smi-timeout",
        type=float,
        default=2.0,
        help="Timeout in seconds for each nvidia-smi sampling attempt",
    )
    parser.add_argument(
        "--min-gpu-mem-usage-mb",
        type=int,
        default=0,
        help="Minimum GPU memory usage (MB) to include PID from nvidia-smi. Default: 0.",
    )
    return parser.parse_args()


def _normalize_pids(pid_args: List[str]) -> List[int]:
    pids: List[int] = []
    seen = set()

    for item in pid_args:
        for part in item.split(","):
            part = part.strip()
            if not part:
                continue
            pid = int(part)
            if pid not in seen:
                pids.append(pid)
                seen.add(pid)

    if not pids:
        raise ValueError("empty pid list")

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


def _filter_stack_data(
    stack_data: Dict[str, Any], main_thread_only: bool
) -> Dict[str, Any]:
    if not main_thread_only:
        return stack_data

    threads = stack_data.get("threads") or []
    main_threads = [t for t in threads if (t.get("name") or "") == "MainThread"]
    if main_threads:
        return {"threads": main_threads}

    return stack_data


def _get_pids_from_npu_smi_info(min_hbm_usage_mb: int = 5000) -> List[int]:
    try:
        result = subprocess.run(
            ["npu-smi", "info"],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        logger.error("npu-smi not found; please provide -p/--pid explicitly")
        return []
    except subprocess.CalledProcessError as e:
        logger.error(f"npu-smi info failed: {(e.stderr or '').strip()}")
        return []

    def _find_col_index(cols: List[str], patterns: List[str]) -> Optional[int]:
        for i, c in enumerate(cols):
            for p in patterns:
                if re.search(p, c):
                    return i
        return None

    def _parse_int(cell: str) -> Optional[int]:
        m = re.search(r"(\d+)", cell)
        return int(m.group(1)) if m else None

    pids: List[int] = []
    seen = set()
    in_process_table = False
    pid_col: Optional[int] = None
    mem_col: Optional[int] = None

    for raw_line in (result.stdout or "").splitlines():
        line = raw_line.rstrip("\n")
        lower = line.lower()
        if re.search(r"process\s*id", lower) and re.search(r"process\s*name", lower):
            in_process_table = True
            if "|" in line:
                header_cols = [
                    p.strip().lower() for p in line.strip().strip("|").split("|")
                ]
                pid_col = _find_col_index(header_cols, [r"\bpid\b", r"process\s*id"])
                mem_col = _find_col_index(
                    header_cols,
                    [
                        r"hbm.*usage",
                        r"memory.*usage",
                        r"\busage\b.*\bmb\b",
                        r"\bmb\b.*\busage\b",
                    ],
                )
            continue
        if not in_process_table:
            continue
        if line.startswith("+"):
            continue
        if not line.strip().startswith("|"):
            continue

        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) < 3:
            continue

        pid_str: Optional[str] = None
        if (
            pid_col is not None
            and 0 <= pid_col < len(parts)
            and parts[pid_col].isdigit()
        ):
            pid_str = parts[pid_col]
        else:
            for token in parts[1:]:
                if token.isdigit():
                    pid_str = token
                    break
        if pid_str is None:
            continue

        mem_mb: Optional[int] = None
        if mem_col is not None and 0 <= mem_col < len(parts):
            mem_mb = _parse_int(parts[mem_col])
        elif parts:
            mem_mb = _parse_int(parts[-1])

        if min_hbm_usage_mb > 0 and mem_mb is not None and mem_mb < min_hbm_usage_mb:
            continue

        pid = int(pid_str)
        if pid not in seen:
            pids.append(pid)
            seen.add(pid)

    return pids


def _detect_accelerator_backend() -> str:
    if shutil.which("npu-smi"):
        return "npu"
    if shutil.which("nvidia-smi"):
        return "gpu"
    return "none"


def _get_pids_from_nvidia_smi_info(min_gpu_mem_usage_mb: int = 0) -> List[int]:
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        logger.error("nvidia-smi not found; please provide -p/--pid explicitly")
        return []
    except subprocess.CalledProcessError as e:
        logger.error(f"nvidia-smi query compute apps failed: {(e.stderr or '').strip()}")
        return []

    pids: List[int] = []
    seen = set()
    for raw_line in (result.stdout or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = [p.strip() for p in line.split(",")]
        if not parts:
            continue

        pid_str = parts[0]
        if not pid_str.isdigit():
            continue

        mem_mb: Optional[int] = None
        if len(parts) >= 2:
            m = re.search(r"(\d+)", parts[1])
            if m:
                mem_mb = int(m.group(1))

        if (
            min_gpu_mem_usage_mb > 0
            and mem_mb is not None
            and mem_mb < min_gpu_mem_usage_mb
        ):
            continue

        pid = int(pid_str)
        if pid not in seen:
            pids.append(pid)
            seen.add(pid)

    return pids


def _sample_npu_usage_rates(
    refresh_interval_s: int, timeout_s: float
) -> Optional[Dict[str, int]]:
    cmd = ["npu-smi", "info", "-t", "usages", "-i", str(int(refresh_interval_s))]

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
            if "aicore usage rate" in lower:
                key = "aicore"
            elif "hbm usage rate" in lower:
                key = "hbm"

            if key is None or key in found:
                continue

            m = re.search(r":\s*(\d+)\b", line)
            if not m:
                continue

            found[key] = int(m.group(1))
            if "aicore" in found and "hbm" in found:
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


def _sample_gpu_usage_rates(timeout_s: float) -> Optional[Dict[str, int]]:
    timeout = max(float(timeout_s or 0.0), 0.1)
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,utilization.memory",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout,
    )

    gpu_utils: List[int] = []
    mem_utils: List[int] = []
    for raw_line in (result.stdout or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue

        m_gpu = re.search(r"(\d+)", parts[0])
        m_mem = re.search(r"(\d+)", parts[1])
        if not m_gpu or not m_mem:
            continue

        gpu_utils.append(int(m_gpu.group(1)))
        mem_utils.append(int(m_mem.group(1)))

    if not gpu_utils or not mem_utils:
        return None

    return {
        "gpu": max(gpu_utils),
        "mem": max(mem_utils),
    }


def _sample_cpu_mem_stats(timeout_s: float) -> Optional[Dict[str, int]]:
    timeout = max(float(timeout_s or 0.0), 0.1)
    result = subprocess.run(
        ["free", "-b"],
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout,
    )

    for raw_line in (result.stdout or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if not (line.startswith("Mem:") or line.lower().startswith("mem:")):
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
            "total_bytes": total,
            "used_bytes": used,
            "free_bytes": free,
            "shared_bytes": shared,
            "buff_cache_bytes": buff_cache,
            "available_bytes": available,
            "usage_rate": usage_rate,
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
        ["py-spy", "dump", "--pid", str(pid), "--native"],
        ["py-spy", "dump", "-p", str(pid)],
    ]

    last_err: Optional[str] = None
    for cmd in cmds:
        stdout, err = _run_pyspy_dump(cmd)
        if stdout is not None:
            return stdout, None
        last_err = err

    return "", last_err


_THREAD_HEADER_RE = re.compile(r"^Thread\s+(?P<rest>.+?)\s*$")
_FILE_FRAME_RE = re.compile(
    r'^\s*File\s+"(?P<file>[^"]+)",\s+line\s+(?P<line>\d+),\s+in\s+(?P<func>.+?)\s*$'
)
_SIMPLE_FRAME_RE = re.compile(
    r"^\s*(?P<func>.+?)\s+\((?P<file>.+?):(?P<line>\d+)\)\s*$"
)
_NATIVE_FRAME_RE = re.compile(r"^\s*\[native\]\s+(?P<func>.+?)\s*$")


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
            current["stack"].append(
                {"name": m_nf.group("func"), "filename": "[native]", "lineno": 0}
            )
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


def close_open_stacks(
    pid: int, ts_us: int, abs_time: str, open_stacks: Dict[int, List[str]]
) -> List[Dict[str, Any]]:
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

    log_path = _ensure_output_log_file_handler(args.output)
    logger.info(f"Timing log file: {log_path}")

    timing_stats: Dict[str, Dict[str, float]] = {}

    manual_pid_list = bool(args.pids)
    accelerator_backend = _detect_accelerator_backend()
    if accelerator_backend == "npu":
        logger.info("Detected accelerator backend: NPU (npu-smi)")
    elif accelerator_backend == "gpu":
        logger.info("Detected accelerator backend: GPU (nvidia-smi)")
    else:
        logger.info("Detected accelerator backend: none")

    if args.pids:
        pids = _normalize_pids(args.pids)
    else:
        if accelerator_backend == "npu":
            min_hbm = 0 if args.debug_pid_discovery else args.min_hbm_usage_mb
            pids = _get_pids_from_npu_smi_info(min_hbm)
        elif accelerator_backend == "gpu":
            min_gpu_mem = 0 if args.debug_pid_discovery else args.min_gpu_mem_usage_mb
            pids = _get_pids_from_nvidia_smi_info(min_gpu_mem)
        else:
            pids = []

    if manual_pid_list:
        alive: List[int] = []
        missing: List[int] = []
        for pid in pids:
            if _pid_exists(pid):
                alive.append(pid)
            else:
                missing.append(pid)

        if missing:
            logger.warning(
                f"Specified PIDs not running and will be skipped: {','.join(map(str, missing))}"
            )

        pids = alive
        if not pids:
            chrome_tracing_data = {
                "traceEvents": [],
                "displayTimeUnit": "us",
            }
            _atomic_write_json(args.output, chrome_tracing_data, indent=2)
            logger.info(
                "All specified PIDs are not running; saved empty trace and exiting"
            )
            logger.info(f"Chrome Tracing file saved to: {args.output}")
            return

    main_thread_only = not args.all_threads

    if pids:
        logger.info(
            f"Starting Python Stack Sniffer for PIDs: {','.join(map(str, pids))}"
        )
    else:
        if accelerator_backend == "npu":
            logger.info(
                "Starting Python Stack Sniffer with auto PID discovery from `npu-smi info` (no initial PIDs)"
            )
        elif accelerator_backend == "gpu":
            logger.info(
                "Starting Python Stack Sniffer with auto PID discovery from `nvidia-smi` (no initial PIDs)"
            )
        else:
            logger.info(
                "Starting Python Stack Sniffer without detected accelerator tool (`npu-smi`/`nvidia-smi` not found)"
            )
        logger.info(
            "PID discovery: refresh every 5s when empty; exit if empty for 180s; refresh every 30s when non-empty"
        )

    logger.info(f"Sampling interval: {args.interval} seconds")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Threads: {'MainThread only' if main_thread_only else 'all'}")

    autosave_interval_s = float(args.autosave_interval or 0.0)
    autosave_output = args.autosave_output or args.output
    if autosave_interval_s > 0:
        logger.info(
            f"Auto-save: every {autosave_interval_s} seconds -> {autosave_output}"
        )

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
    if collect_npu_usage and accelerator_backend != "npu":
        logger.warning(
            "--npu-usage is enabled but NPU backend is not detected; disabling NPU usage sampling"
        )
        collect_npu_usage = False

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

    collect_gpu_usage = bool(args.gpu_usage)
    if collect_gpu_usage and accelerator_backend != "gpu":
        logger.warning(
            "--gpu-usage is enabled but GPU backend is not detected; disabling GPU usage sampling"
        )
        collect_gpu_usage = False

    gpu_smi_disabled = False
    gpu_trace_pid = 1
    if collect_gpu_usage:
        chrome_tracing_data["traceEvents"].append(
            {
                "name": "process_name",
                "ph": "M",
                "ts": 0,
                "pid": gpu_trace_pid,
                "args": {"name": "gpu"},
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
            iter_t0 = time.monotonic()

            now = time.monotonic()
            if args.duration and (now - start_time) >= args.duration:
                break

            ts_us = int((time.monotonic() - start_time) * 1000000)
            abs_time = _format_abs_time(start_wall_time, ts_us)

            stop_capture = False
            if not manual_pid_list:
                refresh_interval_s = (
                    AUTO_PID_REFRESH_WITH_PIDS_S if pids else AUTO_PID_REFRESH_NO_PIDS_S
                )
                if (now - last_pid_refresh_time) >= refresh_interval_s:
                    t0 = time.monotonic()
                    if accelerator_backend == "npu":
                        min_hbm = 0 if args.debug_pid_discovery else args.min_hbm_usage_mb
                        discovered = _get_pids_from_npu_smi_info(min_hbm)
                        timing_name = "npu_smi_info_discover"
                        discover_source = "npu-smi info"
                    elif accelerator_backend == "gpu":
                        min_gpu_mem = (
                            0 if args.debug_pid_discovery else args.min_gpu_mem_usage_mb
                        )
                        discovered = _get_pids_from_nvidia_smi_info(min_gpu_mem)
                        timing_name = "nvidia_smi_discover"
                        discover_source = "nvidia-smi"
                    else:
                        discovered = []
                        timing_name = "auto_discover_none"
                        discover_source = "none"

                    dt = time.monotonic() - t0
                    _timing_add(timing_stats, timing_name, dt)
                    logger.info(
                        f"timing {discover_source} discover: {dt * 1000.0:.2f}ms pids={len(discovered)}"
                    )
                    last_pid_refresh_time = now

                    prev_set = set(pids)
                    new_set = set(discovered)

                    removed = sorted(prev_set - new_set)
                    added = sorted(new_set - prev_set)

                    for rpid in removed:
                        st = states.pop(rpid, None)
                        if st:
                            events = close_open_stacks(
                                rpid, ts_us, abs_time, st["open_stacks"]
                            )
                            chrome_tracing_data["traceEvents"].extend(events)

                    if removed:
                        logger.info(
                            f"Auto PID refresh: removed {','.join(map(str, removed))}"
                        )

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
                                "args": {
                                    "name": f"python pid {apid}",
                                    "abs_time": abs_time,
                                },
                            }
                        )

                    if added:
                        logger.info(
                            f"Auto PID refresh: added {','.join(map(str, added))}"
                        )

                    pids = sorted(new_set)

                    if not pids:
                        if empty_since is None:
                            empty_since = now
                        elif (now - empty_since) >= AUTO_PID_EMPTY_EXIT_AFTER_S:
                            logger.info(
                                f"No PIDs from auto discovery for {int(AUTO_PID_EMPTY_EXIT_AFTER_S)}s; stopping capture"
                            )
                            stop_capture = True
                    else:
                        empty_since = None

            if stop_capture:
                break

            if collect_npu_usage and not npu_smi_disabled:
                try:
                    t0 = time.monotonic()
                    rates = _sample_npu_usage_rates(
                        args.npu_smi_refresh_interval, args.npu_smi_timeout
                    )
                    _timing_add(
                        timing_stats, "npu_smi_usages_sample", time.monotonic() - t0
                    )
                except FileNotFoundError:
                    logger.error("npu-smi not found; disable --npu-usage")
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
                    chrome_tracing_data.setdefault("npu", {}).setdefault(
                        "aicore_usage_rate", []
                    ).append({"ts_us": ts_us, "value": aicore})

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
                    chrome_tracing_data.setdefault("npu", {}).setdefault(
                        "hbm_usage_rate", []
                    ).append({"ts_us": ts_us, "value": hbm})

            if collect_gpu_usage and not gpu_smi_disabled:
                try:
                    t0 = time.monotonic()
                    rates = _sample_gpu_usage_rates(args.gpu_smi_timeout)
                    _timing_add(
                        timing_stats, "nvidia_smi_usages_sample", time.monotonic() - t0
                    )
                except FileNotFoundError:
                    logger.error("nvidia-smi not found; disable --gpu-usage")
                    gpu_smi_disabled = True
                    rates = None
                except subprocess.TimeoutExpired:
                    rates = None
                except subprocess.CalledProcessError:
                    rates = None

                if rates and ("gpu" in rates):
                    gpu = int(rates["gpu"])
                    chrome_tracing_data["traceEvents"].append(
                        {
                            "name": "gpu_utilization_rate",
                            "ph": "C",
                            "ts": ts_us,
                            "pid": gpu_trace_pid,
                            "tid": 0,
                            "args": {"gpu": gpu, "abs_time": abs_time},
                        }
                    )
                    chrome_tracing_data.setdefault("gpu", {}).setdefault(
                        "utilization_rate", []
                    ).append({"ts_us": ts_us, "value": gpu})

                if rates and ("mem" in rates):
                    mem = int(rates["mem"])
                    chrome_tracing_data["traceEvents"].append(
                        {
                            "name": "gpu_memory_usage_rate",
                            "ph": "C",
                            "ts": ts_us,
                            "pid": gpu_trace_pid,
                            "tid": 0,
                            "args": {"memory": mem, "abs_time": abs_time},
                        }
                    )
                    chrome_tracing_data.setdefault("gpu", {}).setdefault(
                        "memory_usage_rate", []
                    ).append({"ts_us": ts_us, "value": mem})

            if collect_cpu_mem_usage and not cpu_mem_disabled:
                try:
                    t0 = time.monotonic()
                    mem = _sample_cpu_mem_stats(args.cpu_mem_timeout)
                    _timing_add(timing_stats, "free_mem_sample", time.monotonic() - t0)
                except FileNotFoundError:
                    logger.error("free not found; disable --cpu-mem-usage")
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
                    chrome_tracing_data.setdefault("cpu", {}).setdefault(
                        "mem", []
                    ).append({"ts_us": ts_us, **mem})

            for pid in list(pids):
                t0 = time.monotonic()
                pyspy_output, pyspy_err = get_stack_traces(pid)
                dt = time.monotonic() - t0
                _timing_add(timing_stats, "pyspy_dump", dt)
                if dt >= 1.0 or (not pyspy_output):
                    logger.info(
                        f"timing py-spy dump: pid={pid} {dt * 1000.0:.2f}ms ok={bool(pyspy_output)}"
                    )

                if not pyspy_output:
                    if pyspy_err:
                        logger.error(f"Failed to capture stack traces: {pyspy_err}")

                    if not _pid_exists(pid):
                        st = states.pop(pid, None)
                        if st:
                            events = close_open_stacks(
                                pid, ts_us, abs_time, st["open_stacks"]
                            )
                            chrome_tracing_data["traceEvents"].extend(events)

                        try:
                            pids.remove(pid)
                        except ValueError:
                            pass

                        logger.info(f"PID {pid} not running; stop tracking")

                        if manual_pid_list:
                            if not pids:
                                logger.info(
                                    "All specified PIDs have exited; stopping capture"
                                )
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

            if (
                autosave_interval_s > 0
                and (now - last_autosave_time) >= autosave_interval_s
            ):
                t0 = time.monotonic()
                _atomic_write_json(autosave_output, chrome_tracing_data, indent=2)
                _timing_add(timing_stats, "autosave_write_json", time.monotonic() - t0)
                last_autosave_time = now

            if (
                autosave_snapshot_interval_s > 0
                and (now - last_autosave_snapshot_time) >= autosave_snapshot_interval_s
            ):
                snapshot_path = _make_time_tagged_path(autosave_snapshot_output)
                t0 = time.monotonic()
                _atomic_write_json(snapshot_path, chrome_tracing_data, indent=2)
                _timing_add(
                    timing_stats, "autosave_snapshot_write_json", time.monotonic() - t0
                )
                last_autosave_snapshot_time = now

            if (not manual_pid_list) and (not pids):
                next_tick = time.monotonic() + AUTO_PID_REFRESH_NO_PIDS_S
            else:
                next_tick += args.interval

            iter_dt = time.monotonic() - iter_t0
            _timing_add(timing_stats, "loop_total", iter_dt)
            if iter_dt >= 2.0:
                logger.info(
                    f"timing loop_total: {iter_dt * 1000.0:.2f}ms pids={len(pids)}"
                )

            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                _timing_add(timing_stats, "loop_sleep", sleep_s)
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

        t0 = time.monotonic()
        _atomic_write_json(args.output, chrome_tracing_data, indent=2)
        _timing_add(timing_stats, "final_write_json", time.monotonic() - t0)

        logger.info(f"Capture completed")
        logger.info(f"Total samples: {sample_count}")
        logger.info(f"Chrome Tracing file saved to: {args.output}")

        summary_lines = _format_timing_summary(timing_stats)
        if summary_lines:
            logger.info("Timing summary:")
            for line in summary_lines:
                logger.info(line)

        logger.info(f"To view, open chrome://tracing in Chrome and load the file")


if __name__ == "__main__":
    main()
