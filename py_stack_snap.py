#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
py_stack_snap.py: 获取 NPU 上所有 Python 进程的堆栈快照
"""

import subprocess
import re
import os
import sys
import argparse
import time
from typing import List, Dict, Optional

def get_npu_processes() -> List[Dict[str, str]]:
    """
    通过 npu-smi info 获取正在使用 NPU 的进程列表 (PID, Name)
    Returns:
        List[Dict]: 包含 PID 和进程名的字典列表
    """
    try:
        # 执行 npu-smi info 命令
        result = subprocess.run(
            ["npu-smi", "info"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        output = result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running 'npu-smi info': {e.stderr}", file=sys.stderr)
        return []
    except FileNotFoundError:
        print("Error: 'npu-smi' command not found. Is the NPU driver installed?", file=sys.stderr)
        return []

    processes = []
    lines = output.splitlines()

    for line in lines:
        line = line.strip()
        # 判断一行内是否有4个竖线（'|'），这样分割后会有5个部分
        if line.count('|') == 4:
            # 按 '|' 分割
            parts = [p.strip() for p in line.split('|')]
            # parts 应该是 ['', 'NPU/Chip Info', 'PID', 'Name', 'Memory'] 或类似
            # 因为分割后有5个部分，所以有效内容在索引 1, 2, 3, 4
            # 根据您之前的格式，NPU/Chip 在 1, PID 在 2, Name 在 3, Memory 在 4
            if len(parts) == 5:
                pid_part = parts[2] # PID 部分
                name_part = parts[3] # Name 部分
                pid = pid_part.strip()
                name = name_part.strip()
                # 验证 PID 是否为数字
                if pid.isdigit():
                     # 检查 Name 是否包含 'python' 关键字
                     if 'python' in name.lower():
                         processes.append({"pid": pid, "name": name})
                         print(f"Found Python process on NPU: PID={pid}, Name={name}")
                else:
                    # 如果 PID 不是数字，可能是标题行或其他非数据行，跳过
                    # print(f"Skipping non-data line: {line}") # Debug
                    continue
            else:
                # 分割后的部分数不为5，虽然有4个|，但也可能是格式异常
                # print(f"Line has 4 '|', but parts != 5: {line}") # Debug
                continue
        # else:
        #     # 不是包含4个|的行，跳过
        #     # print(f"Skipping line without 4 '|': {line}") # Debug
        #     continue

    return processes

def get_all_python_processes() -> List[Dict[str, str]]:
    """
    通过 ps 获取所有 Python 进程列表 (PID, Command)
    Returns:
        List[Dict]: 包含 PID 和命令行的字典列表
    """
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid,comm,args"], # comm 是命令名, args 是完整命令行
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        output = result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running 'ps': {e.stderr}", file=sys.stderr)
        return []

    processes = []
    lines = output.strip().split('\n')
    # 跳过标题行
    headers = lines.pop(0).strip()
    for line in lines:
        parts = line.split(None, 2) # 按空白符分割，最多分3部分 (PID, COMM, ARGS)
        if len(parts) >= 2:
            pid_str, comm = parts[0], parts[1]
            if pid_str.isdigit() and 'python' in comm.lower():
                processes.append({"pid": pid_str, "name": comm})
    return processes

def run_py_spy_dump(pid: str) -> Optional[str]:
    """
    运行 py-spy dump --pid {pid} 并返回输出
    Args:
        pid: 进程 PID
    Returns:
        str: py-spy dump 的输出，如果失败则返回 None
    """
    try:
        # print(f"Dumping stack for PID: {pid}", file=sys.stderr) # Debug info
        result = subprocess.run(
            ["py-spy", "dump", "--pid", pid],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30 # 设置超时，防止 py-spy 卡住
        )
        if result.returncode == 0:
            return result.stdout
        else:
            print(f"py-spy dump failed for PID {pid}. STDERR: {result.stderr.strip()}", file=sys.stderr)
            return None
    except subprocess.TimeoutExpired:
        print(f"py-spy dump timed out for PID {pid}", file=sys.stderr)
        return None
    except FileNotFoundError:
        print("Error: 'py-spy' command not found. Please install py-spy.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error running py-spy on PID {pid}: {e}", file=sys.stderr)
        return None

def parse_stack_dump(dump_output: str) -> List[str]:
    """
    解析 py-spy dump 的输出，提取堆栈行。
    返回前10行。
    """
    lines = dump_output.splitlines()
    # 找到第一个以 "Thread" 开头的行作为堆栈开始
    stack_start_idx = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("Thread"):
            stack_start_idx = i
            break

    if stack_start_idx != -1:
        # 返回从堆栈开始到结尾的行，只取前10行
        return lines[stack_start_idx:stack_start_idx+10]
    else:
        # 如果没找到 Thread 行，返回前10行原始输出
        return lines[:10]


def main():
    parser = argparse.ArgumentParser(description="Get stack snapshots of Python processes running on NPU using py-spy.")
    parser.add_argument("-t", "--top", type=int, default=10, help="Number of top stack lines to print per process (default: 10)")
    parser.add_argument("-a", "--all-python", action='store_true', help="Scan all Python processes, not just those reported by npu-smi")
    args = parser.parse_args()

    target_pids = set()

    if args.all_python:
        print("Scanning ALL Python processes on the system...")
        all_python_procs = get_all_python_processes()
        for proc in all_python_procs:
            target_pids.add(proc['pid'])
    else:
        print("Scanning Python processes reported by npu-smi...")
        npu_procs = get_npu_processes()
        for proc in npu_procs:
            target_pids.add(proc['pid'])

    if not target_pids:
        print("No target Python processes found.")
        return

    print(f"Found {len(target_pids)} target Python process(es): {', '.join(sorted(target_pids))}")

    for pid in sorted(target_pids):
        print(f"\n--- Stack dump for PID {pid} ---")
        dump_output = run_py_spy_dump(pid)
        if dump_output:
            stack_lines = parse_stack_dump(dump_output)
            # 只打印请求的前 N 行
            for line in stack_lines[:args.top]:
                print(line)
            if len(stack_lines) > args.top:
                 print(f"... (truncated to {args.top} lines)")
        else:
            print(f"Failed to get stack for PID {pid}.")


if __name__ == "__main__":
    main()
