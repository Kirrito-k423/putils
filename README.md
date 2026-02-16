# putils

[English](#english) | [中文](#中文)

---

## English

### Overview

`putils` is a utility library for debugging and profiling distributed AI model training workloads, with specialized support for NPU (Neural Processing Unit) and CUDA environments. The library provides non-invasive instrumentation tools that enable developers to inspect tensor states, monitor memory consumption, profile execution performance, and debug multi-process distributed training workflows without modifying core training code.

### Key Features

- Tensor state inspection and tracking (shape, hash, front elements, etc.)
- Memory consumption monitoring and reporting
- Execution performance profiling
- Distributed training debugging support
- Multiple integrated debugging tools (debug, cache, timer, etc.)
- NPU and CUDA environment support
- Low-intrusive design, minimal changes to training code

### Tech Stack

- Language: Python 100%
- Supported Environments: NPU, CUDA
- Related Tools: VizTracer (tracing), Backward Hook (gradient monitoring)

### Project Structure

```
putils/
├── tools/
│   ├── python_stack_sniffer.py   # Main tool: Stack tracing & visualization
│   └── py_stack_snap.py          # Stack snapshot tool
├── debug.py                      # Debug utilities
├── memory.py                     # Memory monitoring
├── timer.py                      # Timing utilities
├── cache.py                      # Caching utilities
├── device.py                     # Device management
├── profiling.py                  # Profiling utilities
├── perf.py                       # Performance tools
├── pprint.py                     # Pretty printing
├── accuracy.py                   # Accuracy metrics
├── burn.py                       # Burn-in testing
├── dataset/                      # Dataset utilities
├── tests/                        # Unit tests
└── README.md
```

---

## 中文

### 简介

`putils` 是一个专为分布式AI模型训练工作负载设计的调试和性能分析工具库，特别支持NPU（神经处理单元）和CUDA环境。该库提供非侵入式的仪器工具，使开发者能够在不修改核心训练代码的情况下，检查张量状态、监控内存消耗、分析执行性能，以及调试多进程分布式训练工作流。

### 主要功能

- 张量状态检查与追踪（支持shape、hash、前10个元素等信息）
- 内存消耗监控和报告
- 执行性能分析和分析（perf profiling）
- 分布式训练调试支持
- 多种调试工具集成（debug、cache、timer等）
- 支持NPU和CUDA环境
- 低侵入式设计，最轻量化修改训练代码

### 技术栈

- 语言：Python 100%
- 支持环境：NPU、CUDA
- 相关工具：VizTracer（用于追踪）、Backward Hook（用于梯度监控）

### 项目结构

```
putils/
├── tools/
│   ├── python_stack_sniffer.py   # 核心工具：栈追踪与可视化
│   └── py_stack_snap.py          # 栈快照工具
├── debug.py                      # 调试工具
├── memory.py                     # 内存监控
├── timer.py                      # 计时工具
├── cache.py                      # 缓存工具
├── device.py                     # 设备管理
├── profiling.py                 # 性能分析
├── perf.py                       # 性能工具
├── pprint.py                     # 格式化打印
├── accuracy.py                   # 精度指标
├── burn.py                       # 烧机测试
├── dataset/                      # 数据集工具
├── tests/                        # 单元测试
└── README.md
```

---

## python_stack_sniffer.py 详解

### English

`python_stack_sniffer.py` is a powerful stack tracing tool that periodically captures stack traces from Python processes and converts them to Chrome Tracing JSON format for visualization.

### Core Features

1. **Stack Trace Capture**: Uses `py-spy` to capture stack traces from running Python processes
2. **Chrome Tracing Format**: Converts stack traces to Chrome Tracing JSON format for visualization in `chrome://tracing`
3. **Automatic PID Discovery**: Automatically discovers Python processes via `npu-smi info` (NPU environment)
4. **NPU Monitoring**: Records NPU AICore and HBM usage rates during tracing
5. **CPU/Memory Monitoring**: Records system CPU memory usage
6. **Multi-thread Support**: Can capture all threads or MainThread only
7. **Auto-save**: Supports periodic snapshots to prevent data loss during long-running tasks

### Usage Examples

```bash
# Auto-discover PIDs from npu-smi (recommended for NPU environments)
python python_stack_sniffer.py -i 60 -o stack_trace.json --autosave-interval 60 --npu-usage --cpu-mem-usage --all-thread

# Manual PID list
python python_stack_sniffer.py -p 44002,44003 -i 0.2 -d 10 -o trace.json

# With NPU monitoring
python python_stack_sniffer.py -p 12345 -i 0.2 -d 10 -o trace.json --npu-usage

# Auto-save every 10 seconds
python python_stack_sniffer.py -p 1667631 -i 0.1 -o stack_trace.json --autosave-interval 10
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `-p/--pid` | Process ID list (comma-separated or repeatable) |
| `-i/--interval` | Sampling interval in seconds (default: 0.1) |
| `-o/--output` | Output JSON file path |
| `-d/--duration` | Duration to run in seconds |
| `--npu-usage` | Enable NPU usage monitoring (AICore/HBM) |
| `--cpu-mem-usage` | Enable CPU memory monitoring |
| `--all-threads` | Capture all threads (default: MainThread only) |
| `--autosave-interval` | Auto-save interval in seconds |
| `--autosave-snapshot-interval` | Snapshot interval with time tag (default: 7200s) |

### Output

The tool generates a Chrome Tracing compatible JSON file that can be opened in:
- Chrome browser: `chrome://tracing`
- VS Code with Chrome Trace Viewer extension
- Any Chrome Tracing compatible viewer

---

## python_stack_sniffer.py 详解

`python_stack_sniffer.py` 是一款强大的栈追踪工具，可以定期捕获Python进程的堆栈跟踪并将其转换为Chrome Tracing JSON格式进行可视化分析。

### 核心功能

1. **栈跟踪捕获**: 使用 `py-spy` 从正在运行的Python进程中捕获堆栈跟踪
2. **Chrome追踪格式**: 将堆栈跟踪转换为Chrome追踪JSON格式，可在 `chrome://tracing` 中可视化
3. **自动PID发现**: 通过 `npu-smi info` 自动发现Python进程（NPU环境）
4. **NPU监控**: 追踪期间记录NPU AICore和HBM使用率
5. **CPU/内存监控**: 记录系统CPU内存使用情况
6. **多线程支持**: 可捕获所有线程或仅主线程
7. **自动保存**: 支持定期快照，防止长时运行任务数据丢失

### 使用示例

```bash
# 从npu-smi自动发现PID（推荐NPU环境使用）
python python_stack_sniffer.py -i 60 -o stack_trace.json --autosave-interval 60 --npu-usage --cpu-mem-usage --all-thread

# 手动指定PID列表
python python_stack_sniffer.py -p 44002,44003 -i 0.2 -d 10 -o trace.json

# 开启NPU监控
python python_stack_sniffer.py -p 12345 -i 0.2 -d 10 -o trace.json --npu-usage

# 每10秒自动保存
python python_stack_sniffer.py -p 1667631 -i 0.1 -o stack_trace.json --autosave-interval 10
```

### 主要参数

| 参数 | 说明 |
|------|------|
| `-p/--pid` | 进程ID列表（逗号分隔或可重复） |
| `-i/--interval` | 采样间隔（秒，默认0.1） |
| `-o/--output` | 输出JSON文件路径 |
| `-d/--duration` | 运行持续时间（秒） |
| `--npu-usage` | 启用NPU使用率监控（AICore/HBM） |
| `--cpu-mem-usage` | 启用CPU内存监控 |
| `--all-threads` | 捕获所有线程（默认仅MainThread） |
| `--autosave-interval` | 自动保存间隔（秒） |
| `--autosave-snapshot-interval` | 带时间戳的快照间隔（默认7200秒） |

### 输出说明

工具生成兼容Chrome Tracing的JSON文件，可在以下工具中打开：
- Chrome浏览器: `chrome://tracing`
- VS Code配合Chrome Trace Viewer扩展
- 任何兼容Chrome Tracing的查看器

### 技术细节

- 使用 `py-spy dump` 命令获取栈跟踪，支持 `--native` 选项捕获原生栈
- 自动解析py-spy输出，提取线程信息、函数名、文件名、行号
- 转换为Chrome Tracing事件格式（B=begin, E=end, M=metadata, C=counter）
- 支持动态PID发现和移除，自动管理open stacks避免数据错误
- 记录各阶段耗时统计，便于性能分析和问题诊断
