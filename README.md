# putils

putils is a utility library for debugging and profiling distributed AI model training workloads, with specialized support for NPU (Neural Processing Unit) and CUDA environments. The library provides non-invasive instrumentation tools that enable developers to inspect tensor states, monitor memory consumption, profile execution performance, and debug multi-process distributed training workflows without modifying core training code.

putils是一个专为分布式AI模型训练工作负载设计的调试和性能分析工具库，特别支持NPU（神经处理单元）和CUDA环境。该库提供非侵入式的仪器工具，使开发者能够在不修改核心训练代码的情况下，检查张量状态、监控内存消耗、分析执行性能，以及调试多进程分布式训练工作流。

### 主要功能点

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




## Next

下一个工具： 将我关心的数据dump（shape，hash，front10， memallocate）下来，在py stack sniffer里集成可视化。
