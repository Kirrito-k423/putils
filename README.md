# putils

dev ai model utils

## requirements

```
pip install portalocker debugpy
```


## Debug

针对VeRL，DeepSpeed, Vllm, 等多进程无法使用vscode_debug和直接breakpoint的情况, 可以尝试sleep attach的方法

```
# export DEBUG_NPU_WORKER=1
import time
print(f"DEBUGGING PAUSE: Worker PID {os.getpid()} is waiting for debugger attachment.")

# 这是一个无限循环，直到您手动修改变量或附加调试器退出
# 请将此行代码添加在 npu_worker.py 中：
# 例如：NPUWorkerProc 里的 worker_main 函数的最前面
if "DEBUG_NPU_WORKER" in os.environ:
    # 打印 PID，这样您就知道要附加哪个进程
    print(f"Worker PID: {os.getpid()} is paused. Attach now!")
    # 您也可以使用一个变量来控制循环，这样附加后可以修改它退出循环
    while True:
        # 可以用一个短的 sleep 来减少 CPU 占用
        time.sleep(0.5) 
        # 您可以在附加调试器后，在 Watch 窗口手动设置一个标志变量来跳出循环
```

没用，vscode attach不上，直接gdb是c++堆栈，需要source python-gdb.py
