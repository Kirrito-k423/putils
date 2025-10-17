# trace_manager.py
import threading
import time
import os
from viztracer import VizTracer

# 常用方法
# from trace_manager import TraceManager

# def main():
#     # ... your code ...
#     if need_debug:
#         trace_manager.save_now("debug_point.json")

# if __name__ == "__main__":
#     with TraceManager(auto_save_interval=60) as trace_manager:
#         main()

class TraceManager:
    def __init__(
        self,
        auto_save_interval: int = 60,          # 自动保存间隔（秒），设为 None 则关闭
        final_output_file: str = "trace_final.json",
        auto_prefix: str = "trace_auto_",
        tracer_entries: int = 1000000,
        verbose: int = 0,
        use_fork_save: bool = True             # True = 分段保存（推荐），False = 全量累积
    ):
        self.auto_save_interval = auto_save_interval
        self.final_output_file = final_output_file
        self.auto_prefix = auto_prefix
        self.use_fork_save = use_fork_save
        self.verbose = verbose

        # 初始化 viztracer
        self.tracer = VizTracer(tracer_entries=tracer_entries, verbose=0)
        self._stop_event = threading.Event()
        self._save_lock = threading.Lock()
        self._auto_counter = 0
        self._started = False

    def start(self):
        """启动追踪和自动保存线程"""
        if self._started:
            return
        self.tracer.start()
        self._started = True

        if self.auto_save_interval and self.auto_save_interval > 0:
            self._auto_thread = threading.Thread(target=self._auto_save_worker, daemon=True)
            self._auto_thread.start()
            if self.verbose:
                print(f"[TraceManager] Auto-save every {self.auto_save_interval}s enabled.")

    def stop(self):
        """停止追踪并保存最终结果"""
        if not self._started:
            return
        self._stop_event.set()
        self.tracer.stop()
        self.save(self.final_output_file)
        self._started = False

    def save(self, output_file: str):
        """立即保存当前 trace（全量）"""
        with self._save_lock:
            if self.verbose:
                print(f"[TraceManager] Saving trace to {output_file}")
            self.tracer.save(output_file=output_file)

    def save_now(self, output_file: str = None):
        """手动触发保存，可指定文件名"""
        if not self._started:
            raise RuntimeError("TraceManager not started. Call start() first.")
        if output_file is None:
            output_file = f"trace_manual_{int(time.time())}.json"
        if self.use_fork_save:
            # fork_save 会保存并清空缓冲区（适合分段）
            with self._save_lock:
                if self.verbose:
                    print(f"[TraceManager] Fork-saving to {output_file}")
                self.tracer.fork_save(output_file=output_file)
        else:
            self.save(output_file)

    def _auto_save_worker(self):
        """后台自动保存线程"""
        while not self._stop_event.is_set():
            time.sleep(self.auto_save_interval)
            if not self._stop_event.is_set():
                self._auto_counter += 1
                filename = f"{self.auto_prefix}{self._auto_counter:04d}.json"
                try:
                    self.save_now(filename)
                except Exception as e:
                    if self.verbose:
                        print(f"[TraceManager] Auto-save failed: {e}")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
