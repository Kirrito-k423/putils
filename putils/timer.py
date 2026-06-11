import time
import torch.distributed as dist
from contextlib import contextmanager
import torch
import torch_npu
import time
from .write2file import log2file

# 全局开关，用于控制是否输出耗时信息
TIMER_VERBOSE = True

# |--Add下发--| |--target func下发--||--Add下发--|
#                     |--空泡--| |--record--||--空泡--||--target func--| |--空泡--| |--record--|
class CUDAEVENT_TIMER():
    def __init__(self):
        self.events = {}
        self.times = {}
        self.cpu_times = []
    def add_cpu(self):
        torch.npu.synchronize()
        self.cpu_times.append(time.time())

    def add(self, label, group=None): #在当前位置添加cuda event
        if label not in self.times:
            self.times[label] = []
            self.events[label] = []
        stream = None
        #process group用于通信打点
        if group is not None:
            collective_stream_id = group._get_backend(torch.device('npu'))._get_stream_id(False)
            if collective_stream_id is None:
                return
            stream = torch_npu.npu.Stream(stream_id=collective_stream_id, device_type=20)

        event = torch_npu.npu.Event(enable_timing=True)
        event.record(stream=stream)
        self.events[label].append(event)

    def flush(self): #根据label对cuda event两两配对得出时间
        torch.npu.synchronize()
        for label in self.times:
            if torch.npu.current_device()==0:
                events = self.events[label]
                if len(events)%2 != 0:
                    print(f"got {len(events)} {label} events, should be paired")
                else:
                    for i in range(0, len(events), 2): 
                        time = events[i].elapsed_time(events[i+1])
                        self.times[label].append(time)
                    print(f"\n、 {label} execute {len(self.times[label])} times, average time is {sum(self.times[label])/len(self.times[label])} ms, each time is: {self.times[label]}")
        if torch.npu.current_device()==0 and len(self.cpu_times) == 2:
            print(f"\n cpu_time: {(self.cpu_times[1] - self.cpu_times[0])*1000} ms")
        self.cpu_times = []
        self.events = {}
        self.times = {}

event_timer = CUDAEVENT_TIMER()     

# with timer("sleep 1s"):
#     time.sleep(1)
@contextmanager
def timer(name="", print_rank0_only=True):
    if TIMER_VERBOSE:
        start = time.perf_counter()
        rank = dist.get_rank() if dist.is_initialized() else 0
        if not print_rank0_only or rank == 0:
            print(f"[Timer] '{name}' started...")
            prefix_str = f"{name} func started"
            log2file(f"{name} {mprint(prefix_str)}")
    try:
        yield
    finally:
        if TIMER_VERBOSE:
            end = time.perf_counter()
            duration = end - start
            rank = dist.get_rank() if dist.is_initialized() else 0
            if not print_rank0_only or rank == 0:
                print(f"[Timer] '{name}' finished in {duration:.6f} seconds.")
                prefix_str = f"{name} func end"
                log2file(f"{name} {mprint(prefix_str)} during {duration}")



# 如果你想用装饰器方式统计整个函数
# @timer_decorator("custom name")
# def my_func():
#     time.sleep(0.5)
def timer_decorator(name=None, print_rank0_only=True):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with timer(name or func.__name__, print_rank0_only=print_rank0_only):
                return func(*args, **kwargs)
        return wrapper
    return decorator

def to_filename_safe(timestamp_str):
    return (
        timestamp_str.replace("-", "")
                   .replace(":", "")
                   .replace(" ", "_")
                   .replace(".", "_ms")
    )

def get_formatted_time():
    now = time.time()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
    milliseconds = int((now * 1000) % 1000)
    return f"{timestamp}.{milliseconds:03d}"

def get_time_str():
    return to_filename_safe(get_formatted_time())

def mprint(msg, level="INFO"):
    formatted_time = get_formatted_time()
    print(f"[{formatted_time}] [{level}] {msg}")
    return to_filename_safe(formatted_time)
