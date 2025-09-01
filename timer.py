import time
from contextlib import contextmanager
from .write2file import log2file

# 全局开关，用于控制是否输出耗时信息
TIMER_VERBOSE = True

# with timer("sleep 1s"):
#     time.sleep(1)
@contextmanager
def timer(name=""):
    if TIMER_VERBOSE:
        start = time.perf_counter()
        print(f"[Timer] '{name}' started...")
        prefix_str = f"{name} func started"
        log2file(f"{name} {mprint(prefix_str)}")
    try:
        yield
    finally:
        if TIMER_VERBOSE:
            end = time.perf_counter()
            duration = end - start
            print(f"[Timer] '{name}' finished in {duration:.6f} seconds.")
            prefix_str = f"{name} func end"
            log2file(f"{name} {mprint(prefix_str)} during {duration}")



# 如果你想用装饰器方式统计整个函数
# @timer_decorator("custom name")
# def my_func():
#     time.sleep(0.5)
def timer_decorator(name=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with timer(name or func.__name__):
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
