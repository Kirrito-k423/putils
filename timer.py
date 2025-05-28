import time
from contextlib import contextmanager

# 全局开关，用于控制是否输出耗时信息
TIMER_VERBOSE = True

# with timer("sleep 1s"):
#     time.sleep(1)
@contextmanager
def timer(name=""):
    if TIMER_VERBOSE:
        start = time.perf_counter()
        print(f"[Timer] '{name}' started...")
    try:
        yield
    finally:
        if TIMER_VERBOSE:
            end = time.perf_counter()
            duration = end - start
            print(f"[Timer] '{name}' finished in {duration:.6f} seconds.")


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