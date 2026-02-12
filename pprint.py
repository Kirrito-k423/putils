
import torch
import torch.distributed
import hashlib
import os
import functools
from .write2file import log2file

def log_or_print(content):
    if os.getenv("log_tag", "") != "":
        log2file(content)
    else:
        print(content, flush=True)


def ifdebug():
    return int(os.environ.get("TSJPRINT", 0)) == 1


def tensor_md5(tensor: torch.Tensor, length=6) -> str:
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()  # BFloat16 → Float32
    # 确保 Tensor 是连续的，并转为 numpy 数组
    tensor_np = tensor.cpu().detach().numpy()
    # 生成字节流
    tensor_bytes = tensor_np.tobytes()
    # 计算 MD5
    return hashlib.md5(tensor_bytes).hexdigest()[:length]

def aprint(name, tensors,file_path=''):
    if ifdebug():
        if isinstance(tensors, torch.Tensor):
            tprint(tensors, name)
        elif isinstance(tensors, tuple) or isinstance(tensors, list):
            for tensor in tensors:
                tprint(tensor, name)
        else:
            print(name , type(tensors))

def tprint(obj, name=""):
    if ifdebug():
        if torch.is_tensor(obj):
            origin_dtype=obj.dtype
            obj=obj.to(torch.float)
            try:
                tmp_mean = obj.mean()
            except Exception as e:
                tmp_mean = e
            mem_size=obj.element_size() * obj.numel() / 1024**3
            tmp_str1= f"'{name}'| {tensor_md5(obj)} | {origin_dtype} | {obj.shape} | continue: {obj.is_contiguous()} | mean {tmp_mean} | sum {obj.sum()} "
            tmp_str1+= f"| Size {obj.numel()} | Memory size: {mem_size:.2f} GB | isNan {torch.isnan(obj).any()}"
            print(tmp_str1, flush=True)
            tmp_str2=obj.flatten()[:10].tolist()
            print(tmp_str2, flush=True)
            return tmp_str1, tmp_str2
        elif isinstance(obj, torch.nn.Module):
            total_params = sum(p.numel() for p in obj.parameters())
            print(f"'{name}' is a nn.Module. |Total parameters: {total_params}")
        else:
            print(f"'{name}' {obj}")
            # mean {obj.mean()} 

# print part model weight
def wprint(pmodel, name="wprint"):
    rank =  torch.distributed.get_rank()
    for param_name, param in pmodel.named_parameters():
        tprint(param,f"{name} rank {rank} {param_name}")


def dis_print(sth, sstr="", toTprint=True):
    rank = torch.distributed.get_rank()
    from megatron.core import parallel_state
    dprank = parallel_state.get_data_parallel_rank()
    print_str = f"rank{rank} dp{dprank} {sstr} "
    if toTprint:
        tprint(sth, print_str)
    else:
        print(f"{print_str} {sth}")


_printed_tags = set()
def print_once_by_tag(tag, message=""):
    global _printed_tags
    if tag not in _printed_tags:
        print(f"[{tag}] {message}")
        _printed_tags.add(tag)


def _pprint_any(obj, name="", max_items=8, _depth=0, _max_depth=3):
    if not ifdebug():
        return
    if _depth > _max_depth:
        log_or_print(f"'{name}' <max_depth_reached> type={type(obj)}")
        return
    if torch.is_tensor(obj) or isinstance(obj, torch.nn.Module):
        tprint(obj, name)
        return
    if isinstance(obj, (list, tuple)):
        log_or_print(f"'{name}' {type(obj).__name__} len={len(obj)}")
        for i, v in enumerate(obj[:max_items]):
            _pprint_any(v, f"{name}[{i}]", max_items=max_items, _depth=_depth + 1, _max_depth=_max_depth)
        if len(obj) > max_items:
            log_or_print(f"'{name}' ... truncated ({len(obj) - max_items} more)")
        return
    if isinstance(obj, dict):
        log_or_print(f"'{name}' dict len={len(obj)}")
        for i, (k, v) in enumerate(list(obj.items())[:max_items]):
            _pprint_any(k, f"{name}.key[{i}]", max_items=max_items, _depth=_depth + 1, _max_depth=_max_depth)
            _pprint_any(v, f"{name}.val[{i}]", max_items=max_items, _depth=_depth + 1, _max_depth=_max_depth)
        if len(obj) > max_items:
            log_or_print(f"'{name}' ... truncated ({len(obj) - max_items} more)")
        return
    log_or_print(f"'{name}' {obj} (type={type(obj)})")


# 函数装饰器 monitor_io(...)，会在 TSJPRINT=1 时对每次调用打印所有输入/输出
# 直接装饰函数：
# @monitor_io()
# 自定义 tag：
# @monitor_io("my_fn")
# 控制容器打印：
# @monitor_io(max_items=4, max_depth=2)
def monitor_io(tag: str = "", max_items: int = 8, max_depth: int = 3):
    def _decorator(func):
        call_idx = {"i": 0}

        @functools.wraps(func)
        def _wrapped(*args, **kwargs):
            if not ifdebug():
                return func(*args, **kwargs)

            call_idx["i"] += 1
            fname = tag or getattr(func, "__qualname__", getattr(func, "__name__", "func"))
            prefix = f"{fname}#{call_idx['i']}"

            for i, a in enumerate(args):
                _pprint_any(a, f"{prefix}.in.args[{i}]", max_items=max_items, _max_depth=max_depth)
            for k, v in kwargs.items():
                _pprint_any(v, f"{prefix}.in.kwargs.{k}", max_items=max_items, _max_depth=max_depth)

            try:
                out = func(*args, **kwargs)
            except Exception as e:
                log_or_print(f"'{prefix}.out.exception' {repr(e)}")
                raise

            _pprint_any(out, f"{prefix}.out", max_items=max_items, _max_depth=max_depth)
            return out

        return _wrapped

    return _decorator
