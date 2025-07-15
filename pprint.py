
import torch
import torch.distributed
import hashlib
import os
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