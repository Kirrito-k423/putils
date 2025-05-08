
import torch
import torch.distributed
import hashlib
import os
from megatron.core import parallel_state

def ifdebug():
    return int(os.environ.get("TSJPRINT", 0)) == 1

def tensor_md5(tensor: torch.Tensor) -> str:
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()  # BFloat16 → Float32
    # 确保 Tensor 是连续的，并转为 numpy 数组
    tensor_np = tensor.cpu().detach().numpy()
    # 生成字节流
    tensor_bytes = tensor_np.tobytes()
    # 计算 MD5
    return hashlib.md5(tensor_bytes).hexdigest()

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
            print(f"'{name}'| Hash {tensor_md5(obj)} | sum {obj.sum()} | Shape: {obj.shape} | Size {obj.numel()} | Memory size: {obj.element_size() * obj.numel() / 1024**3:.2f} GB | isNan {torch.isnan(obj).any()} | {obj.dtype}  "\
                    , flush=True)
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
    dprank = parallel_state.get_data_parallel_rank()
    print_str = f"rank{rank} dp{dprank} {sstr} "
    if toTprint:
        tprint(sth, print_str)
    else:
        print(f"{print_str} {sth}")