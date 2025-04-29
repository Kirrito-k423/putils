
import torch
import torch.distributed
import hashlib

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
    if isinstance(tensors, torch.Tensor):
        tprint(tensors, name)
    elif isinstance(tensors, tuple) or isinstance(tensors, list):
        for tensor in tensors:
            tprint(tensor, name)
    else:
        print(name , type(tensors))

def tprint(obj, name=""):
    if True:
        if torch.is_tensor(obj):
            print(f"TSJTensor '{name}' | Shape: {obj.shape} | Hash {tensor_md5(obj)} | \
                    mean {obj.mean()} | sum {obj.sum()} |{obj.dtype} |\
                    Size {obj.numel()} | Memory size: {obj.element_size() * obj.numel() / 1024**2:.2f} MB "\
                    , flush=True)
        elif isinstance(obj, torch.nn.Module):
            total_params = sum(p.numel() for p in obj.parameters())
            print(f"TSJObject '{name}' is a nn.Module. |Total parameters: {total_params}")
        else:
            print(f"TSJObject '{name}' {obj}")