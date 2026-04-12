import torch
import torch.distributed

from .pprint import aprint, ifdebug

record_rank=1

def hook_func(name, module, file_path):
    def hook_function(module, inputs, outputs):
        print("===========================================================================================")
        print("###########################################################################################")
        aprint(name+' inputs', inputs, file_path)
        aprint(name+' outputs', outputs, file_path)
        print("###########################################################################################")
        print("*******************************************************************************************")
    return hook_function


def hookt(str, t):
    def hook_tensor(grad):
        aprint(str,grad)
    rank = torch.distributed.get_rank()
    if ifdebug():
        if rank == record_rank:
            aprint(f"forward {str}", t)
            if t.requires_grad:
                t.register_hook(hook_tensor)
                print(f"Tensor {str} hook success.")
            else:
                print(f"Tensor {str} does not require gradient. Skipping hook registration.")
                t.requires_grad_(True)  # 显式启用梯度
                t.register_hook(hook_tensor)
                print(f"Tensor {str} force hook success.")

def _normalize_include_list(include_list):
    if include_list is None:
        return set()
    if isinstance(include_list, str):
        include_list = [include_list]
    return {str(name).strip() for name in include_list if str(name).strip()}


def hook_for_model(model, include_list=None):
    if not ifdebug():
        return []

    include_set = _normalize_include_list(include_list)
    if not include_set:
        print("hook_for_model skipped: include_list is empty.")
        return []

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    handles = []
    matched_modules = []

    for name, module in model.named_modules():
        if name not in include_set:
            continue

        matched_modules.append(name)
        print(f"hook_for_model register {name}")
        handles.append(module.register_forward_hook(hook_func('[forward]: '+name, module, f"{rank}.log")))
        handles.append(module.register_full_backward_hook(hook_func('[backward]: '+name, module, f"{rank}.log")))

    missing_modules = sorted(include_set - set(matched_modules))
    for name in missing_modules:
        print(f"hook_for_model skip missing module: {name}")

    return handles
