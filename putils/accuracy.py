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

def _normalize_name_list(name_list):
    if name_list is None:
        return set()
    if isinstance(name_list, str):
        name_list = [name_list]
    return {str(name).strip() for name in name_list if str(name).strip()}


def _get_named_modules(model, include_root=False):
    module_items = []
    for name, module in model.named_modules():
        if not include_root and not name:
            continue
        module_items.append((name, module))
    return module_items


def hook_for_model(model, include_list=None, exclude_list=None, print_structure=True):
    if not ifdebug():
        return []

    module_items = _get_named_modules(model)
    all_module_names = [name for name, _ in module_items]

    if print_structure:
        print(f"hook_for_model available_modules = {all_module_names}")

    include_set = _normalize_name_list(include_list)
    exclude_set = _normalize_name_list(exclude_list)

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    handles = []
    registered_modules = []

    if include_set:
        target_names = [name for name in all_module_names if name in include_set]
    else:
        target_names = list(all_module_names)

    if exclude_set:
        target_names = [name for name in target_names if name not in exclude_set]

    target_set = set(target_names)

    for name, module in module_items:
        if name not in target_set:
            continue

        registered_modules.append(name)
        print(f"hook_for_model register {name}")
        handles.append(module.register_forward_hook(hook_func('[forward]: '+name, module, f"{rank}.log")))
        handles.append(module.register_full_backward_hook(hook_func('[backward]: '+name, module, f"{rank}.log")))

    missing_modules = sorted(include_set - set(all_module_names))
    for name in missing_modules:
        print(f"hook_for_model skip missing module: {name}")

    missing_excludes = sorted(exclude_set - set(all_module_names))
    for name in missing_excludes:
        print(f"hook_for_model skip missing exclude module: {name}")

    print(f"hook_for_model registered_modules = {registered_modules}")

    return handles
