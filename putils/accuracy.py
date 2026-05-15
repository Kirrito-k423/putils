import re
from pathlib import Path

import torch
import torch.distributed

from .pprint import aprint, ifdebug

record_rank=1

def hook_func(name, module, file_path):
    def hook_function(module, inputs, outputs):
        print("===========================================================================================")
        print("###########################################################################################")
        # 在 full backward hook 中，这里的 inputs 表示该 module 前向输入对应的梯度（grad_input）。
        aprint(name+' inputs', inputs, file_path)
        aprint(name+' outputs', outputs, file_path)
        print("###########################################################################################")
        print("*******************************************************************************************")
    return hook_function


def hookt(str, t):
    return hookt_dump(str, t)


def _to_filename_safe(value):
    return re.sub(r"[^0-9a-zA-Z_.-]+", "_", str(value)).strip("_")


def _save_tensor_dump(name, phase, tensor, dump_dir):
    dump_root = Path(dump_dir)
    dump_root.mkdir(parents=True, exist_ok=True)

    safe_name = _to_filename_safe(name)
    file_name = f"{safe_name}.{phase}.pt"
    file_path = dump_root / file_name

    dump_payload = {
        "name": name,
        "phase": phase,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "tensor": tensor.detach().cpu(),
    }
    torch.save(dump_payload, file_path)
    print(f"Tensor {name} {phase} dump saved: {file_path}")
    return str(file_path)


def hookt_dump(
    name,
    t,
    dump_forward=True,
    dump_backward=True,
    dump_dir=None,
    force_requires_grad=True,
):
    """
    Hook one Tensor for forward/backward debug and optional dump.

    Args:
        name: Tensor tag used in logs and dump file names.
        t: Tensor to hook.
        dump_forward: Whether to print/dump forward tensor value.
        dump_backward: Whether to print/dump backward gradient (dLoss/dt).
        dump_dir: Optional directory for saving `.pt` dump files.
        force_requires_grad: If True, set `t.requires_grad_(True)` when needed.
    """
    def hook_tensor(grad):
        # grad 是该 Tensor 在反向传播中收到的梯度（dLoss/dt）。
        if dump_backward:
            aprint(f"backward {name}", grad)
            if dump_dir:
                _save_tensor_dump(name, "backward_grad", grad, dump_dir)

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
    if ifdebug():
        if rank == record_rank:
            # forward 阶段打印/落盘 Tensor 本身，便于和反向梯度对照。
            if dump_forward:
                aprint(f"forward {name}", t)
                if dump_dir:
                    _save_tensor_dump(name, "forward_tensor", t, dump_dir)

            if not dump_backward:
                return None

            if t.requires_grad:
                # backward 时会回调 hook_tensor(grad)。
                handle = t.register_hook(hook_tensor)
                print(f"Tensor {name} hook success.")
                return handle

            print(f"Tensor {name} does not require gradient. Skipping hook registration.")
            if force_requires_grad:
                t.requires_grad_(True)  # 显式启用梯度，便于调试捕获 dLoss/dt。
                handle = t.register_hook(hook_tensor)
                print(f"Tensor {name} force hook success.")
                return handle
    return None

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
