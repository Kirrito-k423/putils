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

def hook_for_model(model):
    exclude = []
    for i in range(24):
        exclude.append(f"image_encoder.encoder.blocks.layers.{i}.cross_attention")
        exclude.append(f"image_encoder.encoder.blocks.layers.{i}.cross_attn_bda")
        
    for i in range(40):
        exclude.append(f"text_decoder.decoder.layers.{i}.cross_attention")
        exclude.append(f"text_decoder.decoder.layers.{i}.cross_attn_bda")

    if ifdebug():
        rank = 0
        # print(f"hook_for_model record rank {rank}")
        # if rank == record_rank:
        for name, module in model.named_modules():
            if name.startswith('image_encoder.encoder.blocks.'):
                continue
            if name not in exclude:
                print(name)
                module.register_forward_hook(hook_func('[forward]: '+name, module, f"{rank}.log"))
                module.register_backward_hook(hook_func('[backward]: '+name, module, f"{rank}.log"))
