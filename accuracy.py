import torch
import torch.distributed

from .pprint import aprint

record_rank=0

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
    if rank == record_rank:
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
    exclude.append("image_encoder.encoder.encoder.layers.0.cross_attention")
    exclude.append("image_encoder.encoder.encoder.layers.0.cross_attn_bda")
    exclude.append("text_decoder.decoder.layers.0.cross_attention")
    exclude.append("text_decoder.decoder.layers.0.cross_attn_bda")
    exclude.append("text_decoder.decoder.layers.1.cross_attention")
    exclude.append("text_decoder.decoder.layers.1.cross_attn_bda")
    

    rank = torch.distributed.get_rank()
    if rank == record_rank:
        for name, module in model.named_modules():
            if name not in exclude:
                print(name)
                module.register_forward_hook(hook_func('[forward]: '+name, module, f"{rank}.log"))
                module.register_backward_hook(hook_func('[backward]: '+name, module, f"{rank}.log"))
