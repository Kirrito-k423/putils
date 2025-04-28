import torch
import torch.distributed

from .pprint import aprint

def hook_func(name, module, file_path):
    def hook_function(module, inputs, outputs):
        print("===========================================================================================")
        print("###########################################################################################")
        aprint(name+' inputs', inputs, file_path)
        aprint(name+' outputs', outputs, file_path)
        print("###########################################################################################")
        print("*******************************************************************************************")
    return hook_function

def hook_for_model(model):
        exclude = []
        exclude.append("image_encoder.encoder.encoder.layers.0.cross_attention")
        exclude.append("image_encoder.encoder.encoder.layers.0.cross_attn_bda")
        exclude.append("text_decoder.decoder.layers.0.cross_attention")
        exclude.append("text_decoder.decoder.layers.0.cross_attn_bda")
        exclude.append("text_decoder.decoder.layers.1.cross_attention")
        exclude.append("text_decoder.decoder.layers.1.cross_attn_bda")
        

        rank = torch.distributed.get_rank()
        for name, module in model.named_modules():
            if name not in exclude:
                print(name)
                module.register_forward_hook(hook_func('[forward]: '+name, module, f"{rank}.log"))
                module.register_backward_hook(hook_func('[backward]: '+name, module, f"{rank}.log"))
