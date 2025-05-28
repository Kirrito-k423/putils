
import torch
import torch.distributed
import torch_npu


# list 
# bt
# n s 
def bk():
    if torch.distributed.get_rank() == 0:
        breakpoint()