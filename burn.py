"""Burn GPU/NPU SMA to see if the machine works fine."""
# nohup python3 /mnt/bn/seutao-hl/Projects/Mammoth2DIT/burn.py --sleep 0.007 --size 5000 > burn.log 2>&1 &
import argparse
import os
import time
import torch
import torch.multiprocessing as mp
from accelerate.utils import is_cuda_available, is_npu_available

def get_args() -> argparse.Namespace:
    """Get burning arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=10000)
    parser.add_argument("--sleep", type=float, default=0.01)
    return parser.parse_args()
def train(rank, args, device_name) -> None:
    """Burn GPU/NPU with matmul op."""
    # Obtain current device
    device = torch.device(f"{device_name}:{rank}")
    size = args.size  # Adjust this to occupy more GPU memory
    sleep_time = args.sleep
    # Create a computation-intensive task (e.g., large-scale matrix multiplication)
    torch.manual_seed(rank)  # Use different seeds for each process
    x = torch.rand(size, size, device=device)
    y = torch.rand(size, size, device=device)
    while True:  # Infinite loop until manually terminated
        z = torch.matmul(x, y)
        # Add a small operation to prevent optimization by the compiler
        z = z * torch.randn(1, device=device)
        del z  # Release memory, but this will still use computational resources
        time.sleep(sleep_time)
if __name__ == "__main__":
    # Initialize the distributed environment.
    if is_npu_available():  # This would import torch_npu
        accelerator = torch.npu
        device_name = "npu"
    elif is_cuda_available():
        accelerator = torch.cuda
        device_name = "cuda"
    else:
        error_msg = "No GPU/NPU available."
        raise RuntimeError(error_msg)
    # Get world size considering the VISIBLE_DEVICES environment variable.
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        world_size = len(os.getenv("CUDA_VISIBLE_DEVICES").split(","))
    elif "ASCEND_RT_VISIBLE_DEVICES" in os.environ:
        world_size = len(os.getenv("ASCEND_RT_VISIBLE_DEVICES").split(","))
    else:
        world_size = accelerator.device_count()  # Get the number of available GPUs
    # Launch multi-process training, running one process per GPU/NPU.
    print(f"Launching {world_size} processes on {device_name} devices. Start burning...")
    mp.spawn(train, args=(get_args(), device_name), nprocs=world_size, join=True)
