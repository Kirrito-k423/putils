"""Burn GPU/NPU SMA to see if the machine works fine."""
import argparse
import fcntl
import hashlib
import json
import os
import sys
import time
import torch
import torch.multiprocessing as mp
from accelerate.utils import is_cuda_available, is_npu_available

def get_args() -> argparse.Namespace:
    """Get burning arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=10000)
    parser.add_argument("--sleep", type=float, default=0.01)
    parser.add_argument("--lock-dir", type=str, default="/tmp/burn_locks")
    parser.add_argument("--disable-lock", action="store_true")
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

def _get_visible_devices_env(device_name: str) -> str:
    if device_name == "cuda":
        return "CUDA_VISIBLE_DEVICES"
    if device_name == "npu":
        return "ASCEND_RT_VISIBLE_DEVICES"
    return ""

def _task_lock_signature(args: argparse.Namespace, device_name: str, world_size: int) -> str:
    env_key = _get_visible_devices_env(device_name)
    visible_devices = os.getenv(env_key, "") if env_key else ""
    payload = {
        "device_name": device_name,
        "size": args.size,
        "sleep": args.sleep,
        "world_size": world_size,
        "visible_devices": visible_devices,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()

def _acquire_task_lock(lock_dir: str, signature: str):
    os.makedirs(lock_dir, exist_ok=True)
    lock_path = os.path.join(lock_dir, f"burn_{signature}.lock")
    fp = open(lock_path, "a+")
    try:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fp.close()
        return None, lock_path
    fp.seek(0)
    fp.truncate()
    fp.write(f"{os.getpid()}\n")
    fp.flush()
    return fp, lock_path

if __name__ == "__main__":
    args = get_args()
    if is_npu_available():
        accelerator = torch.npu
        device_name = "npu"
    elif is_cuda_available():
        accelerator = torch.cuda
        device_name = "cuda"
    else:
        error_msg = "No GPU/NPU available."
        raise RuntimeError(error_msg)
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        world_size = len(os.getenv("CUDA_VISIBLE_DEVICES").split(","))
    elif "ASCEND_RT_VISIBLE_DEVICES" in os.environ:
        world_size = len(os.getenv("ASCEND_RT_VISIBLE_DEVICES").split(","))
    else:
        world_size = accelerator.device_count()
    lock_fp = None
    if not args.disable_lock:
        signature = _task_lock_signature(args, device_name, world_size)
        lock_fp, lock_path = _acquire_task_lock(args.lock_dir, signature)
        if lock_fp is None:
            print(f"Skip launching: same burn task already running (lock: {lock_path}).")
            sys.exit(0)
    print(f"Launching {world_size} processes on {device_name} devices. Start burning...")
    mp.spawn(train, args=(args, device_name), nprocs=world_size, join=True)
