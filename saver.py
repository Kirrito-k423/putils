import os
import uuid
import torch
from concurrent.futures import ThreadPoolExecutor
from filelock import FileLock

# from putils.saver import AsyncTorchSaver
# async_saver = AsyncTorchSaver()
# async_saver.save(full_sd, model_path)


def to_cpu(obj):
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(v) for v in obj)
    elif hasattr(obj, "device"):
        try:
            return obj.detach().cpu()
        except Exception:
            return obj
    else:
        return obj


class AsyncTorchSaver:
    def __init__(self, tmp_dir="/tmp/torch_save_tmp", max_workers=1):
        self.tmp_dir = tmp_dir
        os.makedirs(tmp_dir, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def save(self, obj, final_path):
        """
        非阻塞保存：
        1. CPU
        2. 写本地 tmp
        3. 锁父目录
        4. 原子 replace 到网盘
        """
        return self.executor.submit(self._save_impl, obj, final_path)

    def _save_impl(self, obj, final_path):
        parent_dir = os.path.dirname(final_path)
        os.makedirs(parent_dir, exist_ok=True)

        # 目录级锁
        lock_path = os.path.join(parent_dir, ".write.lock")

        tmp_file = os.path.join(
            self.tmp_dir,
            f"{os.path.basename(final_path)}.{uuid.uuid4().hex}.tmp"
        )

        try:
            # 1. to CPU（避免子线程触 NPU）
            obj_cpu = to_cpu(obj)

            # 2. 写本地 tmp（快、稳定）
            torch.save(obj_cpu, tmp_file)

            # 3. 串行写网盘（目录级锁）
            with FileLock(lock_path):
                os.replace(tmp_file, final_path)

        except Exception as e:
            print(f"[async_torch_save] failed: {final_path}, err={e}")
            try:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
            except Exception:
                pass
