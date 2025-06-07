import torch

import os
import threading

import torch.distributed

def log2file(content, prefix=""):
    """
    将调试信息写入特定的日志文件中。

    :param content: 要写入的内容
    :param prefix: 文件名前缀
    :param rank_id: 进程或rank编号
    """
    rank_id=torch.distributed.get_rank()

    # 获取线程ID
    tid = threading.get_ident()

    # 获取环境变量 log_tag，不存在则为空字符串
    log_tag = os.getenv("log_tag", "")

    # 构建文件名
    filename_parts = [prefix, f"rank{rank_id}", f"tid{tid}"]
    if log_tag:
        filename_parts.append(log_tag)
    filename = "_".join(filename_parts) + ".log"

    # 写入内容到文件（追加模式）
    with open(filename, "a", encoding="utf-8") as f:
        f.write(content + "\n")

    # print(f"[LOG] Wrote to {filename}")  # 可选打印信息