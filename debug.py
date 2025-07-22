
import torch
import torch.distributed
import torch_npu
from torch_npu.contrib import transfer_to_npu
from .pprint import aprint, ifdebug
import portalocker
import socket
import os
import debugpy# 防止多次调用 listen()

def freeze_print(model):
    if not ifdebug():
        return
    
    print("="*200)
    for name, param in model.named_parameters():
        print(f"GRAD / 参数名: {param.requires_grad} / {name}")
    print("="*200)

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0
        
def vscode_gdb(start_port=23325, end_port=24000,lock_file="/tmp/debugpy.lock"):
    if not ifdebug():
        return

    # Step 1: 找到一个可用端口
    selected_port = None
    for port in range(start_port, end_port + 1):
        if not is_port_in_use(port):
            selected_port = port
            break
        else:
            print(f"🔒 Port {port} already in use. Check next port...")


    if selected_port is None:
        print("❌ No available port found in range.")
        lock_fd.close()
        return

    # Step 2: 尝试获取文件锁，只有获取到锁的进程才绑定端口
    lock_dir = os.path.dirname(lock_file)
    if lock_dir and not os.path.exists(lock_dir):
        os.makedirs(lock_dir, exist_ok=True)

    try:
        # 打开文件并尝试加锁
        lock_fd = open(lock_file, "w")
        portalocker.lock(lock_fd, portalocker.LOCK_EX | portalocker.LOCK_NB)

        # 成功获取锁，说明我们可以绑定端口
        debugpy.listen(("0.0.0.0", selected_port))
        print(f"⏳ Waiting for debugger to attach on port {selected_port}...",flush=True)
        debugpy.wait_for_client()
        print(f"🎉 Debugger attached on port {selected_port}!")

        # 锁文件保持打开状态，进程退出时自动释放锁
        # 可以在这里做其他任务
    except (portalocker.exceptions.LockException, OSError):
        # 获取锁失败，说明其他进程已经绑定端口
        print(f"🔒 Another process is managing the debug port {selected_port}.")
        lock_fd.close()
# {
#     // 使用 IntelliSense 了解相关属性。 
#     // 悬停以查看现有属性的描述。
#     // 欲了解更多信息，请访问: https://go.microsoft.com/fwlink/?linkid=830387
#     "version": "0.2.0",
#     "configurations": [

#         {
#             "name": "Python Debugger: Python File",
#             "type": "debugpy",
#             "request": "launch",
#             "program": "${file}"
#         },
#         {
#             "name": "Python: glm4v",
#             "type": "python",
#             "request": "attach",
#             "connect": {
#                 "host": "90.90.94.158",
#                 "port": 23323
#             },
#             "justMyCode": true,
#             "console": "integratedTerminal"
#         }
#     ]
# }

# list 
# bt
# n s 
def bk():
    if torch.distributed.get_rank() == 0:
        breakpoint()