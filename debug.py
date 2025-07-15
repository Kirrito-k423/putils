
import torch
import torch.distributed
import torch_npu
from .pprint import aprint, ifdebug


def vscode_gdb(port=23325):
    if ifdebug():
        import debugpy# 防止多次调用 listen()
        if not hasattr(debugpy, "_already_listened"):
            debugpy._already_listened = True
            debugpy.listen(("0.0.0.0", port))
            print("⏳ Waiting for debugger to attach...")
            debugpy.wait_for_client()
            print("🎉 Debugger attached!")

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