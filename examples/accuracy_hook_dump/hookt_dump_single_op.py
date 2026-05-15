"""Single-op forward/backward tensor dump demo for precision debugging.

Usage:
    TSJPRINT=1 python examples/accuracy_hook_dump/hookt_dump_single_op.py
"""

from pathlib import Path

import torch

from putils.accuracy import hookt_dump


def main():
    dump_dir = Path("/tmp/putils_hookt_dump_demo")
    dump_dir.mkdir(parents=True, exist_ok=True)

    x = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)

    # 单算子示例：y = x^2。我们对中间 Tensor y 做前反向 dump。
    y = x * x
    handle = hookt_dump(
        name="single_op.square_y",
        t=y,
        dump_forward=True,
        dump_backward=True,
        dump_dir=dump_dir,
    )

    # 构造一个简单 loss，触发 backward。
    loss = y.sum()
    loss.backward()

    if handle is not None:
        handle.remove()

    forward_file = dump_dir / "single_op.square_y.forward_tensor.pt"
    backward_file = dump_dir / "single_op.square_y.backward_grad.pt"

    forward_payload = torch.load(forward_file)
    backward_payload = torch.load(backward_file)

    print(f"forward dump: {forward_file}")
    print(f"backward dump: {backward_file}")
    print("forward tensor:", forward_payload["tensor"])  # 期望 [1, 4, 9]
    print("backward grad:", backward_payload["tensor"])  # d(sum(y))/dy = [1, 1, 1]


if __name__ == "__main__":
    main()
