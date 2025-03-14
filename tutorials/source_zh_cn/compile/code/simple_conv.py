"""simple conv module"""

import time
from typing import Optional, Callable
import numpy as np
import mindspore
from mindspore import nn, Tensor


class BasicBlock(nn.Cell):
    """define the basic block of resnet"""
    expansion: int = 1

    def __init__(
            self,
            in_channels: int = 128,
            channels: int = 128,
            stride: int = 1,
            groups: int = 1,
            base_width: int = 64,
            norm: Optional[nn.Cell] = None,
            down_sample: Optional[nn.Cell] = None,
    ) -> None:
        super().__init__()
        if norm is None:
            norm = nn.BatchNorm2d
        assert groups == 1, "BasicBlock only supports groups=1"
        assert base_width == 64, "BasicBlock only supports base_width=64"

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3,
                               stride=stride, padding=1, pad_mode="pad")
        self.bn1 = norm(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3,
                               stride=1, padding=1, pad_mode="pad")
        self.bn2 = norm(channels)
        self.down_sample = down_sample

    def construct(self, x: Tensor) -> Tensor:
        """construct function"""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample is not None:
            identity = self.down_sample(x)

        out += identity
        out = self.relu(out)

        return out


block = BasicBlock()
grad_fn = mindspore.value_and_grad(block, None, block.trainable_params(), has_aux=False)


def fp(x):
    out = block(x)
    return out


def fp_and_bp(x):
    out, grads = grad_fn(x)
    return out, grads


def run_func(f: Callable, des: str = "function"):
    """run_func"""
    s_time = time.time()

    x = Tensor(np.random.randn(1, 128, 256, 256), mindspore.float32)
    out = f(x)

    time_to_prepare = time.time() - s_time
    s_time = time.time()

    for i in range(1000):
        out = f(x*(i/1000))

    time_to_run_thousand_times = time.time() - s_time

    s_out_shape = f"{out.shape}" if isinstance(out, Tensor) else f"{out[0].shape}, grad[0] shape is: {out[1][0].shape}"
    print(f"{des}, output shape is: {s_out_shape}, time to prepare: {time_to_prepare:.2f}s, "
          f"time to run thousand times: {time_to_run_thousand_times:.2f}s")


# run fp
run_func(fp, des="origin block fp")
run_func(mindspore.jit(fp), des="jitted block by default fp")
run_func(mindspore.jit(fp, jit_level="O1"), des="jitted block by O1 fp")

# run fp+bp
run_func(fp_and_bp, des="origin block fp+bp")
run_func(mindspore.jit(fp_and_bp), des="jitted block by default fp+bp")
run_func(mindspore.jit(fp_and_bp, jit_level="O1"), des="jitted block by O1 fp+bp")
