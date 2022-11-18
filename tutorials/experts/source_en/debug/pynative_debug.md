# Debugging in PyNative Mode

<a href="https://gitee.com/mindspore/docs/blob/master/tutorials/experts/source_en/debug/pynative_debug.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>&nbsp;&nbsp;

## Introduction

PyNative, also called the dynamic graph mode. In this mode, Python commands are executed statement by statement based on the Python syntax. After each Python command is executed, the execution result of the Python statement can be obtained. Therefore, in PyNative mode, you can debug network scripts command by command or at a specific command location.

## Breakpoint Debugging

Breakpoint debugging is to set a breakpoint before or after a command in a network script. When the network script runs to the breakpoint, the script stops. You can view the variable information at the breakpoint or debug the script step by step. During the debugging, you can view the current value of each variable. You can determine whether the current code is correct by analyzing whether the variables at the breakpoint are proper. In PyNative mode, Python commands are executed statement by statement based on the Python syntax. Therefore, in PyNative mode, you can use the breakpoint debugging tool pdb provided by Python to debug network scripts.

The following piece of code is used to demonstrate the breakpoint debugging function.

```python
import pdb
import numpy as np
import mindspore as ms
from mindspore import Tensor, nn, set_context
from mindspore import Parameter, ParameterTuple
from mindspore import ops
set_context(mode=ms.PYNATIVE_MODE)
class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.w1 = Parameter(Tensor(np.random.randn(5, 6).astype(np.float32)), name="w1", requires_grad=True)
        self.w2 = Parameter(Tensor(np.random.randn(5, 6).astype(np.float32)), name="w2", requires_grad=True)
        self.relu = nn.ReLU()
        self.pow = ops.Pow()

    def construct(self, x, y):
        x = self.relu(x * self.w1) * self.w2
        pdb.set_trace()
        out = self.pow(x - y, 2)
        return out

x = Tensor(np.random.randn(5, 6).astype(np.float32))
y = Tensor(np.random.randn(5, 6).astype(np.float32))

net = Net()
ret = net(x, y)
weights = ParameterTuple(filter(lambda x : x.requires_grad, net.get_parameters()))
grads = ms.grad(net, grad_position=None, weights=weights)(x, y)
print("grads: ", grads)

```

1. You can import pdb to the script to use the breakpoint debugging function.

    ```python
    import pdb
    ```

2. Set the following command at the position where the breakpoint is required to stop the network script when the command is executed:

    **Demo code:**

    ```python
    x = self.relu(x * self.w1) * self.w2
    pdb.set_trace()
    out = self.pow(x - y, 2)
    return out
    ```

    As shown in Figure 1, the script stops at the `out = self.pow(x-y, 2)` command and waits for the next pdb command.

    ![pynative_debug.png](https://gitee.com/mindspore/docs/raw/master/tutorials/experts/source_zh_cn/debug/images/pynative_debug.png)

    Figure 1

3. When a network script stops at a breakpoint, you can use common pdb debugging commands to debug the network script. For example, you can print variable values, view program call stacks, and perform step-by-step debugging.

    * To print the value of a variable, run the p command, as shown in (1) in Figure 1.
    * To view the program call stack, run the bt command, as shown in (2) in Figure 1.
    * To view the network script context of the breakpoint, run the l command, as shown in (3) in Figure 1.
    * To debug the network script step by step, run the n command, as shown in (4) in Figure 1.

## Common pdb Commands

For details about how to use pdb commands, see the [pdb official document](https://docs.python.org/3/library/pdb.html).
