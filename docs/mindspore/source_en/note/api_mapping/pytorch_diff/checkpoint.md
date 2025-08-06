# Differences with torch.utils.checkpoint.checkpoint

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/br_base/resource/_static/logo_source_en.svg)](https://gitee.com/mindspore/docs/blob/br_base/docs/mindspore/source_en/note/api_mapping/pytorch_diff/checkpoint.md)

## torch.utils.checkpoint.checkpoint

```text
torch.utils.checkpoint.checkpoint(function, preserve_rng_state=True, *args)
```

For more information, see [torch.utils.checkpoint.checkpoint](https://pytorch.org/docs/1.8.1/checkpoint.html#torch.utils.checkpoint.checkpoint).

## mindspore.nn.Cell.recompute

```text
mindspore.nn.Cell.recompute(mp_comm_recompute=True, parallel_optimizer_comm_recompute=False)
```

For more information, see [mindspore.nn.Cell.recompute](https://www.mindspore.cn/docs/en/br_base/api_python/nn/mindspore.nn.Cell.html#mindspore.nn.Cell.recompute).

## Differences

PyTorch: A function of class torch.utils.checkpoint, as a center layer to warp the Cell. Use torch.no_grad() in forward computation, and recompute the needed center nums in back propagation.

MindSpore: Defined as a function of Cell or primitive. Use the concrete cell.recompute() or primitive.recompute() to call recomputation.

| Categories | Subcategories |PyTorch | MindSpore | Difference |
| --- | ---   | ---   | ---        |---  |
| Parameters | Parameter 1  | function | mp_comm_recompute  | function means the cell that needed to be warpped. mp_comm_recompute indicates whether the communication operations introduced by model parallelism within the cell are recalculated in automatic parallel or semi-automatic parallel mode |
|      | Parameter 2  | preserve_rng_state | parallel_optimizer_comm_recompute | preserve_rng_state indicates whether to save the state of the random number generator, and parallel_optimizer_comm_recompute indicates whether the AllGather communication introduced by the optimizer in parallel within the specified cell is recalculated in automatic parallel or semi-automatic parallel mode |
|      | Parameter 3  | *args |   | Indicates the input parameter of the function function |

### Code Example 1

```python
# PyTorch
import torch
from torch.utils.checkpoint import checkpoint as cp
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(12, 24)
        self.fc2 = nn.Linear(24, 48)
        self.fc3 = nn.Linear(48, 2)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = cp(self._checkpoint, x)
        x = self.fc3(x)
        return x

    def _checkpoint(self, x):
        x = nn.functional.relu(self.fc2(x))
        return x

x = torch.randn(2, 12)
model = Net()
output = model(x)
output.sum().backward()
print(model.fc1.weight.grad)
```

```python
# MindSpore
import mindspore
import numpy as np
from mindspore import nn, ops

class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Dense(12, 24)
        self.relu1 = ops.ReLU()
        self.fc2 = nn.Dense(24, 48)
        self.fc2.recompute()
        self.relu2 = ops.ReLU()
        self.relu2.recompute()
        self.fc3 = nn.Dense(48, 2)

    def construct(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

model = Net()

def forward_fn(data, label):
    logits = model(data)
    loss = logits.sum()
    return loss, logits

grad_fn = mindspore.value_and_grad(forward_fn, None, (model.fc1.weight,), has_aux=True)

x = mindspore.Tensor(np.random.randn(2, 12), dtype=mindspore.float32)
label = mindspore.Tensor(np.random.randn(2, 12), dtype=mindspore.float32)
(loss, _), grads = grad_fn(x, label)
print(grads)
```
