# Function Differences with torch.optim.Adadelta

<a href="https://gitee.com/mindspore/docs/blob/r1.7/docs/mindspore/source_en/note/api_mapping/pytorch_diff/ApplyAdadelta.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

## torch.optim.Adadelta

```python
class torch.optim.Adadelta(
    params,
    lr=1.0,
    rho=0.9,
    eps=1e-06,
    weight_decay=0
)
```

For more information, see [torch.optim.Adadelta](https://pytorch.org/docs/1.5.0/optim.html#torch.optim.Adadelta).

## mindspore.ops.ApplyAdadelta

```python
class mindspore.ops.ApplyAdadelta(*args, **kwargs)(
    var,
    accum,
    accum_update,
    lr,
    rho,
    epsilon,
    grad
)
```

For more information, see [mindspore.ops.ApplyAdadelta](https://mindspore.cn/docs/en/r1.7/api_python/ops/mindspore.ops.ApplyAdadelta.html#mindspore.ops.ApplyAdadelta).

## Differences

PyTorch: Parameters to be optimized should be put into an iterable parameter then passed as a whole. The `step` method is also implemented to perform one single step optimization and return loss.

MindSpore: Parameters to be updated: `var`, `accum`, `accum_update`, `grad` should be passed respectively.

## Code Example

```python
# The following implements Adadelta with MindSpore.
import numpy as np
import torch
import mindspore.nn as nn
from mindspore import Tensor, Parameter
import mindspore.ops as ops
from mindspore import dtype as mstype

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.apply_adadelta = ops.ApplyAdadelta()
        self.var = Parameter(Tensor(np.random.rand(1, 1).astype(np.float32)), name="var")
        self.accum = Parameter(Tensor(np.random.rand(1, 1).astype(np.float32)), name="accum")
        self.accum_update = Parameter(Tensor(np.random.rand(1, 1).astype(np.float32)), name="accum_update")
    def construct(self, lr, rho, epsilon, grad):
        return self.apply_adadelta(self.var, self.accum, self.accum_update, lr, rho, epsilon, grad)

np.random.seed(0)
net = Net()
lr = Tensor(0.001, mstype.float32)
rho = Tensor(0.0, mstype.float32)
epsilon = Tensor(1e-6, mstype.float32)
grad = Tensor(np.random.rand(1, 1).astype(np.float32))
var, accum, accum_update = net(lr, rho, epsilon, grad)
print(var)
print(accum)
print(accum_update)
# Out:
# [[0.5480]]
# [[0.2969]]
# [[0.6028]]

# The following implements Adadelta with torch.
input_x = torch.tensor(np.random.rand(1, 20).astype(np.float32))
input_y = torch.tensor([1.])
net = torch.nn.Sequential(torch.nn.Linear(input_x.shape[-1], 1))
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adadelta(net.parameters())
l = loss(net(input_x).view(-1), input_y) / 2
optimizer.zero_grad()
l.backward()
optimizer.step()
print(loss(net(input_x).view(-1), input_y).item() / 2)
# Out:
# 0.5616
```
