# 比较与torch.optim.Adadelta的功能差异

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_zh_cn/note/api_mapping/pytorch_diff/ApplyAdadelta.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source.png"></a>

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

更多内容详见[torch.optim.Adadelta](https://pytorch.org/docs/1.5.0/optim.html#torch.optim.Adadelta)。

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

更多内容详见[mindspore.ops.ApplyAdadelta](https://mindspore.cn/docs/api/zh-CN/master/api_python/ops/mindspore.ops.ApplyAdadelta.html#mindspore.ops.ApplyAdadelta)。

## 使用方式

PyTorch：需要将期望更新的参数放入1个迭代类型参数`params`后传入，且设置了`step`方法执行单步优化返回损失值。

MindSpore：需要分别传入期望更新的参数`var`，`accum`，`accum_update`，`grad`。

## 代码示例

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