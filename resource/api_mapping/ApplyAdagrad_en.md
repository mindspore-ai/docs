# Function Differences with torch.optim.Adagrad

## torch.optim.Adagrad

```python
class torch.optim.Adagrad(
    params,
    lr=0.01,
    lr_decay=0,
    weight_decay=0,
    initial_accumulator_value=0,
    eps=1e-10
)
```

## mindspore.nn.ApplyAdagrad

```python
class mindspore.nn.Adagrad(
    params,
    accum=0.1,
    learning_rate=0.001,
    update_slots=True,
    loss_scale=1.0,
    weight_decay=0.0
)(grads)
```

## Differences

PyTorch: Parameters to be optimized should be put into an iterable parameter then passed as a whole. The `step` method is also implemented to perform one single step optimization and return loss.

MindSpore: Parameters to be updated: `grads`, `params` should be passed respectively.

## Code Example

```python
# The following implements Adagrad with MindSpore.
import numpy as np
import torch
import mindspore.nn as nn
from mindspore import Tensor, Parameter
import mindspore.ops as ops
from mindspore import dtype as mstype

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.apply_adagrad = ops.ApplyAdagrad()
        self.var = Parameter(Tensor(np.random.rand(1, 1).astype(np.float32)), name="var")
        self.accum = Parameter(Tensor(np.random.rand(1, 1).astype(np.float32)), name="accum")

    def construct(self, lr, grad):
        return self.apply_adagrad(self.var, self.accum, lr, grad)

np.random.seed(0)
net = Net()
lr = Tensor(0.001, mstype.float32)
grad = Tensor(np.random.rand(1, 1).astype(np.float32))
var, accum = net(lr, grad)
print(var)
print(accum)
# Out:
# [[0.5482]]
# [[1.0785]]

# The following implements Adagrad with torch.
input_x = torch.tensor(np.random.rand(1, 20).astype(np.float32))
input_y = torch.tensor([1.])
net = torch.nn.Sequential(torch.nn.Linear(input_x.shape[-1], 1))
loss = torch.nn.MSELoss()
optimizer = torch.optim.Adagrad(net.parameters())
l = loss(net(input_x).view(-1), input_y) / 2
optimizer.zero_grad()
l.backward()
optimizer.step()
print(loss(net(input_x).view(-1), input_y).item() / 2)
# Out:
# 0.1830
```
