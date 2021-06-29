# 比较与torch.broadcast_tensors的功能差异

## torch.broadcast_tensors

```python
torch.broadcast_tensors(
    *tensors
)
```

## mindspore.ops.BroadcastTo

```python
class mindspore.ops.BroadcastTo(shape)(input_x)
```

## 使用方式

PyTorch: 按照[一定的规则](https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics)
将输入的若干个tensor广播成1个tensor。

MindSpore：将一个给定的tensor广播成指定形状的tensor。

## 代码示例

```python
from mindspore import Tensor
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore, the parameter shape is passed to reshape input_x.
shape = (2, 3)
input_x = Tensor(np.array([1, 2, 3]).astype(np.float32))
broadcast_to = ops.BroadcastTo(shape)
output = broadcast_to(input_x)
print(output.shape)
# Out：
# (2, 3)

# In torch, two tensors x and y should be separately passed.
# And the final output of the tensor's shape will be determined by these inputs' shapes according to rules mentioned above.
x = torch.Tensor(np.array([1, 2, 3]).astype(np.float32)).view(1, 3)
y = torch.Tensor(np.array([4, 5]).astype(np.float32)).view(2, 1)
m, n = torch.broadcast_tensors(x, y)
print(m.shape)
# Out：
# torch.Size([2, 3])
```