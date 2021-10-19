# Function Differences with torch.broadcast_tensors

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/migration_guide/source_en/api_mapping/pytorch_diff/BroadcastTo.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source_en.png"></a>

## torch.broadcast_tensors

```python
torch.broadcast_tensors(
    *tensors
)
```

For more information, see [torch.broadcast_tensors](https://pytorch.org/docs/1.5.0/torch.html#torch.broadcast_tensors).

## mindspore.ops.BroadcastTo

```python
class mindspore.ops.BroadcastTo(shape)(input_x)
```

For more information, see [mindspore.ops.BroadcastTo](https://mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.BroadcastTo.html#mindspore.ops.BroadcastTo).

## Differences

PyTorch: Broadcasts given tensors according to [Broadcasting-semantics](https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics)
.

MindSpore：Broadcasts a given Tensor to a specified shape Tensor.

## Code Example

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