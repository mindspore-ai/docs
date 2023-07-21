# Function Differences with torch.broadcast_tensors

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r1.9/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r1.9/docs/mindspore/source_en/note/api_mapping/pytorch_diff/BroadcastTo.md)

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

For more information, see [mindspore.ops.BroadcastTo](https://mindspore.cn/docs/en/r1.9/api_python/ops/mindspore.ops.BroadcastTo.html#mindspore.ops.BroadcastTo).

## Differences

PyTorch: Broadcasts given tensors according to [Broadcasting-semantics](https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics)
.

MindSpore：Broadcasts a given Tensor to a specified shape Tensor.

## Code Example

```python
import mindspore as ms
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore, the parameter shape is passed to reshape input_x.
shape = (2, 3)
input_x = ms.Tensor(np.array([1, 2, 3]).astype(np.float32))
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