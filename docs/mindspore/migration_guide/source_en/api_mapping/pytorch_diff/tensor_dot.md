# Function Differences with torch.dot

<a href="https://gitee.com/mindspore/docs/blob/r1.5/docs/mindspore/migration_guide/source_en/api_mapping/pytorch_diff/tensor_dot.md" target="_blank"><img src="https://gitee.com/mindspore/docs/raw/r1.5/resource/_static/logo_source_en.png"></a>

## torch.dot

```python
torch.dot(
    input,
    other,
    out=None
)
```

For more information, see [torch.dot](https://pytorch.org/docs/1.5.0/torch.html#torch.dot).

## mindspore.ops.tensor_dot

```python
mindspore.ops.tensor_dot(
    x1,
    x2,
    axes
)
```

For more information, see [mindspore.ops.tensor_dot](https://mindspore.cn/docs/api/en/r1.5/api_python/ops/mindspore.ops.tensor_dot.html#mindspore.ops.tensor_dot).

## Differences

PyTorch: Calculates the dot product(inner product) of two tensors of the same shape, only 1D is supported.

MindSpore：Calculates the dot product of two tensors on any axis. Support tensor of any dimension, but the shape corresponding to the specified axis should be equal. The function of the PyTorch is the same when the input is 1D and the axis is set to 0.

## Code Example

```python
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore, tensor of any dimension will be supported.
# And parameters will be set to specify how to compute among dimensions.
input_x1 = Tensor(np.array([2, 3, 4]), mindspore.float32)
input_x2 = Tensor(np.array([2, 1, 3]), mindspore.float32)
output = ops.tensor_dot(input_x1, input_x2, 1)
print(output)
# Out：
# 19.0

# In torch, only 1D tensor's computation will be supported.
input_x1 = torch.tensor([2, 3, 4])
input_x2 = torch.tensor([2, 1, 3])
output = torch.dot(input_x1, input_x2)
print(output)
# Out：
# tensor(19)
```