﻿# Function Differences with torch.Tensor.float

[![View Source On Gitee](https://gitee.com/mindspore/docs/raw/r1.6/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r1.6/docs/mindspore/migration_guide/source_en/api_mapping/pytorch_diff/Cast.md)

## torch.Tensor.float

```python
torch.Tensor.float(memory_format=torch.preserve_format)
```

For more information, see [torch.Tensor.float](https://pytorch.org/docs/1.5.0/tensors.html#torch.Tensor.float).

## mindspore.ops.Cast

```python
class mindspore.ops.Cast(*args, **kwargs)(
    input_x,
    type
)
```

For more information, see [mindspore.ops.Cast](https://mindspore.cn/docs/api/en/r1.6/api_python/ops/mindspore.ops.Cast.html#mindspore.ops.Cast).

## Differences

PyTorch: Changes the tensor type to float.

MindSpore：Converts the input type to the specified data type.

## Code Example

```python
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore, you can specify the data type to be transformed into.
input_x = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
cast = ops.Cast()
output = cast(input_x, mindspore.int32)
print(output.dtype)
print(output.shape)
# Out：
# Int32
# (2, 3, 4, 5)

# In torch, the input will be transformed into float.
input_x = torch.Tensor(np.random.randn(2, 3, 4, 5).astype(np.int32))
output = input_x.float()
print(output.dtype)
print(output.shape)
# Out：
# torch.float32
# torch.Size([2, 3, 4, 5])
```