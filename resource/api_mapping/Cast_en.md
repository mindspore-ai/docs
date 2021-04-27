﻿# Function Differences with torch.Tensor.float

## torch.Tensor.float

```python
torch.Tensor.float()
```

## mindspore.ops.Cast

```python
class mindspore.ops.Cast(*args, **kwargs)(
    input_x,
    type
)
```

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