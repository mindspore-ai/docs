# Function Differences with torch.min

## torch.min

```python
torch.min(
    input,
    dim,
    keepdim=False)
```

## mindspore.ops.ArgMinWithValue

```python
class mindspore.ops.ArgMinWithValue(
    axis=0,
    keep_dims=False
)(input_x)
```

## Differences

PyTorch: Output tuple(min, index of min).

MindSpore: Output tuple(index of min, min).

## Code Example

```python
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import torch
import numpy as np

# Output tuple(index of min, min).
input_x = Tensor(np.array([0.0, 0.4, 0.6, 0.7, 0.1]), mindspore.float32)
argmin = ops.ArgMinWithValue()
index, output = argmin(input_x)
print(index)
print(output)
# Out：
# 0
# 0.0

# Output tuple(min, index of min).
input_x = torch.tensor([0.0, 0.4, 0.6, 0.7, 0.1])
output, index = torch.min(input_x, 0)
print(index)
print(output)
# Out：
# tensor(0)
# tensor(0.)
```
