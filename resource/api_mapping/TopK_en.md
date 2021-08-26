# Function Differences with torch.topk

## torch.topk

```python
torch.topk(
    input,
    k,
    dim=None,
    largest=True,
    sorted=True,
    *,
    out=None
)
```

## mindspore.ops.TopK

```python
class mindspore.ops.TopK(
    sorted=False
)(input_x, k)
```

## Differences

PyTorch: Support to obtain the maximum or minimum value of the k entries along the specified dimension.

MindSpore：Only support to obtain the maximum value of the k entries along the last dimension currently.

## Code Example

```python
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore, only obtain the k largest entries along the last dimension.
topk = ops.TopK()
k = 3
input_x = Tensor([[1, 2, 3, 4], [2, 4, 6, 8]], mindspore.float16)
values, indices = topk(input_x, k)
print(values)
print(indices)

# Out：
# [[4. 3. 2.]]
# [[8. 6. 4.]]
# [[3 2 1]]
# [[3 2 1]]

# In torch, you can set parameter dim to obtain the k largest or smallest entries along the specific dimension.
# largest=True
input_x = torch.tensor([[1, 2, 3, 4], [2, 4, 6, 8]], dtype=torch.float)
dim = 1
output = torch.topk(input_x, k, dim=dim, largest=True)
print(output)

# Out：
# torch.return_types.topk(
# values=tensor([[4., 3., 2.],
#                [8., 6., 4.]]),
# indices=tensor([[3, 2, 1],
#                 [3, 2, 1]]))

# largest=False
output = torch.topk(input_x, k, dim=dim, largest=False)
print(output)

# Out：
# torch.return_types.topk(
# values=tensor([[1., 2., 3.],
#                [2., 4., 6.]]),
# indices=tensor([[0, 1, 2],
#                 [0, 1, 2]]))
```
