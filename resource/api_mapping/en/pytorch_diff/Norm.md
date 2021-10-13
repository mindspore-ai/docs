# Function Differences with torch.norm

## torch.norm

```python
torch.norm(
    input,
    p='fro',
    dim=None,
    keepdim=False,
    out=None,
    dtype=None
)
```

For more information, see[torch.norm](https://pytorch.org/docs/1.5.0/torch.html#torch.norm).

## mindspore.nn.Norm

```python
class mindspore.nn.Norm(
    axis=(),
    keep_dims=False
)(input)
```

For more information, see[mindspore.nn.Norm](https://mindspore.cn/docs/api/en/r1.3/api_python/nn/mindspore.nn.Norm.html#mindspore.nn.Norm).

## Differences

PyTorch: Multiple normalizations including L2-norm are supported.

MindSpore: Only supports L2 norm.

## Code Example

```python
import mindspore
from mindspore import Tensor, nn
import torch
import numpy as np

# In MindSpore, only L2 norm is supported.
net = nn.Norm(axis=0)
input_x = Tensor(np.array([[4, 4, 9, 1], [2, 1, 3, 6]]), mindspore.float32)
output = net(input_x)
print(output)
# Out：
# [4.4721 4.1231 9.4868 6.0828]

# In torch, you can set parameter p to implement the desired norm.
input_x = torch.tensor(np.array([[4, 4, 9, 1], [2, 1, 3, 6]]), dtype=torch.float)
output1 = torch.norm(input_x, dim=0, p=2)
print(output1)
# Out：
# tensor([4.4721, 4.1231, 9.4868, 6.0828])

input_x = torch.tensor(np.array([[4, 4, 9, 1], [2, 1, 3, 6]]), dtype=torch.float)
output2 = torch.norm(input_x, dim=0, p=1)
print(output2)
# Out：
# tensor([6., 5., 12., 7.])
```