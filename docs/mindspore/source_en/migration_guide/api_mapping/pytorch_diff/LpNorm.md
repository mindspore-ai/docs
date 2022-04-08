# Function Differences with torch.norm/Tensor.norm

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/migration_guide/source_en/api_mapping/pytorch_diff/LpNorm.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

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

```python
torch.Tensor.norm(
    p='fro',
    dim=None,
    keepdim=False,
    dtype=None
)
```

For more information, see [torch.norm](https://pytorch.org/docs/1.5.0/torch.html#torch.norm).

## mindspore.ops.LpNorm

```python
class mindspore.ops.LpNorm(
    axis,
    p=2,
    keep_dims=False,
    epsilon=1e-12
)(input)
```

For more information, see [mindspore.ops.LpNorm](https://mindspore.cn/docs/api/en/master/api_python/ops/mindspore.ops.LpNorm.html#mindspore.ops.LpNorm).

## Differences

PyTorch: The p parameter can support types or values such as int, float, inf, -inf, 'fro', 'nuc' to calculate different types of normalization.

MindSpore: Currently only normalization for integer p-normal form is supported.

## Code Example

```python
import mindspore
from mindspore import Tensor
import mindspore.ops as ops
import torch
import numpy as np

# In MindSpore, only Lp norm is supported.
net = ops.LpNorm(axis=0, p=2)
input_x = Tensor(np.array([[4, 4, 9, 1], [2, 1, 3, 6]]), mindspore.float32)
output = net(input_x)
print(output)
# Out：
# [4.472136 4.1231055 9.486833 6.0827627]

# In torch, p=2
input_x = torch.tensor(np.array([[4, 4, 9, 1], [2, 1, 3, 6]]), dtype=torch.float)
output1 = torch.norm(input_x, dim=0, p=2)
print(output1)
# Out：
# tensor([4.4721, 4.1231, 9.4868, 6.0828])

# In torch, p='nuc'
input_x = torch.tensor(np.array([[4, 4, 9, 1], [2, 1, 3, 6]]), dtype=torch.float)
output2 = torch.norm(input_x, dim=(0, 1), p='nuc')
print(output2)
# Out：
# tensor([16.8892])
```