# Function Differences with torch.nn.Softmin

[![View Source On Gitee](https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/r2.0/resource/_static/logo_source_en.png)](https://gitee.com/mindspore/docs/blob/r2.0/docs/mindspore/source_en/note/api_mapping/pytorch_diff/softmin.md)

## torch.nn.Softmin

```python
torch.nn.Softmin(
    dim=None
)
```

For more information, see [torch.nn.Softmin](https://pytorch.org/docs/1.8.1/generated/torch.nn.Softmin.html).

## mindspore.nn.Softmin

```python
class mindspore.nn.Softmin(
    axis=-1
)
```

For more information, see [mindspore.nn.Softmin](https://www.mindspore.cn/docs/en/r2.0/api_python/nn/mindspore.nn.Softmin.html).

## Differences

PyTorch: Supports instantiation with the `dim` parameter, which scales the specified dimension elements between [0, 1] and sums to 1. Default value: None.

MindSpore: Supports instantiation with the `axis` parameter, which scales the specified dimension elements between [0, 1] and sums to 1. Default value: -1.

| Classification | Subclass  | PyTorch | MindSpore | difference |
| ---- | ----- | ------- | --------- | -------------------- |
| Parameter | Parameter 1 | dim     | axis      | Same function, different parameter names |

## Code Example

```python
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

# MindSpore
x = ms.Tensor(np.array([1, 2, 3, 4, 5]), ms.float32)
softmin = nn.Softmin()
output1 = softmin(x)
print(output1)
# Out:
# [0.6364086 0.23412167 0.08612854 0.03168492 0.01165623]
x = ms.Tensor(np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]), ms.float32)
softmin = nn.Softmin(axis=0)
output2 = softmin(x)
print(output2)
# out:
# [[0.98201376 0.880797   0.5        0.11920292 0.01798621]
#  [0.01798621 0.11920292 0.5        0.880797   0.98201376]]

# PyTorch
input = torch.tensor(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
output3 = F.softmin(input, dim=0)
print(output3)
# Out:
# tensor([0.6364, 0.2341, 0.0861, 0.0317, 0.0117], dtype=torch.float64)
```
