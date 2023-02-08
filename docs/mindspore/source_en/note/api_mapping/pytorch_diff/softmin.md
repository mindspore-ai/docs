# Function Differences with torch.nn.Softmin

<a href="https://gitee.com/mindspore/docs/blob/master/docs/mindspore/source_en/note/api_mapping/pytorch_diff/softmin.md" target="_blank"><img src="https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/website-images/master/resource/_static/logo_source_en.png"></a>

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

For more information, see [mindspore.nn.Softmin](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Softmin.html).

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
import numpy as np

# Mindspore
x = ms.Tensor(np.array([1, 2, 3, 4, 5]), ms.float32)
softmin = nn.Softmin()
output1 = softmin(x)
print(output1)
# Out:
# [0.01165623 0.03168492 0.08612854 0.23412167 0.6364086 ]
x = ms.Tensor(np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]]), ms.float32)
output2 = softmin(x, axis=0)
print(output2)
# out:
# [[0.01798621 0.11920292 0.5        0.880797   0.98201376]
#  [0.98201376 0.880797   0.5        0.11920292 0.01798621]]

# Pytorch
input = torch.tensor(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
output3 = torch.nn.softmin(dim=0)(input)
print(output3)
# Out:
# tensor([0.0117, 0.0317, 0.0861, 0.2341, 0.6364], dtype=torch.float64)
```
